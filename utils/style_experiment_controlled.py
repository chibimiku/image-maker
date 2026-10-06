"""画风相似度**严格受控实验**的编排、统计与报告（协议 1.0 / 2026-10-06）。

本模块只做三件事，**不重新实现任何相似度公式**：

1. 冻结数据、参数、主指标与验收门槛（写入 run 目录的 `PLAN/*.json` 与 `attempts.jsonl` 账本）；
2. 调用共享入口（`utils.style_similarity.compare_images` /
   `modules.image_analysis.style_comparison.compare_state_in_process`）执行各阶段；
3. 统计、汇总、写报告（Markdown / results.json / HTML），并如实标注未完成与缺口。

规则（协议 §3/§10/§11）在本文件内固化为常量，先声明后使用，运行中出现偏差只记录不改阈值。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_VERSION = "1.0"
PROTOCOL_DATE = "2026-10-06"
SEED = 20261006
EXPERIMENT_ID = "style-similarity-controlled-v1"
RUN_ROOT_RELATIVE = "data/test-result/20261006/style-similarity-controlled-v1/run-20261006-01"
PROMPT_DIR_RELATIVE = "prompts/style-extraction/similarity-validation-v1"

DEVICE = "cpu"
PRECISION = "fp32"
MAIN_METRIC = "csd"
METRIC_DIRECTION = {"gram": "min", "adain": "min", "lpips": "min", "csd": "max",
                    "tone": "max", "edges": "max", "lines": "max", "space": "max"}
FACE_METRIC_DIRECTION = {metric: "max" for metric in (
    "hair_fineness", "hair_continuity", "eye_brightness", "eye_height", "eyelashes", "eye_width",
    "eye_gap", "eye_curvature", "iris_ratio", "eye_highlights", "lid_weight", "eye_tilt", "face_ratios")}

TIE_RTOL = 1e-9
E0_SYMMETRY_ABS = 1e-6
E0_SYMMETRY_RTOL = 1e-5
MAX_ATTEMPTS = 3          # 首次 + 2 次重试（协议 §10）
BOOTSTRAP_ROUNDS = 2000

#: 预注册验收门槛（工程口径，不是行业标准）
ACCEPTANCE = {
    "implementation_usable": "E0 关键项全部通过；不存在错误 hash 复用或 partial 冒充完整",
    "style_retrieval_assist": {"value": 0.80, "label": "E1 试验集 CSD Top-1 ≥ 80% 且各风格 ≥ 2/3；"
                                                     "E2 困难对照正确偏好 ≥ 75%（并列计不正确）"},
    "local_measurement_assist": {"value": 0.90, "label": "E3 目标响应方向命中 ≥ 90%；局部负对照均通过；报告未验证维度与耦合"},
    "auto_region_default": {"coverage": 0.90, "per_metric_coverage": 0.80, "lid_mean_over_eye_width": 0.02,
                            "lid_p95_over_eye_width": 0.05, "hair_band_valid": 0.95,
                            "label": "E4 可用图覆盖 ≥ 90%、每维度有效覆盖 ≥ 80%、眼睑平均偏差 ≤ 眼宽 2%、"
                                     "95 分位 ≤ 5%、发丝采样带有效率 ≥ 95%，并须人工间重复性通过"},
    "candidate_ranking_assist": {"value": 0.80, "min_items": 24,
                                 "label": "E5 两人重复题一致率各 ≥ 80%、≥ 24 个有效独立题；"
                                          "预注册主指标与人类共识方向一致 ≥ 80%；严重主体错误不被选为合格最佳"},
}

STAGE_LEDGER_KEYS = ("planned", "actual", "independent_works", "missing", "reason")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_root() -> Path:
    return PROJECT_ROOT / RUN_ROOT_RELATIVE


def prompt_dir() -> Path:
    return PROJECT_ROOT / PROMPT_DIR_RELATIVE


# --------------------------------------------------------------------------- #
# 尝试账本（协议 §10：每一次请求/失败/费用未知/缓存复用都要记录）
# --------------------------------------------------------------------------- #


class AttemptLedger:
    """逐 job 记录 attempts；`job_id` 稳定，重试不换 run_id、不重置计数。"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.job_index = {}

    def load(self):
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.rows.append(json.loads(line))
                    self.job_index[self.rows[-1]["job_id"]] = self.rows[-1]
        return self

    def next_attempt(self, job_id: str) -> int:
        previous = [row["attempt_id"] for row in self.rows if row["job_id"] == job_id]
        return len(previous) + 1

    def record(self, **row):
        row.setdefault("timestamp", now())
        row.setdefault("protocol", PROTOCOL_VERSION)
        self.rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self.job_index[row["job_id"]] = row
        return row

    def finished_ok(self, job_id: str) -> bool:
        return any(row["job_id"] == job_id and row["status"] in ("ok", "reused") for row in self.rows)

    def summarize(self) -> dict:
        counter = {}
        for row in self.rows:
            key = row.get("stage", "unknown") + "|" + str(row.get("status"))
            counter[key] = counter.get(key, 0) + 1
        charged_unknown = [row["job_id"] for row in self.rows
                           if row.get("may_have_charged") and row.get("cost") in (None, "unknown")]
        return {
            "rows": len(self.rows),
            "by_stage_status": dict(sorted(counter.items())),
            "retried_jobs": sorted({row["job_id"] for row in self.rows
                                    if self.next_attempt(row["job_id"]) > 2 and row.get("status") == "ok"}),
            "jobs_with_multiple_attempts": sorted({row["job_id"] for row in self.rows
                                                   if sum(1 for r in self.rows if r["job_id"] == row["job_id"]) > 1}),
            "cost_unknown_jobs": sorted(set(charged_unknown)),
            "http_attempts_unknown": sorted({row["job_id"] for row in self.rows
                                             if row.get("http_attempts_unknown")}),
            "reused_jobs": sorted({row["job_id"] for row in self.rows if row.get("status") == "reused"}),
        }


def with_retries(ledger: AttemptLedger, job_id: str, stage: str, action, request: dict | None = None,
                 charged: bool = False, allow_retry=True):
    """执行一个技术 job：首次 + 最多 2 次重试。

    只有**技术失败**（异常）才重试；成功但分数低/结果差一律不回炉（协议 §10）。
    """
    attempts = 0
    last_error = None
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        started = now()
        attempt_id = ledger.next_attempt(job_id)
        try:
            value = action()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            ledger.record(job_id=job_id, attempt_id=attempt_id, retry_of=(attempt_id - 1) or None,
                          stage=stage, status="failed", started=started, finished=now(),
                          request_hash=json_hash(request or {}), error_class=type(exc).__name__,
                          error=str(exc)[:2000], may_have_charged=charged, cost="unknown" if charged else 0.0,
                          http_attempts_unknown=charged)
            if not allow_retry or attempts >= MAX_ATTEMPTS:
                break
            time.sleep(1.0)
            continue
        ledger.record(job_id=job_id, attempt_id=attempt_id, retry_of=(attempt_id - 1) or None,
                      stage=stage, status="ok", started=started, finished=now(),
                      request_hash=json_hash(request or {}), error_class=None, error=None,
                      may_have_charged=charged, cost=0.0 if not charged else "unknown",
                      http_attempts_unknown=False)
        return value
    raise RuntimeError(f"job {job_id} 达到 {attempts} 次尝试上限：{last_error}")


def ensure_stage_result(ledger, run_dir, stage, producer, **kwargs):
    """已完成并成功写盘的阶段直接复用，不重复计算（缓存/断点复用，写进账本）。"""
    marker = Path(run_dir) / f"{stage.upper()}-result.json"
    if marker.is_file():
        ledger.record(job_id=f"{stage}:cached", attempt_id=1, retry_of=None, stage=stage, status="reused",
                      started=now(), finished=now(), request_hash=json_hash({"marker": str(marker)}),
                      may_have_charged=False, cost=0.0)
        return read_json(marker)
    value = producer()
    atomic_json(marker, value)
    return value


# --------------------------------------------------------------------------- #
# 冻结包
# --------------------------------------------------------------------------- #


def code_freeze() -> dict:
    files = ["utils/style_similarity.py", "utils/style_face_metrics.py", "utils/style_regions.py",
             "utils/style_render_metrics.py", "utils/style_experiment_attributes.py",
             "utils/style_experiment_synthetic.py", "utils/style_experiment_controlled.py",
             "utils/style_experiment_reports.py",
             "tools/style_metrics_verify.py", "modules/image_analysis/style_comparison.py",
             "modules/image_analysis/style_deep_comparison.py", "modules/image_analysis/style_similarity_tab.py",
             "prompts/style-extraction/face-regions-v1.md", "prompts/style-comparison-v2.md"]
    frozen = {}
    for relative in files:
        path = PROJECT_ROOT / relative
        frozen[relative] = sha256_file(path) if path.is_file() else None
    for path in sorted((PROJECT_ROOT / "utils/style_metrics").glob("*.py")):
        frozen[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = sha256_file(path)
    for path in sorted(prompt_dir().glob("*")):
        if path.is_file():
            frozen[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = sha256_file(path)
    return frozen


def weights_freeze() -> dict:
    from utils.style_metrics import inventory
    records = inventory.list_inventory(verify=False)
    return {key: {"path": value.get("path"), "expected_sha256": value.get("expected_sha256"),
                  "expected_bytes": value.get("expected_bytes"), "exists": value.get("exists"),
                  "size_matches": value.get("size_matches"), "source_url": value.get("source_url")}
            for key, value in records.items()}


def runtime_freeze() -> dict:
    import importlib.metadata
    import platform
    import sys
    versions = {}
    for package in ("torch", "torchvision", "numpy", "Pillow", "opencv-python", "lpips", "clip",
                    "openvino", "openai", "pyqt6"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    from utils.style_metrics.config import CSD_INPUT_SIZE, GRAM_FORMULA_VERSION, LPIPS_INPUT_SIZE, VGG_INPUT_SIZE
    return {"python": sys.version.split()[0], "executable": sys.executable,
            "platform": {"system": platform.system(), "release": platform.release(),
                         "machine": platform.machine()},
            "packages": versions, "device": DEVICE, "precision": PRECISION,
            "preprocessing": {"vgg_input": VGG_INPUT_SIZE, "lpips_input": LPIPS_INPUT_SIZE,
                              "csd_input": CSD_INPUT_SIZE, "gram_formula": GRAM_FORMULA_VERSION},
            "gpu_present": _gpu_note(),
            "note": "主结果固定 CPU FP32；CUDA 仅作旁证，不参与主结果，也不得中途切换设备。"}


def _gpu_note() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)}（本实验主结果不使用）"
    except Exception:
        pass
    return "unavailable"


# --------------------------------------------------------------------------- #
# 数据冻结：候选池 → 属性 → 分层选样
# --------------------------------------------------------------------------- #


def scan_pool(root, extensions=(".jpg", ".jpeg", ".png", ".webp"), min_short_side=800, limit=None,
              marker_files=True):
    """稳定顺序扫描目录（只读），过滤明显不可用与元数据文件。"""
    from PIL import Image
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"数据源不存在：{root}")
    candidates = []
    for extension in extensions:
        candidates.extend(root.rglob("*" + extension))
    stable = []
    for path in sorted(set(candidates), key=lambda item: str(item).lower()):
        name = path.name.lower()
        if marker_files and any(token in name for token in (".mask.", "-bak", "metadata", "thumbnail")):
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            continue
        if min(width, height) < min_short_side:
            continue
        stable.append({"path": str(path.resolve()), "width": int(width), "height": int(height),
                       "work_id": path.stem.lower(), "file_name": path.name,
                       "source_root": str(root), "source_relative": str(path.relative_to(root))})
    stable.sort(key=lambda item: item["path"].lower())
    return stable[:limit] if limit else stable


def stratified_selection(pool, references=6, queries=6, seed=SEED):
    """分层选择：先按确定性规则剔除「同一作品」，再按
    (framing, brightness, background_density) 分层排序交替分配。

    **不看任何画风指标**，只用冻结属性；同一规则用于全部画风，保证可比。
    模糊近重复只标记不删除（协议要求人工核查后冻结），并在返回里列出待核查对。
    """
    from utils.style_experiment_attributes import near_duplicate_decisions
    items = [dict(row) for row in pool]
    decisions = near_duplicate_decisions(items)
    auto_excluded = {item["path"] for item in decisions["auto_excluded"]}
    kept = [row for row in items if row["path"] not in auto_excluded]
    # 稳定排序：按 (framing, brightness, background_density, path)，避免变成亮/暗交替
    ordered = sorted(kept, key=lambda row: (str(row["framing"]), str(row["brightness"]),
                                            str(row["background_density"]), row["path"].lower()))
    reference_rows, query_rows = [], []
    for index, row in enumerate(ordered):
        if index % 2 == 0 and len(reference_rows) < references:
            reference_rows.append(row)
        elif len(query_rows) < queries:
            query_rows.append(row)
        elif len(reference_rows) < references:
            reference_rows.append(row)
    chosen = {row["path"] for row in reference_rows + query_rows}
    extras = [row for row in kept if row["path"] not in chosen]
    return {"references": reference_rows, "queries": query_rows, "extras": extras,
            "near_duplicate_pairs": decisions["awaiting_human_review"],
            "near_duplicate_excluded": sorted(auto_excluded),
            "near_duplicate_rules": decisions["thresholds"],
            "near_duplicate_auto_excluded_detail": decisions["auto_excluded"],
            "fuzzy_pairs_awaiting_review": len(decisions["awaiting_human_review"]),
            "pool_size": len(items), "usable_pool": len(kept)}


STYLE_SOURCES = {
    "puracotte": {"label": "Puracotte", "style_key": "puracotte-style-v2",
                  "source_root": r"C:\data\train\puracotte-2d",
                  "note": "本地原作品集（pixiv 下载目录），非生成图"},
    "sakurapion": {"label": "Sakurapion", "style_key": "sakurapion-style",
                   "source_root": r"C:\data\train\sakurapion-2d\train",
                   "note": "本地原作品集 train 目录"},
    "kishida_mel": {"label": "Kishida Mel", "style_key": "kishida-mel-style",
                    "source_root": r"C:\data\train\kishida_mel-2d\tagged",
                    "note": "本地原作品集 tagged 目录"},
    "renian": {"label": "Renian", "style_key": "renian",
               "source_root": r"C:\data\train\renian-2d\fanbox",
               "note": "FANBOX 原作品集"},
}

#: 协议优先级里 TID 只有单张参考图，本地没有可验证的原作品集 → 用可验证来源的 Renian 替代
STYLE_SUBSTITUTION = {"requested": ["Puracotte", "Sakurapion", "Kishida Mel", "TID"],
                      "used": ["puracotte", "sakurapion", "kishida_mel", "renian"],
                      "reason": "本地 TID 仅有 data/style-ref/tid-fullbody-background-candidate.jpg 与 "
                                "tid.png 两张素材，没有可验证来源的原作品集；改用来源可验证的 "
                                "C:\\data\\train\\renian-2d（FANBOX 原作品）作为第 4 种画风。",
                      "decided_before_scoring": True}


def build_style_pools(measure=True, progress=lambda message: None) -> dict:
    """扫描并测量候选池。测量结果按 (路径, 大小, mtime, 属性版本) 缓存在 cache/ 下，
    代码或数据变化时自动失效；缓存只加速同一冻结流程的重跑，不参与任何分数。"""
    from utils.style_experiment_attributes import VERSION as ATTRIBUTE_VERSION
    from utils.style_experiment_attributes import measure as measure_image
    cache_path = PROJECT_ROOT / "cache" / "style-experiment" / "pool-measurements.json"
    cache = read_json(cache_path, {}) or {}
    result = {}
    dirty = False
    for style_id, source in STYLE_SOURCES.items():
        progress(f"扫描 {style_id}：{source['source_root']}")
        pool = scan_pool(source["source_root"], limit=240)
        if measure:
            records = []
            for index, row in enumerate(pool):
                stat = os.stat(row["path"])
                key = f"{row['path']}|{stat.st_size}|{int(stat.st_mtime)}|{ATTRIBUTE_VERSION}"
                record = cache.get(key)
                if not record:
                    record = measure_image(row["path"])
                    cache[key] = record
                    dirty = True
                record = dict(record)
                record.update(image_id=f"{style_id}-{index + 1:03d}", style_id=style_id,
                              work_id=row["work_id"], source_root=row["source_root"],
                              source_relative=row["source_relative"], is_generated=False,
                              origin_evidence={"kind": "original-artwork-local-corpus",
                                               "source_root": row["source_root"],
                                               "source_relative": row["source_relative"],
                                               "note": source["note"]})
                records.append(record)
                if index % 25 == 0:
                    progress(f"{style_id} 属性 {index + 1}/{len(pool)}"
                             + ("（缓存命中）" if record.get("attribute_version") else ""))
            pool = records
        result[style_id] = {"source": source, "pool": pool}
    if dirty:
        atomic_json(cache_path, cache)
    return result


def freeze_split(pools, references=6, queries=6) -> dict:
    frozen = {}
    for style_id, payload in pools.items():
        selection = stratified_selection(payload["pool"], references, queries)
        for row in selection["references"]:
            row["split"] = "reference"
            row["style_id"] = style_id
        for row in selection["queries"]:
            row["split"] = "query"
            row["style_id"] = style_id
        for row in selection["extras"]:
            row["split"] = "reserved_e2_pool"
            row["style_id"] = style_id
        frozen[style_id] = {"source": payload["source"], **selection}
    return frozen


# --------------------------------------------------------------------------- #
# 共享计算入口调用
# --------------------------------------------------------------------------- #


def compare(queries, references, cache_directory, device=DEVICE, progress=lambda message: None):
    """唯一计算入口：逐 query × 全部 reference 逐张比较（不预先平均图片）。"""
    from utils.style_similarity import compare_images, image_manifest
    inputs = image_manifest([row["path"] for row in queries], [row["path"] for row in references])
    return compare_images(inputs, device, progress, cache_directory)


def pair_lookup(result) -> dict:
    lookup = {}
    for row in result["rows"]:
        for pair in row["pairs"]:
            lookup[(row["path"], pair["reference"])] = pair
    return lookup


def metric_value(pair, metric):
    outcome = pair.get(metric) or {}
    value = outcome.get("value")
    if outcome.get("status") not in ("ok",) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


# --------------------------------------------------------------------------- #
# 统计工具
# --------------------------------------------------------------------------- #


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict:
    if total <= 0:
        return {"n": 0, "k": successes, "p": None, "low": None, "high": None}
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"n": total, "k": successes, "p": p, "low": max(0.0, center - half), "high": min(1.0, center + half)}


def cluster_bootstrap(clusters, statistic, rounds=BOOTSTRAP_ROUNDS, seed=SEED) -> dict:
    """按簇（query/work）重采样的 cluster bootstrap 95% 区间。"""
    generator = random.Random(seed)
    names = sorted(clusters)
    if not names:
        return {"rounds": 0, "clusters": 0, "low": None, "high": None, "note": "没有可用簇"}
    values = []
    for _ in range(rounds):
        sample = [clusters[generator.choice(names)] for _ in range(len(names))]
        try:
            value = statistic(sample)
        except Exception:
            continue
        if value is not None and math.isfinite(value):
            values.append(value)
    values.sort()
    if not values:
        return {"rounds": 0, "clusters": len(names), "low": None, "high": None, "note": "统计量在重采样中不可用"}
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return {"rounds": len(values), "clusters": len(names), "low": low, "high": high, "seed": seed}


def kendall_tau_b(first, second) -> dict:
    """Kendall tau-b（含并列修正）；样本不足返回 None。"""
    pairs = list(zip(first, second))
    n = len(pairs)
    if n < 3:
        return {"n": n, "tau": None, "note": "样本不足"}
    concordant = discordant = ties_first = ties_second = 0
    for i in range(n):
        for j in range(i + 1, n):
            a1, b1 = pairs[i]
            a2, b2 = pairs[j]
            da, db = a1 - a2, b1 - b2
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_first += 1
            elif db == 0:
                ties_second += 1
            elif da * db > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_first) * (concordant + discordant + ties_second))
    tau = (concordant - discordant) / denominator if denominator else None
    return {"n": n, "tau": tau, "concordant": concordant, "discordant": discordant,
            "ties_first": ties_first, "ties_second": ties_second}


def icc_a1(groups) -> dict:
    """双向随机、绝对一致性、单次测量 ICC(A,1)。样本不足或方差为 0 时如实返回 None。"""
    usable = [group for group in groups if len(group) >= 2]
    if len(usable) < 2:
        return {"icc": None, "note": "可用案例不足"}
    k = min(len(group) for group in usable)
    usable = [group[:k] for group in usable]
    n = len(usable)
    grand = statistics.mean(value for group in usable for value in group)
    row_means = [statistics.mean(group) for group in usable]
    column_means = [statistics.mean(group[index] for group in usable) for index in range(k)]
    ss_total = sum((value - grand) ** 2 for group in usable for value in group)
    ss_rows = k * sum((mean - grand) ** 2 for mean in row_means)
    ss_columns = n * sum((mean - grand) ** 2 for mean in column_means)
    ss_error = ss_total - ss_rows - ss_columns
    ms_rows = ss_rows / (n - 1) if n > 1 else None
    ms_columns = ss_columns / (k - 1) if k > 1 else None
    ms_error = ss_error / ((n - 1) * (k - 1)) if n > 1 and k > 1 else None
    if not ms_error or ms_error <= 0:
        return {"icc": None, "note": "组内误差方差为 0 或不可估", "n": n, "k": k}
    if not ms_rows:
        return {"icc": None, "note": "行间方差不可估", "n": n, "k": k}
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    icc = (ms_rows - ms_error) / denominator if denominator else None
    return {"icc": icc, "n": n, "k": k, "ms_rows": ms_rows, "ms_error": ms_error,
            "note": "双向随机 / 绝对一致性 / 单次测量 ICC(A,1)"}


def build_plan(ledger: AttemptLedger, progress=None) -> dict:
    """冻结计划：E0 图片集、四种画风的数据源与 split、E3 图样矩阵、E5 状态与额度。

    本函数**只看数据与配置**，不读取任何指标值；计划写盘后才开始算分。
    """
    from utils import style_experiment_synthetic as syn
    stages_root = run_root()
    plan_dir = stages_root / "PLAN"
    plan_dir.mkdir(parents=True, exist_ok=True)

    # --- E0 图片集：至少 4 张原作品 + 历史两张 Gemini 图
    originals = []
    for style_id, source in STYLE_SOURCES.items():
        pool = scan_pool(source["source_root"], limit=40)
        if not pool:
            continue
        chosen = pool[min(5, len(pool) - 1)]
        originals.append({"image_id": f"E0-{style_id}", "kind": "original", "style_id": style_id,
                          "path": chosen["path"], "sha256": sha256_file(chosen["path"]),
                          "work_id": chosen["work_id"], "source_root": chosen["source_root"],
                          "source_relative": chosen["source_relative"], "is_generated": False})
    generated = []
    for entry in HISTORY_GEMINI_IMAGES:
        path = PROJECT_ROOT / entry["relative"]
        if not path.is_file():
            raise FileNotFoundError("历史 Gemini 产物缺失：" + str(path))
        generated.append({"image_id": entry["image_id"], "kind": "generated", "style_id": entry["style_id"],
                          "path": str(path), "sha256": sha256_file(path), "is_generated": True,
                          "origin_evidence": entry["origin"], "history_note": entry["history_note"]})
    e0_images = originals + generated

    # --- E1/E2 数据源
    pools = build_style_pools(progress=progress)
    split = freeze_split(pools)
    split_records = []
    for style_id, payload in split.items():
        for group in ("references", "queries", "extras"):
            for row in payload[group]:
                split_records.append({key: row.get(key) for key in
                                      ("image_id", "style_id", "split", "path", "sha256", "width", "height",
                                       "work_id", "framing", "brightness", "background_density",
                                       "face_area_ratio", "median_gray_center", "background_border_edge_ratio",
                                       "dominant_hue", "mean_saturation", "source_root", "source_relative")})

    # --- E2 P/N 与变体（选择只读冻结属性；本模块与阶段模块互相引用，这里延迟导入）
    from utils.style_experiment_controlled_phases import build_e2_pairs
    e2_plan = build_e2_pairs({"split": split})

    # --- E5：历史产物状态、冻结参数与额度
    styles_config = json.loads((PROJECT_ROOT / "conf/config-styles.json").read_text(encoding="utf-8"))
    e5_states, e5_runs, blind_pool = {}, {}, []
    for style_id, source in STYLE_SOURCES.items():
        references = [row["path"] for row in split[style_id]["references"]]
        history = next((row for row in generated if row.get("style_id") == style_id), None)
        if not history:
            continue
        candidate_id = f"history/{style_id}/0"
        dataset = {"name": f"{style_id}-e1-references", "images": references}
        candidates = [{"id": candidate_id, "path": history["path"], "sha256": history["sha256"],
                       "record": {"test_prompt": HISTORY_SUBJECT, "style_reference": references[0],
                                  "generated_files": {}, "source": "history-gemini-20261006"}}]
        common = {"dataset": dataset, "dataset_selection": {"split": "reference",
                                                            "chosen_from": source["source_root"]}}
        e5_states[style_id] = {**common,
                               "test_images": {"history": {"generated_files": {"gemini": [history["path"]]},
                                                           "test_prompt": HISTORY_SUBJECT,
                                                           "style_reference": references[0]}},
                               "parameters": {"repaint_reference_mode": "none"}}
        # 两种呈现顺序：参考图正序 / 倒序。候选只有 1 张历史产物，候选级顺序无法检验，
        # 这里检验的是「参考图呈现顺序」的敏感性，并在报告里如实说明该限制。
        forward = dict(common)
        forward["dataset"] = {**dataset, "images": list(references)}
        forward["test_images"] = {"history": {"generated_files": {"gemini": [history["path"]]},
                                              "test_prompt": HISTORY_SUBJECT, "style_reference": references[0]}}
        reverse = dict(common)
        reverse["dataset"] = {**dataset, "images": list(reversed(references))}
        reverse["test_images"] = dict(forward["test_images"])
        reverse["parameter_presentation_order"] = "reference-order-reversed"
        e5_runs[style_id] = {
            "forward": {"state": forward, "state_path": str(stages_root / "e5" / style_id / "state-forward.json")},
            "reverse": {"state": reverse, "state_path": str(stages_root / "e5" / style_id / "state-reverse.json")},
        }
        blind_pool.append({"image_id": f"candidate-{style_id}", "style_id": style_id, "path": history["path"],
                           "role": "candidate"})
        blind_pool.append({"image_id": f"reference-{style_id}", "style_id": style_id,
                           "path": references[0], "role": "reference"})
    reference_showcase = [{"label": f"{style_id} 参考画风图（E1 冻结点）",
                           "path": split[style_id]["references"][0]["path"]}
                          for style_id in sorted(split)]
    e5_plan = {"runs": e5_runs, "states": e5_states, "blind_pool": blind_pool,
               "reference_showcase": reference_showcase,
               "e5_new_generation": {"planned_slots": 18, "subjects": 3, "repeats": 3, "styles": 2,
                                     "spec": "2 种画风 × 3 个主体（正面头像 / 正面半身 / 可读三分之四面部）"
                                             "× 3 次独立重复；仅 Gemini 带参考图，禁止 GPT 首图与后续重绘",
                                     "authorised": False}}

    plan = {
        "run_id": stages_root.name,
        "seed": SEED,
        "device": DEVICE,
        "precision": PRECISION,
        "main_metric": MAIN_METRIC,
        "acceptance": ACCEPTANCE,
        "protocol": {"version": PROTOCOL_VERSION, "date": PROTOCOL_DATE,
                     "document": "docs/style-extraction/DEEPSEEK-SIMILARITY-CONTROLLED-EXPERIMENT-20261006.md",
                     "sha256": sha256_file(PROJECT_ROOT /
                                           "docs/style-extraction/"
                                           "DEEPSEEK-SIMILARITY-CONTROLLED-EXPERIMENT-20261006.md")},
        "e0_images": e0_images,
        "e0_entries": {"query": next(row for row in generated),
                       "references": originals[:3]},
        "split": split,
        "split_records": split_records,
        "e2": e2_plan,
        "e3": {"cases": syn.CASES, "interventions": syn.INTERVENTIONS,
               "negative_controls": syn.NEGATIVE_CONTROLS, "declared_couplings": {}},
        "e5": {key: value for key, value in e5_plan.items() if key != "states"},
        "e5_states": e5_plan["states"],
        "e5_frozen_parameters": HISTORY_FROZEN_PARAMETERS,
        "style_substitution": STYLE_SUBSTITUTION,
        "styles_config_snapshot": {name: {"enabled": entry.get("enabled", True),
                                          "ref_image": entry.get("ref_image")}
                                   for name, entry in styles_config.items()},
        "frozen_at": now(),
    }
    return plan


def plan_files(run_dir: Path) -> list[str]:
    return [str(path) for path in sorted((Path(run_dir) / "PLAN").glob("*.json"))]


def near_duplicate_contact_sheets(plan, out_dir, cell=380, max_pairs=60) -> list[str]:
    """把模糊近重复候选对拼成接触表，供人工核查（协议要求检测后人工核查再冻结）。"""
    from utils.style_experiment_attributes import build_contact_sheet
    sheets = []
    per_style = {}
    for style, payload in plan["split"].items():
        per_style[style] = payload["near_duplicate_pairs"]
    for style, pairs in sorted(per_style.items()):
        for index in range(0, min(len(pairs), max_pairs), 8):
            chunk = pairs[index:index + 8]
            items = []
            for pair in chunk:
                items.append((pair["first"], pair["first_path"]))
                items.append((pair["second"], pair["second_path"]))
            target = out_dir / f"{style}-near-duplicates-{index // 8 + 1}.png"
            build_contact_sheet(items, target, cell=cell, columns=4)
            sheets.append(str(target))
    return sheets


def write_freezes(plan, ledger, progress=None) -> dict:
    """把代码/权重/依赖/账本首行落盘，形成可复核的冻结包。"""
    run_dir = run_root()
    plan_dir = run_dir / "PLAN"
    plan_dir.mkdir(parents=True, exist_ok=True)
    freeze = {"code_and_prompts_sha256": code_freeze(), "weights": weights_freeze(),
              "runtime": runtime_freeze(),
              "generated_at": now()}
    from utils.style_experiment_controlled_phases import freeze_e3
    plan["e3_frozen"] = freeze_e3(progress)
    atomic_json(plan_dir / "plan-full.json", plan)
    atomic_json(plan_dir / "protocol-lock.json", {
        "protocol_version": PROTOCOL_VERSION, "protocol_date": PROTOCOL_DATE,
        "run_id": plan["run_id"], "seed": SEED, "device": DEVICE, "precision": PRECISION,
        "main_metric": MAIN_METRIC, "acceptance_thresholds": ACCEPTANCE,
        "data_frozen_before_scoring": True, "run_version": plan.get("run_version", "v1"),
        "protocol_document_sha256": plan["protocol"]["sha256"],
        "metric_directions": METRIC_DIRECTION, "tie_rtol": TIE_RTOL,
        "e0_symmetry_tolerance": {"atol": E0_SYMMETRY_ABS, "rtol": E0_SYMMETRY_RTOL},
        "max_attempts_per_job": MAX_ATTEMPTS,
        "style_substitution": STYLE_SUBSTITUTION,
        "frozen_at": now()})
    atomic_json(plan_dir / "code-freeze.json", freeze)
    atomic_json(plan_dir / "weights.json", freeze["weights"])
    atomic_json(plan_dir / "runtime.json", freeze["runtime"])
    atomic_json(plan_dir / "images.json",
                {"e0_images": plan["e0_images"], "split_records": plan["split_records"],
                 "e3_variants": plan["e3_frozen"]["records"],
                 "confounds": _confounds(plan)})
    atomic_json(plan_dir / "pairs.json",
                {"e1": {"queries": [row["image_id"] for row in
                                    [item for style in plan["split"].values() for item in style["queries"]]],
                        "references_by_style": {style: [row["image_id"] for row in payload["references"]]
                                                for style, payload in plan["split"].items()},
                        "rule": "同一 query 分别与每种画风的参考集比较；禁止把不同画风参考图混进一个均值"},
                 "e0": {"pairs": [row["image_id"] for row in plan["e0_images"]],
                        "entries": plan["e0_entries"]},
                 "e2": {"anchors": [row["image_id"] for row in plan["e2"]["anchors"]],
                        "positive": {key: value["row"]["image_id"] for key, value in plan["e2"]["positive"].items()},
                        "negative": {key: value["row"]["image_id"] for key, value in plan["e2"]["negative"].items()},
                        "selection_rule": plan["e2"]["selection_rule"]},
                 "e3": {"cases": [row["case_id"] for row in plan["e3"]["cases"]]},
                 "e5": {"existing_pairs": ["history/puracotte/0", "history/sakurapion/0"],
                        "planned_slots": plan["e5"]["e5_new_generation"]["planned_slots"]}})
    atomic_json(plan_dir / "transforms.json",
                {"transform_version": plan["e3_frozen"].get("transform_version", "style-experiment-transforms/1"),
                 "e2_parameters": {"N0": {"operation": "lossless PNG re-encode"},
                                   "N1": {"hue_shift_degrees": 30.0},
                                   "N2": {"intensity_scale": 0.9},
                                   "N3": {"blur_sigma": 6.0, "region": "确定性问题/主体掩膜背景",
                                          "mask_threshold_percentile": 75.0},
                                   "N4": {"shift_fraction_of_width": 0.05}},
                 "e2_variants": plan.get("e2_frozen", {}).get("variants"),
                 "e3_interventions": plan["e3"]["interventions"],
                 "e3_negative_controls": plan["e3"]["negative_controls"]})
    atomic_json(plan_dir / "blind-map.json", {"pending": True,
                                              "seed": SEED,
                                              "note": "盲评包在 E5 执行时生成，见 E5/blind/blind-map.json"})
    atomic_json(plan_dir / "e2-plan.json", plan["e2"])
    atomic_json(plan_dir / "e5-plan.json", plan["e5"])
    # 冻结顺序说明：本函数只写计划与冻结包；plan-full.json 在调用方写完 e3_frozen 后重写
    ledger.record(job_id="plan:freeze", attempt_id=1, retry_of=None, stage="PLAN", status="ok",
                  started=now(), finished=now(), request_hash=json_hash(plan["e0_images"]),
                  may_have_charged=False, cost=0.0)
    return freeze


def _confounds(plan) -> dict:
    import collections
    counter = {}
    for row in plan["split_records"]:
        bucket = counter.setdefault(row["style_id"], collections.Counter())
        bucket[row["split"] + ":" + str(row["framing"])] += 1
        bucket[row["split"] + ":" + str(row["brightness"])] += 1
        bucket[row["split"] + ":" + str(row["background_density"])] += 1
    return {style: dict(value) for style, value in sorted(counter.items())}


#: E0/E5 使用的历史 Gemini 产物（2026-10-06 实测，未重启 app、未发 GPT 生图）
HISTORY_GEMINI_IMAGES = [
    {"image_id": "GEMINI-puracotte", "style_id": "puracotte",
     "relative": "data/test-result/20261006/style-similarity-gemini/puracotte-style-v2/"
                 "test-puracotte-style-v2_111916-5611c0.jpg",
     "origin": {"channel": "gemini", "ui_entry": "app.py SingleGenDebugWidget.generate_image → ImageGenWorkerThread",
                "model": "gemini-3-pro-image-preview", "resolution": "2K", "aspect_ratio": "2:3",
                "style_ref_mode": "priority", "style_key": "puracotte-style-v2",
                "request_snapshot": "data/test-result/20261006/style-similarity-gemini/puracotte-style-v2/"
                                    "test-puracotte-style-v2_aigc2d_replay_111916_1679a5.json"},
     "history_note": "先前实测成功产物；协议 §9 要求补做此前缺失的正式视觉复核"},
    {"image_id": "GEMINI-sakurapion", "style_id": "sakurapion",
     "relative": "data/test-result/20261006/style-similarity-gemini/sakurapion-style/"
                 "test-sakurapion-style_111943-9894bf.jpg",
     "origin": {"channel": "gemini", "ui_entry": "app.py SingleGenDebugWidget.generate_image → ImageGenWorkerThread",
                "model": "gemini-3-pro-image-preview", "resolution": "2K", "aspect_ratio": "2:3",
                "style_ref_mode": "priority", "style_key": "sakurapion-style",
                "request_snapshot": "data/test-result/20261006/style-similarity-gemini/sakurapion-style/"
                                    "test-sakurapion-style_aigc2d_replay_111943_639435.json"},
     "history_note": "先前实测成功产物；Sakurapion 产物在深度指标上更接近 Puracotte 参考，需正式复核"},
]

HISTORY_SUBJECT = ("Create a waist-up portrait of one 25-year-old adult woman, facing the viewer with her head "
                   "upright, both eyes open and fully visible, a relaxed closed-mouth smile. She has long "
                   "chestnut-brown hair with a simple side part, no head ornaments, and emerald-green eyes. "
                   "Her clothing is a fully buttoned ivory blouse with long sleeves and a navy-blue sleeveless "
                   "vest. Her arms rest naturally below the crop; do not show hands. Place her in a quiet "
                   "bookshop with softly simplified bookshelves and a window behind her, in gentle daylight.")

HISTORY_FROZEN_PARAMETERS = {
    "generation_entry": "app.py SingleGenDebugWidget（真实 GUI 生成入口）→ ImageGenWorkerThread → "
                        "modules/others/api_backend.generate_image_aigc2d",
    "channel": "gemini", "model": "gemini-3-pro-image-preview", "resolution": "2K", "aspect_ratio": "2:3",
    "style_ref_mode": "priority",
    "reference_attachment_order": ["画风参考图（config ref_image，位于文后 STYLE 段之前）"],
    "text_length_policy": "参考优先模式使用 prompt_compressed",
    "gpt_image_used": False, "repaint_used": False, "post_process_used": False,
    "subject_text": "prompts/style-extraction/similarity-validation-20261006.txt",
    "note": "冻结自 2026-10-06 的实测快照（见 PLAN/e5-plan.json 与历史 request replay）。",
}


# --------------------------------------------------------------------------- #
# 阶段状态与缺口
# --------------------------------------------------------------------------- #


def stage_statuses(results, plan) -> dict:
    e0 = results.get("E0") or {}
    e1 = results.get("E1") or {}
    e2 = results.get("E2") or {}
    e3 = results.get("E3") or {}
    e4 = results.get("E4") or {}
    e5 = results.get("E5") or {}
    queries = sum(len(payload["queries"]) for payload in plan["split"].values())
    anchors = len((plan.get("e2") or {}).get("anchors") or [])
    variants = len((plan.get("e3_frozen") or {}).get("records") or [])
    e4_images = len(((e4 or {}).get("rows")) or [])
    return {
        "E0": {"status": e0.get("status", "planned"), "planned": len(plan["e0_images"]),
               "actual": len(e0.get("self_check_rows") or []),
               "independent_works": len(plan["e0_images"]),
               "missing_ids": e0.get("critical_failures") or [],
               "reason": "零收费本地计算"},
        "E1": {"status": e1.get("status", "planned"), "planned": queries, "actual": queries,
               "independent_works": queries,
               "missing_ids": [], "reason": "四种画风 × 12 个独立作品（6 参考 / 6 query）"},
        "E2": {"status": e2.get("status", "planned"), "planned": anchors, "actual": anchors,
               "independent_works": anchors,
               "missing_ids": [], "reason": "12 个 anchor + P/N + N0–N4 变体；P/N 由自动匹配代理填写"},
        "E3": {"status": e3.get("status", "planned"), "planned": variants, "actual": len(e3.get("rows") or []),
               "independent_works": 6,
               "missing_ids": [], "reason": "可控绘制图样机制测试，不代表真实图效度"},
        "E4": {"status": "awaiting_human", "planned": e4_images, "actual": e4_images,
               "independent_works": e4_images,
               "missing_ids": ["H1", "H2", "H1-repeat(≥24h)", "人工变体结构核查"],
               "reason": "自动定位已跑；人工标注与人工间重复性必须由实际人类提供"},
        "E5": {"status": "awaiting_budget+awaiting_human", "planned": 18 + 2, "actual": 2,
               "independent_works": 2,
               "missing_ids": ["新增 18 槽位（未授权额度）", "两名人类 ≥24 个有效独立题"],
               "reason": "历史 2 张 Gemini 产物已补正式视觉复核与盲评包；新增生图无授权额度"},
    }


def collect_gaps(results, plan) -> list[dict]:
    e5 = (results.get("E5") or {})
    gaps = [
        {"kind": "human", "what": "E4 的两名人类标注者（H1/H2）与 H1 隔 ≥24 小时复标；"
                                  "眼睑曲线、虹膜、睫毛、脸轮廓、头发 mask/path 叠图核查",
         "blocks": "自动定位偏差、人工间重复性、自动 vs 人工排名翻转率、自动定位能否默认参与正式测量"},
        {"kind": "human", "what": "E5 两名人类各 ≥24 个有效独立 A/B 题（含 6 个重复题检验个人一致性）",
         "blocks": "候选排序效度、视觉模型与人类的一致率"},
        {"kind": "budget", "what": f"E5 新增 18 个生图槽位（2 画风 × 3 主体 × 3 重复）的费用授权；"
                                   f"当前已授权 0",
         "blocks": "E5 新增候选、组内排序与盲评题数"},
        {"kind": "human", "what": "E2 的 N0–N4 变体人工核查（人物结构是否被意外改变）与「配色/构图匹配」人工标注",
         "blocks": "困难对照的选择依据是否成立"},
        {"kind": "data", "what": "TID 画风可验证原作品集（本地仅有 2 张素材），当前用 Renian 替代",
         "blocks": "与协议原始画风清单完全对齐"},
        {"kind": "data", "what": "确认集（每画风 20 个新独立 query 作品）",
         "blocks": "从「试验集诊断」升级为通用可靠性结论"},
    ]
    if (e5.get("prepared_deep_review") or {}).get("errors"):
        gaps.append({"kind": "data", "what": "E5 视觉复核中失败的批次",
                     "blocks": "该候选的正式复核结论"})
    return gaps


def summary_template(results) -> str:
    e1 = results.get("E1") or {}
    primary = ((e1.get("analysis") or {}).get(MAIN_METRIC)) or {}
    e2 = ((results.get("E2") or {}).get("analysis") or {}).get("preference") or {}
    e3 = ((results.get("E3") or {}).get("analysis") or {}).get("gate") or {}
    e4 = ((results.get("E4") or {}).get("coverage")) or {}
    e5 = (results.get("E5") or {})
    ledger = results.get("ledger_summary") or {}
    return f"""协议/代码版本：{results.get('protocol_version')} / 代码 hash 见 PLAN/code-freeze.json
冻结包与运行目录：{run_root()}
实际独立原作品/query/主体组数：{sum(len(v['references']) + len(v['queries']) for v in (results.get('plan') or {{}}).get('split', {{}}).values())}/{(e1.get('counts') or {{}}).get('queries')}/6
E0：{((results.get('E0') or {{}}).get('status'))}；关键失败：{(results.get('E0') or {{}}).get('critical_failures')}
E1：主指标 {MAIN_METRIC} n={primary.get('counts', {{}}).get('total')} Top-1={primary.get('top1_accuracy')} 区间={primary.get('wilson_95')} 每画风={{{', '.join(f'{k}:{v.get("correct")}/{v.get("total")}' for k, v in (primary.get('per_style') or {{}}).items())}}}
E2：困难对照 CSD 正确偏好 {(e2.get('csd') or {{}}).get('correct')}/{(e2.get('csd') or {{}}).get('n')}；变体翻转见 E2-result.json
E3：目标响应 {e3.get('direction_hit')}；负对照 {'通过' if e3.get('negative_controls_passed') else '未通过'}
E4：自动有效 {(e4.get('usable'))}/{(e4.get('images'))}；人工确认 {e4.get('confirmed')}/{(e4.get('images'))}（awaiting_human）
E5：人类有效题数 = 0（awaiting_human）；视觉已复核 {(e5.get('review', {{}}).get('runs') and sum(len(v) for v in e5['review']['runs'].values()))}/2 顺序；排序一致性未计算
重试/未知收费/缓存复用：{ledger.get('jobs_with_multiple_attempts')} / {ledger.get('cost_unknown_jobs')} / {ledger.get('reused_jobs')}
哪些用途可用、哪些只能探索、哪些不能用：见报告第 10 节
尚需人工/额度/数据的具体项目：见报告第 11 节
下一轮唯一优先改进及验证方案：{results.get('next_step', '')}"""


def next_step(results) -> str:
    e4 = (results.get("E4") or {})
    usable = ((e4.get("coverage") or {}).get("usable_rate"))
    if usable is None or usable < 0.9:
        return ("先解决自动定位在真实动漫图上的精度问题：以 6 个可控案例的解析真值 + 4 张历史图建立"
                "人工叠图基线，把眼睑曲线偏差（相对眼宽）与发丝采样带有效率作为唯一验收指标，"
                "先做人工定位条件下的稳定性检验，再考虑改进定位提示词；改提示词属于新版本，"
                "不回改本次结果。")
    return ("先把 E5 的 18 槽位额度与两名人类盲评补齐：只有 ≥24 个有效独立题才能判断排序效度；"
            "在此之前不建议改动任何公式或权重。")


def exact_counts(values) -> dict:
    """精确计数（阈值判定用向上取整，见协议 §11）。"""
    return {"n": len(values), "k": sum(1 for value in values if value)}


def round_or_none(value, digits=8):
    return None if value is None else round(float(value), digits)
