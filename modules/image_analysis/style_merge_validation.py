"""myc0t0xin 按字段合并验证：候选包构造、预检、生图、比较、交付物落盘。

背景：第 2 轮 GPT 首图与第 5 轮 Gemini 直出在旧比较里总分并列，但两者来自**不同字段**
（第 2 轮 `gpt_image_prompt` 合法、第 5 轮不合格）。本模块把「按字段合并」做成可复算的
验证流程，而不是把两段风格描述拼成一段：

- 包 A：三字段全部取第 2 轮（对照）
- 包 B：`gemini_full_prompt` 取第 5 轮，`gpt_image_prompt` 与 `repaint_clauses` 取第 2 轮（按通道合并）
- 包 C：在 B 基础上把 `repaint_clauses` 换成第 5 轮（重绘条款对照）

关键约束（见 `docs/style-extraction/deepseek-myc0t0xin-merged-validation.md`）：

1. **不重跑分析轮次**：只做字段复制 + 现有生图链路，第 2/5 轮字段按原文与 hash 复制，不压缩、不改写。
2. **重绘条款也进入 GPT 首图**：`_generate_test_image` 把 `gemini_repaint_clauses` 当 `extra_clauses`
   放进 GPT 首图请求，所以 A/B/C 的 GPT 首图必须按包分别核验，不能认为复制了 `prompt_gpt` 就等价。
2. **预检与实际发送共用同一段组装代码**：预检直接调用真实业务函数
   `StyleIterativeWorkerThread._generate_test_image`，只把它内部要发出去的四次调用换成探针。
3. **第 5 轮 GPT 短版不合格时**：不发 GPT 首图/重绘请求，状态写 `not_run`，绝不回退到母版顶替。
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from modules.image_analysis.style_analyzer import (
    StyleIterativeWorkerThread,
    _reference_aspect_ratio,
    normalize_style_prompt_package,
)

# 候选包的字段来源轮次（第 1 行是 A，第 2 行是 B，第 3 行是 C）
PACKAGE_PLAN = (
    ("A", "round_2", "round_2", "round_2", "第 2 轮对照"),
    ("B", "round_5", "round_2", "round_2", "按通道合并"),
    ("C", "round_5", "round_2", "round_5", "重绘条款对照"),
)

# `_generate_test_image` 内部会引用的三个通道名 —— 预检只替换发送调用，组装代码全部跑真的
PROBE_CHANNELS = ("gemini_direct", "gpt_first_pass", "gpt_repainted")

# 「按运行记录候选 JSON 取包」模式：允许从候选 JSON 里读进 prompt 包的字段。
PACKAGE_FIELDS = ("gemini_full_prompt", "gpt_image_prompt", "gemini_repaint_clauses",
                  "face_hair_clauses", "optional_motifs", "negative_rules", "usage_profiles",
                  "evidence_summary", "confidence")

# 候选 JSON 本身的形态（只用来**报清来源**，不当作校验通过）：
#   v2     —— app「多图画风提取」的完整训练结果
#   light  —— 只带 prompt_variants / final_test_images 的精简候选（如人工优化候选）
REVIEW_CANDIDATE_SHAPES = ("v2", "light")

# 状态文件名按运行记录派生，避免与 app 的结果文件撞名。
def review_state_basename(candidate_name: str) -> str:
    return f"{candidate_name}_style_iter_result.json"


class ValidationError(RuntimeError):
    """预检/校验失败：调用方必须先停止联网请求再报告。"""


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #


def utc_stamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def text_hash(text):
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_info(path):
    from PIL import Image
    with Image.open(path) as image:
        return {"path": os.path.abspath(path), "sha256": file_hash(path), "bytes": os.path.getsize(path),
                "width": image.width, "height": image.height, "format": image.format,
                "aspect_ratio": round(image.width / image.height, 6)}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path):
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def log_line(directory, message):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "validation.log"), "a", encoding="utf-8") as stream:
        stream.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(message, flush=True)


# --------------------------------------------------------------------------- #
# 任务定义
# --------------------------------------------------------------------------- #


class MergeValidationTask:
    def __init__(self, source, output_root="", subject="", repetitions=2, reuse_cache=True,
                 dry_run=False, package_keys=("A", "B", "C"), channels=PROBE_CHANNELS,
                 enable_deep=True, enable_compare=True, subject_label="primary", stage="stage1",
                 model_overrides=None):
        self.source = os.path.abspath(source)
        self.output_root = os.path.abspath(output_root) if output_root else os.path.abspath(os.path.join(
            "data", datetime.datetime.now().strftime("%Y%m%d"), "style-extraction", "myc0t0xin",
            "merged-validation-" + utc_stamp()))
        self.repetitions = max(1, int(repetitions))
        self.reuse_cache = bool(reuse_cache)
        self.dry_run = bool(dry_run)
        self.package_keys = tuple(package_keys)
        self.channels = tuple(channels)
        self.enable_deep = bool(enable_deep)
        self.enable_compare = bool(enable_compare)
        self.subject_label = str(subject_label or "primary")
        self.stage = str(stage or "stage1")
        # 生图模型默认从当前配置读；只有节点没配或要做跨模型对照时才显式覆盖，值进请求审计
        self.model_overrides = {key: str(value).strip() for key, value in (model_overrides or {}).items()
                                if str(value or "").strip()}
        self._state = None
        self.subject = str(subject or "").strip()

    # -- 源结果 ---------------------------------------------------------- #

    def source_state(self):
        if self._state is None:
            if not os.path.isfile(self.source):
                raise ValidationError(f"结果 JSON 不存在：{self.source}")
            self._state = read_json(self.source)
        return self._state

    def subject_prompt(self):
        """固定测试主体：默认沿用源运行里记录的同一个主体，不另写一套。"""
        if self.subject:
            return self.subject
        state = self.source_state()
        prompts = {str(record.get("test_prompt") or "").strip()
                   for record in (state.get("test_images") or {}).values()
                   if str(record.get("test_prompt") or "").strip()}
        prompts |= {str((state.get("final_test_images") or {}).get("test_prompt") or "").strip()} - {""}
        if len(prompts) != 1:
            raise ValidationError(f"源结果里的测试主体不唯一，无法固定对照：{sorted(prompts)}")
        return prompts.pop()

    def style_reference(self):
        state = self.source_state()
        record = state["test_images"]["round_2"]
        path = str(record.get("style_reference") or "")
        if not path or not os.path.isfile(path):
            raise ValidationError(f"测试画风参考图不可用：{path or '(未记录)'}")
        return os.path.abspath(path)

    def dataset_images(self):
        state = self.source_state()
        images = [os.path.abspath(str(p)) for p in (state.get("dataset") or {}).get("images") or []]
        if len(images) != 12:
            raise ValidationError(f"预期 12 张参考图，实际 {len(images)} 张")
        missing = [p for p in images if not os.path.isfile(p)]
        if missing:
            raise ValidationError("参考图缺失：" + "; ".join(missing))
        return images


# --------------------------------------------------------------------------- #
# 按运行记录候选 JSON 取包（复核对照模式）
# --------------------------------------------------------------------------- #


class ReviewCandidateTask(MergeValidationTask):
    """同一固定主体的「运行记录候选包」对照：读每个候选 JSON 自己的 prompt 包。

    与 `MergeValidationTask`（跨轮**字段合并**）的区别：
    - 字段不再跨轮拼接，整包取自该候选 JSON 记录的 `prompt_variants`；不重跑任何分析轮次。
    - 数据集张数不写死：候选 JSON 自己记录的 `dataset.images` 有几张就用几张。
    - 主体、画风参考图、输出根目录逐个主体独立，避免把场景差异当成版本差异。
    - `--reference` 指向的原运行 JSON 只在候选没记录时兜底数据集/筛图来源/参考图，不改写候选字段。
    """

    def __init__(self, candidates, reference="", output_root="", subject="",
                 repetitions=1, reuse_cache=True, dry_run=False, stage="review",
                 subject_label="primary", model_overrides=None):
        if not candidates:
            raise ValidationError("复核模式至少要有一个候选包")
        self.candidates = [dict(item) for item in candidates]
        self._candidate_states = {}
        self._dataset_cache = None
        merge_source = str(reference or self.candidates[0].get("path") or "")
        super().__init__(source=merge_source, output_root=output_root, subject=subject,
                         repetitions=max(1, int(repetitions)), reuse_cache=reuse_cache,
                         dry_run=dry_run, package_keys=tuple(item["key"] for item in self.candidates),
                         enable_deep=True, enable_compare=True, subject_label=subject_label,
                         stage=stage, model_overrides=model_overrides)

    # -- 来源 ------------------------------------------------------------- #

    def candidate_state(self, item):
        path = os.path.abspath(str(item["path"]))
        if not os.path.isfile(path):
            raise ValidationError(f"候选 JSON 不存在：{path}")
        if path not in self._candidate_states:
            self._candidate_states[path] = read_json(path)
        return self._candidate_states[path]

    def reference_state(self):
        """兜底来源：候选里没写数据集/筛图来源时用它，**不改写候选字段**。"""
        path = os.path.abspath(str(self.source or ""))
        if not os.path.isfile(path):
            raise ValidationError(f"兜底来源 JSON 不存在：{path}")
        return read_json(path)

    def _pick_dataset(self, state):
        images = [os.path.abspath(str(p)) for p in ((state.get("dataset") or {}).get("images") or [])]
        if images:
            return images, state.get("dataset_selection") or {}
        selection = state.get("dataset_selection") or {}
        paths = [os.path.abspath(str(p)) for p in (selection.get("image_paths") or [])]
        if paths:
            return paths, selection
        return [], {}

    def dataset_images(self):
        if self._dataset_cache is None:
            images = []
            for item in self.candidates:
                found, _ = self._pick_dataset(self.candidate_state(item))
                if not images:
                    images = found
                elif found != images:
                    raise ValidationError(
                        f"候选 {item['key']} 记录的数据集与 {self.candidates[0]['key']} 不一致；"
                        "同一主体的对照必须使用同一批源图")
            if not images:
                images, _ = self._pick_dataset(self.reference_state())
            if len(images) < 2:
                raise ValidationError("画风复核至少需要两张源图")
            missing = [p for p in images if not os.path.isfile(p)]
            if missing:
                raise ValidationError("源图缺失：" + "; ".join(missing))
            self._dataset_cache = images
        return list(self._dataset_cache)

    def dataset_selection(self):
        for item in self.candidates:
            _, selection = self._pick_dataset(self.candidate_state(item))
            if selection:
                return selection
        _, selection = self._pick_dataset(self.reference_state())
        return selection

    def style_reference(self):
        for item in self.candidates:
            path = str(item.get("style_reference") or "")
            if not path:
                path = self._candidate_style_reference(self.candidate_state(item))
            if path and os.path.isfile(path):
                return os.path.abspath(path)
        path = str((self.reference_state().get("final_test_images") or {}).get("style_reference") or "")
        if path and os.path.isfile(path):
            return os.path.abspath(path)
        raise ValidationError("候选 JSON 里没有可用的测试画风参考图")

    @staticmethod
    def _candidate_style_reference(state):
        for record in (state.get("test_images") or {}).values():
            path = str((record or {}).get("style_reference") or "")
            if path:
                return path
        return str((state.get("final_test_images") or {}).get("style_reference") or "")

    def subject_prompt(self):
        if self.subject:
            return self.subject
        subjects = []
        for item in self.candidates:
            origin = _candidate_variant_source(self, item)
            state = self.candidate_state(item)
            text = str(item.get("test_prompt") or origin.get("test_prompt") or
                       (state.get("final_test_images") or {}).get("test_prompt") or "").strip()
            if text:
                subjects.append(text)
        if subjects:
            if len(set(subjects)) != 1:
                raise ValidationError("候选测试主体不一致，请显式指定 --subject")
            return subjects[0]
        raise ValidationError("候选 JSON 没有记录测试主体，且没有显式指定 --subject")


# --------------------------------------------------------------------------- #
# 候选包
# --------------------------------------------------------------------------- #


def _candidate_variant_source(task, item):
    """从候选 JSON 里定位该候选实际记录的 prompt 包，并说明来源结构。

    两种形态都读、都如实标注，**不因为能读出来就当作校验通过**：
    - `v2`：完整训练结果 —— `test_images[<package_key>].prompt_variants`（例如 `round_5`）
    - `light`：精简候选 —— 根级 `prompt_variants`（或 `final_test_images.prompt_variants`）

    `package_key` 为空时取候选自己声明的 `source_candidate`（形如 `round_5/gpt_repainted/0`），
    避免用错轮次：原运行要的是 round_5 的包，不是最终 `prompt_variants`。
    """
    state = task.candidate_state(item)
    key = str(item.get("package_key") or "").strip()
    if not key:
        origin = state.get("resume_origin") or {}
        key = str(origin.get("candidate_id") or "").split("/")[0].strip()
    if key and (state.get("test_images") or {}).get(key):
        record = state["test_images"][key]
        variants = record.get("prompt_variants") or {}
        if variants:
            return {"shape": "v2", "package_key": key,
                    "record_path": f"test_images.{key}.prompt_variants",
                    "prompts_used": record.get("prompts_used") or "",
                    "test_prompt": record.get("test_prompt") or "",
                    "style_reference": record.get("style_reference") or "",
                    "variants": variants}
    if item.get("package_key") and key != "final":
        raise ValidationError(f"候选 {item['key']} 找不到显式指定的包 {key!r}，不回退最终包")
    for container, label in ((state, "prompt_variants"),
                             (state.get("final_test_images") or {},
                              "final_test_images.prompt_variants")):
        variants = container.get("prompt_variants") or {}
        if variants:
            return {"shape": "light", "package_key": key or "final",
                    "record_path": label,
                    "prompts_used": str(container.get("prompts_used") or state.get("final_art_style_prompts") or ""),
                    "test_prompt": str(container.get("test_prompt") or ""),
                    "style_reference": str(container.get("style_reference") or ""),
                    "variants": variants}
    raise ValidationError(
        f"候选 {item['key']} 的 JSON 里找不到 prompt 包"
        + (f"（指定 package_key={key!r}）" if key else "（根级与 final_test_images 都没有 prompt_variants）"))


def _field_snapshot(record, key):
    variants = record.get("prompt_variants") or {}
    value = variants.get(key)
    if isinstance(value, list):
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return {"kind": "list", "count": len(value), "sha256": text_hash(payload),
                "chars": sum(len(str(item)) for item in value), "value": value, "payload": payload}
    text = str(value or "")
    return {"kind": "text", "chars": len(text), "sha256": text_hash(text), "value": text, "payload": text}


def build_candidate_package(task, key):
    """按字段复制第 2/5 轮内容，再用 `normalize_style_prompt_package` 重新校验。"""
    plan = next((row for row in PACKAGE_PLAN if row[0] == key), None)
    if plan is None:
        raise ValidationError(f"未知候选包：{key}")
    _, gemini_round, gpt_round, clauses_round, label = plan
    state = task.source_state()
    records = {name: state["test_images"][name] for name in ("round_2", "round_5")}
    gemini_snapshot = _field_snapshot(records[gemini_round], "gemini_full_prompt")
    gpt_snapshot = _field_snapshot(records[gpt_round], "gpt_image_prompt")
    clauses_snapshot = _field_snapshot(records[clauses_round], "gemini_repaint_clauses")

    package = {
        "gemini_full_prompt": gemini_snapshot["value"],
        "gpt_image_prompt": gpt_snapshot["value"],
        "gemini_repaint_clauses": copy.deepcopy(clauses_snapshot["value"]),
        "optional_motifs": [],          # 三包统一关闭可选母题，排除装饰内容注入
        "motif_enabled": False,
        "negative_rules": copy.deepcopy(
            (records[gemini_round].get("prompt_variants") or {}).get("negative_rules") or []),
    }
    normalized = normalize_style_prompt_package(package, records[gemini_round].get("prompts_used") or "")

    # 精确复制校验：规范化不得悄悄改写被复制的字段（lenient 修复必须幂等）
    checks = {
        "gemini_full_prompt_unchanged": normalized.get("gemini_full_prompt") == gemini_snapshot["value"],
        "gpt_image_prompt_unchanged": normalized.get("gpt_image_prompt") == gpt_snapshot["value"],
        "repaint_clauses_unchanged": [str(c).strip() for c in normalized.get("gemini_repaint_clauses") or []]
                                     == [str(c).strip() for c in clauses_snapshot["value"]],
        "optional_motifs_empty": not normalized.get("optional_motifs"),
    }
    if not all(checks.values()):
        raise ValidationError(f"候选包 {key} 规范化改写了字段，已停止：{checks}")

    clause_text = json.dumps(normalized.get("gemini_repaint_clauses") or [], ensure_ascii=False)
    effective = {
        "gemini_full_prompt_chars": len(normalized.get("gemini_full_prompt") or ""),
        "gemini_full_prompt_sha256": text_hash(normalized.get("gemini_full_prompt") or ""),
        "gpt_image_prompt_chars": len(normalized.get("gpt_image_prompt") or ""),
        "gpt_image_prompt_sha256": text_hash(normalized.get("gpt_image_prompt") or ""),
        "repaint_clauses_count": len(normalized.get("gemini_repaint_clauses") or []),
        "repaint_clauses_sha256": text_hash(clause_text),
        "repaint_clauses_chars": sum(len(str(c)) for c in normalized.get("gemini_repaint_clauses") or []),
    }
    return {
        "key": key,
        "label": label,
        "field_sources": {
            "prompt": {"round": gemini_round, "field": "gemini_full_prompt",
                       "chars": gemini_snapshot["chars"], "sha256": gemini_snapshot["sha256"]},
            "prompt_gpt": {"round": gpt_round, "field": "gpt_image_prompt",
                           "chars": gpt_snapshot["chars"], "sha256": gpt_snapshot["sha256"]},
            "repaint_clauses": {"round": clauses_round, "field": "gemini_repaint_clauses",
                                "count": clauses_snapshot["count"], "chars": clauses_snapshot["chars"],
                                "sha256": clauses_snapshot["sha256"]},
        },
        "normalization": {
            "validator": "modules.image_analysis.style_analyzer.normalize_style_prompt_package",
            "copy_checks": checks,
            "gpt_image_prompt_valid": bool(normalized.get("gpt_image_prompt_valid")),
            "gpt_image_prompt_errors": list(normalized.get("gpt_image_prompt_errors") or []),
        },
        "effective": effective,
        "gpt_channel_allowed": bool(normalized.get("gpt_image_prompt_valid")),
        "motif_prompt": "",
        "package": normalized,
    }


# --------------------------------------------------------------------------- #
# 预检：调用真实组装函数，把「要发出去的东西」记下来（不发送任何请求）
# --------------------------------------------------------------------------- #


def build_review_package(task, item):
    """按运行记录整包复制候选字段，再用 `normalize_style_prompt_package` 重新校验。

    与原运行/优化候选的对应关系写在 `field_sources` 里（文件 SHA-256 + 字段 SHA-256），
    规范化不得改写被复制的字段；改写了就停止，不用 validity 布尔值绕过校验。
    """
    from modules.image_analysis.style_analyzer import normalize_style_prompt_package
    from utils.style_gpt import repair_prompt_gpt_lenient

    key = str(item["key"])
    origin = _candidate_variant_source(task, item)
    variants = origin["variants"]
    candidate_path = os.path.abspath(str(item["path"]))

    package = {field: copy.deepcopy(variants.get(field)) for field in PACKAGE_FIELDS
               if variants.get(field) is not None}
    # 可选母题默认**两边统一关闭**：原运行当年是开着母题发的（含花瓣/光点可选条款），
    # 优化包本身母题为空。若只让一边带母题，两包的对照就不成立，且「不新增装饰」这条
    # 特征会被自己的母题条款抵消。所以这里显式关闭，并在请求审计里记录该决定。
    package["optional_motifs"] = list(item.get("optional_motifs") or [])
    package["motif_enabled"] = bool(package["optional_motifs"])
    if item.get("negative_rules") is not None:
        package["negative_rules"] = copy.deepcopy(item["negative_rules"])
    package.setdefault("gemini_repaint_clauses", [])
    package.setdefault("gpt_image_prompt", "")
    package.setdefault("negative_rules", [])

    normalized = normalize_style_prompt_package(package, origin["prompts_used"])
    checks = {
        "gemini_full_prompt_unchanged":
            normalized.get("gemini_full_prompt") == str(variants.get("gemini_full_prompt") or "").strip(),
        "gpt_image_prompt_unchanged":
            normalized.get("gpt_image_prompt") == repair_prompt_gpt_lenient(str(variants.get("gpt_image_prompt") or "")),
        "repaint_clauses_unchanged":
            [str(c).strip() for c in normalized.get("gemini_repaint_clauses") or []]
            == [str(c).strip() for c in variants.get("gemini_repaint_clauses") or []],
        "optional_motifs_empty": not normalized.get("optional_motifs"),
    }
    if not all(checks.values()):
        raise ValidationError(f"候选 {key} 规范化改写了字段，已停止：{checks}")

    unknown = sorted(set(str(name) for name in variants if name not in PACKAGE_FIELDS))
    clause_text = json.dumps(normalized.get("gemini_repaint_clauses") or [], ensure_ascii=False)
    effective = {
        "gemini_full_prompt_chars": len(normalized.get("gemini_full_prompt") or ""),
        "gemini_full_prompt_sha256": text_hash(normalized.get("gemini_full_prompt") or ""),
        "gpt_image_prompt_chars": len(normalized.get("gpt_image_prompt") or ""),
        "gpt_image_prompt_sha256": text_hash(normalized.get("gpt_image_prompt") or ""),
        "repaint_clauses_count": len(normalized.get("gemini_repaint_clauses") or []),
        "repaint_clauses_sha256": text_hash(clause_text),
        "repaint_clauses_chars": sum(len(str(c)) for c in normalized.get("gemini_repaint_clauses") or []),
    }
    return {
        "key": key,
        "label": str(item.get("label") or key),
        "candidate_kind": "run_record",
        "source_shape": origin["shape"],
        "field_sources": {
            "candidate_json": {"path": candidate_path, "sha256": file_hash(candidate_path),
                               "package_key": origin["package_key"], "record_path": origin["record_path"]},
            "prompt": {"field": "gemini_full_prompt", "chars": len(str(variants.get("gemini_full_prompt") or "")),
                       "sha256": text_hash(str(variants.get("gemini_full_prompt") or ""))},
            "prompt_gpt": {"field": "gpt_image_prompt", "chars": len(str(variants.get("gpt_image_prompt") or "")),
                           "sha256": text_hash(str(variants.get("gpt_image_prompt") or ""))},
            "repaint_clauses": {"field": "gemini_repaint_clauses",
                                "count": len(variants.get("gemini_repaint_clauses") or []),
                                "chars": sum(len(str(c)) for c in variants.get("gemini_repaint_clauses") or []),
                                "sha256": text_hash(json.dumps(
                                    variants.get("gemini_repaint_clauses") or [], ensure_ascii=False,
                                    sort_keys=True))},
        },
        "normalization": {
            "validator": "modules.image_analysis.style_analyzer.normalize_style_prompt_package",
            "copy_checks": checks,
            "gpt_image_prompt_valid": bool(normalized.get("gpt_image_prompt_valid")),
            "gpt_image_prompt_errors": list(normalized.get("gpt_image_prompt_errors") or []),
        },
        "effective": effective,
        "gpt_channel_allowed": bool(normalized.get("gpt_image_prompt_valid")),
        "motif_prompt": "",
        "motif_decision": {
            "motif_enabled": bool(package["motif_enabled"]),
            "candidate_declared_motifs": len(variants.get("optional_motifs") or []),
            "note": "可选母题两边统一关闭：只让一边带母题会让对照不成立，且与「不新增装饰」相冲",
        },
        "unused_candidate_fields": unknown,
        "package": normalized,
    }


def build_precheck(task, key, package_entry, subject, style_reference, images, progress=None):
    """无联网预检：调用**真实组装函数**拿到三路要发出去的东西，不发送任何请求。

    这里不自己拼提示词：直接调用 `StyleIterativeWorkerThread._assemble_test_generation_requests`，
    和实跑用的是同一个方法；重绘那一路再调用 `utils.post_process.assemble_repaint_request`，
    也就是 `run_pipeline` 在发送前真正构造的那一份。
    """
    from modules.others import api_backend
    from utils.post_process import assemble_repaint_request, default_pipeline

    gemini_cfg = api_backend.get_api_config(api_type="aigc2d")
    gpt_cfg = api_backend.get_api_config(api_type="aigc-2d-gpt")
    aspect_ratio = _reference_aspect_ratio(style_reference)

    worker = StyleIterativeWorkerThread(
        images, api_key="", base_url="", model_name="",
        total_rounds=0, images_per_round=0, output_dir=os.path.join(task.output_root, "_precheck"),
        enable_test_gen=True, test_prompt=subject, img_api_type="aigc-2d-gpt",
        img_aspect_ratio="", file_prefix="precheck", test_style_ref_path=style_reference,
        repaint_reference_mode="none")
    worker.model_overrides = dict(task.model_overrides)

    variants = package_entry["package"]
    motif_prompt = ""
    source_gpt_model = str(gpt_cfg.get("model") or "")
    assembled = worker._assemble_test_generation_requests(variants, style_reference, aspect_ratio,
                                                          motif_prompt,
                                                          model_overrides=task.model_overrides)
    assembled_error = str(assembled.get("gpt_error") or "")

    channels = {}
    if assembled.get("gemini_direct"):
        channels["gemini_direct"] = _describe_probe("gemini_direct", assembled["gemini_direct"],
                                                    package_entry, subject, style_reference,
                                                    aspect_ratio, gemini_cfg, gpt_cfg)
    if not assembled.get("gpt_first_pass"):
        reasons = list(package_entry["normalization"]["gpt_image_prompt_errors"] or [])
        if assembled_error:
            reasons.append(assembled_error)
        channels["gpt_first_pass"] = _blocked_channel(
            "gpt_first_pass", "GPT 短版校验失败，未发送 GPT 生图请求：" + "；".join(reasons))
        channels["gpt_repainted"] = _blocked_channel("gpt_repainted", "GPT 首图未成功，无法进行重绘测试")
    else:
        channels["gpt_first_pass"] = _describe_probe("gpt_first_pass", assembled["gpt_first_pass"],
                                                     package_entry, subject, style_reference,
                                                     aspect_ratio, gemini_cfg, gpt_cfg)
        # 重绘：用 run_pipeline 同一份组装（source 是本次 GPT 首图的占位路径，只影响录制字段）
        repaint_cfg = dict((assembled.get("gpt_repaint") or {}).get("steps") or {}).get("repaint") or {}
        placeholder = os.path.abspath(os.path.join(task.output_root, "_precheck",
                                                   f"{key}-gpt-first.png"))
        os.makedirs(os.path.dirname(placeholder), exist_ok=True)
        reference_source = None
        for candidate in images:
            reference_source = candidate
            break
        if not os.path.isfile(placeholder):
            shutil.copy2(reference_source, placeholder)
        request = assemble_repaint_request(
            placeholder, repaint_cfg, style_ref_path=style_reference,
            style_clauses=(assembled.get("gpt_repaint") or {}).get("style_clauses") or [],
            sub_dir=os.path.dirname(placeholder), prefix="precheck",
            work_dir=os.path.dirname(placeholder))
        request["channel"] = "gpt_repainted"
        channels["gpt_repainted"] = _describe_repaint_request(request, package_entry, style_reference)
    shutil.rmtree(os.path.join(task.output_root, "_precheck"), ignore_errors=True)

    if "gemini_direct" not in channels:
        raise ValidationError(f"候选包 {key} 的 Gemini 直出组装未完成，预检失败：{assembled_error}")

    ordered = [channels[name] for name in PROBE_CHANNELS if name in channels]
    result = {"key": key, "channels": ordered, "aspect_ratio": aspect_ratio,
              "assembler": ("StyleIterativeWorkerThread._assemble_test_generation_requests"
                            "（预检与实跑同一段组装代码）"),
              "repaint_assembler": "utils.post_process.assemble_repaint_request（run_pipeline 发送前那一份）",
              "gpt_model": source_gpt_model,
              "config_models": {
                  "gemini": {"model": str(gemini_cfg.get("model") or ""),
                             "key_source": api_backend.api_key_source(api_type="aigc2d"),
                             "base_url": str(gemini_cfg.get("base_url") or "")},
                  "gpt": {"model": source_gpt_model,
                          "key_source": api_backend.api_key_source(api_type="aigc-2d-gpt"),
                          "base_url": str(gpt_cfg.get("base_url") or ""),
                          "size": str(gpt_cfg.get("size") or ""),
                          "quality": str(gpt_cfg.get("quality") or "")},
              }}
    result["request_digest"] = request_digest(ordered)
    result["channel_digests"] = channel_digests(ordered)
    if progress:
        progress(f"[预检] 包 {key}：request_digest={result['request_digest'][:16]}；"
                 f"通道={[entry.get('channel') for entry in ordered]}")
    return result


def _describe_repaint_request(request, package_entry, style_reference):
    kwargs = request.get("call_kwargs") or {}
    images = list(request.get("source_paths") or []) + list(request.get("extra_reference_paths") or [])
    info = [{"index": index, "path": os.path.abspath(path), "sha256": file_hash(path),
             "bytes": os.path.getsize(path),
             "role": "source（本次 GPT 首图的占位路径）" if index == 0 else f"extra[{index}]"}
            for index, path in enumerate(images)]
    prompt = str(request.get("prompt") or "")
    warnings = []
    if request.get("reference_mode") != "none":
        warnings.append(f"重绘参考模式不是 none（实际 {request.get('reference_mode')}）")
    if not request.get("clauses_without_image"):
        warnings.append("clauses_without_image 未开启，画风条款可能未被当作文字规格发送")
    if len(images) != 1:
        warnings.append(f"重绘发送了 {len(images)} 张图，文字模式应当只发送本次 GPT 首图")
    if "STYLE LANGUAGE" not in prompt:
        warnings.append("重绘提示词里没有 STYLE LANGUAGE 段落，条款没有真正进入请求")
    if "STYLE REFERENCE" in prompt and "REFERENCE ROLES" in prompt:
        warnings.append("重绘提示词仍带多图参考图指代，说明不是文字模式")
    if warnings:
        raise ValidationError("重绘模式不符合「文字模式」要求：" + "；".join(warnings))
    return {
        "channel": "gpt_repainted",
        "blocked": False,
        "source": "utils.post_process.assemble_repaint_request(...)（预检与实际发送同一份）",
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "prompt_sha256": request.get("prompt_sha256") or text_hash(prompt),
        "prompt_role_note": _prompt_role_note("gpt_repainted"),
        "images": info,
        "image_sha256": [item["sha256"] for item in info],
        "params": {"model": request.get("model"), "resolution": request.get("resolution"),
                   "aspect_ratio": request.get("aspect_ratio"), "repeat": request.get("repeat"),
                   "reference_mode": request.get("reference_mode"),
                   "scope": request.get("scope"),
                   "style_clauses_count": len(request.get("style_clauses") or []),
                   "clauses_without_image": request.get("clauses_without_image"),
                   "prompt_source": request.get("system_prompt_source"),
                   "firmware_source": request.get("firmware_source")},
        "clauses": list(request.get("style_clauses") or []),
        "repaint_clauses_source_round": _clause_source(package_entry),
        "warnings": warnings,
        "reference_mode": request.get("reference_mode"),
        "motif_enabled": False,
        "motif_prompt_chars": 0,
        "log_lines": list(request.get("log_lines") or []),
    }


def _blocked_channel(channel, reason):
    return {"channel": channel, "blocked": True, "reason": reason, "prompt": "", "prompt_sha256": "",
            "images": [], "image_sha256": [], "params": {}, "warnings": []}


def _describe_probe(channel, payload, package_entry, subject, style_reference, aspect_ratio,
                    gemini_cfg, gpt_cfg):
    warnings = []
    if channel == "gemini_direct":
        prompt = str(payload.get("prompt") or "")
        images = list(payload.get("image_paths") or [])
        params = {"model": payload.get("model"), "aspect_ratio": payload.get("aspect_ratio"),
                  "resolution": payload.get("resolution"), "api_type": payload.get("api_type"),
                  "face_quality_boost": payload.get("face_quality_boost"),
                  "model_source": "apis.aigc2d.model",
                  "configured_model": str(gemini_cfg.get("model") or "")}
        source = ("StyleIterativeWorkerThread._assemble_test_generation_requests"
                  " → generate_image_aigc2d(prompt=compose_style_prompt(...))")
    elif channel == "gpt_first_pass":
        prompt = str(payload.get("prompt") or "")
        images = list(payload.get("image_paths") or [])
        from modules.others.api_backend import normalize_gpt_image2_size
        size = normalize_gpt_image2_size(size=payload.get("size"), aspect_ratio=payload.get("aspect_ratio"))
        params = {"model": payload.get("model"), "size": size, "api_type": payload.get("api_type"),
                  "mode": payload.get("mode"), "aspect_ratio": payload.get("aspect_ratio"),
                  "quality": payload.get("quality") or "(未显式传，取节点配置)",
                  "configured_size": str(gpt_cfg.get("size") or ""), "model_source": "apis.aigc-2d-gpt.model",
                  "configured_model": str(gpt_cfg.get("model") or "")}
        source = ("StyleIterativeWorkerThread._assemble_test_generation_requests"
                  " → utils.analysis_gen.build_gpt_image_request(...)")
    else:
        prompt = str(payload.get("prompt") or "")
        images = list(payload.get("source_paths") or []) + list(payload.get("extra_reference_paths") or [])
        params = {"model": payload.get("model"), "resolution": payload.get("resolution"),
                  "aspect_ratio": payload.get("aspect_ratio"), "repeat": payload.get("repeat"),
                  "reference_mode": payload.get("reference_mode"),
                  "style_clauses_count": len(payload.get("style_clauses") or []),
                  "clauses_without_image": (payload.get("call_kwargs") or {}).get("clauses_without_image"),
                  "prompt_source": payload.get("system_prompt_source"),
                  "firmware_source": payload.get("firmware_source")}
        source = "utils.post_process.assemble_repaint_request(...)（预检与实际发送同一份）"

    if channel == "gpt_first_pass" and len(images) == 1:
        warnings.append("首图只挂画风参考图：不做 edits、不挂素材图（原运行同条件）")
    if channel == "gpt_first_pass" and payload.get("quality") is None:
        warnings.append("未显式传 quality，取 apis.aigc-2d-gpt.quality 节点配置")
    if channel == "gpt_repainted":
        if params.get("reference_mode") != "none":
            warnings.append(f"重绘参考模式不是 none（实际 {params.get('reference_mode')}）")
        if not params.get("clauses_without_image"):
            warnings.append("clauses_without_image 未开启，画风条款可能未被当作文字规格发送")
        if len(images) != 1:
            warnings.append(f"重绘发送了 {len(images)} 张图，文字模式应当只发送本次 GPT 首图")
        if warnings:
            raise ValidationError("重绘模式不符合「文字模式」要求：" + "；".join(warnings))

    info = [{"index": index, "path": os.path.abspath(path), "sha256": file_hash(path),
             "bytes": os.path.getsize(path), "role": _image_role(channel, index, len(images), path,
                                                                  style_reference)}
            for index, path in enumerate(images)]
    entry = {
        "channel": channel,
        "blocked": False,
        "source": source,
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "prompt_sha256": text_hash(prompt),
        "prompt_role_note": _prompt_role_note(channel),
        "images": info,
        "image_sha256": [item["sha256"] for item in info],
        "params": params,
        "clauses": [str(c).strip() for c in (package_entry["package"].get("gemini_repaint_clauses") or [])]
                   if channel != "gemini_direct" else [],
        "repaint_clauses_source_round": _clause_source(package_entry),
        "warnings": warnings,
        "reference_mode": "none" if channel == "gpt_repainted" else "",
        "motif_enabled": False,
        "motif_prompt_chars": 0,
    }
    if channel == "gpt_first_pass":
        entry["content_chars"] = payload.get("content_chars")
        entry["style_chars"] = payload.get("style_chars")
    return entry


def _clause_source(package_entry):
    """重绘条款的「来源」标签：合并模式给轮次，复核模式给候选 JSON 内的字段名。"""
    source = (package_entry.get("field_sources") or {}).get("repaint_clauses") or {}
    return str(source.get("round") or source.get("field") or source.get("package_key") or "?")


def _prompt_role_note(channel):
    if channel == "gemini_direct":
        return "画风完整说明（候选包 gemini_full_prompt）+ 固定测试主体 + 画风参考图角色分工"
    if channel == "gpt_first_pass":
        return "prompt_gpt → 内容锚 → 参考图排除句 → RENDERING LANGUAGE（画风条款）→ 画风参考图"
    return "重绘固件 + STYLE LANGUAGE 条款（文字模式，不发送画风参考图）+ 身份锁"


def _image_role(channel, index, total, path, style_reference):
    if channel == "gpt_repainted":
        return "source（本次 GPT 首图）" if index == 0 else f"reference[{index}]"
    if os.path.abspath(path) == os.path.abspath(style_reference):
        return "style reference（测试画风参考图）"
    return f"input[{index}]"


# --------------------------------------------------------------------------- #
# 生图
# --------------------------------------------------------------------------- #


def _attempt_dir(task, key, attempt):
    return os.path.join(task.output_root, task.subject_label, "generated", key, f"attempt-{attempt}")


# 真正决定「发出去的字节」的字段：这些没变就不必再花一次钱。
# 刻意只放**传输内容**：source 说明、prompt_chars、log_lines 之类只影响审计文本，
# 不影响实际请求；把它们算进来会让 A 与 B 这种逐字相同的请求被误判成不同而重复生图。
def effective_request_manifest(channels):
    """把三路通道折算成「实际传输内容」清单（跨分组复用与审计都用它）。"""
    manifest = {}
    for item in channels:
        if item.get("blocked"):
            manifest[item.get("channel")] = {"blocked": True, "reason": item.get("reason")}
            continue
        params = item.get("params") or {}
        manifest[item.get("channel")] = {
            "prompt_sha256": item.get("prompt_sha256"),
            "prompt_chars": item.get("prompt_chars"),
            "images": list(item.get("image_sha256") or []),
            "model": params.get("model"),
            "resolution": params.get("resolution"),
            "size": params.get("size"),
            "aspect_ratio": params.get("aspect_ratio"),
            "quality": params.get("quality"),
            "mode": params.get("mode"),
            "reference_mode": params.get("reference_mode"),
            "scope": params.get("scope"),
            "clauses_without_image": params.get("clauses_without_image"),
            "repeat": params.get("repeat"),
            "face_quality_boost": params.get("face_quality_boost"),
        }
    return manifest


def request_digest(channels):
    return text_hash(json.dumps(effective_request_manifest(channels), ensure_ascii=False,
                                sort_keys=True))


def channel_digests(channels):
    manifest = effective_request_manifest(channels)
    return {name: text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True))
            for name, value in manifest.items()}


def _collect_outputs(directory):
    names = sorted(os.listdir(directory)) if os.path.isdir(directory) else []
    gemini, gpt_first = [], []
    for name in names:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        lower = name.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            if "gemini-direct" in lower:
                gemini.append(path)
            elif "gpt-first" in lower and "-final-rp" not in lower:
                gpt_first.append(path)
    return gemini, gpt_first


def _final_repaint(directory):
    """真正的重绘产物；**排除 `-partial` 回退件**。

    `utils.post_process.run_pipeline` 在失败时会「把最后成功的状态写成一个带 `-partial` 的
    final 文件」，好让调用方还有东西可用。那个文件是**源图的重编码副本，不是重绘结果**：
    真实事故（2026-10-05）—— 复核对照里 `gpt_repainted` 通道因此被判成功，产物哈希与
    `gpt_first_pass` 完全相同；若不拦住，视觉比较会把「没重绘过」的图当成重绘候选评分。
    所以这里只在文件名里筛真正的产物，失败由 manifest 状态判定（见 `_repaint_step_failed`）。
    """
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, name) for name in sorted(os.listdir(directory))
            if "-final-rp" in name.lower() and "partial" not in name.lower()
            and name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]


def _repaint_step_failed(directory, source_path, stage="repaint"):
    """读 pipeline manifest：该源图的这一步是否记为失败。

    只看「有没有产物文件」是不够的 —— 产物可能是上面的 partial 回退件。
    manifest 里 `status=failed` 才说明这一步真的没跑成。
    """
    manifest_path = os.path.join(directory, "gpt-repaint-steps", "pipeline-manifest.json")
    if not os.path.isfile(manifest_path):
        return False
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError):
        return False
    target = os.path.abspath(str(source_path))
    for item in manifest.get("items") or []:
        if os.path.abspath(str(item.get("source") or "")) != target:
            continue
        for step in item.get("steps") or []:
            if step.get("key") == stage and step.get("status") == "failed":
                return str(step.get("error") or "工序失败")
    return False


def _run_missing_channels(worker, package_entry, missing, generated, directory, file_prefix,
                          style_reference, subject, model_overrides=None):
    """只跑缺失通道；已成功的通道直接沿用，绝不重发请求。

    组装仍然走 `worker._assemble_test_generation_requests`（与顺序实跑同一段代码），
    这里只负责「挑着执行」，不自己拼提示词。
    """
    from modules.others.api_backend import generate_image_aigc2d, generate_image_aigc2d_gpt
    from utils.analysis_gen import run_gpt_image_pipeline
    from utils.styles import motif_prompt_from_clauses

    variants = package_entry["package"]
    motif_prompt = motif_prompt_from_clauses(variants.get("optional_motifs") or [])
    aspect_ratio = _reference_aspect_ratio(style_reference)
    # 组装可能整体抛错（例如节点没配模型）；那条路本身也是「不发请求」，按 blocked 处理
    assembled = {}
    assembly_error = ""
    try:
        assembled = worker._assemble_test_generation_requests(variants, style_reference, aspect_ratio,
                                                              motif_prompt,
                                                              model_overrides=model_overrides)
    except Exception as exc:  # noqa: BLE001
        assembly_error = f"{type(exc).__name__}: {exc}"
    assembly_error = assembly_error or str(assembled.get("gpt_error") or "")
    worker._test_generation_errors = {}
    cancel_check = lambda: False
    outputs = {}

    if "gemini_direct" in missing:
        payload = assembled.get("gemini_direct")
        if not payload:
            worker._test_generation_errors["gemini_direct"] = assembly_error or "组装失败"
            outputs["gemini_direct"] = []
        else:
            try:
                outputs["gemini_direct"] = generate_image_aigc2d(
                    prompt=payload["prompt"], image_paths=payload["image_paths"] or None,
                    model=payload["model"], aspect_ratio=aspect_ratio,
                    resolution=payload["resolution"], api_type="aigc2d", save_sub_dir=directory,
                    file_prefix=f"{file_prefix}-gemini-direct", face_quality_boost=False,
                    cancel_check=cancel_check, log_callback=worker.log_signal.emit) or []
            except Exception as exc:  # noqa: BLE001
                worker._test_generation_errors["gemini_direct"] = f"{type(exc).__name__}: {exc}"
                outputs["gemini_direct"] = []

    first_pass = list(generated.get("gpt_first_pass") or [])
    if "gpt_first_pass" in missing:
        payload = assembled.get("gpt_first_pass")
        if not payload:
            worker._test_generation_errors["gpt_first_pass"] = assembly_error or "组装失败"
            outputs["gpt_first_pass"] = []
        else:
            try:
                outputs["gpt_first_pass"] = generate_image_aigc2d_gpt(
                    prompt=payload["prompt"], image_paths=payload["image_paths"],
                    model=payload["model"], aspect_ratio=aspect_ratio,
                    api_type="aigc-2d-gpt", save_sub_dir=directory,
                    file_prefix=f"{file_prefix}-gpt-first", mode="generate",
                    quality=payload.get("quality"), size=payload.get("size"),
                    cancel_check=cancel_check, log_callback=worker.log_signal.emit) or []
                first_pass = list(outputs["gpt_first_pass"]) or first_pass
            except Exception as exc:  # noqa: BLE001
                worker._test_generation_errors["gpt_first_pass"] = f"{type(exc).__name__}: {exc}"
                outputs["gpt_first_pass"] = []

    if "gpt_repainted" in missing:
        repaint = assembled.get("gpt_repaint") or {}
        sources = [path for path in first_pass if os.path.isfile(path)]
        if not sources:
            worker._test_generation_errors["gpt_repainted"] = "GPT 首图未成功，无法进行重绘测试"
            outputs["gpt_repainted"] = []
        else:
            try:
                produced = run_gpt_image_pipeline(
                    sources, repaint.get("steps") or {}, final_dir=directory,
                    work_dir=os.path.join(directory, "gpt-repaint-steps"),
                    style_ref_path=repaint.get("style_ref_path"), style_clauses=repaint.get("style_clauses"),
                    log_callback=worker.log_signal.emit) or []
                # 管道可能返回「失败回退的 partial 副本」：那不算重绘成功。
                failure = _repaint_step_failed(directory, sources[0])
                usable = [p for p in produced if os.path.isfile(p) and "partial" not in os.path.basename(p).lower()]
                if failure or not usable:
                    worker._test_generation_errors["gpt_repainted"] = (
                        f"未取得可信的完成重绘产物，已按失败记录：{failure or '仅有 partial 或无有效文件'}")
                    outputs["gpt_repainted"] = []
                else:
                    outputs["gpt_repainted"] = usable
            except Exception as exc:  # noqa: BLE001
                worker._test_generation_errors["gpt_repainted"] = f"{type(exc).__name__}: {exc}"
                outputs["gpt_repainted"] = []
    return outputs


def run_attempt(task, key, package_entry, attempt, subject, style_reference, images, progress=None,
                records=None):
    """跑一次三路测试；已成功的产物按通道哈希复用，不重复收费。

    复用有两层：① 同一目录上次跑成功过（断点续跑，按通道粒度补跑）；② 同主体内另一个分组的
    **该通道实际传输内容**逐字相同（例如包 B 的首图与重绘条款都取自第 2 轮，与包 A 一致）——
    第二层覆盖的是实际发送的提示词，不是只比配置字段；复用关系写进 attempt.json 并在报告里列出。
    """
    directory = _attempt_dir(task, key, attempt)
    marker_path = os.path.join(directory, "attempt.json")
    digests = dict(package_entry["precheck"].get("channel_digests") or {})
    if not digests:
        digests = {name: text_hash(json.dumps(name)) for name in PROBE_CHANNELS}
    digest = package_entry["precheck"]["request_digest"]

    # ① 断点续跑：本目录上次成功的通道直接复用，只补跑缺的
    generated = {name: [] for name in PROBE_CHANNELS}
    previous = {}
    if task.reuse_cache and os.path.isfile(marker_path):
        previous = read_json(marker_path)
        if previous.get("channel_digests") == digests:
            for name in PROBE_CHANNELS:
                reuse_info = (previous.get("channel_reused") or {}).get(name) or {}
                if reuse_info and reuse_info.get("attempt") != attempt:
                    continue
                files = list((previous.get("generated_files") or {}).get(name) or [])
                expected_hashes = (previous.get("output_sha256") or {}).get(name)
                if expected_hashes and expected_hashes != [file_hash(p) for p in files if os.path.isfile(p)]:
                    continue
                if name == "gpt_repainted" and previous.get("repaint_source_sha256") is not None:
                    if previous["repaint_source_sha256"] != [file_hash(p) for p in generated["gpt_first_pass"]]:
                        continue
                # 旧记录可能把「失败回退的 partial 副本」写成了重绘成功（2026-10-05 修复）：
                # 带 partial 的文件一律不算可复用的重绘产物，必须重新跑。
                if name == "gpt_repainted":
                    files = [p for p in files if "partial" not in os.path.basename(p).lower()]
                    if any(_repaint_step_failed(directory, p) for p in generated["gpt_first_pass"]):
                        files = []
                if files and all(os.path.isfile(path) for path in files) \
                        and ((previous.get("channel_status") or {}).get(name) or {}).get("status") == "success":
                    generated[name] = files
                    if progress:
                        progress(f"[复用] 包 {key} 第 {attempt} 次 / {name}：上次成功产物仍在，跳过")

    # ② 跨分组复用：实际传输内容逐字相同的通道不再重复收费
    reused = {name: info for name, info in (previous.get("channel_reused") or {}).items()
              if generated[name]}
    if task.reuse_cache and records:
        for name in PROBE_CHANNELS:
            if name == "gpt_repainted":
                continue  # 预检占位图不能决定实际重绘的缓存身份。
            if generated[name] or digests.get(name) in (None, ""):
                continue
            shared = _reuse_source(records, name, digests[name], attempt=attempt)
            if not shared:
                continue
            source = records[(shared["package"], shared["attempt"])]
            files = list((source.get("generated_files") or {}).get(name) or [])
            if files and all(os.path.isfile(path) for path in files):
                generated[name] = files
                reused[name] = {**shared, "channel_digest": digests[name]}
                if progress:
                    progress(f"[复用] 包 {key} 第 {attempt} 次 / {name}：与包 {shared['package']} "
                             f"第 {shared['attempt']} 次的实际请求逐字一致，不再重复生图")

    from modules.others import api_backend
    missing = [name for name in PROBE_CHANNELS if not generated[name]]
    errors = {}
    logs = []
    elapsed = 0.0
    gemini_cfg = api_backend.get_api_config(api_type="aigc2d")
    gpt_cfg = api_backend.get_api_config(api_type="aigc-2d-gpt")
    blocked = set()
    for item in package_entry["precheck"]["channels"]:
        if item.get("blocked"):
            blocked.add(item["channel"])
    if missing:
        os.makedirs(directory, exist_ok=True)
        worker = StyleIterativeWorkerThread(
            images, api_key="", base_url="", model_name="",
            total_rounds=0, images_per_round=0, output_dir=directory,
            enable_test_gen=True, test_prompt=subject, img_api_type="aigc-2d-gpt",
            img_aspect_ratio="", file_prefix=f"{key}-a{attempt}", test_style_ref_path=style_reference,
            repaint_reference_mode="none")
        worker.log_signal.connect(logs.append)
        worker.model_overrides = dict(task.model_overrides)
        started = time.time()
        try:
            outputs = {}
            for name in missing:
                if name == "gpt_repainted":
                    source_hashes = [file_hash(p) for p in generated["gpt_first_pass"]]
                    runtime_digest = text_hash(json.dumps({"template": digests[name],
                                                          "source_sha256": source_hashes}, sort_keys=True))
                    shared = _reuse_source(records or {}, name, runtime_digest, attempt=attempt)
                    if task.reuse_cache and shared:
                        generated[name] = list(records[(shared["package"], shared["attempt"])]["generated_files"][name])
                        reused[name] = {**shared, "channel_digest": runtime_digest}
                        continue
                result = _run_missing_channels(worker, package_entry, [name], generated, directory,
                                               f"{key}-a{attempt}", style_reference, subject,
                                               task.model_overrides)
                outputs.update(result)
                generated[name] = list(result.get(name) or [])
                errors.update(getattr(worker, "_test_generation_errors", {}) or {})
                atomic_json(marker_path, {"channel_digests": digests, "generated_files": generated,
                            "channel_reused": reused, "channel_status": {
                                channel: {"status": "success" if files else "failed"}
                                for channel, files in generated.items()},
                            "output_sha256": {channel: [file_hash(p) for p in files]
                                              for channel, files in generated.items()},
                            "repaint_source_sha256": [file_hash(p) for p in generated["gpt_first_pass"]]})
        except Exception as exc:  # noqa: BLE001 - 取消/未预期错误都要留下记录
            logs.append(f"{type(exc).__name__}: {exc}")
            outputs = {}
        elapsed = round(time.time() - started, 2)
        errors.update(getattr(worker, "_test_generation_errors", {}) or {})
        for name in missing:
            files = list(outputs.get(name) or [])
            if files:
                generated[name] = files

    channel_status = {}
    for name in PROBE_CHANNELS:
        files = generated[name]
        if files:
            status = "success"
            error = ""
        elif name in blocked:
            status = "not_run"
            error = next((item.get("reason") or "" for item in package_entry["precheck"]["channels"]
                          if item.get("channel") == name), "")
        elif name == "gpt_repainted" and not generated.get("gpt_first_pass"):
            status, error = "not_run", "GPT 首图未成功，无法进行重绘测试"
        else:
            status = "failed"
            error = errors.get(name, "接口未返回图片；此通道未通过测试")
        channel_status[name] = {"status": status, "image_count": len(files), "error": error}
    passed = sum(bool(generated[name]) for name in PROBE_CHANNELS)
    record = {
        "prompts_used": package_entry["package"].get("master_prompt") or "",
        "prompt_variants": package_entry["package"],
        "test_prompt": subject,
        "style_reference": os.path.abspath(style_reference),
        "generated_files": generated,
        "status": "success" if passed == 3 else ("partial" if passed else "failed"),
        "channel_status": channel_status,
        "repaint_reference_mode": "none",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "package_key": key, "attempt": attempt, "subject_label": task.subject_label,
        "request_digest": digest, "channel_digests": digests, "channel_reused": reused,
        "elapsed_seconds": elapsed,
        "models": {"gemini": str(gemini_cfg.get("model") or ""), "gpt": str(gpt_cfg.get("model") or "")},
        "logs": logs[-200:], "errors": errors,
    }
    source_hashes = [file_hash(p) for p in generated["gpt_first_pass"]]
    record["repaint_source_sha256"] = source_hashes
    record["output_sha256"] = {name: [file_hash(p) for p in files] for name, files in generated.items()}
    record["runtime_channel_digests"] = {**digests, "gpt_repainted": text_hash(json.dumps({
        "template": digests["gpt_repainted"], "source_sha256": source_hashes}, sort_keys=True))}
    atomic_json(marker_path, record)
    if progress:
        summary = ", ".join(f"{name}={len(generated[name])}" for name in PROBE_CHANNELS)
        progress(f"[生图] 包 {key} 第 {attempt} 次：{summary}（{elapsed}s，"
                 f"复用 {len(reused)} 条通道）" + (f"；错误 {errors}" if errors else ""))
    return record


def _reuse_source(records, channel, channel_digest, attempt=None):
    """找出**该通道实际传输内容逐字相同**、且已有成功产物的更早分组（同主体内）。

    注意比的是通道级哈希：包 B 的 Gemini 直出用了第 5 轮描述，与 A 不同；
    但它的 GPT 首图与重绘条款都来自第 2 轮，实际请求与 A 逐字一致 —— 这种才算可复用。
    复用关系写进 attempt.json 与报告，不冒充成「新生成」。
    """
    for (other_key, other_attempt), record in sorted(records.items()):
        if attempt is None or other_attempt != attempt:
            continue
        if record.get("status") not in ("ok", "success", "partial"):
            continue
        if ((record.get("channel_status") or {}).get(channel) or {}).get("status", "success") != "success":
            continue
        hashes = record.get("runtime_channel_digests") if channel == "gpt_repainted" else record.get("channel_digests")
        if (hashes or {}).get(channel) != channel_digest:
            continue
        files = list((record.get("generated_files") or {}).get(channel) or [])
        if channel == "gpt_repainted" and any("partial" in os.path.basename(p).lower() for p in files):
            continue
        if files and all(os.path.isfile(path) for path in files):
            return {"package": other_key, "attempt": other_attempt,
                    "reason": "该通道实际传输内容逐字一致（提示词哈希、输入图哈希、模型与尺寸全部相同）"}
    return None


def _marker_outputs(marker):
    return {name: list((marker.get("generated_files") or {}).get(name) or []) for name in PROBE_CHANNELS}


# --------------------------------------------------------------------------- #
# 训练结果状态文件（app 可读）
# --------------------------------------------------------------------------- #


def build_precheck_document(task, entries, images, subject, dataset_images=None, progress=None):
    """无联网预检文档：每包三条通道的生效提示词、输入图顺序与哈希、模型与参数。

    预检与实际发送共用同一段组装代码，所以「预检里那份提示词」就是「实跑要发出去的那份」。
    记录里只放 key 来源变量名，不放凭据值。
    """
    sources = dataset_images if dataset_images is not None else images
    result = {"generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "source": "run-record candidates" if isinstance(task, ReviewCandidateTask) else task.source,
              "reference_source": task.source if isinstance(task, ReviewCandidateTask) else "",
              "output_root": task.output_root, "subject_label": task.subject_label,
              "subject": subject, "style_reference": task.style_reference(),
              "dataset_images": [{"index": index, "path": path, "sha256": file_hash(path)}
                                 for index, path in enumerate(sources)],
              "package_keys": list(entries), "packages": {}}
    if isinstance(task, ReviewCandidateTask):
        result["candidates"] = [
            {"key": item["key"], "label": item.get("label") or item["key"],
             "path": os.path.abspath(str(item["path"])), "sha256": file_hash(str(item["path"])),
             "package_key": item.get("package_key") or ""}
            for item in task.candidates]
        result["dataset_selection"] = task.dataset_selection()
    for key, entry in entries.items():
        pre = build_precheck(task, key, entry, subject, task.style_reference(), images,
                             progress=progress)
        entry["precheck"] = pre
        result["packages"][key] = pre
    result["assembler"] = ("StyleIterativeWorkerThread._assemble_test_generation_requests"
                           "（预检与实跑同一段组装代码）")
    result["repaint_assembler"] = "utils.post_process.assemble_repaint_request（run_pipeline 发送前那一份）"
    result["network_requests_sent"] = 0
    return result


def build_review_state(task, entries, records):
    """构造 app 可打开、比较/深度指标可直接吃的状态；候选 ID 用「包名-第n次」前缀。

    与 `build_state`（myc0t0xin 专用，候选必须叫 A/B/C）并列存在：
    复核模式包名可自定义，所以 ID 形如 `cccccanh-round5-original-attempt1/gpt_repainted/0`。
    """
    subject_dir = os.path.join(task.output_root, task.subject_label)
    test_images = {}
    state_dataset_selection = task.dataset_selection()
    style_reference = os.path.abspath(task.style_reference())
    subject = task.subject_prompt()
    dataset = [os.path.abspath(p) for p in task.dataset_images()]
    for item in task.candidates:
        key = item["key"]
        entry = entries[key]
        for attempt in range(1, task.repetitions + 1):
            record = records.get((key, attempt))
            if not record:
                continue
            stage = f"{key}-attempt{attempt}"
            test_images[stage] = {
                "prompts_used": entry["package"].get("master_prompt") or "",
                "prompt_variants": entry["package"],
                "test_prompt": subject,
                "style_reference": style_reference,
                "generated_files": {name: list((record.get("generated_files") or {}).get(name) or [])
                                    for name in PROBE_CHANNELS},
                "status": record.get("status") or "failed",
                "channel_status": record.get("channel_status") or {},
                "repaint_reference_mode": "none",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "package_key": key,
                "package_label": entry.get("label") or key,
                "candidate_json": entry["field_sources"]["candidate_json"]["path"],
                "candidate_sha256": entry["field_sources"]["candidate_json"]["sha256"],
                "source_shape": entry.get("source_shape"),
                "attempt": attempt,
                "subject_label": task.subject_label,
                "request_digest": entry["precheck"]["request_digest"],
                "field_sources": entry["field_sources"],
                "normalization": entry["normalization"],
                "effective": entry["effective"],
                "motif_decision": entry.get("motif_decision"),
                "elapsed_seconds": record.get("elapsed_seconds"),
                "errors": record.get("errors") or {},
                "runtime_channel_digests": record.get("runtime_channel_digests") or {},
                "output_sha256": record.get("output_sha256") or {},
                "repaint_source_sha256": record.get("repaint_source_sha256") or [],
                "reused_from": record.get("channel_reused") or {},
            }
    name = str(getattr(task, "state_name", "") or "review_style_iter_result.json")
    return {
        "version": "2.0",
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {"image_count": len(dataset), "images": dataset},
        "dataset_selection": state_dataset_selection,
        "parameters": {"total_rounds": 0, "images_per_round": 0, "repaint_reference_mode": "none",
                       "validation_stage": task.stage, "subject_label": task.subject_label},
        "iterations": [{"step": 0, "type": "run_record_candidate_review", "round": 0,
                        "art_style_prompts": "", "model_used": "candidate json (no analysis rounds)",
                        "note": "整包取自各候选 JSON 记录的 prompt 包；没有重跑分析轮次、没有拼接两段画风描述",
                        "candidates": [{"key": item["key"], "path": os.path.abspath(str(item["path"])),
                                        "sha256": file_hash(str(item["path"]))}
                                       for item in task.candidates]}],
        "final_art_style_prompts": "",
        "final_test_images": {},
        "file_prefix": name[:-len("_style_iter_result.json")] if name.endswith("_style_iter_result.json") else name,
        "test_images": test_images,
        "analysis_warnings": [],
        "review_validation": {
            "mode": "run_record_candidates", "subject_label": task.subject_label,
            "subject": subject, "stage": task.stage, "style_reference": style_reference,
            "candidates": [{"key": item["key"], "label": item.get("label") or item["key"],
                            "path": os.path.abspath(str(item["path"]))} for item in task.candidates],
            "note": "本次只出图与比较，不重跑分析、不改候选 JSON、不写入 conf/config-styles.json",
        },
    }


def review_state_path(task):
    return os.path.join(task.output_root, task.subject_label,
                        str(getattr(task, "state_name", "") or "review_style_iter_result.json"))


def build_state(task, package_entries, records):
    """构造 app「多图画风提取」能打开的 state；比较/深度指标都在同一目录。"""
    source_state = task.source_state()
    subject_dir = os.path.join(task.output_root, task.subject_label)
    test_images = {}
    for key in ("A", "B", "C"):
        if key not in package_entries:
            continue
        entry = package_entries[key]
        for attempt in range(1, task.repetitions + 1):
            record = records.get((key, attempt))
            if not record:
                continue
            stage = f"package{key}-attempt{attempt}"
            stored = {
                "prompts_used": entry["package"].get("master_prompt") or "",
                "prompt_variants": entry["package"],
                "test_prompt": task.subject_prompt(),
                "style_reference": os.path.abspath(task.style_reference()),
                "generated_files": {name: list((record.get("generated_files") or {}).get(name) or [])
                                    for name in PROBE_CHANNELS},
                "status": record.get("status") or "failed",
                "channel_status": record.get("channel_status") or {},
                "repaint_reference_mode": "none",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "package_key": key,
                "attempt": attempt,
                "subject_label": task.subject_label,
                "request_digest": entry["precheck"]["request_digest"],
                "field_sources": entry["field_sources"],
                "normalization": entry["normalization"],
                "elapsed_seconds": record.get("elapsed_seconds"),
                "errors": record.get("errors") or {},
                "reused_from": record.get("reused"),
            }
            test_images[stage] = stored
    state = {
        "version": "2.0",
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {"image_count": len(task.dataset_images()),
                    "images": [os.path.abspath(p) for p in task.dataset_images()]},
        "parameters": {"total_rounds": 0, "images_per_round": 0, "repaint_reference_mode": "none",
                       "validation_stage": task.stage},
        "iterations": [{"step": 0, "type": "merged_validation_seed", "round": 0,
                        "art_style_prompts": "", "model_used": "field-merge (no analysis rounds)",
                        "source_json": task.source,
                        "source_sha256": file_hash(task.source),
                        "note": "本文件不是重跑分析的结果：字段按轮次复制，未拼接、未压缩"}],
        "final_art_style_prompts": "",
        "final_test_images": {},       # 本任务没有"终审版本"这个阶段：字段按轮次复制，不重跑分析
        "file_prefix": f"myc0t0xin-merged-validation-{task.subject_label}",
        "test_images": test_images,
        "dataset_selection": source_state.get("dataset_selection"),
        "analysis_warnings": [],
        "merged_validation": {"subject_label": task.subject_label, "stage": task.stage,
                              "source_json": task.source, "packages": list(task.package_keys),
                              "note": "本文件由 tools/style_merge_validate.py 生成：字段按轮次复制、"
                                      "没有重跑分析轮次、没有拼接两段画风描述"},
    }
    seen = {}
    aliases = []
    for stage, record in test_images.items():
        record["aliased_generated_files"] = {}
        for channel, paths in record["generated_files"].items():
            unique = []
            for path in paths:
                sha = file_hash(path)
                if sha in seen:
                    aliases.append({"stage": stage, "channel": channel, "path": path,
                                    "sha256": sha, "canonical": seen[sha]})
                    record["aliased_generated_files"].setdefault(channel, []).append(path)
                else:
                    seen[sha] = {"stage": stage, "channel": channel, "path": path}
                    unique.append(path)
            record["generated_files"][channel] = unique
    state["sample_inventory"] = {"unique_image_count": len(seen), "alias_count": len(aliases),
                                 "aliases": aliases, "method": "SHA-256 of image file bytes"}
    return state


def state_path(task):
    if isinstance(task, ReviewCandidateTask):
        return review_state_path(task)
    return os.path.join(task.output_root, task.subject_label, "myc0t0xin_style_iter_result.json")


def save_state(task, state):
    path = state_path(task)
    atomic_json(path, state)
    return path


# --------------------------------------------------------------------------- #
# 深度指标
# --------------------------------------------------------------------------- #


def run_deep_metrics(task, state_file, device="auto-npu", progress=None):
    """按 app 的既有链路计算本地指标（独立进程，NPU 优先）。"""
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-X", "utf8", "-u", str(root / "tools" / "style_metrics_verify.py"),
               "--comparison-state", os.path.abspath(state_file), "--device", device]
    if progress:
        progress("[深度指标] " + " ".join(command[1:]))
    process = subprocess.run(command, cwd=root, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
    log_path = os.path.join(os.path.dirname(os.path.abspath(state_file)),
                            "deep-feature-comparison.log")
    with open(log_path, "w", encoding="utf-8") as stream:
        stream.write(process.stdout or "")
        stream.write("\n--- stderr ---\n")
        stream.write(process.stderr or "")
    if progress:
        for line in (process.stdout or "").splitlines()[-12:]:
            progress("    " + line)
    if process.returncode:
        raise ValidationError(f"深度指标进程退出码 {process.returncode}；详情见 {log_path}")
    latest = read_json(state_file)
    return latest.get("deep_feature_comparison") or {}


# --------------------------------------------------------------------------- #
# 交付物
# --------------------------------------------------------------------------- #


def cost_report(package_entries, task, channels=PROBE_CHANNELS, attempts=None):
    from utils import cost_estimate
    rows = []
    total_usd = 0.0
    net_usd = 0.0
    for key in task.package_keys:
        if key not in package_entries:
            continue
        entry = package_entries[key]
        channels_by_name = {item["channel"]: item for item in entry["precheck"]["channels"]}
        for channel in channels:
            item = channels_by_name.get(channel) or {}
            if item.get("blocked"):
                rows.append({"package": key, "channel": channel, "usd": 0.0, "runs": 0, "paid_runs": 0,
                             "note": "未发送请求：" + str(item.get("reason"))})
                continue
            params = item.get("params") or {}
            chars = int(item.get("prompt_chars") or 0)
            if channel == "gemini_direct":
                estimate = cost_estimate.estimate_gemini_repaint_cost(
                    model=str(params.get("model") or "gemini-3-pro-image-preview"), repeat=1)
            else:
                estimate = cost_estimate.estimate_gpt_image_cost(
                    chars, size=str(params.get("size") or "1024x1536"),
                    quality=str(params.get("quality") or "medium").split("（")[0].strip() or "medium",
                    ref_images=max(1, len(item.get("image_sha256") or [])),
                    model=str(params.get("model") or "gpt-image-2"))
            per_run = float(estimate.get("usd") or 0.0)
            generated_runs = task.repetitions
            reused_runs = 0
            if attempts is not None:
                generated_runs = 0
                for attempt in range(1, task.repetitions + 1):
                    record = attempts.get((key, attempt))
                    if not record:
                        continue
                    reuse = (record.get("channel_reused") or {}).get(channel)
                    status = ((record.get("channel_status") or {}).get(channel) or {}).get("status")
                    if reuse and status == "success":
                        reused_runs += 1
                        continue
                    if status in ("success", "failed", "partial"):
                        generated_runs += 1
            usd = per_run * task.repetitions
            total_usd += usd
            net_usd += per_run * generated_runs
            note = "本地估算（utils.cost_estimate 内置公式/缓存价目表），非账单"
            if reused_runs:
                note += f"；其中 {reused_runs} 次因该通道实际请求逐字一致而复用既有产物，未产生新费用"
            if attempts is not None and generated_runs == 0 and not reused_runs:
                note += "；本次没有该通道的实际执行记录"
            rows.append({"package": key, "channel": channel, "usd": round(usd, 5),
                         "net_usd": round(per_run * generated_runs, 5), "per_run_usd": round(per_run, 5),
                         "runs": task.repetitions, "paid_runs": generated_runs,
                         "billed": estimate.get("billed"),
                         "basis": {k: v for k, v in estimate.items() if k in
                                   ("model", "billed", "group", "group_ratio", "text_tokens",
                                    "image_input_tokens", "output_tokens", "price_per_call")},
                         "note": note})
    return {"rows": rows, "total_usd": round(total_usd, 5), "total_cny": round(total_usd * 7.2, 2),
            "net_usd": round(net_usd, 5), "net_cny": round(net_usd * 7.2, 2),
            "note": "按候选包逐通道估算：total_* 是不复用时的名义值，net_* 是已记录通道的本地估算"
                    "（复用不重复计费），不是账单或费用上限；未计入后端重试、失败请求是否计费、视觉复核与本地算力成本"}


def write_deliverables(task, package_entries, precheck, state, comparison, deep, attempts,
                       extra=None, proposed_package=None):
    subject_dir = os.path.dirname(state_path(task))
    os.makedirs(subject_dir, exist_ok=True)
    if "sample_inventory" not in state:
        seen, aliases = {}, []
        for stage, record in (state.get("test_images") or {}).items():
            for channel, paths in (record.get("generated_files") or {}).items():
                for path in paths:
                    sha = file_hash(path)
                    item = {"stage": stage, "channel": channel, "path": path}
                    if sha in seen:
                        aliases.append({**item, "sha256": sha, "canonical": seen[sha]})
                    else:
                        seen[sha] = item
        state["sample_inventory"] = {"unique_image_count": len(seen), "alias_count": len(aliases),
                                     "aliases": aliases, "method": "SHA-256 of image file bytes"}
    written = {}
    written["state"] = state_path(task)
    written["precheck"] = os.path.join(subject_dir, "precheck.json")
    atomic_json(written["precheck"], precheck)
    written["packages"] = os.path.join(subject_dir, "candidate-packages.json")
    atomic_json(written["packages"], {key: {"label": entry["label"], "field_sources": entry["field_sources"],
                                            "normalization": entry["normalization"],
                                            "effective": entry["effective"],
                                            "master_prompt_chars": len(entry["package"].get("master_prompt") or ""),
                                            "gemini_full_prompt": entry["package"].get("gemini_full_prompt"),
                                            "gpt_image_prompt": entry["package"].get("gpt_image_prompt"),
                                            "gemini_repaint_clauses": entry["package"].get("gemini_repaint_clauses")}
                                    for key, entry in package_entries.items()})
    written["requests"] = os.path.join(subject_dir, "request-audit.json")
    atomic_json(written["requests"], {
        "note": "packages 是预检模板；重绘占位图不代表真实输入。attempts 记录实际首图哈希与缓存身份；不含密钥",
        "assembler": precheck.get("assembler"),
        "packages": {key: {"request_digest": entry["precheck"]["request_digest"],
                           "channels": entry["precheck"]["channels"]}
                     for key, entry in package_entries.items()},
        "attempts": [{"package": key, "attempt": attempt, "status": record.get("status"),
                      "channel_status": record.get("channel_status"),
                      "channel_reused": record.get("channel_reused"),
                      "runtime_channel_digests": record.get("runtime_channel_digests"),
                      "repaint_source_sha256": record.get("repaint_source_sha256"),
                      "output_sha256": record.get("output_sha256"),
                      "generated_files": record.get("generated_files"),
                      "elapsed_seconds": record.get("elapsed_seconds"),
                      "models": record.get("models"), "errors": record.get("errors")}
                     for (key, attempt), record in sorted(attempts.items())],
    })
    written["sample_inventory"] = os.path.join(subject_dir, "sample-inventory.json")
    atomic_json(written["sample_inventory"], state.get("sample_inventory") or {})
    written["sample_summary"] = os.path.join(subject_dir, "independent-sample-summary.json")
    atomic_json(written["sample_summary"], summarize_samples(state, comparison, attempts))
    written["cost"] = os.path.join(subject_dir, "cost-estimate.json")
    atomic_json(written["cost"], cost_report(package_entries, task, attempts=attempts))
    written["proposed"] = os.path.join(subject_dir, "proposed-style-entry.json")
    atomic_json(written["proposed"], proposed_style_entry(task, package_entries, comparison, deep,
                                                        package_key=proposed_package))
    if extra:
        written.update(extra)
    return written


def _esc(value):
    import html
    return html.escape(str(value))


def _field_source_label(sources, field):
    snapshot = sources[field]
    return snapshot.get("round") or (sources.get("candidate_json") or {}).get("record_path") or snapshot.get("field", field)


def summarize_samples(state, comparison, attempts):
    """把配对复用的评分映射回各包，同时每包每路只按唯一图片计数。"""
    from modules.image_analysis.style_comparison import candidates_from_state
    rows = {row["id"]: row for row in (comparison or {}).get("rows") or []}
    scores_by_hash = {file_hash(candidate["path"]): rows.get(candidate["id"])
                      for candidate in candidates_from_state(state)}
    result = []
    for key in sorted({key for key, _ in attempts}):
        for channel in PROBE_CHANNELS:
            samples = {}
            for (package, attempt), record in sorted(attempts.items()):
                if package != key:
                    continue
                for path in (record.get("generated_files") or {}).get(channel) or []:
                    sha = file_hash(path)
                    row = scores_by_hash.get(sha) or {}
                    samples.setdefault(sha, {"attempt": attempt, "path": path, "sha256": sha,
                                             "candidate_id": row.get("id"), "total": row.get("total"),
                                             "eligible": row.get("eligible"), "gates": row.get("gates")})
            values = [float(s["total"]) for s in samples.values() if s["total"] is not None]
            result.append({"package": key, "channel": channel, "unique_samples": len(samples),
                           "reviewed": len(values),
                           "eligible": sum(s["eligible"] is True for s in samples.values()),
                           "total_mean": sum(values) / len(values) if values else None,
                           "total_min": min(values) if values else None,
                           "total_max": max(values) if values else None, "samples": list(samples.values())})
    return {"method": "每包每通道按图片 SHA-256 去重；等价配对样本映射同一评分；均值仅作描述统计，门禁独立列出",
            "rows": result}


def build_contact_sheet(task, attempts, thumb=380, columns=6):
    """把全部候选拼成一张接触表，供人工快速看图复核（不改变任何原始产物）。"""
    from PIL import Image, ImageDraw
    from modules.image_analysis.style_comparison import DIMENSIONS  # noqa: F401 - 仅用于一致性提示
    items = []
    for (key, attempt), record in sorted(attempts.items()):
        for channel in PROBE_CHANNELS:
            for path in (record.get("generated_files") or {}).get(channel) or []:
                if os.path.isfile(path):
                    items.append((f"{key} a{attempt} {channel}", path))
    if not items:
        return ""
    rows = (len(items) + columns - 1) // columns
    label_height = 22
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(items):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb, thumb))
            x = (index % columns) * thumb
            y = (index // columns) * (thumb + label_height)
            sheet.paste(image, (x + (thumb - image.width) // 2, y))
            draw.text((x + 4, y + thumb + 4), f"{index + 1}. {label}", fill="black")
    path = os.path.join(task.output_root, task.subject_label, "contact-sheet.jpg")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path, quality=88)
    return path


def _picture(path, max_dim=720, height=220):
    from utils.image_encoding import compress_and_encode_image
    mime, encoded = compress_and_encode_image(path, max_dim=max_dim, quality=85)
    return (f'<img loading="lazy" style="max-width:{max_dim}px;max-height:{height}px" '
            f'src="data:{mime};base64,{encoded}">')


def _attempt_matrix(entries, attempts, packages):
    """逐包逐通道的「已生成 / 已复核 / 未执行」矩阵。"""
    rows = []
    for key in packages:
        entry = entries[key]
        by_channel = {item["channel"]: item for item in entry["precheck"]["channels"]}
        for channel in PROBE_CHANNELS:
            item = by_channel.get(channel) or {}
            generated = failed = reused = 0
            for attempt in range(1, 20):
                record = attempts.get((key, attempt))
                if not record:
                    continue
                status = (record.get("channel_status") or {}).get(channel) or {}
                if (record.get("channel_reused") or {}).get(channel):
                    reused += 1
                if status.get("status") == "success":
                    generated += 1
                elif status.get("status") in ("failed", "partial"):
                    failed += 1
            rows.append({
                "package": key, "channel": channel,
                "field_valid": entry["normalization"]["gpt_image_prompt_valid"] if channel != "gemini_direct" else True,
                "assembled": not item.get("blocked", False),
                "generated": generated, "failed": failed, "reused": reused,
                "blocked_reason": item.get("reason", ""),
                "prompt_sha256": item.get("prompt_sha256", ""),
            })
    return rows


def write_validation_report(task, entries, precheck, state, comparison, deep, attempts,
                            contact_sheet=""):
    subject_dir = os.path.dirname(state_path(task))
    os.makedirs(subject_dir, exist_ok=True)
    recommendation = recommendation_from(comparison, deep)
    costs = cost_report(entries, task, attempts=attempts)
    matrix = _attempt_matrix(entries, attempts, task.package_keys)
    deep_rows = {row["id"]: row for row in (deep or {}).get("rows") or []}
    parts = [
        '<!doctype html><meta charset="utf-8"><title>画风候选验证报告</title>',
        '<style>body{font:15px/1.6 system-ui;margin:24px;background:#f6f6f6;color:#222}'
        'h1,h2,h3{line-height:1.3}article{background:#fff;padding:18px;margin:14px 0;border-radius:6px}'
        'table{border-collapse:collapse;width:100%;margin:8px 0}td,th{border:1px solid #bbb;padding:6px;'
        'vertical-align:top;font-size:13px}th{background:#eee}pre{white-space:pre-wrap;word-break:break-word;'
        'font-size:12px;background:#fafafa;padding:8px}img{border:1px solid #ddd;background:#fff}'
        '.ok{color:#0a7d24}.bad{color:#b00020}.warn{color:#a06000}figure{display:inline-block;margin:6px}'
        'figcaption{font-size:11px;max-width:240px;word-break:break-all}</style>',
        '<h1>画风候选验证</h1>',
        '<article><h2>一、这次到底做了什么</h2>',
        f'<p>源结果：<code>{_esc(task.source)}</code><br>源 SHA-256：<code>{_esc(file_hash(task.source))}</code></p>',
        f'<p>输出根目录：<code>{_esc(task.output_root)}</code>｜主体分组：<b>{_esc(task.subject_label)}</b>'
        f'｜固定测试主体：<code>{_esc(task.subject_prompt())}</code></p>',
        '<p><b>没有重跑 5 轮分析</b>：本报告里的候选包是「按字段复制指定轮次 + 重新校验」的结果，'
        '既没有把两段画风描述拼成一段，也没有把旧分数赋给新组合。三路测试的组装代码与 app 的多图画风提取一致'
        '（<code>StyleIterativeWorkerThread._generate_test_image</code>）。</p>',
        '<p><b>重绘模式变了</b>：旧运行的六张重绘是把完整画风图一起发出去的；本次统一改成文字模式'
        '（<code>reference_mode=none</code>、<code>clauses_without_image=true</code>、<code>scope=full</code>），'
        '只发送当次 GPT 首图。所以本次重绘<b>不是</b>旧完整参考图重绘的同条件复现。</p>',
        '<p><b>重绘条款也进入 GPT 首图</b>：测试逻辑把它们当 <code>extra_clauses</code> 放进首图请求，'
        '所以 A/B/C 的首图必须按包分别核验；下表给出每包每条通道的生效提示词哈希。</p></article>',
    ]

    # 参考图
    parts.append(f'<article><h2>二、原始参考图（{len(precheck.get("dataset_images") or [])} 张）</h2>')
    for index, item in enumerate(precheck.get("dataset_images") or []):
        parts.append('<figure>' + _picture(item["path"], max_dim=360, height=360)
                     + f'<figcaption>参考 {index}<br>{_esc(os.path.basename(item["path"]))}<br>'
                       f'{_esc(item["sha256"][:12])}</figcaption></figure>')
    parts.append(f'<p>测试画风参考图：<code>{_esc(precheck.get("style_reference"))}</code></p></article>')

    # 字段来源
    parts.append('<article><h2>三、候选包与逐字段来源</h2>'
                 '<table><tr><th>包</th><th>prompt（Gemini 完整描述）</th><th>prompt_gpt（GPT 短版）</th>'
                 '<th>repaint_clauses（重绘条款）</th><th>母题</th><th>校验</th></tr>')
    for key in task.package_keys:
        entry = entries[key]
        sources = entry["field_sources"]
        parts.append(
            f'<tr><td><b>{key}</b><br>{_esc(entry["label"])}</td>'
            f'<td>{_esc(_field_source_label(sources, "prompt"))}<br>{sources["prompt"]["chars"]} 字<br>'
            f'<code>{_esc(sources["prompt"]["sha256"][:16])}</code></td>'
            f'<td>{_esc(_field_source_label(sources, "prompt_gpt"))}<br>{sources["prompt_gpt"]["chars"]} 字<br>'
            f'<code>{_esc(sources["prompt_gpt"]["sha256"][:16])}</code></td>'
            f'<td>{_esc(_field_source_label(sources, "repaint_clauses"))}（{sources["repaint_clauses"]["count"]} 条）<br>'
            f'<code>{_esc(sources["repaint_clauses"]["sha256"][:16])}</code></td>'
            f'<td>关闭（空）</td>'
            f'<td>{"通过" if entry["normalization"]["gpt_image_prompt_valid"] else "GPT 短版不合格"}'
            + ("" if entry["normalization"]["gpt_image_prompt_valid"] else
               "<br>" + _esc("；".join(entry["normalization"]["gpt_image_prompt_errors"])))
            + '</td></tr>')
    parts.append('</table>'
                 '<p>复制校验：规范化后三个字段与来源轮次原文逐字比对，全部必须一致；'
                 '不一致就停止，绝不改 validity 布尔值绕过校验。</p>'
                 + '<pre>' + _esc(json.dumps({key: entries[key]["normalization"]["copy_checks"]
                                              for key in task.package_keys}, ensure_ascii=False, indent=2))
                 + '</pre></article>')

    # 顶层字段原文
    parts.append('<article><h2>四、候选字段原文（可逐字比对）</h2>')
    for key in task.package_keys:
        entry = entries[key]
        parts.append(f'<h3>包 {key}：{_esc(entry["label"])}</h3>')
        parts.append(f'<p>gemini_full_prompt（来源 {_esc(_field_source_label(entry["field_sources"], "prompt"))}）'
                     + '<details><summary>展开</summary><pre>'
                     + _esc(entry["package"].get("gemini_full_prompt")) + '</pre></details></p>')
        parts.append(f'<p>gpt_image_prompt（来源 {_esc(_field_source_label(entry["field_sources"], "prompt_gpt"))}）'
                     + '<pre>' + _esc(entry["package"].get("gpt_image_prompt")) + '</pre></p>')
        parts.append('<p>gemini_repaint_clauses（来源 '
                     + _esc(_field_source_label(entry["field_sources"], "repaint_clauses")) + '）'
                     + '<pre>' + _esc("\n".join("- " + str(c) for c in
                                                entry["package"].get("gemini_repaint_clauses") or []))
                     + '</pre></p>')
    parts.append('</article>')

    # 预检与实际请求
    parts.append('<article><h2>五、预检与实际发送的请求</h2>'
                 '<p>预检不发送请求：它把真实组装函数内部要发出去的调用换成探针，'
                 '所以「预检里那份提示词」就是「实跑要发出去的那份」。'
                 '每包每通道的实际提示词全文、输入图顺序与哈希见 '
                 f'<code>{_esc(os.path.join(subject_dir, "request-audit.json"))}</code>。</p>'
                 '<table><tr><th>包</th><th>通道</th><th>组装</th><th>提示词字符/哈希</th>'
                 '<th>输入图（顺序）</th><th>参数</th></tr>')
    for key in task.package_keys:
        for item in entries[key]["precheck"]["channels"]:
            if item.get("blocked"):
                parts.append(f'<tr><td>{key}</td><td>{_esc(item["channel"])}</td>'
                             f'<td class="bad">未组装（{_esc(item.get("reason"))}）</td>'
                             f'<td>-</td><td>-</td><td>-</td></tr>')
                continue
            images = "<br>".join(
                f'{index}. {_esc(os.path.basename(img["path"]))} '
                f'<code>{_esc(img["sha256"][:10])}</code> ({_esc(img["role"])})'
                for index, img in enumerate(item["images"])) or "（无）"
            params = item.get("params") or {}
            parts.append(f'<tr><td>{key}</td><td>{_esc(item["channel"])}</td><td class="ok">已组装</td>'
                         f'<td>{item.get("prompt_chars")} 字<br><code>{_esc((item.get("prompt_sha256") or "")[:16])}</code></td>'
                         f'<td>{images}</td><td><code>{_esc(json.dumps(params, ensure_ascii=False))}</code></td></tr>')
    parts.append('</table></article>')

    # 生成结果
    parts.append('<article><h2>六、实际生成与执行状态</h2>'
                 '<p>「复用」= 该通道<b>实际传输内容</b>与更早分组逐字一致，直接沿用既有产物；'
                 '复用不等于没测过，它证明两包的这一路请求本来就一样（提示词哈希、输入图哈希、'
                 '模型与尺寸全部相同）。</p>'
                 '<table><tr><th>包</th><th>通道</th><th>字段合法</th><th>已组装</th>'
                 '<th>已生成</th><th>复用</th><th>失败</th><th>说明</th></tr>')
    for row in matrix:
        parts.append(f'<tr><td>{row["package"]}</td><td>{_esc(row["channel"])}</td>'
                     f'<td>{"是" if row["field_valid"] else "否"}</td>'
                     f'<td>{"是" if row["assembled"] else "否"}</td>'
                     f'<td>{row["generated"]}</td><td>{row["reused"]}</td><td>{row["failed"]}</td>'
                     f'<td>{_esc(row["blocked_reason"] or "")}</td></tr>')
    parts.append('</table>')
    not_run = [(key, channel) for (key, channel), record in sorted(attempts.items())
               for channel in PROBE_CHANNELS
               if ((record.get("channel_status") or {}).get(channel) or {}).get("status") == "not_run"]
    parts.append('<p>未执行项：' + (_esc("；".join(f"{key}/{channel}" for key, channel in not_run)) or "无")
                 + '</p></article>')

    # 视觉评分
    visual = comparison or {}
    parts.append('<article><h2>七、原有 12 项视觉评分与四项主体门禁</h2>')
    if not visual:
        parts.append('<p class="bad">视觉复核未完成：没有 automatic_comparison。缺失就是缺失，不用 0 分代替。</p>')
    else:
        parts.append(f'<p>{_esc(visual.get("formula"))}<br>模型 {_esc(visual.get("model"))}；'
                     f'批次 {_esc(visual.get("batch_count"))}；输入摘要 '
                     f'<code>{_esc(str(visual.get("input_hash"))[:16])}</code>；'
                     f'{"复用已保存评分" if visual.get("cached") else "本次评分"}</p>')
        from modules.image_analysis.style_comparison import DIMENSIONS, DIMENSION_LABELS
        parts.append('<table><tr><th>候选</th>'
                     + "".join(f'<th>{_esc(DIMENSION_LABELS[k])}</th>' for k in DIMENSIONS)
                     + '<th>画风分</th><th>主体符合度</th><th>总分</th><th>门禁</th><th>依据</th></tr>')
        for row in visual.get("rows") or []:
            gates = [label for key, label in (("reference_content_copied", "复制参考内容"),
                                              ("major_structure_defect", "严重结构缺陷"),
                                              ("explicit_subject_mismatch", "主体不符"),
                                              ("uncertain", "证据不确定")) if (row.get("gates") or {}).get(key)]
            parts.append(f'<tr><td>{_esc(row["id"])}</td>'
                         + "".join(f'<td>{_esc(row.get(k))}</td>' for k in DIMENSIONS)
                         + f'<td>{_esc(row.get("style_score"))}</td><td>{_esc(row.get("subject_fidelity"))}</td>'
                           f'<td>{_esc(row.get("total"))}</td>'
                           f'<td class="{"bad" if gates else "ok"}">{_esc("排除：" + "、".join(gates) if gates else "通过")}</td>'
                           f'<td>{_esc(row.get("reason"))}</td></tr>')
        parts.append('</table>'
                     f'<p>排序第一（仅按分数与 ID 排序，不是唯一最佳）：'
                     f'<b>{_esc(visual.get("best_id") or "无候选通过门禁")}</b>'
                     + ('；并列需人工复核：' + _esc("、".join(visual.get("tied_best_ids") or []))
                        if len(visual.get("tied_best_ids") or []) > 1 else "")
                     + '</p>')
    parts.append('</article>')

    # 深度指标
    parts.append('<article><h2>八、独立深度指标（Gram / AdaIN / LPIPS / CSD）</h2>')
    if not deep:
        parts.append('<p class="bad">深度指标未完成或未运行；状态缺失不置零，也不并入视觉总分。</p>')
    else:
        backend = deep.get("backend") or {}
        parts.append(f'<p>设备：请求 {_esc(backend.get("requested"))} → 实际 {_esc(backend.get("actual"))}'
                     f'（{_esc(backend.get("backend"))} / {_esc(backend.get("precision"))}）；'
                     f'回退：{_esc(backend.get("fallback"))}；输入摘要 '
                     f'<code>{_esc(str(deep.get("input_hash"))[:16])}</code></p>'
                     f'<p>同图自检：<code>{_esc(json.dumps(deep.get("self_check"), ensure_ascii=False))}</code></p>'
                     '<table><tr><th>候选</th><th>CSD ↑</th><th>Gram ↓</th><th>AdaIN ↓</th><th>LPIPS ↓</th></tr>')
        for row in deep.get("rows") or []:
            cells = []
            for metric in ("csd", "gram", "adain", "lpips"):
                summary = (row.get("summary") or {}).get(metric) or {}
                cells.append(f'<td>{format(summary["mean"], ".6g") if summary.get("mean") is not None else "缺失"}</td>')
            parts.append(f'<tr><td>{_esc(row["id"])}</td>' + "".join(cells) + '</tr>')
        parts.append('</table>'
                     f'<p>{_esc(deep.get("limits"))}</p>'
                     f'<p>报告：<code>{_esc(deep.get("report_path"))}</code></p>')
    parts.append('</article>')

    # 候选图与接触表
    parts.append('<article><h2>九、候选图（按包分组）</h2>')
    for key in task.package_keys:
        parts.append(f'<h3>包 {key}</h3>')
        for attempt in range(1, 20):
            record = attempts.get((key, attempt))
            if not record:
                continue
            for channel in PROBE_CHANNELS:
                paths = list((record.get("generated_files") or {}).get(channel) or [])
                status = ((record.get("channel_status") or {}).get(channel) or {})
                parts.append(f'<h4>{key} / 第 {attempt} 次 / {_esc(channel)}'
                             f'（{_esc(status.get("status") or "未执行")}）</h4>')
                if not paths:
                    parts.append('<p class="warn">无产物：' + _esc(status.get("error") or "未记录原因") + '</p>')
                for path in paths:
                    parts.append('<figure>' + _picture(path)
                                 + f'<figcaption>{_esc(os.path.basename(path))}<br>'
                                   f'{_esc(str(image_info(path)["sha256"])[:12])}｜'
                                   f'{image_info(path)["width"]}x{image_info(path)["height"]}</figcaption></figure>')
    if contact_sheet and os.path.isfile(contact_sheet):
        parts.append('<h3>全部候选接触表</h3>' + _picture(contact_sheet, max_dim=1400, height=1600))
    parts.append('</article>')

    # 建议
    parts.append('<article><h2>十、逐字段与逐通道建议</h2>'
                 f'<pre>{_esc(json.dumps(recommendation, ensure_ascii=False, indent=2))}</pre>'
                 f'<p>花费估算（本地公式，非账单）：名义 {_esc(costs["total_usd"])} USD ≈ '
                 f'{_esc(costs["total_cny"])} CNY；'
                 f'已记录通道的估算约 {_esc(costs.get("net_usd"))} USD ≈ '
                 f'{_esc(costs.get("net_cny"))} CNY（复用不重复计费）。</p>'
                 '<p>人工看图复核记录：<code>'
                 f'{_esc(os.path.join(subject_dir, "manual-review-notes.md"))}</code></p>'
                 '<p>本次结论只提供方案，未写入 <code>conf/config-styles.json</code>，'
                 '也没有提交或发布画风。</p></article>')

    inventory = state.get("sample_inventory") or {}
    parts.append('<article><h2>样本独立性与去重</h2>'
                 f'<p>按图片文件 SHA-256 去重：{inventory.get("unique_image_count", 0)} 张唯一图片；'
                 f'{inventory.get("alias_count", 0)} 条复用别名。比较与深度指标只计算唯一图片。'
                 '不同重复次数禁止共用样本；同一次试验中等价请求可共用配对样本。</p>'
                 f'<pre>{_esc(json.dumps(inventory, ensure_ascii=False, indent=2))}</pre></article>')
    parts.append('<article><h2>各包各通道的独立样本统计</h2><pre>'
                 + _esc(json.dumps(summarize_samples(state, comparison, attempts), ensure_ascii=False, indent=2))
                 + '</pre></article>')
    path = os.path.join(subject_dir, "merged-validation-report.html")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(parts))
    return path


def write_conclusion_markdown(task, entries, precheck, state, comparison, deep, attempts,
                              cost=None):
    subject_dir = os.path.dirname(state_path(task))
    recommendation = recommendation_from(comparison, deep)
    costs = cost or cost_report(entries, task, attempts=attempts)
    matrix = _attempt_matrix(entries, attempts, task.package_keys)
    generated_total = sum(row["generated"] for row in matrix)
    failed_total = sum(row["failed"] for row in matrix)
    lines = [
        "# 画风候选验证结论（中文）",
        "",
        f"- 源结果：`{task.source}`",
        f"- 源 SHA-256：`{file_hash(task.source)}`",
        f"- 输出根目录：`{task.output_root}`",
        f"- 主体分组：{task.subject_label}；固定测试主体：`{task.subject_prompt()}`",
        f"- 测试画风参考图：`{precheck.get('style_reference')}`",
        "",
        "## 一、真实执行数量",
        "",
        f"- 预检：{len(entries)} 个候选包（未发送任何联网请求）。",
        f"- 生图：已生成 {generated_total} 张通道产物，失败 {failed_total} 条通道记录。",
        f"- 按 SHA-256 去重：{(state.get('sample_inventory') or {}).get('unique_image_count', 0)} 张唯一图片，"
        f"{(state.get('sample_inventory') or {}).get('alias_count', 0)} 条复用别名；评分不重复计算别名。",
        "",
        "| 包 | 通道 | 字段合法 | 已组装 | 已生成 | 失败 | 说明 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in matrix:
        lines.append(f'| {row["package"]} | {row["channel"]} | {"是" if row["field_valid"] else "否"} | '
                     f'{"是" if row["assembled"] else "否"} | {row["generated"]} | {row["failed"]} | '
                     f'{row["blocked_reason"] or ""} |')
    lines += [
        "",
        "## 二、字段来源",
        "",
        "| 包 | prompt | prompt_gpt | repaint_clauses |",
        "|---|---|---|---|",
    ]
    for key in task.package_keys:
        entry = entries[key]
        lines.append(f'| {key}（{entry["label"]}） | {_field_source_label(entry["field_sources"], "prompt")} | '
                     f'{_field_source_label(entry["field_sources"], "prompt_gpt")} | '
                     f'{_field_source_label(entry["field_sources"], "repaint_clauses")} |')
    lines += [
        "",
        ("按各候选记录独立读取提示词；字段校验与已出图不代表视觉门禁通过。"
         if isinstance(task, ReviewCandidateTask) else "第 5 轮 GPT 短版只有 158 字符、只有 `Palette` 一个有值字段，`validate_prompt_gpt` 判定不合格，"
        "因此它没有被任何一包采用：包 B/C 的 `prompt_gpt` 按设计取第 2 轮，GPT 首图与重绘**照常发送**"
        "（没有用母版顶替，也没有拿不合格短版顶上）。第 5 轮完整描述和重绘条款是否适合，"
        "须依据独立样本的主体门禁与画风复核判断，不能由短版字段不合法推断。"),
        "",
        "## 三、费用估算依据",
        "",
        f"- 名义合计（完全不复用时）约 {costs['total_usd']} USD ≈ {costs['total_cny']} CNY；"
        f"已记录 {sum(row.get('paid_runs', 0) for row in costs['rows'])} 次非复用通道执行，"
        f"估算净额约 {costs.get('net_usd')} USD ≈ {costs.get('net_cny')} CNY"
        "（`utils.cost_estimate` 的公式 + 缓存价目表，本地估算，非账单）。",
        "- 逐通道明细见 `cost-estimate.json`；复用与失败通道都不重复计费。",
        "",
        "## 四、视觉复核与深度指标",
        "",
        f"- 视觉复核：{'已完成' if comparison else '未完成'}；"
        f"排序第一 {recommendation.get('selected_candidate') or '无'}。",
        f"- 深度指标：{'已完成' if deep else '未完成'}；"
        f"设备 {(deep or {}).get('backend', {}).get('actual', '未运行')}。",
        "- 复制参考内容的候选一律排除，高视觉分不能抵消主体门禁；深度指标不并入视觉总分。",
        "",
        "## 五、建议",
        "",
        f"```json\n{json.dumps(recommendation, ensure_ascii=False, indent=2)}\n```",
        "",
        "区分的四层状态：**字段合法**（能通过 `validate_prompt_gpt`）→ **已生成**（该通道真的出了图）"
        "→ **复核通过**（12 项量表 + 四项门禁）→ **尚未覆盖的场景**（未测试主体、未测试画幅、"
        "未测试的母题与场景配置）。",
        "",
        "## 六、限制与可复现入口",
        "",
        "- 本次重绘是文字模式（`reference_mode=none`），与旧运行的完整画风图重绘不是同条件复现。",
        "- 视觉分项是模型按量表判断，未经人类标注校准；深度指标未经人工校准，不能替代门禁。",
        "- 复现入口：`python tools/style_merge_validate.py precheck|run|compare|deep|report "
        "--source <源 JSON> --out <本目录根>`。",
        "",
        "本次仅验证与提出方案，未写入 `conf/config-styles.json`，未提交或发布画风。",
    ]
    path = os.path.join(subject_dir, "conclusion-zh.md")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return path


def proposed_style_entry(task, package_entries, comparison, deep, package_key=None):
    """待用户观察报告后手动选用的画风条目草稿（默认 enabled=false，不写入正式配置）。"""
    recommendation = recommendation_from(comparison, deep)
    key = package_key or recommendation.get("package") or next(iter(package_entries))
    if package_key and package_key not in package_entries:
        raise ValidationError(f"报告中没有候选包 {package_key}")
    entry = package_entries[key]
    package = entry["package"]
    return {
        "_comment": "候选方案草稿：本次仅验证与提出建议，未写入 conf/config-styles.json；"
                    "enabled 默认 false，需用户观察报告后自行选用。",
        "prompt": package.get("gemini_full_prompt") or "",
        "prompt_gpt": package.get("gpt_image_prompt") or "",
        "repaint_clauses": list(package.get("gemini_repaint_clauses") or []),
        "motif_clauses": [],
        "motif_enabled": False,
        "enabled": False,
        "ref_image": os.path.abspath(task.style_reference()),
        "field_sources": {field: source for field, source in entry["field_sources"].items()},
        "validation": {
            "run_root": task.output_root,
            "state_json": state_path(task),
            "precheck": os.path.join(os.path.dirname(state_path(task)), "precheck.json"),
            "request_audit": os.path.join(os.path.dirname(state_path(task)), "request-audit.json"),
            "comparison_report": comparison.get("report_path") if comparison else "",
            "deep_report": deep.get("report_path") if deep else "",
            "recommendation": recommendation,
            "proposed_package": key,
            "proposal_basis": "指定候选草稿，未启用；自动最高分保留在 recommendation"
                              if package_key else "按单张最高门禁通过分生成草稿，不代表整包统计最优",
            "source_json": task.source,
        },
    }


def recommendation_from(comparison, deep):
    """逐通道建议：只在门禁通过的候选里挑，且明确区分「字段合法/已生成/复核通过」。"""
    rows = (comparison or {}).get("rows") or []
    deep_rows = {row["id"]: row for row in (deep or {}).get("rows") or []}
    eligible = [row for row in rows if row.get("eligible")]
    eligible.sort(key=lambda row: (-float(row["sort_value"]), row["id"]))
    import re

    def candidate_package(candidate_id):
        stage = str(candidate_id).split("/")[0]
        if stage.startswith("package"):
            stage = stage[len("package"):]
        return re.sub(r"-attempt\d+$", "", stage)

    per_key = {}
    for row in rows:
        key = candidate_package(row["id"])
        per_key.setdefault(key, []).append(row)
    summary = {}
    for key, items in per_key.items():
        ok = [row for row in items if row.get("eligible")]
        best = max(ok, key=lambda row: float(row["sort_value"])) if ok else None
        summary[key] = {
            "candidates": len(items),
            "eligible": len(ok),
            "best_id": best["id"] if best else None,
            "best_total": best["total"] if best else None,
            "channels": sorted({row["id"].split("/")[1] for row in items}),
        }
    best_row = eligible[0] if eligible else None
    package_key = None
    if best_row:
        package_key = candidate_package(best_row["id"])
    return {
        "selected_candidate": best_row["id"] if best_row else "",
        "package": package_key or "",
        "per_package": summary,
        "deep_metric_support": {row_id: {"csd": (deep_rows.get(row_id, {}).get("summary", {}).get("csd", {}) or {}).get("mean"),
                                         "gram": (deep_rows.get(row_id, {}).get("summary", {}).get("gram", {}) or {}).get("mean")}
                                for row_id in [row["id"] for row in eligible]},
        "caveats": [
            "视觉分项是模型按量表判断，不是相似度百分比，也未做人类标注校准。",
            "深度指标不参与视觉总分，不能抵消主体门禁。",
            "同一包内重复样本的波动必须一起看，不能凭单张好图断言整包更优。",
        ],
    }
