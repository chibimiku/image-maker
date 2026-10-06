# -*- coding: utf-8 -*-
"""App 二阶段色调/主次冻结包的无头执行支持：离线校验 → 运行参数冻结 → 无图文字首图 → 分组视觉评审 → 报告。

对应任务 `prompts/color-knowledge/DEEPSEEK-STAGE2-DIRECTION-VALIDATION-20261005.md`。
本模块只服务这一件事，范围**只有**无图冻结首图、ledger、成功缓存与本轮评审：

- 不新增 HTTP 客户端：生成走 `modules.others.api_backend.generate_image_aigc2d`，
  评审走 `utils.analysis_gpt_prompt.call_text_model`（既有视觉文本通道）。
- 每次发送前先原子落盘 `sending`；发送结果落 `generated` / `failed` / `unknown_after_send`。
  `ledger.json` 是恢复时的唯一权威：`unknown_after_send` 之后不再自动发送，等用户决定。
- 「明确失败」消耗该槽位预算，不自动补图；成功产图按 hash 复用，不重复收费。
- 生图落盘、评审落盘与报告渲染**分离**：报告渲染异常只写 `RUN-REPORT-render-failure.txt`，
  绝不影响已落盘的生成/评审证据，也绝不为补报告重新联网。
- 落盘只保留脱敏端点与 key 来源名称，凭据不进入报告、ledger、请求快照或日志。
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import random
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from utils.atomic_io import write_json_atomic
from utils.theme_color import (apply_theme_color, digest, freeze_theme_color,
                               record_outputs, record_request, validate_snapshot)

BASE = Path(__file__).resolve().parents[1]

LABEL = "color-direction-app-v2-validation-v1"
PACK_PATH = "prompts/color-knowledge/theme-color-direction-validation-v1-pack.json"
SCENES_PATH = "prompts/color-knowledge/theme-color-direction-validation-v1-scenes.json"
REVIEW_PROMPT_PATH = "prompts/color-knowledge/theme-color-direction-validation-v1-review.md"
EXPECTED_PACK_HASH = "3210d0796f8e5d385284c63e4a916be27ba6fff23fecd63d4fe4481a911d09ff"
APPROVED_PACKS = {
    EXPECTED_PACK_HASH: {"out_dir": "data/test-result/20261005/" + LABEL, "variants": list("ABCDE"), "images": 10, "reviews": 4},
    "e8ef077a0079b91ba5829d120a6d719762b0608ed38d7611f0f380c249058334": {
        "out_dir": "data/test-result/20261006/color-chroma-app-v2-validation-v2", "variants": list("ABC"), "images": 6, "reviews": 2},
    # 2026-10-06 清色重复验证（A 基准 / B 当前生产清色 / C 同规则 + 唯一候选追加正文）：
    # 9 次图片请求 + 3 次视觉评审是与 v2 同一轮交接里登记的预算，不是新增额度。
    "0c3178c7795a3456c553196cccced84ae449115974208dd52f1ba852fa789298": {
        "out_dir": "data/test-result/20261006/color-clear-repeat-validation-v3",
        "variants": ["A", "A", "A", "B", "B", "B", "C", "C", "C"], "images": 9, "reviews": 3},
}

AUTHORISED_HOST = "new.aigc2d.com"
IMAGE_RETRY_ENV = "IMAGE_MAKER_IMAGE_MAX_RETRIES"
IMAGE_API_TYPE = "aigc2d"
REVIEW_MAX_TOKENS = 12000          # 与项目既有视觉评审（reassessment）同一口径
REVIEW_TIMEOUT_SECONDS = 420
RUN_ROOT = ("data", "test-result")

UNCOVERED_ITEMS = (
    "清色 / 浊色（第三、第四种色调）未验证",
    "只选色调、不开启色板（无配色方案的 tone-only 路径）未验证",
    "「主色主导 / 少量强调」这一额外主次模式未验证",
    "情绪 / 季节（generation_atmosphere）未验证",
    "另外两套色板（深绿+玫红、蓝紫+金）未验证",
    "GPT-image 通道未验证（本轮固定 Gemini 无图文字首图）",
    "带参考图（内容图 / 画风图）未验证",
    "App GUI 内实际生效未验证（本轮为同一编译器的无头复现）",
    "重绘 / 后处理保留未验证（本轮不含任何后续工序）",
)


# --- 基础工具 ---------------------------------------------------------------

def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def compose_slot_prompt(theme_prompt, candidate_text=""):
    """槽位最终正文 = 主题配色编译器输出 + 唯一的候选追加正文。

    追加正文是**实验候选**，不属于生产模板：它逐字来自 pack 里登记的候选文本、
    单独参与 hash，绝不写回 App 或共享配置。没有候选文本时与旧轮逐字一致。
    """
    text = str(candidate_text or "")
    if not text:
        return str(theme_prompt)
    return str(theme_prompt) + "\n\n" + text


def _rel(path):
    try:
        return str(Path(path).resolve().relative_to(BASE)).replace("\\", "/")
    except Exception:  # noqa: BLE001
        return str(path)


def _resolve_out(out_dir):
    """实验目录必须落在 `data/test-result/` 下（测试可用环境变量指定临时根）。"""
    root_env = str(os.environ.get("IMAGE_MAKER_COLOUR_TEST_OUTPUT_ROOT") or "").strip()
    out = Path(out_dir).resolve()
    if root_env and out.is_relative_to(Path(root_env).resolve()):
        return out
    if not out.is_relative_to(BASE.joinpath(*RUN_ROOT)):
        raise ValueError("冻结包实验的输出目录必须在 data/test-result/ 下：%s" % out)
    return out


def _write_once(path: Path, data, log=print, ignore_keys=()):
    """写 JSON；已存在且内容相同则复用，不同则另存时间戳副本，绝不覆盖已有内容。

    `ignore_keys` 用于忽略只表示「什么时候写的」的易变字段（如 `at`），
    否则同一份冻结参数每次重算都会产生一份新副本、并让 hash 不稳定。
    """
    path = Path(path)

    def comparable(value):
        if isinstance(value, dict):
            return {key: comparable(item) for key, item in value.items() if key not in ignore_keys}
        return value

    if path.is_file():
        try:
            if comparable(json.loads(path.read_text(encoding="utf-8"))) == comparable(data):
                return {"path": str(path), "reused": True}
        except Exception:  # noqa: BLE001
            pass
        alt = path.with_name("%s-%s%s" % (path.stem, time.strftime("%H%M%S"), path.suffix))
        write_json_atomic(str(alt), data)
        log("[direction] 已有同名文件内容不同，另存 %s（原文件未覆盖）" % _rel(alt))
        return {"path": str(alt), "reused": False, "previous": str(path)}
    write_json_atomic(str(path), data)
    return {"path": str(path), "reused": False}


def _write_derived(path: Path, data, log=print):
    """派生报告类产物：内容变化时先留旧版本副本，再原地刷新（证据类文件不走这里）。"""
    path = Path(path)
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2)
    if path.is_file():
        previous = path.read_text(encoding="utf-8")
        if previous == text:
            return {"path": str(path), "changed": False}
        backup = path.with_name("%s.%s.prev" % (path.name, time.strftime("%Y%m%d-%H%M%S")))
        backup.write_text(previous, encoding="utf-8")
        log("[direction] 报告刷新，旧版本留存为 %s" % backup.name)
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "changed": True}


def load_pack(path=None):
    return json.loads(Path(path or (BASE / PACK_PATH)).read_text(encoding="utf-8"))


def pack_digest(pack) -> str:
    return digest({key: value for key, value in pack.items() if key != "pack_hash"})


def slot_ids(pack):
    return [slot["id"] for slot in pack["slots"]]


# --- 1. 离线校验 ------------------------------------------------------------

def verify_offline(out_dir, *, pack_path=None, scenes_path=None, review_path=None, log=print):
    """逐字节复核冻结包：来源 hash、pack hash、正文/plan hash、App 编译器逐字复算、结构完整性。

    重新调用的 `freeze_theme_color` / `apply_theme_color` 结果**只用于比较**，
    绝不复写回冻结包。
    """
    out = _resolve_out(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pack_file = Path(pack_path or (BASE / PACK_PATH))
    scenes_file = Path(scenes_path or (BASE / SCENES_PATH))
    review_file = Path(review_path or (BASE / REVIEW_PROMPT_PATH))
    pack = json.loads(pack_file.read_text(encoding="utf-8"))
    problems, notes, rows = [], [], []

    # 1) pack hash（canonical JSON 去掉顶层 pack_hash）与既定期望值
    approval = APPROVED_PACKS.get(pack.get("pack_hash")) or {}
    expected_hash = pack.get("pack_hash") if approval else EXPECTED_PACK_HASH
    recomputed = pack_digest(pack)
    if recomputed != pack.get("pack_hash"):
        problems.append("pack_hash 与自身内容不一致：%s != %s" % (recomputed, pack.get("pack_hash")))
    if pack.get("pack_hash") != expected_hash:
        problems.append("pack_hash 与任务期望值不一致：%s != %s" % (pack.get("pack_hash"), expected_hash))
    if pack.get("out_dir") != approval.get("out_dir"):
        problems.append("冻结包 out_dir 与本轮输出目录不一致：%s" % pack.get("out_dir"))
    if pack.get("automatic_retry") is not False:
        problems.append("冻结包未声明关闭自动重试")
    if int(pack.get("image_http_budget") or 0) != approval.get("images"):
        problems.append("冻结包图像预算与登记值不符：%s" % pack.get("image_http_budget"))
    if int(pack.get("review_http_budget") or 0) != approval.get("reviews"):
        problems.append("冻结包评审预算与登记值不符：%s" % pack.get("review_http_budget"))

    # 2) 来源文件字节 SHA-256
    source_hashes = pack.get("source_hashes") or {}
    source_rows = []
    for rel, want in sorted(source_hashes.items()):
        path = BASE / rel
        if not path.is_file():
            problems.append("来源文件缺失：%s" % rel)
            source_rows.append({"path": rel, "status": "missing", "expected": want})
            continue
        got = _sha256_file(path)
        ok = got == want
        source_rows.append({"path": rel, "status": "match" if ok else "mismatch",
                            "expected": want, "actual": got})
        if not ok:
            problems.append("来源文件字节 hash 不一致：%s" % rel)
    for required in (scenes_file, review_file):
        rel = _rel(required)
        if rel not in source_hashes:
            notes.append("来源清单未登记 %s（仍按 pack 内文本复核）" % rel)

    # 2b) 候选追加正文：来源文件必须与 pack 内登记的文本逐字一致（尾部换行除外）
    candidate_texts = pack.get("candidate_texts") or {}
    for text_id, rel in sorted((pack.get("candidate_text_sources") or {}).items()):
        path = BASE / rel
        if not path.is_file():
            problems.append("候选正文来源文件缺失：%s" % rel)
            continue
        on_disk = path.read_text(encoding="utf-8").rstrip("\n")
        if on_disk != str(candidate_texts.get(text_id) or "").rstrip("\n"):
            problems.append("候选正文来源文件与 pack 内登记文本不一致：%s" % rel)

    # 3) 逐槽位：结构、prompt hash、plan hash、App 编译器逐字复算
    slots = pack.get("slots") or []
    ids = [slot.get("id") for slot in slots]
    if len(ids) != approval.get("images"):
        problems.append("槽位数量与登记值不符：%d" % len(ids))
    if len(set(ids)) != len(ids):
        problems.append("槽位 id 有重复：%s" % ids)
    by_scene = {}
    for slot in slots:
        by_scene.setdefault(slot.get("scene_id"), []).append(slot)
    for scene_id, group in sorted(by_scene.items()):
        variants = sorted(slot.get("variant_id") for slot in group)
        if variants != approval.get("variants"):
            problems.append("%s 的变体与登记值不符：%s" % (scene_id, variants))
        contents = {slot.get("content_prompt") for slot in group}
        if len(contents) != 1:
            problems.append("%s 的 A–E 正文不是逐字相同" % scene_id)

    for slot in slots:
        row = {"slot_id": slot.get("id"), "scene_id": slot.get("scene_id"),
               "variant_id": slot.get("variant_id"), "problems": []}
        prompt = str(slot.get("prompt") or "")
        if _sha256_text(prompt) != slot.get("prompt_sha256"):
            row["problems"].append("prompt_sha256 与正文不一致")
        plan = slot.get("color_plan") or {}
        try:
            validate_snapshot(plan)
        except Exception as exc:  # noqa: BLE001
            row["problems"].append("plan_hash 校验失败：%s" % exc)
        selection = slot.get("selection") or {}
        candidate_id = slot.get("candidate_text_id")
        candidate_text = ""
        if candidate_id:
            candidate_text = str((pack.get("candidate_texts") or {}).get(candidate_id) or "")
            if not candidate_text:
                row["problems"].append("candidate_text_id 未在 pack.candidate_texts 中登记：%s" % candidate_id)
        append = str(slot.get("candidate_append") or "")
        if append != candidate_text:
            row["problems"].append("candidate_append 与 pack 登记的候选正文不一致（追加正文必须逐字唯一）")
        if not candidate_id and append:
            row["problems"].append("出现未登记的 candidate_append（必须用 candidate_text_id 引用 pack 文本）")
        if "candidate_append_sha256" in slot and _sha256_text(append) != slot.get("candidate_append_sha256"):
            row["problems"].append("candidate_append_sha256 与追加正文不一致")
        try:
            recomputed_plan = freeze_theme_color(selection)
            if recomputed_plan != plan:
                row["problems"].append("用原 selection 重新 freeze_theme_color 与冻结 plan 不一致")
            recomputed_prompt = compose_slot_prompt(
                apply_theme_color(str(slot.get("content_prompt") or ""), recomputed_plan,
                                  image_paths=(), mode="generate", post_enabled=False),
                candidate_text)
            if recomputed_prompt != prompt:
                row["problems"].append("用原 selection + 候选正文重新 apply_theme_color 与冻结 prompt 不一致")
        except Exception as exc:  # noqa: BLE001
            row["problems"].append("App 编译器离线复算失败：%s" % exc)
        if slot.get("instructions") or slot.get("post_instructions") or slot.get("prompt_suffix"):
            row["problems"].append("instructions/post_instructions/prompt_suffix 必须为空")
        if list(slot.get("image_paths") or []):
            row["problems"].append("image_paths 必须为空（无图文字首图）")
        if slot.get("face_quality_boost") is not False:
            row["problems"].append("face_quality_boost 必须为 False")
        if int(slot.get("expected_output_count") or 0) != 1:
            row["problems"].append("expected_output_count 必须为 1")
        if slot.get("auxiliary_note") and slot.get("variant_id") != "E":
            notes.append("%s 带场景级 auxiliary_note（A–D 保持原色，只有 E 才授权辅助绑定）" % slot.get("id"))
        problems.extend("%s: %s" % (row["slot_id"], item) for item in row["problems"])
        rows.append(row)

    # 3b) 候选追加的通用一致性：同一条件逐字一致，且只有 pack 声明的条件带追加
    append_by_variant = {}
    for slot in slots:
        append_by_variant.setdefault(slot.get("variant_id"), set()).add(str(slot.get("candidate_append") or ""))
    for variant, values in sorted(append_by_variant.items(), key=lambda item: str(item[0])):
        if len(values) != 1:
            problems.append("条件 %s 的不同重复使用了不同的追加正文" % variant)
    if pack.get("candidate_append_variants") is not None:
        carrying = sorted(variant for variant, values in append_by_variant.items() if values != {""})
        if carrying != sorted(pack["candidate_append_variants"]):
            problems.append("带追加正文的条件与 pack 声明不符：%s != %s"
                            % (carrying, sorted(pack["candidate_append_variants"])))

    # 4) 场景定义与冻结槽位一致
    scenes = json.loads(scenes_file.read_text(encoding="utf-8"))
    scene_by_id = {scene["id"]: scene for scene in scenes.get("scenes", [])}
    variant_by_id = {variant["id"]: variant for variant in scenes.get("variants", [])}
    for slot in slots:
        scene = scene_by_id.get(slot.get("scene_id"))
        if not scene:
            problems.append("%s 引用了不存在的场景 %s" % (slot.get("id"), slot.get("scene_id")))
            continue
        if slot.get("content_prompt") != scene.get("content_prompt"):
            problems.append("%s 正文与场景定义不一致" % slot.get("id"))
        for key, value in (scene.get("selection") or {}).items():
            if (slot.get("selection") or {}).get(key) != value:
                problems.append("%s selection 的 %s 与场景定义不一致（%r != %r）"
                                % (slot.get("id"), key, (slot.get("selection") or {}).get(key), value))
        variant = variant_by_id.get(slot.get("variant_id")) or {}
        expected = dict(scene.get("selection") or {})
        expected.update(variant.get("change") or {})
        if variant.get("use_auxiliary_regions"):
            expected["auxiliary_regions"] = scene.get("auxiliary_regions")
        if slot.get("selection") != expected:
            problems.append("%s selection 与「场景 + 变体定义」推导结果不一致" % slot.get("id"))

    # 5) 评审分组：显式槽位、每组 3 张、覆盖全部槽位
    groups = pack.get("review_groups") or []
    if len(groups) != approval.get("reviews"):
        problems.append("评审组数量与登记值不符：%d" % len(groups))
    seen_slots = set()
    for group in groups:
        if len(group.get("slot_ids") or []) != 3:
            problems.append("%s 不是 3 张一组" % group.get("id"))
        for item in group.get("slot_ids") or []:
            if item not in ids:
                problems.append("%s 引用了不存在的槽位 %s" % (group.get("id"), item))
            seen_slots.add(item)
        if group.get("kind") not in ("tone", "hierarchy"):
            problems.append("%s 的 kind 非法：%s" % (group.get("id"), group.get("kind")))
        order = group.get("neutral_order")
        if order is not None and sorted(order) != sorted(group.get("slot_ids") or []):
            problems.append("%s 声明的平衡展示顺序不是该组槽位的排列：%s" % (group.get("id"), order))
    if seen_slots != set(ids):
        problems.append("存在未被任何评审组覆盖的槽位：%s" % sorted(set(ids) - seen_slots))

    # 5b) 通用比较计划：显式的（基准槽位, 候选槽位）解码，不依赖 S1/S2 命名约定
    group_ids = {group.get("id") for group in groups}
    for item in pack.get("comparisons") or []:
        label = item.get("name") or "<未命名>"
        for key in ("name", "group_id", "baseline", "candidate"):
            if not item.get(key):
                problems.append("comparisons 条目缺少 %s：%s" % (key, label))
        if item.get("baseline") not in ids or item.get("candidate") not in ids:
            problems.append("comparisons 引用了不存在的槽位：%s" % label)
        if item.get("group_id") not in group_ids:
            problems.append("comparisons 引用了不存在的评审组：%s" % label)
        if item.get("expect") not in (None, "candidate", "baseline"):
            problems.append("comparisons 的 expect 非法：%s" % label)
        if item.get("baseline") == item.get("candidate"):
            problems.append("comparisons 的基准与候选是同一槽位：%s" % label)

    review_text = review_file.read_text(encoding="utf-8")
    if "variant" in review_text.lower():
        notes.append("评审模板提到 variant，需人工确认不泄露变体名（模板本身不含具体名称）")

    result = {"label": (pack or {}).get("id", LABEL), "at": _now(), "no_network": True,
              "pack": _rel(pack_file), "scenes": _rel(scenes_file), "review_prompt": _rel(review_file),
              "pack_hash_expected": expected_hash, "pack_hash_actual": pack.get("pack_hash"),
              "pack_hash_recomputed": recomputed,
              "pack_hash_ok": recomputed == pack.get("pack_hash") == expected_hash,
              "source_files": source_rows, "slots": rows,
              "review_prompt_sha256": hashlib.sha256(review_text.encode("utf-8")).hexdigest(),
              "review_groups": [{"id": group.get("id"), "kind": group.get("kind"),
                                 "slot_ids": group.get("slot_ids")} for group in groups],
              "problems": problems, "notes": notes,
              "status": "problems" if problems else "passed"}
    written = _write_once(out / "offline-verification.json", result, log=log)
    log("[direction] 离线校验 %s：%d 槽位 / %d 来源文件 / %d 个问题 / %d 条提示（未联网）"
        % (result["status"], len(slots), len(source_rows), len(problems), len(notes)))
    for problem in problems:
        log("[direction] 问题：%s" % problem)
    return {**result, "written": written}


# --- 2. 运行参数冻结 --------------------------------------------------------

def freeze_runtime(out_dir, pack, *, log=print):
    """冻结本次实际运行参数（不含任何凭据），并落 `RUNTIME-FROZEN.json`。"""
    out = _resolve_out(out_dir)
    problems, notes = [], []
    from modules.others.api_backend import get_api_config
    from utils.analysis_gpt_prompt import load_text_api_config

    generation, review = {}, {}
    try:
        cfg = get_api_config(api_type=IMAGE_API_TYPE)
        base_url = str(cfg.get("base_url") or "")
        host = urlparse(base_url).hostname
        generation = {
            "api_node": IMAGE_API_TYPE, "api_type": IMAGE_API_TYPE,
            "host": host, "base_url_without_credentials": base_url,
            "model": cfg.get("model"), "model_source": "conf/config.json apis.%s.model（未手写替换）" % IMAGE_API_TYPE,
            "aspect_ratio": cfg.get("default_aspect_ratio"),
            "aspect_ratio_source": "conf/config.json apis.%s.default_aspect_ratio（配置未给才回落既有默认解析）" % IMAGE_API_TYPE,
            "resolution": cfg.get("resolution"),
            "resolution_source": "conf/config.json apis.%s.resolution" % IMAGE_API_TYPE,
            "transport_timeout_seconds": cfg.get("timeout"),
            "retries": {"automatic": False,
                        "env_override": {IMAGE_RETRY_ENV: "0"},
                        "configured_max_retries": cfg.get("max_retries"),
                        "note": "每次发送只发一次 HTTP；包装器与底层都不重发"},
            "image_paths": [], "instructions": "", "post_instructions": "", "prompt_suffix": "",
            "face_quality_boost": False, "expected_output_count_per_slot": 1,
            "key_present": bool(cfg.get("api_key")), "key_source": cfg.get("_api_key_source"),
        }
        if host != AUTHORISED_HOST:
            problems.append("图片通道主机不是授权节点：%s" % host)
        if not cfg.get("api_key"):
            problems.append("图片通道 key 不可用（离线可跑，生成前必须解决）")
        for key in ("model", "resolution", "aspect_ratio"):
            if not generation.get(key):
                problems.append("图片通道缺少合法 %s" % key)
    except Exception as exc:  # noqa: BLE001
        problems.append("读取图片通道配置失败：%s" % exc)

    try:
        tcfg = load_text_api_config()
        host = urlparse(str(tcfg.get("base_url") or "")).hostname
        review = {"endpoint_without_credentials": tcfg.get("base_url"), "host": host,
                  "model": tcfg.get("model"), "model_source": "conf/config.json 顶层 model",
                  "max_completion_tokens": REVIEW_MAX_TOKENS, "timeout_seconds": REVIEW_TIMEOUT_SECONDS,
                  "images_per_call": 3, "calls_planned": pack["review_http_budget"],
                  "key_present": bool(tcfg.get("api_key")), "key_source": "IMAGE_MAKER_TEXT_API_KEY（解析入口 resolve_text_api_key）",
                  "role": "视觉文本通道（支持图片输入）；DeepSeek 文字回复不算视觉评审"}
        if host != AUTHORISED_HOST:
            problems.append("评审端点主机不在授权范围：%s" % host)
        if not tcfg.get("api_key"):
            problems.append("评审通道 key 不可用")
        if not tcfg.get("model"):
            problems.append("评审通道缺少模型")
    except Exception as exc:  # noqa: BLE001
        problems.append("读取评审通道配置失败：%s" % exc)

    runtime = {"label": (pack or {}).get("id", LABEL), "at": _now(), "no_network": True,
               "pack_hash": pack.get("pack_hash"),
               "generation": generation, "review": review,
               "budget": {"image_sends": pack["image_http_budget"], "review_sends": pack["review_http_budget"],
                          "per_slot_max_sends": 1, "per_review_group_max_sends": 1},
               "sameness": "全部槽位使用同一份生成参数（模型/比例/分辨率/超时/重试策略）",
               "redaction": "只记录脱敏端点与 key 来源名称；凭据不进入本文件、ledger、请求快照或日志",
               "problems": problems, "notes": notes,
               "status": "problems" if problems else "frozen"}
    runtime["runtime_hash"] = digest({key: value for key, value in runtime.items() if key != "at"})
    written = _write_once(out / "RUNTIME-FROZEN.json", runtime, log=log, ignore_keys=("at",))
    log("[direction] 运行参数冻结 %s：%s / %s / %s / %s / 超时 %ss；评审 %s @ %s"
        % (runtime["status"], generation.get("host"), generation.get("model"),
           generation.get("resolution"), generation.get("aspect_ratio"),
           generation.get("transport_timeout_seconds"), review.get("model"), review.get("host")))
    for problem in problems:
        log("[direction] 问题：%s" % problem)
    return {**runtime, "written": written}


def load_runtime(out_dir):
    return json.loads((_resolve_out(out_dir) / "RUNTIME-FROZEN.json").read_text(encoding="utf-8"))


def publish_frozen_pack(out_dir, *, pack_path=None, log=print):
    """把源冻结包**按字节**复制进输出目录；已存在同一份就复用，内容不同则停止（不静默升级版本）。"""
    out = _resolve_out(out_dir)
    source = Path(pack_path or (BASE / PACK_PATH))
    text = source.read_text(encoding="utf-8")
    source_sha = _sha256_text(text)
    target = out / "FROZEN-PACK.json"
    if target.is_file():
        existing_text = target.read_text(encoding="utf-8")
        existing = json.loads(existing_text)
        if existing.get("pack_hash") != json.loads(text).get("pack_hash"):
            raise ValueError("输出目录里的 FROZEN-PACK.json 与源冻结包 pack_hash 不一致：停止执行，人工核对版本")
        if _sha256_text(existing_text) == source_sha:
            log("[direction] FROZEN-PACK.json 已存在且逐字节一致，复用（不覆盖）")
            return {"path": str(target), "reused": True, "pack_hash": existing.get("pack_hash"),
                    "source_sha256": source_sha, "byte_identical": True}
        backup = target.with_name(target.name + ".reserialized.prev")
        if not backup.is_file():
            backup.write_text(existing_text, encoding="utf-8")
        target.write_text(text, encoding="utf-8")
        log("[direction] FROZEN-PACK.json 语义相同但字节不同，已改写为逐字节副本（旧文件留存 %s）"
            % backup.name)
        return {"path": str(target), "reused": False, "rewritten": True,
                "pack_hash": existing.get("pack_hash"), "source_sha256": source_sha,
                "previous": str(backup), "byte_identical": True}
    target.write_text(text, encoding="utf-8")
    return {"path": str(target), "reused": False, "pack_hash": json.loads(text).get("pack_hash"),
            "source_sha256": source_sha, "byte_identical": True}


# --- 2b. 离线编译冻结包（通用：场景 × 条件 × 重复 + 候选追加 + 平衡顺序） ----

def build_pack(spec_path, out_pack_path=None, *, log=print):
    """按实验规格编译冻结包与场景文件（**完全离线**，不联网、不改生产模板）。

    规格（spec）与生产代码解耦：场景正文、条件（变体）改动、重复次数、评审组、
    平衡展示顺序、候选追加正文与来源文件都写在 spec 里，由 App 既有编译器
    （``freeze_theme_color`` / ``apply_theme_color``）产出 plan 与正文。
    候选追加正文单独登记、单独 hash，只进入本实验包的最终正文。

    产物：``<...>-pack.json``（含 pack_hash）与 spec 指定的 scenes 文件。
    已存在且 pack_hash 不同的文件不会被覆盖，避免覆盖历史证据。
    """
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    scenes_rel = str(spec["scenes_path"])
    scenes_file = BASE / scenes_rel
    review_rel = str(spec["review_prompt_path"])
    review_file = BASE / review_rel
    if not review_file.is_file():
        raise FileNotFoundError("评审模板不存在：%s" % review_rel)
    candidate_texts = {str(k): str(v) for k, v in (spec.get("candidate_texts") or {}).items()}
    candidate_sources = {str(k): str(v) for k, v in (spec.get("candidate_text_sources") or {}).items()}

    scenes_out, variant_out, slots_out = [], [], []
    for scene in spec["scenes"]:
        scenes_out.append({key: scene[key] for key in
                           ("id", "label", "content_prompt", "selection", "auxiliary_regions",
                            "protected_facts", "auxiliary_note") if key in scene})
    for index, condition in enumerate(spec["conditions"]):
        variant_out.append({"id": condition["variant_id"], "label": condition.get("label", ""),
                            "change": dict(condition.get("change") or {})})
    for scene in spec["scenes"]:
        for condition in spec["conditions"]:
            variant_id = condition["variant_id"]
            selection = dict(scene.get("selection") or {})
            selection.update(condition.get("change") or {})
            plan = freeze_theme_color(selection)
            base_prompt = apply_theme_color(str(scene["content_prompt"]), plan,
                                            image_paths=(), mode="generate", post_enabled=False)
            text_id = condition.get("candidate_text_id")
            append = candidate_texts.get(str(text_id) or "", "") if text_id else ""
            prompt = compose_slot_prompt(base_prompt, append)
            for repeat in range(1, int(condition.get("repeats") or 1) + 1):
                slots_out.append({
                    "id": "%s%d" % (variant_id, repeat), "scene_id": scene["id"],
                    "variant_id": variant_id,
                    "content_prompt": scene["content_prompt"],
                    "selection": selection, "color_plan": plan,
                    "prompt": prompt, "prompt_sha256": _sha256_text(prompt),
                    "candidate_text_id": text_id, "candidate_append": append,
                    "candidate_append_sha256": _sha256_text(append),
                    "protected_facts": list(scene.get("protected_facts") or []),
                    "auxiliary_note": scene.get("auxiliary_note", ""),
                    "instructions": "", "post_instructions": "", "prompt_suffix": "",
                    "image_paths": [], "face_quality_boost": False, "expected_output_count": 1,
                })

    source_files = list(spec.get("source_files") or [])
    for rel in [scenes_rel, review_rel] + list(candidate_sources.values()):
        if rel not in source_files:
            source_files.append(rel)
    scenes_doc = {"version": int(spec.get("version") or 1),
                  "style_policy": spec.get("style_policy", ""),
                  "scenes": scenes_out, "variants": variant_out,
                  "comparison_policy": spec.get("comparison_policy", "")}
    target_scenes = _write_once(scenes_file, scenes_doc, log=log, ignore_keys=())
    scenes_written_rel = _rel(Path(target_scenes["path"]))
    source_hashes = {}
    for rel in source_files:
        path = BASE / rel
        if not path.is_file():
            raise FileNotFoundError("来源文件不存在：%s" % rel)
        source_hashes[rel] = _sha256_file(path)
    if scenes_written_rel not in source_hashes:
        source_hashes[scenes_written_rel] = _sha256_file(BASE / scenes_written_rel)

    pack = {
        "kind": spec.get("kind", "app-theme-color-direction-validation"),
        "version": int(spec.get("version") or 1),
        "id": spec["id"],
        "status": "prompts_frozen_runtime_parameters_not_yet_frozen",
        "out_dir": spec["out_dir"],
        "generation_channel": spec.get("generation_channel", "gemini_text_only"),
        "image_http_budget": int(spec["image_http_budget"]),
        "review_http_budget": int(spec["review_http_budget"]),
        "automatic_retry": False,
        "runtime_freeze_required": list(spec.get("runtime_freeze_required") or [
            "api_node", "base_url_without_credentials", "model", "aspect_ratio", "resolution",
            "transport_timeout", "retry_policy", "review_model",
            "review_endpoint_without_credentials", "review_timeout"]),
        "review_prompt_path": review_rel,
        "review_groups": [dict(group) for group in spec.get("review_groups") or []],
        "slots": slots_out,
        "source_hashes": source_hashes,
        "limitations": list(spec.get("limitations") or []),
        "comparison_policy": spec.get("comparison_policy", ""),
        "scenes_path": scenes_written_rel,
    }
    for key in ("tone_names", "tone_measure", "tone_pairwise_field", "comparisons",
                "verdict_texts", "candidate_texts", "candidate_text_sources",
                "candidate_append_variants", "candidate_policy"):
        if spec.get(key) is not None:
            pack[key] = spec[key]
    pack["source_spec"] = _rel(spec_path)
    pack["pack_hash"] = pack_digest(pack)

    written = {"scenes": target_scenes}
    out_pack = Path(out_pack_path or (BASE / ("prompts/color-knowledge/%s-pack.json" % spec["id"])))
    if out_pack.is_file():
        existing = json.loads(out_pack.read_text(encoding="utf-8"))
        if existing.get("pack_hash") not in (None, pack["pack_hash"]):
            raise ValueError("目标冻结包已存在且 pack_hash 不同，拒绝覆盖：%s" % _rel(out_pack))
    write_json_atomic(str(out_pack), pack)
    written["pack"] = {"path": str(out_pack), "pack_hash": pack["pack_hash"]}
    log("[build] 冻结包已编译：%s（%d 槽位 / %d 评审组 / pack_hash=%s）"
        % (_rel(out_pack), len(slots_out), len(pack["review_groups"]), pack["pack_hash"][:12]))
    return {"pack_hash": pack["pack_hash"], "pack": str(out_pack), "scenes": str(target_scenes["path"]),
            "slots": [slot["id"] for slot in slots_out], "written": written}


# --- 3. ledger --------------------------------------------------------------

LEDGER_FILENAME = "ledger.json"
SLOT_STATES = ("planned", "sending", "generated", "failed", "unknown_after_send")


def ledger_path(out_dir):
    return _resolve_out(out_dir) / LEDGER_FILENAME


def load_ledger(out_dir, pack=None):
    path = ledger_path(out_dir)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"label": (pack or {}).get("id", LABEL), "created_at": _now(),
            "pack_hash": (pack or {}).get("pack_hash"),
            "budget": {"image_http_budget": (pack or {}).get("image_http_budget", 10), "review_http_budget": (pack or {}).get("review_http_budget", 4)},
            "counters": {"image_sends": 0, "review_sends": 0},
            "slots": {}, "reviews": {}, "events": [], "paused": None,
            "note": "sending 在发送前原子落盘；generated/failed/unknown_after_send 为终态"}


def save_ledger(out_dir, ledger):
    write_json_atomic(str(ledger_path(out_dir)), ledger)


def _event(ledger, kind, **fields):
    ledger.setdefault("events", []).append({"at": _now(), "kind": kind, **fields})


def slot_inputs(out_dir, slot_id):
    raw = _resolve_out(out_dir) / "images" / "raw" / slot_id
    return sorted(raw.glob("*")) if raw.is_dir() else []


def slot_outputs_intact(out_dir, entry):
    """成功槽位复用判据：文件仍在且 hash 未变。"""
    outputs = entry.get("outputs") or []
    if not outputs:
        return False
    for row in outputs:
        path = Path(row.get("path") or "")
        if not path.is_file():
            return False
        if _sha256_file(path) != row.get("sha256"):
            return False
    return True


def slot_request(pack, slot, runtime):
    """一条槽位的实际发送参数（后端实参）；wire body 证据在执行后补记。"""
    generation = runtime.get("generation") or {}
    return {
        "label": (pack or {}).get("id", LABEL), "slot_id": slot["id"], "scene_id": slot["scene_id"],
        "variant_id": slot["variant_id"], "protocol": pack.get("id"),
        "channel": IMAGE_API_TYPE, "api_type": IMAGE_API_TYPE,
        "host": generation.get("host"), "base_url_without_credentials": generation.get("base_url_without_credentials"),
        "model": generation.get("model"),
        "prompt": slot["prompt"], "prompt_sha256": slot["prompt_sha256"],
        "content_prompt_sha256": _sha256_text(slot["content_prompt"]),
        "plan_hash": (slot.get("color_plan") or {}).get("plan_hash"),
        "color_plan_schema_version": (slot.get("color_plan") or {}).get("schema_version"),
        "resolution": generation.get("resolution"), "aspect_ratio": generation.get("aspect_ratio"),
        "transport_timeout_seconds": generation.get("transport_timeout_seconds"),
        "image_paths": [], "instructions": "", "post_instructions": "", "prompt_suffix": "",
        "face_quality_boost": False, "expected_output_count": 1,
        "evidence": {"kind": "backend_arguments",
                     "note": "这些是传给 generate_image_aigc2d 的实参；脱敏 HTTP 证据在发送后补记"},
    }


def _log_candidates(log_file=None):
    """api_backend 的日志文件名在**导入时**定死，跨零点后仍写前一天的文件，所以两天都要找。"""
    if log_file:
        return [Path(log_file)]
    candidates = []
    now = time.localtime()
    for delta in (0, 1):
        day = time.strftime("%Y-%m-%d", time.localtime(time.mktime(now) - delta * 86400))
        path = BASE / "log" / (day + ".log")
        if path.is_file():
            candidates.append(path)
    return candidates


def _http_evidence(slot_id, prompt, log_file=None):
    """从 api_backend 的日志里取回该次请求的脱敏 wire body（不含 headers/凭据）。

    同一场景的 A–E 共享逐字相同的内容正文，所以**不能**用前缀判定是哪一条请求：
    必须要求日志里的 `text` 与本次正文**逐字全等**才算命中；只前缀相同的另记为
    `prefix_only` 候选（保留数量），不冒充证据。同一槽位重发过就会有多条全等命中，
    取最后一条作本次产物对应的请求，并记录命中总数。
    """
    paths = _log_candidates(log_file)
    if not paths:
        return {"status": "log_not_found", "log_file": _rel(BASE / "log")}
    head = prompt[:80]
    exact, prefix_only = [], []
    for path in paths:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            if "请求数据:" not in line:
                continue
            block = []
            for follow in lines[index + 1:]:
                if re.match(r"^\[\d{4}-\d{2}-\d{2} ", follow) and block:
                    break
                block.append(follow)
            blob = "\n".join(block).strip()
            if not blob.startswith("{"):
                continue
            try:
                payload = json.loads(blob)
            except Exception:  # noqa: BLE001
                continue
            parts = ((payload.get("contents") or [{}])[0].get("parts") or [])
            text_part = next((part.get("text") for part in parts if isinstance(part, dict) and "text" in part), "")
            if not text_part:
                continue
            url_line = ""
            for back in range(index, max(index - 40, 0), -1):
                if lines[back].startswith("[") and "请求 URL:" in lines[back]:
                    url_line = lines[back].split("请求 URL:", 1)[1].strip()
                    break
            row = {"slot_id": slot_id, "log_file": _rel(path), "line": index + 1,
                   "request_url": url_line,
                   "image_config": (((payload.get("generationConfig") or {}).get("imageConfig")) or {}),
                   "inline_data_parts": sum(1 for part in parts if isinstance(part, dict) and "inline_data" in part),
                   "payload_sha256": _sha256_text(blob),
                   "text_chars": len(text_part),
                   "headers_note": "api_backend 的 headers 行含部分 key 明文，按约定**不**抄录进实验记录"}
            if text_part == prompt:
                exact.append(row)
            elif text_part.startswith(head):
                common = 0
                for a in range(min(len(text_part), len(prompt))):
                    if text_part[a] != prompt[a]:
                        break
                    common += 1
                prefix_only.append({**row, "common_prefix_chars": common})
    if exact:
        primary = dict(exact[-1])
        primary.update({"status": "captured", "prompt_matches_sent_request": True,
                        "matches": len(exact), "earlier_matches": len(exact) - 1,
                        "prefix_only_candidates": len(prefix_only)})
        return primary
    if prefix_only:
        return {**prefix_only[-1], "status": "prefix_only_not_exact",
                "prompt_matches_sent_request": False, "candidates": len(prefix_only)}
    return {"status": "not_found_in_log", "log_file": _rel(paths[0]), "slot_id": slot_id}


def _replay_evidence(out, slot_id, prompt):
    """收集后端落下的请求回放文件（本次是 api_backend 新版行为），只保留脱敏摘要。

    回放文件里 headers 含明文 key，已按 `run-notes/credential-redaction.json` 替换为占位符；
    这里只记录路径、URL、body hash 与「body 正文与发送正文逐字一致」的校验结果。
    """
    folder = out / "images" / "raw" / slot_id
    rows = []
    for path in sorted(folder.glob("*replay*.json")) if folder.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            rows.append({"file": path.name, "status": "unreadable"})
            continue
        body = data.get("body") or {}
        parts = ((body.get("contents") or [{}])[0].get("parts") or [])
        text = next((part.get("text") for part in parts if isinstance(part, dict) and "text" in part), "")
        rows.append({"file": path.name, "status": "captured", "url": data.get("url"),
                     "body_sha256": _sha256_text(json.dumps(body, ensure_ascii=False, sort_keys=True)),
                     "prompt_matches_body": text == prompt,
                     "image_config": ((body.get("generationConfig") or {}).get("imageConfig")) or {},
                     "credentials": "headers 里的 key 已替换为 <REDACTED_API_KEY>"})
    return rows


def collect_evidence(out_dir, pack, *, log=print):
    """离线补记/刷新已生成槽位的 wire 证据（日志窗口 + 后端请求回放），0 联网、不改任何产物。"""
    out = _resolve_out(out_dir)
    ledger = load_ledger(out, pack)
    updated = []
    for slot in pack["slots"]:
        entry = (ledger.get("slots") or {}).get(slot["id"]) or {}
        if entry.get("status") != "generated":
            continue
        entry["http_evidence"] = _http_evidence(slot["id"], slot["prompt"])
        entry["wire_replay"] = _replay_evidence(out, slot["id"], slot["prompt"])
        updated.append({"slot_id": slot["id"],
                        "log_evidence": entry["http_evidence"].get("status"),
                        "replay_files": len(entry["wire_replay"])})
    save_ledger(out, ledger)
    for row in updated:
        log("[evidence] %s：log=%s / replay=%d" % (row["slot_id"], row["log_evidence"], row["replay_files"]))
    return {"updated": updated}


def _image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as image:
            return [int(image.width), int(image.height)]
    except Exception:  # noqa: BLE001
        return None


# --- 4. 生成（每槽位最多一次发送） ------------------------------------------

def run_slots(out_dir, pack, runtime, *, only=None, log=print, dry_run=False):
    out = _resolve_out(out_dir)
    ledger = load_ledger(out, pack)
    if ledger.get("pack_hash") and ledger["pack_hash"] != pack.get("pack_hash"):
        raise ValueError("ledger 属于另一份冻结包，停止执行（不覆盖旧记录）")
    ledger["pack_hash"] = pack.get("pack_hash")
    slots = [slot for slot in pack["slots"] if not only or slot["id"] in only]
    extra = int(ledger["counters"].get("authorized_extra_sends") or 0)
    budget = int((pack.get("image_http_budget") or 10)) + extra
    if extra:
        log("[budget] 冻结包预算 %s + 用户已授权额外发送 %d = 有效预算 %d（超出部分会在报告里列出）"
            % (pack.get("image_http_budget"), extra, budget))
    results = []
    for slot in slots:
        sid = slot["id"]
        entry = ledger["slots"].setdefault(sid, {
            "slot_id": sid, "scene_id": slot["scene_id"], "variant_id": slot["variant_id"],
            "status": "planned", "attempts": 0, "pack_hash": pack.get("pack_hash"),
            "plan_hash": (slot.get("color_plan") or {}).get("plan_hash"),
            "prompt_sha256": slot["prompt_sha256"]})
        request_path = out / "requests" / sid
        if entry.get("status") == "generated" and slot_outputs_intact(out, entry):
            log("[cached] %s：成功产图仍在且 hash 一致，零联网复用" % sid)
            results.append({"slot_id": sid, "status": "generated", "reused": True})
            continue
        if entry.get("status") == "failed":
            log("[skip] %s：已明确失败，按预算不自动补图（需人工授权新目录）" % sid)
            results.append({"slot_id": sid, "status": "failed", "reused": False,
                            "error_type": entry.get("error_type")})
            continue
        if entry.get("status") in ("sending", "unknown_after_send"):
            ledger["paused"] = {"at": _now(), "slot_id": sid, "status": entry.get("status"),
                                "reason": "发送已发生但结果未知；按约定暂停后续联网，等待用户决定"}
            save_ledger(out, ledger)
            log("[paused] %s 状态为 %s：暂停后续联网，等待用户决定（不自动重发）" % (sid, entry.get("status")))
            results.append({"slot_id": sid, "status": entry.get("status"), "paused": True})
            return {"results": results, "ledger": ledger, "paused": True}
        if ledger["counters"]["image_sends"] >= budget:
            log("[budget] 图像发送已达上限 %d，停止后续槽位" % budget)
            break

        plan = slot["color_plan"]
        try:
            validate_snapshot(plan)
        except Exception as exc:  # noqa: BLE001
            entry.update(status="failed", error_type="snapshot_invalid", error=str(exc)[:300])
            save_ledger(out, ledger)
            results.append({"slot_id": sid, "status": "failed", "error_type": "snapshot_invalid"})
            continue

        request = slot_request(pack, slot, runtime)
        if dry_run:
            _write_once(request_path / "planned-request.json", request, log=log)
            entry.update(status="planned", planned_request=str(request_path / "planned-request.json"))
            results.append({"slot_id": sid, "status": "planned", "dry_run": True})
            continue

        # 发送前：先落 sending（原子），再记一次预算
        entry.update(status="sending", attempts=int(entry.get("attempts") or 0) + 1, sent_at=_now(),
                     error_type=None, error=None)
        ledger["counters"]["image_sends"] += 1
        _event(ledger, "image_send", slot_id=sid,
               send_index=ledger["counters"]["image_sends"], prompt_sha256=slot["prompt_sha256"])
        save_ledger(out, ledger)
        request_path.mkdir(parents=True, exist_ok=True)
        _write_once(request_path / "request.json", request, log=log)
        (request_path / "prompt.txt").write_text(slot["prompt"], encoding="utf-8")

        # 主题配色的 App 记录（同一份正文/plan 快照）：发送前登记，成功后补 outputs
        record = record_request(plan, slot["prompt"], model=(runtime.get("generation") or {}).get("model") or "",
                                channel=IMAGE_API_TYPE,
                                context={"slot_id": sid, "variant_id": slot["variant_id"],
                                         "resolution": (runtime.get("generation") or {}).get("resolution"),
                                         "aspect_ratio": (runtime.get("generation") or {}).get("aspect_ratio"),
                                         "entry": "utils.color_direction_pack"},
                                root=request_path)

        staging = out / "images" / "raw" / sid
        staging.mkdir(parents=True, exist_ok=True)
        log("[generate] %s：第 %d 次发送（本槽位第 %d 次尝试）"
            % (sid, ledger["counters"]["image_sends"], entry["attempts"]))
        started = time.perf_counter()
        try:
            outcome, payload_result = _send_once(slot, runtime, staging, log)
        except Exception as exc:  # noqa: BLE001 —— 发送后任何崩溃都不能让槽位停在 sending
            import traceback
            detail_dir = out / "run-notes" / "send-errors"
            detail_dir.mkdir(parents=True, exist_ok=True)
            detail = detail_dir / ("%s-%s.txt" % (sid, time.strftime("%H%M%S")))
            detail.write_text("slot: %s\nat: %s\n%s: %s\n\n%s"
                              % (sid, _now(), type(exc).__name__, exc, traceback.format_exc()),
                              encoding="utf-8")
            entry.update(status="unknown_after_send", error_type="send_exception:" + type(exc).__name__,
                         error=str(exc)[:300], elapsed_seconds=round(time.perf_counter() - started, 2),
                         finished_at=_now(), http_evidence={"status": "unknown_after_send",
                                                            "error_detail": _rel(detail)})
            ledger["paused"] = {"at": _now(), "slot_id": sid, "status": "unknown_after_send",
                                "reason": "发送过程中抛出异常（%s）；是否成功未知，暂停后续联网，等待用户决定"
                                          % type(exc).__name__}
            _event(ledger, "image_unknown_after_send", slot_id=sid,
                   error_type="send_exception:" + type(exc).__name__)
            save_ledger(out, ledger)
            log("[unknown_after_send] %s：%s: %s；已暂停后续联网（细节 %s）"
                % (sid, type(exc).__name__, str(exc)[:160], detail))
            results.append({"slot_id": sid, "status": "unknown_after_send"})
            return {"results": results, "ledger": ledger, "paused": True}
        elapsed = round(time.perf_counter() - started, 2)
        entry["elapsed_seconds"] = elapsed
        entry["finished_at"] = _now()
        entry["request_record"] = record.get("path")
        server_meta = payload_result.get("server_meta") or {}

        if outcome == "generated":
            returned = [Path(p) for p in payload_result.get("saved_files") or []]
            finals, outputs = [], []
            for index, source in enumerate(returned, 1):
                suffix = source.suffix.lower() if source.suffix else ".png"
                target = out / "images" / ("%s-%d%s" % (sid, index, suffix))
                if target.exists():
                    target = out / "images" / ("%s-%d-%s%s" % (sid, index, time.strftime("%H%M%S"), suffix))
                os.replace(str(source), str(target))          # 只改名，不改字节
                outputs.append({"path": str(target), "name": target.name, "bytes": target.stat().st_size,
                                "sha256": _sha256_file(target), "size": _image_size(target)})
                finals.append(str(target))
            record_outputs(record, finals)
            entry.update(status="generated", outputs=outputs, server_meta=server_meta,
                         protocol_deviation=("multi_candidate" if len(outputs) > 1 else None),
                         returned_names=[p.name for p in returned],
                         http_evidence=_http_evidence(sid, slot["prompt"]))
            _event(ledger, "image_generated", slot_id=sid, outputs=[row["name"] for row in outputs],
                   elapsed_seconds=elapsed)
            log("[generated] %s：%d 张，%.1fs → %s" % (sid, len(outputs), elapsed, outputs[0]["name"]))
        elif outcome == "unknown_after_send":
            entry.update(status="unknown_after_send", error_type=payload_result.get("error_type"),
                         error=payload_result.get("error"),
                         http_evidence={"status": "unknown_after_send"})
            ledger["paused"] = {"at": _now(), "slot_id": sid, "status": "unknown_after_send",
                                "reason": "发送已发生但超时/连接断开，是否成功未知；暂停后续联网，等待用户决定"}
            _event(ledger, "image_unknown_after_send", slot_id=sid,
                   error_type=payload_result.get("error_type"))
            save_ledger(out, ledger)
            log("[unknown_after_send] %s：%s；暂停后续联网，等待用户决定（不自动重发、不当作未收费）"
                % (sid, payload_result.get("error_type")))
            results.append({"slot_id": sid, "status": "unknown_after_send"})
            return {"results": results, "ledger": ledger, "paused": True}
        else:
            entry.update(status="failed", error_type=payload_result.get("error_type"),
                         error=payload_result.get("error"), server_meta=server_meta,
                         protocol_deviation=payload_result.get("protocol_deviation"),
                         http_evidence=_http_evidence(sid, slot["prompt"]))
            _event(ledger, "image_failed", slot_id=sid, error_type=payload_result.get("error_type"))
            log("[failed] %s：%s（消耗该槽位预算，不自动补图）"
                % (sid, payload_result.get("error_type")))
        save_ledger(out, ledger)
        results.append({"slot_id": sid, "status": entry["status"]})
    save_ledger(out, ledger)
    return {"results": results, "ledger": ledger, "paused": bool(ledger.get("paused"))}


def _send_once(slot, runtime, staging: Path, log):
    """一次发送（不重发）。返回 (outcome, payload)。"""
    from modules.others.api_backend import generate_image_aigc2d
    from utils.color_experiment import _redact_server_meta
    generation = runtime.get("generation") or {}
    result = generate_image_aigc2d(
        prompt=slot["prompt"], image_paths=[], model=generation.get("model"),
        aspect_ratio=generation.get("aspect_ratio"), resolution=generation.get("resolution"),
        instructions="", post_instructions="", prompt_suffix="", face_quality_boost=False,
        api_type=IMAGE_API_TYPE, save_sub_dir=str(staging), file_prefix="dir-%s" % slot["id"],
        return_metadata=True, log_callback=log)
    if not isinstance(result, dict):
        return "failed", {"error_type": "unexpected_return_type", "error": type(result).__name__}
    raw = result.get("server_response_raw")
    meta = _redact_server_meta(result)
    saved = [str(path) for path in (result.get("saved_files") or []) if Path(path).is_file()]
    if saved:
        return "generated", {"saved_files": saved, "server_meta": meta}
    if isinstance(raw, dict) and raw.get("error"):
        text = str(raw.get("raw_text") or "")
        return "unknown_after_send", {"error_type": "transport_error",
                                      "error": str(raw.get("error"))[:300],
                                      "error_body_head": text[:200]}
    if isinstance(raw, dict) and raw.get("parse_error"):
        return "failed", {"error_type": "response_parse_error", "error": str(raw.get("parse_error"))[:200],
                          "server_meta": meta}
    candidates = raw.get("candidates") if isinstance(raw, dict) else None
    if isinstance(candidates, list):
        return "failed", {"error_type": "no_image_in_response",
                          "error": "candidates=%d 但没有可保存的图片片段" % len(candidates),
                          "server_meta": meta, "protocol_deviation": "response_without_image"}
    return "failed", {"error_type": "empty_result", "error": "后端返回空结果", "server_meta": meta}


# --- 5. 视觉评审（每组最多一次发送） ----------------------------------------

def _binding_map(slot):
    plan = slot["color_plan"]
    profile = plan.get("profile") or {}
    selection = plan.get("selection") or {}
    direction = plan.get("direction") or {}
    bindings = {
        "base": {"colour": profile.get("base"), "regions": selection.get("environment")},
        "accent": {"colour": profile.get("accent"), "regions": selection.get("accent_regions")},
    }
    if direction.get("area") == "three-level":
        auxiliary = direction.get("auxiliary") or {}
        bindings["auxiliary"] = {"colour": auxiliary.get("colour"),
                                 "regions": direction.get("auxiliary_regions"),
                                 "authorisation": "预先声明的辅助色绑定：该区域允许改色，不算保护色违规"}
    else:
        bindings["auxiliary"] = None
    return bindings


def _scene_facts(pack, scene_id):
    scene = next(scene for scene in pack["slots"] if scene["scene_id"] == scene_id)
    return {"content_facts": scene["content_prompt"], "protected_facts": scene.get("protected_facts") or [],
            "auxiliary_regions": scene.get("auxiliary_regions"),
            "auxiliary_note": scene.get("auxiliary_note")}


def _review_user_prompt(group, pack, rows):
    """评审输入：具体正文事实 + 共同色板 + 每图允许的 binding map；不含变体名与预期方向。"""
    scene_id = group["scene_id"]
    facts = _scene_facts(pack, scene_id)
    plan = rows[0]["slot"]["color_plan"]
    profile = plan.get("profile") or {}
    lines = ["GROUP KIND: %s" % group["kind"],
             "SCENE FACTS (identical for every supplied image; from the frozen content text):",
             facts["content_facts"], "",
             "DECLARED PROTECTED FACTS (identical for every supplied image):",
             "; ".join(facts["protected_facts"]), "",
             "COMMON PALETTE (identical for every supplied image):",
             "base = %s ; accent = %s ; swatches = %s" % (profile.get("base"), profile.get("accent"),
                                                          ", ".join(profile.get("swatches") or [])),
             "",
             "IMAGES (neutral IDs only; each is one independent random generation):"]
    for row in rows:
        bindings = _binding_map(row["slot"])
        lines.append("ID %s bindings: base -> %s (%s) ; accent -> %s (%s)"
                     % (row["neutral_id"], bindings["base"]["regions"], bindings["base"]["colour"],
                        bindings["accent"]["regions"], bindings["accent"]["colour"]))
        if bindings["auxiliary"]:
            aux = bindings["auxiliary"]
            lines.append("ID %s additional declared auxiliary binding: %s -> %s (%s)"
                         % (row["neutral_id"], aux["regions"], aux["colour"], aux["authorisation"]))
        else:
            lines.append("ID %s additional declared auxiliary binding: none"
                         % row["neutral_id"])
    lines += ["", "Neutral IDs supplied to you: %s" % ", ".join(row["neutral_id"] for row in rows),
              "Return the JSON schema from your system instructions. Chinese explanations. "
              "No experiment success labels, no aesthetic winner, no numeric total score."]
    return "\n".join(lines)


def _review_slots(out_dir, pack, group, ledger):
    rows = []
    for slot_id in group["slot_ids"]:
        entry = (ledger.get("slots") or {}).get(slot_id) or {}
        slot = next((item for item in pack["slots"] if item["id"] == slot_id), None)
        if not slot:
            continue
        intact = entry.get("status") == "generated" and slot_outputs_intact(out_dir, entry)
        rows.append({"slot_id": slot_id, "slot": slot, "entry": entry, "usable": intact})
    return rows


def _shuffled_mapping(usable_ids):
    rng = random.SystemRandom()
    order = list(usable_ids)
    rng.shuffle(order)
    return {("N%d" % (index + 1)): slot_id for index, slot_id in enumerate(order)}


def review_groups(out_dir, pack, runtime, *, only=None, log=print, dry_run=False):
    out = _resolve_out(out_dir)
    ledger = load_ledger(out, pack)
    review_prompt = (BASE / pack.get("review_prompt_path", REVIEW_PROMPT_PATH)).read_text(encoding="utf-8")
    rules_hash = _sha256_text(review_prompt)
    budget = int(pack.get("review_http_budget") or 4)
    results = []
    for group in pack["review_groups"]:
        gid = group["id"]
        if only and gid not in only:
            continue
        entry = ledger["reviews"].setdefault(gid, {
            "review_id": gid, "kind": group["kind"], "scene_id": group["scene_id"],
            "slot_ids": group["slot_ids"], "status": "planned", "attempts": 0})
        folder = out / "reviews" / gid
        folder.mkdir(parents=True, exist_ok=True)
        rows = _review_slots(out, pack, group, ledger)
        usable = [row for row in rows if row["usable"]]
        missing = [row["slot_id"] for row in rows if not row["usable"]]
        entry["missing_slots"] = missing
        if entry.get("status") == "reviewed" and (folder / "review.json").is_file():
            cached = json.loads((folder / "review.json").read_text(encoding="utf-8"))
            if cached.get("rules_hash") == rules_hash:
                log("[cached] 评审 %s：成功结果复用，零联网" % gid)
                results.append({"review_id": gid, "status": "reviewed", "reused": True})
                continue
        if entry.get("status") in ("failed", "parse_error", "sending", "unknown_after_send"):
            log("[skip] 评审 %s：上一次发送已是终态（%s），不做第二次「修复评审」"
                % (gid, entry.get("status")))
            results.append({"review_id": gid, "status": entry.get("status"), "reused": False})
            continue
        if len(usable) < 1 and not dry_run:
            entry.update(status="skipped_no_images", reason="该组没有可用的成功产图")
            save_ledger(out, ledger)
            log("[skip] 评审 %s：没有可用产图，跳过（不拿别的图替代）" % gid)
            results.append({"review_id": gid, "status": "skipped_no_images"})
            continue
        if ledger["counters"]["review_sends"] >= budget:
            log("[budget] 评审发送已达上限 %d，停止后续评审组" % budget)
            break
        if not usable:
            log("[plan] 评审 %s：预演阶段暂无可用产图（不写终态、不发送）" % gid)
            results.append({"review_id": gid, "status": "planned_no_images_yet", "dry_run": True})
            continue

        mapping_file = folder / "mapping.json"
        mapping_preview_only = False
        if mapping_file.is_file():
            mapping = json.loads(mapping_file.read_text(encoding="utf-8"))["neutral_to_slot"]
            if sorted(mapping.values()) != sorted(row["slot_id"] for row in usable):
                entry.update(status="mapping_stale",
                             reason="已冻结的中性映射与当前可用产图集合不一致；不自动重排、不发送")
                save_ledger(out, ledger)
                log("[skip] 评审 %s：映射与可用产图集合不一致，保持原映射不发送（人工决定）" % gid)
                results.append({"review_id": gid, "status": "mapping_stale"})
                continue
        else:
            usable_ids = {row["slot_id"] for row in usable}
            declared = [sid for sid in (group.get("neutral_order") or []) if sid in usable_ids]
            if declared:
                # 平衡顺序：逐字沿用冻结包声明的展示位置，不随机、不挑选
                mapping = {"N%d" % (index + 1): slot_id for index, slot_id in enumerate(declared)}
                if len(declared) != len(usable):
                    log("[review] %s：声明顺序中有 %d 张产图不可用，其余按声明的相对顺序保留"
                        % (gid, len(usable) - len(declared)))
                else:
                    log("[review] %s：使用冻结包声明的平衡展示顺序" % gid)
            else:
                mapping = _shuffled_mapping([row["slot_id"] for row in usable])
            mapping_preview_only = (bool(dry_run) or len(usable) < 3) and not declared
            if not mapping_preview_only:
                write_json_atomic(str(mapping_file), {"review_id": gid, "at": _now(),
                                                      "neutral_to_slot": mapping,
                                                      "note": "映射单独保存，不出现在评审输入里"})
        by_slot = {row["slot_id"]: row for row in usable}
        ordered = [{"neutral_id": neutral, "slot": by_slot[slot_id]["slot"], "slot_id": slot_id,
                    "entry": by_slot[slot_id]["entry"]}
                   for neutral, slot_id in sorted(mapping.items()) if slot_id in by_slot]
        user = _review_user_prompt(group, pack, ordered)
        images = [(row["neutral_id"], row["entry"]["outputs"][0]["path"], row["entry"]["outputs"][0]["sha256"])
                  for row in ordered]
        cache_key = digest({"rules_hash": rules_hash, "model": (runtime.get("review") or {}).get("model"),
                            "endpoint": (runtime.get("review") or {}).get("endpoint_without_credentials"),
                            "facts": _scene_facts(pack, group["scene_id"]),
                            "kind": group["kind"],
                            "images": sorted([{"neutral_id": neutral, "sha256": sha}
                                              for neutral, _path, sha in images],
                                             key=lambda row: row["sha256"]),
                            "bindings": user})
        request = {"review_id": gid, "kind": group["kind"], "scene_id": group["scene_id"],
                   "model": (runtime.get("review") or {}).get("model"),
                   "endpoint_without_credentials": (runtime.get("review") or {}).get("endpoint_without_credentials"),
                   "max_completion_tokens": (runtime.get("review") or {}).get("max_completion_tokens"),
                   "timeout_seconds": (runtime.get("review") or {}).get("timeout_seconds"),
                   "system_prompt_file": pack.get("review_prompt_path", REVIEW_PROMPT_PATH), "rules_hash": rules_hash,
                   "user_prompt": user, "cache_key": cache_key,
                   "images": [{"neutral_id": neutral, "file": Path(path).name, "sha256": sha}
                              for neutral, path, sha in images],
                   "missing_slots": missing,
                   "display_order_source": "declared_balanced" if declared else "random_shuffled",
                   "planned_neutral_map": mapping if declared else None,
                   "note": "评审输入不含变体名、不含期望方向；neutral ID 到槽位的映射单独保存"}
        if len(usable) < 2:
            request["note"] = (request["note"] + "；不足 2 图：本次只做单图观察，tone_pairwise 应为空数组")
        if dry_run:
            request["mapping_preview_only"] = True
            _write_once(folder / "planned-request.json", request, log=log)
            results.append({"review_id": gid, "status": "planned", "dry_run": True,
                            "usable": len(usable), "missing": missing})
            continue

        entry.update(status="sending", attempts=int(entry.get("attempts") or 0) + 1, sent_at=_now(),
                     cache_key=cache_key, neutral_map=mapping, missing_slots=missing)
        ledger["counters"]["review_sends"] += 1
        _event(ledger, "review_send", review_id=gid, send_index=ledger["counters"]["review_sends"],
               cache_key=cache_key)
        save_ledger(out, ledger)
        (folder / "system-prompt.md").write_text(review_prompt, encoding="utf-8")
        _write_once(folder / "request.json", request, log=log)
        log("[review] %s：第 %d 次评审发送" % (gid, ledger["counters"]["review_sends"]))

        outcome, payload = _review_once(runtime, review_prompt, user,
                                        [path for _n, path, _s in images], log)
        entry["finished_at"] = _now()
        (folder / "response.raw.txt").write_text(payload.get("raw") or "", encoding="utf-8")
        if outcome == "reviewed":
            record = {"review_id": gid, "kind": group["kind"], "scene_id": group["scene_id"],
                      "at": _now(), "model": request["model"],
                      "endpoint_without_credentials": request["endpoint_without_credentials"],
                      "rules_hash": rules_hash, "cache_key": cache_key,
                      "neutral_to_slot": mapping, "missing_slots": missing,
                      "images": request["images"], "response": payload["parsed"],
                      "response_sha256": _sha256_text(json.dumps(payload["parsed"], ensure_ascii=False, sort_keys=True)),
                      "raw_sha256": _sha256_text(payload.get("raw") or ""),
                      "status": "model_observation_not_gate"}
            _write_once(folder / "review.json", record, log=log)
            entry.update(status="reviewed", review_record=str(folder / "review.json"))
            _event(ledger, "review_completed", review_id=gid)
            log("[reviewed] %s：已解析并落盘" % gid)
        elif outcome == "unknown_after_send":
            entry.update(status="unknown_after_send", error_type=payload.get("error_type"),
                         error=payload.get("error"))
            ledger["paused"] = {"at": _now(), "review_id": gid, "status": "unknown_after_send",
                                "reason": "评审发送已发生但结果未知；暂停后续联网，等待用户决定"}
            _event(ledger, "review_unknown_after_send", review_id=gid,
                   error_type=payload.get("error_type"))
            save_ledger(out, ledger)
            log("[unknown_after_send] 评审 %s：暂停后续联网，等待用户决定（不自动重发）" % gid)
            results.append({"review_id": gid, "status": "unknown_after_send"})
            return {"results": results, "ledger": ledger, "paused": True}
        else:
            entry.update(status=payload.get("status") or "failed", error_type=payload.get("error_type"),
                         error=payload.get("error"))
            _event(ledger, "review_failed", review_id=gid, error_type=payload.get("error_type"))
            log("[failed] 评审 %s：%s；原响应已保留，禁止自动发第二次「修复评审」"
                % (gid, payload.get("error_type")))
        save_ledger(out, ledger)
        results.append({"review_id": gid, "status": entry["status"]})
    save_ledger(out, ledger)
    return {"results": results, "ledger": ledger, "paused": bool(ledger.get("paused"))}


def _review_once(runtime, system, user, image_paths, log):
    from utils.analysis_gpt_prompt import call_text_model
    review = runtime.get("review") or {}
    from utils.analysis_gpt_prompt import load_text_api_config
    cfg = load_text_api_config()
    if not cfg.get("api_key"):
        return "failed", {"status": "failed", "error_type": "no_review_key",
                          "error": "评审通道 key 不可用", "raw": ""}
    try:
        raw = call_text_model(cfg["base_url"], cfg["api_key"], review.get("model") or cfg.get("model"),
                              system, user, timeout=int(review.get("timeout_seconds") or REVIEW_TIMEOUT_SECONDS),
                              max_tokens=int(review.get("max_completion_tokens") or REVIEW_MAX_TOKENS),
                              image_paths=list(image_paths))
    except Exception as exc:  # noqa: BLE001
        return "unknown_after_send", {"error_type": type(exc).__name__, "error": str(exc)[:300], "raw": ""}
    if not str(raw or "").strip():
        return "failed", {"status": "empty_response", "error_type": "empty_response",
                          "error": "视觉通道返回空内容", "raw": ""}
    try:
        parsed = _parse_json_object(raw)
    except Exception as exc:  # noqa: BLE001
        return "failed", {"status": "parse_error", "error_type": "json_parse_error",
                          "error": str(exc)[:200], "raw": raw}
    expected, seen = [], set()
    for found in re.findall(r"ID (N\d+)\b", user):
        if found not in seen:
            seen.add(found)
            expected.append(found)
    returned = [str(row.get("id")) for row in (parsed.get("images") or []) if isinstance(row, dict)]
    if sorted(set(expected)) != sorted(set(returned)):
        return "failed", {"status": "id_mismatch", "error_type": "neutral_id_mismatch",
                          "error": "返回的 neutral ID 与输入不一致：%s vs %s" % (sorted(set(expected)), sorted(set(returned))),
                          "raw": raw, "parsed": parsed}
    return "reviewed", {"parsed": parsed, "raw": raw}


def _parse_json_object(text):
    body = str(text).strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body)
    try:
        value = json.loads(body)
    except Exception:  # noqa: BLE001
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(body[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("评审返回不是 JSON 对象")
    return value


# --- 6. 报告（渲染失败不影响证据） ------------------------------------------

def _decode_tone(pairwise, mapping, slot_a, slot_other, measure="lighter"):
    """从 tone 组的客观两两明暗里解出「另一张相对 A 更亮/更暗」。"""
    id_for_slot = {slot: neutral for neutral, slot in mapping.items()}
    if slot_a not in id_for_slot or slot_other not in id_for_slot:
        return {"verdict": "missing_mapping"}
    id_a, id_b = id_for_slot[slot_a], id_for_slot[slot_other]
    for row in pairwise or []:
        ids = [str(item) for item in (row.get("ids") or [])]
        if sorted(ids) != sorted([id_a, id_b]):
            continue
        lighter = str(row.get(measure) or "")
        uncertain = (not lighter) or ("tie" in lighter.lower()) or ("uncertain" in lighter.lower()) \
            or lighter.lower() in ("不确定",)
        return {"lighter": lighter, "ids": ids, "evidence_regions": row.get("evidence_regions") or [],
                "measure": measure, "confounded": row.get("confounded"), "reason": row.get("reason"),
                "other_is_lighter": lighter == id_b, "a_is_lighter": lighter == id_a,
                "tie_or_uncertain": uncertain}
    return {"verdict": "no_pair_entry"}


def _observation(parsed, neutral_id):
    for row in (parsed or {}).get("images") or []:
        if isinstance(row, dict) and str(row.get("id")) == neutral_id:
            return row
    return {}


def render_report(out_dir, pack, runtime, *, decoding=None, log=print):
    """渲染 RESULTS.json / RUN-REPORT.md / gallery.html。

    步骤刻意分离：任何渲染异常先写 `RUN-REPORT-render-failure.txt`，已落盘的证据不动，
    修复只允许离线重跑本函数（0 联网）。
    """
    out = _resolve_out(out_dir)
    ledger = load_ledger(out, pack)
    decoding = decoding if decoding is not None else _load_decoding(out)
    slots_view, review_view = [], []
    by_slot = {slot["id"]: slot for slot in pack["slots"]}

    previous = {}
    for row in ledger.get("previous_attempts") or []:
        previous.setdefault(row.get("slot_id"), []).append(
            {key: row.get(key) for key in ("status", "attempts", "sent_at", "error_type", "error",
                                           "moved_at", "moved_reason", "moved_note")})

    for slot in pack["slots"]:
        entry = (ledger.get("slots") or {}).get(slot["id"]) or {}
        row = {"slot_id": slot["id"], "scene_id": slot["scene_id"], "variant_id": slot["variant_id"],
               "status": entry.get("status", "planned"), "attempts": entry.get("attempts", 0),
               "sent_at": entry.get("sent_at"), "finished_at": entry.get("finished_at"),
               "elapsed_seconds": entry.get("elapsed_seconds"),
               "error_type": entry.get("error_type"), "error": entry.get("error"),
               "plan_hash": (slot.get("color_plan") or {}).get("plan_hash"),
               "prompt_sha256": slot["prompt_sha256"],
               "outputs": entry.get("outputs") or [], "protocol_deviation": entry.get("protocol_deviation"),
               "http_evidence_status": (entry.get("http_evidence") or {}).get("status"),
               "wire_replay": [{"file": row.get("file"), "status": row.get("status"),
                                "prompt_matches_body": row.get("prompt_matches_body")}
                               for row in (entry.get("wire_replay") or [])],
               "previous_attempts": previous.get(slot["id"], []),
               "resend_authorised": entry.get("resend_authorised"),
               "settled": entry.get("settled"),
               "auxiliary_regions": ((slot.get("color_plan") or {}).get("direction") or {}).get("auxiliary_regions")}
        slots_view.append(row)

    for group in pack["review_groups"]:
        entry = (ledger.get("reviews") or {}).get(group["id"]) or {}
        folder = out / "reviews" / group["id"]
        parsed, mapping = {}, entry.get("neutral_map") or {}
        if (folder / "review.json").is_file():
            record = json.loads((folder / "review.json").read_text(encoding="utf-8"))
            parsed, mapping = record.get("response") or {}, record.get("neutral_to_slot") or mapping
        review_view.append({"review_id": group["id"], "kind": group["kind"], "scene_id": group["scene_id"],
                            "slot_ids": group["slot_ids"], "status": entry.get("status", "planned"),
                            "attempts": entry.get("attempts", 0), "sent_at": entry.get("sent_at"),
                            "error_type": entry.get("error_type"), "error": entry.get("error"),
                            "neutral_to_slot": mapping, "missing_slots": entry.get("missing_slots") or [],
                            "parsed": parsed, "review_record": entry.get("review_record")})

    directions = _decode_directions(pack, review_view, decoding)
    anomalies = _collect_anomalies(ledger, slots_view, review_view)
    results = {
        "label": (pack or {}).get("id", LABEL), "at": _now(), "out_dir": str(out),
        "pack_hash": pack.get("pack_hash"), "runtime_hash": runtime.get("runtime_hash"),
        "budget": {"image_http_budget": pack.get("image_http_budget"),
                   "review_http_budget": pack.get("review_http_budget")},
        "counters": ledger.get("counters"),
        "paused": ledger.get("paused"),
        "slots": slots_view, "review_groups": review_view,
        "directions": directions,
        "incidents": _load_incidents(out),
        "uncovered_items": (list(pack["limitations"]) if (pack.get("tone_names") or pack.get("comparisons"))
                            and pack.get("limitations") else list(UNCOVERED_ITEMS)),
        "anomalies": anomalies,
        "notes": [
            "generated 只表示产图成功，不等于配色验收成功",
            "色彩方向可观察，但内容失败的图不算可用成功",
            ("本轮为同方向内 A/B/C 三条件的重复验证，不含辅助绑定与面积变体" if pack.get("comparisons")
             else ("本轮仅清色/浊色，不含辅助绑定与面积变体" if pack.get("tone_names")
                   else "E 是「辅助色绑定 + 三层主次」的合并变体，不是纯面积变体")),
            "重复样本只说明随机波动与提示词作用的相对大小，无 seed 保证，不能提供统计显著性、生产门禁或跨重绘泛化结论",
        ],
    }
    _write_derived(out / "RESULTS.json", results, log=log)
    report = _render_markdown(results, pack, runtime)
    _write_derived(out / "RUN-REPORT.md", report, log=log)
    gallery = _render_gallery(results, pack)
    _write_derived(out / "gallery.html", gallery, log=log)
    log("[direction] 报告已渲染：RESULTS.json / RUN-REPORT.md / gallery.html（0 联网）")
    return results


def _load_decoding(out_dir):
    path = _resolve_out(out_dir) / "decoding.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _usability_gate(observation, expected_regions=(), require_ids=False):
    """Experimental evidence qualification, separate from visible direction."""
    checks = {"content_status": "pass", "palette_status": "conform",
              "protected_hues_status": "pass", "detail_readability": "pass"}
    issues = [key + ":" + str(observation.get(key, "missing"))
              for key, expected in checks.items() if observation.get(key) != expected]
    bindings = observation.get("binding_observations") or []
    if not bindings:
        issues.append("binding_observations:missing")
    for binding in bindings:
        if binding.get("status") != "conform" or binding.get("visible") != "yes":
            issues.append("binding:" + str(binding.get("region", "unknown")))
    if "unauthorized_additions" not in observation or observation.get("unauthorized_additions"):
        issues.append("unauthorized_additions:missing_or_present")
    if observation.get("content_issues") or observation.get("protected_hues_issues"):
        issues.append("reported_compliance_issues")
    # Old reviews used translated region names. Do not invent failures by
    # comparing Chinese labels with English prompt text. New reviews use IDs.
    identified = any("region_id" in b for b in bindings)
    if require_ids and not identified:
        issues.append("binding_region_ids:missing")
    if identified:
        seen = {str(b.get("region_id", "")).strip().casefold() for b in bindings}
        for region in expected_regions:
            if region.strip().casefold() not in seen:
                issues.append("binding_missing:" + region)
    return {"usable": not issues, "issues": issues,
            "coverage": "literal_ids_checked" if identified else "legacy_labels_require_human_check"}


def _qualify_directions(out, review_view, pack):
    slots = {slot["id"]: slot for slot in pack["slots"]}

    def regions(slot_id):
        selection = (slots.get(slot_id) or {}).get("selection") or {}
        return [r.strip() for key in ("environment", "accent_regions", "auxiliary_regions")
                for r in str(selection.get(key) or "").split(",") if r.strip()]

    require_ids = bool(pack.get("tone_names") or pack.get("comparisons"))
    for direction in out.values():
        for scene in direction["scenes"]:
            slot_id = scene.get("slot_id") or ""
            group = next((row for row in review_view
                          if (scene.get("review_id") and row.get("review_id") == scene["review_id"])
                          or (not scene.get("review_id")
                              and slot_id in (row.get("neutral_to_slot") or {}).values()
                              and row.get("status") == "reviewed"
                              and row.get("review_id", "").endswith(
                                  "tone" if "pairwise" in scene else "hierarchy"))), {})
            mapping = group.get("neutral_to_slot") or {}
            reverse = {slot: neutral for neutral, slot in mapping.items()}
            parsed = group.get("parsed") or {}
            baseline_id = scene.get("baseline_slot_id") or (slot_id.split("-")[0] + "-A")
            expect_candidate = ((scene.get("expect", "candidate") != "baseline") if scene.get("expect")
                                else slot_id.endswith("-B"))
            baseline = _observation(parsed, reverse.get(baseline_id, ""))
            scene["candidate_gate"] = _usability_gate(scene.get("observation") or {}, regions(slot_id), require_ids)
            scene["baseline_gate"] = _usability_gate(baseline, regions(baseline_id), require_ids)
            pair = scene.get("pairwise") or {}
            scene["direction_observed"] = ((bool(pair.get("other_is_lighter")) if expect_candidate
                                            else bool(pair.get("a_is_lighter"))) if "pairwise" in scene
                                           else scene.get("verdict") in ("direction_observed", "layer_observed"))
            scene["candidate_usable_success"] = scene["direction_observed"] and scene["candidate_gate"]["usable"]
            scene["usable_success"] = (scene["candidate_usable_success"] and scene["baseline_gate"]["usable"]
                                       and (pair.get("confounded") is False if "pairwise" in scene
                                            else not (scene.get("observation", {}).get("geometry_confounds") or baseline.get("geometry_confounds"))))
        direction["gate_version"] = 2
        direction["observed_direction_count"] = sum(bool(row.get("direction_observed")) for row in direction["scenes"])
        direction["candidate_usable_success"] = sum(bool(row.get("candidate_usable_success")) for row in direction["scenes"])
        direction["usable_success"] = sum(bool(row.get("usable_success")) for row in direction["scenes"])
    return out


def _decode_comparisons(pack, by_id, decoding):
    """通用比较解码：每个方向由 pack 显式给出的（基准槽位, 候选槽位）决定。

    与旧轮的 S1/S2 命名约定无关：重复次数、条件数量、评审组划分全部来自冻结包。
    """
    field = pack.get("tone_pairwise_field", "tone_pairwise")
    measure = pack.get("tone_measure", "lighter")
    out = {}
    for item in pack.get("comparisons") or []:
        name = item["name"]
        row = by_id.get(item.get("group_id")) or {}
        mapping = row.get("neutral_to_slot") or {}
        parsed = row.get("parsed") or {}
        reverse = {slot: neutral for neutral, slot in mapping.items()}
        candidate, baseline = item["candidate"], item["baseline"]
        scene = {"scene_id": item.get("scene_id") or row.get("scene_id"),
                 "slot_id": candidate, "baseline_slot_id": baseline,
                 "review_id": item.get("group_id"), "group_id": item.get("group_id"),
                 "expect": item.get("expect") or "candidate",
                 "variant_id": item.get("variant_id") or (next(
                     (slot.get("variant_id") for slot in pack["slots"] if slot.get("id") == candidate), ""))}
        bucket = out.setdefault(name, {"label": item.get("label"), "scenes": []})
        if row.get("status") != "reviewed":
            scene.update(verdict="missing_evidence", review_status=row.get("status"))
            bucket["scenes"].append(scene)
            continue
        decoded = _decode_tone(parsed.get(field), mapping, baseline, candidate, measure)
        neutral_candidate = reverse.get(candidate)
        observation = _observation(parsed, neutral_candidate or "")
        content = str(observation.get("content_status") or "uncertain")
        if decoded.get("verdict") in ("missing_mapping", "no_pair_entry"):
            verdict = "missing_evidence"
        else:
            observed = (bool(decoded.get("a_is_lighter")) if scene["expect"] == "baseline"
                        else bool(decoded.get("other_is_lighter")))
            verdict = ("direction_observed" if observed else
                       ("no_clear_difference" if decoded.get("tie_or_uncertain") else "opposite_or_unclear"))
            if content == "fail":
                verdict = "content_failure"
            elif content == "uncertain" and verdict == "direction_observed":
                verdict = "observed_but_content_uncertain"
        scene.update(review_status=row.get("status"), pairwise=decoded, content_status=content,
                     observation=observation, verdict=verdict)
        bucket["scenes"].append(scene)
    for name, direction in out.items():
        scenes = direction["scenes"]
        direction["valid_samples"] = sum(
            1 for scene in scenes if scene.get("verdict") in
            ("direction_observed", "observed_but_content_uncertain",
             "no_clear_difference", "opposite_or_unclear"))
        direction["usable_success"] = sum(1 for scene in scenes
                                          if scene.get("verdict") == "direction_observed")
        notes = (decoding.get("directions") or {}).get(name) or {}
        direction["summary_note"] = notes.get("summary_note")
        direction["decoded_by"] = notes.get("decoded_by")
    return out


def _decode_directions(pack, review_view, decoding):
    """按显式槽位解码；pack 声明 comparisons 时走通用比较解码，否则沿用旧轮的两场景形状。"""
    by_id = {row["review_id"]: row for row in review_view}
    if pack.get("comparisons"):
        return _qualify_directions(_decode_comparisons(pack, by_id, decoding), review_view, pack)
    out = {}
    direction_notes = decoding.get("directions") or {}

    def usable(parsed, neutral_id):
        row = _observation(parsed, neutral_id)
        return str(row.get("content_status") or "uncertain")

    names = pack.get("tone_names") or ["bright", "deep"]
    tone_map = {names[0]: ("S1-B", "S2-B", "B"), names[1]: ("S1-C", "S2-C", "C")}
    for name, (slot_s1, slot_s2, variant) in tone_map.items():
        scenes = []
        for group_id, slot_id, base_id in (("S1-tone", slot_s1, "S1-A"), ("S2-tone", slot_s2, "S2-A")):
            row = by_id.get(group_id) or {}
            mapping = row.get("neutral_to_slot") or {}
            parsed = row.get("parsed") or {}
            if row.get("status") != "reviewed":
                scenes.append({"scene_id": group_id.split("-")[0], "slot_id": slot_id,
                               "verdict": "missing_evidence", "review_status": row.get("status")})
                continue
            decoded = _decode_tone(parsed.get(pack.get("tone_pairwise_field", "tone_pairwise")), mapping, base_id, slot_id, pack.get("tone_measure", "lighter"))
            neutral_other = {slot: neutral for neutral, slot in mapping.items()}.get(slot_id)
            content = usable(parsed, neutral_other) if neutral_other else "uncertain"
            if name == names[0]:
                observed = bool(decoded.get("other_is_lighter"))
            else:
                observed = bool(decoded.get("a_is_lighter"))
            verdict = ("direction_observed" if observed else
                       ("no_clear_difference" if decoded.get("tie_or_uncertain") else "opposite_or_unclear"))
            if content == "fail":
                verdict = "content_failure"
            elif content == "uncertain" and verdict == "direction_observed":
                verdict = "observed_but_content_uncertain"
            scenes.append({"scene_id": group_id.split("-")[0], "slot_id": slot_id, "variant_id": variant,
                           "review_status": row.get("status"), "pairwise": decoded,
                           "content_status": content, "verdict": verdict,
                           "observation": _observation(parsed, neutral_other or "")})
        out[name] = {"scenes": scenes,
                     "valid_samples": sum(1 for row in scenes if row["verdict"] in
                                          ("direction_observed", "observed_but_content_uncertain",
                                           "no_clear_difference", "opposite_or_unclear")),
                     "usable_success": sum(1 for row in scenes if row["verdict"] == "direction_observed"),
                     "summary_note": (direction_notes.get(name) or {}).get("summary_note"),
                     "decoded_by": (direction_notes.get(name) or {}).get("decoded_by")}

    for name, (slot_s1, slot_s2) in (("balanced", ("S1-D", "S2-D")), ("three_level", ("S1-E", "S2-E"))):
        if slot_s1 not in slot_ids(pack):
            continue
        scenes = []
        for group_id, slot_id, variant in (("S1-hierarchy", slot_s1, "D" if name == "balanced" else "E"),
                                           ("S2-hierarchy", slot_s2, "D" if name == "balanced" else "E")):
            row = by_id.get(group_id) or {}
            mapping = row.get("neutral_to_slot") or {}
            parsed = row.get("parsed") or {}
            neutral = {slot: neutral for neutral, slot in mapping.items()}.get(slot_id)
            if row.get("status") != "reviewed":
                scenes.append({"scene_id": group_id.split("-")[0], "slot_id": slot_id, "variant_id": variant,
                               "verdict": "missing_evidence", "review_status": row.get("status")})
                continue
            observation = _observation(parsed, neutral or "")
            note = ((decoding.get("hierarchy") or {}).get(slot_id) or {})
            scenes.append({"scene_id": group_id.split("-")[0], "slot_id": slot_id, "variant_id": variant,
                           "review_status": row.get("status"),
                           "content_status": str(observation.get("content_status") or "uncertain"),
                           "observation": observation,
                           "verdict": note.get("verdict", "decoding_pending"),
                           "decode_note": note.get("note"),
                           "decode_evidence": note.get("evidence"),
                           "decode_basis": note.get("basis"),
                           "decoded_by": note.get("decoded_by")})
        out[name] = {"scenes": scenes,
                     "valid_samples": sum(1 for row in scenes if row.get("verdict") not in
                                          ("missing_evidence", "decoding_pending")),
                     "usable_success": sum(1 for row in scenes
                                           if row.get("verdict") in ("direction_observed", "layer_observed")),
                     "summary_note": (direction_notes.get(name) or {}).get("summary_note"),
                     "decoded_by": (direction_notes.get(name) or {}).get("decoded_by")}
    return _qualify_directions(out, review_view, pack)


def _load_incidents(out_dir):
    """读取 `run-notes/incidents.json`（本轮真实发生的外部/流程事件），报告照实列出。"""
    path = _resolve_out(out_dir) / "run-notes" / "incidents.json"
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("incidents") or []
    except Exception:  # noqa: BLE001
        return []


def _collect_anomalies(ledger, slots_view, review_view):
    rows = []
    for row in slots_view:
        if row["status"] != "generated":
            rows.append({"kind": "slot_not_generated", "slot_id": row["slot_id"], "status": row["status"],
                         "error_type": row.get("error_type")})
        if row.get("protocol_deviation"):
            rows.append({"kind": "protocol_deviation", "slot_id": row["slot_id"],
                         "detail": row["protocol_deviation"]})
        if row["status"] == "generated" and row.get("http_evidence_status") != "captured":
            rows.append({"kind": "http_evidence_missing", "slot_id": row["slot_id"],
                         "detail": row.get("http_evidence_status")})
        for previous in row.get("previous_attempts") or []:
            rows.append({"kind": "previous_failed_attempt", "slot_id": row["slot_id"],
                         "detail": {key: previous.get(key) for key in ("status", "attempts", "sent_at",
                                                                       "error_type", "moved_reason")}})
        if row.get("resend_authorised"):
            rows.append({"kind": "resend_authorised", "slot_id": row["slot_id"],
                         "detail": row["resend_authorised"]})
    extra = int((ledger.get("counters") or {}).get("authorized_extra_sends") or 0)
    if extra:
        rows.append({"kind": "budget_deviation", "detail":
                     "冻结包图像预算 10 次之外另有 %d 次用户明确授权的补发（S1-A 首次发送因既有后端崩溃丢图）"
                     % extra})
    for row in review_view:
        if row["status"] != "reviewed":
            rows.append({"kind": "review_not_completed", "review_id": row["review_id"],
                         "status": row["status"], "error_type": row.get("error_type")})
        if row.get("missing_slots"):
            rows.append({"kind": "review_missing_slots", "review_id": row["review_id"],
                         "detail": row["missing_slots"]})
    if ledger.get("paused"):
        rows.append({"kind": "paused", "detail": ledger["paused"]})
    return rows


def _verdict_text(name, pack=None):
    declared = ((pack or {}).get("verdict_texts") or {}).get(name)
    if declared:
        return declared
    return {"bright": "B 相对 A 是否更明亮且原色相仍可辨",
            "deep": "C 相对 A 是否更深沉且暗部仍可读",
            "balanced": "D 的第二色视觉份量是否提高且没有扩大物件/串色",
            "three_level": "E 的辅助层是否在预先声明区域可辨且主次成立",
            "clear": "B 相对 A 是否更清晰鲜明且强调色仍为红色",
            "muted": "C 相对 A 是否更含蓄灰浊且强调色仍为红色"}.get(name, name)


def _direction_summary(name, row, pack=None):
    scenes = row.get("scenes") or []
    usable = row.get("usable_success", 0)
    scene_total = len([scene for scene in scenes if scene.get("verdict") != "missing_evidence"]) or len(scenes)
    valid = int(row.get("valid_samples") or 0)
    if valid <= 0:
        head = "无有效样本（尚未完成或证据全部缺失，不能读成负面结论）"
    elif scene_total == 2:          # 旧轮固定两场景的措辞，保持不变
        head = {2: "2/2：两个单次样本方向一致，值得下一轮验证",
                1: "1/2：不稳定"}.get(usable, "0/2：不支持推广（或证据不足/内容失败）")
    elif scene_total <= 0:
        head = "无可用样本"
    elif usable == scene_total:
        head = "%d/%d：全部重复同向，值得下一轮验证" % (usable, scene_total)
    elif usable:
        head = "%d/%d：不稳定（存在反向或混杂样本）" % (usable, scene_total)
    else:
        head = "0/%d：不支持推广（或证据不足/内容失败）" % scene_total
    return "%s —— %s" % (_verdict_text(name, pack), head), scenes


def _render_markdown(results, pack, runtime):
    counts = results.get("counters") or {}
    lines = ["# App 二阶段色调与主次冻结包执行报告", "",
             "- 输出目录：`%s`" % _rel(results["out_dir"]),
             "- 冻结包 hash：`%s`" % results.get("pack_hash"),
             "- 运行参数 hash：`%s`" % results.get("runtime_hash"),
             "- 实际图像发送：**%s** / 预算 %s" % (counts.get("image_sends"), results["budget"]["image_http_budget"]),
             "- 实际视觉评审发送：**%s** / 预算 %s" % (counts.get("review_sends"), results["budget"]["review_http_budget"]),
             "- 状态：%s" % ("已暂停（等待用户决定）" if results.get("paused") else "未暂停"),
             ""]
    lines += ["- 验收口径 v2：方向观察、候选合规、对照合规分开；缺项/不确定/串色不计完整成功。", ""]
    for name, direction in results["directions"].items():
        lines += ["- %s：方向观察 %s；候选合规成功 %s；完整对照成功 %s。" %
                  (name, direction.get("observed_direction_count"), direction.get("candidate_usable_success"), direction.get("usable_success"))]
        for scene in direction.get("scenes", []):
            lines += ["  - %s 候选问题=%s；基准问题=%s；比较混杂=%s" %
                      (scene["slot_id"], scene.get("candidate_gate", {}).get("issues"), scene.get("baseline_gate", {}).get("issues"), scene.get("pairwise", {}).get("confounded"))]
    lines.append("")
    if counts.get("authorized_extra_sends"):
        lines += ["- **预算偏差（已授权）**：冻结包预算 %s 次 + 用户明确授权的补发 %s 次 = 有效预算 %s 次；"
                  "S1-A 的首次发送因既有后端在保存前崩溃而丢图，按约定记为 `unknown_after_send` 并暂停，"
                  "用户授权后才补发，历史留在 `ledger.json` 的 `previous_attempts`。"
                  % (results["budget"]["image_http_budget"], counts["authorized_extra_sends"],
                     int(results["budget"]["image_http_budget"]) + counts["authorized_extra_sends"]), ""]
    if results.get("paused"):
        lines += ["> 暂停原因：%s" % json.dumps(results["paused"], ensure_ascii=False), ""]
    lines += ["## 1. 槽位状态", "",
              "| 槽位 | 场景 | 变体 | 状态 | 发送次数 | 产图 | SHA256（前 12）| 备注 |",
              "|---|---|---|---|---|---|---|---|"]
    for row in results["slots"]:
        outputs = row.get("outputs") or []
        names = ", ".join("[%s](%s)" % (item["name"], "images/" + item["name"]) for item in outputs) or "—"
        hashes = ", ".join((item.get("sha256") or "")[:12] for item in outputs) or "—"
        note = row.get("error_type") or row.get("protocol_deviation") or ""
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (row["slot_id"], row["scene_id"], row["variant_id"], row["status"],
                        row.get("attempts"), names, hashes, note))
    lines += ["", "## 2. 四组视觉评审", "",
              "| 评审组 | 类型 | 状态 | 发送次数 | 中性映射 | 缺项 |", "|---|---|---|---|---|---|"]
    for row in results["review_groups"]:
        mapping = ", ".join("%s=%s" % (k, v) for k, v in sorted((row.get("neutral_to_slot") or {}).items()))
        lines.append("| %s | %s | %s | %s | %s | %s |"
                     % (row["review_id"], row["kind"], row["status"], row.get("attempts"),
                        mapping or "—", ", ".join(row.get("missing_slots") or []) or "—"))
    lines += ["", "## 3. 方向结果（按显式槽位解码）", ""]
    for name in results["directions"]:
        head, scenes = _direction_summary(name, results["directions"].get(name) or {}, pack)
        lines += ["### %s" % head, ""]
        note = (results["directions"].get(name) or {}).get("summary_note")
        if note:
            lines += ["> 解码注记：%s" % note, ""]
        for scene in scenes:
            lines.append("- `%s`（%s）：%s" % (scene.get("slot_id"), scene.get("scene_id"),
                                              scene.get("verdict")))
            if scene.get("pairwise"):
                lines.append("  - 两两方向观察：`%s`" % json.dumps(scene["pairwise"], ensure_ascii=False))
            observation = scene.get("observation") or {}
            for key in ("content_status", "tonal_expression", "hierarchy_observation",
                        "protected_hues_status", "unauthorized_additions", "geometry_confounds", "uncertainty"):
                if observation.get(key):
                    lines.append("  - %s：%s" % (key, json.dumps(observation[key], ensure_ascii=False)))
            if scene.get("decode_note"):
                lines.append("  - 解码注记：%s（依据：%s）"
                             % (scene["decode_note"], scene.get("decode_evidence") or scene.get("decode_basis")))
        lines.append("")
    lines += ["## 4. 异常与不确定项", ""]
    for incident in results.get("incidents") or []:
        lines.append("- 【事件 %s】%s" % (incident.get("at", "?"), incident.get("detail")))
        if incident.get("impact"):
            lines.append("  - 影响：%s" % incident["impact"])
        if incident.get("status"):
            lines.append("  - 处理：%s" % incident["status"])
    if results.get("anomalies"):
        for row in results["anomalies"]:
            lines.append("- %s" % json.dumps(row, ensure_ascii=False))
    else:
        lines.append("- 无")
    lines += ["", "## 5. 本轮未覆盖项", ""]
    lines += ["- %s" % item for item in results["uncovered_items"]]
    lines += ["", "## 6. 口径与限制", ""]
    lines += ["- %s" % item for item in results["notes"]]
    lines += [("- 评审只收到中性 ID、相同事实与绑定，不收到条件标签、候选措辞或预期方向。" if pack.get("comparisons")
               else ("- 评审只收到中性 ID、相同事实与绑定，不收到清色/浊色标签。" if pack.get("tone_names")
                     else "- 评审输入保留正文事实、共同色板与每图 binding map；E 的辅助绑定属于事先声明的授权，因此层级组只隐藏标签与期望方向。")),
              "- 全部原图按相对链接列出，未做任何本地叠加、裁剪重绘或色调处理。"]
    return "\n".join(lines) + "\n"


def _render_gallery(results, pack):
    def esc(value):
        return html.escape(str(value))

    parts = ["<!DOCTYPE html>", "<html lang='zh-CN'><head><meta charset='utf-8'>",
             "<title>App 二阶段色调与主次验证</title>",
             "<style>body{font-family:sans-serif;margin:24px;background:#fafafa;color:#222}"
             "figure{display:inline-block;margin:12px;text-align:center;vertical-align:top}"
             "img{max-width:340px;border:1px solid #ccc;background:#fff}"
             "figcaption{font-size:12px;color:#444;max-width:340px}"
             "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}"
             ".pending{color:#a00}</style></head><body>",
             "<h1>App 二阶段色调与主次冻结包验证</h1>",
             "<p>pack_hash <code>%s</code> · runtime_hash <code>%s</code> · 图像发送 %s / 评审发送 %s</p>"
             % (esc(results.get("pack_hash")), esc(results.get("runtime_hash")),
                esc((results.get("counters") or {}).get("image_sends")),
                esc((results.get("counters") or {}).get("review_sends"))),
             "<h2>全部原图（未作任何处理）</h2>"]
    for row in results["slots"]:
        outputs = row.get("outputs") or []
        if not outputs:
            parts.append("<figure><div class='pending'>无产图</div><figcaption>%s · %s · %s</figcaption></figure>"
                         % (esc(row["slot_id"]), esc(row["variant_id"]), esc(row["status"])))
            continue
        for item in outputs:
            parts.append("<figure><img src='images/%s' alt='%s'><figcaption>%s（槽位 %s / 变体 %s）<br>"
                         "sha256 %s</figcaption></figure>"
                         % (esc(item["name"]), esc(item["name"]), esc(item["name"]),
                            esc(row["slot_id"]), esc(row["variant_id"]), esc((item.get("sha256") or "")[:16])))
    parts.append("<h2>槽位状态</h2><table><tr><th>槽位</th><th>场景</th><th>变体</th><th>状态</th>"
                 "<th>尝试</th><th>错误</th></tr>")
    for row in results["slots"]:
        parts.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                     % (esc(row["slot_id"]), esc(row["scene_id"]), esc(row["variant_id"]),
                        esc(row["status"]), esc(row.get("attempts")), esc(row.get("error_type") or "")))
    parts.append("</table><h2>方向结果</h2>")
    for name in results["directions"]:
        head, scenes = _direction_summary(name, results["directions"].get(name) or {}, pack)
        parts.append("<h3>%s</h3><ul>" % esc(head))
        for scene in scenes:
            parts.append("<li><code>%s</code> %s：%s</li>"
                         % (esc(scene.get("slot_id")), esc(scene.get("scene_id")), esc(scene.get("verdict"))))
        parts.append("</ul>")
    parts.append("<h2>评审组与中性映射</h2><table><tr><th>组</th><th>类型</th><th>状态</th>"
                 "<th>中性 ID → 槽位</th><th>缺项</th></tr>")
    for row in results["review_groups"]:
        mapping = ", ".join("%s=%s" % (k, v) for k, v in sorted((row.get("neutral_to_slot") or {}).items()))
        parts.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                     % (esc(row["review_id"]), esc(row["kind"]), esc(row["status"]), esc(mapping or "—"),
                        esc(", ".join(row.get("missing_slots") or []) or "—")))
    parts.append("</table><h2>未覆盖项</h2><ul>")
    parts += ["<li>%s</li>" % esc(item) for item in results["uncovered_items"]]
    parts.append("</ul></body></html>")
    return "\n".join(parts) + "\n"


def render_report_safely(out_dir, pack, runtime, *, log=print):
    """渲染失败只落 `RUN-REPORT-render-failure.txt`，不丢证据、不重新联网。"""
    out = _resolve_out(out_dir)
    try:
        return {"ok": True, "results": render_report(out_dir, pack, runtime, log=log)}
    except Exception as exc:  # noqa: BLE001
        (out / "RUN-REPORT-render-failure.txt").write_text(
            "%s\n%s: %s\n" % (_now(), type(exc).__name__, exc), encoding="utf-8")
        log("[direction] 报告渲染失败（已写入 RUN-REPORT-render-failure.txt）：%s" % exc)
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}


# --- 7. 状态与恢复 ----------------------------------------------------------

def _assert_ledger_belongs_to(ledger, pack):
    """恢复动作一律先确认 ledger 属于当前冻结包，避免在别的轮次目录里改状态。"""
    recorded = ledger.get("pack_hash")
    current = (pack or {}).get("pack_hash")
    if recorded and current and recorded != current:
        raise ValueError("ledger 属于另一份冻结包（%s != %s），拒绝改动"
                         % (str(recorded)[:12], str(current)[:12]))

def authorize_resend(out_dir, slot_id, *, note, log=print, pack=None):
    """人工显式授权后把一个已发送的槽位放回可发送（**绝不自动发生**）。

    - 只接受 `sending` / `unknown_after_send` / `failed` 的槽位；
    - 旧记录整体移进 `previous_attempts`（状态、尝试数、发送时间、错误、证据都留着）；
    - 尝试计数保留，于是下一次发送是 attempt 2，不会被伪装成首次；
    - 同时把这次授权计进 `counters.authorized_extra_sends`：预算按
      「冻结包预算 + 已授权额外次数」生效，超出部分在报告里单独列出。
    """
    out = _resolve_out(out_dir)
    pack = pack or load_pack()
    ledger = load_ledger(out, pack)
    _assert_ledger_belongs_to(ledger, pack)
    entry = (ledger.get("slots") or {}).get(slot_id)
    if not entry:
        raise ValueError("槽位不存在于 ledger：%s" % slot_id)
    if entry.get("status") not in ("sending", "unknown_after_send", "failed"):
        raise ValueError("只允许授权重发已发送的槽位；%s 当前是 %s" % (slot_id, entry.get("status")))
    if not str(note or "").strip():
        raise ValueError("授权重发必须写明授权备注（谁、为什么）")
    ledger.setdefault("previous_attempts", []).append({**entry, "moved_at": _now(),
                                                       "moved_reason": "human_authorised_resend",
                                                       "moved_note": note})
    entry.update(status="planned", error_type=None, error=None, outputs=[], server_meta=None,
                 resend_authorised={"at": _now(), "by": "human_authorised_resend", "note": note})
    ledger["counters"]["authorized_extra_sends"] = int(
        ledger["counters"].get("authorized_extra_sends") or 0) + 1
    if (ledger.get("paused") or {}).get("slot_id") == slot_id:
        ledger["paused"] = None
    _event(ledger, "slot_resend_authorised", slot_id=slot_id, note=note,
           attempts_so_far=entry.get("attempts"))
    save_ledger(out, ledger)
    log("[authorize-resend] %s 放回可发送（attempt 计数保留为 %s；已授权额外发送 +1，"
        "有效图像预算 = %s）" % (slot_id, entry.get("attempts"),
                              int(pack.get("image_http_budget") or 10)
                              + ledger["counters"]["authorized_extra_sends"]))
    return entry


def settle_slot(out_dir, slot_id, *, state, error_type="", error="", evidence_file="", note="", log=print,
                pack=None):
    """人工授权后结算一个停在 sending 的槽位（绝不自动转回可发送）。

    只允许把 `sending` 结算成 `unknown_after_send` / `failed`：已发送状态不能自动转回
    `planned`，恢复时以 ledger 为准；此函数是唯一的（人工触发）出口。
    """
    out = _resolve_out(out_dir)
    pack = pack or load_pack()
    if state not in ("unknown_after_send", "failed"):
        raise ValueError("settle 只允许结算成 unknown_after_send 或 failed")
    ledger = load_ledger(out, pack)
    _assert_ledger_belongs_to(ledger, pack)
    entry = (ledger.get("slots") or {}).get(slot_id)
    if not entry:
        raise ValueError("槽位不存在于 ledger：%s" % slot_id)
    if entry.get("status") not in ("sending", "unknown_after_send", "failed"):
        raise ValueError("只结算已发送状态的槽位；%s 当前是 %s" % (slot_id, entry.get("status")))
    if entry.get("status") == state and not note:
        log("[settle] %s 已经是 %s，无需改动" % (slot_id, state))
        return entry
    evidence = None
    if evidence_file:
        source = Path(evidence_file)
        if not source.is_file():
            raise ValueError("找不到证据文件：%s" % evidence_file)
        target_dir = out / "run-notes" / "send-errors"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / ("%s-evidence%s" % (slot_id, source.suffix or ".txt"))
        target.write_text(source.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        evidence = _rel(target)
    entry.update(status=state, error_type=error_type or entry.get("error_type"),
                 error=error or entry.get("error"),
                 settled={"at": _now(), "by": "human_authorised_settle", "state": state,
                          "note": note, "evidence": evidence})
    _event(ledger, "slot_settled", slot_id=slot_id, state=state, note=note, evidence=evidence)
    if state == "unknown_after_send":
        ledger["paused"] = {"at": _now(), "slot_id": slot_id, "status": state,
                            "reason": note or "已发送但结果未知：暂停后续联网，等待用户决定"}
    elif (ledger.get("paused") or {}).get("slot_id") == slot_id:
        ledger["paused"] = None
    save_ledger(out, ledger)
    log("[settle] %s → %s（证据 %s）" % (slot_id, state, evidence or "—"))
    return entry


def status(out_dir, pack=None):
    out = _resolve_out(out_dir)
    pack = pack or load_pack()
    ledger = load_ledger(out, pack)
    slots = []
    for slot in pack["slots"]:
        entry = (ledger.get("slots") or {}).get(slot["id"]) or {}
        outputs = entry.get("outputs") or []
        slots.append({"slot_id": slot["id"], "status": entry.get("status", "planned"),
                      "attempts": entry.get("attempts", 0),
                      "reusable": entry.get("status") == "generated" and slot_outputs_intact(out, entry),
                      "outputs": [row["name"] for row in outputs]})
    reviews = []
    for group in pack["review_groups"]:
        entry = (ledger.get("reviews") or {}).get(group["id"]) or {}
        reviews.append({"review_id": group["id"], "status": entry.get("status", "planned"),
                        "attempts": entry.get("attempts", 0)})
    return {"label": (pack or {}).get("id", LABEL), "out_dir": str(out), "pack_hash": pack.get("pack_hash"),
            "ledger_pack_hash": ledger.get("pack_hash"),
            "counters": ledger.get("counters"), "paused": ledger.get("paused"),
            "slots": slots, "reviews": reviews,
            "remaining_image_budget": int(pack.get("image_http_budget") or 10)
                                      - int((ledger.get("counters") or {}).get("image_sends") or 0),
            "remaining_review_budget": int(pack.get("review_http_budget") or 4)
                                       - int((ledger.get("counters") or {}).get("review_sends") or 0)}
