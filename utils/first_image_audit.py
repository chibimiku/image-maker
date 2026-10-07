# -*- coding: utf-8 -*-
"""Gemini 直出首图的「审计 + 重试」判定层（不负责发请求）。

背景（2026-10-06，任务 `b51aa600`）：参考优先 + 短内容锚时，Gemini 会把画风参考图**当成待复刻的
内容**（该次复刻出参考图的角色/服装/道具/月夜城堡）。护栏（`gemini-analysis-content-lock.md`、
`REF_PRIORITY_POSTAMBLE`）实测拦不住，所以补一层可判定的审计：

- 审计由外部回调执行（`utils.first_image_audit.audit_first_image`，走视觉文本通道）；
- **本模块只做判定**：这一张要不要重画、重画时正文怎么改、几次之后用哪一张；
- 重试上限硬编码在调用方（默认 2 次），审计异常按「结论无效」处理，**不会**因为没有结论就重试到爆。

判定原则（与 `utils/identity_audit.normalize_audit` 同源）：缺字段/格式非法 ≠ 通过，一律记
`conclusion_valid=False`，由 `should_retry` 决定是否补一次；`uncertain` 不重试（抽卡不解决不确定）。
"""
from __future__ import annotations

import json
import re
import time

VERDICTS = {"pass", "fail", "uncertain"}
SEVERITIES = {"minor", "major"}

# 默认语义：两个 0.65 阈值与身份审计一致；置信度不足的 major 不触发重画
FAIL_CONFIDENCE = 0.65

# 未配置 `reference_content_exclusions` 时的兜底禁令：这些是画风样本里最容易被搬走的"内容类"元素。
# 只在确实挂了参考图时使用；具名清单（画风条目字段）优先。
DEFAULT_FORBIDDEN_ITEMS = (
    "the sample's character, face and hairstyle",
    "the sample's hair colour or eye colour",
    "the sample's costume design, accessories and colour scheme",
    "the sample's pose and camera framing",
    "the sample's setting, props, animals and background architecture",
    "the sample's decorative border, frame, panel layout or watermark",
)


def extract_json_object(text) -> dict:
    """从模型回复里取 JSON 对象；取不到返回空 dict（由调用方按无效结论处理）。"""
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
    for candidate in (raw,):
        try:
            value = json.loads(candidate)
        except Exception:  # noqa: BLE001
            break
        if isinstance(value, dict):
            return value
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except Exception:  # noqa: BLE001
        return {}
    return value if isinstance(value, dict) else {}


def _issues(value) -> list:
    rows = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        element = re.sub(r"[^a-z0-9_]+", "_", str(item.get("element") or "").lower()).strip("_")
        observed = str(item.get("observed") or "").strip()
        severity = str(item.get("severity") or "").strip().lower()
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if not element or not observed or severity not in SEVERITIES:
            continue
        rows.append({"element": element,
                     "expected": str(item.get("expected") or "").strip(),
                     "observed": observed,
                     "severity": severity,
                     "confidence": max(0.0, min(1.0, confidence))})
    return rows


def normalize_audit(value) -> dict:
    """把审计模型回复规范化成可判定结论；缺字段时 `conclusion_valid=False`。"""
    data = value if isinstance(value, dict) else {}
    content_issues = _issues(data.get("content_issues"))
    style_issues = _issues(data.get("style_issues"))
    missing = []
    content_ok = data.get("content_ok")
    style_ok = data.get("style_ok")
    if not isinstance(content_ok, bool):
        missing.append("content_ok")
    if not isinstance(style_ok, bool):
        missing.append("style_ok")
    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        missing.append("verdict")
    copied = data.get("copied_from_reference")
    if not isinstance(copied, bool):
        missing.append("copied_from_reference")
    major = [row for row in (content_issues + style_issues)
             if row["severity"] == "major" and row["confidence"] >= FAIL_CONFIDENCE]
    try:
        confidence = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    result = {
        "content_ok": content_ok if isinstance(content_ok, bool) else None,
        "style_ok": style_ok if isinstance(style_ok, bool) else None,
        "copied_from_reference": copied if isinstance(copied, bool) else None,
        "verdict": verdict if verdict in VERDICTS else "unknown",
        "content_issues": content_issues,
        "style_issues": style_issues,
        "copied_elements": [str(v).strip() for v in (data.get("copied_elements") or []) if str(v).strip()],
        "uncertain": [str(v).strip() for v in (data.get("uncertain") or []) if str(v).strip()],
        "confidence": max(0.0, min(1.0, confidence)),
        "summary": str(data.get("summary") or "").strip(),
        "major_issues": major,
        "missing_conclusion_fields": missing,
        "conclusion_valid": not missing,
    }
    # 结论自相矛盾时（声称 pass 却报了 major）以 major 为准，绝不放过
    if result["conclusion_valid"] and result["verdict"] == "pass" and major:
        result["verdict"] = "fail"
        result["summary"] = (result["summary"] + " [major 问题与 pass 冲突，按 fail 处理]").strip()
    # 结论不完整时**不能**保留一个 pass：缺字段的回复里写着 pass 只是模型漏答，不是通过
    if not result["conclusion_valid"]:
        result["verdict"] = "unknown"
    return result


def _to_positive_float(value):
    """把可选的秒数/预算转成正浮点；给不出数就当"没预算信息"（None）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _audit_budget(per_audit_seconds, cap: float) -> float:
    """估算"再跑一轮"要用的秒数：一次审计（默认 180 秒上限）的 cap 倍。"""
    base = per_audit_seconds if per_audit_seconds else 180.0
    return float(base) * float(cap)


def audit_call_failed(audit: dict) -> bool:
    """审计**调用**失败（超时 / 400 / 解析炸了），不是「模型回了个缺字段的答案」。

    这条区分很关键（2026-10-07 实盘）：把 ReadTimeout / HTTPError 当成"要重画"，
    会白白多出一张收费图、大概率还是同样超时，最后按「重画额度用尽、全部不合格」把
    队列标红 —— 而图其实已经出好、还能用。调用失败就该**跳过重画、把已有产物交出去**。

    注意别把"模型漏答字段"（`missing_conclusion_fields`）也算进来：那种确实是模型没给结论，
    补画一次是原本的设计，`tests/test_first_image_audit.py` 有断言守着。
    """
    data = audit if isinstance(audit, dict) else {}
    return bool(str(data.get("audit_error") or "").strip())


def audit_cancelled(audit: dict) -> bool:
    """审计期间任务被取消/超时（图还在，只是没审完）→ 不重画，直接交付这张。"""
    data = audit if isinstance(audit, dict) else {}
    return bool(str(data.get("audit_cancelled") or "").strip())


def audit_blocked_by_reason(audit: dict) -> bool:
    """审计给出了**明确**的打回理由（复制参考图内容 / 高置信 major 不符），不是"没结论"。"""
    data = audit if isinstance(audit, dict) else {}
    if not data.get("conclusion_valid"):
        return False
    if audit_cancelled(data):
        return False
    copied = data.get("copied_from_reference")
    if copied is True:
        return True
    return bool(data.get("major_issues"))


def should_retry(audit: dict) -> tuple[bool, str]:
    """这一张要不要重画，以及人话原因（写日志与重试指令）。"""
    data = audit if isinstance(audit, dict) else {}
    if not data.get("conclusion_valid"):
        # 调用失败（超时/400）与"模型漏答"是两件事：前者重画也救不回来，且要多花一张图的钱
        if audit_cancelled(data):
            return False, "审计被取消/超时（不是图的问题），跳过重画、直接交付这张"
        if audit_call_failed(data):
            return False, "审计调用失败（不是图的问题），跳过重画、直接交付这张"
        return True, "审计未给出有效结论（缺字段或格式非法），补画一次"
    copied = data.get("copied_from_reference")
    major = list(data.get("major_issues") or [])
    if copied is True:
        return True, "复制了画风样本的内容元素"
    if data.get("content_ok") is False and major:
        return True, "内容与本次要求明显不符"
    if data.get("style_ok") is False and major:
        return True, "没有跟上画风样本的画法"
    if str(data.get("verdict")) == "fail":
        return True, "审计判为不合格"
    return False, "审计通过（或仅有不确定项，不重画）"


def audit_reasons(audit: dict) -> list:
    """把审计结果压成给模型的短句列表（重画指令里用）。"""
    data = audit if isinstance(audit, dict) else {}
    reasons = []
    if not data.get("conclusion_valid"):
        reasons.append("The previous audit could not confirm the requested content.")
    for row in (data.get("major_issues") or []) + list(data.get("content_issues") or []) \
            + list(data.get("style_issues") or []):
        expected = row.get("expected") or "as specified"
        reasons.append(f"{row['element']}: expected {expected}; the picture showed {row['observed']}")
        if len(reasons) >= 8:
            break
    for element in data.get("copied_elements") or []:
        reasons.append("imported from the style sample: " + str(element))
    if not reasons:
        reasons.append("The previous attempt did not follow the content specification.")
    # 去重但保序
    seen, unique = set(), []
    for item in reasons:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique[:10]


def build_retry_prompt(audit: dict, content: str, forbidden=None, attempt: int = 2,
                       max_attempts: int = 2) -> str:
    """重画指令：把「内容权威 + 参考图禁区 + 这次为什么被打回」重申在请求尾部。"""
    from utils.prompt_loader import render_prompt_file
    items = [str(v).strip() for v in (forbidden or []) if str(v).strip()] or list(DEFAULT_FORBIDDEN_ITEMS)
    return render_prompt_file("first-image-audit-retry.md", {
        "attempt": max(1, int(attempt)),
        "max_attempts": max(1, int(max_attempts)),
        "reasons": "- " + "\n- ".join(audit_reasons(audit)),
        "content": str(content or "").strip() or "(see the content block above)",
        "forbidden": "- " + "\n- ".join(items),
    }).strip()


def attempt_score(record: dict) -> tuple:
    """候选排序键：先内容、再风格、再复制惩罚、最后尝试轮次。

    全部用「越大越好」的形式；**审计失败排在最后**（宁可用审计未跑过的图，也不用明确不合格的图）。
    """
    data = record if isinstance(record, dict) else {}
    audit = data.get("audit")
    if not isinstance(audit, dict) or not audit.get("conclusion_valid"):
        rank = 0                      # 没结论：只比审计过的差、比 fail 好
    elif should_retry(audit)[0]:
        rank = -1                     # 明确不合格
    else:
        rank = 1
    content_ok = 1 if audit.get("content_ok") is True else 0
    style_ok = 1 if audit.get("style_ok") is True else 0
    copied = 1 if audit.get("copied_from_reference") is True else 0
    return (rank, content_ok, style_ok, -copied, int(data.get("round") or 0))


def choose_best_candidate(candidates) -> dict:
    """最后一轮仍然不合格时用哪一张：内容 > 风格 > 未复制 > 最新尝试。"""
    rows = [row for row in (candidates or []) if isinstance(row, dict) and row.get("file")]
    if not rows:
        return {}
    return max(rows, key=attempt_score)


def promote_candidate(path: str, publish_dir: str, filename: str, log_callback=None) -> str:
    """把选中的候选复制成发布文件名（原候选保留在候选目录里，绝不删除、绝不覆盖）。

    这里保证两件事：

    1. **后缀跟真实字节走**：发布名过去被写死 `.png`（`single_analyzer.first_image_publish_target`），
       而 aigc2d 的 Gemini 网关会自己决定回 PNG 还是 JPEG，于是 `.png` 里装的是 JPEG
       —— 同目录两张文件字节完全相同、大小分毫不差（用户 2026-10-07 反馈「看着大小一样」，
       那两张本来就是同一个字节流）。现在按 magic bytes 纠正后缀。
    2. **重名不覆盖**：目标已存在且**内容不同**就另存 `<name>-1.ext`、`-2.ext`…，两份都留下，
       再交给 `utils.image_dedup` 按 SHA-256 对账；内容相同则直接认已存在的那份，不重复写入。

    复制失败时**回退候选本身**（绝不返回空串）：空串会被上层当成"本次没有产物"，
    于是图明明出好了、队列却被标红失败（用户 2026-10-07 反馈「有结果的也变红」）。
    """
    import os
    import shutil

    from utils.image_identity import file_sha256, name_with_extension_of, unique_path
    if not path or not os.path.isfile(path):
        if log_callback:
            log_callback(f"⚠️ 选中的候选文件不存在，跳过发布副本：{path}")
        return ""
    os.makedirs(publish_dir, exist_ok=True)
    target = os.path.join(publish_dir, str(filename or os.path.basename(path)))

    # ① 后缀别撒谎：副本与候选是同一个字节流，格式以**候选**为准（目标此刻还不存在）
    corrected = name_with_extension_of(target, path)
    if corrected != target:
        if log_callback:
            log_callback("📎 发布名后缀与真实字节不符，按 magic bytes 纠正："
                         f"{os.path.basename(target)} → {os.path.basename(corrected)}")
        target = corrected

    if os.path.abspath(target) == os.path.abspath(path):
        return target

    # ② 重名容错：先都保存，用 SHA-256 判断是不是同一份
    if os.path.isfile(target):
        source_hash = file_sha256(path)
        target_hash = file_sha256(target)
        if source_hash and source_hash == target_hash:
            if log_callback:
                log_callback("🧬 发布文件已存在且内容相同（SHA-256 一致），复用不重复写入："
                             f"{os.path.basename(target)}")
            return target
        collision = unique_path(target)
        if log_callback:
            log_callback("⚠️ 发布名已被另一份内容占用（SHA-256 不同），两份都保留，"
                         f"本次另存为：{os.path.basename(collision)}")
        target = collision

    try:
        shutil.copy2(path, target)          # 保留候选原件（审计台账/复核都要指着它）
    except OSError as exc:
        if log_callback:
            log_callback(f"⚠️ 发布副本写入失败（{exc}），改用候选原文件：{path}")
        return path
    return target


def audit_record_path(directory: str, prefix: str) -> str:
    import os
    return os.path.join(directory, f"{prefix}-first-image-audit.json")


AUDIT_SYSTEM_FILE = "first-image-audit-system.md"
DEFAULT_MAX_ATTEMPTS = 2
CONFIG_ENABLE_KEY = "gemini_audit_retry_enabled"
CONFIG_ATTEMPTS_KEY = "gemini_audit_retry_max_attempts"
CONFIG_TIMEOUT_KEY = "gemini_audit_timeout_seconds"
DEFAULT_AUDIT_TIMEOUT = 180


def audit_settings(config: dict = None) -> dict:
    """读 `conf/config.json` 里的「审计 + 重试」设置（缺失时用安全默认）。

    - `gemini_audit_retry_enabled`：默认 True（用户 2026-10-06 明确要求默认开启）；
    - `gemini_audit_retry_max_attempts`：默认 2（首图 + 1 次重画），硬上限 3；
    - `gemini_audit_timeout_seconds`：单次审计请求秒数上限，缺省 180、夹在 [60, 600]。
      实盘 `ReadTimeout: read timeout=180` 说明一次性视觉审计（生成图+画风样本+源图）
      确实会顶到上限，慢机器可以调大。
    """
    import os
    if config is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "conf", "config.json")
        try:
            with open(path, encoding="utf-8") as handle:
                config = json.load(handle) or {}
        except (OSError, json.JSONDecodeError):
            config = {}
    data = config if isinstance(config, dict) else {}
    enabled = data.get(CONFIG_ENABLE_KEY, True) is not False
    try:
        attempts = int(data.get(CONFIG_ATTEMPTS_KEY, DEFAULT_MAX_ATTEMPTS))
    except (TypeError, ValueError):
        attempts = DEFAULT_MAX_ATTEMPTS
    try:
        timeout = int(data.get(CONFIG_TIMEOUT_KEY, DEFAULT_AUDIT_TIMEOUT))
    except (TypeError, ValueError):
        timeout = DEFAULT_AUDIT_TIMEOUT
    return {"enabled": bool(enabled), "max_attempts": max(1, min(attempts, 3)),
            "timeout_seconds": max(60, min(timeout, 600))}


def resolve_audit_content(analysis_result: dict, tier: str = "full", max_chars: int = 1800) -> str:
    """审计用的内容规格：优先完整内容锚，其次短锚/描述；不做生成改写。"""
    data = analysis_result if isinstance(analysis_result, dict) else {}
    order = ("gpt_image_prompt", "gpt_image_prompt_short", "short_description", "english_description")
    if str(tier) == "short":
        order = ("gpt_image_prompt_short", "short_description", "gpt_image_prompt", "english_description")
    for key in order:
        text = str(data.get(key) or "").strip()
        if text:
            return text[:max_chars]
    return ""


def audit_first_image(image_path: str, analysis_result: dict, text_cfg: dict = None, *,
                      content: str = "", style_ref_path: str = "",
                      source_image_path: str = "", forbidden=None, timeout: int = 180,
                      max_tokens: int = 1600) -> dict:
    """对一张 Gemini 直出图做「内容保真 / 参考图内容复制 / 画风覆盖」审计。

    走视觉文本通道（`load_text_api_config`，与身份审计同一家），附图顺序固定：
    **生成图 → 画风样本（若有）→ 源图（若有）**，与 user 提示里的说明一致。
    审计本身失败时抛异常，由 `generate_with_audit` 兜成「结论无效」（不会静默当通过）。
    """
    import os
    from utils.analysis_gpt_prompt import call_text_model, load_text_api_config
    from utils.prompt_loader import read_prompt_file
    from utils.send_budget import guarded_text_call

    if not os.path.isfile(str(image_path or "")):
        raise FileNotFoundError(image_path)
    expected = str(content or "").strip() or resolve_audit_content(analysis_result)
    if not expected:
        raise ValueError("审计缺少内容规格（分析产物没有可用描述）")
    cfg = text_cfg or load_text_api_config()
    images = [str(image_path)]
    roles = ["1. GENERATED IMAGE — the picture being audited."]
    if style_ref_path and os.path.isfile(str(style_ref_path)):
        images.append(str(style_ref_path))
        roles.append("2. STYLE SAMPLE — the rendering reference that was attached; it defines technique only.")
    source = str(source_image_path or "").strip() or str(
        (analysis_result or {}).get("source_image_path") or "").strip()
    if source and os.path.isfile(source):
        images.append(source)
        roles.append("3. SOURCE IMAGE — the original that the content specification describes.")
    forbidden_items = [str(v).strip() for v in (forbidden or []) if str(v).strip()]
    user = "\n".join([
        "IMAGE ROLES:",
        "\n".join(roles),
        "",
        "CONTENT SPECIFICATION (what the generated image was asked to render):",
        expected,
        "",
        "REFERENCE CONTENT THAT MUST NOT APPEAR IN THE OUTPUT:",
        ("- " + "\n- ".join(forbidden_items)) if forbidden_items
        else "- " + "\n- ".join(DEFAULT_FORBIDDEN_ITEMS),
        "",
        'Audit the generated image and return exactly one JSON object.',
    ])
    system = read_prompt_file(AUDIT_SYSTEM_FILE)
    raw = guarded_text_call("first-image-audit", call_text_model, cfg["base_url"], cfg["api_key"],
                            cfg["model"], system, user, timeout=timeout, max_tokens=max_tokens,
                            image_paths=images)
    report = normalize_audit(extract_json_object(raw))
    report.update({"image": str(image_path), "model": str(cfg.get("model") or ""), "raw": raw})
    return report


def write_audit_record(directory: str, prefix: str, payload: dict) -> str:
    """把每个候选的审计与选择理由落盘（可复核，不含任何凭据）。"""
    import os
    path = audit_record_path(directory, prefix)
    os.makedirs(directory, exist_ok=True)
    existing = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as handle:
                loaded = json.load(handle)
            existing = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            existing = {}
    existing.update(payload if isinstance(payload, dict) else {})
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(existing, handle, ensure_ascii=False, indent=2)
    return path


def generate_with_audit(generate, *, prompt: str, max_attempts: int = 2, audit=None,
                        content: str = "", forbidden=None, candidate_dir: str = "",
                        publish_dir: str = "", publish_name: str = "",
                        record_prefix: str = "first", log_callback=None,
                        cancel_check=None, retry_budget=None, audit_timeout=None) -> dict:
    """出图 → 审计 → 不满意就重画（有上限），最后落一张发布图。

    - `generate(prompt, round_index) -> list[str]`：出一次图并返回落盘文件；
    - `audit(candidate_path, prompt, round_index) -> dict`：审计回调（`None` 时只出图不审计）；
    - `max_attempts`：本次任务最多出几张（默认 2 = 首图 + 1 次重画），**这是唯一的花费闸门**；
    - `retry_budget`：本次任务还剩多少秒（调用方的生图倒计时）；不够再跑一次"审计 + 重画"时
      提前收工、用眼前这张，不让倒计时在半路把结果掐掉（用户 2026-10-07 反馈「有结果的也变红」）；
    - `audit_timeout`：单次审计请求的秒数上限（与 `audit_first_image(timeout=…)` 同一口径，用于估算）；
    - `stopped_reason`：`passed` / `exhausted` / `cancelled` / `no_output` / `audit_disabled`
      / `audit_unavailable` / `audit_failed` / `audit_cancelled` / `budget_tight`；
    - `suspended=True`：所有候选都被审计**明确**判为不合格 → 调用方必须标记人工复核。
      注意：审计自己失败（超时/400）**不算**"图不合格"，那种情况只写警告、不封锁产物。
    - `published`：**优先**是发布名副本；发布副本写不出来时回退选中的候选文件本身
      （绝不返回空串 —— 那会被上层当成"没有产物"而误标失败）。
    - 返回 dict：`files` / `best` / `published` / `candidates` / `audits` / `stopped_reason` / `suspended`。
    """
    log = log_callback or (lambda message: None)
    limit = max(1, int(max_attempts or 1))
    candidates: list = []
    audits: list = []
    current_prompt = str(prompt or "")
    target_dir = str(candidate_dir or "").strip()
    stopped_reason = ""
    remaining_budget = _to_positive_float(retry_budget)
    per_audit_seconds = _to_positive_float(audit_timeout)

    for round_index in range(1, limit + 1):
        if callable(cancel_check) and cancel_check():
            stopped_reason = "cancelled"
            break
        # 预算还能不能再跑一轮「审计（可能 180 秒超时）+ 重画」？不够就用眼前这张，
        # 别让倒计时在审计中途把整条任务掐掉 —— 那时图已经画好、只是没交付。
        if (round_index > 1 and remaining_budget is not None
                and remaining_budget < _audit_budget(per_audit_seconds, cap=0.75)):
            stopped_reason = "budget_tight"
            log(f"⏳ 剩余时间只够交付、不够再跑一轮审计+重画（约需 "
                f"{int(_audit_budget(per_audit_seconds, cap=0.75))} 秒）：改用眼前这张，按「审计未完成」交付。")
            break
        round_started_at = time.monotonic()
        log(f"🎨 第 {round_index}/{limit} 次出图…" if round_index > 1 else "🎨 正在出图…")
        files = list(generate(current_prompt, round_index) or [])
        files = [path for path in files if path]
        candidates.append({"round": round_index, "file": files[0] if files else "",
                           "files": files, "prompt": current_prompt, "audit": {}})
        if not files:
            stopped_reason = "no_output"
            log(f"⚠️ 第 {round_index} 次出图没有产物。")
            continue
        if audit is None:
            stopped_reason = "audit_disabled"
            break
        if not callable(audit):
            stopped_reason = "audit_unavailable"
            break
        log(f"🔍 第 {round_index} 次出图审计中（内容保真 / 参考图内容复制 / 画风覆盖）…")
        try:
            raw = audit(files[0], current_prompt, round_index)
        except Exception as exc:  # noqa: BLE001 - 审计失败不能让整条链断掉
            raw = {"_error": f"{type(exc).__name__}: {exc}"}
        cancelled_during_audit = bool(callable(cancel_check) and cancel_check())
        if remaining_budget is not None:
            remaining_budget -= (time.monotonic() - round_started_at)
        verdict = normalize_audit(raw)
        if raw.get("_error"):
            verdict["audit_error"] = raw["_error"]
        if cancelled_during_audit:
            # 倒计时在审计期间到了：图还在，判「结果未审完」而不是「任务失败」
            verdict["audit_cancelled"] = "任务在审计期间被取消/超时；产物保留，按未审完交付"
            stopped_reason = "audit_cancelled"
        candidates[-1]["audit"] = verdict
        audits.append({"round": round_index, "audit": verdict, "image": files[0]})
        if cancelled_during_audit:
            log("⏹ 审计被取消（多为生图倒计时到点）：图已经出好，保留并按「未审完」交付，"
                "不再重画、也不算生图失败。")
            break
        retry, reason = should_retry(verdict)
        if audit_call_failed(verdict):
            # 审计自己没跑成（超时 / 400）：跳过重画，把这张交出去；只在记录里留警告
            stopped_reason = "audit_failed"
            log(f"⚠️ 审计调用失败，跳过重画并保留这张图：{verdict.get('audit_error') or reason}")
            break
        if not retry:
            log(f"✅ 第 {round_index} 次出图通过审计：{verdict.get('summary') or reason}")
            stopped_reason = "passed"
            break
        log(f"⚠️ 第 {round_index} 次出图未通过：{reason}（{verdict.get('summary') or '无摘要'}）")
        if round_index >= limit:
            stopped_reason = "exhausted"
            log(f"🛑 重画额度已用尽（{limit} 次），在候选中选最可用的一张，并按需人工复核。")
            break
        if callable(cancel_check) and cancel_check():
            stopped_reason = "cancelled"
            break
        current_prompt = build_retry_prompt(verdict, content, forbidden,
                                            attempt=round_index + 1, max_attempts=limit)

    best = choose_best_candidate(candidates) or {}
    best_file = str(best.get("file") or "")
    published = ""
    if best_file:
        name = ""
        if callable(publish_name):
            name = str(publish_name(int(best.get("round") or 1)) or "")
        else:
            name = str(publish_name or "")
        if not name or not publish_dir:
            published = best_file
        else:
            published = promote_candidate(best_file, publish_dir, name, log_callback=log)
        if target_dir:
            from utils.output_isolation import resolve_output_target
            target_dir, record_prefix = resolve_output_target(
                target_dir, f"{str(record_prefix or 'first')}-audit")
            write_audit_record(target_dir, record_prefix, {
                "max_attempts": limit,
                "stopped_reason": stopped_reason,
                "selected_file": published or best_file,
                "selected_round": best.get("round"),
                "candidates": [{"round": row.get("round"), "file": row.get("file"),
                                "prompt": row.get("prompt"), "audit": row.get("audit")}
                               for row in candidates],
            })
            log(f"📝 首图审计记录：{audit_record_path(target_dir, record_prefix)}")
    blocked = bool(suspended(candidates))
    return {"files": [row.get("file") for row in candidates if row.get("file")],
            "best": best, "published": published or best_file,
            "candidates": candidates, "audits": audits,
            "stopped_reason": stopped_reason, "suspended": blocked}

def suspended(candidates) -> bool:
    """整组候选里没有任何一张可用（全部被审计**明确**打回）→ 这条任务需要人工复核。

    审计调用失败（超时/400）的候选不算"明确不合格"：那是审计没跑成，不是图有问题
    （用户 2026-10-07 反馈「有结果的也变红」—— 图已经出好，不该因为对面超时被封锁）。
    """
    rows = [row for row in (candidates or []) if isinstance(row, dict)]
    if not rows:
        return False
    for row in rows:
        audit = row.get("audit")
        if not isinstance(audit, dict) or not audit:
            return False          # 有没审过的候选（例如审计本身关闭/异常），不算"全部不合格"
        if not audit_blocked_by_reason(audit):
            return False
    return True
