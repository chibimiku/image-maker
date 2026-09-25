# -*- coding: utf-8 -*-
"""「分析图片并生图」在 gpt-image 通道上的请求组装（与单图分析 Tab 共用，便于脱离 Qt 测试）。

和 Gemini 通道的关系：
- GUI 首图优先用分析产物的 `gpt_image_prompt`（约 1400 字符的身份完整锚）。2026-09-25 实测：
  4400+ 字符全文会压弱画风参考图，500 字符短锚又可能漏发色/瞳色；完整锚兼顾身份和画风；
- **画风 `prompt_gpt` 拼在最前面**，后面接内容描述；
- 挂画风参考图时追加「只借渲染语法、不得搬参考图角色/发色/瞳色/服装/姿势/构图」的排除句；
- **不挂分析素材图**（素材图只用来做分析、产出文字）。
"""
import os

from utils.analysis_gpt_prompt import (FIELD_KEY, FIELD_MAX_CHARS, SHORT_FIELD_KEY,
                                       SHORT_FIELD_MAX_CHARS, STYLE_REF_EXCLUSION)


def save_generation_manifest(first_image, request, *, model, size, quality, mode, steps,
                             firmware, outputs=None):
    """为 GUI 产物留可复核的提示词/工序快照；不序列化 API 配置或密钥。"""
    import hashlib
    import json
    from utils.post_process import resolve_firmware_text
    references = []
    for path in request.get("image_paths") or []:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                references.append({"path": os.path.abspath(path), "sha256": hashlib.sha256(f.read()).hexdigest()})
    snapshot = {"first_image": first_image, "prompt": request.get("prompt", ""),
                "style_name": request.get("style_name", ""), "references": references,
                "requested_mode": mode, "model": model, "size": size, "quality": quality,
                "steps": steps, "firmware_text": resolve_firmware_text(firmware),
                "outputs": list(outputs or [])}
    target = str(first_image) + ".request.json"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return target

def first_pass_sub_dir(steps: dict, run_key: str = "") -> str:
    """gpt-image 首图的落盘子目录。

    没有后续工序 → 首图就是最终产物 → 返回 ""（直接落 `data/<日期>/`，方便发布）；
    有工序 → 首图是中间产物 → 落 `analysis-gpt-image/<单次任务>/`。
    `run_key` 留空时保留旧返回值，供旧调用方与测试兼容。
    """
    enabled = [k for k, v in (steps or {}).items() if isinstance(v, dict) and v.get("enabled")]
    if not enabled:
        return ""
    if not str(run_key or "").strip():
        return "analysis-gpt-image"
    import re
    import time
    import uuid
    safe = re.sub(r'[^0-9A-Za-z._-]+', '-', str(run_key)).strip('-_.') or "run"
    return os.path.join("analysis-gpt-image", f"{safe[:80]}-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}")


def publish_final_output(source_path: str, *, style_name: str = "", process_dir: str = "",
                         final_dir: str = "") -> str:
    """把工作目录里选中的最终图只发布一份到 `data/<YYYYMMDD>/`。

    首图、首次重绘、质量/身份修订和审计 JSON 均保留在 `process_dir`；
    发布目录不再暴露中间版本。
    """
    import json
    import re
    import shutil
    from utils import output_isolation
    from utils.post_process import date_output_dir

    source = os.path.abspath(str(source_path or ""))
    if not os.path.isfile(source):
        raise FileNotFoundError(source)
    raw_final_dir = final_dir or date_output_dir()
    style = re.sub(r'[\\/*?:"<>|]+', "-", str(style_name or "")).strip(" .-")
    source_name = os.path.basename(source)
    stem, ext = os.path.splitext(source_name)
    if "final" not in stem.lower():
        stem += "-final"
    if style and not stem.lower().startswith(style.lower() + "-"):
        stem = f"{style}-{stem}"
    target_dir, filename = output_isolation.resolve_output_target(raw_final_dir, stem + ext)
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.abspath(os.path.join(target_dir, filename))
    if os.path.normcase(target) != os.path.normcase(source):
        base, suffix = os.path.splitext(target)
        idx = 1
        while os.path.exists(target):
            target = f"{base}_{idx}{suffix}"
            idx += 1
        shutil.copy2(source, target)
    trace_dir = os.path.abspath(process_dir or os.path.dirname(source))
    os.makedirs(trace_dir, exist_ok=True)
    with open(os.path.join(trace_dir, "published-final.json"), "w", encoding="utf-8") as f:
        json.dump({"selected_process_output": source, "published_final": target,
                   "style_name": str(style_name or "")}, f, ensure_ascii=False, indent=2)
    return target


def resolve_content_text(analysis_result: dict, tier: str = "short") -> str:
    """老路径兜底：按档位取 gpt 专用字段（prompt_gpt 时代留下的兼容路径）。

    GUI 会显式传身份完整锚；这个函数供 CLI 选档和老调用方兜底。
    """
    data = analysis_result or {}
    short = str(tier).lower() == "short"
    limit = SHORT_FIELD_MAX_CHARS if short else FIELD_MAX_CHARS
    primary, secondary = (SHORT_FIELD_KEY, FIELD_KEY) if short else (FIELD_KEY, SHORT_FIELD_KEY)
    text = str(data.get(primary) or "").strip()
    if not text:
        text = str(data.get(secondary) or "").strip()
    if not text:  # 两档都缺（老产物）：退回 refined 描述并按档位上限裁剪
        text = str(data.get("english_description") or data.get("original_english_description") or "").strip()
    if len(text) > limit:
        cut = text[:limit]
        for sep in (". ", "; ", ", "):
            if sep in cut:
                cut = cut[: cut.rfind(sep)]
                break
        text = cut.strip().rstrip(",;.")
    return text


def build_gpt_image_request(analysis_result: dict, style_text: str = "", style_ref_path: str = "",
                           user_hint: str = "", tier: str = "short",
                           extra_clauses=None, content_image_path: str = "",
                           content_text: str = "") -> dict:
    """组装 gpt-image 通道的 (prompt, image_paths)。

    - `style_text` 传 prompt_gpt（取自 utils.style_gpt.resolve_style_prompt / utils.styles.style_prompt_gpt），
      **排在提示词最前面**
    - `content_text`：**和 Gemini 通道同源**的分析描述（素材特征 / 构图 / 服装 / 道具）。显式给了就用它、
      不再按档位截断；没给才退回 `resolve_content_text`（老调用方兼容）。
    - `style_ref_path` 传画风的参考图；路径不存在则不挂
    - `content_image_path`：分析原图。**默认不传**（GUI 的 gpt 通道从不挂素材图）；只有明确要"描图"时才用
    - `user_hint` 是可选的人工补充（放在内容描述之后）
    - `extra_clauses`：画风条目的"渲染语言条款"（如 repaint_clauses），追加到提示词末尾，
      让**眼睛/睫毛/头发/蕾丝**这类作画方法在**首图生成阶段**就正确（不然重绘阶段只能修线条，
      修不回五官画法）。
    """
    content = str(content_text or "").strip() or resolve_content_text(analysis_result, tier=tier)
    content_ref = str(content_image_path or "").strip()
    has_content_ref = bool(content_ref) and os.path.isfile(content_ref)
    parts = []
    if has_content_ref and (style_ref_path and os.path.isfile(str(style_ref_path))):
        # 两类图同时存在 → 必须先讲清角色分工（内容图在前、画风图最后）
        from utils.styles import STYLE_REF_ROLE_INSTRUCTION
        parts.append(STYLE_REF_ROLE_INSTRUCTION.replace("{content_count}", "1"))
    if style_text:                       # 画风说明放最前面
        parts.append(str(style_text).strip())
    if content:
        parts.append(content)
    if user_hint:
        parts.append(str(user_hint).strip())
    images = []
    if has_content_ref:
        images.append(content_ref)
    if style_ref_path and os.path.isfile(style_ref_path):
        images.append(str(style_ref_path))
        # 挂了画风参考图 → 必须写清"只借渲染、别搬角色"
        parts.append(STYLE_REF_EXCLUSION)
    clauses = [str(c).strip() for c in (extra_clauses or []) if str(c).strip()]
    if clauses:
        parts.append("RENDERING LANGUAGE (follow exactly):\n- " + "\n- ".join(clauses))
    prompt = "\n\n".join(p for p in parts if p)
    return {"prompt": prompt, "image_paths": images, "content_chars": len(content),
            "style_chars": len(str(style_text or "")), "has_content_ref": has_content_ref}


def build_first_pass_request(styles_data, style_name, analysis_result, content_text: str = "",
                             user_hint: str = "", tier: str = "short",
                             content_image_path: str = "", api_type: str = "",
                             prompt_recipe: str = "legacy") -> dict:
    """**首图请求的唯一组装点**（GUI 的 gpt 通道与 `tools/analysis_gpt_run.py` 共用）。

    以前 GUI 自己拼这一段，漏了渲染语言条款（`extra_clauses` 没传），于是 GUI 出的首图永远比 CLI
    少一截"线条怎么画 / 颜色锚点在哪 / 别画成什么"的约束 —— 实测这正是首图偏白偏灰、线条碎的主因。
    现在两处都从这里走，缺参数这类偏差不会再出现。

    组装顺序：画风 `prompt_gpt` → 内容锚 → 参考图排除句 → `RENDERING LANGUAGE` 条款。
    条款来源：画风条目手写的 `repaint_clauses` 优先，缺失时按 `prompt_gpt` 字段确定性派生。
    """
    from utils.style_gpt import resolve_style_clauses, style_prompt_gpt
    from utils.styles import style_ref_image, ref_image_valid

    name = str(style_name or "")
    entry = (styles_data or {}).get(name) if name else None
    style_text = style_prompt_gpt(styles_data or {}, name).strip()
    if not style_text and isinstance(entry, str):
        style_text = entry
    elif not style_text and isinstance(entry, dict):
        style_text = str(entry.get("prompt") or "").strip()
    ref = style_ref_image(styles_data or {}, name) if name else ""
    ref = str(ref or "") if ref_image_valid(ref) else ""
    clauses, clauses_source = resolve_style_clauses(entry)
    payload = build_gpt_image_request(analysis_result, style_text=style_text, style_ref_path=ref,
                                      user_hint=user_hint, tier=tier, extra_clauses=clauses or None,
                                      content_image_path=content_image_path, content_text=content_text)
    payload.update({"style_name": name, "style_ref_path": ref, "clauses": clauses,
                    "clauses_source": clauses_source, "api_type": str(api_type or "")})
    if prompt_recipe == "reference" and ref and not content_image_path:
        from utils.prompt_loader import render_prompt_file
        content = str(content_text or "").strip() or resolve_content_text(analysis_result, tier)
        payload["prompt"] = render_prompt_file("analysis-style-reference.md", {
            "style_text": style_text, "content_text": content, "user_hint": user_hint or "None."})
        # Only genuinely authored additions are retained; derived clauses duplicate the eight fields.
        if clauses_source == "entry" and clauses:
            payload["prompt"] += "\n\nRENDERING LANGUAGE:\n- " + "\n- ".join(clauses)
    payload["prompt_recipe"] = prompt_recipe
    return payload


def pipeline_steps_from_flags(repaint: bool = False, structure: bool = False, local: bool = False,
                              structure_strength: float = 0.5, local_region: str = "hair",
                              local_feather: int = 48, resolution: str = "2K",
                              detail_boost: bool = False, repaint_ref_mode: str = "style",
                              local_regions=None, tone: bool = False, tone_target: str = "style",
                              ink: bool = False, ink_target: float = 8.0,
                              repaint_scope: str = "full") -> dict:
    """把界面上的勾选翻译成 utils.post_process 的流水线步骤（repaint 由调用方单独处理）。

    - `repaint_scope`：**「重绘编辑范围」** —— 不裁切、不贴回，只在重绘提示词里要求模型保留不该动的部分：
      `full` / `person_only` / `person_noface` / `details` / `lines_only`（文本见
      `post_process.REPAINT_SCOPE_CLAUSES`）。默认 `full`，让 v5 固件保守修复整图。
    - `detail_boost=False`（默认）：局部重绘统一用 2K（局部重绘现已不是默认工序，见下）。
    - `repaint_ref_mode="style"`（默认）：重绘接收 GPT 首图和完整画风图，以提高主轮廓和线稿连续性；
      角色偏离由首图 prompt 驱动的身份审计与最多两轮定点修订处理。`none` 可用于保真对照。
    - `local` / `local_regions`：**裁切→重绘→贴回**式的局部重绘。**默认关闭、慎用**：模型会在裁切里
      重新构图，贴回原坐标就是一块错位内容（§三十：5 画风里报废 2 张）。要修细节请优先用 `repaint_scope`。
    - `tone` / `ink`：**本地工序**（不调 API）—— 色调校准（目标默认取画风参考图）+ 线条加墨。
      实测（§二十八）：当前首图那条链 163.9/68.5/+0.18 → **138.6/93.8/+2.54**；
      A 基底上 136.9/113.2/+1.78 → **131.7/117.8/+3.21**（参考图 135.6/114.3/+2.55）。
    """
    regions = [str(r).strip() for r in (local_regions or []) if str(r).strip()]
    if not regions:
        regions = [str(local_region or "hair")]
    return {
        "repaint": {"enabled": bool(repaint), "resolution": resolution,
                    "reference_mode": str(repaint_ref_mode or "style"),
                    "scope": str(repaint_scope or "full")},
        "structure": {"enabled": bool(structure), "strength": float(structure_strength)},
        "local": {"enabled": bool(local), "region": regions[0], "regions": regions,
                  "feather": int(local_feather), "resolution": resolution,
                  "detail_boost": bool(detail_boost)},
        "tone": {"enabled": bool(tone), "tone_target": str(tone_target or "style"),
                 "reference_path": "", "contrast": 1.00, "chroma": 1.10,
                 "highlight_strength": 0.85, "skin_warm": 0.0, "sat_target_scale": 1.0},
        "ink": {"enabled": bool(ink), "target_sep": float(ink_target), "max_darken": 40.0},
    }


def run_gpt_image_pipeline(paths, steps, firmware: str = "", log_callback=None,
                           final_dir: str = None, work_dir: str = None,
                           style_ref_path: str = "", style_clauses=None) -> list:
    """按勾选对 gpt-image 产物跑「重绘 → 结构线叠加 → 局部重绘 → 色调校准 → 加墨」。

    统一委托给 `utils.post_process.run_pipeline`：最后一道工序的产物落 `data/<日期>/`（发布目录），
    中间产物落 `<产物目录>/pipeline-steps/`，并支持断点重试（resume）。

    `style_ref_path` / `style_clauses` 传给重绘：`reference_mode="style"` 时第二张参考就是这张画风图
    （GUI 以前没传 → 走默认的线锚图，会把线网印刷到画面上）。
    色调校准（`tone`）的目标参考图也取 `style_ref_path`：目标=画风参考图时才能把画面压到参考图的亮度/饱和度。
    """
    from utils import post_process as pp
    log = log_callback or (lambda m: None)
    steps = steps or {}
    if not any((steps.get(k) or {}).get("enabled")
               for k in ("repaint", "structure", "local", "tone", "ink")):
        return [p for p in (paths or []) if p and os.path.isfile(p)]
    tone_cfg = dict(steps.get("tone") or {"enabled": False})
    if tone_cfg.get("enabled") and str(tone_cfg.get("tone_target") or "style") == "style" \
            and style_ref_path and os.path.isfile(str(style_ref_path)):
        tone_cfg["reference_path"] = str(style_ref_path)      # 目标 = 画风参考图
    return pp.run_pipeline(paths, {"structure": steps.get("structure") or {"enabled": False},
                                   "local": steps.get("local") or {"enabled": False},
                                   "tone": tone_cfg,
                                   "ink": steps.get("ink") or {"enabled": False},
                                   "repaint": steps.get("repaint") or {"enabled": False}},
                           firmware=firmware, log_callback=log,
                           final_dir=final_dir, work_dir=work_dir,
                           style_ref_path=style_ref_path, style_clauses=style_clauses)
