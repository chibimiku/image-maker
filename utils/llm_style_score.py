#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多图对比 → 文字/视觉模型 0-100 画风相似度打分。

与本地指标的定位区别：
- 本地指标（`utils/style_similarity.py`）算的是**深度特征距离/统计贴近度**，是数值不是判断。
- 本模块让**视觉模型自己看图**给出 0-100 分 + 分项 + 依据，属于「大模型视觉检测」。

复用现有文字分析图接口（`utils/analysis_gpt_prompt.call_text_model`，走 `POST /chat/completions`
的多模态 `image_url`），**不新增任何后端**。提示词要点（与实际业务约束对应）：
1. 一次请求发多张图：先给**全部参考图**（画风来源常常 10+ 张），再给**待评候选图**，带编号标签。
2. 明确「评的是画风而不是内容」：人物长相/发色/服装/姿势/场景差异**不计入**扣分。
3. 参考集内部本来就多样，要求模型先在心里归纳参考集的共性，再评候选，而不是逐张比对。
4. 禁止把「候选是否抄了参考图内容」当作加分（抄袭与画风相似是两件事），只评画法。
5. 输出严格 JSON：0-100 总分 + 5 个分项（线条笔触/五官头发/上色光影/材质纹理/背景处理）+ 依据。

用法（CLI 见 `tools/llm_style_score.py`）：
    score_candidates(references, candidates) -> {"assessments": [...], "raw": "..."}
"""
import json
import os
import re

# 分项权重：与「画风」直接相关的四项为主，背景处理权重略低（内容差异影响最大）
FACETS = (
    ("linework", "线条与笔触：线的粗细变化、收尾方式、是否封边、笔触可见度", 0.25),
    ("face_hair", "五官与头发：脸型抽象程度、眼睛画法、发束组织与高光方式", 0.20),
    ("shading", "上色与光影：明暗层次、阴影形状、高光/环境光处理、整体色调倾向", 0.25),
    ("texture", "材质与层次：留白、颗粒/水彩/网点质感、细节密度与主次关系", 0.20),
    ("background", "背景处理：背景画法、透视强弱、背景细节量与主体的关系", 0.10),
)

SYSTEM_PROMPT = """You are a conservative anime art-style reviewer. You receive several STYLE REFERENCE images
of ONE art style, followed by CANDIDATE images that were generated with that style as the target.

Judge ONLY the rendering style (how it is drawn and painted). IGNORE content differences entirely:
characters, faces, hair colour, eye colour, clothing, pose, camera angle, props and scene must NOT
affect the score. A candidate showing a completely different character in the same style is a match.

Rules:
1. First infer the common style of the reference set (they are individually diverse; look for the shared
   drawing conventions instead of comparing against one favourite reference).
2. Score each candidate 0-100 for style fidelity:
   0-20 different style, 21-40 mainly different with a few shared habits, 41-60 mixed,
   61-80 clearly the same family with visible deviations, 81-100 convincingly the same style.
3. Also give five 0-100 sub-scores: linework, face_hair, shading, texture, background.
4. Do NOT reward a candidate for copying the reference's content, and do NOT penalise it for using
   different content. Content similarity is irrelevant here.
5. Be conservative: if you would hesitate between two bands, choose the lower one.

Return ONLY JSON, no prose and no code fence:
{"overall_note": "<one sentence about the reference set's shared style>",
 "assessments": [{"id": "<candidate id>", "style_score": <0-100>, "linework": <0-100>,
   "face_hair": <0-100>, "shading": <0-100>, "texture": <0-100>, "background": <0-100>,
   "reason": "<Chinese, 1-2 sentences citing the drawing habits you compared>",
   "biggest_gap": "<Chinese, the single most obvious deviation>"}]}"""

BATCH_SIZE = 4


def build_inputs(reference_paths, candidates, batch_size=BATCH_SIZE):
    """按批组装请求：每批都发全部参考图 + 该批候选。

    候选标签用**短安全 id**（c1/c2/...）：真实 id 里带 `·`、空格、长数字，模型回抄时容易
    丢字符或改写，导致「覆盖校验」误报。短 id 与真实 id 的映射在本函数的 `id_map` 里。
    """
    batches = []
    for offset in range(0, len(candidates), batch_size):
        chunk = candidates[offset:offset + batch_size]
        id_map = {}
        mapped = []
        for index, candidate in enumerate(chunk, 1):
            label = f"c{index}"
            id_map[label] = candidate["id"]
            mapped.append({"label": label, "path": candidate["path"]})
        batches.append({"references": list(reference_paths), "candidates": mapped, "id_map": id_map})
    return batches


def build_user_prompt(batch, note=""):
    lines = [
        f"STYLE REFERENCE SET — {len(batch['references'])} image(s) of the SAME style follow, in order:",
    ]
    for index, path in enumerate(batch["references"], 1):
        lines.append(f"  reference_{index}: {os.path.basename(path)}")
    lines.append("")
    lines.append(f"Then {len(batch['candidates'])} candidate image(s):")
    for candidate in batch["candidates"]:
        lines.append(f"  id \"{candidate['label']}\": {os.path.basename(candidate['path'])}")
    lines.append("")
    lines.append("Score every candidate on style fidelity only (content is irrelevant).")
    lines.append("Use exactly the ids given above (c1, c2, ...) in the assessments array.")
    if note:
        lines.append(note)
    return "\n".join(lines)


def parse_scores(raw, expected_ids):
    """解析模型返回，校验覆盖与取值范围；不合法直接抛错（不静默补值）。"""
    text = str(raw or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("模型没有返回 JSON")
    data = json.loads(match.group(0))
    items = data.get("assessments")
    if not isinstance(items, list):
        raise ValueError("assessments 不是数组")
    ids = [item.get("id") for item in items]
    if sorted(map(str, ids)) != sorted(map(str, expected_ids)):
        raise ValueError(f"候选覆盖不符：期望 {sorted(expected_ids)}，得到 {sorted(map(str, ids))}")
    out = []
    for item in items:
        row = {"id": str(item["id"]), "reason": str(item.get("reason") or ""),
               "biggest_gap": str(item.get("biggest_gap") or "")}
        for key in ("style_score", *(facet[0] for facet in FACETS)):
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 100:
                raise ValueError(f"{row['id']} 的 {key} 必须是 0-100 数值")
            row[key] = float(value)
        # 分项加权复核：总分与分项差超过 15 分时记录，不擅自改分
        weighted = sum(row[key] * weight for key, _, weight in FACETS)
        row["facet_weighted"] = round(weighted, 1)
        row["score_gap"] = round(row["style_score"] - weighted, 1)
        out.append(row)
    return {"overall_note": str(data.get("overall_note") or ""), "assessments": out}


def score_candidates(reference_paths, candidates, config=None, timeout=300, log=None,
                     call_model=None, batch_size=BATCH_SIZE):
    """对候选逐批打分，返回 {overall_note, assessments, batches, raw[]}。

    call_model：注入用（默认 `utils.analysis_gpt_prompt.call_text_model`），便于离线测试。
    config：`utils.analysis_gpt_prompt.load_text_api_config()` 的返回；缺省时自行加载。
    """
    if not reference_paths or not candidates:
        raise ValueError("需要至少一张参考图与一个候选")
    if call_model is None:
        from utils.analysis_gpt_prompt import call_text_model as call_model  # noqa
    if config is None:
        from utils.analysis_gpt_prompt import load_text_api_config
        config = load_text_api_config()
    base_url = config.get("base_url")
    api_key = config.get("api_key")
    model = config.get("model")
    if not (base_url and api_key and model):
        raise RuntimeError("缺少文本 API 配置（base_url / api_key / model）")

    collected, raws, notes = [], [], []
    for batch in build_inputs(reference_paths, candidates, batch_size=batch_size):
        prompt = build_user_prompt(batch)
        paths = [str(p) for p in batch["references"]] + [str(c["path"]) for c in batch["candidates"]]
        if log:
            log(f"[llm-score] {len(batch['references'])} 参考 + {len(batch['candidates'])} 候选，{len(paths)} 张图")
        raw = call_model(base_url, api_key, model, SYSTEM_PROMPT, prompt,
                         timeout=timeout, max_tokens=6000, image_paths=paths)
        parsed = parse_scores(raw, [c["label"] for c in batch["candidates"]])
        for item in parsed["assessments"]:
            item["label"] = item["id"]
            item["id"] = batch["id_map"].get(item["id"], item["id"])
        collected.extend(parsed["assessments"])
        if parsed.get("overall_note"):
            notes.append(parsed["overall_note"])
        raws.append(raw)
    return {"overall_note": " / ".join(notes), "assessments": collected, "raw": raws,
            "model": model, "references": len(reference_paths), "candidates": len(candidates)}


def score_to_verdict(score):
    """0-100 → 三档（与本地指标判定并列展示，不互相替代）。"""
    if score >= 80:
        return "通过"
    if score >= 60:
        return "存疑"
    return "不通过"
