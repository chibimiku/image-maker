"""画风测试的固定评分、可复算记录及完整画风导入。"""
import hashlib
import json
import math
import os
import re
import shutil
import html
import base64
import mimetypes
from decimal import Decimal, ROUND_HALF_UP

from openai import OpenAI
from PyQt6.QtCore import QThread, pyqtSignal
from utils.image_encoding import compress_and_encode_image
from utils.prompt_loader import read_prompt_file

VERSION = "style-comparison-v2"
LEGACY_DIMENSIONS = ("linework", "face_hair", "shading", "color_logic", "texture_finish")
STYLE_GROUPS = {
    "线条与边缘": ("linework", "edge_hierarchy"),
    "五官与头发": ("face_geometry", "eye_rendering", "hair_grouping"),
    "明暗与光照": ("shading", "lighting"),
    "配色逻辑": ("color_logic",),
    "表面与层次": ("texture_finish", "material_response", "detail_hierarchy", "background_rendering"),
}
DIMENSIONS = tuple(key for keys in STYLE_GROUPS.values() for key in keys)
DIMENSION_LABELS = dict(zip(DIMENSIONS, ("线条笔触", "边缘层级", "脸型抽象", "眼睛画法", "发束组织", "明暗塑形", "光照处理", "配色逻辑", "纹理质感", "材质响应", "细节层级", "背景画法")))
DIMENSION_LABELS["face_hair"] = "五官头发（v1）"
FORMULA = "v2：12 项分成 5 组，组内等权平均、每组占 20%；画风分 S=5 组均分；总分=0.85×S+0.15×主体符合度（0–10）"


def research_basis():
    """论文依据与实际计算状态分开，避免把模型打分冒充深度特征指标。"""
    return [
        {"method": "Gram style loss", "url": "https://arxiv.org/abs/1508.06576", "status": "未计算",
         "basis": "Gatys 等以多层 CNN 特征的 Gram 矩阵匹配风格统计，启发纹理、笔触和多尺度评价。",
         "limit": "需要预训练特征网络；受配色和纹理分布影响，不保证保留身份或精确评价动漫五官。"},
        {"method": "AdaIN feature statistics", "url": "https://arxiv.org/abs/1703.06868", "status": "未计算",
         "basis": "Huang 与 Belongie 对齐深度特征通道均值与标准差，说明风格统计与内容空间结构应分开。",
         "limit": "深度特征均值/标准差并非像素配色均值；实际计算结果单独列于深度指标报告。"},
        {"method": "LPIPS", "url": "https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html", "status": "未计算",
         "basis": "Zhang 等研究学习深度特征距离与人类感知判断的关系，适合作为同内容图像的感知差异参考。",
         "limit": "不是纯画风距离；不同角色、姿势和构图的比较有内容混杂，不能把 1−LPIPS 叫画风相似百分比。"},
        {"method": "CSD style descriptor", "url": "https://arxiv.org/abs/2404.01292", "status": "未计算",
         "basis": "Somepalli 等通过风格对比学习构建专用描述子，用于风格检索与归因，关注颜色、纹理和形状的交互。",
         "limit": "需匹配官方模型、权重与预处理；论文检索基准的效果不等于本项目动漫画风已经校准。"},
        {"method": "LoRA / diffusion training loss", "url": "https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py", "status": "不适用",
         "basis": "官方 LoRA 示例对 epsilon 或 v_prediction 目标计算 MSE，支持 Min-SNR 时间步加权；LoRA 本身是低秩参数更新方式。",
         "limit": "当前流程仅迭代提示词，不训练模型权重；无加噪 latent、时间步及模型预测，不能计算真实训练 loss，更不能由 10−视觉分伪造。"},
    ]


def candidates_from_state(state):
    records = dict(state.get("test_images") or {})
    records["final"] = state.get("final_test_images") or {}
    candidates = []
    for stage, record in sorted(records.items()):
        for channel, paths in sorted((record.get("generated_files") or {}).items()):
            for index, path in enumerate(paths):
                candidates.append({"id": f"{stage}/{channel}/{index}", "path": path,
                                   "stage": stage, "channel": channel, "record": record})
    return candidates


def calculate_ranking(candidates, assessments, version=VERSION):
    """只接收原始分数，严格验证覆盖率；忽略模型声称的总分及最佳版本。"""
    if version not in (VERSION, "style-comparison-v1"):
        raise ValueError("不支持的评分版本")
    dimensions = DIMENSIONS if version == VERSION else LEGACY_DIMENSIONS
    ids = [c["id"] for c in candidates]
    if len(set(ids)) != len(ids) or not isinstance(assessments, list):
        raise ValueError("候选或评分格式错误")
    if len(assessments) != len(ids) or {a.get("id") for a in assessments} != set(ids):
        raise ValueError("评分必须恰好覆盖全部候选，不能遗漏、重复或添加未知候选")
    by_id = {a["id"]: a for a in assessments}
    rows = []
    for candidate in candidates:
        raw = by_id[candidate["id"]]
        scores = {}
        for key in (*dimensions, "subject_fidelity"):
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 10:
                raise ValueError(f"{candidate['id']} 的 {key} 必须是 0–10 有限数值")
            scores[key] = Decimal(str(value))
        gates = ("reference_content_copied", "major_structure_defect", "explicit_subject_mismatch", "uncertain")
        if any(type(raw.get(key)) is not bool for key in gates):
            raise ValueError("评分必须包含四个明确布尔门禁")
        reasons = str(raw.get("reason") or "").strip()
        if not reasons:
            raise ValueError("评分缺少依据")
        evidence = raw.get("dimension_evidence")
        if version == VERSION:
            if not isinstance(evidence, dict) or set(evidence) != set(dimensions):
                raise ValueError("v2 每个维度都须提供源图与候选差异依据")
            for key, item in evidence.items():
                if not isinstance(item, dict) or any(not isinstance(item.get(field), str) or not item[field].strip() for field in ("reference", "candidate", "difference")):
                    raise ValueError(f"{key} 缺少 reference/candidate/difference 证据")
            groups = {group: sum(scores[key] for key in keys) / Decimal(len(keys)) for group, keys in STYLE_GROUPS.items()}
            style = sum(groups.values()) / Decimal(5)
        else:
            groups = {}
            style = sum(scores[key] for key in dimensions) / Decimal(5)
        total = style * Decimal("0.85") + scores["subject_fidelity"] * Decimal("0.15")
        rounded = lambda number: float(number.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
        rows.append({"id": candidate["id"], "path": candidate["path"],
                     **{key: float(value) for key, value in scores.items()},
                     "style_score": rounded(style), "total": rounded(total),
                     "eligible": not any(raw[key] for key in gates),
                     "gates": {key: raw[key] for key in gates}, "reason": reasons,
                     "sort_value": str(total)})
        rows[-1].update(group_scores={key: rounded(value) for key, value in groups.items()}, dimension_evidence=evidence or {})
    rows.sort(key=lambda row: (-Decimal(row["sort_value"]), row["id"]))
    best = next((row["id"] for row in rows if row["eligible"]), None)
    formula = FORMULA if version == VERSION else "v1：五项等权均分 S；总分=0.85×S+0.15×主体符合度（0–10）"
    return {"version": version, "formula": formula, "dimensions": list(dimensions),
            "groups": STYLE_GROUPS if version == VERSION else {},
            "research_basis": research_basis(), "score_type": "vision_model_rubric", "rows": rows, "best_id": best}


def comparison_inputs(state, model, base_url):
    candidates = candidates_from_state(state)
    refs = state.get("dataset", {}).get("images") or []
    if not refs or not candidates:
        raise ValueError("需要原始画风图及已生成的测试图片，未测试的版本不能比较")
    prompt = read_prompt_file("style-comparison-v2.md")
    signature = {"version": VERSION, "model": model, "endpoint": base_url, "prompt": prompt,
                 "references": [], "candidates": []}
    def fingerprint(path):
        with open(path, "rb") as source:
            return hashlib.sha256(source.read()).hexdigest()
    for path in refs:
        signature["references"].append(fingerprint(path))
    subjects = {c["record"].get("test_prompt", "") for c in candidates}
    references = {c["record"].get("style_reference", "") for c in candidates}
    if len(subjects) != 1 or not next(iter(subjects)) or len(references) != 1:
        raise ValueError("候选必须使用相同非空测试主体及同一画风参考图")
    for candidate in candidates:
        record = candidate["record"]
        signature["candidates"].append({"id": candidate["id"], "image": fingerprint(candidate["path"]),
                                        "subject": record.get("test_prompt"), "package": record.get("prompt_variants"),
                                        "master": record.get("prompts_used"),
                                        "style_reference": fingerprint(record["style_reference"])})
    digest = hashlib.sha256(json.dumps(signature, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return candidates, refs, prompt, digest


def write_comparison_report(state, result, directory):
    """自包含报告：原图、测试图、分项和版本对应的提示词均可离线查看。"""
    def picture(path):
        with open(path, "rb") as source:
            encoded = base64.b64encode(source.read()).decode()
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        return f'<img loading="lazy" src="data:{mime};base64,{encoded}">'
    esc = lambda value: html.escape(str(value))
    candidate_map = {c["id"]: c for c in candidates_from_state(state)}
    parameters = state.get("parameters") or {}
    training_rows = [("训练轮次（本次新增）", parameters.get("total_rounds", "未记录")),
                     ("每轮检查图片数", parameters.get("images_per_round", "未记录")),
                     ("参考数据集图片数", len(state["dataset"]["images"])),
                     ("分析模型", ", ".join(sorted({str(it["model_used"]) for it in state.get("iterations", []) if it.get("model_used")})) or "未记录"),
                     ("评分模型", result["model"]), ("评分温度", 0),
                     ("评分图片最长边 / JPEG 质量", "1536 px / 90"),
                     ("评分版本", result["version"]),
                     ("评分缓存", "复用已保存评分" if result.get("cached") else "本次评分")]
    parameters_html = '<h2>运行参数</h2><table>' + ''.join('<tr><th>' + esc(label) + '</th><td>' + esc(value) + '</td></tr>' for label, value in training_rows) + '</table>'
    dimensions = result.get("dimensions") or (LEGACY_DIMENSIONS if result["version"] == "style-comparison-v1" else DIMENSIONS)
    papers = [dict(p) for p in (result.get("research_basis") or research_basis())]
    deep = state.get("deep_feature_comparison") or {}
    if deep:
        for paper, metric in zip(papers[:4], ("gram", "adain", "lpips", "csd")):
            complete = sum(r["summary"][metric]["status"] == "ok" for r in deep.get("rows", []))
            paper["status"] = f"已计算 {complete}/{len(deep.get('rows', []))} 个完整候选；{deep['backend']['actual']} / {deep['backend']['precision']}"
    research_html = '<h2>评分依据与算法状态</h2><p>视觉评分表由模型按结构化量表判断；12 项与组权重是本项目设计，未经论文或人类标注校准。本地深度指标是否运行以状态栏为准，实际数值独立附表，不混入视觉总分。论文基准效果尚未在本项目复现。缺失指标不置零、不参与排名。当前提示词迭代无真实 LoRA training loss。</p><table><tr><th>方法与来源</th><th>状态</th><th>参考依据</th><th>适用限制</th></tr>'
    for paper in papers:
        research_html += '<tr><td><a href="' + esc(paper["url"]) + '">' + esc(paper["method"]) + '</a></td><td>' + esc(paper["status"]) + '</td><td>' + esc(paper["basis"]) + '</td><td>' + esc(paper["limit"]) + '</td></tr>'
    research_html += '</table>'
    if state.get("resume_origin"):
        parameters_html += '<p>续训来源：' + esc(state["resume_origin"].get("candidate_id", "未记录")) + '</p>'
    sections = ['<!doctype html><meta charset="utf-8"><title>画风比较报告</title>',
                '<style>body{font:16px sans-serif;margin:24px;background:#f5f5f5}img{max-width:260px;max-height:360px}article{background:white;padding:20px;margin:16px 0}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}pre{white-space:pre-wrap}a{margin-right:12px}</style>',
                '<h1>画风比较报告</h1><p>' + esc(result.get("formula", FORMULA)) + '</p>',
                '<p>视觉分项是模型判断，不是相似度百分比。四项门禁任何一项为真均不入选；同分按候选 ID 排序。排名选择提示词版本，图像仍属于表中具体通道。</p>',
                '<p>模型：' + esc(result["model"]) + '；输入摘要：' + esc(result["input_hash"]) + '</p>',
                parameters_html,
                research_html,
                '<h2>原始画风图</h2>' + ''.join(picture(p) for p in state["dataset"]["images"]),
                '<h2>最佳候选：' + esc(result["best_id"] or "无候选通过门禁") + '</h2>',
                '<table><tr><th>候选</th>' + ''.join('<th>' + esc(DIMENSION_LABELS.get(k, k)) + '</th>' for k in (*dimensions, "style_score", "subject_fidelity", "total", "eligible")) + '</tr>']
    for row in result["rows"]:
        sections.append('<tr><td><a href="#' + esc(row["id"]) + '">' + esc(row["id"]) + '</a></td>' + ''.join('<td>' + esc(row[k]) + '</td>' for k in (*dimensions, "style_score", "subject_fidelity", "total", "eligible")) + '</tr>')
    sections.append('</table>')
    if deep:
        from pathlib import Path
        from modules.image_analysis.style_deep_comparison import LIMITS
        sections.append('<h2>真实深度特征指标</h2><p>' + esc(LIMITS) + '</p><p><a href="' + esc(Path(deep["report_path"]).as_uri()) + '">逐张数值与模型参数报告</a></p><table><tr><th>候选</th><th>CSD ↑</th><th>Gram ↓</th><th>AdaIN ↓</th><th>LPIPS ↓</th></tr>')
        for row in deep.get("rows", []):
            sections.append('<tr><td>' + esc(row["id"]) + '</td>' + ''.join('<td>' + (format(row["summary"][m]["mean"], '.8g') if row["summary"][m]["mean"] is not None else '缺失') + '</td>' for m in ("csd", "gram", "adain", "lpips")) + '</tr>')
        sections.append('</table>')
    for row in result["rows"]:
        record = candidate_map[row["id"]]["record"]
        sections.append('<h3>' + esc(row["id"]) + '：分组分数与分项依据</h3><p>' + esc(json.dumps(row.get("group_scores", {}), ensure_ascii=False)) + '</p><table><tr><th>维度</th><th>源图依据</th><th>候选依据</th><th>差异</th></tr>' + ''.join('<tr><th>' + esc(DIMENSION_LABELS.get(key, key)) + '</th>' + ''.join('<td>' + esc(item[field]) + '</td>' for field in ("reference", "candidate", "difference")) + '</tr>' for key, item in row.get("dimension_evidence", {}).items()) + '</table>')
        sections.append('<article id="' + esc(row["id"]) + '"><h2>' + esc(row["id"]) + '</h2>' + picture(row["path"]) + '<p>' + esc(row["reason"]) + '</p><p>门禁：' + esc(json.dumps(row["gates"], ensure_ascii=False)) + '</p><p>测试主体：' + esc(record["test_prompt"]) + '</p><p>测试画风参考图：' + esc(record.get("style_reference", "未记录")) + '</p><details><summary>本版本完整提示词包</summary><pre>' + esc(json.dumps({"master": record.get("prompts_used"), "variants": record.get("prompt_variants")}, ensure_ascii=False, indent=2)) + '</pre></details></article>')
    path = os.path.join(directory, "automatic-comparison.html")
    with open(path, "w", encoding="utf-8") as target:
        target.write('\n'.join(sections))
    return path


def resume_seed(state, candidate_id, source_path):
    candidate = next((c for c in candidates_from_state(state) if c["id"] == candidate_id), None)
    if not candidate or not candidate["record"].get("prompts_used"):
        raise ValueError("选中的版本没有可用分析提示词")
    record = candidate["record"]
    return {"dataset": state["dataset"], "iterations": [{"round": 0, "type": "selected_version_seed", "art_style_prompts": record["prompts_used"]}],
            "file_prefix": state.get("file_prefix", "style"),
            "resume_origin": {"source_json": os.path.abspath(source_path), "candidate_id": candidate_id,
                              "prompt_variants": record.get("prompt_variants", {})}}, record


class StyleComparisonWorker(QThread):
    completed = pyqtSignal(object, str)

    def __init__(self, path, config, timeout=600, parent=None):
        super().__init__(parent)
        self.path, self.config, self.timeout = path, config, timeout

    def run(self):
        try:
            with open(self.path, encoding="utf-8") as source:
                state = json.load(source)
            base_url, api_key, model = self.config
            candidates, refs, prompt, digest = comparison_inputs(state, model, base_url)
            cached = state.get("automatic_comparison") or {}
            if cached.get("input_hash") == digest:
                result = calculate_ranking(candidates, cached["assessments"])
                result.update(input_hash=digest, assessments=cached["assessments"], model=model, endpoint=base_url, cached=True)
            else:
                content = [{"type": "text", "text": prompt}]
                def add_image(label, path):
                    mime, encoded = compress_and_encode_image(path, max_dim=1536, quality=90)
                    if not encoded:
                        raise ValueError(f"无法读取图片: {path}")
                    content.extend([{"type": "text", "text": label},
                                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "high"}}])
                for index, path in enumerate(refs):
                    add_image(f"STYLE DATASET REFERENCE {index}", path)
                for candidate in candidates:
                    add_image("CANDIDATE " + candidate["id"] + "\nRequested subject: " + candidate["record"]["test_prompt"], candidate["path"])
                if self.isInterruptionRequested():
                    raise ValueError("比较已取消")
                client = OpenAI(base_url=base_url, api_key=api_key, timeout=self.timeout)
                response = client.chat.completions.create(model=model, temperature=0,
                    response_format={"type": "json_object"}, max_completion_tokens=24000,
                    messages=[{"role": "user", "content": content}], timeout=self.timeout)
                assessments = json.loads(response.choices[0].message.content)["assessments"]
                result = calculate_ranking(candidates, assessments)
                result.update(input_hash=digest, assessments=assessments, model=model, endpoint=base_url, cached=False)
            if self.isInterruptionRequested():
                raise ValueError("比较已取消")
            # 保留请求执行期间其他工序已经保存的字段。
            with open(self.path, encoding="utf-8") as source:
                latest = json.load(source)
            if comparison_inputs(latest, model, base_url)[3] != digest:
                raise ValueError("比较期间训练数据已变更，请重新比较")
            latest["automatic_comparison"] = result
            result["report_path"] = write_comparison_report(latest, result, os.path.dirname(self.path))
            with open(self.path, "w", encoding="utf-8") as target:
                json.dump(latest, target, ensure_ascii=False, indent=2)
            with open(os.path.join(os.path.dirname(self.path), "automatic-comparison.json"), "w", encoding="utf-8") as target:
                json.dump(result, target, ensure_ascii=False, indent=2)
            self.completed.emit(result, "")
        except Exception as exc:
            self.completed.emit({}, str(exc))


def import_style_candidate(state, candidate_id, name, config_path, reference_dir, allow_excluded=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("画风名限 1–80 个英文、数字、连字符或下划线")
    comparison = state.get("automatic_comparison") or {}
    if comparison.get("version") != VERSION or comparison_inputs(state, comparison.get("model"), comparison.get("endpoint", ""))[3] != comparison.get("input_hash"):
        raise ValueError("比较结果已失效，请先重新比较")
    candidates = candidates_from_state(state)
    verified = calculate_ranking(candidates, comparison.get("assessments"))
    row = next((r for r in verified["rows"] if r["id"] == candidate_id), None)
    if not row or (not row.get("eligible") and not allow_excluded):
        raise ValueError("只能导入比较通过的已测试候选")
    candidate = next(c for c in candidates_from_state(state) if c["id"] == candidate_id)
    record = candidate["record"]
    package = record.get("prompt_variants") or {}
    if not package.get("gpt_image_prompt_valid"):
        raise ValueError("该版本缺少通过校验的 GPT 提示词包")
    entry = dict(package.get("style_entry") or {})
    if not entry.get("prompt") or not entry.get("prompt_gpt") or not entry.get("repaint_clauses"):
        raise ValueError("该版本缺少完整描述、GPT 短版或重绘条款")
    with open(config_path, encoding="utf-8") as source:
        styles = json.load(source)
    if name in styles:
        raise ValueError("该画风名已存在，请使用新名字，避免覆盖已有配置")
    ref = record["style_reference"]
    with open(ref, "rb") as source:
        ref_hash = hashlib.sha256(source.read()).hexdigest()[:12]
    os.makedirs(reference_dir, exist_ok=True)
    dest = os.path.abspath(os.path.join(reference_dir, name + "-" + ref_hash + os.path.splitext(ref)[1].lower()))
    if os.path.abspath(ref) != dest:
        shutil.copy2(ref, dest)
    entry.update(ref_image=dest, enabled=True, prompt_compressed=entry.get("prompt_compressed") or entry["prompt_gpt"],
                 motif_enabled=False, extraction_selection={"candidate": candidate_id, "version": comparison["version"], "input_hash": comparison["input_hash"],
                                                           "manual_override": not row["eligible"], "gates": row["gates"]})
    styles[name] = entry
    with open(config_path, "w", encoding="utf-8") as target:
        json.dump(styles, target, ensure_ascii=False, indent=4)
    return entry
