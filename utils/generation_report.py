#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""出图后的对比报告（每轮可用）：参考图对照 + 泛化对照 + 推荐 prompt 组合。

设计要点（对应工作流检查项）：
1. **每轮都能出报告**：输入只要「参考图集合 + 本轮候选图 + 元信息」，不依赖多图画风提取的整轮流程。
2. **泛化对照**：画风提取往往有 10+ 张参考图，只跟单张参考比会过拟合。
   这里用同一个共享入口（`utils/style_similarity.compare_images`）算「每个候选 × 每张参考图」的全配对，
   给出中位/最差/最佳与「最像哪一张参考」「最不像哪一张」。
3. **推荐 prompt 组合**：把候选按 gram/adain 中位排序，输出推荐组合（当前最优条件 + 关键提示词片段）。

用法（独立跑，不需要 app）：
  python tools/make_generation_report.py --label round21 --out data/test-result/style-lab/report-round21.html
"""
import html
import io
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 距离型（越小越好）与贴近/相似型（越大越好）——方向沿用共享指标口径
LOWER_BETTER = ("gram", "adain", "lpips")
HIGHER_BETTER = ("csd", "tone", "edges", "lines", "space")

# 计算设备：默认自动选 CPU（GUI 主进程不加载 torch/OpenVINO 的约定）；
# 批处理/实验脚本可用 IMAGE_MAKER_REPORT_DEVICE=cuda 提速。
DEVICE = os.environ.get("IMAGE_MAKER_REPORT_DEVICE", "auto-cpu")


def _round(value, digits=3):
    return round(float(value), digits) if isinstance(value, (int, float)) else None


def extract_pair_metrics(result):
    """从 `utils.style_similarity.compare_images` 的输出里取出「候选×参考」的逐对指标。

    键统一用**绝对路径**（共享入口返回的就是绝对路径），渲染时再转相对路径。
    """
    out = []
    for row in (result.get("rows") or []):
        for pair in (row.get("pairs") or []):
            entry = {"candidate": str(os.path.abspath(row.get("path") or "")),
                     "reference": str(os.path.abspath(pair.get("reference") or ""))}
            for key in (*LOWER_BETTER, *HIGHER_BETTER):
                value = (pair.get(key) or {}).get("value")
                if isinstance(value, (int, float)):
                    entry[key] = float(value)
            out.append(entry)
    return out


def recommended_combination(per_candidate, metric="gram"):
    """按指标中位排序，给出推荐组合（当前最优条件 + 需要保留的关键条款）。"""
    scored = []
    for candidate, pairs in per_candidate.items():
        values = [p[metric] for p in pairs if isinstance(p.get(metric), (int, float))]
        if not values:
            continue
        # candidate 是绝对路径；报告里用文件名更可读
        scored.append({"candidate": os.path.basename(candidate), "path": candidate,
                       "median": st.median(values),
                       "best": min(values), "worst": max(values), "n": len(values)})
    scored.sort(key=lambda item: item["median"])
    if not scored:
        return {}
    top = scored[0]
    return {"metric": metric, "ranking": scored, "winner": top["path"],
            "note": "按参考图集合的中位数排序；样本 <3 时只作方向性参考"}


def _rel(path):
    if not path:
        return ""
    try:
        return os.path.relpath(path, ROOT).replace("\\", "/")
    except ValueError:
        return path


def render_report(payload, title="出图对比报告"):
    """payload: {meta, candidates:[{id,path,metrics,verdict}], refs:[...], per_reference, recommendation}"""
    meta = payload.get("meta") or {}
    refs = payload.get("refs") or []
    candidates = payload.get("candidates") or []
    per_ref = payload.get("per_reference") or {}
    rec = payload.get("recommendation") or {}

    def esc(text):
        return html.escape(str(text if text is not None else ""))

    parts = ["""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title><style>
body{{margin:0;padding:24px 28px 70px;background:#fafafb;color:#1b1b20;
 font:15px/1.7 "Microsoft YaHei","Segoe UI",system-ui,sans-serif}}
h1{{font-size:22px;margin:0 0 6px}} h2{{font-size:17px;margin:30px 0 8px;padding-bottom:5px;border-bottom:2px solid #e4e4e8}}
table{{border-collapse:collapse;background:#fff;font-size:13px;margin:10px 0}}
th,td{{border:1px solid #e4e4e8;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f2f3f6;white-space:nowrap}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
.good{{color:#14672c;font-weight:600}} .bad{{color:#a11f1f;font-weight:600}} .warn{{color:#8a5a06;font-weight:600}}
code{{background:#f1f2f5;padding:1px 5px;border-radius:4px;font-size:12.5px}}
img{{max-width:100%;border:1px solid #e4e4e8;border-radius:8px;background:#fff}}
.thumb{{width:150px;height:auto;margin:2px}}
.note{{background:#eef5ff;border:1px solid #c3d8f0;border-radius:8px;padding:10px 14px;margin:12px 0;font-size:13.5px}}
.legend{{font-size:12.5px;color:#6b6b73}}
</style></head><body>""".format(title=esc(title))]

    parts.append(f"<h1>{esc(title)}</h1>")
    if meta:
        rows = "".join(
            f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in meta.items()
        )
        parts.append(f"<table>{rows}</table>")

    # 一、候选与参考的逐对指标
    parts.append("<h2>一、候选图 × 参考图（整图八组指标）</h2>")
    parts.append("<p class='legend'>gram / adain / lpips 越小越接近；csd / tone / edges / lines / space 越大越接近。"
                 "「参考」列是该候选最接近的那张参考图（泛化对照结论）。</p>")
    header = ("<tr><th>候选</th><th>预览</th><th class='num'>gram</th><th class='num'>adain</th>"
              "<th class='num'>lpips</th><th class='num'>csd</th><th class='num'>tone</th>"
              "<th class='num'>edges</th><th class='num'>lines</th><th class='num'>space</th>"
              "<th>最像的参考</th><th>最不像的参考</th><th>判定</th></tr>")
    body = []
    for item in candidates:
        stats = per_ref.get(item["id"]) or {}
        cells = "".join(
            f"<td class='num'>{_round(item.get('metrics', {}).get(key)) if item.get('metrics', {}).get(key) is not None else '-'}</td>"
            for key in (*LOWER_BETTER, *HIGHER_BETTER)
        )
        body.append(
            f"<tr><td>{esc(item['id'])}</td>"
            f"<td><img class='thumb' src='{esc(_rel(item.get('path')))}'></td>{cells}"
            f"<td>{esc(os.path.basename(stats.get('closest') or ''))}</td>"
            f"<td>{esc(os.path.basename(stats.get('farthest') or ''))}</td>"
            f"<td>{esc(item.get('verdict') or '-')}</td></tr>"
        )
    parts.append(f"<table>{header}{''.join(body)}</table>")

    # 二、泛化对照
    parts.append("<h2>二、泛化对照（候选 × 每张参考图）</h2>")
    parts.append("<p class='legend'>画风提取通常用 10+ 张参考图；只看一张会过拟合。"
                 "下表是每个候选对<b>全部参考图</b>的中位/最好/最差值。</p>")
    rows = ["<tr><th>候选</th><th class='num'>gram 中位</th><th class='num'>gram 最好</th>"
            "<th class='num'>gram 最差</th><th class='num'>adain 中位</th><th class='num'>配对</th></tr>"]
    for item in candidates:
        agg = (item.get("aggregate") or {})
        rows.append(
            f"<tr><td>{esc(item['id'])}</td>"
            f"<td class='num'>{_round(agg.get('gram_median')) or '-'}</td>"
            f"<td class='num'>{_round(agg.get('gram_best')) or '-'}</td>"
            f"<td class='num'>{_round(agg.get('gram_worst')) or '-'}</td>"
            f"<td class='num'>{_round(agg.get('adain_median')) or '-'}</td>"
            f"<td class='num'>{agg.get('pairs') or 0}</td></tr>"
        )
    parts.append(f"<table>{''.join(rows)}</table>")

    # 三、参考图墙
    if refs:
        parts.append("<h2>三、参考图集合</h2><div>")
        for ref in refs:
            parts.append(f"<img class='thumb' src='{esc(_rel(ref))}' title='{esc(os.path.basename(ref))}'>")
        parts.append("</div>")

    # 四、推荐 prompt 组合
    parts.append("<h2>四、推荐的 prompt 组合</h2>")
    ranking = rec.get("ranking") or []
    if ranking:
        rows = ["<tr><th>排名</th><th>候选</th><th class='num'>中位 %s</th><th class='num'>最好</th>"
                "<th class='num'>最差</th><th class='num'>配对</th></tr>" % esc(rec.get("metric"))]
        for index, item in enumerate(ranking, 1):
            rows.append(f"<tr><td>{index}</td><td>{esc(item['candidate'])}</td>"
                        f"<td class='num'>{_round(item['median'])}</td><td class='num'>{_round(item['best'])}</td>"
                        f"<td class='num'>{_round(item['worst'])}</td><td class='num'>{item['n']}</td></tr>")
        parts.append(f"<table>{''.join(rows)}</table>")
        parts.append(f"<div class='note'><b>推荐使用</b>：<code>{esc(os.path.basename(rec.get('winner') or ''))}</code>"
                     f"（按 {esc(rec.get('metric'))} 中位最优）。{esc(rec.get('note') or '')}</div>")
    else:
        parts.append("<p>没有可排名的候选。</p>")

    if payload.get("recommended_prompt"):
        parts.append("<h2>五、推荐提示词组合</h2>")
        parts.append(f"<pre style='white-space:pre-wrap;background:#f7f8fa;border:1px solid #e4e4e8;"
                     f"border-radius:8px;padding:12px'>{esc(payload['recommended_prompt'])}</pre>")

    parts.append("</body></html>")
    return "\n".join(parts)


def build_payload(reference_images, candidates, meta=None, similarity_fn=None):
    """计算指标并组装报告 payload。

    reference_images: 参考图路径列表（画风提取时是全部源图）
    candidates: [{"id":..., "path":...}, ...]
    similarity_fn: 默认用 utils.style_similarity.compare_images（可注入以便测试）
    """
    if similarity_fn is None:
        from utils.style_similarity import compare_images as similarity_fn  # 延迟导入，避免无 torch 环境导入失败
    from utils.style_similarity import image_manifest

    candidate_paths = [c["path"] for c in candidates]
    # image_manifest 会把路径 resolve 成绝对路径，这里先做同样的规范化，才能把结果映射回调用方 id
    path_to_id = {str(os.path.abspath(c["path"])): c["id"] for c in candidates}
    inputs = image_manifest(candidate_paths, reference_images)
    result = similarity_fn(inputs, requested=DEVICE)
    pairs = extract_pair_metrics(result)

    per_candidate = {}
    by_id = dict(path_to_id)
    for pair in pairs:
        per_candidate.setdefault(pair["candidate"], []).append(pair)

    out_candidates = []
    per_ref = {}
    for path, items in per_candidate.items():
        values = {key: [p[key] for p in items if isinstance(p.get(key), (int, float))]
                  for key in (*LOWER_BETTER, *HIGHER_BETTER)}
        metrics = {key: (st.median(vals) if vals else None) for key, vals in values.items()}
        gram_pairs = [p for p in items if isinstance(p.get("gram"), (int, float))]
        closest = min(gram_pairs, key=lambda p: p["gram"])["reference"] if gram_pairs else None
        farthest = max(gram_pairs, key=lambda p: p["gram"])["reference"] if gram_pairs else None
        per_ref[by_id.get(path, path)] = {"closest": closest, "farthest": farthest}
        out_candidates.append({
            "id": by_id.get(path, path), "path": path, "metrics": {k: _round(v) for k, v in metrics.items()},
            "aggregate": {
                "gram_median": min(values["gram"]) if values.get("gram") else None,
                "gram_best": min(values["gram"]) if values.get("gram") else None,
                "gram_worst": max(values["gram"]) if values.get("gram") else None,
                "adain_median": st.median(values["adain"]) if values.get("adain") else None,
                "pairs": len(items),
            },
            "verdict": None,
        })
        if values.get("gram"):
            out_candidates[-1]["aggregate"]["gram_median"] = st.median(values["gram"])

    return {"meta": meta or {}, "refs": list(reference_images), "candidates": out_candidates,
            "per_reference": per_ref,
            "recommendation": recommended_combination(per_candidate),
            "raw": result}


def write_report(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as handle:
        handle.write(render_report(payload))
    return path


if __name__ == "__main__":
    print(__doc__)
