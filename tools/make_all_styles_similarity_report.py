# -*- coding: utf-8 -*-
"""Build the local, reference-backed all-style similarity report.

The report intentionally makes no generation requests. It reuses the 2026-09-25
same-source sweep and later local refinement outputs, then scores every enabled
reference-backed style with tests/calc_style_similarity.py.
"""
from __future__ import annotations

import base64
import html
import importlib.util
import io
import json
import os
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "conf" / "config-styles.json"
BASELINE_ROOT = ROOT / "data" / "20260925" / "analysis-gpt-image" / "style-sweep-20260925"
OUT_HTML = ROOT / "docs" / "gpt-image-tid-style" / "ALL-STYLES-SIMILARITY-20260927.html"
OUT_JSON = ROOT / "docs" / "gpt-image-tid-style" / "all-styles-similarity-20260927.json"


CURRENT_OVERRIDES = {
    "puracotte-style-v2": "data/test-result/20260927/expanded-horizontal/gpt-full/puracotte-style-v2-single-ref/quality-refine_014440-84b34b.jpg",
    "tinkle-style": "data/test-result/20260927/reference-candidate-ab/tinkle-style/new-ref/C_tinkle-style_style_ref_test_113640-d6bf89.jpg",
    "fuzichoco-v2": "data/test-result/20260927/expanded-horizontal/gpt-full/fuzichoco-v2/quality-refine_015322-70e7eb.jpg",
    "satou_kuuki-style-v2": "data/test-result/20260926/analysis-gpt-image/style-third-refine/satou_kuuki-style-v2-color/20260921-234549-2026-07-_output_105159_0_29346a-115219-396fa1-final-rp_115249-4c44e4.jpg",
    "waterink-style": "data/test-result/20260926/analysis-gpt-image/style-second-refine/waterink-style/post-preview/style-adjusted-v2.jpg",
    "iris-mix-style": "data/test-result/20260927/expanded-horizontal/gpt-full/iris-mix-style/20260926-232324-be2c69cd_output_013648_0_1c13f2-013658-45fc37-final-rp+tone+ink.png",
    "sheya-style": "data/test-result/20260926/analysis-gpt-image/style-third-refine/sheya-style-fresh-retry/20260921-234549-2026-07-_output_114843_0_b8a609-114853-b3c0bf-final-rp_114924-bca62e.jpg",
    "say-hana-v4": "data/test-result/20260926/analysis-gpt-image/style-third-refine/say-hana-v4/identity-correct-1_114036-2ae70e.jpg",
    "tid": "data/test-result/20260927/reference-candidate-ab/tid/new-ref/C_tid_style_ref_test_113726-0f6ff5.jpg",
    "ajicoma": "data/test-result/20260926/analysis-gpt-image/style-second-refine/ajicoma/post-preview/style-adjusted-v2.jpg",
    "inf-nikki-v1": "data/test-result/20260926/analysis-gpt-image/style-third-refine/inf-nikki-v1/20260921-234549-2026-07-_output_114448_0_eb2af3-114457-eecf2d-final-rp_114524-f4755a.jpg",
    "noir-aiart": "data/test-result/20260927/expanded-horizontal/gpt-full/noir-aiart/quality-refine_020221-78f623.jpg",
    "goto-p": "data/test-result/20260926/analysis-gpt-image/style-second-refine/goto-p/20260921-234549-2026-07-_output_213940_0_4c8d66-105933-eb94af-final-rp_110003-777207.jpg",
    "kishida-mel-style": "data/test-result/20260927/gpt-proportion-v2/kishida-mel-style/quality-refine_005624-53f6c7.jpg",
    "sakurapion-style": "data/test-result/20260927/gpt-proportion-v2/sakurapion-style/identity-correct-1_005531-d6e197.jpg",
    "cute-lingerie-wardrobe": "data/test-result/20260926/cute-lingerie-wardrobe/gpt-ref-v2/adult-study-gpt-ref-v2_output_200413_0_facaf0.png",
}

REFERENCE_ADVICE = {
    "puracotte-style-v2": ("观察", "单人图已消除多人歧义，但仍是半身近景；若后续以全身生成居多，建议补一张同作者全身环境图做 A/B。"),
    "tinkle-style": ("已替换", "旧图只覆盖脸和上身；新图覆盖全身、礼服、复杂环境与强光。需同步重做 Gemini 文本锚，避免旧蓝色调与新红紫参考冲突。"),
    "fuzichoco-v2": ("建议替换", "画框、旋转木马与群鸟占据主导，内容泄漏风险高；优先找单人全身、正常场景、无嵌套画框的作品。"),
    "dall-e-v2": ("建议替换", "近景正面胸像且设计特征过强；应换成全身角色＋环境，降低脸型、发饰和服装复制。"),
    "satou_kuuki-style-v2": ("建议替换", "近距离成人服装特写，姿势、材质和配色偏置很强；应换成非特写、完整四肢、同类光照材质的全身图。"),
    "waterink-style": ("保留", "人物身份弱、媒介和留白证据强，作为纯画法参考较安全。"),
    "iris-mix-style": ("条件保留", "该条目本身允许 Q 版比例，近景角色能提供脸部语言；若要生成全身 Q 版，可再补全身参考，不应套用写实比例规则。"),
    "shiratamaco-v2-style": ("建议替换", "半身猫耳角色身份过强且缺腿部/鞋袜/环境证据；建议同作者单人全身图。"),
    "sheya-style": ("观察", "动作和环境证据较丰富，但仍为坐姿中近景；可补一张站立全身图验证比例泛化。"),
    "say-hana-v4": ("保留", "单人全身、留白、植物和水面语言齐全，覆盖较均衡。"),
    "tid": ("已替换", "新图同时覆盖全身比例、海滩背景、强光与水面，A/B 提升明显且复制风险更低。"),
    "ajicoma": ("保留", "全身轮廓与留白清楚，适合该轻淡线稿语言；背景简单是风格事实而非缺陷。"),
    "inf-nikki-v1": ("建议替换", "当前是上半身 3D 特写且带 UID/界面痕迹；建议找无水印的全身 3D 场景截图。"),
    "noir-aiart": ("建议替换", "胸像、蓝蝶和武器主题过强；优先无文字无武器的单人全身环境图。"),
    "noir-art-style": ("临时采用", "97372077_p0 在现有下载中最均衡：脸部可辨、约四分之三身、有背景且无文字；但脚部不完整并带猫耳/饰品偏置。后续更适合拆成脸部画法参考＋全身环境参考。"),
    "goto-p": ("观察", "环境和水彩边缘有效，但人物覆盖不足全身；可补全身同作者作品后做 A/B。"),
    "komori-hikki-style": ("保留", "单人、近全身、服装层次和背景均有覆盖。"),
    "kishida-mel-style": ("建议替换", "热带水果和桌面道具过强、人物非全身；建议找单人全身、有适量背景但无强主题道具的作品。"),
    "sakurapion-style": ("条件保留", "全身比例证据很好，但纯白背景无法教环境笔触；若常生成场景图，建议换成或补充有背景的全身图。"),
    "cute-lingerie-wardrobe": ("保留", "这是服装结构预设而非普通画风；全身白底能降低场景泄漏，符合该条目的用途。"),
}

REFERENCE_AB = {
    "tinkle-style": {
        "old_ref": "data/style-ref/tinkle-test.png",
        "new_ref": "data/style-ref/tinkle-fullbody-candidate.png",
        "old_output": "data/test-result/20260927/reference-candidate-ab/tinkle-style/old-ref/C_tinkle-style_style_ref_test_113614-708f18.jpg",
        "new_output": "data/test-result/20260927/reference-candidate-ab/tinkle-style/new-ref/C_tinkle-style_style_ref_test_113640-d6bf89.jpg",
    },
    "tid": {
        "old_ref": "data/style-ref/tid.png",
        "new_ref": "data/style-ref/tid-fullbody-background-candidate.jpg",
        "old_output": "data/test-result/20260927/reference-candidate-ab/tid/old-ref/C_tid_style_ref_test_113658-6487b2.jpg",
        "new_output": "data/test-result/20260927/reference-candidate-ab/tid/new-ref/C_tid_style_ref_test_113726-0f6ff5.jpg",
    },
}


def load_metric_module():
    path = ROOT / "tests" / "calc_style_similarity.py"
    spec = importlib.util.spec_from_file_location("style_similarity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def pick_pipeline_final(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    for marker in ("identity-correct", "quality-refine", "final-rp"):
        matches = sorted((p for p in images if marker in p.name), key=lambda p: p.stat().st_mtime)
        if matches:
            return matches[-1]
    return sorted(images, key=lambda p: p.stat().st_mtime)[-1] if images else None


def thumbnail_data(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((420, 420), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def rel_link(path: Path) -> str:
    return Path(os.path.relpath(path, OUT_HTML.parent)).as_posix()


def score_one(metric, ref: Path, image: Path):
    return metric.evaluate(str(ref), [str(image)], include_clip=False)[0]


def main() -> int:
    styles = json.loads(CONFIG.read_text(encoding="utf-8"))
    enabled = [(name, entry) for name, entry in styles.items() if entry.get("enabled", True)]
    metric = load_metric_module()
    rows = []
    skipped = []
    for name, entry in enabled:
        ref_value = entry.get("ref_image")
        if not ref_value:
            skipped.append({"style": name, "reason": "无参考图，参考相似度不适用"})
            continue
        ref = Path(ref_value)
        if not ref.is_absolute():
            ref = ROOT / ref
        if not ref.exists():
            skipped.append({"style": name, "reason": f"参考图不存在：{ref}"})
            continue
        baseline = pick_pipeline_final(BASELINE_ROOT / name)
        current = ROOT / CURRENT_OVERRIDES[name] if name in CURRENT_OVERRIDES else baseline
        if current is None or not current.exists():
            skipped.append({"style": name, "reason": "没有可复用的本地生成结果"})
            continue
        current_score = score_one(metric, ref, current)
        baseline_score = score_one(metric, ref, baseline) if baseline and baseline.exists() else None
        status, advice = REFERENCE_ADVICE.get(name, ("观察", "尚未人工复核参考图覆盖。"))
        rows.append({
            "style": name,
            "reference": str(ref),
            "baseline": None if baseline is None else str(baseline),
            "current": str(current),
            "baseline_score": None if baseline_score is None else baseline_score["style_score"],
            "score": current_score["style_score"],
            "gram": current_score["gram_style"],
            "palette": current_score["palette"],
            "render": current_score["render"],
            "copy_risk": current_score["copy_risk"],
            "delta": None if baseline_score is None else round(current_score["style_score"] - baseline_score["style_score"], 4),
            "reference_status": status,
            "reference_advice": advice,
        })
    rows.sort(key=lambda row: row["score"], reverse=True)
    reference_ab = []
    for style, item in REFERENCE_AB.items():
        paths = {key: ROOT / value for key, value in item.items()}
        old_against_old = score_one(metric, paths["old_ref"], paths["old_output"])
        old_against_new = score_one(metric, paths["new_ref"], paths["old_output"])
        new_against_old = score_one(metric, paths["old_ref"], paths["new_output"])
        new_against_new = score_one(metric, paths["new_ref"], paths["new_output"])
        old_mean = round((old_against_old["style_score"] + old_against_new["style_score"]) / 2, 4)
        new_mean = round((new_against_old["style_score"] + new_against_new["style_score"]) / 2, 4)
        reference_ab.append({
            "style": style, **{key: str(value) for key, value in paths.items()},
            "old_vs_old": old_against_old["style_score"], "old_vs_new": old_against_new["style_score"],
            "new_vs_old": new_against_old["style_score"], "new_vs_new": new_against_new["style_score"],
            "old_mean": old_mean, "new_mean": new_mean, "delta": round(new_mean - old_mean, 4),
            "old_copy_risk_max": max(old_against_old["copy_risk"], old_against_new["copy_risk"]),
            "new_copy_risk_max": max(new_against_old["copy_risk"], new_against_new["copy_risk"]),
        })
    payload = {"metric": "Gram 55% + HSV 20% + render signature 25%; CLIP excluded",
               "reference_ab": reference_ab, "rows": rows, "skipped": skipped}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    scores = [r["score"] for r in rows]
    high = sum(s >= 0.82 for s in scores)
    medium = sum(0.74 <= s < 0.82 for s in scores)
    low = sum(s < 0.74 for s in scores)
    body = []
    for rank, row in enumerate(rows, 1):
        current = Path(row["current"])
        ref = Path(row["reference"])
        delta = "—" if row["delta"] is None else f"{row['delta']:+.4f}"
        baseline = "—" if row["baseline_score"] is None else f"{row['baseline_score']:.4f}"
        risk_class = "bad" if row["copy_risk"] >= 0.18 else ("warn" if row["copy_risk"] >= 0.08 else "good")
        body.append(f"""
        <tr>
          <td>{rank}</td><td><b>{html.escape(row['style'])}</b><br><span class='tag'>{html.escape(row['reference_status'])}</span></td>
          <td><a href='{rel_link(ref)}'><img src='{thumbnail_data(ref)}'></a></td>
          <td><a href='{rel_link(current)}'><img src='{thumbnail_data(current)}'></a></td>
          <td class='num score'>{row['score']:.4f}</td><td class='num'>{baseline}</td><td class='num'>{delta}</td>
          <td class='num'>{row['gram']:.4f}</td><td class='num'>{row['palette']:.4f}</td><td class='num'>{row['render']:.4f}</td>
          <td class='num {risk_class}'>{row['copy_risk']:.4f}</td>
          <td class='advice'>{html.escape(row['reference_advice'])}<br><a href='{rel_link(current)}'>打开生成原图</a> · <a href='{rel_link(ref)}'>打开参考图</a></td>
        </tr>""")
    skipped_html = "".join(f"<li><b>{html.escape(x['style'])}</b>：{html.escape(x['reason'])}</li>" for x in skipped)
    ab_html = []
    for item in reference_ab:
        ab_html.append(f"""<tr><td><b>{html.escape(item['style'])}</b></td>
        <td><a href='{rel_link(Path(item['old_output']))}'><img src='{thumbnail_data(Path(item['old_output']))}'></a></td>
        <td><a href='{rel_link(Path(item['new_output']))}'><img src='{thumbnail_data(Path(item['new_output']))}'></a></td>
        <td class='num'>{item['old_vs_old']:.4f}</td><td class='num'>{item['old_vs_new']:.4f}</td><td class='num score'>{item['old_mean']:.4f}</td>
        <td class='num'>{item['new_vs_old']:.4f}</td><td class='num'>{item['new_vs_new']:.4f}</td><td class='num score'>{item['new_mean']:.4f}</td>
        <td class='num'>{item['delta']:+.4f}</td><td class='num'>{item['old_copy_risk_max']:.4f} → {item['new_copy_risk_max']:.4f}</td></tr>""")
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>启用画风横向相似度评估 2026-09-27</title>
<style>
body{{font:15px/1.55 system-ui,'Microsoft YaHei',sans-serif;margin:0;background:#11151c;color:#e8edf4}}main{{max-width:1680px;margin:auto;padding:28px}}
h1,h2{{color:#fff}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{background:#1c2330;border:1px solid #344054;border-radius:10px;padding:12px 18px}}
table{{width:100%;border-collapse:collapse;background:#171d27}}th{{position:sticky;top:0;background:#253044;z-index:2}}th,td{{border:1px solid #344054;padding:8px;vertical-align:top}}img{{width:150px;height:150px;object-fit:contain;background:#0b0e13}}
.num{{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}}.score{{font-size:18px;font-weight:700;color:#9fe3b1}}.good{{color:#8ee59f}}.warn{{color:#ffd479}}.bad{{color:#ff8d8d}}.advice{{min-width:270px;max-width:390px}}a{{color:#8ecbff}}.tag{{font-size:12px;color:#c9d4e5;background:#303c50;border-radius:12px;padding:2px 7px}}code{{color:#b9ddff}}
</style></head><body><main>
<h1>启用画风横向相似度评估（2026-09-27）</h1>
<p>覆盖配置中全部启用条目：有参考图的条目进行本地评分；默认无附加与无参考图条目明确列为不适用。优先复用 9 月 25 日同素材全画风测试及 26–27 日后续结果，没有为了凑齐表格重新调用生图接口。</p>
<div class='cards'><div class='card'><b>{len(rows)}</b><br>可评分画风</div><div class='card'><b>{high}</b><br>≥ 0.82</div><div class='card'><b>{medium}</b><br>0.74–0.82</div><div class='card'><b>{low}</b><br>&lt; 0.74</div><div class='card'><b>{len(skipped)}</b><br>不适用/缺参考</div></div>
<h2>口径</h2><p>总分 = ResNet 中浅层 Gram 风格统计 55% + HSV 色彩 20% + 线条/边缘/明暗渲染签名 25%。CLIP 不参与排名；copy-risk 单列，避免把复制参考内容误当作风格提升。不同素材的分数可用于故障筛查和分层，不能解释为绝对审美排名。</p>
<h2>tinkle / TID 参考图 A/B</h2><p>同一 Gemini 参考优先提示分别换旧图和新图。为避免“拿各自产物只对各自参考图评分”的偏置，每张产物同时对旧、新两张参考评分，再取均值。tinkle 是小幅提升；TID 是明显提升且复制风险下降。</p>
<table><thead><tr><th>画风</th><th>旧参考产物</th><th>新参考产物</th><th>旧产物→旧参</th><th>旧产物→新参</th><th>旧均值</th><th>新产物→旧参</th><th>新产物→新参</th><th>新均值</th><th>提升</th><th>最大复制风险</th></tr></thead><tbody>{''.join(ab_html)}</tbody></table>
<h2>结果</h2><table><thead><tr><th>#</th><th>画风</th><th>当前参考图</th><th>代表生成图</th><th>总分</th><th>9/25基线</th><th>变化</th><th>Gram</th><th>色彩</th><th>渲染</th><th>复制风险</th><th>参考图建议</th></tr></thead><tbody>{''.join(body)}</tbody></table>
<h2>不适用条目</h2><ul>{skipped_html}</ul>
<h2>解释与下一步</h2><ol><li>先处理低分且参考图覆盖差的条目；换参考图时必须同时重做该条目的 Gemini 专用文本锚，不能只换路径。</li><li>每个候选参考图用固定提示至少跑 3 次，比较中位数和最差值；单次提升容易被 Gemini 随机性误导。</li><li>建议逐步支持“画法近景 + 全身环境”双参考：第一张约束脸、眼、发，第二张约束全身比例、服装与环境笔触；请求中明确角色分工并持续监控 copy-risk。</li><li>分数低但视觉合理的媒介类画风（如水墨、Q版）应按各自目标复核，不用统一写实人体或背景密度标准。</li></ol>
<p>原始数据：<a href='{rel_link(OUT_JSON)}'>{OUT_JSON.name}</a></p></main></body></html>"""
    OUT_HTML.write_text(document, encoding="utf-8")
    print(f"wrote {OUT_HTML}")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
