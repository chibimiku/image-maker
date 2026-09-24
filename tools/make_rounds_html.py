# -*- coding: utf-8 -*-
"""生成「本轮迭代 + 迁移测试」单文件 HTML 画廊（图内嵌 + 指标 + Sol 结论）。"""
import base64
import glob
import io
import os
import sys

from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "tests"))
sys.stdout.reconfigure(errors="replace")
import style_render_metrics as m  # noqa: E402

OUT = os.path.join(BASE, "docs", "gpt-image-tid-style", "rounds-20260923.html")


def uri(path, max_edge=560, quality=74):
    if not path or not os.path.isfile(path):
        return None
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if max(w, h) > max_edge:
        s = max_edge / max(w, h)
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def metrics(path):
    if not os.path.isfile(path):
        return "-"
    import cv2
    import numpy as np
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * 900 / h)), 900), interpolation=cv2.INTER_AREA)
    tmp = os.path.join(BASE, "data", "_g.png")
    cv2.imwrite(tmp, small)
    cur = m.analyze(tmp)
    os.remove(tmp)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
    return (f"亮度 {gray.mean():.0f} / 饱和 {sat:.0f} / 段长 {cur['line_avg_len']:.0f} / "
            f"端点 {cur['line_endpoint_density']:.1f} / 碎线 {cur['line_frag_ratio']:.3f}")


def card(title, path, note=""):
    src = uri(path)
    if not src:
        return ""
    return (f'<figure class="card"><img loading="lazy" src="{src}">'
            f'<figcaption><b>{title}</b><br><span class="fname">{note}</span></figcaption></figure>')


DAY = os.path.join(BASE, "data", "20260923")
ITEMS = [
    ("迁移 T2d（精炼管线）：goto-p × 洛丽塔店铺", "goto-p 参考", os.path.join(BASE, "data", "style-ref", "goto-p.jpg"),
     os.path.join(DAY, "20260923-030133-天气好的时候穿上_output_042726_0_ba7e22-042726-3f5013-final-rp+sline50+local-subject_no_face+tone.png"),
     "Sol：无拼接/重复结构/硬边，goto-p 干净细线成立；背景略偏浓、鞋被画框裁到"),
    ("迁移 T3e（精炼管线）：waterink × 舞姬", "waterink 参考", os.path.join(BASE, "data", "style-ref", "water-ink.jpg"),
     os.path.join(DAY, "20260923-030315-00000-33_output_043854_0_a39cc7-043854-1ef6b9-final-rp+sline50+local-subject_no_face+tone.png"),
     "Sol：无拼接/重复/半截肢体，固有特征一致，水彩墨染成立；腰带扣件可读"),
    ("迁移 T1c（修正贴回限制后）：ajicoma × Lolita 屋顶", "ajicoma 参考", os.path.join(BASE, "data", "style-ref", "ajicoma.jpg"),
     os.path.join(DAY, "20260923-025808-Lolita｜🌟_output_042122_0_2c11e9-042122-1f53ba-final-rp+sline50+local-subject_no_face+tone.png"),
     "Sol：矩形带/硬边消失，天空屋顶远山连续，ajicoma 细线成立"),
    ("迁移 T1b（修复前，可见矩形拼接带）", "ajicoma 参考", os.path.join(BASE, "data", "style-ref", "ajicoma.jpg"),
     os.path.join(BASE, "docs", "gpt-image-tid-style", "sol-cases", "round-T1b-full.png"),
     "Sol：无拼接/重复结构、固有特征一致、ajicoma 细线成立；我另发现 waist/thigh 区域有矩形拼接带（已修）"),
    ("★★★★ R5：主用例阶段最佳（高光再压 + 肤色暖化 + 饱和再降）", "tinkle 参考",
     os.path.join(BASE, "data", "style-ref", "tinkle-test.png"),
     os.path.join(BASE, "docs", "gpt-image-tid-style", "sol-cases", "r5-高光080-肤暖04-饱和093.png"),
     "亮度157.6（照片155.3）、饱和78.8、局部RMS 45.67；Sol：白裙层次优于 R4b、无拼接；白裁剪瓶颈在上游"),
    ("★★★ R4b：小硬件紧框 + 饱和度自动匹配（当前最佳）", "tinkle 参考",
     os.path.join(BASE, "data", "style-ref", "tinkle-test.png"),
     os.path.join(BASE, "docs", "gpt-image-tid-style", "sol-cases", "satmatch-1.0-1.20.png"),
     "亮度161.5（照片155.3）、饱和78.3（照片69.1，上一版 105.4 过饱和已修）、无拼接；Sol：荧光感消除，小硬件仍需再修"),
    ("★★ R3 完整链路：内容参考图 + 细节 4K + 色调校准（当前最佳）", "tinkle 参考",
     os.path.join(BASE, "data", "style-ref", "tinkle-test.png"),
     os.path.join(DAY, "20260921-234549-2026-07-_output_034325_0_2f13a3-034325-041352-final-rp+sline50+local-subject_no_face+tone.png"),
     "亮度158.2（照片155.3）、饱和82.3、局部RMS 49.0（R2T原样32.5）；Sol：固有特征保真、无拼接；仍缺「可数有扣件」的小硬件结构"),
    ("★ R2T+tone：tinkle × 主素材（第 2 轮，首图挂内容参考图 + 色调校准）", "tinkle 参考",
     os.path.join(BASE, "data", "style-ref", "tinkle-test.png"),
     os.path.join(BASE, "docs", "gpt-image-tid-style", "sol-cases", "tonecheck-118.png"),
     "亮度176→162（照片155）、饱和70→94、高光裁剪0.070→0.064、局部RMS 32.5→41.6；Sol：发白改善、无拼接；鞋带/鞋扣仍被高光吃掉"),
    ("R2T：tinkle × 主素材（首图加挂内容参考图，未加色调校准）", "tinkle 参考",
     os.path.join(BASE, "data", "style-ref", "tinkle-test.png"),
     os.path.join(DAY, "20260921-234549-2026-07-_output_033415_0_1f6acf-033415-80902d-final-rp+sline50+local-subject_no_face.png"),
     "发色回到照片暖棕（固有特征修复在主用例同样生效）；Sol 指出白色材质高光把结构冲淡"),
    ("① tinkle × 主素材（第 4 轮，无内容参考图）", "tinkle 参考", os.path.join(BASE, "data", "style-ref", "tinkle-test.png"),
     os.path.join(DAY, "20260921-234549-2026-07-_output_024958_0_afd8ee-024958-0ab2a1-final-rp+sline50+local-subject_no_face.png"),
     "通道：首图带画风条款 → 重绘(源图+tinkle 参考+18 条条款) → 结构线 0.28/0.18/1px → 局部(整身·保脸 + 鞋 + 束腰 + 袜带，均限制在主体遮罩内)"),
    ("② 迁移 T1：ajicoma × Lolita 屋顶", "ajicoma 参考", os.path.join(BASE, "data", "style-ref", "ajicoma.jpg"),
     os.path.join(DAY, "20260923-025808-Lolita｜🌟_output_030403_0_5138c6-030403-652b9e-final-rp+sline50+local-subject_no_face.png"),
     "同一套工序与参数，只换画风与输入；无拼接/重复结构"),
    ("③ 迁移 T2：goto-p × 洛丽塔店铺", "goto-p 参考", os.path.join(BASE, "data", "style-ref", "goto-p.jpg"),
     os.path.join(DAY, "20260923-030133-天气好的时候穿上_output_030736_0_247b98-030737-e3d194-final-rp+sline50+local-subject_no_face.png"),
     "Sol：配色匹配、但线条偏软（goto-p 以干净细线著称）"),
    ("④ 迁移 T3b：waterink × 粉发舞姬（只加文字约束）", "waterink 参考", os.path.join(BASE, "data", "style-ref", "water-ink.jpg"),
     os.path.join(DAY, "20260923-030315-00000-33_output_032121_0_3438f9-032121-f9fef6-final-rp+sline50+local-subject_no_face.png"),
     "仅加「固有特征」文字条款：发色由暗navy→银白，但仍未回到玫瑰粉（头发区中位色相 140=偏紫）"),
    ("⑤ **迁移 T4：waterink × 粉发舞姬（首图挂内容参考图）**", "waterink 参考", os.path.join(BASE, "data", "style-ref", "water-ink.jpg"),
     os.path.join(DAY, "20260923-030315-00000-33_output_032727_0_0fad92-032727-f50f9a-final-rp+sline50+local-subject_no_face.png"),
     "**修好了固有特征漂移**：头发区中位色相 140→8（暖玫瑰，源图 171）、饱和 14→22；Sol：发色/瞳色/服装配色基本保住、水彩语言仍成立、无拼接"),
]

# 分组：最终产物（当前精炼管线的端到端输出）/ 过程版本 / 色调变体（基于中间态的后处理变体，不是端到端）
FINAL_TAGS = ("迁移 T2d", "迁移 T3e", "迁移 T1c", "R3 完整链路", "R2T：")
VARIANT_TAGS = ("R5", "R4b", "R2T+tone")


def group_of(title):
    if any(t in title for t in FINAL_TAGS):
        return "最终产物"
    if any(t in title for t in VARIANT_TAGS):
        return "色调变体（非端到端）"
    return "过程版本"


def build(rows, cards, kinds):
    out_rows, out_cards = [], []
    for title, ref_label, ref_path, final_path, note in ITEMS:
        if group_of(title) not in kinds:
            continue
        r = uri(ref_path, max_edge=380)
        ref_html = f'<img class="ref" src="{r}">' if r else "（无参考图）"
        tag = {"最终产物": "【最终产物】", "色调变体（非端到端）": "【色调变体】", "过程版本": "【过程版本】"}[group_of(title)]
        out_rows.append(f"<tr><td>{tag} {title}</td><td>{ref_html}</td><td>{metrics(final_path)}</td>"
                        f"<td>{note}</td></tr>")
        out_cards.append(card(f"{tag} {title}", final_path, os.path.basename(final_path)))
    return ("<table><thead><tr><th>用例</th><th>画风参考</th><th>产物指标</th><th>说明 / Sol 结论</th></tr></thead><tbody>"
            + "".join(out_rows) + "</tbody></table>", "".join(out_cards))


final_table, final_cards = build(ITEMS, None, {"最终产物"})
other_table, other_cards = build(ITEMS, None, {"色调变体（非端到端）", "过程版本"})

html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>迭代与迁移测试 · 2026-09-23</title><style>
 body {{ font: 14px/1.7 system-ui,"Segoe UI",sans-serif; margin:0; padding:24px 30px 80px; background:#fafafa; color:#222; }}
 h1 {{ font-size:22px; }} h2 {{ font-size:16px; margin-top:26px; border-top:1px solid #ddd; padding-top:12px; }}
 table {{ border-collapse:collapse; font-size:12.5px; margin:10px 0; }} th,td {{ border:1px solid #ddd; padding:5px 9px; vertical-align:top; }}
 th {{ background:#f0f0f0; }} img.ref {{ max-width:110px; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:12px; }}
 .card {{ margin:0; background:#fff; border:1px solid #e3e3e3; border-radius:8px; overflow:hidden; }}
 .card img {{ width:100%; display:block; }} .card figcaption {{ padding:7px 9px; font-size:12px; }}
 .fname {{ color:#888; font-size:10.5px; word-break:break-all; }}
 ul {{ margin:6px 0 0 18px; }} code {{ background:#f0f0f0; padding:1px 4px; border-radius:3px; }}
 @media (prefers-color-scheme: dark) {{ body {{ background:#1b1b1b; color:#e9e9e9; }} .card {{ background:#262626; border-color:#3a3a3a; }} th {{ background:#333; }} code {{ background:#333; }} }}
</style></head><body>
<h1>迭代 + 迁移测试（2026-09-23）</h1>
<p>本文件由 <code>tools/make_rounds_html.py</code> 生成；指标归一到 900px 高；每轮产物都由我与 <code>gpt-5.6-sol</code> 双方看图复核。</p>

<h2>① 最终产物（当前精炼管线的端到端输出，只有这几张）</h2>
{final_table}
<div class="grid">{final_cards}</div>

<h2>② 过程版本 / 色调变体（不是最终产物，仅用于对比与回看）</h2>
{other_table}
<div class="grid">{other_cards}</div>

<h2>本轮修掉的问题</h2>
<ul>
 <li><b>下半截拼接错误</b>（多出椅子腿、半截脚、水平接缝）：三处根因全部修掉——
   ① 局部重绘不再把裁切区硬拉成官方比例，改为<b>补边到该比例、回来再裁掉</b>；
   ② 贴回时<b>限制在人物主体遮罩内</b>（GrabCut 最大连通域），背景/家具保持原像素（实测背景改动比例 0.656 → 0.006）；
   ③ 每次贴回都有 <code>[QA]</code> 日志（改动像素占比 + 补边量）。</li>
 <li><b>线条过粗/发黑</b>：结构线掩膜默认 1px 骨架化 + <code>darken 0.18</code> + 高频细节区自动减弱（花边、鞋带、地毯花纹不再被压成黑线）。</li>
 <li><b>画风偏离参考图</b>：tinkle 画风条目按参考图实际内容重写（prompt / prompt_gpt / 18 条渲染语言条款）；首图阶段也带上条款。</li>
 <li><b>比例跟随输入</b>：重绘/局部重绘显式下发最接近源图的官方比例（曾出现源图 2:3 却输出 2048x2048）。</li>
 <li><b>角色固有特征被画风带偏</b>：两步修——① 加「发色/瞳色/肤色不属于画风」条款（T3→T3b 把偏转从暗navy改成银白）；
   ② **首图阶段把分析原图作为内容参考图**（Image 1=内容真值、Image 2=画风只给渲染语言，<code>--content-ref on</code> 已设为默认），
   T4 实测头发区中位色相 140→8、饱和 14→22，回到源图的暖玫瑰族。</li>
</ul>

<h2>仍待解决</h2>
<ul>
 <li>水彩/厚涂类风格（waterink）下线条与扣件偏软：需要按画风登记「线条锐度」，或对脸部/发丝/扣件单独提高局部重绘分辨率（下一步）。</li>
 <li>goto-p / waterink 这类风格下线条偏软：需要按画风登记「线条锐度」参数，或对脸部/发丝单独提高局部重绘分辨率。</li>
 <li>Sol 仍指出：束腰系带的交叉与扣眼、袜带扣件、鞋带的结扣路径在细节处不够确定。</li>
</ul>
</body></html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print("written", os.path.relpath(OUT, BASE), f"{os.path.getsize(OUT)/1024/1024:.2f} MB")
