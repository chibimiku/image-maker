# -*- coding: utf-8 -*-
"""生成「今晚这五项」的测试集合 HTML（单文件、图内嵌），单独一份供明天查看。

输出：docs/gpt-image-tid-style/tonight-tests.html
"""
import base64
import glob
import html
import io
import json
import os
from datetime import datetime

from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "docs", "gpt-image-tid-style", "tonight-tests.html")
STYLE_FILE = os.path.join(BASE, "conf", "config-styles.json")
SUB_STYLE_FILE = os.path.join(BASE, "submodules", "image-maker-artstyle", "config-styles.json")


def uri(path, max_edge=720, quality=78):
    if not path or not os.path.isfile(path):
        return None
    try:
        im = Image.open(path).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    w, h = im.size
    if max(w, h) > max_edge:
        s = max_edge / max(w, h)
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def first(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(BASE, pat)))
        if hits:
            return hits[-1]
    return ""


def card(title, path, note=""):
    src = uri(path)
    if not src:
        return ""
    return (f'<figure class="card"><img loading="lazy" src="{src}" alt="{html.escape(title)}">'
            f'<figcaption><b>{html.escape(title)}</b>{("<br><span class=fname>" + html.escape(note) + "</span>") if note else ""}'
            f'</figcaption></figure>')


# ---------------- 数据收集 ----------------
styles = json.load(open(STYLE_FILE, encoding="utf-8"))
sub_styles = json.load(open(SUB_STYLE_FILE, encoding="utf-8")) if os.path.isfile(SUB_STYLE_FILE) else {}
def cover(data):
    have = [n for n, e in data.items() if isinstance(e, dict) and str(e.get("prompt_gpt") or "").strip()]
    miss = [n for n, e in data.items() if isinstance(e, dict) and not str(e.get("prompt_gpt") or "").strip()]
    return have, miss
have_local, miss_local = cover(styles)
have_sub, miss_sub = cover(sub_styles)

style_rows = []
for name, entry in styles.items():
    if not isinstance(entry, dict):
        continue
    gpt = str(entry.get("prompt_gpt") or "")
    ref = str(entry.get("ref_image") or "")
    style_rows.append((name, len(gpt), os.path.basename(ref) if ref else "—",
                       "OK" if gpt.strip() else "占位/缺失"))

pair_raw = first(["data/*/style-pair-test/pair-tid_output*.png"])
fashion_raw = first(["data/*/fashion-gpt-test/fashion_output*.png"])
fashion_final = first(["data/*/fashion-gpt-test/*-sline50-local-*.jpg"])
w5 = first(["docs/gpt-image-tid-style/workflow-analysis/W5-*.png"])
w7 = first(["docs/gpt-image-tid-style/workflow-analysis/W7-*.png"])
parts = sorted(glob.glob(os.path.join(BASE, "data", "fashion-collector", "**", "20260905_190142_5", "*", "*"), recursive=True))

VERDICTS = [
    ("1. gpt-image-2 生图界面加画风选择", "已完成",
     "Tab 顶部新增「画风」下拉（首项=默认(无附加)），显示字符数/短版来源/参考图；切换画风会实时刷新长度与成本估算。"
     "回归用例：`test_style_combo_lists_styles_with_default_first`、`test_style_info_line_renders`。"),
    ("2. 画风参考图与用户输入图的处理", "已完成并出图验证",
     "内容图在前、**画风参考图放最后**，并自动追加 IMAGE ROLES 角色分工说明（前 N 张=内容、最后一张=只借渲染语法，"
     "不得搬角色/发色/瞳色/服装/姿势/构图）。实测：素材照 + tid 画风参考图 → 保留了照片的姿势/服装/场景，"
     "发色瞳色仍是照片里的（没有变成 tid 参考图的粉发金瞳）。用例：`test_style_with_content_image_adds_role_block_and_orders_style_ref_last`。"),
    ("3. 其他画风补 gpt-image 专用 prompts", "已完成（本地 28/29 + 已同步子模块）",
     "`tools/convert_styles_gpt.py --check`：28 个画风全部 OK（498~637 字符），唯一缺的是「默认(无附加)」占位项。"
     "**同时把本地 conf 的 prompt_gpt 同步进了子模块版本化文件**（此前子模块里一个都没有，这是之前只改了 gitignore 本地文件的缺口）。"
     "新增回归用例 `test_styles_all_have_gpt_prompt_except_placeholder`。"),
    ("4. 分析图片并生图：Gemini / gpt-image-2 单选框", "已完成（接线+单测），真实出图沿用同一请求逻辑",
     "分析 Tab 顶部新增「生图通道」两个单选（默认 Gemini 老逻辑），选 gpt-image-2 时显示「gpt 工序」勾选行"
     "（出图后重绘提线 / 结构线叠加 / 局部重绘+羽化贴回）+ 区域下拉；开关了服装搭配检查/去除照片风格不受影响。"
     "请求逻辑抽到 `utils/analysis_gen.py` 并单测 7 项、Tab 接线单测 5 项；同一逻辑的真实产图见下面 W5/W7。"),
    ("5. 服饰采集 → 带画风生成穿着这套衣服的角色 + 复用后处理", "已完成并出图验证",
     "取最新采集包（dress+shoes+socks 三件套）→ 部件图在前 + tid 画风参考图最后 → gpt-image-2 出图 → "
     "复用 `utils/post_process.run_pipeline`（结构线叠加 0.5 + 人物不含面部局部重绘）。"
     "新增 `utils/styles.py` 的 GARMENT_REF_ROLE_INSTRUCTION / compose_garment_prompt。"),
]


def verdict_table():
    rows = []
    for name, status, detail in VERDICTS:
        color = "#1b7f3b" if status.startswith("已完成") else "#b26a00"
        rows.append(f"<tr><td><b>{html.escape(name)}</b></td>"
                    f"<td style='color:{color};white-space:nowrap'>{html.escape(status)}</td>"
                    f"<td>{html.escape(detail)}</td></tr>")
    return ("<table><thead><tr><th>事项</th><th>结论</th><th>证据 / 说明</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def style_table():
    rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{ln or '—'}</td><td>{html.escape(ref)}</td>"
        f"<td style='color:{'#1b7f3b' if st == 'OK' else '#b26a00'}'>{st}</td></tr>"
        for n, ln, ref, st in style_rows)
    return ("<table><thead><tr><th>画风</th><th>prompt_gpt 字符</th><th>参考图</th><th>状态</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


part_cards = "".join(
    card(os.path.splitext(os.path.basename(p))[0], p, os.path.relpath(p, BASE)) for p in parts[:6])

doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>今晚五项测试集合 · gpt-image-2 画风生图</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font: 14px/1.7 system-ui,"Segoe UI",sans-serif; margin:0; padding:26px 32px 90px; background:#fafafa; color:#222; }}
 h1 {{ font-size:23px; margin:0 0 6px; }}
 h2 {{ font-size:17px; margin:32px 0 6px; padding-top:12px; border-top:1px solid #ddd; }}
 h3 {{ font-size:14px; margin:18px 0 4px; }}
 .note {{ color:#555; }}
 table {{ border-collapse:collapse; font-size:12.5px; margin:8px 0 12px; }}
 th,td {{ border:1px solid #ddd; padding:4px 9px; text-align:left; vertical-align:top; }}
 th {{ background:#f0f0f0; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:12px; }}
 .card {{ margin:0; background:#fff; border:1px solid #e3e3e3; border-radius:8px; overflow:hidden; }}
 .card img {{ width:100%; display:block; }}
 .card figcaption {{ padding:7px 9px; font-size:12px; }}
 .fname {{ color:#888; font-size:10.5px; word-break:break-all; }}
 pre {{ white-space:pre-wrap; background:#f4f4f4; padding:8px; border-radius:6px; font-size:11.5px; max-height:260px; overflow:auto; }}
 code {{ background:#f0f0f0; padding:1px 4px; border-radius:3px; }}
 @media (prefers-color-scheme: dark) {{
   body {{ background:#1b1b1b; color:#e9e9e9; }} .card {{ background:#262626; border-color:#3a3a3a; }}
   th {{ background:#333; }} pre, code {{ background:#333; }}
 }}
</style></head><body>
<h1>今晚五项测试集合</h1>
<p class="note">生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')} ·
画风配置 <code>conf/config-styles.json</code> ·
本文档由 <code>tools/make_tonight_tests.py</code> 生成</p>

<h2>一、目标达成判定</h2>
{verdict_table()}

<h2>二、① 画风选择 + ② 参考图职责（gpt-image-2 Tab）</h2>
<h3>UI 变化</h3>
<ul>
 <li>新增「画风」下拉：首项 <code>默认(无附加)</code>，其余为 <code>config-styles.json</code> 的全部画风；
     右侧信息行显示 <code>字符数 / 来源(prompt_gpt 或全量) / 画风参考图文件名</code>，缺 prompt_gpt 会成为橙色提醒。</li>
 <li>提交顺序：<b>用户内容图在前 → 画风参考图最后</b>；两类图同时存在时提示词自动带上 IMAGE ROLES 分工块。</li>
 <li>提示词长度与成本估算把画风段也算进去（软上限 2000 / 硬上限 15000 字符）。</li>
</ul>
<h3>真实出图验证（素材照 + tid 画风参考图）</h3>
<div class="grid">{card("pair-tid（内容图 1 张 + 画风参考图 1 张）", pair_raw, "发色/瞳色保持照片设定，未搬 tid 的角色特征")}</div>
<h3>请求里的关键段落</h3>
<pre>IMAGE ROLES (read carefully):
- The FIRST 1 image(s) are CONTENT REFERENCES: they define the subject, character, pose, outfit design, props,
  scene and composition.
- The LAST image is the ART-STYLE REFERENCE ONLY: use its palette, colour temperature, lighting, brushwork,
  edge treatment, line character, texture and overall rendering grammar.
Do NOT copy the style reference's character, face, hairstyle, hair colour, eye colour, outfit, pose, props,
background or composition. ... If the two disagree about content, the content references win.

Palette: White-dominant, low-saturation, warm pink-beige, muted cyan-green
Lighting: High-key diffuse illumination, soft ambient fill, gentle shadows
... （prompt_gpt 的 8 个字段）

&lt;用户提示词原文&gt;</pre>

<h2>三、③ 画风 prompt_gpt 覆盖</h2>
<p class="note">本地 <code>conf/config-styles.json</code>：有 prompt_gpt <b>{len(have_local)}</b> / 缺 <b>{len(miss_local)}</b>
（缺的只有占位项 {html.escape(str(miss_local))}）。
子模块版本化文件 <code>submodules/image-maker-artstyle/config-styles.json</code>：有 <b>{len(have_sub)}</b> / 缺 <b>{len(miss_sub)}</b>
—— 本轮已把本地已转换的 28 个 <code>prompt_gpt</code> 同步进子模块（之前只改了 gitignore 的本地文件，等于没进版本库）。</p>
{style_table()}

<h2>四、④ 分析图片并生图：生图通道单选框</h2>
<h3>UI 变化</h3>
<ul>
 <li><b>生图通道</b> 两个单选：<code>Gemini（老逻辑）</code>（默认） / <code>gpt-image-2（gpt 专用短锚 + 画风参考图 + 工序）</code>。</li>
 <li>选 gpt-image-2 时出现「gpt 工序」行：出图后重绘提线（Gemini） / 结构线叠加 / 局部重绘+羽化贴回 + 区域下拉。</li>
 <li>「服装搭配检查」「去除照片风格」「重算 pixiv_tags」等原有勾选完全不受影响（它们发生在分析阶段）。</li>
</ul>
<h3>请求组装逻辑（utils/analysis_gen.py）</h3>
<ul>
 <li>提示词 = <code>prompt_gpt</code> + 分析产物的 <code>gpt_image_prompt_short</code>（≤500，缺则完整档≤1400，再缺则裁描述）+ 参考图排除句。</li>
 <li>参考图 <b>只挂画风参考图</b>，不挂原分析图。</li>
 <li>工序按勾选顺序执行：重绘（传 <code>use_detail_suffix=False</code>，避免旧 cel 后缀污染）→ 结构线叠加 → 局部重绘。</li>
</ul>
<h3>同一逻辑的真实产图（素材照分析产物 → gpt-image-2）</h3>
<div class="grid">
 {card("W5：短锚 1325 字符 + 排除句", w5, "与 tid 画风相似度 0.2804、饱和 29.0")}
 {card("W7：完整锚 2116 字符 + 排除句", w7, "与 tid 画风相似度 0.3234、饱和 25.7（≈ tid 的 25.0）")}
</div>

<h2>五、⑤ 服饰采集 → 穿着这套衣服的角色（含后处理）</h2>
<h3>采集到的服饰部件（内容参考）</h3>
<div class="grid">{part_cards}</div>
<h3>生成与后处理结果</h3>
<div class="grid">
 {card("gpt-image-2 出图（3 部件 + 画风参考图）", fashion_raw, "部件在前、tid 画风图最后")}
 {card("后处理：结构线叠加 0.5 + 人物不含面部局部重绘", fashion_final, "复用 utils/post_process.run_pipeline")}
</div>
<h3>本轮为服饰场景新增的接口</h3>
<pre>from utils.styles import compose_garment_prompt
prompt, images = compose_garment_prompt(part_paths, prompt_gpt_text,
                                        character_hint="全body站立、三视图外的三角构图",
                                        style_ref_path=tid_ref)
# images = [dress, shoes, socks, tid_ref]   ← 部件在前、画风图最后
# prompt 里会写：前 N 张是 GARMENT REFERENCES（照做这些款式/配色/蕾丝/扣件），最后一张只借渲染语法</pre>

<h2>六、本轮改动清单</h2>
<table><thead><tr><th>类型</th><th>文件</th><th>说明</th></tr></thead><tbody>
<tr><td>utils</td><td>utils/styles.py</td><td>STYLE_REF_ROLE_INSTRUCTION / GARMENT_REF_ROLE_INSTRUCTION / compose_style_prompt / ordered_reference_images / compose_garment_prompt</td></tr>
<tr><td>utils</td><td>utils/analysis_gen.py</td><td>分析→gpt-image 的请求组装 + 工序流水线（新）</td></tr>
<tr><td>GUI</td><td>modules/image_generation/gpt_image2_tab.py</td><td>画风下拉 + 信息行 + 请求组装接入 + 预算含画风段</td></tr>
<tr><td>GUI</td><td>modules/image_analysis/single_analyzer.py</td><td>生图通道单选 + gpt 工序勾选 + GptImageGenWorkerThread（新）</td></tr>
<tr><td>CLI</td><td>tools/gpt_image2_gen.py</td><td>--style 与 GUI 共用角色分工与图片顺序</td></tr>
<tr><td>工具</td><td>tools/sync_styles_to_submodule.py</td><td>把本地画风字段同步进子模块（新）</td></tr>
<tr><td>测试</td><td>tests/test_analysis_gen.py(7) / tests/test_analysis_channel.py(5) / test_gpt_image2_api.py 新增 6 项</td><td>请求组装、工序、单选框接线、画风顺序与覆盖</td></tr>
</tbody></table>
</body></html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(doc)
print(f"written {OUT}  {os.path.getsize(OUT)/1024/1024:.2f} MB")
