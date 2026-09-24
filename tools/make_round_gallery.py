# -*- coding: utf-8 -*-
"""生成本轮（2026-09-21）全部比较组的 HTML 画廊。

用法（仓库根目录）：
  & "C:\\Program Files\\Python310\\python.exe" tools/tmp_make_gallery.py
输出：
  docs/gpt-image-tid-style/round-20260921.html
"""
import base64
import glob
import html
import json
import os
import re
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY = os.path.join(BASE, "data", "20260921")
OUT = os.path.join(BASE, "docs", "gpt-image-tid-style", "round-20260921.html")

# 目录 -> 这一组是什么（标题 / 说明 / 排序）
BATCH_META = {
    "tid-verify": ("① 真实提示词验证（4 主题 × 3 注入方式）",
                   "基线组：短说明+参考图(V1) / 全量13k+参考图(V2) / 全量13k纯文本(V3)。"
                   "结论：V1 四项指标全面更好，V2/V3 把参考图挤掉。", 1),
    "tid-luna": ("② Luna 建议轮（8 个变体 + 无主体裁剪/风格板）",
                 "R1 角色分工版 / R2 crop_white / R3 crop_left / R4 风格板 / R5 结构化字段版（当时最优，"
                 "后来发现过白+糊线）/ R6 字段版+身份排除 / R7 字段版+quality=high / R8 字段版无参考图。", 2),
    "tid-sol": ("③ Sol 修正轮 S1/S2/S3（对着 R5 的两个问题改）",
                "S1 = Sol 完整修正版 + high；S2 = 完整版 + medium；S3 = 精简版(2.9k) + medium。", 3),
    "tid-sol-cross": ("④ Sol 精简版跨主体复验（舞蹈 / 洛可可 / 和服）",
                      "验证 S3 的修正不是只对「少女站浅水」有效。", 4),
    "tid-sol-v5": ("⑤ Sol 最终版 v5（≤2000 字符）",
                   "浅水 / 舞蹈 / 和服三个主体，medium。", 5),
    "tid-sol-v6": ("⑥ Sol 第三轮建议版 v6 + 跨画风验证（当前首图推荐）",
                   "v6h = 全文(2.6k) + gpt-image-2 high；v6m = 全文 + medium；v6h-flare = 全文 + gpt-image-2.5-flare high；"
                   "v6x-* = 同一 v6 提示词换三个不同画风的参考图（say-hana / noir-aiart / ajicoma），验证不是只对 tid 有效。"
                   "改善点：双手可见、发束分组、中间调恢复。", 6),
    "tid-stylegpt": ("⑦ prompt_gpt 字段版实拍（config-styles.json 落地验证）",
                     "用 tools/gpt-image2_gen.py --style <画风> 走的完整链路：tid / say-hana-style / noir-aiart。", 7),
    "tid-sol-v6-repaint": ("⑧ v6 产物 + Gemini 重绘（默认固件 = cel shading）",
                           "用项目现成重绘链路 gemini-3-pro-image-preview @2K（固件写死了 flat cel shading）。"
                           "线稿连通性大幅提升（平均段长×2、端点密度÷3），但把水彩画风改写成平涂，"
                           "与源图的配色相似度在 say-hana/noir 上掉到 0.71 / 0.52。", 8),
    "tid-sol-v6-repaint-neutral": ("⑨ v6 产物 + Gemini 重绘（画风中性固件）",
                                   "同样走 Gemini@2K，但把固件里写死的 cel shading 换成 STYLE FIDELITY 段"
                                   "（先判定输入自身的渲染语言、禁止跨媒介转换）。保真度大幅回升："
                                   "say-hana 配色相似 0.71→0.97、tid 0.95→0.94，线稿改善仍然拿到。", 9),
    "line-continuity": ("⑩ 线条连续性控制变量：只换参考图（问题 1）",
                        "同一 v6 提示词、同一主体：A tid / B say-hana / C say-hana 放大到 1000px / D noir / "
                        "E noir 放大 / F 无参考图（纯文本）/ G ajicoma。结论：决定线条连续性的是**参考图自身的"
                        "边缘语言**，不是分辨率——放大 say-hana 反而更差，纯文本比给软边参考图更好。", 10),
    "line-continuity-fix": ("⑪ 想用文字/预处理救线条连续性的尝试（均未成功）",
                            "H1~H4 = 在 v6 里加 EDGE TRANSLATION 段（把柔边翻译成连续线）。"
                            "say-hana 段长 100→114.7、noir 98→108，端点密度几乎没动 → 文字压不过图像条件。", 11),
    "line-continuity-refprep": ("⑫ 参考图预处理（unsharp / 结构线叠加）后再喂",
                                "E1 = 参考图做 unsharp；E2 = 再加结构线暗化。像素级锐化不产生连续长轮廓，"
                                "所以对线条连续性无实质帮助。", 12),
    "workflow-tid-photo": ("⑬ 常用工作流实测：素材照 → gpt-image-2（tid 画风）→ Gemini 重绘（任务 3）",
                           "wf-tid = 全身构图（重绘被中转站内容审核拦，见 README §4.12）；"
                           "wf-tid2 = 腰以上构图 → 重绘成功，头部 MAE 14.6、配色相似 0.8896（面部保真明显好于浅水那两组）。", 13),
    "repaint-diag": ("⑭ 重绘被拦的诊断（无产物）",
                     "D1~D5 全 400：一开始是缺 `role: user`（已修 api_backend），修完这条又撞中转站 "
                     "PROHIBITED_CONTENT 内容审核；1K/JPEG/flash 换法都无效。", 14),
    "repaint-v2": ("⑮ 重绘固件第二轮：Sol 保守修复版（中性 v2）",
                   "对 tid / say-hana / noir / 工作流图各重绘一次。保真大幅提升（say-hana 头部配色 0.221→0.966、"
                   "noir 全图 0.524→0.966），但清线指标回落；工作流图出现头部过淡（MAE 56.6、饱和 −18.5）。", 15),
    "repaint-v3": ("⑯ 重绘固件第三轮：只改色彩（v3）/ 色彩+线条（v4）/ 改良锚图双参考（v5-dual）",
                   "v3 修好了工作流图的过淡（MAE 17.8）但单张方差大；v4 线条指标全部改善且不 cel 化（当前默认固件）；"
                   "v5-dual 用「源图 + 改良锚图」双参考，清线最强（noir 段长 173.7、端点 11.4）但保真下降。", 16),
    "anchor-test": ("⑰ 锚图实验 v1（Sol 方案①）：旧锚图提示词",
                    "两阶段：先生成锚图、再用锚图当参考图出图。方向对（say-hana 段长 99.8→120.5、noir 98.1→118.6），"
                    "但锚图本身内容泄漏（与源图配色仅 0.445）。", 17),
    "anchor-v2": ("⑱ 改良锚图（保留身份/构图/色彩，只增强主要边缘）",
                  "锚图与源图配色相似度从 0.445 提到 0.961，且锚图自身线条更连贯（say-hana 段长 103→111、端点 22.2→20.8）。", 18),
    "line-v2": ("⑲ Sol A/B/E 三组验证（今晚 23:4x）：硬化双参考 / 三参考+结构线 / tid 双参考",
                 "A = say-hana 双参考 + 可验证结构要求；B = 三参考（源图+锚图+本地生成的结构线图）；E = tid 双参考基线。"
                 "结论：A ≈ v5-dual（提示词硬化无效）；B 的线条没更好但保真回升；E 对 tid 提升线条却明显掉保真——"
                 "印证 Sol「水彩厚涂与线条连贯存在原理上限」。", 19),
    "workflow-analysis": ("⑳ 分析工作流改造实测（今晚 23:4x~23:5x）：分析 prompts → gpt-image-2 → Gemini 重绘",
                          "P1 = 现有产物组合（11k 画风 + 4.4k 描述 = 15.7k）**连接被重置（复现『太长』）**；"
                          "P2 短画风+全量描述 5.0k 成功；P3 压到 1.7k 成功但内容跑偏；P4 加素材照参考图更保真；"
                          "P2/P4 再走 Gemini 重绘 v4 固件，段长 +20~27%、端点密度 −22~31%。", 20),
    # 其它目录（tmp_repaint 的空占位文件、gemini_repaint_verify 的单张 jpg）不是本轮比较组，不收录
}

PREFIX_LABEL = {
    "tv-c1-v1": "C1 双人百合 · V1 短说明+图", "tv-c1-v2": "C1 双人百合 · V2 全量+图",
    "tv-c1-v3": "C1 双人百合 · V3 全量纯文本",
    "tv-c2-v1": "C2 舞蹈动态 · V1 短说明+图", "tv-c2-v2": "C2 舞蹈动态 · V2 全量+图",
    "tv-c2-v3": "C2 舞蹈动态 · V3 全量纯文本",
    "tv-c3-v1": "C3 洛可可肖像 · V1 短说明+图", "tv-c3-v2": "C3 洛可可肖像 · V2 全量+图",
    "tv-c3-v3": "C3 洛可可肖像 · V3 全量纯文本",
    "tv-c4-v1": "C4 洛丽塔街拍 · V1 短说明+图", "tv-c4-v2": "C4 洛丽塔街拍 · V2 全量+图",
    "tv-c4-v3": "C4 洛丽塔街拍 · V3 全量纯文本",
    "luna-r1": "R1 角色分工版 + 原参考图", "luna-r2": "R2 角色分工 + crop_white（无主体裁剪）",
    "luna-r3": "R3 角色分工 + crop_left", "luna-r4": "R4 角色分工 + 风格板",
    "luna-r5": "R5 结构化字段版（当时最优）", "luna-r6": "R6 字段版 + 具体身份排除",
    "luna-r7": "R7 角色分工版 + quality=high", "luna-r8": "R8 字段版 + 无参考图",
    "sol-s1": "S1 Sol 完整修正版 + high", "sol-s2": "S2 Sol 完整修正版 + medium",
    "sol-s3": "S3 Sol 精简版(2.9k) + medium",
    "cross-c2": "跨主体 · 舞蹈（Sol 精简版）", "cross-c3": "跨主体 · 洛可可（Sol 精简版）",
    "cross-c5": "跨主体 · 和服（Sol 精简版）",
    "v5-v5a": "V5-A 浅水（Sol 最终版 ≤2000字符）", "v5-v5b": "V5-B 舞蹈（Sol 最终版）",
    "v5-v5c": "V5-C 和服（Sol 最终版）",
    "sg-tid": "prompt_gpt · tid", "sg-sayhana": "prompt_gpt · say-hana-style",
    "sg-noir": "prompt_gpt · noir-aiart",
    "v6-v6h": "V6 全文 + high（gpt-image-2）", "v6-v6m": "V6 全文 + medium（gpt-image-2）",
    "v6x-say_hana": "V6 跨画风 · say-hana-style（high）", "v6x-noir": "V6 跨画风 · noir-aiart（high）",
    "v6x-ajicoma": "V6 跨画风 · ajicoma（high）",
}


def img_data_uri(path: str, max_edge: int = 900) -> str:
    """把图片压成 data URI（最长边 ≤ max_edge），保证 HTML 单文件可离线看；读不了的文件返回 None。"""
    try:
        if os.path.getsize(path) < 1024:      # 0 字节 / 占位文件
            return None
    except OSError:
        return None
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    import io

    try:
        im = Image.open(path).convert("RGB")
    except Exception:  # noqa: BLE001 - 损坏 / 非图片文件直接跳过
        return None
    w, h = im.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=86)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def prompt_from_log(log_path: str) -> str:
    if not os.path.isfile(log_path):
        return ""
    text = open(log_path, encoding="utf-8", errors="replace").read()
    if "---" in text:
        head = text.split("---", 1)[0]
    else:
        head = text
    m = re.search(r'"prompt":\s*"((?:[^"\\]|\\.)*)"', text, re.S)
    if m:
        try:
            head = json.loads('"' + m.group(1) + '"')
        except Exception:  # noqa: BLE001
            head = m.group(1)
    return head.strip()


def usage_of(json_path: str) -> dict:
    if not os.path.isfile(json_path):
        return {}
    try:
        d = json.load(open(json_path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    u = d.get("usage") or {}
    return u


LOG_DIR = os.path.join(BASE, "data", "gpt-image-tid", "logs")
LOG_GUESS = {
    "tv-c1-v1": "verify_C1_yuri_V1_short_ref.log", "tv-c1-v2": "verify_C1_yuri_V2_full_head.log",
    "tv-c1-v3": "verify_C1_yuri_V3_full_text.log",
    "tv-c2-v1": "verify_C2_dance_V1_short_ref.log", "tv-c2-v2": "verify_C2_dance_V2_full_head.log",
    "tv-c2-v3": "verify_C2_dance_V3_full_text.log",
    "tv-c3-v1": "verify_C3_rococo_V1_short_ref.log", "tv-c3-v2": "verify_C3_rococo_V2_full_head.log",
    "tv-c3-v3": "verify_C3_rococo_V3_full_text.log",
    "tv-c4-v1": "verify_C4_lolita_V1_short_ref.log", "tv-c4-v2": "verify_C4_lolita_V2_full_head.log",
    "tv-c4-v3": "verify_C4_lolita_V3_full_text.log",
    "luna-r1": "luna_R1_role_ref.log", "luna-r2": "luna_R2_role_cropwhite.log",
    "luna-r3": "luna_R3_role_cropleft.log", "luna-r4": "luna_R4_role_board.log",
    "luna-r5": "luna_R5_fields_ref.log", "luna-r6": "luna_R6_fields_exclude.log",
    "luna-r7": "luna_R7_role_ref_high.log", "luna-r8": "luna_R8_fields_noref.log",
    "sol-s1": "sol_S1_sol_full_high.log", "sol-s2": "sol_S2_sol_full_medium.log",
    "sol-s3": "sol_S3_sol_short_medium.log",
    "cross-c2": "solcross_C2_dance.log", "cross-c3": "solcross_C3_rococo.log",
    "cross-c5": "solcross_C5_kimono.log",
    "v5-v5a": "v5_V5A_water.log", "v5-v5b": "v5_V5B_dance.log", "v5-v5c": "v5_V5C_kimono.log",
}

ref_stats = {}
stats_path = os.path.join(BASE, "docs", "gpt-image-tid-style", "style-gpt-cases", "style-ref-stats.json")
if os.path.isfile(stats_path):
    ref_stats = json.load(open(stats_path, encoding="utf-8"))

batches = {}
for name, meta in BATCH_META.items():
    d = os.path.join(DAY, name)
    if not os.path.isdir(d):
        continue
    items = []
    found = sorted(glob.glob(os.path.join(d, "*output*.png")))
    found += sorted(p for p in glob.glob(os.path.join(d, "*.jpg"))
                    + glob.glob(os.path.join(d, "*.jpeg"))
                    + glob.glob(os.path.join(d, "*.png"))
                    if "_output" not in p and not os.path.basename(p).endswith(".tmp"))
    seen = set()
    for png in [p for p in found if not (p in seen or seen.add(p))]:
        base = os.path.basename(png)
        if "_output" in base and "-" in base:
            prefix = base.split("-", 1)[1].rsplit("_output", 1)[0]
            key = base.split("_output")[0]
        else:
            prefix = os.path.splitext(base)[0]
            key = prefix
        json_path = os.path.join(d, base.replace("_output", "_aigc-2d-gpt_server_response").split(".png")[0] + ".json")
        log_path = os.path.join(LOG_DIR, LOG_GUESS.get(key, "")) if LOG_GUESS.get(key) else ""
        items.append({
            "file": base,
            "path": png,
            "label": PREFIX_LABEL.get(key, key),
            "prompt": prompt_from_log(log_path) if log_path else "",
            "usage": usage_of(json_path),
            "mtime": os.path.getmtime(png),
        })
    if items:
        items.sort(key=lambda x: x["mtime"])
        batches[name] = {"meta": meta, "items": items}

# ---------- HTML ----------
rows_html = []
for name, data in sorted(batches.items(), key=lambda kv: kv[1]["meta"][2]):
    title, desc, _ = data["meta"]
    cards = []
    for it in data["items"]:
        uri = img_data_uri(it["path"])
        if not uri:
            continue
        u = it["usage"] or {}
        det = u.get("input_tokens_details") or {}
        meta_bits = []
        if u:
            meta_bits.append(f"text_tokens {det.get('text_tokens', '-')}")
            meta_bits.append(f"image_tokens {det.get('image_tokens', '-')}")
            meta_bits.append(f"output_tokens {u.get('output_tokens', '-')}")
        prompt_html = ""
        if it["prompt"]:
            prompt_html = f'<details><summary>查看实际提示词（{len(it["prompt"])} 字符）</summary><pre>{html.escape(it["prompt"])}</pre></details>'
        cards.append(f"""
      <figure class="card">
        <img loading="lazy" src="{uri}" alt="{html.escape(it['label'])}">
        <figcaption>
          <b>{html.escape(it['label'])}</b>
          <span class="fname">{html.escape(it['file'])}</span>
          <span class="meta">{' · '.join(meta_bits)}</span>
          {prompt_html}
        </figcaption>
      </figure>""")
    rows_html.append(f"""
  <section id="{name}">
    <h2>{html.escape(title)}</h2>
    <p class="desc">{html.escape(desc)}</p>
    <p class="path">目录：<code>data/20260921/{name}/</code>（{len(data['items'])} 张）</p>
    <div class="grid">{''.join(cards)}</div>
  </section>""")

toc = "\n".join(
    f'<li><a href="#{name}">{html.escape(batches[name]["meta"][0])}</a> <span class="n">{len(batches[name]["items"])} 张</span></li>'
    for name in sorted(batches, key=lambda k: batches[k]["meta"][2])
)

ref_stats_html = ""
if ref_stats:
    rows = []
    for k, v in ref_stats.items():
        if k == "tid":
            rows.append(f"<tr class='hl'><td>{html.escape(k)}</td><td>{v.get('brightness_mean')}</td>"
                        f"<td>{v.get('saturation_mean')}</td><td>{v.get('near_white_ratio_245')}</td>"
                        f"<td>{v.get('edge_density')}</td></tr>")
    ref_stats_html = f"""
  <section id="ref">
    <h2>参考图客观统计（用于对比的基准）</h2>
    <table><thead><tr><th>画风</th><th>亮度</th><th>饱和度</th><th>白底&gt;245</th><th>边缘密度</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table>
    <p class="desc">完整 27 个画风的统计见 <code>style-gpt-cases/style-ref-stats.json</code>。</p>
  </section>"""

doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>gpt-image 画风生图 · 2026-09-21 比较组总览</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.6 system-ui, "Segoe UI", sans-serif; margin: 0; padding: 24px 28px 80px; background: #fafafa; color: #222; }}
  h1 {{ font-size: 22px; margin: 0 0 6px; }}
  h2 {{ font-size: 17px; margin: 34px 0 4px; padding-top: 10px; border-top: 1px solid #ddd; }}
  .desc {{ margin: 2px 0 6px; color: #555; }}
  .path {{ margin: 0 0 12px; color: #777; font-size: 12.5px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }}
  .card {{ margin: 0; background: #fff; border: 1px solid #e3e3e3; border-radius: 8px; overflow: hidden; }}
  .card img {{ width: 100%; display: block; }}
  .card figcaption {{ padding: 8px 10px; font-size: 12.5px; }}
  .fname {{ display: block; color: #888; font-size: 11px; word-break: break-all; }}
  .meta {{ display: block; color: #666; font-size: 11px; }}
  details {{ margin-top: 6px; }}
  pre {{ white-space: pre-wrap; word-break: break-word; background: #f4f4f4; padding: 8px; border-radius: 6px;
         max-height: 260px; overflow: auto; font-size: 11.5px; }}
  ul.toc {{ columns: 2; }}
  ul.toc .n {{ color: #999; font-size: 12px; }}
  code {{ background: #eee; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }}
  table {{ border-collapse: collapse; font-size: 12.5px; }}
  th, td {{ border: 1px solid #ddd; padding: 3px 10px; }}
  tr.hl {{ background: #fff8dc; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #1b1b1b; color: #e8e8e8; }}
    .card {{ background: #262626; border-color: #3a3a3a; }}
    pre {{ background: #333; }}
    code {{ background: #333; }}
    tr.hl {{ background: #4a4326; }}
  }}
</style>
</head>
<body>
<h1>gpt-image 画风生图 · 2026-09-21 比较组总览</h1>
<p class="desc">本轮所有比较组（共 {sum(len(b['items']) for b in batches.values())} 张），按实验顺序排列。所有图内嵌在本文件里，可离线查看。</p>

<h2 style="border:none;padding:0">目录</h2>
<ul class="toc">{toc}</ul>

{ref_stats_html}
{''.join(rows_html)}

<p class="desc" style="margin-top:40px">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} ·
脚本：<code>tools/make_round_gallery.py</code> · 实验报告：<code>docs/gpt-image-tid-style/README.md</code></p>
</body>
</html>
"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(doc)
print(f"written {OUT}  ({os.path.getsize(OUT) / 1024 / 1024:.1f} MB, {sum(len(b['items']) for b in batches.values())} 张图)")
for n, (name, data) in enumerate(sorted(batches.items(), key=lambda kv: kv[1]["meta"][2]), 1):
    print(f"  [{n}] {len(data['items'])} imgs")
