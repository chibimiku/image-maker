# -*- coding: utf-8 -*-
"""《超人气配色手册》转录页 → 结构化配色知识。

**这个模块只做一件事：把已经逐字转录的 150 页 md，整理成能直接喂给下游的 JSON。**

设计上的三条硬约束（都是踩过坑之后定下来的）：

1. **不新增任何书里没有的说法。** 每个条款都必须带 `source.pdf_page` 与 `evidence`；
   `evidence` 用书里的原话。早期页面（PDF 17/19/36 等）的表格里混了我自己写的英文对照，
   这些行一律标 `evidence_type="derived"`，与 `"stated"` 分开，下游可以按需过滤。

2. **案例配色的色名以原文为准，不做同义改写。** 书里混用「固有色名」（紫藤色）与
   「色彩系统名」（蓝紫色）两套命名，能对上 130 色表的就补上标称 RGB/CMYK，
   对不上的就显式标 `system_coordinate: None` + `name_kind: "固有色名"`，
   **不允许"就近映射"到某个色表条目** —— 那是编造。

3. **分类体系用书自己的。** 条款的 `category` 取自 PDF 146（笔记：插画中的色彩搭配）
   给出的五步流程 + 两条注意事项，再补上光效/材质/排列/知觉四类
   （这四类在书里各有大量独立内容）。
"""

from __future__ import annotations

import io
import json
import os
import re

from utils.color_knowledge import parse_page_markdown

# ---------------------------------------------------------------------------
# 条款分类：先用书里的五步流程做主轴，再补四类书里独立成篇的内容
# ---------------------------------------------------------------------------

#: `(category, 中文名, 书里的出处, 匹配关键词)`
#: **顺序即优先级** —— 越靠前的越具体，先命中先归类。
CLAUSE_CATEGORIES = [
    ("perception", "知觉与错觉", "PDF 51 / 99 / 100 / 115 / 142",
     ["视像", "错觉", "恒常", "记忆色", "联觉", "斯特鲁普", "膨胀色", "收缩色",
      "锥体", "视网膜", "波长", "散射", "浦肯野", "亥姆霍兹",
      "触感", "通感", "五感", "感官"]),
    ("light", "光效", "PDF 91 / 136 等",
     ["逆光", "高光", "反光", "投影", "阴影", "影子", "亚光", "透光", "透明",
      "光泽", "光幕", "炫光", "光粒", "光源", "阳光", "光线",
      "受光", "背光", "背阴", "明暗"]),
    ("material", "材质与质感", "PDF 91 / 92",
     ["质感", "材质", "纹理", "湿润", "金属", "不锈钢", "玻璃", "瓷器", "鹅卵石",
      "酥脆", "蓬松", "木头", "木材", "皮革", "纸"]),
    ("arrangement", "排列手法（渐变 / 分离 / 主题色彩链）", "PDF 18 / 63 / 133",
     ["渐变", "分离", "排列", "色彩链", "间隔", "主题色"]),
    ("composition", "构图", "PDF 71 / 84 / 124 / 134",
     ["构图", "画框", "居中", "三角", "圆形框架", "对称", "视线", "焦点"]),
    ("accent", "第 5 步 · 强调色", "PDF 146 第 5 步",
     ["强调色", "强调"]),
    ("base", "第 4 步 · 基调色", "PDF 146 第 4 步",
     ["基调色", "基调", "辅助色", "大面积"]),
    ("area", "面积配比", "PDF 59 / 77 / 79",
     ["面积", "占比", "百分比", "比例", "60%", "70%", "80%", "90%"]),
    ("contrast", "对比", "PDF 146 注意事项 a",
     ["对比", "反差"]),
    ("assimilation", "同化与相似性", "PDF 117 / 145 / 146 注意事项 a",
     ["同化", "相似性", "亲和", "共通性", "和谐", "秩序", "明晰", "衬托",
      "浑然一体", "统一感"]),
    ("selection", "第 3 步 · 选色彩（色相数量与搭配）", "PDF 146 第 3 步",
     ["色相数量", "色相多", "色相少", "类似色", "补色", "相反色", "同色系",
      "色相范围", "色相", "色相环", "无彩色", "冷暖", "保留", "剔除", "不用"]),
    ("tone", "第 2 步 · 色调（清浊 / 明暗 / 彩度）", "PDF 146 第 2 步",
     ["清色", "浊色", "清浊", "色调", "明度", "彩度", "饱和度", "暗沉", "淡调",
      "锐调", "钝调", "涩调", "弱调", "浓调"]),
    ("story", "第 1 步 · 故事与意象", "PDF 146 第 1 步",
     ["故事", "主题", "意象", "情绪", "意图", "氛围", "情感", "感受"]),
    ("relativity", "配色是相对的", "PDF 146 注意事项 b",
     ["相对", "视差", "明明是", "看起来像"]),
]


def _category_of(text: str, page_type: str = "") -> tuple[str, str]:
    """给一条条款归类。

    **`case_process` 页的表格是逐步操作记录，不是可复用的配色条款**，
    单独归到 `process`（"加背景山""人物底稿"这类），
    免得它们混进条款库被下游当成配色规则用。
    """
    if page_type == "case_process":
        return "process", "绘画过程步骤"
    for cat, zh, _src, kws in CLAUSE_CATEGORIES:
        for kw in kws:
            if kw in text:
                return cat, zh
    return "other", "其他"

#: 早期页面（PDF 8–37）里我用的是另一个小节标题，这些行混有我自己写的英文对照
DERIVED_SECTION = re.compile(r"^##\s*配色要点")
#: 后期页面的小节标题 —— 表里是书里的原话
STATED_SECTION = re.compile(
    r"^##\s*(本页直接给出的|本页是全书的方法论总纲|本页的价值|本页的推理链)")


def _split_tables(body: str):
    """把 markdown 正文切成 [(小节标题, [表的行]), ...]。

    只认 `## ` 级小节，且小节里要有 markdown 表（表头 + 分隔行 + ≥1 数据行）。
    """
    out = []
    cur_title, cur_lines = None, []
    for line in body.split("\n"):
        if line.startswith("## "):
            if cur_title is not None:
                out.append((cur_title, cur_lines))
            cur_title, cur_lines = line[3:].strip(), []
        elif cur_title is not None:
            cur_lines.append(line)
    if cur_title is not None:
        out.append((cur_title, cur_lines))

    tables = []
    for title, lines in out:
        rows, i = [], 0
        while i < len(lines):
            s = lines[i].strip()
            if (s.startswith("|") and i + 1 < len(lines)
                    and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip())):
                head = [c.strip() for c in s.strip("|").split("|")]
                i += 2
                while i < len(lines) and lines[i].strip().startswith("|"):
                    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    if len(cells) == len(head):
                        rows.append((head, cells))
                    i += 1
            else:
                i += 1
        if rows:
            tables.append((title, rows))
    return tables


def _clean_md(s: str) -> str:
    """去掉 markdown 记号，留纯文本（条款要么进提示词、要么给人看，都不该带 `**`）。"""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"<u>(.+?)</u>", r"\1", s)
    s = s.replace("\\*", "*").strip()
    return s


def extract_clauses(pages_dir: str, log_callback=None) -> dict:
    """扫全书页面，抽出所有条款。

    只取两类小节：
    · `## 本页直接给出的…` 等 —— 表里是**书里的原话** → `evidence_type="stated"`
    · `## 配色要点…` —— 早期页面，表里混了我写的英文对照 → `evidence_type="derived"`
    """
    def log(m):
        if log_callback:
            log_callback(m)

    clauses, skipped = [], []
    for num in range(1, 151):
        path = os.path.join(pages_dir, "page-%03d.md" % num)
        if not os.path.isfile(path):
            continue
        meta, body = parse_page_markdown(io.open(path, encoding="utf-8").read())
        for title, rows in _split_tables(body):
            if DERIVED_SECTION.match("## " + title):
                kind = "derived"
            elif STATED_SECTION.match("## " + title):
                kind = "stated"
            else:
                continue
            for head, cells in rows:
                # 表格第一列是要点，最后一列是依据 / 英文对照 / 定义
                key = _clean_md(cells[0])
                val = _clean_md(cells[-1])
                if not key or len(key) < 4:
                    continue
                if len(cells) >= 3:                      # 三列以上：中间列也算内容
                    mid = _clean_md(cells[1])
                    if mid and mid != val:
                        val = mid + " ｜ " + val
                text = key + " " + val
                cat, cat_zh = _category_of(text, meta.get("page_type") or "")
                clauses.append({
                    "id": "clause-p%03d-%02d" % (num, len([c for c in clauses
                                                           if c["source"]["pdf_page"] == num]) + 1),
                    "category": cat,
                    "category_zh": cat_zh,
                    "item": key,
                    "detail": val,
                    # 整行原样留着 —— 自检要用它判断"这条到底在不在来源页里"。
                    # 只留 item/detail 时会漏判：有些行的 detail 我写的是
                    # "三条图注"这类占位，真正的原话在中间列。
                    "cells": [_clean_md(c) for c in cells],
                    "evidence_type": kind,
                    "section": title,
                    "source": {"pdf_page": num,
                               "printed_page": meta.get("printed_page"),
                               "title": meta.get("title"),
                               "page_type": meta.get("page_type")},
                })
            if not rows:
                skipped.append(num)
    by_cat = {}
    for c in clauses:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
    log("[clauses] 共 %d 条，覆盖 %d 页；分类分布：%s"
        % (len(clauses), len({c["source"]["pdf_page"] for c in clauses}),
           " ".join("%s=%d" % (k, v) for k, v in sorted(by_cat.items(), key=lambda x: -x[1]))))
    stated = sum(1 for c in clauses if c["evidence_type"] == "stated")
    log("[clauses] 书里原话 %d 条 / 早期派生 %d 条" % (stated, len(clauses) - stated))
    return {"version": 1, "count": len(clauses),
            "source": "docs/261003-color-improve/book-html/pages/*.md",
            "taxonomy": [{"category": c, "zh": z, "from": f}
                         for c, z, f, _ in CLAUSE_CATEGORIES] + [
                         {"category": "process", "zh": "绘画过程步骤",
                          "from": "case_process 页（逐步操作记录，不是可复用条款）"},
                         {"category": "other", "zh": "其他", "from": ""}],
            "clauses": clauses}


# ---------------------------------------------------------------------------
# 案例配色库
# ---------------------------------------------------------------------------

def _norm_name(n: str) -> str:
    """色名归一化：去空白、去首尾标点，用于和 130 色表比对。"""
    return re.sub(r"[\s　·・]+", "", n or "").strip("、，,。.")


def build_palette_library(pages_dir: str, ncd_path: str, log_callback=None) -> dict:
    """从所有 `case_analysis` 页建案例配色库。

    色名**原样保留**；能对上 130 色表的补标称值，对不上的显式标注为固有色名。
    """
    def log(m):
        if log_callback:
            log_callback(m)

    ncd = json.load(io.open(ncd_path, encoding="utf-8"))
    table = {}
    for c in ncd.get("colors", []) + ncd.get("neutrals", []):
        table[_norm_name(c["name"])] = c

    palettes, unmatched = [], []
    for num in range(1, 151):
        path = os.path.join(pages_dir, "page-%03d.md" % num)
        if not os.path.isfile(path):
            continue
        meta, body = parse_page_markdown(io.open(path, encoding="utf-8").read())
        if meta.get("page_type") != "case_analysis" or not meta.get("palette"):
            continue

        names = [s.strip() for s in str(meta["palette"]).split("|") if s.strip()]
        colors = []
        for i, raw in enumerate(names, 1):
            # meta 里可能写成「蓝紫色 #2e2f5d」——带实测 hex
            m = re.match(r"^(.*?)\s*(#[0-9a-fA-F]{6})$", raw)
            name, measured = (m.group(1).strip(), m.group(2)) if m else (raw, None)
            hit = table.get(_norm_name(name))
            entry = {"order": i, "name": name, "measured_hex": measured}
            if hit:
                entry.update({"name_kind": "色彩系统名", "ncd_no": hit.get("no"),
                              "rgb": hit.get("rgb"), "cmyk": hit.get("cmyk"),
                              "hue": hit.get("hue"), "tone": hit.get("tone")})
            else:
                entry.update({"name_kind": "固有色名", "ncd_no": None,
                              "rgb": None, "cmyk": None, "hue": None, "tone": None})
                unmatched.append((num, name))
            colors.append(entry)

        # 从正文里抽「色相平衡 / 色调平衡」这两段的结论句
        facts = {}
        for seg_title, seg_text in re.findall(
                r"^## (.+?)$\n\n((?:>.*\n?)+)", body, re.M):
            t = _clean_md(re.sub(r"^>\s?", "", seg_text.strip(), flags=re.M))
            t = " ".join(t.split())
            if "色相平衡" in seg_title:
                facts["hue_balance"] = t
            elif "色调平衡" in seg_title:
                facts["tone_balance"] = t
            elif "色调范围" in seg_title or "色相和色调" in seg_title:
                facts["hue_tone_range"] = t
        # 面积占比：连**所在的那句话**一起留。
        # 只留数字会误导 —— p140 里 80% 指的是紫色系占的面积，
        # 40% 指的是高彩度鲜艳色调占的比例，两个 40% 各有所指，
        # 光看 [80, 40, 40] 根本分不清。
        area = []
        for m in re.finditer(r"[^。；\n]{0,40}?\d+\s*%[^。；\n]{0,20}", body):
            s = _clean_md(" ".join(re.sub(r"^[>\-|]\s*", "", m.group(0)).split()))
            if s and s not in area:
                area.append(s)

        palettes.append({
            "slug": "color-case-p%03d" % num,
            "title": meta.get("title"),
            "artist": meta.get("artist"),
            "chapter": meta.get("chapter"),
            "story_theme": meta.get("story_theme"),
            "source": {"pdf_page": num, "printed_page": meta.get("printed_page")},
            "colors": colors,
            "facts": facts,
            "area_statements": area,
            "tags": [t for t in str(meta.get("tags") or "").split() if t],
        })

    matched = sum(1 for p in palettes for c in p["colors"] if c["ncd_no"])
    total = sum(len(p["colors"]) for p in palettes)
    log("[palettes] %d 个案例，共 %d 个色格；对上 130 色表 %d 个（%.0f%%），"
        "固有色名 %d 个" % (len(palettes), total, matched, 100.0 * matched / max(total, 1),
                          total - matched))
    log("[palettes] 面积占比句子抽到 %d 条（带上下文，不是裸数字）"
        % sum(len(p["area_statements"]) for p in palettes))
    return {"version": 1, "count": len(palettes),
            "ncd_table": os.path.basename(ncd_path),
            "naming_note": ("书里混用「固有色名」与「色彩系统名」两套命名"
                            "（PDF 34 笔记已明说）。本表**不做就近映射**："
                            "对不上的色名 system 字段留空，绝不拿相似色名顶替。"),
            "palettes": palettes,
            "unmatched_color_names": sorted({n for _, n in unmatched})}


# ---------------------------------------------------------------------------
# 五套词表
# ---------------------------------------------------------------------------

#: 每套词表的**来源小节标题**（我在转录时用的），据此定位。
#: 小节名必须**逐字对上**我在 md 里写的 `## ` 标题，写错就整段抓不到
#: （第一版把 p10 的小节写成「了解色彩搭配的方法和技巧」，
#: 实际表在「欣赏配色表达的故事」下，于是 topic 词表整抓不到）。
LEXICON_SOURCES = [
    {"key": "topic", "zh": "题材五切入点", "pdf_page": 10,
     "sections": ["欣赏配色表达的故事"]},
    {"key": "emotion_quadrant", "zh": "情绪环状模式（罗素）", "pdf_page": 53,
     "sections": ["情绪环状模式", "用这个模型拆《即使这样也要画下去》"]},
    {"key": "imagery", "zh": "形象坐标意象词", "pdf_page": 74,
     "sections": ["形象坐标图上的完整词表（已按 2.6 倍放大逐格核对）"]},
    # p60 用的是**引用块**而不是表格（`> **甜味**：以红色为主……`），
    # 所以这一套走 blockquote 抽取。
    {"key": "taste", "zh": "味觉与口感", "pdf_page": 60,
     "sections": ["五味", "另外三种口感（页面另列）"], "mode": "quote"},
    {"key": "sound", "zh": "音乐与色彩", "pdf_page": 137,
     "sections": ["明度 ↔ 音高 / 音强 / 音厚", "彩度 ↔ 音色",
                  "配色手法 ↔ 音乐结构"]},
]

#: 引用块里的「**词**：解释」形态
QUOTE_ITEM = re.compile(r"^>\s*\*\*(.+?)\*\*\s*[：:]\s*(.+)$")


def _extract_quotes(body: str, sections) -> list:
    """从引用块里抽 `> **词**：解释` 形态的词条（p60 的味觉表就是这种）。"""
    out, cur = [], None
    for line in body.split("\n"):
        if line.startswith("## "):
            cur = line[3:].strip()
        elif line.startswith("### "):
            continue
        if cur is None or not any(s in cur for s in sections):
            continue
        m = QUOTE_ITEM.match(line.strip())
        if m:
            out.append({"group": cur, "key": _clean_md(m.group(1)),
                        "value": _clean_md(m.group(2)), "note": ""})
    return out


def build_lexicons(pages_dir: str, log_callback=None) -> dict:
    """把书里五套封闭词表抓出来，各自带来源页。"""
    def log(m):
        if log_callback:
            log_callback(m)

    out = {}
    for spec in LEXICON_SOURCES:
        path = os.path.join(pages_dir, "page-%03d.md" % spec["pdf_page"])
        if not os.path.isfile(path):
            log("[lexicons] 缺页 %d，跳过 %s" % (spec["pdf_page"], spec["key"]))
            continue
        meta, body = parse_page_markdown(io.open(path, encoding="utf-8").read())
        if spec.get("mode") == "quote":
            entries = _extract_quotes(body, spec["sections"])
            picked = [(spec["sections"][0], [])]
        else:
            tables = _split_tables(body)
            picked = []
            for want in spec["sections"]:
                for title, rows in tables:
                    if want in title or title in want:
                        picked.append((title, rows))
            entries = []
            for title, rows in picked:
                for head, cells in rows:
                    cells = [_clean_md(c) for c in cells]
                    if len(cells) >= 2:
                        entries.append({"group": title, "key": cells[0],
                                        "value": cells[1],
                                        "note": cells[2] if len(cells) > 2 else ""})
        out[spec["key"]] = {
            "zh": spec["zh"],
            "source": {"pdf_page": spec["pdf_page"], "printed_page": meta.get("printed_page"),
                       "title": meta.get("title")},
            "count": len(entries),
            "entries": entries,
            "_sections_found": [t for t, _ in picked],
        }
        log("[lexicons] %-16s 页 %-4d 抓到 %d 条（小节：%s）"
            % (spec["key"], spec["pdf_page"], len(entries),
               " / ".join(t for t, _ in picked) or "无"))
    return {"version": 1,
            "note": "五套封闭词表，全部来自书中原表；每条都能由 source.pdf_page 回查原文。",
            "lexicons": out}


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def load_kb(data_dir: str) -> dict:
    """载入三份产出 + 案例索引。"""
    out = {}
    for name in ("palette_library", "color_clauses", "lexicons", "case_index"):
        p = os.path.join(data_dir, name + ".json")
        out[name] = json.load(io.open(p, encoding="utf-8")) if os.path.isfile(p) else None
    return out


def resolve_color_case(query: str, data_dir: str):
    """按 slug / 作品名 / 画师 / 标签 / 色名 找案例，返回 (case, 命中理由) 或 (None, 原因)。

    匹配优先级：slug 精确 → 作品名精确 → 作品名包含 → 画师 → 色名 → 标签。
    **命中理由要返回给调用方** —— 用户搜「秋天」时到底命中了哪条、凭什么，
    必须能说清楚，不能让检索看起来像黑盒。
    """
    kb = load_kb(data_dir)
    pal = kb.get("palette_library")
    if not pal:
        return None, "palette_library.json 不存在，先跑 kb"
    q = (query or "").strip()
    if not q:
        return None, "query 为空"

    cases = pal["palettes"]
    for c in cases:
        if c["slug"] == q:
            return c, "slug 精确命中"
    for c in cases:
        if c["title"] == q:
            return c, "作品名精确命中"
    for c in cases:
        if q in (c["title"] or ""):
            return c, "作品名包含"
    for c in cases:
        if q in (c["artist"] or ""):
            return c, "画师命中"
    for c in cases:
        if any(q in (col["name"] or "") for col in c["colors"]):
            return c, "色名命中（该案例用到过这个颜色）"
    for c in cases:
        if any(q in t for t in c["tags"]):
            return c, "标签命中"
    return None, "没找到匹配的案例"


#: 这几页是全书的总纲 / 定义页，任何案例都可以附上它们的条款
#: 只收**方法论页**。p117 虽然编号接近，但它是《伯劳鸟/鸟》的配色亮点页
#: （属作品分析），混进来会把个案当成通用规则。
CORE_CLAUSE_PAGES = (79, 110, 129, 146)


def compose_color_block(case, data_dir: str, with_core: bool = True,
                        max_rules: int = 14) -> str:
    """把一个案例 + 相关条款拼成**可直接追加到生图请求**的文本块。

    只输出书里有的内容：色板来自该作品的「主色调」栏，规则来自该作品的版面
    与总纲页，**一句都不额外编**。
    """
    kb = load_kb(data_dir)
    cl = (kb.get("color_clauses") or {}).get("clauses") or []

    # 该作品的「版组」：作品页本身 + 紧随其后、标题带同一作品名的亮点/过程页
    base = case["source"]["pdf_page"]
    head = (case["title"] or "")[:4]

    def _in_group(c):
        src = c["source"]
        if src["pdf_page"] == base:
            return True
        if src.get("page_type") not in ("case_highlight", "case_process"):
            return False
        if not (0 < src["pdf_page"] - base <= 7):
            return False
        return bool(head) and head in (src.get("title") or "")

    # `process`（过程步骤）与 `other` 不是可复用的配色规则，
    # 混进规则清单会让下游把它们当成配色指令；单独留给"过程"用途。
    own = [c for c in cl if _in_group(c) and c["category"] not in ("process", "other")]
    # 核心条款只取书里那五步 + 两条注意事项，且**按步聚顺序**排，
    # 这样块里的规则顺序与 PDF 146 的流程一致，读起来是一套方法而不是一锅杂烩。
    core_order = ["story", "tone", "selection", "base", "accent",
                  "contrast", "assimilation", "relativity", "area"]
    core = [c for c in cl if with_core and c["source"]["pdf_page"] in CORE_CLAUSE_PAGES
            and c["evidence_type"] == "stated" and c["category"] in core_order]

    lines = ["COLOR SCHEME — 《%s》（%s）" % (case["title"], case["artist"])]
    if case.get("story_theme"):
        lines.append("Story: %s" % case["story_theme"])
    lines.append("Palette (5):")
    for c in case["colors"]:
        if c["ncd_no"]:
            lines.append("  %d. %s  → NCD #%03d  RGB %s  %s/%s"
                         % (c["order"], c["name"], c["ncd_no"],
                            "-".join(str(x) for x in c["rgb"]), c["hue"], c["tone"]))
        else:
            lines.append("  %d. %s  → 固有色名（该书 130 色表里没有同名条目）"
                         % (c["order"], c["name"]))
    f = case.get("facts") or {}
    for key, label in (("hue_tone_range", "Hue/tone range"),
                       ("hue_balance", "Hue balance"),
                       ("tone_balance", "Tone balance")):
        if f.get(key):
            lines.append("%s: %s" % (label, f[key][:160]))
    for s in (case.get("area_statements") or [])[:4]:
        lines.append("Area: %s" % s)

    if own:
        lines.append("Rules from this work's own pages:")
        for c in own[:max_rules]:
            lines.append("  · [%s] %s" % (c["category_zh"], c["item"]))
    if core:
        lines.append("Core rules from the book's method pages (PDF 146 的五步法):")
        seen = set()
        for cat in core_order:
            hit = next((c for c in core if c["category"] == cat), None)
            if hit:
                lines.append("  · [%s] %s" % (hit["category_zh"], hit["item"]))
                seen.add(cat)
            if len(seen) >= max_rules:
                break
    return "\n".join(lines)


def audit_knowledge(out_dir: str, log_callback=None) -> dict:
    """检查三份产出是否自洽：条条款可回查、色名不凭空出现、词表非空。"""
    def log(m):
        if log_callback:
            log_callback(m)

    problems = []
    cl_path = os.path.join(out_dir, "color_clauses.json")
    pa_path = os.path.join(out_dir, "palette_library.json")
    lx_path = os.path.join(out_dir, "lexicons.json")
    for p in (cl_path, pa_path, lx_path):
        if not os.path.isfile(p):
            return {"error": "缺文件 " + os.path.basename(p)}

    cl = json.load(io.open(cl_path, encoding="utf-8"))
    pa = json.load(io.open(pa_path, encoding="utf-8"))
    lx = json.load(io.open(lx_path, encoding="utf-8"))

    pages_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(out_dir))),
                             "docs", "261003-color-improve", "book-html", "pages")

    # 1. 每条 `stated` 条款都要能在来源页里找到出处。
    #    **比对的是 `detail`（书里的原话）而不是 `item`（我自己的概括）** ——
    #    第一版拿 `item` 去比对，64 条全报"查不到"，全是误报。
    placeholder = re.compile(r"^(原文.*|原页注释|步骤\s*\d+|图注|\d+\s*[条组]图注|\d+\s*[条组]引注|"
                             r"\d+\s*[条组]色板|引注|同左.*|（同左.*|[^，。]{0,10}说明|"
                             r"图\s*[a-z]\s*打叉|散点图读数|柱状图读数|$)")
    punct = re.compile(r"[\s「」『』（）()【】\[\]，,。.、；;：:！!？?—\-…·|`*]")

    def _norm(s):
        return punct.sub("", s or "")

    def _has_common(probe, page_norm, least=6):
        """probe 与页面正文是否有 ≥least 字的公共子串。

        门槛取 6 而不是 8：书里不少条款本身就是短的
        （如「两种色调相搭配」只有 7 字），门槛太高会全部误报。
        `punct` 已经把引号/标点/markdown 记号都抹平，所以 6 字足够有辨识度。
        """
        p = _norm(probe)
        if len(p) < least:
            return True                      # 太短没法判，不报错
        for n in range(min(len(p), 40), least - 1, -1):
            for i in range(0, len(p) - n + 1):
                if p[i:i + n] in page_norm:
                    return True
        return False

    for c in cl["clauses"]:
        pg = c["source"]["pdf_page"]
        f = os.path.join(pages_dir, "page-%03d.md" % pg)
        if not os.path.isfile(f):
            problems.append({"kind": "条款来源页不存在", "id": c["id"], "page": pg})
            continue
        page_norm = _norm(io.open(f, encoding="utf-8").read())
        # 逐个单元格试：只要**任意一格**能在页里找到，就算这条有出处。
        # 单元格全是占位词（"三条图注"这种）时无从判定 → 归到 unverifiable，不算错。
        candidates = [x for x in (c.get("cells") or [c["item"], c["detail"]])
                      if x and x.strip() and not placeholder.match(x.strip())]
        if not candidates:
            c["evidence_type"] = "stated_unverifiable"
            continue
        if not any(_has_common(x, page_norm) for x in candidates):
            problems.append({"kind": "条款在来源页里查不到", "id": c["id"],
                             "page": pg, "evidence_type": c["evidence_type"],
                             "cells": candidates[:3]})

    # 2. 案例色格：对不上色表的必须显式标固有色名，不许有空缺字段
    for p in pa["palettes"]:
        if len(p["colors"]) != 5:
            problems.append({"kind": "案例不是 5 色", "slug": p["slug"],
                             "n": len(p["colors"])})
        for c in p["colors"]:
            if not c["name"]:
                problems.append({"kind": "色格无色名", "slug": p["slug"], "order": c["order"]})
            if c["name_kind"] == "色彩系统名" and not c["rgb"]:
                problems.append({"kind": "系统名但无 RGB", "slug": p["slug"], "name": c["name"]})

    # 3. 词表非空
    for k, v in lx["lexicons"].items():
        if v["count"] == 0:
            problems.append({"kind": "词表为空", "key": k, "page": v["source"]["pdf_page"]})

    log("[audit] 条款 %d 条 / 案例 %d 个 / 词表 %d 套"
        % (cl["count"], pa["count"], len(lx["lexicons"])))
    if problems:
        for pr in problems[:40]:
            log("  [!] %s %s" % (pr["kind"], {k: v for k, v in pr.items() if k != "kind"}))
        log("  [x] 共 %d 个问题" % len(problems))
    else:
        log("  [ok] 三份产出自洽：条款可回查、色格无空缺、词表非空")
    return {"clauses": cl["count"], "palettes": pa["count"],
            "lexicons": len(lx["lexicons"]), "problems": problems}


def build_all(pages_dir: str, data_dir: str, ncd_path: str, log_callback=None) -> dict:
    """一把梭：三份产出全部重建并自检。"""
    def log(m):
        if log_callback:
            log_callback(m)

    os.makedirs(data_dir, exist_ok=True)
    result = {}
    for name, fn, args in (
            ("color_clauses.json", extract_clauses, (pages_dir,)),
            ("palette_library.json", build_palette_library, (pages_dir, ncd_path)),
            ("lexicons.json", build_lexicons, (pages_dir,))):
        data = fn(*args, log_callback=log)
        result[name] = data
        # 原子写：先写临时文件再替换，避免中途失败留下半个 JSON
        tmp = os.path.join(data_dir, name + ".tmp")
        with io.open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(data_dir, name))
        log("[write] %s（%.1f KB）" % (name, os.path.getsize(os.path.join(data_dir, name)) / 1024))

    # 案例索引单独出一份小的，便于检索
    idx = [{"slug": p["slug"], "title": p["title"], "artist": p["artist"],
            "story_theme": p["story_theme"], "pdf_page": p["source"]["pdf_page"],
            "printed_page": p["source"]["printed_page"],
            "colors": [c["name"] for c in p["colors"]],
            "tags": p["tags"]} for p in result["palette_library.json"]["palettes"]]
    with io.open(os.path.join(data_dir, "case_index.json"), "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "count": len(idx), "cases": idx},
                  fh, ensure_ascii=False, indent=1)
    log("[write] case_index.json（%d 个案例）" % len(idx))

    audit = audit_knowledge(data_dir, log_callback=log)
    return {"built": {k: v.get("count") for k, v in result.items()}, "audit": audit}
