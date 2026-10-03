# -*- coding: utf-8 -*-
"""把一本扫描版配色书（PDF）变成项目可用的配色知识库。无 PyQt，可无头跑。

用法速查（印刷页码 = PDF 页序号 − 5，本书固定如此）：

  # 0. 先看这本书的结构表，确认页码换算与页型分布
  python tools/color_knowledge.py map --pages-dir cache/temp/color-book/pages

  # 1. 从 PDF 抽页图（不重编码，原样切出内嵌 JPEG）
  python tools/color_knowledge.py extract --pdf docs/261003-color-improve/15659159.pdf \\
      --out cache/temp/color-book/pages

  # 2. 逐页视觉提取（可续跑：已存在的页跳过）
  python tools/color_knowledge.py analyze --pages-dir cache/temp/color-book/pages \\
      --out cache/temp/color-book/page-json --only 13,14,20,22

  # 3. 色板实测（OpenCV，不问模型要 hex）
  python tools/color_knowledge.py measure --pages-dir cache/temp/color-book/pages \\
      --out cache/temp/color-book/swatches

  # 4. 合成知识库（把实测色值贴回视觉 JSON）
  python tools/color_knowledge.py merge --page-json cache/temp/color-book/page-json \\
      --swatches cache/temp/color-book/swatches --out data/color-knowledge

  # 5. 把案例落成画风条目（dry-run 先看，确认后才写 conf/config-styles.json）
  python tools/color_knowledge.py styles --kb data/color-knowledge --dry-run
"""
import argparse
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

DEFAULT_PAGES = os.path.join(BASE, "cache", "temp", "color-book", "pages")
DEFAULT_PAGE_JSON = os.path.join(BASE, "cache", "temp", "color-book", "page-json")
DEFAULT_SWATCHES = os.path.join(BASE, "cache", "temp", "color-book", "swatches")
DEFAULT_KB = os.path.join(BASE, "data", "color-knowledge")


def _parse_ints(text):
    if not text:
        return None
    out = []
    for part in str(text).replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _log(msg):
    print(msg, flush=True)


def cmd_extract(args):
    from utils.color_knowledge import extract_pdf_pages

    rng = _parse_ints(args.pages)
    r = extract_pdf_pages(args.pdf, args.out, page_range=(rng[0], rng[-1]) if rng else None, log_callback=_log)
    _log(json.dumps({k: v for k, v in r.items() if k != "pages"}, ensure_ascii=False))
    if r.get("skipped"):
        _log(f"[color-knowledge] 有 {r['skipped']} 页被跳过 —— 抽页不算成功，先查 manifest 里的 skipped 原因")
        return 1
    return 0


def cmd_targets(args):
    """生成 tone 工序的目标参考图（无纸白、无文字的整页画面）。"""
    from utils.color_knowledge import make_tone_targets

    out = args.out or os.path.join(args.kb, "targets")
    made = make_tone_targets(args.page_json, args.pages_dir, out,
                             from_page=args.from_page, log_callback=_log)
    return 0 if made else 1


def cmd_analyze(args):
    from utils.color_knowledge import analyze_pages

    r = analyze_pages(args.pages_dir, args.out, offset=args.offset, only=_parse_ints(args.only),
                      force=args.force, limit=args.limit, model=args.model, log_callback=_log)
    _log(json.dumps(r, ensure_ascii=False))
    return 0 if not r.get("failed") else 1


def cmd_measure(args):
    from utils.color_knowledge import measure_swatches

    manifest_path = os.path.join(args.pages_dir, "manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8"))["pages"] if os.path.isfile(manifest_path) else []
    only = _parse_ints(args.only)
    os.makedirs(args.out, exist_ok=True)
    done = 0
    for m in manifest:
        if not m.get("file") or (only and m["pdf_page"] not in only):
            continue
        dest = os.path.join(args.out, "page-%03d.json" % m["pdf_page"])
        if os.path.isfile(dest) and not args.force:
            continue
        sw = measure_swatches(os.path.join(args.pages_dir, m["file"]))
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(sw, f, ensure_ascii=False, indent=1)
        done += 1
        _log(f"[color-knowledge] 第 {m['pdf_page']} 页量到 {len(sw)} 个圆形色板")
    _log(f"[color-knowledge] 色板实测完成：{done} 页 → {args.out}")
    return 0


def cmd_merge(args):
    import glob as _g

    from utils.color_knowledge import attach_swatches, build_knowledge_base

    attached = 0
    for path in sorted(_g.glob(os.path.join(args.page_json, "page-*.json"))):
        page = json.load(open(path, encoding="utf-8"))
        pdf_page = page.get("pdf_page") or int(path[-8:-5])
        sw_path = os.path.join(args.swatches, "page-%03d.json" % pdf_page)
        if os.path.isfile(sw_path):
            sw = json.load(open(sw_path, encoding="utf-8"))
            img = os.path.join(args.pages_dir, "page-%03d.jpg" % pdf_page)
            attach_swatches(page, sw, image_path=img if os.path.isfile(img) else "")
            attached += 1
            with open(path, "w", encoding="utf-8") as f:
                json.dump(page, f, ensure_ascii=False, indent=2)
    _log(f"[color-knowledge] 贴回实测色值：{attached} 页")
    written = build_knowledge_base(args.page_json, args.out, log_callback=_log)
    for k, v in written.items():
        _log(f"  {k} → {os.path.relpath(v, BASE)}")
    return 0


def cmd_map(args):
    """打印结构表：PDF 页 / 印刷页 / 页型（有分析 JSON 时带标题）。"""
    import glob as _g

    manifest_path = os.path.join(args.pages_dir, "manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8"))["pages"] if os.path.isfile(manifest_path) else []
    for m in manifest:
        if not m.get("file"):
            continue
        row = {"pdf_page": m["pdf_page"], "printed_page": m["pdf_page"] - args.offset}
        j = os.path.join(args.page_json or DEFAULT_PAGE_JSON, "page-%03d.json" % m["pdf_page"])
        if os.path.isfile(j):
            p = json.load(open(j, encoding="utf-8"))
            row.update({"page_type": p.get("page_type"),
                        "title": p.get("work_title") or p.get("title"),
                        "chapter": p.get("chapter"),
                        "swatches": len(p.get("palette_row_measured") or [])})
        _log(json.dumps(row, ensure_ascii=False))
    return 0


def cmd_verify(args):
    """把所有带实测色板的页拼成一张核对大图（人眼扫一遍 = 完成一轮色值验收）。"""
    from utils.color_knowledge import build_check_sheet

    out = args.out or os.path.join(args.kb, "palette-check.png")
    got = build_check_sheet(args.page_json, args.pages_dir, out, limit=args.limit)
    if not got:
        _log("[color-knowledge] 没有任何一页测到色板排，先跑 measure + merge")
        return 1
    _log(f"[color-knowledge] 核对图 → {os.path.relpath(got, BASE)}（打开看一眼色名与色块是否对得上）")
    return 0


def cmd_figures(args):
    """把扫描页里的配图（插图 / 图表面板 / 色板行）抠成独立 PNG。"""
    from utils.color_knowledge import extract_page_figures

    out = args.out or os.path.join(args.pages_dir, "..", "figures")
    # 核对叠加图是**工序中间产物**，不是站点素材（150 张约 200 MB）。
    # 默认落 cache/temp/，别往交付目录里塞。
    overlay = args.overlay_dir
    if overlay is None:
        overlay = os.path.join(BASE, "cache", "temp", "color-book", "overlays")
    r = extract_page_figures(args.pages_dir, out, only=_parse_ints(args.only),
                             min_area_ratio=args.min_area, min_colorful=args.min_colorful,
                             overwrite=args.force, overlay_dir=overlay,
                             override_dir=args.override_dir, log_callback=_log)
    if overlay:
        _log(f"[color-knowledge] 核对叠加图 → {os.path.relpath(overlay, BASE)}（看一眼方框扣得对不对）")
    return 0 if r.get("total") else 1


def cmd_html(args):
    """把整本书渲染成「目录页 + 每章一页」的多文件站点。"""
    from utils.color_knowledge import build_book_site

    root = args.pages_md
    if os.path.isdir(os.path.join(root, "pages")):
        root = os.path.abspath(root)
    else:
        root = os.path.dirname(os.path.abspath(root))
    r = build_book_site(root, out_dir=args.out, log_callback=_log)
    return 0 if r else 1


def cmd_images(args):
    """把扫描页降采样成站点用的页图（站点必须自带图，不能指向 cache/）。"""
    import glob as _g

    from PIL import Image

    os.makedirs(args.out, exist_ok=True)
    n = 0
    for p in sorted(_g.glob(os.path.join(args.pages_dir, "page-*.jpg"))):
        dest = os.path.join(args.out, os.path.basename(p))
        if os.path.isfile(dest) and not args.force:
            continue
        im = Image.open(p).convert("RGB")
        im.thumbnail((args.max_edge, int(args.max_edge * 1.6)), Image.LANCZOS)
        im.save(dest, quality=args.quality, optimize=True, progressive=True)
        n += 1
    _log(f"[color-knowledge] 页图写出 {n} 张（已有跳过）→ {os.path.relpath(args.out, BASE)}")
    return 0


def cmd_stubs(args):
    """给还没转录的页写占位 md：带完整 `<!--meta-->` 页头 + 一句"待转录"。

    这样每一页在站点里都有标题、章节、页型、印刷页码，也能被创建者直接打开续写；
    已有正文的 md **不会被覆盖**（除非 `--force`）。
    """
    import glob as _g

    from utils.color_knowledge import load_book_structure, page_meta

    root = os.path.abspath(args.root)
    pages_dir = os.path.join(root, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    structure = load_book_structure(root)
    have = {int(re.search(r"(\d+)", os.path.basename(p)).group(1))
            for p in _g.glob(os.path.join(pages_dir, "page-*.md"))}
    nums = sorted({int(re.search(r"(\d+)", os.path.basename(p)).group(1))
                   for p in _g.glob(os.path.join(root, "images", "page-*.jpg"))})
    made = 0
    for num in nums:
        m = page_meta(structure, num)
        dest = os.path.join(pages_dir, "page-%03d.md" % num)
        if os.path.isfile(dest) and not args.force:
            continue
        head = ["<!--meta", "pdf_page: %d" % num,
                "printed_page: %s" % (m["printed"] if m["printed"] else "null"),
                "page_type: %s" % (m["kind"] or "other"),
                "chapter: %s" % (m["chapter_title"] or "")]
        if m.get("theme"):
            head.append("story_theme: %s" % m["theme"])
        head.append("title: %s" % (m["title"] or "PDF 第 %d 页" % num))
        head.append("transcribed: false")
        head.append("-->")
        body = "# %s\n\n> 尚未逐字转录。\n" % (m["title"] or "PDF 第 %d 页" % num)
        with open(dest, "w", encoding="utf-8") as f:
            f.write("\n".join(head) + "\n\n" + body)
        made += 1
    _log(f"[color-knowledge] 占位页写出 {made} 份（已存在的跳过）→ {os.path.relpath(pages_dir, BASE)}")
    return 0


def cmd_color_block(args):
    """按作品名 / 画师 / 色名 / 标签 取一条案例，输出可直接用的配色块。"""
    from utils.color_extract import resolve_color_case, compose_color_block

    if args.list:
        from utils.color_extract import load_kb
        kb = load_kb(args.data)
        idx = kb.get("case_index") or {"cases": []}
        _log("[color] 共 %d 个案例：" % idx["count"])
        for c in idx["cases"]:
            _log("  %-12s %-16s %-14s %s"
                 % (c["slug"], (c["title"] or "")[:16], (c["artist"] or "")[:14],
                    " / ".join(c["colors"])))
        return 0

    case, why = resolve_color_case(args.query, args.data)
    if not case:
        _log("[color] %s（query=%r）" % (why, args.query))
        return 1
    _log("[color] 命中《%s》（%s）—— %s"
         % (case["title"], case["artist"], why))
    out = compose_color_block(case, args.data, with_core=not args.no_core,
                              max_rules=args.max_rules)
    _log("-" * 72)
    _log(out)
    _log("-" * 72)
    if args.out:
        import io as _io
        _io.open(args.out, "w", encoding="utf-8").write(out)
        _log("[color] 已写出 %s（%d 字符）" % (args.out, len(out)))
    return 0


def cmd_check(args):
    """抽检整站（每转录 10 页跑一次）。"""
    from utils.color_knowledge import audit_book_site

    r = audit_book_site(args.root, short_chars=args.short_chars, log_callback=_log)
    if r.get("error"):
        _log("[audit] " + r["error"])
        return 1
    return 0


def cmd_kb(args):
    """把转录好的页面整理成结构化配色知识（条款 / 案例配色 / 词表）。"""
    from utils.color_extract import build_all

    pages_dir = os.path.join(args.root, "pages")
    data_dir = args.data
    ncd = os.path.join(BASE, "data", "color-knowledge", "ncd-palette-130.json")
    if not os.path.isdir(pages_dir):
        _log("[kb] 找不到页面目录：%s" % pages_dir)
        return 1
    if not os.path.isfile(ncd):
        _log("[kb] 找不到 130 色表：%s" % ncd)
        return 1
    r = build_all(pages_dir, data_dir, ncd, log_callback=_log)
    if args.audit_only:
        return 1 if r["audit"]["problems"] else 0
    _log("[kb] 完成：%s" % r["built"])
    return 1 if r["audit"]["problems"] else 0


def cmd_styles(args):
    """把案例配色落成 `conf/config-styles.json` 的 `color-*` 条目（默认只 dry-run）。"""
    from utils.color_style_builder import build_color_style_entries, write_style_entries

    entries = build_color_style_entries(args.kb, min_palette=args.min_palette,
                                        model=args.model, optimize=not args.no_llm, log_callback=_log)
    _log(f"[color-knowledge] 生成 {len(entries)} 条配色画风条目")
    for e in entries:
        _log(f"  {e['name']}：{'/'.join(c.get('hex', '?') for c in e.get('_palette', [])[:5])}")
    if args.dry_run:
        _log("[color-knowledge] --dry-run：未写入 conf/config-styles.json")
        return 0
    n = write_style_entries(entries, path=args.styles_file, log_callback=_log)
    _log(f"[color-knowledge] 写入 {n} 条 → {args.styles_file}")
    return 0


def cmd_pilot(args):
    from utils.color_experiment import (load_spec, run_pilot, select_profile,
                                       summarize_pilot, evaluate_pilot)
    spec = load_spec(args.spec)
    if args.dry_run and args.action != "generate":
        _log("--dry-run only supported for generate")
        return 2
    if args.action == "generate":
        result = run_pilot(spec, args.out, model=args.model, dry_run=args.dry_run,
                           retry_failed=args.retry_failed, log=_log)
        return 1 if any(s["status"] != "success" for s in result.get("samples", [])) else 0
    if args.action == "select":
        result = select_profile(spec, args.out)
    elif args.action == "summarize":
        result = summarize_pilot(args.out)
    else:
        result = evaluate_pilot(args.out)
    _log(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser(description="扫描版配色书 → 项目配色知识库")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pilot", help="隔离的 Gemini 配色对照实验 / 自动选择 / 色彩评价")
    p.add_argument("action", choices=["generate", "select", "summarize", "evaluate"])
    p.add_argument("--spec", default=os.path.join(BASE, "prompts", "color-knowledge", "pilot-spring-fishing.json"))
    p.add_argument("--out", required=True, help="data/test-result/ 下的独立实验目录")
    p.add_argument("--model", default=None, help="本次 Gemini 模型覆盖，不写配置")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--retry-failed", action="store_true", help="显式重试失败/中断记录，成功产物复用")
    p.set_defaults(func=cmd_pilot)

    p = sub.add_parser("extract", help="从 PDF 抽页图")
    p.add_argument("--pdf", required=True)
    p.add_argument("--out", default=DEFAULT_PAGES)
    p.add_argument("--pages", default="", help="如 13-40")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("analyze", help="逐页视觉提取（可续跑）")
    p.add_argument("--pages-dir", default=DEFAULT_PAGES)
    p.add_argument("--out", default=DEFAULT_PAGE_JSON)
    p.add_argument("--only", default="", help="如 13,14,22")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=5, help="printed = pdf_page - offset")
    p.add_argument("--model", default="gpt-6-sol")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("measure", help="OpenCV 实测圆形色板")
    p.add_argument("--pages-dir", default=DEFAULT_PAGES)
    p.add_argument("--out", default=DEFAULT_SWATCHES)
    p.add_argument("--only", default="")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("merge", help="贴回实测色值并合成知识库")
    p.add_argument("--page-json", default=DEFAULT_PAGE_JSON)
    p.add_argument("--swatches", default=DEFAULT_SWATCHES)
    p.add_argument("--pages-dir", default=DEFAULT_PAGES)
    p.add_argument("--out", default=DEFAULT_KB)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("map", help="打印页型结构表")
    p.add_argument("--pages-dir", default=DEFAULT_PAGES)
    p.add_argument("--page-json", default=DEFAULT_PAGE_JSON)
    p.add_argument("--offset", type=int, default=5)
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("targets", help="生成 tone 目标参考图（无纸白无文字的整页画面）")
    p.add_argument("--page-json", default=DEFAULT_PAGE_JSON)
    p.add_argument("--pages-dir", default=DEFAULT_PAGES)
    p.add_argument("--kb", default=DEFAULT_KB)
    p.add_argument("--out", default="")
    p.add_argument("--from-page", default="auto", choices=["auto", "illustration", "analysis"])
    p.set_defaults(func=cmd_targets)

    p = sub.add_parser("verify", help="生成色板核对大图（人工验收）")
    p.add_argument("--page-json", default=DEFAULT_PAGE_JSON)
    p.add_argument("--pages-dir", default=DEFAULT_PAGES)
    p.add_argument("--kb", default=DEFAULT_KB)
    p.add_argument("--out", default="")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("figures", help="把扫描页里的配图抠成独立 PNG")
    p.add_argument("--pages-dir", default=DEFAULT_PAGES)
    p.add_argument("--out", default="")
    p.add_argument("--overlay-dir", default=None)
    p.add_argument("--override-dir", default="", help="人工框覆盖目录（overrides/page-NNN.json）")
    p.add_argument("--only", default="")
    p.add_argument("--min-area", type=float, default=0.006, help="块面积 / 整页面积 的下限")
    p.add_argument("--min-colorful", type=float, default=0.05, help="块内有彩色像素占比的下限")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("html", help="整本书 → 目录页 + 每章一页的静态站点")
    p.add_argument("--pages-md", default=os.path.join(BASE, "docs", "261003-color-improve", "book-html"),
                   help="book-html 根目录（含 structure.json / pages/ / images/ / figures/）")
    p.add_argument("--images-dir", default="")
    p.add_argument("--out", default="")
    p.add_argument("--title", default="")
    p.set_defaults(func=cmd_html)

    p = sub.add_parser("kb", help="转录页 → 结构化配色知识（条款 / 案例配色 / 词表）")
    p.add_argument("--root", default=os.path.join(BASE, "docs", "261003-color-improve", "book-html"))
    p.add_argument("--data", default=os.path.join(BASE, "data", "color-knowledge"))
    p.add_argument("--audit-only", action="store_true", help="只自检，不重建")
    p.set_defaults(func=cmd_kb)

    p = sub.add_parser("color-block",
                       help="按作品名/画师/色名/标签取案例，输出可直接用的配色块")
    p.add_argument("--query", default="", help="作品名 / 画师 / 色名 / 标签 / slug")
    p.add_argument("--data", default=os.path.join(BASE, "data", "color-knowledge"))
    p.add_argument("--list", action="store_true", help="只列出全部案例")
    p.add_argument("--no-core", action="store_true", help="不附总纲页的通用条款")
    p.add_argument("--max-rules", type=int, default=14)
    p.add_argument("--out", default="", help="把配色块写到文件")
    p.set_defaults(func=cmd_color_block)

    p = sub.add_parser("check", help="抽检整站：断链 / 缺标题 / 体量 / 色名一致性")
    p.add_argument("--root", default=os.path.join(BASE, "docs", "261003-color-improve", "book-html"))
    p.add_argument("--short-chars", type=int, default=700)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("stubs", help="按 structure.json 给还没转录的页生成占位 md（含页头 meta）")
    p.add_argument("--root", default=os.path.join(BASE, "docs", "261003-color-improve", "book-html"))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_stubs)

    p = sub.add_parser("images", help="把扫描页降采样成站点用的页图")
    p.add_argument("--pages-dir", default=DEFAULT_PAGES)
    p.add_argument("--out", default=os.path.join(BASE, "docs", "261003-color-improve", "book-html", "images"))
    p.add_argument("--max-edge", type=int, default=1400)
    p.add_argument("--quality", type=int, default=82)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_images)

    p = sub.add_parser("styles", help="把案例配色落成画风条目")
    p.add_argument("--kb", default=DEFAULT_KB)
    p.add_argument("--styles-file", default=os.path.join(BASE, "conf", "config-styles.json"))
    p.add_argument("--model", default="gpt-5.6-luna")
    p.add_argument("--min-palette", type=int, default=3)
    p.add_argument("--no-llm", action="store_true", help="不调文本模型，直接用实测色值拼条款")
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--apply", dest="dry_run", action="store_false")
    p.set_defaults(func=cmd_styles)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
