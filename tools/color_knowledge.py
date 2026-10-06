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


def cmd_generation_knowledge(args):
    if args.action in ("plan", "preflight", "preview", "run", "review", "report"):
        from utils.color_experiment import (freeze_book_ablation, preview_book_ablation, run_book_ablation,
                                            review_book_ablation, book_ablation_report, book_preflight)
        if args.action == "plan":
            result = freeze_book_ablation(args.spec, args.out)
        elif args.action == "preflight":
            result = book_preflight(args.spec, args.out, compare_preview=args.compare_preview, log=_log)
        elif args.action == "preview":
            result = preview_book_ablation(args.spec, args.out)
        elif args.action == "run":
            result = run_book_ablation(args.spec, args.out, dry_run=args.dry_run, retry_failed=args.retry_failed, log=_log)
        elif args.action == "review":
            result = review_book_ablation(args.spec, args.out, log=_log)
        else:
            result = book_ablation_report(args.out)
        offline = args.action in ("plan", "preflight", "preview")
        _log(json.dumps({"plan_hash": result.get("plan_hash"),
                         "groups": (len(result.get("groups") or []) if offline else result.get("groups")),
                         "planned": result.get("planned", result.get("planned_images")),
                         "preflight_ok": result.get("ok") if args.action == "preflight" else None}, ensure_ascii=False, indent=2))
        return 0
    from utils.color_extract import export_generation_knowledge, compile_generation_palette
    if args.action == "extract":
        result = export_generation_knowledge(args.data, args.pages, args.out, args.chart or None, args.scoped_rules or None)
    elif args.action == "measure-chart":
        from utils.color_extract import export_book_image_chart
        result = export_book_image_chart(args.manifest, BASE, args.out)
    else:
        if not args.selection:
            raise ValueError("compile requires --selection")
        with open(args.knowledge, encoding="utf-8") as handle:
            knowledge = json.load(handle)
        with open(args.selection, encoding="utf-8") as handle:
            selection = json.load(handle)
        result = compile_generation_palette(knowledge, selection)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    if args.action == "compile":
        with open(os.path.splitext(args.out)[0] + ".prompt.txt", "w", encoding="utf-8") as handle:
            handle.write(result["prompt"])
    _log(json.dumps({"out": args.out, "count": result.get("count"),
                     "status": result.get("status", "extracted_not_generated")}, ensure_ascii=False))
    return 0


def cmd_catalog(args):
    from utils.color_extract import export_book_palette_catalog
    result = export_book_palette_catalog(args.data, args.out)
    _log(json.dumps(result, ensure_ascii=False, indent=2))
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


def cmd_stage2(args):
    """第二阶段：双模型八组套件（E0 校准 + A 主试验 + 匿名比较 + 符合度 + 汇总）。"""
    from utils.color_experiment import (calibrate_e0, compare_groups, evaluate_conformance,
                                        load_suite_spec, run_suite, suite_status, summarize_suite)
    action = args.action
    if action == "generate":
        spec = load_suite_spec(args.spec)
        rounds = _parse_ints(args.rounds) if args.rounds else None
        channels = [c.strip() for c in args.channels.split(",") if c.strip()] if args.channels else None
        result = run_suite(spec, args.out, rounds=rounds, channels=channels, dry_run=args.dry_run,
                           retry_failed=args.retry_failed, log=_log)
        if args.dry_run:
            return 0
        failed = [s for s in result.get("samples", []) if s.get("status") != "success"]
        aborted = [k for k, v in (result.get("node_blocks") or {}).items() if v.get("aborted")]
        if aborted:
            _log("[stage2] 模型区块被停止：%s（连续两次请求失败，不自动换模型）" % ", ".join(aborted))
        _log(json.dumps(suite_status(args.out), ensure_ascii=False, indent=2))
        return 1 if (failed or aborted) else 0
    if action == "tone-evaluate":
        from utils.color_experiment import evaluate_tone
        spec = load_suite_spec(args.spec)
        summary = evaluate_tone(args.out, spec, log=_log)
        _log(json.dumps(summary["per_group"], ensure_ascii=False, indent=2))
        return 0
    if action == "tone-summarize":
        from utils.color_experiment import summarize_tone
        spec = load_suite_spec(args.spec)
        result = summarize_tone(args.out, spec)
        _log("[stage2] tone gallery → %s" % result["gallery"])
        return 0
    if action == "resume-block":
        from utils.color_experiment import resume_block
        result = resume_block(args.out, args.model, note=args.note or "user authorised one resume")
        _log(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("resumed") else 2
    if action == "calibrate":
        spec = load_suite_spec(args.spec)
        verdict = calibrate_e0(args.out, spec, log=_log)
        return 0 if verdict.get("status") == "passed" else 1
    if action == "compare":
        spec = load_suite_spec(args.spec)
        compare_groups(args.out, spec, model_key=args.model, log=_log, recheck=args.recheck)
        return 0
    if action == "conformance":
        spec = load_suite_spec(args.spec)
        evaluate_conformance(args.out, spec, log=_log)
        return 0
    if action == "summarize":
        spec = load_suite_spec(args.spec)
        result = summarize_suite(args.out, spec)
        _log("[stage2] gallery → %s" % result["gallery"])
        _log("[stage2] plan cards → %s" % result["plan_cards"])
        return 0
    _log(json.dumps(suite_status(args.out), ensure_ascii=False, indent=2))
    return 0


def cmd_direction_pack(args):
    """App 二阶段色调/主次冻结包（无图首图）：离线校验 → 冻结运行参数 → 生成 → 评审 → 报告。

    薄壳：只做动作分发与退出码（`utils/color_direction_pack.py` 承担全部流程）。
    退出码：0 正常 / 2 预检或校验未过 / 3 已暂停（unknown_after_send，等用户决定）/ 1 有未完成项。
    """
    from utils import color_direction_pack as dp
    out = args.out
    if args.action == "build-pack":
        if not args.spec:
            _log("build-pack 需要 --spec <规格 JSON>")
            return 2
        built = dp.build_pack(args.spec, out_pack_path=args.out_pack or None, log=_log)
        _log(json.dumps(built, ensure_ascii=False, indent=2))
        return 0
    pack = dp.load_pack(args.pack)
    if args.action in ("freeze", "preflight", "run", "review"):
        verified = dp.verify_offline(out, pack_path=args.pack, scenes_path=args.scenes,
                                     review_path=args.review_prompt, log=_log)
        if verified["problems"]:
            _log(json.dumps(verified["problems"], ensure_ascii=False))
            return 2
    if args.action == "verify":
        result = dp.verify_offline(out, pack_path=args.pack, scenes_path=args.scenes,
                                   review_path=args.review_prompt, log=_log)
        _log(json.dumps({"status": result["status"], "problems": result["problems"],
                         "notes": result["notes"], "pack_hash_ok": result["pack_hash_ok"],
                         "file": result["written"]["path"]}, ensure_ascii=False, indent=2))
        return 0 if not result["problems"] else 2
    if args.action == "freeze":
        published = dp.publish_frozen_pack(out, pack_path=args.pack, log=_log)
        _log(json.dumps(published, ensure_ascii=False, indent=2))
        runtime = dp.freeze_runtime(out, pack, log=_log)
        if runtime["problems"]:
            return 2
        _log(json.dumps({"runtime_hash": runtime["runtime_hash"], "generation": {
            k: runtime["generation"].get(k) for k in ("host", "model", "resolution", "aspect_ratio",
                                                      "transport_timeout_seconds", "key_source")},
            "review": {k: runtime["review"].get(k) for k in ("host", "model", "max_completion_tokens",
                                                             "timeout_seconds", "key_source")}},
            ensure_ascii=False, indent=2))
        return 0
    if args.action == "preflight":
        published = dp.publish_frozen_pack(out, pack_path=args.pack, log=_log)
        runtime = dp.freeze_runtime(out, pack, log=_log)
        if runtime["problems"]:
            return 2
        mock = dp.run_slots(out, pack, runtime, dry_run=True, log=_log)
        review_mock = dp.review_groups(out, pack, runtime, dry_run=True, log=_log)
        _log(json.dumps({"status": "mock_preflight", "no_network": True,
                         "planned_slots": len(mock["results"]), "planned_reviews": len(review_mock["results"]),
                         "image_sends": mock["ledger"]["counters"]["image_sends"],
                         "review_sends": review_mock["ledger"]["counters"]["review_sends"],
                         "frozen_pack": published["path"],
                         "runtime_hash": runtime["runtime_hash"],
                         "budget": {"image": pack["image_http_budget"], "review": pack["review_http_budget"]},
                         "retry_env": {dp.IMAGE_RETRY_ENV: "0"}}, ensure_ascii=False, indent=2))
        return 0
    if args.action == "run":
        os.environ[dp.IMAGE_RETRY_ENV] = "0"
        _log("[direction] 本次运行已设置 %s=0（一次执行只发一次 HTTP，包装器与底层都不重发）"
             % dp.IMAGE_RETRY_ENV)
        runtime = dp.load_runtime(out)
        only = {name.strip() for name in (args.only or "").split(",") if name.strip()}
        result = dp.run_slots(out, pack, runtime, only=only or None, log=_log)
        counts = result["ledger"]["counters"]
        _log(json.dumps({"image_sends": counts["image_sends"], "paused": result["paused"],
                         "slots": result["results"]}, ensure_ascii=False, indent=2))
        if result["paused"]:
            return 3
        pending = [row["slot_id"] for row in result["results"]
                   if row.get("status") not in ("generated",)]
        return 0 if not pending else 1
    if args.action == "review":
        runtime = dp.load_runtime(out)
        only = {name.strip() for name in (args.only or "").split(",") if name.strip()}
        result = dp.review_groups(out, pack, runtime, only=only or None, log=_log)
        counts = result["ledger"]["counters"]
        _log(json.dumps({"review_sends": counts["review_sends"], "paused": result["paused"],
                         "reviews": result["results"]}, ensure_ascii=False, indent=2))
        if result["paused"]:
            return 3
        pending = [row["review_id"] for row in result["results"]
                   if row.get("status") not in ("reviewed", "skipped_no_images")]
        return 0 if not pending else 1
    if args.action == "authorize-resend":
        entry = dp.authorize_resend(out, args.slot, note=args.note, log=_log, pack=pack)
        _log(json.dumps({k: entry.get(k) for k in ("slot_id", "status", "attempts",
                                                  "resend_authorised")}, ensure_ascii=False, indent=2))
        return 0
    if args.action == "settle":
        entry = dp.settle_slot(out, args.slot, state=args.state, error_type=args.error_type,
                              error=args.error, evidence_file=args.evidence_file,
                              note=args.note or "用户授权结算", log=_log, pack=pack)
        _log(json.dumps({k: entry.get(k) for k in ("slot_id", "status", "attempts", "error_type",
                                                  "error", "settled")}, ensure_ascii=False, indent=2))
        return 0
    if args.action == "evidence":
        result = dp.collect_evidence(out, pack, log=_log)
        _log(json.dumps({"updated": len(result["updated"]),
                         "slots": result["updated"]}, ensure_ascii=False, indent=2))
        return 0
    if args.action == "report":
        runtime = dp.load_runtime(out)
        outcome = dp.render_report_safely(out, pack, runtime, log=_log)
        if not outcome["ok"]:
            return 1
        results = outcome["results"]
        _log(json.dumps({"counters": results["counters"], "paused": results["paused"],
                         "anomalies": len(results["anomalies"])}, ensure_ascii=False, indent=2))
        return 0
    _log(json.dumps(dp.status(out, pack), ensure_ascii=False, indent=2))
    return 0


def cmd_reassess(args):
    """配色方案符合度重评（2026-10-04）：逐张逐条对照实际方案条款，不做美学排名。"""
    from utils.color_experiment import (REASSESS_LABEL, evaluate_reassessment, load_reassess_items,
                                        reassess_plan, reassess_preflight, reassess_summary,
                                        write_reassessment_gallery)
    out = args.out
    if args.action == "preflight":
        items = load_reassess_items()
        plan = reassess_plan(items, chunk_size=args.chunk_size)
        result = reassess_preflight(out, items, plan, log=_log)
        for row in result["items"]:
            stats = row.get("local_stats") or {}
            _log("  %-22s %s 条款 %d 张 %sx%s 彩度像素 %.3f 冷占比 %.3f 暖占比 %.3f" % (
                row["item_id"], row["group_id"], row["clause_count"],
                (stats.get("size") or ["?", "?"])[0], (stats.get("size") or ["?", "?"])[1],
                stats.get("chromatic_pixel_fraction", 0.0), stats.get("chromatic_cool_fraction", 0.0),
                stats.get("chromatic_warm_fraction", 0.0)))
        return 1 if result["problems"] else 0
    if args.action == "recover":
        from utils.color_experiment import recover_reassessment
        only = {name.strip() for name in (args.only or "").split(",") if name.strip()}
        result = recover_reassessment(out, only or None, request_dir=args.request_dir or None, log=_log)
        _log(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["recovered"] or not result["still_unusable"] else 1
    if args.action == "evaluate":
        only = {name.strip() for name in (args.only or "").split(",") if name.strip()}
        result = evaluate_reassessment(out, text_cfg=None, log=_log, chunk_size=args.chunk_size,
                                      dry_run=args.dry_run, call_cap=args.call_cap, only=only or None)
        if args.dry_run:
            return 0
        summary = result["summary"]
        _log("[reassess] 已评价 %d / %d 张；业务调用 %d 次（上限 %d）" % (
            summary["reviewed_items"], summary["planned_items"],
            summary["calls"]["business_calls"], summary["calls"]["business_call_cap"]))
        path = write_reassessment_gallery(out, log=_log)
        _log("[reassess] 逐张评价页 → %s" % path)
        return 0 if not summary["missing_items"] and not summary["stopped"] else 1
    if args.action == "audit-ledger":
        from utils.color_experiment import reassess_call_ledger
        ledger = reassess_call_ledger(out, log=_log)
        _log(json.dumps({key: value for key, value in ledger.items() if key != "rows"},
                        ensure_ascii=False, indent=2))
        return 0
    if args.action == "summarize":
        summary = reassess_summary(out, log=_log)
        path = write_reassessment_gallery(out, log=_log)
        _log("[reassess] %s / %s" % (REASSESS_LABEL, path))
        return 0 if not summary["missing_items"] else 1
    summary_path = os.path.join(out, "reassessment-summary.json")
    if os.path.isfile(summary_path):
        _log(json.dumps(json.loads(open(summary_path, encoding="utf-8").read())["calls"],
                        ensure_ascii=False, indent=2))
    else:
        _log(json.dumps({"label": REASSESS_LABEL, "out": out, "note": "尚未执行评价"}, ensure_ascii=False))
    return 0


def cmd_palette(args):
    """固定色板执行验证（2026-10-04）：离线预检 → 生成 → 按组评审。不做美学排名。"""
    from utils.color_experiment import (PALETTE_LABEL, PALETTE_RETRY_ENV, load_palette_spec,
                                        palette_preflight, palette_review, palette_review_plan,
                                        run_palette, write_palette_plan)
    spec = load_palette_spec(args.spec or None)
    out = args.out
    scale = args.scale
    if args.action == "preflight":
        result = palette_preflight(out, spec_path=args.spec or None, scale=scale, log=_log)
        path = write_palette_plan(out, result, log=_log)
        _log(json.dumps({"status": result["status"], "problems": result["problems"],
                         "notes": result["notes"], "budget": result["budget"],
                         "channel": result["channel"], "plan": str(path)}, ensure_ascii=False, indent=2))
        return 1 if result["problems"] else 0
    if args.action == "preview":
        result = palette_preflight(out, spec_path=args.spec or None, scale=scale, log=_log)
        preview = result["request_preview"]
        _log("[palette] 请求预览：%d 组 · 长度 %s 字符" % (preview["count"], preview["length_chars"]))
        for row in preview["prompts"]:
            _log("=" * 70)
            _log("[%s] %s（%s，条款 %d 条，%d 字符）" % (
                row["group_id"], row["label"], row["variant"] or "无条款", row["clauses"], row["prompt_chars"]))
            _log(row["prompt"])
        return 1 if result["problems"] else 0
    if args.action == "generate":
        os.environ[PALETTE_RETRY_ENV] = "0"
        _log("[palette] 本次运行已设置 %s=0（关闭底层 HTTP 重试，使收费调用数 = 产物数）" % PALETTE_RETRY_ENV)
        result = run_palette(spec, out, scale=scale, retry_failed=args.retry_failed,
                             dry_run=args.dry_run, log=_log)
        if args.dry_run:
            return 0
        failed = [s for s in result["manifest"].get("samples", []) if s.get("status") != "success"]
        return 1 if failed else 0
    if args.action == "review-plan":
        plan = palette_review_plan(spec, out, scale, extra=args.extra)
        _log(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.action == "summary":
        from utils.color_experiment import palette_review_summary, write_palette_gallery
        summary = palette_review_summary(out, scale=scale, log=_log)
        path = write_palette_gallery(out, scale=scale, log=_log)
        _log("[palette] 逐张评价页 → %s" % path)
        missing = [g["group_id"] for g in summary["groups"] if g["reviewed"] < g["planned"]]
        return 1 if missing else 0
    if args.action == "review-recover":
        from utils.color_experiment import palette_review_recover
        only = {name.strip() for name in (args.only or "").split(",") if name.strip()}
        result = palette_review_recover(out, names=only or None, scale=scale, log=_log)
        _log(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["recovered"] or not result["still_unusable"] else 1
    if args.action == "review":
        import json as _json
        result = palette_review(out, scale=scale, extra=args.extra,
                               only={name.strip() for name in (args.only or "").split(",") if name.strip()} or None,
                               log=_log)
        _log(_json.dumps(result["summary"]["variant_pairs"], ensure_ascii=False, indent=2))
        missing = [group["group_id"] for group in result["summary"]["groups"] if group["reviewed"] < group["planned"]]
        from utils.color_experiment import write_palette_gallery
        write_palette_gallery(out, scale=scale, log=_log)
        return 1 if missing else 0
    if args.action == "status":
        plan = palette_review_plan(spec, out, scale, extra=args.extra)
        _log(json.dumps({"label": PALETTE_LABEL, "out": out, "scale": scale, "review": plan},
                        ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"未知动作 {args.action}")


def cmd_retention(args):
    """固定色板 → 画风重绘保留验证（A/B）：预检 → 冻结 → 执行 → 汇总（不评画质）。"""
def cmd_retention(args):
    """固定色板 → 画风重绘保留验证（A/B）：预检 → 冻结 → 执行 → 只补审计 → 汇总。

    `--spec` 选协议（v2 = 修复实验，默认；v1 = 原实验留档）；
    `--out` 覆盖输出目录，`--reuse-from` 指定可复用成功审计的旧实验目录。
    """
    from utils.color_experiment import (RETENTION_SPEC_PATH, RETENTION_V2_SPEC_PATH,
                                        load_retention_spec, retention_audit_only, retention_colour_review,
                                        retention_corrections, retention_freeze, retention_gate_stages,
                                        retention_ledger, retention_preflight, retention_recheck_preview,
                                        retention_run_dir, retention_summary, retention_verify_corrections,
                                        retention_recheck_preflight, retention_recheck_run,
                                        run_retention, write_retention_corrections_page,
                                        write_retention_gallery)
    spec_path = args.spec or RETENTION_V2_SPEC_PATH
    if args.spec == "v1":
        spec_path = RETENTION_SPEC_PATH
    spec = load_retention_spec(spec_path)
    if args.action in ("recheck-preflight", "recheck-run"):
        if not args.out:
            raise ValueError("recheck-preflight requires a separate --out directory")
        if args.action == "recheck-run":
            result = retention_recheck_run(args.out, spec_path=spec_path,
                                          authorized=args.authorize_recheck, log=_log)
            return 0 if all(row["status"] == "success" for row in result["reviews"]) else 1
        result = retention_recheck_preflight(args.out, spec_path=spec_path, log=_log)
        _log(json.dumps({"plan_hash": result["plan_hash"], "budget": result["budget"],
                         "calls_made": result["calls_made"]}, ensure_ascii=False, indent=2))
        return 0
    out = str(retention_run_dir(spec, args.out))
    _log("[retention] 协议 %s → 目录 %s" % (spec["id"], out))
    if args.action == "preflight":
        result = retention_preflight(out, spec_path=spec_path, only_b=args.only_b, log=_log)
        _log(json.dumps({"status": result["status"], "problems": result["problems"],
                         "notes": result["notes"], "budget": result["budget"],
                         "a_baseline_check": result.get("a_baseline_check", {}).get("status"),
                         "dispatch_check": result["dispatch_check"]}, ensure_ascii=False, indent=2))
        return 1 if result["problems"] else 0
    if args.action == "preview":
        result = retention_preflight(out, spec_path=spec_path, only_b=args.only_b, log=_log)
        for row in result["requests"]:
            _log("=" * 70)
            _log("[%s] 顶层 %d 字符 / 实际发送 %d 字符 / 一致=%s / 含合同=%s" % (
                row["request_id"], row["prompt_chars"], row["sent_prompt_chars"],
                row["prompt_matches_sent"], row["colour_contract_appended"]))
            _log(row["prompt"])
        for diff in result["request_diffs"]:
            _log("[diff] %s" % json.dumps(diff, ensure_ascii=False))
        return 1 if result["problems"] else 0
    if args.action == "freeze":
        result = retention_preflight(out, spec_path=spec_path, only_b=args.only_b, log=_log)
        if result["problems"]:
            _log("[retention] 预检有问题，不冻结、不执行")
            return 1
        frozen = retention_freeze(out, result, log=_log, reuse_from=args.reuse_from)
        _log(json.dumps(frozen, ensure_ascii=False, indent=2))
        return 0
    if args.action == "audit-only":
        os.environ["IMAGE_MAKER_IMAGE_MAX_RETRIES"] = "0"
        result = retention_audit_only(out, spec_path=spec_path, reuse_from=args.reuse_from, log=_log)
        _log(json.dumps({k: v for k, v in result.items()
                         if k in ("text_requests", "text_limit", "image_requests")},
                        ensure_ascii=False, indent=2))
        return 0
    if args.action == "colour-review":
        result = retention_colour_review(out, spec_path=spec_path, log=_log)
        _log(json.dumps({"reviews": [row.get("name") for row in result]}, ensure_ascii=False))
        return 0
    if args.action == "outcomes":
        rows = retention_gate_stages(out, spec=spec, log=_log)
        _log(json.dumps(rows, ensure_ascii=False, indent=2))
        retention_summary(out, spec=spec, log=_log)
        write_retention_gallery(out, spec=spec, log=_log)
        return 0
    if args.action == "correct":
        os.environ["IMAGE_MAKER_IMAGE_MAX_RETRIES"] = "0"
        corrections_dir = str(retention_run_dir(spec, args.out or spec.get("corrections_dir")))
        summary = retention_corrections(corrections_dir, spec_path=spec_path, reuse_from=args.reuse_from,
                                        evidence=not args.no_evidence, log=_log)
        page = write_retention_corrections_page(summary["corrections_dir"], spec_path=spec_path, log=_log)
        retention_recheck_preview(summary["corrections_dir"], spec_path=spec_path, log=_log)
        result = retention_verify_corrections(summary["corrections_dir"], spec_path=spec_path, log=_log)
        _log(json.dumps({"status_counts": summary["status_counts"], "page": str(page),
                         "verification": result["status"], "problems": result["problems"]},
                        ensure_ascii=False, indent=2))
        return 1 if result["problems"] else 0
    if args.action == "verify-corrections":
        result = retention_verify_corrections(args.out or None, spec_path=spec_path, log=_log)
        _log(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result["problems"] else 0
    if args.action == "reuse-audit":
        spec_obj = spec
        from utils.color_experiment import _retention_reuse_scan
        source = args.reuse_from or out
        scan = _retention_reuse_scan(retention_run_dir({"id": spec_obj["id"],
                                                        "_out_dir": source}, None), spec_obj)
        rows = [{"branch": key[0], "sample_id": key[1], "audit": key[2], "candidate_sha256": key[3],
                 "state": (value.get("reuse_eligibility") or {}).get("state"),
                 "missing": (value.get("reuse_eligibility") or {}).get("missing")}
                for key, value in sorted(scan.items())]
        _log(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if args.action == "run":
        os.environ["IMAGE_MAKER_IMAGE_MAX_RETRIES"] = "0"
        _log("[retention] 本次运行已设置 IMAGE_MAKER_IMAGE_MAX_RETRIES=0")
        result = run_retention(out, spec_path=spec_path, dry_run=args.dry_run,
                               reuse_from=args.reuse_from, only_b=args.only_b, log=_log)
        if args.dry_run or result.get("stopped"):
            _log(json.dumps({"stopped": result.get("stopped")}, ensure_ascii=False))
            return 1 if result.get("stopped") else 0
        failed = [row for row in (result["summary"].get("stages") or [])
                  if row.get("status") == "request_failed"]
        _log("[retention] A/B 对照页 → %s" % result.get("page"))
        return 1 if failed else 0
    if args.action == "summary":
        summary = retention_summary(out, spec=spec, log=_log)
        ledger = retention_ledger(out, spec=spec, log=_log)
        path = write_retention_gallery(out, spec=spec, log=_log)
        _log(json.dumps({"stages": len(summary.get("stages") or []),
                         "initial_success": summary["initial_success"],
                         "inherited": summary.get("inherited_candidates"),
                         "colour_verdicts": summary["colour_retention"]["verdicts"],
                         "available_after_gates": summary["gates"]["available_after_gates"],
                         "image_sent": ledger["image"]["sent_success"],
                         "text_fresh": ledger["text"]["fresh_gate_audits"],
                         "gallery": str(path)}, ensure_ascii=False, indent=2))
        return 0
    if args.action == "status":
        preflight_path = os.path.join(out, "retention-preflight.json")
        ledger = retention_ledger(out, spec=spec, log=_log)
        _log(json.dumps({"label": spec["id"], "out": out,
                         "preflight_present": os.path.isfile(preflight_path),
                         "ledger": {k: v for k, v in ledger.items() if k in ("image", "text")}},
                        ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"未知动作 {args.action}")


def cmd_cross(args):
    """跨主题固定配色执行验证（无画风参考图）：plan/conflicts/preflight/freeze/run/review/summary/snapshot。"""
    from utils.color_experiment import (CROSS_THEME_SPEC_PATH, cross_theme_colour_review,
                                        cross_theme_conflict_report, cross_theme_freeze, cross_theme_ledger,
                                        cross_theme_plan, cross_theme_preflight, cross_theme_run,
                                        cross_theme_snapshot, cross_theme_summary, load_cross_theme_spec,
                                        write_cross_theme_gallery)
    spec_path = args.spec or CROSS_THEME_SPEC_PATH
    spec = load_cross_theme_spec(spec_path)
    out = args.out or spec["_out_dir"]
    if args.action == "plan":
        plan = cross_theme_plan(out, spec_path=spec_path, log=_log)
        _log(json.dumps({"requests": len(plan["requests"]),
                         "contract_hash": plan["contract"]["contract_hash"],
                         "first_prompt_chars": plan["requests"][0]["prompt_chars"]},
                        ensure_ascii=False, indent=2))
        return 0
    if args.action == "conflicts":
        report = cross_theme_conflict_report(spec)
        _log(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if report["problems"] else 0
    if args.action == "preflight":
        result = cross_theme_preflight(out, spec_path=spec_path, log=_log)
        _log(json.dumps({"status": result["status"], "problems": result["problems"],
                         "resolved_generation": result["resolved_generation"],
                         "retries": result["retries_resolved"], "budget": result["budget"]},
                        ensure_ascii=False, indent=2))
        return 1 if result["problems"] else 0
    if args.action == "freeze":
        result = cross_theme_preflight(out, spec_path=spec_path, log=_log)
        if result["problems"]:
            _log("[cross] 预检有问题，不冻结、不执行")
            return 1
        frozen = cross_theme_freeze(out, result, log=_log)
        _log(json.dumps({"freeze_hash": frozen["freeze_hash"],
                         "generation": frozen["generation"]}, ensure_ascii=False, indent=2))
        return 0
    if args.action == "run":
        result = cross_theme_run(out, spec_path=spec_path, dry_run=args.dry_run, log=_log)
        if args.dry_run or result.get("stopped"):
            _log(json.dumps({"stopped": result.get("stopped")}, ensure_ascii=False))
            return 1 if result.get("stopped") else 0
        summary = result["summary"]
        failed = summary["generation_total"] - summary["generation_success"]
        _log(json.dumps({"success": summary["generation_success"], "failed": failed,
                         "groups": len(summary["groups"])}, ensure_ascii=False, indent=2))
        return 1 if failed else 0
    if args.action == "review":
        records = cross_theme_colour_review(out, spec=spec, spec_path=spec_path, log=_log)
        _log(json.dumps({"groups": len(records),
                         "statuses": {row.get("group_id"): row.get("status") for row in records}},
                        ensure_ascii=False, indent=2))
        return 0
    if args.action == "summary":
        summary = cross_theme_summary(out, spec=spec, log=_log)
        ledger = cross_theme_ledger(out, spec=spec, log=_log)
        page = write_cross_theme_gallery(out, spec=spec, log=_log)
        _log(json.dumps({"success": summary["generation_success"],
                         "total": summary["generation_total"],
                         "image_calls": ledger["image"]["success"],
                         "text_calls": ledger["text"]["reserved"], "page": str(page)},
                        ensure_ascii=False, indent=2))
        return 0
    if args.action == "snapshot":
        snap = cross_theme_snapshot(out, spec=spec, log=_log)
        _log(json.dumps({"done": snap["done"], "total": snap["total"]}, ensure_ascii=False))
        return 0
    raise ValueError(f"未知动作 {args.action}")


def main():
    ap = argparse.ArgumentParser(description="扫描版配色书 → 项目配色知识库")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("catalog", help="离线导出全部书籍案例配色及色格预览，不生图")
    p.add_argument("--data", default=os.path.join(BASE, "data", "color-knowledge"))
    p.add_argument("--out", default=os.path.join(BASE, "prompts", "color-knowledge", "book-palette-catalog-v1.json"))
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("pilot", help="隔离的 Gemini 配色对照实验 / 自动选择 / 色彩评价")
    p.add_argument("action", choices=["generate", "select", "summarize", "evaluate"])
    p.add_argument("--spec", default=os.path.join(BASE, "prompts", "color-knowledge", "pilot-spring-fishing.json"))
    p.add_argument("--out", required=True, help="data/test-result/ 下的独立实验目录")
    p.add_argument("--model", default=None, help="本次 Gemini 模型覆盖，不写配置")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--retry-failed", action="store_true", help="显式重试失败/中断记录，成功产物复用")
    p.set_defaults(func=cmd_pilot)

    p = sub.add_parser("stage2", help="第二阶段：双模型八组配色套件（E0 校准 / A 主试验 / 匿名比较 / 汇总）")
    p.add_argument("action", choices=["generate", "calibrate", "compare", "conformance", "summarize", "status",
                                      "resume-block", "tone-evaluate", "tone-summarize"])
    p.add_argument("--spec", default=os.path.join(BASE, "prompts", "color-knowledge", "stage2-spring-fishing.json"))
    p.add_argument("--out", required=True, help="data/test-result/ 下的独立实验目录")
    p.add_argument("--rounds", default="", help="如 1 或 1,2,3（默认全部三轮）")
    p.add_argument("--channels", default="", help="如 gemini-flash,gpt-image-2（默认两通道）")
    p.add_argument("--model", default=None, help="只跑/只恢复某个通道（gemini-flash / gpt-image-2）")
    p.add_argument("--note", default="", help="resume-block 的授权备注（记进 suite.json 的 block_history）")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--recheck", action="store_true", help="只重跑交换复核（严格镜像），写成 -swap2，旧记录保留")
    p.add_argument("--retry-failed", action="store_true", help="显式重试失败记录，成功产物复用")
    p.set_defaults(func=cmd_stage2)

    p = sub.add_parser("reassess", help="配色方案符合度重评：逐张逐条对照实际方案条款（不做美学排名）")
    p.add_argument("action", choices=["preflight", "evaluate", "recover", "summarize", "audit-ledger", "status"])
    p.add_argument("--out", required=True, help="data/test-result/ 下的独立重评目录")
    p.add_argument("--chunk-size", type=int, default=None, help="每块图片数（默认 6；同方案三张不拆块）")
    p.add_argument("--call-cap", type=int, default=None,
                   help="本次运行允许新增的业务调用数（默认 10；授权总数 12，跨目录历史调用另记）")
    p.add_argument("--only", default="", help="只跑指定块/组，如 reassess-12,reassess-13")
    p.add_argument("--request-dir", default="",
                   help="recover 时从别处读 request.json（测试夹具/只读副本）；原始响应仍从 --out 读")
    p.add_argument("--dry-run", action="store_true", help="只做离线预检与协议快照，不发任何调用")
    p.set_defaults(func=cmd_reassess)

    p = sub.add_parser("palette", help="固定色板执行验证：离线预检 → 生成 → 按组评审（不做美学排名）")
    p.add_argument("action", choices=["preflight", "preview", "generate", "review-plan", "review",
                                      "review-recover", "summary", "status"])
    p.add_argument("--out", required=True, help="data/test-result/ 下的独立实验目录")
    p.add_argument("--scale", default="full", choices=["full", "reduced"],
                   help="full=21 张（7 组×3）/ reduced=9 张（3 组×3）")
    p.add_argument("--spec", default="", help="覆盖默认 spec 路径")
    p.add_argument("--extra", type=int, default=0,
                   help="已使用的额外评审次数（只用于事先约定的失败恢复/证据不足）")
    p.add_argument("--only", default="", help="review 时只评审指定组，如 P1-simple,B0")
    p.add_argument("--retry-failed", action="store_true", help="生成时显式重试失败样本（成功产物仍复用）")
    p.add_argument("--dry-run", action="store_true", help="只打印计划，不发出任何生成请求")
    p.set_defaults(func=cmd_palette)

    p = sub.add_parser("direction-pack",
                       help="App 二阶段色调/主次冻结包（无图首图）：verify/freeze/preflight/run/review/report/status")
    p.add_argument("action", choices=["verify", "freeze", "preflight", "run", "review", "report",
                                      "status", "settle", "authorize-resend", "evidence", "build-pack"])
    p.add_argument("--out", required=True, help="data/test-result/ 下的独立实验目录")
    p.add_argument("--spec", default="", help="build-pack 的实验规格 JSON（离线编译冻结包）")
    p.add_argument("--out-pack", default="", help="build-pack 的冻结包输出路径（默认按 id 派生）")
    p.add_argument("--slot", default="", help="settle 的槽位 id（人工授权结算停在 sending 的槽位）")
    p.add_argument("--state", default="unknown_after_send",
                   choices=["unknown_after_send", "failed"], help="settle 的目标状态")
    p.add_argument("--error-type", default="", help="settle 记录的错误类型")
    p.add_argument("--error", default="", help="settle 记录的错误摘要")
    p.add_argument("--evidence-file", default="", help="settle 附带的证据文件（会复制进 run-notes/send-errors/）")
    p.add_argument("--note", default="", help="settle 的授权备注")
    p.add_argument("--pack", default=os.path.join(BASE, "prompts", "color-knowledge",
                                                  "theme-color-direction-validation-v1-pack.json"),
                   help="冻结包路径（默认 App 编译器产出的 v1 冻结包）")
    p.add_argument("--scenes", default=os.path.join(BASE, "prompts", "color-knowledge",
                                                    "theme-color-direction-validation-v1-scenes.json"))
    p.add_argument("--review-prompt", default=os.path.join(BASE, "prompts", "color-knowledge",
                                                           "theme-color-direction-validation-v1-review.md"))
    p.add_argument("--only", default="", help="run/review 时只处理指定槽位或评审组，逗号分隔")
    p.set_defaults(func=cmd_direction_pack)

    p = sub.add_parser("retention", help="固定色板 → 画风重绘保留验证（A/B，跑现有门禁）")
    p.add_argument("action", choices=["preflight", "preview", "freeze", "run", "audit-only",
                                      "colour-review", "outcomes", "correct", "verify-corrections",
                                      "reuse-audit", "summary", "status", "recheck-preflight", "recheck-run"])
    p.add_argument("--spec", default="", help="协议：v2（默认，修复版）或 v1（原实验留档）或 spec 路径")
    p.add_argument("--out", default="", help="覆盖输出目录（默认用 spec 里的 out_dir）")
    p.add_argument("--reuse-from", default="", help="可复用成功审计的旧实验目录（按候选 hash 匹配）")
    p.add_argument("--only-b", action="store_true", help="只生成 B 支（A 沿用已核对的旧同请求产物）")
    p.add_argument("--dry-run", action="store_true", help="只做预检与冻结，不发任何请求")
    p.add_argument("--no-evidence", action="store_true", help="修正交付里不重建输入证据（更快的只读核对）")
    p.add_argument("--authorize-recheck", action="store_true", help="授权冻结的三次只读文本复核；不生图、不重试")
    p.set_defaults(func=cmd_retention)

    p = sub.add_parser("cross", help="跨主题固定配色执行验证（无画风参考图，只判配色与落点）")
    p.add_argument("action", choices=["plan", "conflicts", "preflight", "freeze", "run",
                                      "review", "summary", "snapshot"])
    p.add_argument("--spec", default="", help="协议路径（默认 prompts/color-knowledge/cross-theme-palette-v1.json）")
    p.add_argument("--out", default="", help="输出目录（默认用 spec 里的 out_dir）")
    p.add_argument("--dry-run", action="store_true", help="只做预检与冻结，不发任何请求")
    p.set_defaults(func=cmd_cross)

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

    p = sub.add_parser("generation-knowledge", help="提取/编译书籍配色；preflight/plan 离线冻结，run 生图，review 视觉评审")
    p.add_argument("--retry-failed", action="store_true", help="已授权失败样本原样重试一次，保留首次失败")
    p.add_argument("action", choices=["extract", "compile", "measure-chart", "plan", "preflight", "preview", "run", "review", "report"])
    p.add_argument("--compare-preview", default="", help="preflight 可选的旧 preview 目录：逐条记录请求差异（只读本地快照）")
    p.add_argument("--spec", default=os.path.join(BASE, "prompts", "color-knowledge", "book-palette-ablation-v1.json"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--manifest", default=os.path.join(BASE, "prompts", "color-knowledge", "book-image-chart-rois-v1.json"))
    p.add_argument("--chart", default="")
    p.add_argument("--scoped-rules", default="", help="显式导入经过来源核对的条件规则包；不覆盖旧知识版本")
    p.add_argument("--data", default=os.path.join(BASE, "data", "color-knowledge"))
    p.add_argument("--pages", default=os.path.join(BASE, "docs", "261003-color-improve", "book-html", "pages"))
    p.add_argument("--knowledge", default=os.path.join(BASE, "prompts", "color-knowledge", "book-generation-knowledge-v1.json"))
    p.add_argument("--selection", default="")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_generation_knowledge)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
