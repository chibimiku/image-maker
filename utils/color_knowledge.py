# -*- coding: utf-8 -*-
"""配色书知识提取运行时（扫描版 PDF → 结构化色彩知识 → AI 生图可用规则）。

为什么单独一个模块：本项目的分析 Tab 只处理**单张插画**，而这里要处理的是
**一本扫描书**（150 页 JPEG，无文字层）。两者共用同一条文本/视觉 API 通道
（`utils/analysis_gpt_prompt.load_text_api_config` + `call_text_model`），但
工序、缓存粒度、产物 schema 完全不同，所以独立成模块，不塞进 single_analyzer。

三段工序（GUI 不接入，全部走 `tools/color_knowledge.py`）：

1. `extract_pdf_pages` —— 从 PDF 里按页抽出原始 JPEG 页图。
   本仓库不装 PyMuPDF（`PROJECT_REQUIREMENTS.md` 第 2 节：唯一 Python 是系统
   Python，禁止 pip 装依赖），所以这里直接解析 PDF 对象：每个 `/Type /Page`
   的 `/XObject` 指向一个 `/DCTDecode`（JPEG）图像流，把那段字节原样切出来即可，
   不重新编码、不丢画质。扫描件几乎都是这个形态。
2. `analyze_page` —— 一页图 → 一段严格 JSON（走视觉模型）。
   **绝不向模型要 hex 色值**：扫描印刷品的颜色是 CMYK 网点，模型只能靠印象瞎编。
   颜色名（中文传统色名）靠视觉读，真实色值靠 `measure_swatches` 用 OpenCV 量。
3. `measure_swatches` —— 用 OpenCV 在页面上找圆形色板并量出平均色。
   5 个「主色调」色板是在白底上排成一行的圆，形状极其规整，检测比让模型猜靠谱得多。

最后 `build_knowledge_base` 把逐页 JSON + 实测色值合成知识库，交给
`tools/color_knowledge.py merge` 写盘。
"""
import base64
import json
import os
import re
import time

from utils.prompt_loader import render_prompt_file

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KB_DIR = os.path.join(BASE_DIR, "data", "color-knowledge")
SYSTEM_PROMPT_FILE = "color-knowledge/page-extract-system.md"
USER_PROMPT_FILE = "color-knowledge/page-extract-user.md"
DEFAULT_MODEL = "gpt-6-sol"
FALLBACK_MODEL = "gpt-5.6-luna"
DEFAULT_TIMEOUT = 900
# 页图送模型前的最长边：1824px 的原始页图已经够读正文小字，再大只是烧 token
PAGE_MAX_EDGE = 1824
PAGE_JPEG_QUALITY = 92

PAGE_TYPES = (
    "front_matter",     # 版权页 / 目录 / 扉页 / 阅读方法 / 图表的解说
    "theory",           # 「如何看色彩」「配色的基础」「NCD 色相环」
    "note",             # 「笔记」页：色彩与心理/五感/季节/意象
    "chapter_cover",    # 画师章节封面
    "illustration",     # 整页作品
    "case_analysis",    # 作品分析首页：主色调 + 色相/色调平衡
    "case_highlight",   # 配色亮点 A/B/C/D
    "case_process",     # 绘画过程中的配色要点
    "other",
)

# 画师章节在 PDF 里的起始页（= 目录里章节首页的印刷页码 + 5）。
# 有这张表模型才知道自己读的是谁的章节——不给的话它会瞎猜（实测第 22 页被标成"理论"）。
DEFAULT_CHAPTER_MAP = {
    13: "理论（如何看色彩 / 配色的基础）",
    20: "给米（Gemi）",
    38: "波特歌（potg）",
    54: "樱田千寻（Sakurada Chihiro）",
    66: "亚纪（Aki）",
    80: "约尔·福杰（Fajyobore）",
    94: "巴尼石门（Banishment）",
    106: "寺田寺（Terada Tera）",
    118: "木野花日兰子（Konohana Hiranko）",
    130: "三土（REDUM）",
    138: "拉斯库（RASUKU）",
}


# ---------------------------------------------------------------- 页图抽取

def _iter_pdf_objects(raw: bytes):
    """把 `N 0 obj ... endobj` 切出来（本 PDF 是线性 xref，未压缩对象）。"""
    for m in re.finditer(rb"(\d+)\s+(\d+)\s+obj", raw):
        num = int(m.group(1))
        start = m.end()
        end = raw.find(b"endobj", start)
        yield num, raw[start:end if end > 0 else len(raw)]


def extract_pdf_pages(pdf_path: str, out_dir: str, page_range=None, log_callback=None) -> dict:
    """把 PDF 的每一页抽成 `page-NNN.jpg`，并写 `manifest.json`。

    只处理「一页一张 DCTDecode 图」的扫描件；遇到没有内嵌 JPEG 的页面会跳过并在
    manifest 里记 `skipped`（正文书是矢量排版，那种情况需要另一条路径）。
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    raw = open(pdf_path, "rb").read()
    objs = dict(_iter_pdf_objects(raw))
    pages = []
    for num, body in objs.items():
        if not re.search(rb"/Type\s*/Page[^s]", body):
            continue
        xobj = re.search(rb"/XObject\s*<<\s*/Im0\s+(\d+)\s+0\s+R", body)
        if xobj:
            pages.append((num, int(xobj.group(1))))
    pages.sort()
    lo, hi = (page_range or (1, len(pages)))
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for idx, (page_obj, img_obj) in enumerate(pages, 1):
        entry = {"pdf_page": idx, "page_obj": page_obj, "image_obj": img_obj, "file": ""}
        if not (lo <= idx <= hi):
            entry["skipped"] = "out_of_range"
            manifest.append(entry)
            continue
        body = objs.get(img_obj, b"")
        if not re.search(rb"/DCTDecode", body):
            entry["skipped"] = "not_dctdecode"
            manifest.append(entry)
            continue
        s = body.find(b"stream")
        blob = body[s + 6:].lstrip(b"\r\n")
        blob = blob[:blob.rfind(b"endstream")].rstrip(b"\r\n")
        if blob[:2] != b"\xff\xd8":
            entry["skipped"] = "bad_jpeg_header"
            manifest.append(entry)
            continue
        name = "page-%03d.jpg" % idx
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(blob)
        entry["file"] = name
        entry["bytes"] = len(blob)
        manifest.append(entry)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"pdf": pdf_path, "pages": manifest}, f, ensure_ascii=False, indent=1)
    done = sum(1 for m in manifest if m.get("file"))
    skipped = sum(1 for m in manifest if m.get("skipped"))
    log(f"[color-knowledge] 抽页完成：{done} 页写出，{skipped} 页跳过 → {out_dir}")
    return {"pages": manifest, "written": done, "skipped": skipped, "out_dir": out_dir}


# ---------------------------------------------------------------- 视觉分析

def _data_url(path: str, max_edge: int = PAGE_MAX_EDGE, quality: int = PAGE_JPEG_QUALITY) -> str:
    """按最长边压到 max_edge 再编码（复用 opencv，和 sol_review.py 同款）。"""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读不到图片: {path}")
    h, w = img.shape[:2]
    scale = float(max_edge) / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def parse_json_reply(text: str) -> dict:
    """从模型回复里抠出 JSON（容忍 ```json 围栏与前后废话）。"""
    t = str(text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise RuntimeError(f"回复里没有 JSON：{t[:200]}")
        return json.loads(m.group(0))


def analyze_page(image_path: str, pdf_page: int, printed_page=None, chapter_hint: str = "",
                 text_cfg: dict = None, model: str = DEFAULT_MODEL, timeout: float = DEFAULT_TIMEOUT,
                 log_callback=None, attempts: int = 2) -> dict:
    """一页图 → 页级知识 JSON（视觉模型）。失败抛异常，由调用方决定是否写 cache。"""
    from utils.analysis_gpt_prompt import call_text_model, load_text_api_config, normalize_chat_base

    cfg = text_cfg or load_text_api_config()
    if not (cfg.get("base_url") and cfg.get("api_key")):
        raise RuntimeError("缺少文本 API 配置（conf/config.json 的 base_url + IMAGE_MAKER_TEXT_API_KEY）")
    system_prompt = render_prompt_file(SYSTEM_PROMPT_FILE, {})
    user_prompt = render_prompt_file(USER_PROMPT_FILE, {
        "pdf_page": pdf_page,
        "printed_page": printed_page if printed_page is not None else "",
        "chapter_hint": chapter_hint or "（未知，自行判断）",
        "page_types": " | ".join(PAGE_TYPES),
    })
    last = ""
    for attempt in range(1, max(1, attempts) + 1):
        use_model = model if attempt == 1 else (FALLBACK_MODEL or model)
        try:
            # 走项目既有的文本通道（`call_text_model`），附图用 image_paths——
            # 不自己拼 POST，免得密钥解析/归一化/超时口径和别的链路漂移。
            raw = call_text_model(normalize_chat_base(cfg["base_url"]), cfg["api_key"], use_model,
                                  system_prompt, user_prompt, timeout=timeout, max_tokens=16000,
                                  image_paths=[image_path])
            payload = parse_json_reply(raw)
            payload.setdefault("pdf_page", pdf_page)
            # 页码以「PDF 序号 − offset」为准（本书已用 TOC + 页脚交叉验证）；
            # 模型读出来的页码只留作自检，读错了不能让整本书的索引跟着错。
            payload["printed_page_read"] = payload.get("printed_page")
            payload["printed_page"] = printed_page
            read_pg, want_pg = payload.get("printed_page_read"), printed_page
            if isinstance(read_pg, int) and isinstance(want_pg, int) and read_pg != want_pg:
                payload.setdefault("warnings", []).append(
                    f"模型读出的页码 {read_pg} 与推算页码 {want_pg} 不一致（以推算为准）")
            payload["_model"] = use_model
            return payload
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            if log_callback:
                log_callback(f"[color-knowledge] 第 {pdf_page} 页分析失败（{use_model}，第 {attempt} 次）：{last}")
            time.sleep(2)
    raise RuntimeError(f"第 {pdf_page} 页分析失败：{last}")


REQUIRED_PAGE_KEYS = ("pdf_page", "page_type", "printed_page")


def is_complete_page_json(path: str) -> bool:
    """判断一份页级 JSON 是不是「完整产物」。

    只按「文件存在」跳过是会出事的：模型超时/返回非 JSON 时可能留下半截文件，
    下一轮就被当成已完成跳过，最后知识库里默默缺页。所以必须校验关键字段。
    """
    try:
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(p, dict):
        return False
    if any(k not in p for k in REQUIRED_PAGE_KEYS):
        return False
    return str(p.get("page_type") or "") in PAGE_TYPES


def analyze_pages(pages_dir: str, out_dir: str, offset: int = 5, only=None, force: bool = False,
                  limit: int = 0, model: str = DEFAULT_MODEL, chapter_map=None,
                  log_callback=None) -> dict:
    """批量跑 `analyze_page`，逐页落盘，天然可续跑（已存在的页跳过）。

    `offset` = 印刷页码与 PDF 序号之差（本书 = 5，即 printed = pdf_page − 5）。
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    manifest_path = os.path.join(pages_dir, "manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8"))["pages"] if os.path.isfile(manifest_path) else []
    files = [m for m in manifest if m.get("file")] or [
        {"pdf_page": int(re.search(r"(\d+)", os.path.basename(p)).group(1)), "file": os.path.basename(p)}
        for p in sorted(_glob_pages(pages_dir))]
    if only:
        want = set(int(x) for x in only)
        files = [m for m in files if m["pdf_page"] in want]
    if limit:
        files = files[:limit]
    os.makedirs(out_dir, exist_ok=True)
    done = skipped = failed = 0
    for m in files:
        pdf_page = m["pdf_page"]
        dest = os.path.join(out_dir, "page-%03d.json" % pdf_page)
        if os.path.isfile(dest) and not force:
            if is_complete_page_json(dest):
                skipped += 1
                continue
            log(f"[color-knowledge] 第 {pdf_page} 页已有文件但字段不全，重跑")
            os.remove(dest)
        printed = pdf_page - offset
        chapter = ""
        for lo, name in sorted((chapter_map if chapter_map is not None else DEFAULT_CHAPTER_MAP).items(),
                               reverse=True):
            if pdf_page >= int(lo):
                chapter = name
                break
        try:
            payload = analyze_page(os.path.join(pages_dir, m["file"]), pdf_page, printed,
                                   chapter_hint=chapter, model=model, log_callback=log)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log(f"[color-knowledge] 放弃第 {pdf_page} 页：{exc}")
            continue
        # 章节以查表为准：模型会猜（实测第 22 页被判成「理论」，实际是「给米」）。
        read_chapter = str(payload.get("chapter") or "").strip()
        if chapter:
            if read_chapter and read_chapter != chapter:
                payload.setdefault("warnings", []).append(
                    f"模型判定的章节「{read_chapter}」与查表「{chapter}」不一致（以查表为准）")
            payload["chapter"] = chapter
        if not payload.get("work_title") and payload.get("title"):
            payload["work_title"] = payload["title"]
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        done += 1
        log(f"[color-knowledge] {pdf_page}/{pdf_page} 页完成：{payload.get('page_type')} "
            f"{payload.get('work_title') or payload.get('title') or ''}")
    log(f"[color-knowledge] 分析完成：新增 {done}，跳过 {skipped}，失败 {failed}")
    return {"done": done, "skipped": skipped, "failed": failed, "out_dir": out_dir}


def _glob_pages(pages_dir):
    import glob as _g
    return _g.glob(os.path.join(pages_dir, "page-*.jpg"))


# ---------------------------------------------------------------- 色板实测

def measure_swatches(image_path: str, min_radius: int = 12, max_radius: int = 80,
                     min_saturation: int = 40, min_value: int = 30, max_value: int = 250,
                     min_circularity: float = 0.72) -> list:
    """在页面里找圆形色板并量出平均色（HSV 阈值 → 轮廓 → 圆度过滤）。

    返回按 (y, x) 排序的列表：`{"x","y","r","hex","rgb","hsv","area","circularity"}`。
    只做几何筛选，不判断"这是不是主色调色板"——那一步交给视觉 JSON 的名字对齐。
    """
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读不到图片: {image_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat >= min_saturation) & (val >= min_value) & (val <= max_value)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        per = cv2.arcLength(c, True)
        if per <= 0:
            continue
        circularity = 4.0 * 3.141592653589793 * area / (per * per)
        if circularity < min_circularity:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if not (min_radius <= r <= max_radius):
            continue
        if abs(area / (3.141592653589793 * r * r) - 1.0) > 0.45:
            continue
        yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
        inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r * 0.6) ** 2
        if inner.sum() < 20:
            continue
        bgr = img[inner].reshape(-1, 3).mean(axis=0)
        hsv_m = hsv[inner].reshape(-1, 3).mean(axis=0)
        r8, g8, b8 = [int(round(v)) for v in (bgr[2], bgr[1], bgr[0])]
        out.append({
            "x": int(round(cx)), "y": int(round(cy)), "r": int(round(r)),
            "hex": "#%02x%02x%02x" % (r8, g8, b8),
            "rgb": [r8, g8, b8],
            "hsv": [int(round(hsv_m[0] * 360 / 180)), int(round(hsv_m[1] * 100 / 255)), int(round(hsv_m[2] * 100 / 255))],
            "area": int(area), "circularity": round(circularity, 3),
        })
    out.sort(key=lambda d: (d["y"], d["x"]))
    return out


def group_swatch_rows(swatches: list, tol: int = 40) -> list:
    """把实测色板按 y 聚成「排」（同一排的色板 y 只差几十像素）。"""
    rows = []
    for s in sorted(swatches, key=lambda d: d["y"]):
        for row in rows:
            if abs(row[0]["y"] - s["y"]) <= tol:
                row.append(s)
                break
        else:
            rows.append([s])
    for row in rows:
        row.sort(key=lambda d: d["x"])
    return rows


def _row_unit(row: list) -> float:
    """把一排圆点的「基准间距」估出来，允许中间缺圆（缺一个时 gap 就是 2 倍）。

    返回 0 表示这排不像等距色板。判据：以最小间距为基准，其余间距都要接近它的整数倍。
    """
    xs = [s["x"] for s in row]
    if len(xs) < 2:
        return 0.0
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    span = xs[-1] - xs[0]
    if span <= 0 or min(gaps) <= 0:
        return 0.0
    unit = span / max(1, round(span / min(gaps)))
    for g in gaps:
        if abs(g - round(g / unit) * unit) > 0.22 * unit:
            return 0.0
    return unit


def pick_palette_row(swatches: list, want: int = 5) -> list:
    """从实测色板里挑出「主色调」那一排（齐整的情况）。

    判据（纯几何，不问模型）：凑够 `want` 个圆、半径彼此相差不超过 15%、
    间距是等距栅格（`_row_unit`）；多排合格时取**最靠上**的一排（案例页主色调在上部）。
    """
    for row in group_swatch_rows(swatches):
        if len(row) != want:
            continue
        radii = [s["r"] for s in row]
        if max(radii) - min(radii) > max(2, 0.15 * max(radii)):
            continue
        if _row_unit(row):
            return row
    return []


def _best_partial_row(swatches: list, want: int, tol: int = 40) -> list:
    """挑出「最像主色调那一排」的半成品行：凑不够 `want` 个圆，但间距是等距栅格。

    排序偏好：圆越多越好 → 越靠上越好（案例页的主色调排在版面靠上位置）。
    不限制 y：有些作品的插画占满上半页，主色调会排到页面中部以下。
    """
    best = []
    for row in group_swatch_rows(swatches, tol):
        if len(row) < 3 or len(row) > want:
            continue
        if not _row_unit(row):
            continue
        if len(row) > len(best) or (len(row) == len(best) and best and row[0]["y"] < best[0]["y"]):
            best = row
    return best


def _paper_color(img, y: int, band: int = 120):
    """估这一行附近的纸白（取行带里最亮的 5% 像素的中位色）。

    扫描页有网点噪声，绝对阈值不可用，只能和"这一页的纸"比。
    """
    import numpy as np

    y0, y1 = max(0, y - band), min(img.shape[0], y + band)
    strip = img[y0:y1].reshape(-1, 3).astype(np.int32)
    order = strip.sum(axis=1).argsort()[::-1]
    top = strip[order[:max(50, len(order) // 20)]]
    return np.median(top, axis=0)


def _sample_disc(img, blurred, cx: int, cy: int, radius: int, paper):
    """取圆盘内像素均值 + "去噪后"的标准差。

    标准差必须在高斯模糊过的图上算：扫描页的网点噪声让原始 std 动辄 50+，
    那反映的是印刷颗粒而不是内容。纸上真正的结构（文字、边缘、花纹）模糊后依然在。
    """
    import numpy as np

    r = max(2, int(radius))
    if cx - r < 0 or cy - r < 0 or cy + r >= img.shape[0] or cx + r >= img.shape[1]:
        return None
    yy, xx = np.ogrid[-r:r, -r:r]
    m = ((xx ** 2 + yy ** 2) <= (r * 0.85) ** 2).reshape(-1)
    sel = img[cy - r:cy + r, cx - r:cx + r].reshape(-1, 3)[m]
    sel_b = blurred[cy - r:cy + r, cx - r:cx + r].reshape(-1, 3)[m]
    if sel.size < 12:
        return None
    mean = sel.mean(axis=0)
    std = float(sel_b.std())
    dist = float(np.abs(mean - paper).max())
    paper_frac = float((np.abs(sel - paper).max(axis=1) < 25).mean())
    return mean, std, dist, paper_frac


def _complete_row(image_path: str, row: list, want: int = 5) -> list:
    """半成品行 → 补齐：按估出的等距栅格外推位置，再采样验证"那里是块平色"。

    书里的主色调是等距圆点，认到 3~4 个，剩下的位置就是确定的。
    浅色色板（粉红、米黄）在 HSV 阈值下容易被漏掉，靠几何补回来比反复调阈值稳。
    要求至少认到 3 个真圆——只认到 2 个的话那一排更可能是插画上的巧合，不该外推。
    """
    import numpy as np

    if len(row) < 3:
        return []
    import cv2

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []
    unit = _row_unit(row)
    r = int(np.median([s["r"] for s in row]))
    if not unit or unit < 2.2 * r:
        return []
    y = int(np.median([s["y"] for s in row]))
    paper = _paper_color(img, y)
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    hit_x = [s["x"] for s in row]
    by_x = {s["x"]: s for s in row}

    def slot_ok(x):
        """某个格位能不能当色板：和纸白差得远、圆盘里几乎没有纸白。"""
        got = _sample_disc(img, blurred, x, y, int(r * 0.55), paper)
        if got is None:
            return None
        mean, std, dist, paper_frac = got
        # 判据只认「和纸白差得远」+「圆盘里几乎没有纸白」：
        # 印刷网点让 std 天然很大（浅色色板能到 60+），所以 std 只当兜底护栏，
        # 真正的判据是 paper_frac——色板里不该有纸白，文字旁边一定有一半是纸白。
        if dist < 22 or paper_frac > 0.25 or std > 95:
            return None
        return mean

    # 已知圆可能整排缺一个，所以缺的那个既可能在左边也可能在右边：
    # 把栅格往两边各延一格，再滑窗取「命中已知圆最多、其次最靠左」的那 `want` 格。
    best_window = None
    base = min(hit_x) - unit
    for start in range(0, 4):
        window = [int(round(base + start * unit + k * unit)) for k in range(want)]
        if window[0] < r or window[-1] + r >= img.shape[1]:
            continue
        hits = sum(1 for x in window if any(abs(x - hx) <= 0.3 * unit for hx in hit_x))
        score = (hits, -window[0])
        if best_window is None or score > best_window[0]:
            best_window = (score, window)
    if best_window is None:
        return []
    filled = []
    for x in best_window[1]:
        hit = None
        for sx, s in by_x.items():
            if abs(sx - x) <= 0.3 * unit:
                hit = s
        if hit is not None:
            filled.append(hit)
            continue
        mean = slot_ok(x)
        if mean is None:
            return []
        b, g, ch = [int(round(v)) for v in mean]
        hsv = cv2.cvtColor(np.uint8([[[b, g, ch]]]), cv2.COLOR_BGR2HSV)[0][0]
        filled.append({"x": x, "y": y, "r": r, "hex": "#%02x%02x%02x" % (ch, g, b),
                       "rgb": [ch, g, b],
                       "hsv": [int(round(int(hsv[0]) * 360 / 180)),
                               int(round(int(hsv[1]) * 100 / 255)),
                               int(round(int(hsv[2]) * 100 / 255))],
                       "area": int(3.1416 * r * r), "circularity": 1.0, "source": "grid_fill"})
    return filled


def detect_palette_row(image_path: str, swatches=None, want: int = 5) -> list:
    """案例页「主色调」那一排色板的实测色值（按从左到右）。

    先按严格阈值找圆；找不到就逐级放宽阈值；再不行用几何补齐。
    这样浅色色板（粉红、米黄）不会因为饱和度低被整排丢掉。
    """
    import cv2
    import numpy as np

    sw = swatches if swatches is not None else measure_swatches(image_path)
    row = pick_palette_row(sw, want)
    if row:
        return row
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    for sat_floor in (25, 14, 6):
        row = pick_palette_row(measure_swatches(image_path, min_saturation=sat_floor), want)
        if row:
            return row
    partial = _best_partial_row(sw, want)
    return _complete_row(image_path, partial, want) if partial else []


def attach_swatches(page_json: dict, swatches: list, want: int = 5, image_path: str = "") -> dict:
    """把实测色值按「同一排、从左到右」贴到视觉 JSON 的 `main_palette` 上。

    中文色名来自视觉读取（最可靠），真实色值来自 OpenCV 实测（最可靠），
    两边按**位置顺序**对齐——这也是为什么 system prompt 要求 `main_palette` 按从左到右排序。
    """
    palette = list(page_json.get("main_palette") or [])
    row = pick_palette_row(swatches, want=want) if palette else []
    if palette and len(row) != want and image_path:
        row = detect_palette_row(image_path, swatches=swatches, want=want)
    for i, (item, s) in enumerate(zip(palette, row)):
        item["hex"] = s["hex"]
        item["hsv"] = s["hsv"]
        item["hex_source"] = "measured" if s.get("source") != "grid_fill" else "measured_grid_fill"
        item["swatch_xy"] = [s["x"], s["y"]]
        # 需要人眼过一遍的两种情况：色值是几何补齐出来的（不是真检测到的圆），
        # 或者模型给的色名条数与实测圆点数不一致（说明位置对齐可能是错的）。
        item["needs_review"] = bool(s.get("source") == "grid_fill" or len(palette) != len(row))
    if palette and row and len(palette) != len(row):
        page_json.setdefault("warnings", []).append(
            f"色板排有 {len(row)} 个圆、色名有 {len(palette)} 条，只对齐了 {min(len(row), len(palette))} 组")
    elif palette and not row:
        page_json.setdefault("warnings", []).append("这一页没测到等距色板排，主色调只有色名没有实测色值")
    page_json["main_palette"] = palette
    page_json["palette_row_measured"] = row
    page_json["swatches_measured"] = swatches
    return page_json


# ---------------------------------------------------------------- 合成知识库

def render_palette_check(page_path: str, row: list, label: str = "", names=None, row_height: int = 190,
                         chip: int = 150) -> "object":
    """画一条核对图：上面是页面上色板那一排的原图裁剪，下面是从实测色值复原的色块 + 色名 + hex。

    这是给**人眼**看一眼用的：模型读的色名和 OpenCV 量的色值对不对得上，一眼就能发现
    （实测第 29 页有一格是栅格补齐补出来的深黑，必须在写进知识库前被人拦一下；
     同一页模型还把印刷的「丁子色」读成了「丁香色」，只有对照着看才发现得了）。
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    font = _cjk_font(22)
    img = Image.open(page_path).convert("RGB")
    w, h = img.size
    if row:
        y = int(sum(s["y"] for s in row) / len(row))
        top = max(0, y - row_height // 2)
        crop = img.crop((0, top, w, min(h, top + row_height))).resize((w, row_height))
    else:
        crop = Image.new("RGB", (w, row_height), "white")
    canvas = Image.new("RGB", (w, row_height + chip + 34), "white")
    canvas.paste(crop, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((8, row_height + 6), label, fill="black", font=font)
    names = list(names or [])
    x = 8
    for i, s in enumerate(row):
        d.rectangle([x, row_height + 34, x + chip - 14, row_height + 34 + chip - 62], fill=s["hex"])
        d.text((x + 2, row_height + 34 + chip - 56), s["hex"], fill="black", font=font)
        d.text((x + 2, row_height + 34 + chip - 30), names[i] if i < len(names) else "", fill="black", font=font)
        x += chip
    return canvas


def _cjk_font(size: int):
    """找一份中文字体给 PIL 用；没有就退回默认字体（中文会变方框，不影响色值）。"""
    from PIL import ImageFont

    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf"):
        path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def build_check_sheet(page_json_dir: str, pages_dir: str, out_path: str, limit: int = 0) -> str:
    """把所有带实测色板的页拼成一张核对大图（人眼扫一遍 = 完成一轮色值验收）。"""
    import glob as _g

    from PIL import Image

    tiles = []
    for path in sorted(_g.glob(os.path.join(page_json_dir, "page-*.json"))):
        p = json.load(open(path, encoding="utf-8"))
        row = p.get("palette_row_measured") or []
        if not row:
            continue
        img = os.path.join(pages_dir, "page-%03d.jpg" % int(p.get("pdf_page") or 0))
        if not os.path.isfile(img):
            continue
        names = [str(c.get("name_zh") or "?") for c in (p.get("main_palette") or [])]
        label = f"pdf p{p.get('pdf_page')} / 印刷 p{p.get('printed_page')}  {p.get('work_title') or p.get('title') or ''}"
        tiles.append(render_palette_check(img, row, label=label, names=names))
        if limit and len(tiles) >= limit:
            break
    if not tiles:
        return ""
    width = max(t.size[0] for t in tiles)
    canvas = Image.new("RGB", (width, sum(t.size[1] for t in tiles)), "white")
    y = 0
    for t in tiles:
        canvas.paste(t, (0, y))
        y += t.size[1]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    return out_path
def _rule_key(rule: dict) -> str:
    return re.sub(r"\W+", "", str(rule.get("concept") or rule.get("title") or "").lower())[:60]


CHAPTER_SLUGS = {
    "给米": "gemi", "波特歌": "potg", "樱田千寻": "sakurada", "亚纪": "aki", "约尔·福杰": "fajyobore",
    "巴尼石门": "banishment", "寺田寺": "terada", "木野花日兰子": "konohana", "三土": "redum",
    "拉斯库": "rasuku", "理论": "theory",
}


def case_slug(page_json: dict) -> str:
    """案例的稳定 ASCII 标识：`color-<画师>-p<印刷页>`。GUI 下拉、文件名、CLI 参数都用它。"""
    chapter = str(page_json.get("chapter") or "")
    slug = next((v for k, v in CHAPTER_SLUGS.items() if k and k in chapter), "case")
    return "color-%s-p%s" % (slug, page_json.get("printed_page"))


def crop_to_art(page_path: str, margin: int = 8, min_area_ratio: float = 0.15):
    """把扫描页裁到「最大的那一块画面」：去掉纸白边、页码、书眉，以及旁边的小图表。

    为什么要裁：`tone_calibrate` 只取参考图的 **LAB 明度均值 + HSV 饱和度均值**。
    纸白是纯白、说明文字与图表是浅底，会把目标明度往上拉、饱和度往下拉，
    等于给了一道"偏白偏灰"的假目标（实测：作品页下方挂着色相平衡/色调平衡两个白底小图，
    不剔掉的话参考图里有三成是纸）。

    做法：与这一页纸色差 > 38 的像素构成前景 → 形态学合并 → **只留最大连通域** → 取它的包围盒。
    最大连通域就是插画本体；旁边的小图表各自成块，会被自然丢掉。
    """
    import cv2
    import numpy as np
    from PIL import Image

    img = cv2.imdecode(np.fromfile(page_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    paper = _paper_color(img, h // 2, band=h // 2)
    dist = np.abs(img.astype(np.int32) - paper.reshape(1, 1, 3)).max(axis=2)
    mask = (dist > 38).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    # 核不能太大：61 会把插画与它下面那两块白底图表（色相平衡/色调平衡）粘成一块，
    # 于是参考图里又混进三成纸白。实测 25 刚好 —— 在 9~41 之间，最大连通域都是插画本体。
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    # 第 0 个是背景，跳过；按面积取最大
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[idx]
    if area < min_area_ratio * w * h or bw < 0.3 * w:
        return None
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
    return Image.fromarray(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))


def _art_coverage(page_path: str) -> float:
    """这一页「有画」的面积占比。整页插图接近 1，分析页/文字页只有两三成。"""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(page_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return -1.0
    h, w = img.shape[:2]
    paper = _paper_color(img, h // 2, band=h // 2)
    dist = np.abs(img.astype(np.int32) - paper.reshape(1, 1, 3)).max(axis=2)
    return float((dist > 38).mean())


def make_tone_targets(page_json_dir: str, pages_dir: str, out_dir: str, from_page: str = "auto",
                      log_callback=None) -> list:
    """为每个案例生成一张**无纸白、无文字**的 tone 目标图 → `data/color-knowledge/targets/`。

    `from_page="auto"`：在分析页的前一页/后一页里挑「画面占比更高」的那张——
    书里「整页插图 + 分析页」的前后顺序不固定（实测带上你的文字是插在前、落色是插在后），
    所以不能写死 `pdf_page - 1`，得按画面占比选。
    `from_page="analysis"`：直接用分析页本身（裁掉纸白后仍可用，但会带着图表和色板）。
    """
    import glob as _g

    from PIL import Image

    def log(msg):
        if log_callback:
            log_callback(msg)

    os.makedirs(out_dir, exist_ok=True)
    made = []
    for path in sorted(_g.glob(os.path.join(page_json_dir, "page-*.json"))):
        p = json.load(open(path, encoding="utf-8"))
        if p.get("page_type") != "case_analysis":
            continue
        pdf_page = int(p.get("pdf_page") or 0)
        if from_page == "analysis":
            candidates = [pdf_page]
        elif from_page == "illustration":
            candidates = [pdf_page - 1]
        else:
            candidates = [pdf_page - 1, pdf_page + 1]
        best, best_cov = None, -1.0
        for cand in candidates:
            src = os.path.join(pages_dir, "page-%03d.jpg" % cand)
            if not os.path.isfile(src):
                continue
            cov = _art_coverage(src)
            if cov > best_cov:
                best, best_cov = src, cov
        if best is None:
            log(f"[color-knowledge] 第 {pdf_page} 页：候选页都不存在，跳过")
            continue
        art = crop_to_art(best)
        if art is None:
            log(f"[color-knowledge] 第 {pdf_page} 页：{os.path.basename(best)} 裁不出画面区域，跳过")
            continue
        dest = os.path.join(out_dir, "%s.png" % case_slug(p))
        art.save(dest)
        made.append({"slug": case_slug(p), "path": dest, "pdf_page": pdf_page,
                     "source_page": int(re.search(r"(\d+)", os.path.basename(best)).group(1)),
                     "art_coverage": round(best_cov, 3), "size": list(art.size),
                     "work_title": p.get("work_title") or p.get("title")})
        log(f"[color-knowledge] {case_slug(p)} ← {os.path.basename(best)}（画面占比 {best_cov:.0%}）"
            f"裁出 {art.size[0]}x{art.size[1]}")
    with open(os.path.join(out_dir, "targets.json"), "w", encoding="utf-8") as f:
        json.dump({"targets": made, "count": len(made)}, f, ensure_ascii=False, indent=2)
    log(f"[color-knowledge] tone 目标图 {len(made)} 张 → {out_dir}")
    return made


# ---------------------------------------------------------------- 扫描页 → 可检索 HTML

_META_RE = re.compile(r"^\s*<!--meta\s*(.*?)-->", re.S)


def parse_page_markdown(text: str) -> dict:
    """拆出页头 `<!--meta ... -->`（key: value 行）与正文。"""
    text = text.replace("\r\n", "\n")
    meta = {}
    body = text
    m = _META_RE.match(text)
    if m:
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            value = value.strip()
            if value in ("", "null", "None", "~"):
                value = None
            meta[key.strip()] = value
        body = text[m.end():].lstrip("\n")
    return meta, body


def _inline_md(s: str) -> str:
    """行内标记：先转义，再处理 `code` 与 **bold**。

    `<br>` 是唯一放行的 HTML 标签（表格单元格里要换行；其余标签一律转义，免得转录内容里
    出现了尖括号就把整页版式带歪）。实测漏掉这一步会让 12 处 `<br>` 原样显示成文字。
    """
    s = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    s = s.replace("&lt;br&gt;", "<br>").replace("&lt;br/&gt;", "<br>")
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    return s


def _strip_leading_h1(body: str) -> str:
    """去掉正文开头的 `# 标题`。

    页面标题由 `.phead` 单独渲染（那里还带章节/页码信息），正文再顶一个 `<h1>` 就是重复。
    """
    lines = body.lstrip("\n").split("\n")
    if lines and re.match(r"^#\s+\S", lines[0].strip()):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines)


def _md_to_html(md: str) -> str:
    """够用就好的 Markdown 子集渲染：标题 / 表格 / 引用 / 列表 / 代码块 / 分隔线 / 行内标记。

    不引第三方库（`PROJECT_REQUIREMENTS.md`：不许装依赖），只支持我们自己写的这几种语法。
    """
    out, i = [], 0
    lines = md.split("\n")
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):                       # 代码块
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(
                l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for l in buf) + "</code></pre>")
            continue

        if re.match(r"^(-{3,}|\*{3,})$", stripped):          # 分隔线
            out.append("<hr>")
            i += 1
            continue

        m = re.match(r"^!\[([^\]]*)\]\(([^)\s]+)\)\s*$", stripped)   # 独占一行的配图
        if m:
            src = m.group(2).replace('"', "")
            out.append(f'<figure class="fig"><img src="{src}" loading="lazy" alt="{_inline_md(m.group(1))}">'
                       f'<figcaption>{_inline_md(m.group(1))}</figcaption></figure>')
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)         # 标题
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{_inline_md(m.group(2).strip())}</h{level}>")
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            head = [c.strip() for c in stripped.strip("|").split("|")]      # 表格
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            html = ["<table><thead><tr>"]
            html += [f"<th>{_inline_md(c)}</th>" for c in head]
            html.append("</tr></thead><tbody>")
            for r in rows:
                html.append("<tr>" + "".join(f"<td>{_inline_md(c)}</td>" for c in r) + "</tr>")
            html.append("</tbody></table>")
            out.append("".join(html))
            continue

        if stripped.startswith(">"):                          # 引用（连续行合并）
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + _md_to_html("\n\n".join(buf)) + "</blockquote>")
            continue

        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)     # 列表
        if m:
            ordered = bool(re.match(r"\d+\.", m.group(2)))
            tag = "ol" if ordered else "ul"
            items = []
            while i < len(lines):
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if not mm:
                    break
                items.append(mm.group(3))
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{_inline_md(x)}</li>" for x in items) + f"</{tag}>")
            continue

        if not stripped:                                      # 空行
            i += 1
            continue

        buf = []                                              # 段落
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^\s*(#{1,6}\s|>|\||```|[-*]\s|\d+\.\s|-{3,}$)", lines[i]):
            buf.append(lines[i].strip())
            i += 1
        if buf:
            out.append("<p>" + _inline_md(" ".join(buf)) + "</p>")
        else:
            # 兜底：这一行没被上面任何分支吃掉（例如孤零零一行以 `|` 开头但不是表格），
            # 这里若不推进 `i`，外层 while 就会**原地死循环**。
            # 实测：PDF 15 那份 130 行色表触发过一次，整个站点构建卡死十几分钟没有输出。
            out.append("<p>" + _inline_md(stripped) + "</p>")
            i += 1
    return "\n".join(out)


_HTML_CSS = """
:root{--ink:#1b1b1f;--mut:#6b6b76;--line:#e3e3ea;--bg:#fbfbfd;--acc:#7a4bd0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:16px/1.85 "Microsoft YaHei","PingFang SC","Hiragino Sans GB",sans-serif}
#top{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid var(--line);
 padding:10px 18px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
#top h1{font-size:17px;margin:0;font-weight:600;white-space:nowrap}
#q{flex:1;min-width:200px;padding:8px 12px;border:1px solid var(--line);border-radius:8px;
 font-size:15px;font-family:inherit}
#q:focus{outline:2px solid var(--acc);outline-offset:-1px}
#count{color:var(--mut);font-size:13px;white-space:nowrap}
#nav{position:sticky;top:53px;z-index:8;background:#fff;border-bottom:1px solid var(--line);
 padding:7px 18px;display:flex;gap:8px;flex-wrap:wrap;max-height:132px;overflow:auto}
#nav a{font-size:13px;color:var(--ink);text-decoration:none;border:1px solid var(--line);
 border-radius:999px;padding:2px 11px;background:#fff}
#nav a:hover,#nav a.on{border-color:var(--acc);color:var(--acc)}
main{max-width:1180px;margin:0 auto;padding:18px}
.page{background:#fff;border:1px solid var(--line);border-radius:12px;padding:20px 26px;
 margin:0 0 22px;scroll-margin-top:110px}
.page.hide{display:none}
.phead{display:flex;justify-content:space-between;align-items:baseline;gap:14px;
 border-bottom:2px solid var(--acc);padding-bottom:8px;margin-bottom:14px;flex-wrap:wrap}
.phead .who{color:var(--mut);font-size:13px}
.phead .ttl{font-size:19px;font-weight:700}
/* 正文单栏铺满：原来这里是「左文右扫描图」两列，
   扫描图改成页眉里的一个小链接之后，右列空间还给正文。 */
.grid{display:block}
/* 扫描原页的入口 —— 只在页眉占一个小胶囊，点击新开标签看原图 */
.src{font-size:12px;color:var(--mut);border:1px solid var(--line);border-radius:999px;
 padding:1px 9px;text-decoration:none;white-space:nowrap;margin-left:8px}
.src:hover{border-color:var(--acc);color:var(--acc)}
figure{margin:0}
figure img{width:100%;border:1px solid var(--line);border-radius:6px;display:block}
figcaption{color:var(--mut);font-size:12px;margin-top:6px;text-align:center}
figure.fig{margin:16px 0;padding:11px;background:#faf9fd;border:1px solid var(--line);border-radius:9px}
figure.fig img{background:#fff}
.pal{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0 0}
.pal span{border:1px solid var(--line);border-radius:5px;padding:3px 8px;font-size:13px;
 font-family:ui-monospace,Consolas,monospace}
.sw{display:inline-block;width:15px;height:15px;border-radius:3px;border:1px solid #0002;
 vertical-align:-3px;margin-right:6px}
h1,h2,h3,h4{line-height:1.4}
.page h1{font-size:19px;margin:14px 0 8px}
h2{font-size:18px;margin:22px 0 8px;padding-left:9px;border-left:4px solid var(--acc)}
h3{font-size:16px;margin:17px 0 6px;color:#333}
p{margin:9px 0}
blockquote{margin:12px 0;padding:9px 15px;background:#f6f3fd;border-left:4px solid var(--acc);
 border-radius:0 7px 7px 0;color:#3a3a45}
blockquote p{margin:3px 0}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
th{background:#f4f2fa;font-weight:600}
code{background:#f2f2f6;border-radius:4px;padding:1px 5px;font-family:ui-monospace,Consolas,monospace;
 font-size:13px}
pre{background:#26262e;color:#e8e8f0;border-radius:8px;padding:12px 14px;overflow:auto;font-size:13px}
pre code{background:none;color:inherit;padding:0}
hr{border:none;border-top:1px dashed var(--line);margin:20px 0}
mark{background:#ffe680;padding:0 2px;border-radius:2px}
ul,ol{margin:9px 0;padding-left:24px}
li{margin:3px 0}
.empty{color:var(--mut);text-align:center;padding:40px}
"""

_HTML_JS = """
var pages=[].slice.call(document.querySelectorAll('.page'));
var navLinks=[].slice.call(document.querySelectorAll('#nav a'));
var q=document.getElementById('q'), cnt=document.getElementById('count');
function plain(el){return (el.dataset.raw||'').toLowerCase();}
function run(){
  var t=q.value.trim().toLowerCase(); var n=0;
  pages.forEach(function(p){
    var hit=!t||plain(p).indexOf(t)>=0;
    p.classList.toggle('hide',!hit); if(hit)n++;
  });
  navLinks.forEach(function(a){
    var p=document.getElementById(a.getAttribute('href').slice(1));
    a.style.display=(p&&!p.classList.contains('hide'))?'':'none';
  });
  cnt.textContent=t?('命中 '+n+' / '+pages.length+' 页'):(pages.length+' 页');
}
q.addEventListener('input',run); run();
document.addEventListener('keydown',function(e){
  if(e.key==='/'&&document.activeElement!==q){e.preventDefault();q.focus();}
});
"""


def load_book_structure(path_or_dir: str) -> dict:
    """读 `structure.json`（章节划分 + 逐页标题）。

    接受目录或文件路径。目录下没有 `structure.json` 时返回一个"整本一章"的兜底结构，
    这样 `html` 永远不会因为缺结构文件而失败。
    """
    path = path_or_dir
    if os.path.isdir(path):
        path = os.path.join(path, "structure.json")
    if not os.path.isfile(path):
        return {"title": "扫描页全文", "chapters": [{"id": "ch00", "title": "全部", "pdf": [1, 9999]}],
                "pages": {}, "front_pages": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fallback_page_title(structure: dict, printed, chapter_range=None) -> str:
    """没被 `structure.json` 命名的页，用**同一章内最近的前一篇作品**做上下文标签。

    本书每篇作品的版面是「整页插图 + 分析页 + 配色亮点若干页 + 绘画过程若干页」，
    只有分析页在目录里挂了名字。整站 150 页里约 60 页属于这种"无名版面"。

    两条必须遵守的约束（都是实测踩出来的）：
    1. **锚点用 `kind == "case_analysis"` 的作品页**，不能只找"最近一个有名页"——
       笔记页插在作品之间，会把作品的亮点页错挂到笔记名下。
    2. **只在同一章范围内找**——`chapter_range` 是 (本章印刷页起, 止)。
       不限制的话，章首页会挂到上一章最后一篇作品名下（实测 PDF 67 属第 5 章，
       却被挂成了第 4 章的《天空之色果酱 · 版面》）。
    """
    pages_map = structure.get("pages") or {}
    if printed is None:
        return "未标注版面"
    lo, hi = chapter_range or (None, None)

    def _nums(kinds):
        out = []
        for k, v in pages_map.items():
            v = v or {}
            if not v.get("title"):
                continue
            if kinds is not None and v.get("kind") not in kinds:
                continue
            n = int(k)
            if lo is not None and n < lo:
                continue
            if hi is not None and n > hi:
                continue
            out.append(n)
        return sorted(out)

    works = _nums({"case_analysis"})
    # 「有名字的页」里要**排除章节封面**：封面上的名字是画师名，不是作品名，
    # 拿它当锚点会把章首那张整页插图挂成「《亚纪（Aki）》· 版面」（实测 PDF 67）。
    # 排除之后 PDF 67 会落到下面的 "after" 分支上，正确归属为《星泉的守护神》· 前置版面
    # —— 因为本书的版面就是「章节封面 → 整页插图 → 作品分析页」。
    named = [n for n in _nums(None)
             if (pages_map.get(str(n)) or {}).get("kind") != "chapter_cover"]
    for pool in (works, named):
        before = [n for n in pool if n < printed]
        if before:
            return "《%s》· 版面（目录未单列）" % pages_map[str(before[-1])]["title"]
    after = [n for n in named if n > printed]
    if after:
        return "《%s》· 前置版面" % pages_map[str(after[0])]["title"]
    return "未标注版面"


def page_meta(structure: dict, pdf_page: int) -> dict:
    """PDF 页序号 → {chapter_id, chapter_title, printed, title, kind, theme}。"""
    offset = int(structure.get("offset") or 5)
    printed = pdf_page - offset
    chapter = {"id": "ch00", "title": "未分章"}
    for ch in structure.get("chapters") or []:
        lo, hi = (ch.get("pdf") or [0, 0])[:2]
        if int(lo) <= pdf_page <= int(hi):
            chapter = ch
            break
    info = (structure.get("pages") or {}).get(str(printed)) or {}
    front = (structure.get("front_pages") or {}).get(str(pdf_page))
    title = info.get("title") or front or ""
    if not title:
        pdf_range = (chapter.get("pdf") or [None, None])[:2]
        ch_range = (int(pdf_range[0]) - offset, int(pdf_range[1]) - offset) \
            if pdf_range[0] is not None else None
        title = _fallback_page_title(structure, printed if printed >= 1 else None, ch_range)
    kind = info.get("kind") or ("front_matter" if printed <= 7 else "")
    return {"chapter_id": chapter.get("id"), "chapter_title": chapter.get("title"),
            "printed": printed if printed >= 1 else None, "title": title,
            "kind": kind, "theme": info.get("theme") or "",
            "named": bool(info.get("title") or front)}


def _page_figure_files(fig_dir: str, pdf_page: int) -> list:
    """这一页已抠出的配图文件（按 fig 编号排序）。"""
    import glob as _g
    out = []
    for p in sorted(_g.glob(os.path.join(fig_dir, "page-%03d-fig*.*" % pdf_page))):
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".webp"):
            out.append(os.path.basename(p))
    return out


def _figure_captions(fig_dir: str, pdf_page: int) -> dict:
    """读抠图工序写的 `page-NNN.json`，取出 id → caption。"""
    path = os.path.join(fig_dir, "page-%03d.json" % pdf_page)
    if not os.path.isfile(path):
        return {}
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return {f.get("id"): f.get("caption") for f in (data.get("figures") or []) if f.get("caption")}


_NAV_CSS = """
.crumb{color:var(--mut);font-size:13px}
.crumb a{color:var(--acc);text-decoration:none}
.pager{margin-left:auto;display:flex;gap:8px}
.pager a{font-size:13px;border:1px solid var(--line);border-radius:8px;padding:4px 12px;
 color:var(--ink);text-decoration:none;background:#fff}
.pager a:hover{border-color:var(--acc);color:var(--acc)}
.pager a.off{color:#c8c8d0;pointer-events:none}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px;margin-top:18px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px;
 text-decoration:none;color:var(--ink);display:block}
.card:hover{border-color:var(--acc);box-shadow:0 3px 14px #7a4bd020}
.card h3{margin:0 0 6px;font-size:17px}
.card .meta{color:var(--mut);font-size:13px;margin-bottom:9px}
.card .note{color:#4a4a55;font-size:13px;margin-bottom:9px}
.bar{height:6px;border-radius:3px;background:#eee;overflow:hidden}
.bar i{display:block;height:100%;background:var(--acc)}
.card .pct{color:var(--mut);font-size:12px;margin-top:5px}
.thumbs{display:flex;gap:5px;margin-top:9px}
.thumbs img{width:52px;border:1px solid var(--line);border-radius:4px;display:block}
.todo{color:var(--mut);background:#faf9fd;border:1px dashed var(--line);border-radius:9px;
 padding:12px 14px;font-size:14px;margin:10px 0}
.todo b{color:#8a6a00}
#pnav{position:sticky;top:53px;z-index:8;background:#fff;border-bottom:1px solid var(--line);
 padding:7px 18px;display:flex;flex-direction:column;gap:3px;max-height:190px;overflow:auto}
#pnav a{font-size:13px;color:var(--ink);text-decoration:none;border-radius:6px;padding:3px 8px}
#pnav a:hover{background:#f4f2fa;color:var(--acc)}
.pnavlink{display:flex;justify-content:space-between;gap:8px}
.pnavlink em{color:#b08a00;font-style:normal;font-size:11px}
"""


def _load_pages(root: str, structure: dict, log) -> list:
    """收集全书每一页的渲染素材（有 md 就用 md，没有就只靠结构表 + 页图 + 配图）。"""
    pages_dir = os.path.join(root, "pages")
    img_dir = os.path.join(root, "images")
    fig_dir = os.path.join(root, "figures")
    entries = []
    for path in sorted((__import__("glob").glob(os.path.join(pages_dir, "page-*.md")))):
        num = int(re.search(r"(\d+)", os.path.basename(path)).group(1))
        entries.append(num)
    # 结构表 + 页图覆盖到的页也要出现（即使还没转录）
    import glob as _g
    have = set(entries)
    for p in _g.glob(os.path.join(img_dir, "page-*.jpg")):
        num = int(re.search(r"(\d+)", os.path.basename(p)).group(1))
        if num not in have:
            entries.append(num)
            have.add(num)
    out = []
    for num in sorted(entries):
        meta = page_meta(structure, num)
        md_path = os.path.join(pages_dir, "page-%03d.md" % num)
        body, md_meta = "", {}
        if os.path.isfile(md_path):
            md_meta, body = parse_page_markdown(open(md_path, encoding="utf-8").read())
            body = _strip_leading_h1(body)
        # 转录状态以页头显式声明的 `transcribed:` 为准；老文件没这个键就按正文字数判。
        # 不能用「文件存在」判——占位页也有文件，那样全书都会显示成"已转录"。
        flag = md_meta.get("transcribed")
        if flag is None:
            done = len(body.strip()) > 60
        else:
            done = str(flag).strip().lower() not in ("false", "0", "no", "none", "")
        # 标题优先级：**转录时写的 meta 最权威**（那是看着扫描页写的），
        # 其次才是 structure.json / 兜底规则。
        # 实测：印刷 44（PDF 49）是《即使这样也要画下去》的整页插图，
        # 但兜底规则只看"最近的前一篇作品"，把它挂成了《水底花圃 · 插图》——
        # 因为插图页排在**它所属作品的分析页之前**，而亮点页排在**之后**，
        # 光靠位置推不出来。转录过的页以 md 为准就不会有这类错。
        meta_final = dict(meta)
        if md_meta.get("title"):
            meta_final["title"] = str(md_meta["title"]).strip()
        if md_meta.get("page_type"):
            meta_final["kind"] = str(md_meta["page_type"]).strip()
        rec = {"pdf_page": num, "meta": meta_final, "md_meta": md_meta, "body": body,
               "transcribed": done,
               "img": "images/page-%03d.jpg" % num if os.path.isfile(os.path.join(img_dir, "page-%03d.jpg" % num)) else "",
               "figs": _page_figure_files(fig_dir, num),
               "fig_captions": _figure_captions(fig_dir, num)}
        out.append(rec)
    return out


def _render_body(rec: dict) -> str:
    """正文：有转录就渲染 md；没转录就给出结构表信息 + 已抠出的配图。"""
    if rec["transcribed"]:
        html = _md_to_html(rec["body"])
    else:
        html = ""
    if rec["figs"]:
        extra = []
        for name in rec["figs"]:
            fig_id = re.search(r"-(fig\d+)\.png$", name)
            cap = rec["fig_captions"].get(fig_id.group(1) if fig_id else "", "")
            if not cap:
                cap = "从扫描页自动抠出的配图（%s）" % (fig_id.group(1) if fig_id else name)
            extra.append(f'<figure class="fig"><img src="figures/{name}" loading="lazy" '
                         f'alt="{_inline_md(cap)}"><figcaption>{_inline_md(cap)}</figcaption></figure>')
        if rec["transcribed"]:
            html += "".join(extra)
        else:
            html = ('<div class="todo">这一页<b>尚未逐字转录</b>。'
                    '下面是从扫描页自动抠出的配图，页图见右栏。</div>' + "".join(extra))
    elif not rec["transcribed"]:
        html = '<div class="todo">这一页<b>尚未逐字转录</b>。页图见右栏。</div>'
    pal = ""
    if rec["md_meta"].get("palette"):
        chips = []
        for part in str(rec["md_meta"]["palette"]).split("|"):
            part = part.strip()
            mm = re.match(r"^(.*?)\s*(#[0-9a-fA-F]{6})$", part)
            if mm:
                chips.append(f'<span><i class="sw" style="background:{mm.group(2)}"></i>'
                             f'{_inline_md(mm.group(1).strip())} {mm.group(2)}</span>')
            elif part:
                chips.append(f"<span>{_inline_md(part)}</span>")
        pal = '<div class="pal">' + "".join(chips) + "</div>"
    return html + pal


def _render_section(rec: dict, heading: str) -> str:
    meta = rec["meta"]
    who = " · ".join(x for x in [
        "PDF p%d" % rec["pdf_page"],
        ("印刷 p%s" % meta["printed"]) if meta.get("printed") else "前置页",
        meta.get("kind") or "",
        meta.get("theme") or ""] if x)
    img = ""
    if rec["img"]:
        # 扫描原页不再占右侧半屏，只在页眉留一个可点的小入口。
        img = (f'<a class="src" href="{rec["img"]}" target="_blank" '
               f'title="打开 PDF 第 {rec["pdf_page"]} 页的扫描原图">扫描原页 ↗</a>')
    raw = (str(heading) + " " + who + " " + rec["body"]).replace('"', "&quot;").replace("\n", " ")
    return (f'<section class="page" id="p{rec["pdf_page"]:03d}" data-raw="{raw}">'
            f'<div class="phead"><div class="ttl">{_inline_md(heading)}</div>'
            f'<div class="who">{_inline_md(who)}{img}</div></div>'
            f'<div class="grid">{_render_body(rec)}</div></section>')


def _page_files(rec: dict) -> str:
    return f'<a class="pf" href="#p{rec["pdf_page"]:03d}">{_inline_md(rec["meta"].get("title") or ("PDF p%d" % rec["pdf_page"]))}</a>'


def build_book_site(root: str, out_dir: str = "", log_callback=None) -> dict:
    """把整本书渲染成「目录页 + 每章一页」的多文件站点（纯静态，双击即看）。

    - `index.html`：书名 + 进度条 + 每章一张卡片（页数 / 已转录数 / 缩略图）→ 点进章节；
    - `chapter-<id>.html`：本章的逐页图文（左文右图），带章内页码导航、搜索框、
      上一章 / 下一章 / 返回目录；
    - `book.md`、`index.json`：全文纯文本与结构化索引，给 grep 与工具链用。

    没转录的页也会出现：给出结构表里的标题、页型、印刷页码，右栏是扫描原页，
    正文位置列出已自动抠出的配图。这样"骨架"先把全书 150 页铺满，正文可以按章慢慢补。
    """
    def log(msg):
        if log_callback:
            log_callback(msg)

    root = os.path.abspath(root)
    out_dir = os.path.abspath(out_dir or root)
    os.makedirs(out_dir, exist_ok=True)
    structure = load_book_structure(root)
    pages = _load_pages(root, structure, log)
    if not pages:
        log("[color-knowledge] 没找到任何页图或 pages/*.md，先跑 extract / figures")
        return {}
    chapters = structure.get("chapters") or [{"id": "ch00", "title": "全部", "pdf": [1, 9999]}]
    by_ch = {}
    for rec in pages:
        by_ch.setdefault(rec["meta"]["chapter_id"], []).append(rec)

    # ---- 每章一页
    live = [c for c in chapters if by_ch.get(c["id"])]
    ch_files = {c["id"]: "chapter-%s.html" % c["id"] for c in live}
    for i, ch in enumerate(live):
        recs = by_ch[ch["id"]]
        name = ch_files[ch["id"]]
        prev_ch = live[i - 1] if i > 0 else None
        next_ch = live[i + 1] if i + 1 < len(live) else None
        done = sum(1 for r in recs if r["transcribed"])
        nav = ['<a href="index.html">← 目录</a>']
        nav.append(f'<a href="{ch_files[prev_ch["id"]]}">← {_inline_md(prev_ch["title"])}</a>' if prev_ch
                   else '<a class="off">← 上一章</a>')
        nav.append(f'<a href="{ch_files[next_ch["id"]]}">下一章：{_inline_md(next_ch["title"])} →</a>' if next_ch
                   else '<a class="off">已经是最后一章</a>')
        plist = "".join(
            f'<a href="#p{r["pdf_page"]:03d}" class="pnavlink">{_inline_md(r["meta"].get("title") or "PDF p%d" % r["pdf_page"])}'
            f'<em>{"" if r["transcribed"] else "待转录"}</em></a>' for r in recs)
        secs = []
        for k, r in enumerate(recs):
            heading = r["meta"].get("title") or ("PDF 第 %d 页" % r["pdf_page"])
            secs.append(_render_section(r, heading))
        html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_inline_md(ch["title"])} · {_inline_md(structure.get("title") or "")}</title>
<style>{_HTML_CSS}{_NAV_CSS}</style></head><body>
<div id="top"><h1>{_inline_md(ch["title"])}</h1>
<input id="q" type="search" placeholder="本章内过滤（/ 聚焦）— Ctrl+F 全局查找" autocomplete="off">
<span id="count"></span><span class="pager">{"".join(nav)}</span></div>
<div id="pnav">{plist}</div>
<main><div class="crumb">{_inline_md(structure.get("title") or "")} ／ 本章 {len(recs)} 页，
已转录 <b>{done}</b> 页</div>{"".join(secs)}
<p class="empty">— 由 tools/color_knowledge.py html 生成 —</p></main>
<script>{_HTML_JS}</script></body></html>
"""
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(html)

    # ---- 目录页
    total, done_total = len(pages), sum(1 for r in pages if r["transcribed"])
    cards = []
    for ch in chapters:
        recs = by_ch.get(ch["id"]) or []
        if not recs:
            continue
        done = sum(1 for r in recs if r["transcribed"])
        pct = int(round(100 * done / max(1, len(recs))))
        lo = min(r["pdf_page"] for r in recs)
        hi = max(r["pdf_page"] for r in recs)
        thumbs = "".join(f'<img src="images/page-{r["pdf_page"]:03d}.jpg" loading="lazy" alt="">'
                         for r in recs[:7] if r["img"])
        pr = ch.get("printed_range") or ""
        cards.append(
            f'<a class="card" href="{ch_files[ch["id"]]}"><h3>{_inline_md(ch["title"])}</h3>'
            f'<div class="meta">PDF {lo}–{hi} 页（{len(recs)} 页）{pr}</div>'
            f'<div class="note">{_inline_md(ch.get("note") or "")}</div>'
            f'<div class="bar"><i style="width:{pct}%"></i></div>'
            f'<div class="pct">已转录 {done} / {len(recs)} 页（{pct}%）</div>'
            f'<div class="thumbs">{thumbs}</div></a>')
    idx_html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_inline_md(structure.get("title") or "扫描页全文")}</title>
<style>{_HTML_CSS}{_NAV_CSS}</style></head><body>
<div id="top"><h1>{_inline_md(structure.get("title") or "扫描页全文")}</h1>
<span id="count">{len(chapters)} 章 · {total} 页 · 已转录 {done_total} 页</span></div>
<main>
<p class="crumb">源文件 <code>{_inline_md(structure.get("source_pdf") or "")}</code>
 ／ 印刷页码 = PDF 页序号 − {structure.get("offset") or 5}
 ／ 全文纯文本 <a href="book.md">book.md</a>（可 grep）· 结构化索引 <a href="index.json">index.json</a></p>
<div class="cards">{"".join(cards)}</div>
<p class="empty">点任意一章进入该章的逐页图文（左文右图，可搜索、可翻页）。</p>
</main></body></html>
"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(idx_html)

    # ---- 纯文本 + 索引
    raw, index = [], []
    for rec in pages:
        m = rec["meta"]
        raw.append(f"\n\n<!-- ===== PDF p{rec['pdf_page']} / 印刷 p{m.get('printed')} / "
                   f"{m.get('kind') or ''} / {m.get('title') or ''} ===== -->\n"
                   f"{m.get('title') or ''}\n\n" + (rec["body"].strip() or "（尚未转录）"))
        index.append({"pdf_page": rec["pdf_page"], "printed_page": m.get("printed"),
                      "chapter": m.get("chapter_title"), "chapter_id": m.get("chapter_id"),
                      "title": m.get("title"), "page_type": m.get("kind"), "theme": m.get("theme"),
                      "transcribed": rec["transcribed"], "image": rec["img"],
                      "figures": rec["figs"],
                      "markdown": ("pages/page-%03d.md" % rec["pdf_page"]) if rec["transcribed"] else None})
    with open(os.path.join(out_dir, "book.md"), "w", encoding="utf-8") as f:
        f.write("# " + (structure.get("title") or "扫描页全文") + "\n" + "".join(raw) + "\n")
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"title": structure.get("title"), "offset": structure.get("offset") or 5,
                   "count": len(index), "transcribed": done_total,
                   "chapters": [{"id": c["id"], "title": c["title"], "file": ch_files.get(c["id"]),
                                 "pdf": c.get("pdf"), "pages": len(by_ch.get(c["id"]) or []),
                                 "transcribed": sum(1 for r in (by_ch.get(c["id"]) or []) if r["transcribed"])}
                                for c in chapters if by_ch.get(c["id"])],
                   "pages": index}, f, ensure_ascii=False, indent=2)
    log(f"[color-knowledge] 目录页 → {os.path.join(out_dir, 'index.html')}（{len(ch_files)} 章 / {total} 页 / 已转录 {done_total}）")
    for c in chapters:
        if by_ch.get(c["id"]):
            log(f"  {c['title']}：{len(by_ch[c['id']])} 页 → {ch_files.get(c['id'])}")
    log(f"[color-knowledge] 纯文本 → {os.path.join(out_dir, 'book.md')}；索引 → {os.path.join(out_dir, 'index.json')}")
    return {"chapters": len(ch_files), "pages": total, "transcribed": done_total,
            "index": os.path.join(out_dir, "index.html")}


def build_book_html(pages_dir: str, images_dir: str, out_path: str, book_title: str = "", log_callback=None) -> dict:
    """把逐页 Markdown（`<!--meta ... -->` + 正文）拼成一份可离线、可检索的单文件 HTML。

    检索能力有两层：
    1. 浏览器原生 Ctrl+F —— 所有页都渲染在同一个文档里，不折叠、不按需加载正文；
    2. 页面自带的搜索框 —— 按整页文本过滤，`/` 键快速聚焦。

    同时写 `book.md`（全文纯文本，给 grep / 工具链用）与 `index.json`（结构化页表）。
    """
    import glob as _g

    def log(msg):
        if log_callback:
            log_callback(msg)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for path in sorted(_g.glob(os.path.join(pages_dir, "page-*.json".replace(".json", ".md")))):
        meta, body = parse_page_markdown(open(path, encoding="utf-8").read())
        body = _strip_leading_h1(body)
        name = os.path.basename(path)
        num = int(re.search(r"(\d+)", name).group(1))
        img = "images/page-%03d.jpg" % num
        if not os.path.isfile(os.path.join(out_dir, img)):
            img = ""
        entries.append({"file": name, "id": "p%03d" % num, "meta": meta, "body": body,
                        "img": img, "pdf_page": meta.get("pdf_page") or num,
                        "md_name": name})
    if not entries:
        log("[color-knowledge] 没找到任何 pages/page-*.md")
        return {}

    def _sort_key(e):
        try:
            return int(e["pdf_page"])
        except (TypeError, ValueError):
            return 9999
    entries.sort(key=_sort_key)

    nav, secs, raw_all, index = [], [], [], []
    for e in entries:
        m = e["meta"]
        title = m.get("title") or m.get("work_title") or e["md_name"]
        printed = m.get("printed_page")
        who = " · ".join(x for x in [
            ("印刷 p" + str(printed)) if printed else "前置页（未印页码）",
            m.get("chapter"), m.get("page_type")] if x)
        nav.append(f'<a href="#{e["id"]}">{_inline_md(str(title))}</a>')
        pal = ""
        if m.get("palette"):
            chips = []
            for part in str(m["palette"]).split("|"):
                part = part.strip()
                mm = re.match(r"^(.*?)\s*(#[0-9a-fA-F]{6})$", part)
                if mm:
                    chips.append(f'<span><i class="sw" style="background:{mm.group(2)}"></i>'
                                 f'{_inline_md(mm.group(1).strip())} {mm.group(2)}</span>')
                elif part:
                    chips.append(f"<span>{_inline_md(part)}</span>")
            pal = '<div class="pal">' + "".join(chips) + "</div>"
        # 同一处口径：扫描原页只留一个页眉链接，不占右列。
        fig = (f'<a class="src" href="{e["img"]}" target="_blank" '
               f'title="打开 PDF 第 {e["pdf_page"]} 页的扫描原图">扫描原页 ↗</a>') if e["img"] else ""
        raw = (str(title) + " " + who + " " + e["body"]).replace('"', "&quot;")
        secs.append(
            f'<section class="page" id="{e["id"]}" data-raw="{raw}">'
            f'<div class="phead"><div class="ttl">{_inline_md(str(title))}</div>'
            f'<div class="who">{_inline_md(who)}{fig}</div></div>'
            f'<div class="grid">{_md_to_html(e["body"])}{pal}</div></section>')
        raw_all.append(f"\n\n<!-- ===== PDF p{e['pdf_page']} / 印刷 p{printed} / {title} ===== -->\n"
                       + str(title) + "\n\n" + e["body"].strip())
        index.append({"pdf_page": e["pdf_page"], "printed_page": printed, "title": title,
                      "chapter": m.get("chapter"), "page_type": m.get("page_type"),
                      "image": e["img"], "markdown": "pages/" + e["md_name"]})

    head = book_title or "《超人气配色手册》扫描页全文（图 + 文）"
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_inline_md(head)}</title>
<style>{_HTML_CSS}</style></head><body>
<div id="top"><h1>{_inline_md(head)}</h1>
<input id="q" type="search" placeholder="输入关键词过滤（按 / 快速聚焦）— 也可直接用 Ctrl+F 全局查找" autocomplete="off">
<span id="count"></span></div>
<div id="nav">{"".join(nav)}</div>
<main>{"".join(secs)}<p class="empty">— 本文件由 tools/color_knowledge.py html 从 pages/*.md 生成 —</p></main>
<script>{_HTML_JS}</script></body></html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    md_path = os.path.join(out_dir, "book.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {head}\n" + "".join(raw_all) + "\n")
    json_path = os.path.join(out_dir, "index.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"title": head, "count": len(index), "pages": index}, f, ensure_ascii=False, indent=2)
    log(f"[color-knowledge] HTML {len(index)} 页 → {out_path}")
    log(f"[color-knowledge] 纯文本 {len(index)} 页 → {md_path}（可 grep）")
    log(f"[color-knowledge] 页索引 → {json_path}")
    return {"pages": len(index), "html": out_path, "md": md_path, "index": json_path}


# ---------------------------------------------------------------- 版面切图（扫描页 → 配图）

def _ink_mask(img, paper_dist: int = 38):
    """墨色掩膜：与这一页纸色差 > `paper_dist` 的像素（不做膨胀，保留真实边界）。"""
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    paper = _paper_color(img, h // 2, band=h // 2)
    dist = np.abs(img.astype(np.int32) - paper.reshape(1, 1, 3)).max(axis=2)
    mask = (dist > paper_dist).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _gutter_runs(profile, min_gutter: int, thresh: float):
    """在一维密度曲线上找「连续空白段」，返回 [(start, end), ...]。"""
    runs, start = [], None
    for i, v in enumerate(profile):
        if v <= thresh:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_gutter:
                runs.append((start, i))
            start = None
    if start is not None and len(profile) - start >= min_gutter:
        runs.append((start, len(profile)))
    return runs


def _bands(profile, min_gutter: int, thresh: float, min_band: int):
    """按空白段把一维密度切成若干「有内容」的区间。"""
    out, cursor = [], 0
    for g0, g1 in _gutter_runs(profile, min_gutter, thresh):
        if g0 - cursor >= min_band:
            out.append((cursor, g0))
        cursor = g1
    if len(profile) - cursor >= min_band:
        out.append((cursor, len(profile)))
    return out


def _merge_stacked(boxes, max_gap: int = 22, min_overlap: float = 0.85):
    """把上下相邻、横向范围几乎一致的框合成一块。

    本书的大图表（如第 15 页那张 120 色总表）行与行之间只有十来像素的白缝，
    按白缝切会被切成 7 条横带。合并条件是 **横向范围几乎一致**，
    所以它不会把"色相平衡"和右边的"色调平衡"并到一起（两者 x 范围差得远）。
    """
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ov = min(a[2], b[2]) - max(a[0], b[0])
                if ov < min_overlap * min(a[2] - a[0], b[2] - b[0]):
                    continue
                top, bot = (a, b) if a[1] <= b[1] else (b, a)
                gap = bot[1] - top[3]
                if gap < -6 or gap > max_gap:
                    continue
                merged = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                boxes.pop(max(i, j))
                boxes.pop(min(i, j))
                boxes.append(merged)
                changed = True
                break
            if changed:
                break
    return boxes


def _panel_boxes(image_path: str, paper_dist: int = 24, min_gutter_h: int = 10, min_gutter_v: int = 12,
                 thresh: float = 0.003, min_band: int = 16):
    """用**版面上的白缝**把页面切成一块块面板，返回 (img, [(x0,y0,x1,y1), ...])。

    为什么要按白缝切、而不是按连通域切：本书的版面是"白底 + 面板"的规整网格，
    一块面板（标题 + 图 + 图注）内部元素之间只有几像素缝隙，面板之间却有大片白。
    按连通域（膨胀）切会把一块图表拆成三五块（实测第 22 页：色相平衡的柱、色调平衡的两个圆
    各自成图，标题行也被当成一块）。按白缝切才和"一块配图"对齐。

    `paper_dist` 取 24（而不是抠图用的 38）：色相色调范围表那种浅灰网格线
    与纸白只差 20 上下，用 38 会把整张表格当成空白，只剩标题行。
    """
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    mask = _ink_mask(img, paper_dist=paper_dist)
    rows = mask.mean(axis=1) / 255.0
    boxes = []
    for y0, y1 in _bands(rows, min_gutter_h, thresh, min_band):
        strip = mask[y0:y1]
        cols = strip.mean(axis=0) / 255.0
        if cols.max() <= thresh:
            continue
        for x0, x1 in _bands(cols, min_gutter_v, thresh, min_band):
            boxes.append([x0, y0, x1, y1])

    # 合并一：同一排里被小缝切开的（色相平衡的两簇柱子中间只有一条窄白缝）
    boxes = _merge_in_row(boxes, max_gap=44)
    # 合并二：上下相邻、横向范围几乎一致的小缝（大图表会被行间白缝切成多条横带）
    boxes = _merge_stacked(boxes, max_gap=22)
    # 合并三：把紧贴在上方的小高度标题行并进它下面的图（"主色调""色相平衡"这种）
    boxes = _merge_headings(boxes, mask, max_gap=30, max_head_h=90)
    return img, [tuple(b) for b in boxes]


def _merge_in_row(boxes, max_gap: int = 44):
    """纵向重叠 ≥50%、横向间隙 ≤`max_gap` 的框合成一个。"""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                oy = min(a[3], b[3]) - max(a[1], b[1])
                if oy <= 0.5 * min(a[3] - a[1], b[3] - b[1]):
                    continue
                gap = max(a[0], b[0]) - min(a[2], b[2])
                if gap > max_gap:
                    continue
                boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                boxes.pop(j)
                changed = True
                break
            if changed:
                break
    return boxes


def _merge_headings(boxes, mask, max_gap: int = 30, max_head_h: int = 90):
    """把「矮、宽、墨色稀疏」的标题行并进它正下方那块图。"""
    boxes = [list(b) for b in boxes]
    heads = [b for b in boxes if (b[3] - b[1]) <= max_head_h]
    bodies = [b for b in boxes if b not in heads]
    for h in list(heads):
        target = None
        best_gap = 10 ** 9
        for b in bodies:
            gap = b[1] - h[3]
            if gap < -8 or gap > max_gap:
                continue
            ov = min(h[2], b[2]) - max(h[0], b[0])
            if ov <= 0.35 * (h[2] - h[0]):
                continue
            if gap < best_gap:
                best_gap, target = gap, b
        if target is not None:
            target[0] = min(target[0], h[0])
            target[1] = min(target[1], h[1])
            target[2] = max(target[2], h[2])
            target[3] = max(target[3], h[3])
            heads.remove(h)
    return bodies + heads


def _max_true_run(arr) -> int:
    """一维布尔数组里最长的一段连续 True。"""
    best = cur = 0
    for v in arr:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def _panel_features(img, mask, box) -> dict:
    """面板特征：用来判断「这块是配图还是正文段落」。"""
    import cv2
    import numpy as np

    x0, y0, x1, y1 = box
    sub, sub_mask = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    ink = float((sub_mask > 0).mean())
    colorful = float((((s >= 40) & (v >= 40)) & (sub_mask > 0)).mean())
    dark = float(((v < 80) & (sub_mask > 0)).mean())
    # 「通长直线」= 整行/整列里存在一段**连续**墨迹，长度超过该边的 45%。
    # 表格规则线、坐标轴、网格线满足；正文行不满足（字与字之间有缝，最长一段就是一笔的宽度）。
    # 早期版本这里用的是"行墨密度 ≥ 0.6"，结果是紧贴一行的文字框（两端对齐、字距小）
    # 会被判成表格线，把整段正文抠成 5 张"配图"。
    bw, bh = x1 - x0, y1 - y0
    bin_mask = (sub_mask > 0)
    n_rules = 0
    if bw >= 40:
        n_rules += int(sum(1 for r in bin_mask if _max_true_run(r) >= 0.45 * bw))
    if bh >= 40:
        n_rules += int(sum(1 for c in bin_mask.T if _max_true_run(c) >= 0.45 * bh))
    n_lab, _l, st, _c = cv2.connectedComponentsWithStats(
        (cv2.morphologyEx(sub_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0).astype(np.uint8),
        connectivity=8)
    n_small = 0
    if n_lab > 1:
        areas = st[1:, cv2.CC_STAT_AREA]
        n_small = int(((areas > 4) & (areas < 0.002 * sub_mask.size)).sum())
    return {"crop": [int(x0), int(y0), int(x1), int(y1)],
            "bbox_ratio": round(((x1 - x0) * (y1 - y0)) / (w * h), 4),
            "aspect": round((x1 - x0) / max(1, y1 - y0), 3),
            "ink": round(ink, 4), "colorful": round(colorful, 4), "dark": round(dark, 4),
            "n_rules": n_rules, "n_small": n_small}


def detect_page_figures(image_path: str, min_area_ratio: float = 0.008, min_colorful: float = 0.05,
                        max_text_components: int = 60, pad: int = 8, max_figures: int = 24,
                        max_bbox_ratio: float = 0.75) -> list:
    """找出一页里的配图块（先按面积判，再按阅读顺序编号）。

    判据（满足任意一条即可，但都要先过大面积关）：
    - **有彩色**：`colorful >= min_colorful` —— 插画 / 色板 / 彩色图表；
    - **有通长直线**：`n_rules >= 2` —— 表格、坐标轴、网格（色相色调范围表靠这条救回来）；
    - **大面积深色**：`bbox_ratio >= 0.06 且 ink >= 0.5` —— 深色整页插画（本身几乎没有饱和色）。

    排除条件：小连通域个数 `n_small >= max_text_components` 且 `colorful < 0.10` ——
    正文段落会碎成上百个小块，图表和色板只有几个到十几个。
    """
    img, boxes = _panel_boxes(image_path)
    if img is None:
        return []
    mask = _ink_mask(img, paper_dist=24)
    kept = []
    for box in boxes:
        f = _panel_features(img, mask, box)
        if f["bbox_ratio"] < min_area_ratio or f["aspect"] < 0.28:
            continue
        # 整页插图不抠：它和页图就是同一张，抠出来只是多存一份（占全书约三分之一页数）。
        if f["bbox_ratio"] > max_bbox_ratio:
            continue
        # "碎成一片"只在**同时没有颜色**时才判为正文：色相色调范围表那种图
        # 由上百个色点组成，n_small 天然很高，但它明显是彩色的。
        if f["n_small"] >= max_text_components and f["colorful"] < min_colorful:
            continue
        # 矮块必须"本身就是一片颜色"：标题行、图注、表头都是宽而矮、以文字为主的块，
        # 它们的行墨迹会因为加粗/下划线凑出通长直线，或者因为扫描色噪凑出一点彩色，
        # 单靠 colorful / n_rules 挡不住。真正的矮配图（色板条、色卡行）有彩色占比很高。
        if (f["crop"][3] - f["crop"][1]) <= 110 and f["colorful"] < 0.25:
            continue
        colorful_ok = f["colorful"] >= min_colorful
        # 靠"通长直线"过关的块必须够高：页面大标题下面都有一条粗下划线，
        # 只按 n_rules 判会把「本书的阅读方法」「如何看色彩」这类标题整行抠成配图。
        # 真正的表格/坐标图不会低于 150px。
        rules_ok = f["n_rules"] >= 2 and f["ink"] >= 0.02 and (f["crop"][3] - f["crop"][1]) >= 150
        dark_ok = f["bbox_ratio"] >= 0.06 and f["ink"] >= 0.5
        if not (colorful_ok or rules_ok or dark_ok):
            continue
        kept.append(f)
    kept.sort(key=lambda f: -f["bbox_ratio"])
    kept = kept[:max_figures]
    # 按阅读顺序（先上后下、再左到右）重新编号，方便和页面对照
    kept.sort(key=lambda f: (f["crop"][1] // 60, f["crop"][0]))
    h, w = img.shape[:2]
    for f in kept:
        x0, y0, x1, y1 = f["crop"]
        f["crop"] = [max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)]
    return kept


def load_page_override(override_dir: str, pdf_page: int):
    """读人工覆盖文件 `overrides/page-NNN.json`；没有就返回 None。

    格式（crop 用**原始页图**的像素坐标）::

        {"figures": [{"id": "fig01", "crop": [60, 780, 1000, 1080], "caption": "主色调色板"}]}

    为什么要留这扇门：版面自动切分对"一块面板被白缝切成两半"这种情况无解
    （实测第 22 页的色相平衡：两簇柱子中间有 250px 白缝，和它到右边色调平衡面板的间距
    是同一个量级，任何阈值都会误伤一边）。这一类用 2 行 JSON 钉死比继续调参划算。
    """
    if not override_dir:
        return None
    path = os.path.join(override_dir, "page-%03d.json" % pdf_page)
    if not os.path.isfile(path):
        return None
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return data.get("figures") or None


def extract_page_figures(pages_dir: str, out_dir: str, only=None, min_area_ratio: float = 0.008,
                         min_colorful: float = 0.05, overwrite: bool = False,
                         overlay_dir: str = "", override_dir: str = "", log_callback=None,
                         max_figures: int = 24) -> dict:
    """把每一页的配图抠成独立 PNG，并写 `page-NNN.json`（含 bbox，供人工核对与回填说明）。

    命名 `page-NNN-figNN.png`，编号按阅读顺序（先上后下、再左到右）。
    **存 PNG、原分辨率**：全书 150 页抠下来四百多块、约 190 MB —— 这是刻意选的，
    配图是要被反复回看和二次利用的素材（色值、表格线、小字都要能放大看清），
    用 JPEG 换来的体积收益不值得赔上画质。
    `overlay_dir` 非空时同时输出一张「编号方框叠加图」，人眼扫一眼就知道抠得对不对。
    `override_dir` 下存在 `page-NNN.json` 时，**该页改用人工框**（见 `load_page_override`）。
    """
    import glob as _g

    from PIL import Image

    def log(msg):
        if log_callback:
            log_callback(msg)

    os.makedirs(out_dir, exist_ok=True)
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
    want = set(int(x) for x in (only or []))
    total = 0
    for path in sorted(_g.glob(os.path.join(pages_dir, "page-*.jpg"))):
        num = int(re.search(r"(\d+)", os.path.basename(path)).group(1))
        if want and num not in want:
            continue
        figs = load_page_override(override_dir, num)
        source = "override" if figs else "auto"
        if not figs:
            figs = detect_page_figures(path, min_area_ratio=min_area_ratio,
                                       min_colorful=min_colorful, max_figures=max_figures)
        im = Image.open(path).convert("RGB")
        for i, f in enumerate(figs, 1):
            f.setdefault("id", "fig%02d" % i)
            f["file"] = "page-%03d-%s.png" % (num, f["id"])
            dest = os.path.join(out_dir, f["file"])
            if overwrite or not os.path.isfile(dest):
                im.crop(tuple(f["crop"])).save(dest, optimize=True)
        with open(os.path.join(out_dir, "page-%03d.json" % num), "w", encoding="utf-8") as fh:
            json.dump({"pdf_page": num, "source": os.path.basename(path), "origin": source,
                       "figures": figs}, fh, ensure_ascii=False, indent=1)
        total += len(figs)
        log(f"[color-knowledge] p{num}：{len(figs)} 块配图（{source}）")
        if overlay_dir:
            _draw_overlay(path, figs, os.path.join(overlay_dir, "overlay-page-%03d.png" % num))
    log(f"[color-knowledge] 抠图完成：{total} 块 → {out_dir}")
    return {"total": total, "out_dir": out_dir}


def _draw_overlay(page_path: str, figs: list, out_path: str):
    """在缩小的页图上画出每个配图块的编号与范围，用于人工核对。"""
    from PIL import Image, ImageDraw

    im = Image.open(page_path).convert("RGB")
    w, h = im.size
    im.thumbnail((1100, 1700))
    sx, sy = im.size[0] / w, im.size[1] / h
    d = ImageDraw.Draw(im)
    font = _cjk_font(28)
    for i, f in enumerate(figs, 1):
        x0, y0, x1, y1 = f["crop"]
        d.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=(220, 0, 0), width=4)
        d.rectangle([x0 * sx, y0 * sy, x0 * sx + 62, y0 * sy + 40], fill=(220, 0, 0))
        d.text((x0 * sx + 8, y0 * sy + 3), f["id"], fill="white", font=font)
    im.save(out_path)


def audit_book_site(root: str, short_chars: int = 700, log_callback=None) -> dict:
    """抽检整站：断链图片 / 缺标题 / 正文体量 / 色名与 130 色表一致性。

    **每转录 10 页跑一次**。这几个检查都是"抄错/挂错"的高发点：
    图被删了但 md 还引用、标题兜底规则挂错作品、色名写成了表里没有的（要么是固有色名，要么是抄错）。
    返回的 `problems` 里每条都带页码，便于回改。
    """
    import glob as _g

    def log(msg):
        if log_callback:
            log_callback(msg)

    root = os.path.abspath(root)
    problems = []
    idx_path = os.path.join(root, "index.json")
    if not os.path.isfile(idx_path):
        return {"error": "index.json 不存在，先跑 html"}
    idx = json.load(open(idx_path, encoding="utf-8"))

    # 1. 断链图片
    for f in _g.glob(os.path.join(root, "chapter-*.html")) + [os.path.join(root, "index.html")]:
        if not os.path.isfile(f):
            continue
        for x in set(re.findall(r'src="((?:figures|images)/[^"]+)"', open(f, encoding="utf-8").read())):
            if not os.path.isfile(os.path.join(root, x)):
                problems.append({"kind": "断链图片", "page": os.path.basename(f), "detail": x})

    # 2. 缺标题
    for p in idx["pages"]:
        if not p.get("title"):
            problems.append({"kind": "缺标题", "page": p["pdf_page"], "detail": ""})

    # 3. 130 色表一致性（色名 → 是否在表内）
    pal_path = os.path.join(BASE_DIR, "data", "color-knowledge", "ncd-palette-130.json")
    known = set()
    if os.path.isfile(pal_path):
        pal = json.load(open(pal_path, encoding="utf-8"))
        known = {c["name"] for c in pal["colors"]} | {n["name"] for n in pal["neutrals"]}
    from_palette = []
    done = [p for p in idx["pages"] if p.get("transcribed")]
    for p in done:
        md_path = os.path.join(root, p["markdown"] or "")
        if not p.get("markdown") or not os.path.isfile(md_path):
            continue
        text = open(md_path, encoding="utf-8").read()
        # 体量检查要在 palette 判断**之前**做：没有 palette 的页（笔记 / 理论 / 章节封面）
        # 同样需要看体量，早先把这一条写在 palette 分支里，结果整类页面被跳过、永远查不出来。
        body = text.split("-->", 1)[-1]
        if len(body) < short_chars:
            problems.append({"kind": "正文偏短", "page": p["pdf_page"],
                             "detail": "%d 字 / %s" % (len(body), p.get("page_type") or "")})
        if "palette:" not in text:
            continue
        raw = text.split("palette:", 1)[1].split("\n")[0]
        for part in raw.split("|"):
            nm = re.sub(r"\s*#[0-9a-fA-F]{6}$", "", part.strip()).strip()
            if not nm:
                continue
            if known and nm not in known:
                from_palette.append({"pdf_page": p["pdf_page"], "name": nm})
    log("[audit] 已转录 %d / %d 页" % (idx["transcribed"], idx["count"]))
    for pr in problems:
        log("  [!] %-8s p%s  %s" % (pr["kind"], pr["page"], pr["detail"]))
    if not problems:
        log("  [ok] 断链 / 缺标题 / 体量 均无问题")
    log("[audit] 色名不在 130 色表内的共 %d 个（多数是固有色名，属预期）：" % len(from_palette))
    log("       " + "、".join("%s(p%s)" % (x["name"], x["pdf_page"]) for x in from_palette))
    return {"transcribed": idx["transcribed"], "total": idx["count"],
            "problems": problems, "names_not_in_table": from_palette}


def build_knowledge_base(pages_dir: str, out_dir: str, log_callback=None) -> dict:
    """把逐页 JSON 合成四份可被下游消费的知识文件。

    - `color_theory.json`    规则表（concept / prompt_keywords / post_process …），按 concept 去重
    - `palette_library.json` 案例配色库（一篇作品一条：主色调 + 色相/色调平衡 + 规则）
    - `lighting_rules.json`  从规则里筛出光照/氛围条目
    - `knowledge_index.json` 页码索引与工序统计（给下游查证用）
    """
    import glob as _g

    def log(msg):
        if log_callback:
            log_callback(msg)

    page_files = sorted(_g.glob(os.path.join(pages_dir, "page-*.json")))
    rules, palettes, lighting, index = [], [], [], []
    for path in page_files:
        try:
            p = json.load(open(path, encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log(f"[color-knowledge] 跳过坏 JSON {path}：{exc}")
            continue
        index.append({"pdf_page": p.get("pdf_page"), "printed_page": p.get("printed_page"),
                      "page_type": p.get("page_type"),
                      "title": p.get("work_title") or p.get("title") or p.get("story_theme"),
                      "chapter": p.get("chapter")})
        work_title = p.get("work_title") or p.get("title")
        for r in (p.get("rules") or []):
            if not isinstance(r, dict):
                continue
            r = dict(r)
            r["source"] = {"pdf_page": p.get("pdf_page"), "printed_page": p.get("printed_page"),
                           "work_title": work_title, "chapter": p.get("chapter"),
                           "page_type": p.get("page_type")}
            rules.append(r)
            if str(r.get("applies_to") or "").lower() in ("lighting", "mood", "atmosphere"):
                lighting.append(r)
        if p.get("page_type") in ("case_analysis", "case_highlight", "case_process", "note") and \
                (p.get("main_palette") or p.get("page_type") == "note"):
            palettes.append({
                "name": work_title,
                "artist": p.get("artist"), "chapter": p.get("chapter"),
                "story_theme": p.get("story_theme"), "page_type": p.get("page_type"),
                "printed_page": p.get("printed_page"), "pdf_page": p.get("pdf_page"),
                "main_palette": p.get("main_palette") or [],
                "hue_tone_range": p.get("hue_tone_range"), "hue_balance": p.get("hue_balance"),
                "tone_balance": p.get("tone_balance"),
                "color_relations": p.get("color_relations") or [],
                "highlight_points": p.get("highlight_points") or [],
                "process_points": p.get("process_points") or [],
                "note_points": p.get("note_points") or [],
            })
    seen, uniq = set(), []
    for r in rules:
        k = _rule_key(r)
        if k and k not in seen:
            seen.add(k)
            uniq.append(r)
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for name, payload in (("color_theory.json", {"rules": uniq, "count": len(uniq)}),
                          ("palette_library.json", {"palettes": palettes, "count": len(palettes)}),
                          ("lighting_rules.json", {"rules": lighting, "count": len(lighting)}),
                          ("knowledge_index.json", {"pages": index, "count": len(index)})):
        dest = os.path.join(out_dir, name)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        written[name] = dest
    log(f"[color-knowledge] 合成完成：规则 {len(uniq)}（去重自 {len(rules)}）、"
        f"光照 {len(lighting)}、配色库 {len(palettes)}、页索引 {len(index)} → {out_dir}")
    return written
