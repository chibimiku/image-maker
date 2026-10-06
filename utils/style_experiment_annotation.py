"""E4 人工标注辅助：把「怎么标、标完怎么导入」固化成可交付说明。

**本模块不生成任何人工数值**：人类的标注必须由实际人类在看图工具里完成，
写完的 JSON 由 `utils.style_regions.save_annotation` 校验并保存（hash 必须匹配），
再重新运行 `tools/style_similarity_experiment.py run --stages E4` 才会参与统计。
"""

from __future__ import annotations

import base64
import html as html_module
import json
from pathlib import Path

from utils.style_experiment_controlled import atomic_json

SCHEMA_NOTE = """
标注 JSON 必须满足 `style-regions/1`：

```json
{
  "version": "style-regions/1",
  "image_sha256": "<被测图片自身的 SHA-256>",
  "confirmed": true,
  "pose": "frontal|three_quarter|profile|unknown",
  "face_outline": [[x,y], ...],            // 可见面部轮廓，不含头发，≥3 点
  "eyes": {
    "viewer_left":  {"state": "open|closed|occluded|unknown",
                     "upper_lid": [[x,y], ...],   // ≥7 点，左→右，含共用眼角
                     "lower_lid": [[x,y], ...],
                     "iris": [[x,y], ...],        // 可见虹膜部分的闭合多边形
                     "lashes": [[[x,y], ...], ...]},
    "viewer_right": { ... }
  },
  "hair_regions": [[[x,y], ...], ...],     // 必须排除脸、衣服、背景与外轮廓
  "hair_strands": [{"points": [[x,y], ...], "polarity": "dark|light"}, ...],
  "nose": [x,y], "mouth": [x,y]
}
```

坐标是 **EXIF 转正后**归一化到 0–1。没有 3 根可读细发丝时 `hair_strands` 留空，
测量保持 `unavailable`——**不得**用发束轮廓或衣服边缘替代细发丝。
闭眼 / 遮挡不补线。
"""


def annotation_brief(images) -> dict:
    rows = []
    for item in images:
        rows.append({"image_id": item["image_id"], "kind": item["kind"],
                     "path": item["path"], "sha256": item["sha256"],
                     "style_id": item.get("style_id"),
                     "history_failure": item.get("history_failure"),
                     "overlay_for_review": item.get("overlay")})
    return {"schema": "style-regions/1", "images": rows,
            "annotators_required": 2, "repeat_delay_hours": 24,
            "instructions": SCHEMA_NOTE,
            "forbidden": ["用模型或 agent 代填标注", "把自动候选标成 confirmed",
                          "对不可见部位补线", "用发束轮廓冒充细发丝"],
            "import_command": "python -u tools/style_metrics_verify.py --similarity <A> <B> --regions <annotation.json>",
            "note": "标注完成后重跑 E4 才会进入统计；未提供的部分保持 awaiting_human。"}


def review_sheet(images, out_path) -> str:
    """生成一张「人工核查/标注入口」的单文件 HTML：每张图 + 叠图 + 该图的 hash。"""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    parts = ["<!doctype html><meta charset='utf-8'><title>E4 人工标注入口</title>",
             "<style>body{font-family:system-ui;margin:24px;max-width:1400px}"
             "figure{display:inline-block;margin:10px;vertical-align:top}"
             "img{max-height:520px;border:1px solid #ccc}"
             "code{background:#f6f6f6;padding:2px 4px;overflow-wrap:anywhere}</style>",
             "<h1>E4 人工标注 / 核查入口</h1>",
             "<p>本页只用于<strong>人类</strong>看图：先在叠图上核查自动候选哪里偏了，"
             "再按 schema 写标注 JSON。自动候选永远是 <code>confirmed=false</code>。</p>",
             "<pre>" + html_module.escape(SCHEMA_NOTE) + "</pre>"]
    for item in images:
        for label, path in (("原图", item["path"]), ("自动候选叠图", item.get("overlay"))):
            if not path or not Path(path).is_file():
                continue
            data = Path(path).read_bytes()
            if len(data) > 8_000_000:
                continue
            mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
            uri = f"data:{mime};base64," + base64.b64encode(data).decode()
            parts.append(f"<figure><figcaption>{html_module.escape(item['image_id'])} · {label}</figcaption>"
                         f"<img src='{uri}'></figure>")
        parts.append(f"<p><code>{html_module.escape(item['image_id'])}</code><br>"
                     f"path: <code>{html_module.escape(item['path'])}</code><br>"
                     f"sha256: <code>{html_module.escape(item['sha256'])}</code></p>")
    target.write_text("\n".join(parts), encoding="utf-8")
    return str(target)


def write_annotation_assets(images, run_dir) -> dict:
    run_dir = Path(run_dir)
    brief = annotation_brief(images)
    atomic_json(run_dir / "E4" / "annotation-brief.json", brief)
    sheet = review_sheet(images, run_dir / "E4" / "human-annotation-entry.html")
    atomic_json(run_dir / "E4" / "annotation-status.json",
                {"status": "awaiting_human", "H1": None, "H2": None, "H1_repeat_after_24h": None,
                 "images": [item["image_id"] for item in images],
                 "note": "没有任何人工标注；不得用模型输出填充。"})
    return {"brief": str(run_dir / "E4" / "annotation-brief.json"), "sheet": sheet,
            "status": str(run_dir / "E4" / "annotation-status.json")}
