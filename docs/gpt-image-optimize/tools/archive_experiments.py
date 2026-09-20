# -*- coding: utf-8 -*-
"""归档四批实验：把**全分辨率原图**迁进 docs，并改写对比页引用、刷新 manifest。

为什么必须迁进来：`data/` 在 `.gitignore` 里，实验产物留在那儿等于克隆后全丢。

产物结构（全部原始分辨率，不压缩、不转码）：
    experiments/img/batch-A/  批次 A：三模型 + 超长版对照
    experiments/img/batch-B/  批次 B：前置画风变体 + 复跑
    experiments/img/batch-C/  批次 C：重绘（修手 / 连通发丝）+ A1/A2
    experiments/img/batch-D/  批次 D：蕾丝保真
    experiments/img/evidence/ 手/发丝/蕾丝放大证据、最早的重绘产物

关键坑：报告页 payload 里的图片键是**页内命名**（`v3_source.png` / `b0_baseline.png` / `src.png`…），
不是原始产物文件名；改写时必须按页内键映射（见 PAGE_MAP），否则整页图会被兜底成占位图。

用法：python docs/gpt-image-optimize/tools/archive_experiments.py
"""
import hashlib
import json
import os
import re
import shutil

import cv2
import numpy as np

D19 = r"D:\code\image-maker\data\20260919"
D20 = r"D:\code\image-maker\data\20260920"
DST = r"D:\code\image-maker\docs\gpt-image-optimize\experiments"
IMG = os.path.join(DST, "img")

V3 = os.path.join(D19, "anime_lolita_cmp_v3", "gpt-image-2_output_193138_0_d09c70.png")
C = os.path.join(D20, "gemini_repaint_C")
DD = os.path.join(D20, "gemini_repaint_D")
L = os.path.join(D20, "gemini_repaint_L")
PB = os.path.join(D19, "anime_lolita_prefixB")
PB2 = os.path.join(D19, "anime_lolita_prefixB_rep2")

# 归档相对路径（img/ 下） -> 原始文件全路径
MIGRATE = {
    # 批次 A
    "batch-A/01-source-v3-gptimage2.png": V3,
    "batch-A/02-gptimage2.png": V3,
    "batch-A/03-gptimage2.5-flare.png": os.path.join(D19, "anime_lolita_cmp_v3", "gpt-image-2.5-flare_output_193215_0_75fc03.png"),
    "batch-A/04-gptimage2.5-sunburst.png": os.path.join(D19, "anime_lolita_cmp_v3", "gpt-image-2.5-sunburst_output_193255_0_a26a6e.png"),
    "batch-A/05-gptimage2-long-prompt.png": os.path.join(D19, "anime_lolita_cmp_v2", "gpt-image-2_output_192922_0_b000d3.png"),
    "batch-A/06-sunburst-long-prompt.png": os.path.join(D19, "anime_lolita_cmp_v2", "gpt-image-2.5-sunburst_output_192939_0_ecad6e.png"),
    # 批次 B
    "batch-B/B0-baseline.png": os.path.join(PB, "B0_baseline_output_233314_0_4c9eae.png"),
    "batch-B/B0-baseline-r2.png": os.path.join(PB2, "B0_baseline_r2_output_233840_0_4804bb.png"),
    "batch-B/B1-cel.png": os.path.join(PB, "B1_cel_output_233355_0_aa4b52.png"),
    "batch-B/B2-vector.png": os.path.join(PB, "B2_vector_output_233435_0_e556ba.png"),
    "batch-B/B2-vector-r2.png": os.path.join(PB2, "B2_vector_r2_output_233923_0_2d8ea1.png"),
    "batch-B/B3-animator.png": os.path.join(PB, "B3_animator_output_233521_0_5899b0.png"),
    "batch-B/B4-metric.png": os.path.join(PB, "B4_metric_output_233607_0_6f8173.png"),
    "batch-B/B4-metric-r2.png": os.path.join(PB2, "B4_metric_r2_output_234004_0_0dbc86.png"),
    "batch-B/B5-both.png": os.path.join(PB, "B5_both_output_233651_0_7e699f.png"),
    "batch-B/B5-both-r2.png": os.path.join(PB2, "B5_both_r2_output_234043_0_afc6ca.png"),
    # 批次 C（含 ab_report「A · 重绘 prompt 对比」用的两张）
    "batch-C/A1-style-run1.jpg": os.path.join(D19, "gemini_repaint_A", "v3src_A1_style_233348-3245a2.jpg"),
    "batch-C/A2-lines-only.png": os.path.join(D19, "gemini_repaint_A", "v3src_A2_lines_233442-f7d041.png"),
    "batch-C/01-source.png": V3,
    "batch-C/A1-no-repair.png": os.path.join(C, "C_A1_style_000222-9edc7f.png"),
    "batch-C/C1-long.png": os.path.join(C, "C_C1_hands_hair_000310-87d4b1.png"),
    "batch-C/C2-terse.png": os.path.join(C, "C_C2_hands_hair_terse_000311-05666e.png"),
    "batch-C/C3-steps.png": os.path.join(C, "C_C3_steps_000415-e06da3.png"),
    "batch-C/C4-negative.png": os.path.join(C, "C_C4_negative_000228-ad6c24.png"),
    "batch-C/C4-negative-r2.png": os.path.join(DD, "D_C4_negative_r2_000701-a8a7b7.png"),
    "batch-C/C5-merged.png": os.path.join(DD, "D_C5_merged_r2_000759-352557.png"),
    "batch-C/C6-stylefirst.jpg": os.path.join(DD, "D_C6_stylefirst_000641-655f76.jpg"),
    "batch-C/C7-final.png": os.path.join(DD, "D_C7_c4refined_000733-e24609.png"),
    # 批次 D
    "batch-D/L1-structure.png": os.path.join(L, "L_L1_lace_structure_002422-aec16d.png"),
    "batch-D/L2-rules.png": os.path.join(L, "L_L2_lace_rules_002522-bde650.png"),
    "batch-D/L3-short.png": os.path.join(L, "L_L3_lace_short_002430-e3d631.png"),
    "batch-D/L4-priority.png": os.path.join(L, "L_L4_lace_priority_002523-f7dd0f.png"),
    "batch-D/L5-final-firmware.png": os.path.join(L, "L_L5_lace_conservative_r2_002425-ffa18a.png"),
    "batch-D/C7-rerun.png": os.path.join(L, "L_C7_c4refined_r2_002526-c8aaa3.png"),
    # 证据图
    "evidence/hands-L5-vs-source-3x.jpg": os.path.join(DD, "inspect_D", "inspect_handL2.jpg"),
    "evidence/hands-face-hand-3x.jpg": os.path.join(DD, "inspect_D", "inspect_handR.jpg"),
    "evidence/lace-L5-vs-C7-3.4x.jpg": os.path.join(L, "inspect_L", "inspect_lace.jpg"),
    "evidence/hem-frill-rows-2.2x.jpg": os.path.join(DD, "inspect_lace", "inspect_hem.jpg"),
    "evidence/chest-lace-2.2x.jpg": os.path.join(DD, "inspect_lace", "inspect_chest.jpg"),
    "evidence/hair-right-2.2x.jpg": os.path.join(L, "inspect_L", "inspect_hairr.jpg"),
    "evidence/repaint-first-test-v3src.jpg": os.path.join(D19, "gemini_repaint", "gemini3pro_repaint_232331-cfe5ca.jpg"),
    "evidence/repaint-first-test-v2src.png": os.path.join(D19, "gemini_repaint", "gemini3pro_repaint_v2src_232441-415457.png"),
    "evidence/repaint-before-after-v2src.jpg": os.path.join(D19, "gemini_repaint", "compare_v2src", "side_by_side.jpg"),
    "evidence/repaint-lace-v2src.jpg": os.path.join(D19, "gemini_repaint", "compare_v2src", "crop_lace_2x.jpg"),
    "evidence/repaint-A1-lace-vs-src.jpg": os.path.join(D19, "gemini_repaint_A", "cmp_A1", "crop_lace_2x.jpg"),
    "evidence/repaint-A2-lace-vs-src.jpg": os.path.join(D19, "gemini_repaint_A", "cmp_A2", "crop_lace_2x.jpg"),
}

# 报告页「页内键 -> 归档相对路径」
PAGE_MAP = {
    "report-A": {
        "v3_source.png": "batch-A/01-source-v3-gptimage2.png",
        "v3_image2.png": "batch-A/02-gptimage2.png",
        "v3_flare.png": "batch-A/03-gptimage2.5-flare.png",
        "v3_sunburst.png": "batch-A/04-gptimage2.5-sunburst.png",
        "v2_image2.png": "batch-A/05-gptimage2-long-prompt.png",
        "v2_sunburst.png": "batch-A/06-sunburst-long-prompt.png",
    },
    "report-A-B": {
        "v3_source.png": "batch-A/01-source-v3-gptimage2.png",
        "a1_style.jpg": "batch-C/A1-style-run1.jpg",
        "a2_lines.png": "batch-C/A2-lines-only.png",
        "b0_baseline.png": "batch-B/B0-baseline.png",
        "b0_baseline_r2.png": "batch-B/B0-baseline-r2.png",
        "b1_cel.png": "batch-B/B1-cel.png",
        "b3_animator.png": "batch-B/B3-animator.png",
        "b2_vector.png": "batch-B/B2-vector.png",
        "b2_vector_r2.png": "batch-B/B2-vector-r2.png",
        "b4_metric.png": "batch-B/B4-metric.png",
        "b4_metric_r2.png": "batch-B/B4-metric-r2.png",
        "b5_both.png": "batch-B/B5-both.png",
        "b5_both_r2.png": "batch-B/B5-both-r2.png",
    },
    "report-C": {
        "src.png": "batch-C/01-source.png",
        "c4.png": "batch-C/C4-negative.png",
        "c5.png": "batch-C/C5-merged.png",
        "c7.png": "batch-C/C7-final.png",
        "a1.png": "batch-C/A1-no-repair.png",
        "c1.png": "batch-C/C1-long.png",
        "c2.png": "batch-C/C2-terse.png",
        "c3.png": "batch-C/C3-steps.png",
        "c4_r2.png": "batch-C/C4-negative-r2.png",
        "c6.png": "batch-C/C6-stylefirst.jpg",
    },
    "report-D": {
        "src.png": "batch-C/01-source.png",
        "c7.png": "batch-C/C7-final.png",
        "c7_r2.png": "batch-D/C7-rerun.png",
        "l1.png": "batch-D/L1-structure.png",
        "l2.png": "batch-D/L2-rules.png",
        "l3.png": "batch-D/L3-short.png",
        "l4.png": "batch-D/L4-priority.png",
        "l5.png": "batch-D/L5-final-firmware.png",
    },
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def image_size(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else [int(img.shape[1]), int(img.shape[0])]


def main():
    if os.path.isdir(IMG):
        shutil.rmtree(IMG)
    os.makedirs(IMG, exist_ok=True)

    print("== 1. 迁移全分辨率原图 ==")
    records = []
    for rel, src in MIGRATE.items():
        if not os.path.isfile(src):
            print(f"   [缺失] {rel} <- {src}")
            continue
        dst = os.path.join(IMG, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        records.append({
            "archived": f"img/{rel}",
            "size": image_size(dst),
            "archived_kb": os.path.getsize(dst) // 1024,
            "source": os.path.relpath(src, r"D:\code\image-maker").replace("\\", "/"),
            "source_sha256": sha256(src),
        })
        print(f"   {rel:46s} {records[-1]['archived_kb']:5d} KB  {records[-1]['size']}")
    print(f"   共 {len(records)} 个文件，{sum(r['archived_kb'] for r in records) // 1024} MB")

    print("== 2. 改写页面引用（页内键 -> 归档路径）==")
    for page, mapping in PAGE_MAP.items():
        path = os.path.join(DST, page, "index.html")
        if not os.path.isfile(path):
            print(f"   [跳过] 缺页面 {page}")
            continue
        text = open(path, encoding="utf-8").read()
        hits = 0
        for key, rel in mapping.items():
            for old in (f"../img/{key}", f"img/{key}"):
                if old in text:
                    text = text.replace(old, f"../img/{rel}")
                    hits += 1
        leftover = sorted(set(re.findall(r"(?<![\w./])img/([A-Za-z0-9_.\-]+)", text)))
        if leftover:
            for miss in leftover:
                text = text.replace(f"img/{miss}", f"../img/evidence/{miss}")
        open(path, "w", encoding="utf-8").write(text)
        print(f"   {page}: 改写 {hits} 处"
              f"{'，未映射（兜到 evidence/）: ' + ', '.join(leftover) if leftover else '（全部命中）'}")

    print("== 3. 校验 ==")
    ok = True
    for page in PAGE_MAP:
        path = os.path.join(DST, page, "index.html")
        text = open(path, encoding="utf-8").read()
        refs = sorted(set(re.findall(r"\.\./img/([A-Za-z0-9_./\-]+)", text)))
        missing = [r for r in refs if not os.path.isfile(os.path.join(DST, "img", r.replace("/", os.sep)))]
        print(f"   {page}: 引用 {len(refs)} 张，缺 {len(missing)}")
        for m in missing:
            ok = False
            print(f"      [缺] {m}")
    print("   校验:", "全部命中" if ok else "存在缺失")

    manifest = {
        "note": "四批实验的**全分辨率原图**归档（未压缩未转码，PNG/JPG 原样）。"
                "页面图在 img/batch-A|B|C|D/，证据图在 img/evidence/。"
                "source 为原始存放路径（data/ 已被 gitignore），source_sha256 可与原图核对。",
        "batches": {
            "A": {"doc": "experiments-A-gpt-image-vs-25.md", "report": "report-A/index.html", "images": "img/batch-A/"},
            "B": {"doc": "experiments-B-style-prefix.md", "report": "report-A-B/index.html", "images": "img/batch-B/ 与 img/batch-C/A1,A2"},
            "C": {"doc": "experiments-C-gemini-repaint.md", "report": "report-C/index.html", "images": "img/batch-C/"},
            "D": {"doc": "experiments-D-lace-fidelity.md", "report": "report-D/index.html", "images": "img/batch-D/"},
        },
        "images": records,
    }
    with open(os.path.join(DST, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    grand = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(DST) for x in fs)
    cnt = sum(1 for r, _, fs in os.walk(DST) for _ in fs)
    print(f"== 完成：experiments/ {cnt} 文件 / {grand // 1024 // 1024} MB ==")


if __name__ == "__main__":
    main()
