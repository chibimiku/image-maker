# -*- coding: utf-8 -*-
"""Luna 建议的第三轮实验（8 张，同一主体「C1 少女站浅水」）：

R1 角色分工版模板 + 原始 tid 参考图          （对照之前最优解 G1）
R2 无主体裁剪 crop_white + 角色分工版
R3 无主体裁剪 crop_left + 角色分工版
R4 风格板 style_board + 角色分工版
R5 结构化风格字段（~250 字）+ 原始参考图
R6 结构化字段 + 具体身份排除 + 原始参考图
R7 角色分工版 + quality=high（测 high 是否压掉水彩的简化/留白）
R8 结构化字段 + 无参考图（纯文本对照）
"""
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
CLI = os.path.join(BASE, "tools", "gpt_image2_gen.py")
REF = os.path.join(BASE, "data", "style-ref", "tid.png")
CROPS = os.path.join(BASE, "data", "gpt-image-tid", "style-crops")
OUT_SUB = "tid-luna"
LOG_DIR = os.path.join(BASE, "data", "gpt-image-tid", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

SUBJECT = ("A girl stands in shallow water at dusk, waist-up, looking at the camera, "
           "wearing a white dress.")

TMPL_ROLE = """Create a new image.

Use Image 1 as a visual style reference only. Match its:
- color palette and saturation
- brightness and amount of white negative space
- brushwork and edge softness
- line quality
- lighting mood
- level of detail

Image 1 is a style reference, not a subject or composition reference. Image 1 provides visual style only.
The scene below provides all content and composition.
Create a different character and a different composition from Image 1.

Scene: {subject}
The scene description controls the character, pose, clothing, framing, perspective and composition.
The reference image controls only the visual style, color treatment, lighting quality, mark-making and
overall degree of detail."""

TMPL_ROLE_EXCLUDE = TMPL_ROLE + "\nDo not reuse Image 1's character identity, hair color, eye color, costume, or pose."

TMPL_FIELDS = """Create a new image in the visual style described below.

Palette: pale pastel, very low saturation, lots of clean white.
Lighting: bright high-key, soft even light, luminous and airy.
Brushwork: transparent watercolor-like washes with delicate fine lineart.
Edges: soft, occasionally lost.
Texture: subtle paper grain, tiny floating droplets and bubbles as light accents.
Composition density: low; generous white negative space around the subject.
Detail level: restrained; a few crisp accents (hair strands, glossy eyes) on a simplified base.
Avoid: heavy shading, thick outlines, saturated or dark color, dense background clutter.

This style description controls the rendering only. The scene below controls all content:
{subject}"""

VARIANTS = [
    ("R1_role_ref", TMPL_ROLE.format(subject=SUBJECT), [REF], "medium"),
    ("R2_role_cropwhite", TMPL_ROLE.format(subject=SUBJECT), [os.path.join(CROPS, "crop_white.png")], "medium"),
    ("R3_role_cropleft", TMPL_ROLE.format(subject=SUBJECT), [os.path.join(CROPS, "crop_left.png")], "medium"),
    ("R4_role_board", TMPL_ROLE.format(subject=SUBJECT), [os.path.join(CROPS, "style_board.png")], "medium"),
    ("R5_fields_ref", TMPL_FIELDS.format(subject=SUBJECT), [REF], "medium"),
    ("R6_fields_exclude", TMPL_FIELDS.format(subject=SUBJECT) + "\nDo not reuse Image 1's character identity, hair color, eye color, costume, or pose.", [REF], "medium"),
    ("R7_role_ref_high", TMPL_ROLE.format(subject=SUBJECT), [REF], "high"),
    ("R8_fields_noref", TMPL_FIELDS.format(subject=SUBJECT), [], "medium"),
]

only = [a for a in sys.argv[1:] if not a.startswith("--")] or None

for name, prompt, images, quality in VARIANTS:
    if only and name not in only:
        continue
    prefix = "luna-" + name.split("_")[0].lower()
    cmd = [PY, "-u", CLI, "--site", "new.aigc2d", "--model", "gpt-image-2",
           "--size", "1024x1536", "--quality", quality,
           "--output-subdir", OUT_SUB, "--prefix", prefix,
           "--timeout", "900", "--json", "--prompt", prompt]
    for p in images:
        cmd += ["--image", p]
    log_path = os.path.join(LOG_DIR, f"luna_{name}.log")
    print(f"\n===== {name}  quality={quality} prompt_chars={len(prompt)} images={len(images)}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        lf.write(f"VARIANT: {name}\nQUALITY: {quality}\nIMAGES: {images}\nPROMPT_CHARS: {len(prompt)}\n\n")
        lf.write(prompt + "\n\n---\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=BASE, stdout=lf, stderr=subprocess.STDOUT)
    lines = open(log_path, encoding="utf-8", errors="replace").read().strip().splitlines()
    print("\n".join(lines[-5:]), flush=True)
    print(f"----- EXIT={proc.returncode} ELAPSED={time.time() - t0:.1f}s", flush=True)

print("\nALL DONE")
