# -*- coding: utf-8 -*-
"""真实提示词验证：用项目里 4 个主题的真实主体，对比三种画风注入方式。

V1 最优解 = 678 字符短说明 + tid 参考图
V2 现状   = head（13k 全量指令 + 参考说明块）+ tid 参考图
V3 纯文本 = off 全量指令，无参考图
"""
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
CLI = os.path.join(BASE, "tools", "gpt_image2_gen.py")
REF = r"D:\code\image-maker\data\style-ref\tid.png"
OUT_SUB = "tid-verify"
LOG_DIR = os.path.join(BASE, "data", "gpt-image-tid", "logs")

STYLE_HEAD = ("Use the art style of the attached reference image (Image 1) and draw a NEW image of the scene below.\n"
              "Keep from Image 1: the gentle watercolor-like digital painting finish, delicate thin lineart, soft "
              "luminous pastel palette, high-key bright ambience, airy pale background with floating translucent "
              "droplets and bubbles, glossy detailed eyes, and the calm elegant mood.\n"
              "Change from Image 1: the character, her pose, her outfit, the camera framing and the background layout "
              "\u2014 all of these come only from the scene description. Do not copy the reference's person or composition.\n")

FULL_HEAD = open(os.path.join(BASE, "docs", "gpt-image-tid-style", "prompts", "B2_full_head.txt"),
                 encoding="utf-8").read().strip()

# 四个真实主题（取自 prompts/2girls-yuri-v2.txt、prompts/sd-batch/dance.txt /
# prompts/rococo_portrait.txt、prompts/sd-batch/lolita_lace.txt 的主体设定，重写成 gpt-image 友好的自然语言）
CASES = {
    "C1_yuri": ("Two young women stand close together in a sunlit rose garden, holding hands and looking at each "
                "other with gentle smiles, both in elegant lace dresses. Half-body, soft golden-hour light."),
    "C2_dance": ("A girl in a flowing dress dances mid-spin on an empty stage, skirt flaring outward, one arm raised, "
                 "eyes closed. Full body, dynamic motion."),
    "C3_rococo": ("A young woman in a gorgeous rococo gown with intricate lace and ribbon trims sits in a three-quarter "
                  "portrait, elegant posture, one hand holding a small bouquet. Half-body, ornate background."),
    "C4_lolita": ("A lolita fashion girl stands in a quiet street, layered lace dress with ribbon bows, holding a lace "
                  "parasol, looking at the camera with a shy smile. Waist-up."),
}

VARIANTS = [
    ("V1_short_ref", "short", True),
    ("V2_full_head", "full", True),
    ("V3_full_text", "full", False),
]

only_cases = [a for a in sys.argv[1:] if not a.startswith("--")] or None

for ckey, subject in CASES.items():
    if only_cases and ckey not in only_cases:
        continue
    for vname, flavor, use_ref in VARIANTS:
        prompt = (STYLE_HEAD + "Scene: " + subject) if flavor == "short" else (FULL_HEAD + "\n\n" + subject)
        prefix = f"tv-{ckey.split('_')[0]}-{vname.split('_')[0]}"
        cmd = [PY, "-u", CLI, "--site", "new.aigc2d", "--model", "gpt-image-2",
               "--size", "1024x1536", "--quality", "medium",
               "--output-subdir", OUT_SUB, "--prefix", prefix,
               "--timeout", "600", "--json", "--prompt", prompt]
        if use_ref:
            cmd += ["--image", REF]
        print(f"\n===== {ckey} {vname} ref={use_ref} prompt_chars={len(prompt)}", flush=True)
        t0 = time.time()
        log_path = os.path.join(LOG_DIR, f"verify_{ckey}_{vname}.log")
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
            proc = subprocess.run(cmd, cwd=BASE, stdout=lf, stderr=subprocess.STDOUT)
        lines = open(log_path, encoding="utf-8", errors="replace").read().strip().splitlines()
        print("\n".join(lines[-6:]), flush=True)
        print(f"----- EXIT={proc.returncode} ELAPSED={time.time() - t0:.1f}s", flush=True)

print("\nALL DONE")
