# -*- coding: utf-8 -*-
"""从 tid.png 生成「无主体风格裁剪」与「风格板」，供参考图用法实验使用。"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(BASE, "data", "style-ref", "tid.png")
OUT = os.path.join(BASE, "data", "gpt-image-tid", "style-crops")
os.makedirs(OUT, exist_ok=True)

try:
    from PIL import Image
except Exception as e:  # noqa: BLE001
    print("PIL 不可用:", e)
    sys.exit(1)

im = Image.open(REF).convert("RGB")
W, H = im.size
print("参考图", im.size)

# 主体（少女）大致集中在画面中部偏右；取几块远离主体的区域
crops = {
    # 左上纯白 + 少量草叶/气泡：最干净的「留白 + 色板」
    "crop_white": (0, 0, int(W * 0.34), int(H * 0.30)),
    # 左下：水草 + 蝴蝶 + 气泡 + 白底（有笔触、无人物）
    "crop_left": (0, int(H * 0.30), int(W * 0.32), int(H * 0.95)),
    # 底部：水波纹 + 鞋/倒影之外的水面（笔触最明显）
    "crop_bottom": (0, int(H * 0.72), W, H),
    # 右上：发尾之外的白色空间 + 少量草
    "crop_topright": (int(W * 0.62), 0, W, int(H * 0.28)),
}
paths = {}
for name, box in crops.items():
    c = im.crop(box)
    p = os.path.join(OUT, f"{name}.png")
    c.save(p)
    paths[name] = p
    print(f"{name:14s} {c.size}  {p}")

# 风格板：三块**确认无人物**的区域横排 + 白色间隔（外框留 24px 白边）
pad = 24
boxes = [
    (0, 0, int(W * 0.30), int(H * 0.62)),          # 左上：留白 + 气泡 + 草叶
    (0, int(H * 0.55), int(W * 0.30), H),          # 左下：水波纹 + 气泡 + 鞋边（无人物）
    (int(W * 0.72), 0, W, int(H * 0.10)),          # 右上角落横条：纯白 + 水滴（避开头发）
]
cols = [im.crop(b) for b in boxes]
ch = max(c.height for c in cols)
scaled = []
for c in cols:
    ratio = ch / c.height
    scaled.append(c.resize((max(1, int(c.width * ratio)), ch)))
board_w = pad + sum(c.width + pad for c in scaled)
board = Image.new("RGB", (board_w, ch + 2 * pad), (255, 255, 255))
x = pad
for c in scaled:
    board.paste(c, (x, pad))
    x += c.width + pad
bp = os.path.join(OUT, "style_board.png")
board.save(bp)
print(f"{'style_board':14s} {board.size}  {bp}  (三块无人物区域)")
