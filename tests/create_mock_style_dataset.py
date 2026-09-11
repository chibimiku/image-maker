"""
生成 mock 测试数据集，用于本地测试多轮迭代画风提取功能。

产出：
  - data/mock_style_test/img_01.png ~ img_05.png  （5 张模拟同画风的测试图片）
  - data/mock_style_test/style_iter_result.json   （预置 2 轮迭代历史的 JSON，可测试导入续训）

用法：
  python create_mock_style_dataset.py
"""

import os
import json
import datetime
from PIL import Image, ImageDraw

OUT_DIR = os.path.join("data", "mock_style_test")


def create_mock_images():
    """生成 5 张模拟"同画风"的测试图片。

    共性特征（模拟同一画风）：
      - 所有图片左上角有一个相同颜色的圆（模拟共同色彩倾向）
      - 所有图片使用暖色调为主
      - 相似的构图方式
      - 第 5 张稍有变化（模拟 outlier / 边界情况）
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    specs = [
        # (文件名, 背景色, 形状, 形状颜色, 描述)
        ("img_01.png", (255, 220, 200), "circle", (180, 100, 120), "粉色圆形"),
        ("img_02.png", (255, 210, 190), "rect",   (180, 110, 130), "粉色矩形"),
        ("img_03.png", (240, 200, 180), "circle", (170, 90, 110),  "深粉圆形"),
        ("img_04.png", (250, 215, 195), "rect",   (185, 105, 125), "粉色矩形"),
        ("img_05.png", (200, 230, 210), "circle", (160, 130, 140), "偏绿圆形（边界情况）"),
    ]

    for idx, (fname, bg, shape, shape_color, desc) in enumerate(specs, start=1):
        img = Image.new("RGB", (512, 512), bg)
        draw = ImageDraw.Draw(img)

        # 共同元素：左上角小圆（模拟共通标志）
        draw.ellipse([20, 20, 60, 60], fill=shape_color, outline=(255, 255, 255), width=2)

        if shape == "circle":
            # 中央大圆
            draw.ellipse([156, 156, 356, 356], fill=shape_color, outline=(255, 255, 255), width=3)
        else:
            # 中央矩形
            draw.rectangle([156, 156, 356, 356], fill=shape_color, outline=(255, 255, 255), width=3)

        # 共同元素：底边装饰线（模拟共通构图习惯）
        draw.line([50, 460, 462, 460], fill=shape_color, width=4)

        # 共同元素：文字标注
        text = f"Mock #{idx}: {desc}"
        draw.text((30, 470), text, fill=(100, 100, 100))

        filepath = os.path.join(OUT_DIR, fname)
        img.save(filepath, "PNG")
        print(f"  [OK] Created: {filepath}")

    print(f"\n[OK] Generated {len(specs)} test images -> {os.path.abspath(OUT_DIR)}/")


def create_mock_json():
    """创建一个预置了 2 轮迭代历史的 JSON 文件，用于测试「导入已有 JSON 继续训练」功能。"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    image_paths = [os.path.abspath(os.path.join(OUT_DIR, f"img_{i:02d}.png")) for i in range(1, 6)]

    state = {
        "version": "1.0",
        "created_at": now,
        "updated_at": now,
        "dataset": {
            "image_count": 5,
            "images": image_paths,
        },
        "parameters": {
            "total_rounds": 3,
            "images_per_round": 2,
        },
        "iterations": [
            {
                "step": 1,
                "type": "commonality_extraction",
                "round": 1,
                "timestamp": now,
                "art_style_prompts": (
                    "You are an advanced generative AI model specialized in creating images "
                    "in the following art style. Adhere strictly to these stylistic rules:\n\n"
                    "1. Overall Artistic Vibe: Soft, warm-toned illustration with a gentle, "
                    "dreamy atmosphere. The style leans toward semi-realistic digital painting "
                    "with smooth gradients and soft focus.\n"
                    "2. Color Palette: Dominantly warm peach and rose tones (R: 240-255, G: 200-220, B: 180-210). "
                    "Low to medium saturation, giving a pastel-like quality. Gentle tonal transitions.\n"
                    "3. Lighting: Soft diffuse lighting from upper-left, creating subtle shadows "
                    "with feathered edges. No harsh highlights — everything is blended smoothly.\n"
                    "4. Line Art: Minimal visible line art — shapes are defined by color blocks "
                    "and soft edges rather than outlines. Painterly approach.\n"
                    "5. Composition: Central subject placement with decorative framing elements "
                    "along the bottom edge. Simple, uncluttered backgrounds that keep focus on the main subject.\n"
                    "6. Texture: Smooth, airbrushed finish with subtle grain texture overlay. "
                    "Matte surface quality without glossy reflections."
                ),
                "model_used": "mock-model-v1",
            },
            {
                "step": 2,
                "type": "refinement_check",
                "round": 1,
                "check_index": 1,
                "timestamp": now,
                "image": image_paths[1],
                "prompts_before": (
                    "You are an advanced generative AI model specialized in creating images "
                    "in the following art style. Adhere strictly to these stylistic rules:\n\n"
                    "1. Overall Artistic Vibe: Soft, warm-toned illustration with a gentle, "
                    "dreamy atmosphere…"
                ),
                "differences_analysis": (
                    "The image img_02.png uses cooler pink tones than described. The shape is "
                    "rectangular rather than circular, suggesting more geometric variety in the dataset. "
                    "The decorative line at the bottom is thinner than typical."
                ),
                "prompts_after": (
                    "You are an advanced generative AI model specialized in creating images "
                    "in the following art style. Adhere strictly to these stylistic rules:\n\n"
                    "1. Overall Artistic Vibe: Soft, warm-toned illustration with a gentle, "
                    "dreamy atmosphere. The style leans toward semi-realistic digital painting "
                    "with smooth gradients and soft focus. Subject shapes vary between organic "
                    "(circular) and geometric (rectangular) forms, all rendered with the same "
                    "soft treatment.\n"
                    "2. Color Palette: Dominantly warm rose and coral tones (R: 240-255, G: 200-220, B: 180-210) "
                    "with slight variation toward cooler pinks in some pieces. Low to medium saturation.\n"
                    "3. Lighting: Soft diffuse lighting from upper-left, creating subtle shadows "
                    "with feathered edges. No harsh highlights.\n"
                    "4. Line Art: Minimal visible line art — shapes defined by color blocks "
                    "and thin white outlines (2-3px) for emphasis.\n"
                    "5. Composition: Central subject placement with decorative framing elements "
                    "along the bottom edge (line motif). Clean, uncluttered backgrounds.\n"
                    "6. Texture: Smooth, airbrushed finish. Matte surface quality."
                ),
                "confidence": 0.78,
                "model_used": "mock-model-v1",
            },
            {
                "step": 3,
                "type": "refinement_check",
                "round": 1,
                "check_index": 2,
                "timestamp": now,
                "image": image_paths[3],
                "prompts_before": (
                    "You are an advanced generative AI model…\n"
                    "[Same as prompts_after from step 2]"
                ),
                "differences_analysis": (
                    "img_04.png confirms the rectangular form variant. The color saturation is "
                    "slightly higher than described. The white outline is consistent at 3px. "
                    "Background treatment matches — simple flat fill without gradients."
                ),
                "prompts_after": (
                    "You are an advanced generative AI model specialized in creating images "
                    "in the following art style. Adhere strictly to these stylistic rules:\n\n"
                    "1. Overall Artistic Vibe: Soft, warm-toned digital illustration with gentle, "
                    "dreamy atmosphere. Semi-realistic approach with smooth gradients. "
                    "Subject forms: both organic circles and geometric rectangles, unified by "
                    "consistent rendering treatment.\n"
                    "2. Color Palette: Warm rose/coral dominant (R: 240-255, G: 200-230, B: 180-210), "
                    "medium-low saturation, pastel quality. Individual pieces may lean slightly "
                    "warmer or cooler within this range.\n"
                    "3. Lighting: Soft diffuse from upper-left, feathered shadow edges, no hard speculars.\n"
                    "4. Line Art: Painterly — minimal outlines. Thin white strokes (2-3px) for form emphasis.\n"
                    "5. Composition: Centered subject, bottom-edge decorative line motif, flat/simple background.\n"
                    "6. Texture: Smooth airbrush finish, matte surface, subtle grain overlay."
                ),
                "confidence": 0.85,
                "model_used": "mock-model-v1",
            },
            {
                "step": 4,
                "type": "commonality_extraction",
                "round": 2,
                "timestamp": now,
                "art_style_prompts": (
                    "You are an advanced generative AI model specialized in creating images "
                    "in the following art style. Adhere strictly to these stylistic rules:\n\n"
                    "1. Artistic Vibe: Warm, soft digital illustration — dreamy and gentle. "
                    "Semi-realistic painterly rendering with smooth color transitions.\n"
                    "2. Color Palette: Rose/coral dominant warm palette (pastel). "
                    "Background colors: warm beige/cream base (R: 240-255, G: 200-230, B: 180-210). "
                    "Subject colors: deeper rose/pink tones. Slight variation acceptable within "
                    "warm pastel family.\n"
                    "3. Lighting: Soft diffuse top-left lighting. Feathered shadow edges. "
                    "No hard highlights or sharp shadow boundaries.\n"
                    "4. Edge Treatment: Painterly — minimal line art. Thin white outlines (2-3px) "
                    "for form definition. Shapes defined primarily by color contrast.\n"
                    "5. Composition: Centered single subject. Bottom decorative line motif. "
                    "Flat/simple background without gradients or patterns. Clean, uncluttered.\n"
                    "6. Texture & Finish: Smooth airbrush rendering. Matte surface quality. "
                    "Subtle film grain texture overlay throughout."
                ),
                "model_used": "mock-model-v1",
            },
            {
                "step": 5,
                "type": "refinement_check",
                "round": 2,
                "check_index": 1,
                "timestamp": now,
                "image": image_paths[2],
                "prompts_before": (
                    "[Prompts from Round 2 commonality extraction]"
                ),
                "differences_analysis": (
                    "img_03.png shows deeper, slightly more saturated pink tones than the pastel "
                    "description suggests. The circle form is rendered with the same soft treatment. "
                    "Background is slightly darker cream, pushing contrast higher."
                ),
                "prompts_after": (
                    "You are an advanced generative AI model specialized in creating images "
                    "in the following art style. Adhere strictly to these stylistic rules:\n\n"
                    "1. Artistic Vibe: Warm, soft digital illustration — dreamy and gentle. "
                    "Semi-realistic painterly rendering with smooth color transitions.\n"
                    "2. Color Palette: Rose/coral dominant warm palette. Base backgrounds: "
                    "warm cream/peach (R: 240-255, G: 200-230, B: 180-210). Subject colors: "
                    "rose/pink range from pastel to moderately saturated. Natural variation "
                    "in saturation across pieces is a feature, not a bug.\n"
                    "3. Lighting: Soft diffuse top-left lighting. Feathered shadow edges.\n"
                    "4. Edge Treatment: Painterly, minimal line art. Thin white outlines (2-3px).\n"
                    "5. Composition: Centered subject, bottom decorative line, flat clean background.\n"
                    "6. Texture: Smooth airbrush, matte finish, subtle grain."
                ),
                "confidence": 0.82,
                "model_used": "mock-model-v1",
            },
            {
                "step": 6,
                "type": "refinement_check",
                "round": 2,
                "check_index": 2,
                "timestamp": now,
                "image": image_paths[4],
                "prompts_before": (
                    "[Prompts after step 5]"
                ),
                "differences_analysis": (
                    "img_05.png is the outlier — it uses a slightly green-tinted background "
                    "instead of the warm cream/peach palette. The rose subject color is still "
                    "present but the overall temperature is cooler. This may indicate the "
                    "dataset includes occasional cool variants, or this is a true outlier that "
                    "should be excluded from the 80% threshold."
                ),
                "prompts_after": (
                    "You are an advanced generative AI model specialized in creating images "
                    "in the following art style. Adhere strictly to these stylistic rules:\n\n"
                    "1. Artistic Vibe: Warm, soft digital illustration with dreamy atmosphere. "
                    "Semi-realistic painterly rendering. Occasional pieces may have cooler "
                    "accents but the dominant aesthetic is warm-toned.\n"
                    "2. Color Palette: Rose/coral dominant warm palette. Background: warm "
                    "cream/peach base (occasional cool variant acceptable as secondary motif). "
                    "Subject: rose/pink tones, pastel to moderate saturation.\n"
                    "3. Lighting: Soft diffuse top-left. Feathered shadows.\n"
                    "4. Edge Treatment: Painterly minimal line art, thin white outlines (2-3px).\n"
                    "5. Composition: Centered subject, bottom decorative line, flat background.\n"
                    "6. Texture: Smooth airbrush, matte finish, subtle grain."
                ),
                "confidence": 0.75,
                "model_used": "mock-model-v1",
            },
        ],
        "final_art_style_prompts": (
            "You are an advanced generative AI model specialized in creating images "
            "in the following art style. Adhere strictly to these stylistic rules:\n\n"
            "1. Artistic Vibe: Warm, soft digital illustration with dreamy atmosphere. "
            "Semi-realistic painterly rendering. Occasional pieces may have cooler "
            "accents but the dominant aesthetic is warm-toned.\n"
            "2. Color Palette: Rose/coral dominant warm palette. Background: warm "
            "cream/peach base (occasional cool variant acceptable as secondary motif). "
            "Subject: rose/pink tones, pastel to moderate saturation.\n"
            "3. Lighting: Soft diffuse top-left. Feathered shadows.\n"
            "4. Edge Treatment: Painterly minimal line art, thin white outlines (2-3px).\n"
            "5. Composition: Centered subject, bottom decorative line, flat background.\n"
            "6. Texture: Smooth airbrush, matte finish, subtle grain."
        ),
    }

    json_path = os.path.join(OUT_DIR, "style_iter_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Created mock JSON: {json_path}")
    print(f"   Contains {len(state['iterations'])} iteration steps (2 rounds of training)")
    print(f"   Can be used to test 'import existing JSON to continue training'")


if __name__ == "__main__":
    print("=" * 60)
    print("Mock Style Dataset Generator")
    print("=" * 60)
    create_mock_images()
    create_mock_json()
    print(f"\n[OK] Done. Dataset dir: {os.path.abspath(OUT_DIR)}/")
    print(f"\nTest steps:")
    print(f"  1. Launch app.py")
    print(f"  2. Switch to 'multi-image style extraction' tab")
    print(f"  3. Drag the 5 images from {os.path.abspath(OUT_DIR)}/ into the image list")
    print(f"  4. Or click 'import existing JSON' -> select style_iter_result.json")
    print(f"  5. Set rounds and images per round, click start")
