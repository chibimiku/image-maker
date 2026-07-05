请分析以下英文图片描述，移除其中所有明确要求照片类、写实类风格的词语和短语，但保持原文其他内容不变。

需要移除的关键词包括但不限于：
- realistic, realism, photo-realistic, photorealistic
- photo, photograph, photography, photo-like, photoshoot
- real, actual, lifelike, naturalistic
- high resolution, 8k, 4k, ultra detailed (仅当用于描述照片真实感时)
- cinematic lighting (仅当用于照片风格时)
- DSLR, camera, shutter, aperture
- RAW photo, professional photo

处理规则：
1. 只移除与照片风格直接相关的词语，保持人物描述、动作、服装、场景等内容不变。
2. 如果某个词语同时有其他含义且不影响风格，可以保留（例如 "natural" 在描述自然场景时可保留）。
3. 保持英文描述的流畅性和连贯性。
4. 输出的 JSON 结构必须与输入完全一致，修改 english_description、original_english_description 和 short_description 字段。
5. 同时清理 booru-tags 中的照片风格相关标签（如 realistic, photo, photography 等）。

输入数据：
{input_json}

预期输出 JSON 结构（与输入相同，仅修改内容）：
{{
    "english_description": "<移除照片风格词后的英文描述>",
    "original_english_description": "<移除照片风格词后的原始英文描述>",
    "short_description": "<移除照片风格词后的英文简述>",
    "booru-tags": ["<清理后的标签数组>"],
    "japanese_title": "<保留不变>",
    "chinese_title": "<保留不变>",
    "pixiv_tags": ["<保留不变>"],
    "aspect_ratio": "<保留不变>"
}}