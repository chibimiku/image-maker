请分析以下英文图片描述，移除其中所有明确要求照片类、写实类风格的词语和短语，但保持原文其他内容不变；另外，若存在自拍挡脸（手/手机/相机/头发/口罩/道具遮住面部）的情况，请删除相关遮挡描述并替换为其他不挡脸的动作。

需要移除的关键词包括但不限于：
- realistic, realism, photo-realistic, photorealistic
- photo, photograph, photography, photo-like, photoshoot
- real, actual, lifelike, naturalistic
- high resolution, 8k, 4k, ultra detailed (仅当用于描述照片真实感时)
- cinematic lighting (仅当用于照片风格时)
- DSLR, camera, shutter, aperture
- RAW photo, professional photo

处理规则：
1. 除了下方第 6 条的「挡脸动作替换」之外，只移除与照片风格直接相关的词语，保持人物描述、动作、服装、场景等内容不变。
2. 如果某个词语同时有其他含义且不影响风格，可以保留（例如 "natural" 在描述自然场景时可保留）。
3. 保持英文描述的流畅性和连贯性。
4. 输出的 JSON 结构必须与输入完全一致，修改 english_description、original_english_description 和 short_description 字段。
5. 同时清理 booru-tags 中的照片风格相关标签（如 realistic, photo, photography 等）。
6. 自拍挡脸检测与动作替换（必须执行）：
   - 判定：描述里带有自拍（selfie、taking a selfie、holding a phone/camera toward herself）或类似手持机位的拍摄方式，并且脸部被遮挡或挡住视线焦点时，即判定为「自拍挡脸」。常见形态包括：手/手指/手掌挡脸（hand over face、hand over own mouth、fingers covering face、hand near face）、手机或相机贴在脸前挡住脸（phone covering face、camera in front of face、holding phone up to face）、头发散落遮住脸（hair covering face/eyes）、口罩或面罩遮脸（mask covering face、surgical mask）、用抱枕/玩偶/花束/书本等道具挡在脸前（holding pillow/book/bouquet over face、object covering face）。
   - 处理：删除造成挡脸的机位与道具描述（自拍、举手机/相机、手挡脸、遮挡面部的那件道具等），改为一个不遮挡面部、视线焦点回到脸上的其他动作，并让面部特征清晰可见。除该动作外，不要改动人物身份、发色瞳色、服装、场景、画幅与整体氛围。
   - 替换动作要求：与原有服装、场景和氛围保持一致，不得新增未出现过的道具或改变机位。优先从这些方向里选一个最贴合的：looking at viewer（正视镜头）、smile / gentle smile、looking away / looking up / looking down、turning head slightly、head tilted、one hand raised near shoulder 或 resting on chest/waist、standing with hands behind back、hands clasped in front of chest、arms relaxed at her sides、leaning forward slightly。
   - 表情与视线：脸部必须完整可见，视线与表情保持原有情绪（温柔、骄傲、俏皮等）不变；如果原描述因遮挡而无法判断表情，就用中性且正面的表达（例如 calm, gentle expression）。
   - 标签同步：把 booru-tags 与 pixiv_tags 中表示挡脸的标签一并替换为上述新动作对应的标签（例如把 hand_over_own_mouth、hand_over_face、phone_in_front_of_face、hair_over_eyes、mask 等替换为 looking_at_viewer、smile、head_tilt、hand_on_own_chest 等；pixiv_tags 仍须是日文标签），不保留任何遮挡面部的标签。
   - 三个描述字段（english_description、original_english_description、short_description）都要做同样的替换，并保持原有篇幅量级（不要明显变短或变长）。
   - 如果输入描述本来就没有挡脸内容，则本条不生效，其余内容一律保持不变，不要主动改动任何动作。

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