You will check whether the following drawing prompts contain human characters.
If there is no human character, keep both prompts unchanged.
If there are human characters, revise only the clothing-related parts so the outfit is internally coherent.
Focus on clothing first, and keep shoes, socks, gloves, headwear, jewelry, ribbons, and other accessories stylistically consistent with the clothing.
Avoid mismatched combinations such as a school uniform paired with luxury stilettos unless the whole outfit clearly supports that concept.
Keep identity, pose, scene, composition, lighting, camera language, art style, mood, and other non-clothing content as unchanged as possible.
Keep the language in English and preserve roughly the same detail level and writing style.
Do not add new characters, remove major scene elements, or introduce NSFW content.
If `Target clothing style` is empty, do not force a style override. If it is not empty and human characters exist, rewrite the clothing to match that target style while keeping the rest of the prompt as stable as possible. Adjust shoes, socks, hats, gloves, ribbons, jewelry, and other accessories when necessary so they match the target clothing style.
When modifying the description, also update the pixiv_tags to stay consistent with the changes (e.g., if hair color is changed, update the corresponding tag). Keep non-clothing related tags unchanged.
Return strict JSON with keys: has_person, modified, english_description, original_english_description, pixiv_tags, reason.

Target clothing style: {outfit_style_override}

Input JSON:
{input_json}
