AUTHORIZED CLOTHING SCOPE:
{wardrobe}
Extract protected character anchors from the actual prompt, but do not treat the old source outfit's category/cut as immutable when the policy authorizes redesign. Judge clothing against this target and any explicit keep-original/colour/accessory conditions. Other identity facts remain fully protected. The mere authorized wardrobe change is not an identity mismatch. Do not infer age from the clothing style.


BILINGUAL DISPLAY OUTPUT: Keep all existing schema keys, enum values and audit findings in English. In the SAME JSON object also return "display_zh": {"summary": "简体中文结论摘要", "details": "简体中文完整展示说明"}. Translate the actual conclusion, each finding and its region, evidence, expected/observed differences, repair advice, accepted variations, limitations and uncertainty into Simplified Chinese. Preserve ordering, confidence values and meaning; do not add or remove findings. This display-only object must never override the English decision fields. Return both languages in one response, with no separate translation request.
