# Cute lingerie wardrobe preset (2026-09-26)

This is an adult wardrobe transformation preset rather than a normal art style. The 12 local training references were filtered to exclude visible nudity, sexual acts and sex toys. The runtime prompt requires an explicitly adult subject and extracts garment construction only.

## Runtime decision

- Gemini direct generation succeeds while preserving brown hair, violet eyes, the seated pose, chair and study.
- GPT-image-2 generation succeeds with coherent coordinated lingerie, stockings and a chiffon layer.
- Sending the complete reference to Gemini repaint causes severe reference-character leakage, including silver hair, red eyes, a white void and an unrelated weapon.
- A source-only repaint attempt was blocked by upstream safety policy.
- The preset therefore sets `skip_repaint=true`, `repaint_reference_mode=none`, `skip_identity_refine=true`. GPT first-pass output is the final output. Other styles keep their existing repaint behavior.

The recommended reference was changed from `06-blue-white-fullbody.png` to `04-white-bridal-set.png` because the former contained an unrelated weapon. The replacement presents a complete coordinated garment construction without that contaminating prop.

See [the embedded comparison](cute-lingerie-wardrobe-20260926.html).
