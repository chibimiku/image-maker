# Sakurapion Gemini cross-subject correction (2026-09-26)

The reported output received the configured tropical-table reference correctly. The reference is intentionally retained: an art-style preset must transfer across unrelated subjects and scenes. Replacing it with a school-fashion image produced an easier same-domain result, but that diagnostic approach was rejected because it hides the generalization problem.

## Actual cause

1. `sakurapion-style` originally lacked `prompt_compressed`, so reference-priority mode dropped the distinctive face, iris and hair rules.
2. The Gemini content prompt was a multi-thousand-character prose description containing competing rendering phrases such as polished, realistic texture and subdued metropolitan palette. Those phrases acted as a second style prompt and outweighed the reference.
3. The issue was prompt-role separation, not subject mismatch.

## Final fix

- Restored `02-tropical-table.jpg` as the reference.
- Added a complete compressed style specification based on invariant drawing traits rather than tropical content.
- Added the optional style field `gemini_content_field`. For Sakurapion it is `gpt_image_prompt`, so reference-priority Gemini receives the existing content-only anchor rather than prose containing competing style language.
- The field is opt-in per style and does not affect the GPT-image channel or other styles.

The controlled tropical-reference result retains the unrelated staircase, black-haired subject, uniform and dynamic pose while improving gem-like eyes, grouped hair highlights, coloured contour hierarchy, garment detail and warm-cool separation. See [the embedded comparison](sakurapion-gemini-fix-20260926.html).
