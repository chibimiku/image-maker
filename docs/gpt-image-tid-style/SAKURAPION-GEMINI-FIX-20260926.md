# Sakurapion Gemini direct-generation correction (2026-09-26)

The reported output `data/20260926/c4ae2603_194635-3e784f.jpg` received the correct reference file, so this was not a missing-attachment bug.

## Root causes

1. `sakurapion-style` had no `prompt_compressed`. Gemini reference-priority mode therefore used the local heuristic compression, whose omitted middle contained the distinctive face, iris and hair construction.
2. The former reference `02-tropical-table.jpg` is prop-heavy and semantically far from a monochrome stairwell school-uniform scene. Gemini followed the content description and fell back to generic anime rendering.
3. The run used the original description, which explicitly requested a surgical mask. The refined analysis description removes it; this content difference is independent of the style bug.

## Fix

- Added an authored compressed Gemini specification that retains face shape, gem-like iris construction, grouped lashes, ribbon-like hair masses, segmented highlights, coloured contour hierarchy and tinted cel shadows.
- Changed the representative reference to `10-school-fashion.jpg`, which exposes the same author-specific facial and line treatment in a full-body fashion composition with fewer unrelated props.
- Corrected a malformed character in the full prompt.

The controlled same-content/same-model/same-resolution comparison shows stronger Sakurapion eye construction, grouped hair highlights, coloured contours and luminous pastel separation. See [the embedded comparison](sakurapion-gemini-fix-20260926.html).
