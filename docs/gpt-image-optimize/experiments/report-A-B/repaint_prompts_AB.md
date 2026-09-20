# 任务 A · 两种重绘 prompt 对比（同一张 v3 图做源）

## A1 · 带画风要求的重绘（`repaint_A1_style.md`）

```text
Repaint this entire image as a clean cel-shaded anime illustration. Keep the exact same composition, pose, camera angle, character design, outfit design and background layout — this is a re-rendering pass, not a new picture. Do not add or remove characters or props, do not crop, do not reframe.

Art style to apply:
- Traditional Japanese anime cel animation look: flat base colors filled to the line, exactly one hard-edged shadow tone per material, no soft airbrush gradients, no painterly blending, minimal highlights.
- Bold, even-weight, continuous outlines around every shape, drawn like a clean pen line. No sketchy, hairy or broken strokes.
- Limited pastel palette, clearly separated colours, no colour bleeding between adjacent materials.
- Every shape must read as a countable structure: each lace scallop, each frill layer, each ribbon, each hair clump and each leaf drawn as a distinct closed shape rather than a soft blur.
- Uniform crispness across the whole frame: background, tree branch, lake and scenery rendered with the same clean edge definition as the character.

Keep the palette, lighting direction and mood of the original. Output one image at the same aspect ratio, drawn in this flat cel style, no blur, no noise, no text, no watermark.
```

## A2 · 只要求线条连贯（`repaint_A2_lines.md`）

```text
Redraw this image so that its linework is continuous and its details are crisp.

Requirements:
- Every outline must be unbroken and closed. Lines must connect where they meet, with no gaps, no double strokes and no sketchy or hairy fragments.
- Sharpened detail: hair strands, eyelashes, iris detail, lace, frills, ribbons, fabric seams, leaves and water ripples must be rendered as clearly separated structures instead of a soft blur.
- Consistent line quality across the whole frame. No region may be left softer or blurrier than another.
- Keep everything else as it is: same composition, same pose, same character, same colours, same lighting, same background layout, same aspect ratio.
- No blur, no noise, no compression artefacts, no text, no watermark.
```
