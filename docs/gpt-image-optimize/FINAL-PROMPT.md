# 最终固件（重绘提线 Prompt）快照

> 运行时读的是 `prompts/gpt-image-optimize/` 下的**两个文件**，拼接方式为 `主prompt + "\n\n" + 细节后缀`。
> 本文件是同内容快照，**改 prompt 请改 prompts/ 下的文件**，然后回来同步这份快照。
>
> 来源：批次 D 的 `L5` 变体（结构保真 + 修手/发丝），由 `docs/gpt-image-optimize/experiments-D-lace-fidelity.md` 的实验数据支撑。

## 主 prompt（`prompts/gpt-image-optimize/repaint-system.md`）

```text
Repaint this image as a clean cel-shaded anime illustration. This is a rendering-quality upgrade and defect repair, not a redesign: keep the same composition, pose, camera angle, character design, the exact outfit design, and the background layout. Add and remove nothing; do not crop or reframe.

PRESERVE EVERYTHING THE CHARACTER IS WEARING OR CARRYING - this comes first.
Go over the source image item by item and keep every accessory, garment and prop exactly as it is. Nothing may be dropped, simplified away, or merged into the body. In particular keep:
- gloves and any hand coverings, in the same material, colour, length and cut as the source (mittens, wrist cuffs, lace half-gloves, fingerless gloves all stay, with the same coverage of the fingers and the same trim); if the source shows a covered hand, the repainted hand stays covered;
- legwear: tights, pantyhose, stockings, thigh-highs, knee-highs, socks, their opacity, colour and where they end (above or below the knee, above the ankle) are all part of the design and must stay exactly as drawn, including when the legwear is fully opaque and dark;
- footwear: shoes, boots, sandals and their straps, buckles, ribbons and heels;
- jewellery and hair ornaments: earrings, necklaces, chokers, hairpins, ribbons, hairbands;
- small emotional and facial details: tears, tear streaks, sweat drops, blush, freckles, a beauty mark, glitter - keep them in the same place with the same size;
- held objects: books, fans, cups, umbrellas, bags and anything else in her hands;
- the dress itself: every ruffle, frill, lace trim, bow, ribbon, seam and fold, with the same count and the same layering;
- the hammock or any other support, with the same weave and the same contact points with her body.
Treat this list as a checklist: before output, scan the source and confirm none of these has disappeared.

THEN apply the rendering upgrade:
- flat cel shading, flat base colours filled to the line, one hard-edged shadow tone per material, no soft airbrush gradients, no painterly blending;
- bold, even-weight, continuous pen outlines that close at every corner, no gaps, no double strokes, no sketchy or hairy fragments;
- limited palette with clearly separated colours and no bleeding between adjacent materials;
- every element read as a countable structure: each lace scallop, frill layer, ribbon, bow, hair clump and leaf a distinct closed shape rather than a soft blur;
- uniform crispness across the frame - the hammock, water, sky and foliage as sharply edged as the figure; no blurry region anywhere.

THEN correct the defects:
1. Hands. Fix the anatomy WITHOUT changing what the hands are wearing. Five fingers each, correct thumb placement and finger order, correct joints, no fused, melted, duplicated, missing or extra digits, no finger merging into the palm, the wrist or the sleeve. If the hands are gloved, the corrected hands keep the same glove; if the fingers are bare, the bare skin keeps the same colour and the same nails.
2. Hair. Every lock is one unbroken tapered strand from root to tip; no strand fades into blur or stops in mid air. Keep the original hairstyle, volume and silhouette.
3. Any other broken anatomy or merged shape you find must be repaired the same way - but never by deleting an accessory or a garment detail.

Keep the palette, lighting direction and mood of the original. Output one image at the same aspect ratio, with the same palette and lighting. Before finishing, verify: every accessory, garment and small detail from the checklist is still present and unchanged. No blur, no noise, no text, no watermark.
```

## 细节后缀（`prompts/gpt-image-optimize/repaint-detail-suffix.md`）

```text
Promote the detail quality of this artwork WITHOUT changing a single thing about what the character wears or carries. This pass is about line and textile clarity, not about simplifying the design.

NON-NEGOTIABLE COMPLETENESS CHECK (go through the source first, item by item):
- gloves or any hand covering: keep the same material, colour, length, cut and finger coverage; a covered hand stays covered;
- legwear: tights, pantyhose, stockings, thigh-highs, knee-highs, socks - keep the same opacity, colour and ending height, including fully opaque dark tights;
- footwear and its straps, buckles, ribbons, heels;
- jewellery, earrings, choker, necklaces, hairpins, hair ribbons, headbands;
- facial micro-details in the same place and size: tears, tear streaks, sweat, blush, freckles, beauty marks;
- props held in the hands: books, fans, cups, umbrellas, bags;
- the dress: same number of ruffles, frills, lace rows, bows, ribbon threading, seams and folds, with the same layering;
- the hammock or support: same weave and the same contact points with the body.
Nothing on this list may be dropped, simplified away, merged into the skin, or recoloured.

Rendering upgrade to apply on top of that - flat cel shading, flat colours filled to the line, one hard-edged shadow tone per material, no soft gradients, no painterly blending, bold even-weight continuous outlines that close at every corner, limited palette with no colour bleeding. Lace trims stay narrow repeating bands of many small evenly spaced scalloped arcs, each closed and separated from its neighbour - never one large smooth wave, never an empty band. Every frill row keeps its own outline and its own scalloped bottom edge, and the row count is unchanged. Hair is drawn as unbroken tapered strands with countable clumps. Uniform crispness everywhere, no blurry region left.

Defect repair, done without deleting anything: hands get correct anatomy (five fingers, correct thumb, correct joints, no fused or missing digits) while keeping whatever the hands are wearing exactly as drawn; any merged or broken shape elsewhere gets the same treatment, including the branch, moss, water, foliage and far shore.

Before output, do the completeness pass again: confirm every accessory, garment detail and facial micro-detail is still there and unchanged, and that the frill row count and the lace scallop spacing match the source. Same aspect ratio, same palette, same lighting. No blur, no noise, no text, no watermark.
```

## 逐条解释：为什么每段都在

| 段落 | 作用 | 删掉会怎样（实验依据） |
|---|---|---|
| `rendering-quality upgrade and defect repair, not a redesign` + 保留清单 | 钉死"不是新图"，保住构图/pose/服装/背景 | 构图漂移会明显变大（批次 C 里 A1 就是靠这句保住构图） |
| `ABSOLUTE RULE - NO SIMPLIFICATION` + 6 条 | 阻止模型把蕾丝/荷叶边"合并简化" | 蕾丝被并成大圆瓣、荷叶边行数被吃掉（批次 D 的病因） |
| `small, evenly spaced scallop arcs ... with a small hole or gap` | 给出**正确的单元形态** | 只说"保持细节"时模型不知道要画成什么（L3 掉了 44% 的 lace_teng） |
| `a row of small scallops must never become one large smooth wave` | 点名最常见的偷懒方式 | 这是 C4/C5/C7 的实际失败模式 |
| `every frill row keeps its own outline ... number stays the same` | 保住层级结构 | 行数被合并 |
| `pintuck pleats / piped seams / eyelet ribbon / waist seam / apron panel` | 保住低频但重要的结构 | 裙身被抹平成一块布 |
| `THEN CORRECT THE DEFECTS` + 手 / 发 / 其他（分优先级） | 修源图的解剖错误 | 只写"重绘风格"时模型**忠实复制坏手**（批次 C 的 A1） |
| `the source hands are malformed, so reconstruct both instead of copying them` | 明确授权改动手部 | 这句是手部改善的开关 |
| `no fused, melted, duplicated, missing or extra digits ...` | 负面清单 | 正面形容词效果差（B 组结论） |
| `no strand fades into blur or stops in mid air` + 点名"画面右侧长发" | 发丝连续性 | 不点名区域时模型只精修脸附近 |
| `STYLE: flat cel shading ... uniform crispness` | 画风与全画面一致性 | 背景会保持糊（源图的典型问题） |
| `FINAL CHECK BEFORE OUTPUT: ... count ... trace ...` | 让模型自检 | 自检句与手/发丝命中率正相关（批次 C 第 3 条发现） |
| 细节后缀（第二段） | 用独立段落再强调纺织物可读性 + 手 + 发 + 均匀清晰 | 挤在结尾逗号里效果更弱；独立段落更稳（固件设计） |

## 调参与抽卡建议

- 模型：`gemini-3-pro-image-preview`（质量最好）；`gemini-3.1-flash-image*` 更快但细节弱一些。
- 分辨率：`2K` 起步；要印刷级可用 `4K`（更慢更贵）。
- 宽高比：设成与源图一致（本实验是 `2:3`）。
- **重复次数 2–3**：重绘是抽卡，同一 prompt 两次的手/发丝质量会有差异（`metrics.md` 3.4）。
