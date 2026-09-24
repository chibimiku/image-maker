先下结论：**图 4 还没到顶，但提示词层只剩一轮有意义的小修；主要增益应来自 `high` 和更明确的参考图约束。** 再继续堆风格形容词，收益很低。

**1. 图 4 最明显的三个不足**

1. **人物尤其脸、胸肩和手臂的明暗过于平滑，缺少图 1 的“清晰线稿 + 分块上色”**
   
   图 4 的脸颊、锁骨、肩膀形成大面积柔滑渐变，鼻口也偏空气感。图 1 虽然整体明亮，但眼睑、下巴、发束、衣褶都有明确且连续的细轮廓，阴影更像浅色块而不是柔焦塑形。这是目前最大的“画风不像”。

2. **发丝呈现两极化：主体发束太宽太塑料，外围又有过多细碎游丝**
   
   图 4 头顶和刘海是宽而亮的带状高光，像现代高完成度商业动漫渲染；外围发梢则出现大量细、浅、断续的线。图 1 的发束内部有更多连续长线和窄色块，主次更清楚。指标也支持这一点：图 4 细尺度边缘仍为 `0.587`，高于参考的 `0.478`，说明碎边仍偏多。

3. **背景与主体抢层次，构图也比图 1 更静、更模板化**
   
   水面高频反光、云层和右侧太阳都很显眼，人物却正中、正面、近似左右对称。图 1 的长发、躯干、手臂和植物形成明显方向性，背景留白承担衬托作用。图 4 局部 RMS 只有 `19.8`，不是没有亮暗，而是对比大量消耗在水面细碎纹理上，人物内部结构反而偏软。

另外，图 4 没有真正生成手部，而是让前臂入水规避了手。它不算当前画面的视觉错误，但对“首图可直接使用”是明显缺口。

**2. 对 v5 的小幅修改**

你没有在本条消息中贴出 v5 原文，所以我只能按上一版的语义字段给出精确替换，不能保证字段名逐字一致。

- **线稿字段**
  
  将类似 `fine delicate linework` 改成：
  
  `continuous, confident colored linework; long connected contours around hair, jaw, dress and arms; medium-fine primary outlines with fewer stray micro-lines`
  
  预期：减少外围断发和碎线，同时保留轮廓，不再靠“越细越好”。

- **上色/光影字段**
  
  将 `soft luminous shading / glowing sunset light` 改成：
  
  `restrained cel-painted shading with clearly separated midtones; narrow highlights only; no bloom or airbrushed skin`
  
  预期：把对比移回脸、头发和衣褶，增加中灰与局部 RMS，避免宽高光带。

- **头发字段**
  
  补充：
  
  `hair organized into readable tapered locks, each defined by one outer contour and a few long interior strokes; matte color masses, no plastic ribbon highlights`
  
  预期：解决“主发束太宽、外围发丝太碎”。

- **背景字段**
  
  将详细水面/天空描写压缩为：
  
  `quiet simplified dusk sea, broad watercolor shapes, sparse reflections, subdued clouds`
  
  预期：降低背景细边缘密度，让人物成为最高信息区。

- **构图/姿态字段**
  
  增加：
  
  `a slight three-quarter torso turn and gentle shoulder tilt while her eyes face the camera; both hands visible at or just above the waterline`
  
  预期：避免证件照式正面对称，并阻止模型用水面藏手。若“半身”严格只到腰部，双手必须明确写出，否则仍可能被裁掉。

- **参考图约束字段**
  
  增加：
  
  `Use the reference for rendering grammar only: contour hierarchy, hair construction, flat midtone separation, restrained highlights, and watercolor accents. Do not copy its character, outfit, pose, or objects.`
  
  预期：把参考图影响集中到真正需要的视觉语法，而不是泛泛的“same style”。

**3. 参数层**

- **`quality`: 选 `high`。**
  
  图 3 已经表明 high 对长线连续性、中间调和局部结构有实际收益。Gemini 后续能修手、接发丝和放大，但无法无损恢复首图中已经被 medium 合并掉的发束、衣褶和面部轮廓。既然目标是首图本身够好，medium 的成本优势不应压过源图质量。

- **`size`: `1024x1536` 合适。**
  
  这是该构图合理的 2:3 竖幅。不要为了脸部细节改成方图，也不建议先生成横图再裁。人物约占画面高度 75% 到 85%，手要纳入构图。

- **`gpt-image-2.5-flare`: 暂不作为最终首选。**
  
  我无法从公开、稳定的模型契约验证它相对 `gpt-image-2` 的线条质量；“更快、output token 约 1/3”只证明效率，不证明长线连续性和风格遵循更好。最合理的是用同一提示词、同一输入图，各跑至少 4 张，比较长线占比、局部 RMS、手部成功率和人工风格评分。未完成这个 A/B 前，首图上限仍选你已经验证过的 `gpt-image-2 high`。

- **其他参数**
  
  使用无损 `PNG`；若只能用 WebP，设最高质量。透明背景对此场景无益。没有 seed 时，最有效的“参数”其实是一次生成多个候选再择优，而不是反复改提示词。`input_fidelity` 不要传；你已确认 `gpt-image-2` 不接受它。

**4. 是否增加结构图**

**当前这个简单站姿不建议默认增加。** `/images/edits` 的第二张图不是 ControlNet；模型可能同时吸收它的线宽、脸型、服装甚至灰度关系。结构图能提高姿态和手的位置确定性，但也可能削弱图 1 的风格控制。

只有在“必须露出双手且连续失败”时再加。最省事的做法：

1. 在 SD-WebUI 用 OpenPose/DWpose 预处理器，从一张姿势接近的照片或草图提取骨架。
2. 结构图不要只给彩色 OpenPose 骨架；最好用 SD 根据骨架生成一张**低细节灰度人体草图**，明确头、肩、肘、腕、手掌和水线。
3. 去掉脸部细节、材质、复杂衣褶和光影，避免它成为第二张风格参考。
4. 提示词明确指定：`Image 1 is the sole style reference. Image 2 controls only pose, crop, hand placement and waterline; ignore its style, colors, face, clothing and rendering.`

Gemini 也可以把草图清理成灰度结构稿，但它可能自行美化和补风格；就“只提供几何约束”而言，SD-WebUI 的 DWpose/OpenPose 更可控。

**5. 首图最优组合**

推荐：**`gpt-image-2` + `quality=high` + `1024x1536` + PNG + 图 1 作为唯一输入参考图**。先不加结构图；只有双手连续失败时才加入低细节灰度姿态稿。

可直接用的提示词如下，约 1,700 个 ASCII 字符：

```text
Create a polished vertical anime illustration.

SUBJECT
A young adult woman stands waist-deep in shallow water at dusk, shown from mid-thigh/waist upward. She looks directly at the camera with a calm, subtle smile. Give her torso a slight three-quarter turn and a gentle shoulder tilt rather than a rigid symmetrical pose. Both arms are naturally lowered, with both anatomically clear hands visible at or just above the waterline. She wears a modest flowing white summer dress with thin straps, small ruffles, and readable folds. Long dusty-pink hair moves lightly in the sea breeze.

COMPOSITION
2:3 portrait composition. The figure is the dominant focal point and occupies about 80% of the image height. Keep clear space around the head and hair tips. Use an elegant directional flow through the shoulders, hair and arms. The horizon remains secondary. No extreme perspective and no cropped hands.

STYLE REFERENCE
Image 1 is the sole style reference. Use it for rendering grammar only: crisp colored contour hierarchy, long connected lines, organized tapered hair locks, restrained watercolor texture, clean anime facial construction, flat midtone separation, and sparse decorative accents. Do not copy its character, pose, outfit, plants, fish, butterflies, or composition.

LINE AND COLOR
Use continuous, confident colored linework. Draw medium-fine primary contours around the hair silhouette, jaw, dress, arms and fingers, with thinner interior details. Prefer long connected strokes and clean closed forms; avoid faint broken outlines and excessive stray micro-hairs. Organize hair into readable tapered locks, each with one outer contour and only a few long interior strokes. Use matte color masses, not plastic ribbon highlights.

Apply restrained cel-painted shading with clearly separated light, midtone and shadow shapes. Preserve substantial midtones on the face, neck, hair and white dress. Keep highlights narrow and selective. Skin and fabric must retain visible structure under the sunset light. No bloom, haze, airbrushed skin or washed-out whites.

BACKGROUND
A quiet simplified dusk sea with broad watercolor shapes, sparse reflections, subdued pink-blue clouds and a small low sun. Keep background edges softer and less detailed than the figure. Water around the hands may ripple gently but must not hide the fingers.

AVOID
Overexposure, glowing white skin, broad clipped highlights, soft-focus rendering, glossy 3D hair, fragmented flyaway lines, noisy water texture, photographic realism, exaggerated bust, malformed or submerged hands, extra fingers, text, border or signature.
```

一句话结论：**当前可验证的首图上限是 `gpt-image-2 high + 1024x1536 PNG + 单张图 1 风格参考 + 强化长线/中间调/双手可见的 v5 小修版`，因为它优先修复 medium 最明显丢失的线条连续性与人物内部层次，同时避免第二张结构图稀释画风。**

还没有“完全到顶”，但已经接近提示词层上限。即便使用上述组合，**手指精确结构、极细发丝逐根连通、局部断线、眼睫和蕾丝的像素级清理、最终高分辨率锐化**仍属于 Gemini 重绘工序；这些无法靠单次无 seed 生成稳定保证。