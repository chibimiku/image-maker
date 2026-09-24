# gpt-5.6-sol：线粗 / 风格偏离 / 细节乱画（2026-09-23 01:56）

每轮只带 1 张图（中转站对多图载荷会断连）。

## 第 1 轮：针对最终产物的改法

## 按优先级排序的改动清单

### 1. 先压低结构线叠加的存在感，解决“粗、黑、硬”

**改什么参数 / 条款**

将当前结构线叠加参数改为：

- `strength：0.50 → 0.26～0.32`，建议先用 `0.28`
- `darken：0.45 → 0.14～0.22`，建议先用 `0.18`
- `掩膜膨胀：2×2 → 1×1`；如果仍显粗，直接取消膨胀
- 线层混合强度降低到原来的 `55%～65%`
- 结构线只作用于人物和服装，不覆盖背景家具、地毯、窗框等高对比边缘
- 线色不要使用纯黑或深棕，改为低饱和暖灰紫，例如：
  - `#8F858B`
  - 或 `RGB(143,133,139)`
- 增加线条约束：

```text
hairline-thin contours, variable line weight, tapered stroke endings,
delicate broken contours, low-contrast gray-lilac linework,
soft disappearing line ends, preserve small white gaps between adjacent forms
```

负面条款：

```text
no thick outline, no uniform line width, no black ink contour,
no hard brown-gray line, no comic inking, no closed heavy contours,
no double outline, no dark edge enhancement
```

**期望效果**

- 人物轮廓从现在的棕灰色硬边，变成更细、更浅、更有断续感的线。
- 头发、裙摆和褶皱的线尾能自然变细、消失，而不是每条线都以同样粗细结束。
- 面部、手部和裙边不会被结构线压黑。
- 保留必要的轮廓识别，但不再有“描边后贴上去”的感觉。

**风险**

- `strength` 和 `darken` 同时降低过多时，丝袜边缘、裙褶和靴子结构可能变得不够清楚。
- 取消膨胀后，局部线条可能出现断裂。建议先使用 `1×1`，不要一开始完全取消。
- 线色过偏紫会让整张图发冷，建议先用低饱和灰紫，而不是明显的紫色。

---

### 2. 改写 Gemini 重绘提示词，纠正“粉白厚涂 + 棕灰硬线”

**改什么参数 / 条款**

在 Gemini 的风格提示中，明确区分两张参考图的职责：

```text
Use the original photo/result as the primary reference for anatomy,
pose, proportions, facial identity, clothing construction and object placement.

Use the supplied tinkle reference only for rendering language:
delicate hairline contours, transparent pale watercolor layers,
fine tapered line endings, airy negative space, subtle paper texture,
rich but translucent layering, soft cool-warm pastel harmony.
Do not copy the reference composition or character design.
```

加入明确的画面风格条款：

```text
delicate fine-line illustration,
transparent watercolor and very light gouache washes,
layered translucent colors,
soft ivory paper base,
restrained blush pink,
pale lavender-gray,
muted blue-green,
quiet warm beige,
cool gray-lilac linework,
small controlled color accents,
visible light underpainting,
soft edges combined with selected crisp details,
rich layered but airy rendering
```

针对当前画面应增加的负面条款：

```text
avoid milky white overpainting,
avoid opaque pink-white paint,
avoid chalky thick gouache,
avoid flat pastel fill,
avoid muddy brown-gray outlines,
avoid dark rigid contour lines,
avoid uniformly heavy linework,
avoid glossy airbrush skin,
avoid blown-out white highlights,
avoid low-detail blurry lace and shoe details
```

如果 Gemini 的参考图选择和权重可控，建议：

- 源图：作为**结构主参考**
- tinkle 参考图：作为**风格主参考**
- 白底线锚图：只作为**局部结构参考**，不要让它控制整张图的线条黑度
- 全图重绘时不要把白底线锚图作为高强度全局参考

如果能设置风格影响强度，可先试：

- 源图结构保持：`0.70～0.80`
- tinkle 风格影响：`0.35～0.45`
- 白底结构线影响：`0.20～0.30`

**期望效果**

- 减少现在画面中大面积粉白、奶油色覆盖造成的厚重感。
- 让裙子、头发和背景出现更明显的透明叠色，而不是一层浅粉色糊住。
- 线条从棕灰硬线改成低对比的灰紫或暖冷中性线。
- 保留当前人物姿势、服装轮廓和室内构图，不把风格迁移变成重新设计人物。

**风险**

- 风格影响过高时，Gemini 可能重新设计服装褶皱、花边或脸部细节。
- 透明度要求过强时，皮肤和白裙可能发灰、发脏，尤其是腿部高光区域。
- 负面提示中如果同时强调“soft、delicate、low contrast”过多，可能导致眼睛、蕾丝和鞋带细节消失。应保留 `selected crisp details` 和 `clear small-scale construction details`。

---

### 3. 不要再用“整身排除脸部”的方式处理靴子，单独建立靴子局部重绘

**改什么参数 / 条款**

将靴子处理从全身局部重绘中拆出来，建立单独的局部步骤：

#### 局部范围

只框选：

- 靴筒
- 脚踝绑带
- 鞋带
- 鞋眼
- 靴口附近的少量袜边

尽量排除：

- 大腿和小腿皮肤
- 裙摆
- 地毯
- 椅腿
- 靴子外部背景

遮罩不要覆盖整条腿，也不要把靴子周围大面积背景一起交给模型。

#### 局部重绘参数

建议初始值：

- 局部裁剪放大：`2.5×～4×`
- 局部重绘强度：`0.28～0.38`，建议先用 `0.32`
- 局部结构控制强度：`0.65～0.78`，建议先用 `0.70`
- 遮罩膨胀：`1～2 px`
- 羽化：`2～4 px`
- 贴回时只做轻微色彩匹配，不要再次强行加深线条

这里应当使用“高结构控制、低创意重绘”，而不是提高整个画面的重绘强度。

#### 靴子局部提示词

```text
Reconstruct only the ankle boot details.
Keep the original boot silhouette, perspective, material and position.

The ankle straps must be clean, continuous and physically plausible.
Each strap has a clear beginning and ending.
The visible eyelets are evenly spaced and aligned with the boot opening.
The lace crosses alternately through the eyelets in an orderly sequence,
with a small controlled knot and two readable lace ends.
Preserve clear gaps between straps, laces and the boot surface.
Use delicate thin tapered linework and translucent light shading.
```

负面提示词：

```text
no scribbled lines, no tangled spaghetti lace,
no random loops, no melted straps, no extra shoelaces,
no duplicated eyelets, no dense black knot,
no hairy texture, no chaotic crosshatching,
no fused straps, no illegible shoe construction
```

如果模型仍然把鞋带画成线团，可分成两次：

1. **先只重建鞋带几何结构**  
   使用白底线锚图或手工画出的简化线稿，局部结构强度 `0.75～0.85`，颜色先保持浅灰。
2. **再进行低强度颜色和材质重绘**  
   重绘强度降到 `0.20～0.28`，避免模型重新改变鞋带走向。

线锚图最好只画出：

- 鞋眼位置
- 鞋带交叉路径
- 绑带的起止点
- 鞋带结的大致轮廓

不要在锚图中加入过多阴影，否则模型会把阴影误认为额外的鞋带。

**期望效果**

- 脚踝处的绑带变成几条独立、连续、可读的带子。
- 鞋带交叉关系清楚，不再堆成一团随机棕线。
- 鞋眼、鞋带和靴面之间保留小面积空隙。
- 靴子的透视和原有轮廓不被重新设计。

**风险**

- 局部裁剪太小会让模型看不懂靴子的空间关系，容易出现错位；至少要保留靴筒、鞋面和少量小腿作为上下文。
- 结构控制太强时，鞋带可能过于机械、像矢量线稿；完成几何修复后应再降低线层强度。
- 羽化过大可能把靴子边缘的浅色线带到地毯或腿部，建议控制在 `2～4 px`。

---

### 4. 最后再做一次“线条专用”轻量修正，不要用重绘解决线条问题

**改什么参数 / 条款**

在局部靴子修复和 Gemini 风格重绘之后，只对人物线层做一次轻量处理：

- 结构线 `strength：0.20～0.26`
- `darken：0.10～0.16`
- 不再扩大掩膜
- 线层透明度：`35%～45%`
- 对外轮廓使用更低透明度，对鞋带、花边、袖口等局部结构使用略高透明度
- 不对脸部重新叠加结构线，继续保持脸部排除

可加入一条层级约束：

```text
Line hierarchy:
hair, lace, shoe straps and small garment details may be slightly clearer;
outer body contour and background edges must remain lighter and thinner;
do not give every edge equal visual weight.
```

**期望效果**

- 线条层只负责补充结构，不再重新定义整张图的明暗。
- 鞋带等小细节能够清楚，但不会把人物轮廓和背景一起压黑。
- 形成“局部细节清晰、整体轮廓轻”的 tinkle 风格层次。

**风险**

- 局部提高细节线透明度时，鞋带和蕾丝可能再次变得抢眼；上限建议不要超过整体线层的 `1.25×`。
- 如果线层来自全图自动边缘检测，背景窗框、椅子和地毯也可能被误增强，因此应优先使用人物遮罩。

## 第 2 轮：靴子区绑带问题

### 判断

这不是单纯的“分辨率不足”。更可能是：

1. **局部重绘在高频细节区失控，是主因**：鞋带、蝴蝶结、鞋孔、鞋舌都属于细而密的线条，重绘时模型容易重复生成多条相近路径，造成线头不闭合、回折、互相穿插。
2. **结构线叠加进一步强化了杂边**：靴筒边缘、鞋舌边缘、鞋带和鞋孔的线被同时保留，导致局部出现多重轮廓。图中蝴蝶结和踝部附近尤其明显。
3. **分辨率不足是放大问题，不是根本原因**：在这个裁切尺寸下，鞋带单根线只有约 1–3 像素宽，确实容易粘连；但单靠提高分辨率不会修复错误的走向，只会得到更清晰的“乱线”。

也就是说，优先级大致是：**局部重绘失控 ＞ 结构线叠加 ＞ 分辨率不足**。

---

## 修法一：只重绘鞋带，不重绘靴子结构

**适用：**靴子轮廓、鞋底和鞋帮基本正常，只是踝部蝴蝶结及鞋带乱。

### 流程

1. 建立精确蒙版，只覆盖：
   - 踝部蝴蝶结；
   - 鞋舌上的鞋带；
   - 左右鞋孔之间的交叉线；
   - 不要覆盖靴筒外轮廓、鞋底、腿部和裙摆。
2. 蒙版边缘羽化 **4–8 px**，原图尺寸下操作。
3. Inpaint 参数建议：
   - Denoise / 重绘强度：**0.35–0.48**
   - CFG：**5–7**
   - 采样步数：**28–40**
   - 若有 ControlNet Lineart / SoftEdge：权重 **0.55–0.7**
4. 提示词明确写成结构，而不是泛化的“lace”：

   > clean pair of crossed boot laces, two symmetrical eyelet rows, one continuous thin shoelace, neat bow knot, visible lace ends, consistent line art

   负面词：

   > tangled scribble, random loops, spaghetti lines, extra shoelaces, broken strands, messy knot, duplicate outlines

### 预期效果

保留原本靴子的形状和线稿，只让鞋带重新组织成少量连续路径。重点是**不要把鞋帮和鞋底一起重绘**，否则模型会重新改动整个靴子的结构。

---

## 修法二：先清除旧线，再按“鞋孔—交叉—蝴蝶结”分段重建

**适用：**当前鞋带和鞋孔已经互相污染，直接局部重绘仍会继承旧的杂线。

### 流程

分两次处理，不要一次让模型同时生成所有细节。

#### 第一步：清理鞋带区域

蒙版覆盖：

- 踝部蝴蝶结；
- 两排鞋孔之间的鞋舌；
- 所有明显的重复乱线。

但保留：

- 靴子左右外边缘；
- 鞋底；
- 鞋跟；
- 靴筒与腿的交界。

参数：

- Denoise：**0.55–0.65**
- CFG：**5–6**
- 先使用较低细节、较平整的提示词，例如 `clean plain boot upper, simple smooth leather panel`。
- 这一步不要加入 `intricate lace` 或 `ornamental bow`。

#### 第二步：在干净鞋面上重画鞋带

把区域缩小为：

- 左右鞋孔列；
- 鞋舌中央；
- 蝴蝶结所在的踝部横向区域。

参数：

- Denoise：**0.30–0.42**
- ControlNet Lineart 或手工草图权重：**0.65–0.8**
- 鞋孔数量固定为 **每侧 3–4 个**，不要让模型自由增加数量。
- 提示词加入：
  > exactly two eyelet rows, three crossing segments on each side, one bow at the ankle, clean separated strands

如果工作流支持，最好先用画笔手工画出两排鞋孔和大致鞋带路径，再用低强度 img2img 统一线稿。

### 预期效果

先消除“旧线叠加”，再让模型生成新鞋带，能明显减少幽灵线、重复轮廓和无来源的线头。相比单次重绘，这种方法更适合当前这种已经形成线团的区域。

---

## 修法三：提高局部有效分辨率，再做低强度细节修复

**适用：**鞋带走向基本对，但线条挤在一起、局部糊成一团，属于像素不足或缩放后粘连。

### 流程

1. 只裁出靴子区域，建议包含：
   - 从膝下到鞋底；
   - 左右各多留 **10–15% 边界**；
   - 不要把整条裙摆和大面积地毯一起放进细节重绘。
2. 局部放大 **2–3 倍**，不要直接放大 4 倍以上。  
   例如原局部宽度 250 px，放大到 **500–750 px**。
3. 先进行普通放大或 latent upscale：
   - 放大倍率：**2x**
   - 去噪：**0.15–0.25**
4. 然后只对鞋带蒙版做一次细节修复：
   - Denoise：**0.20–0.32**
   - Steps：**20–30**
   - CFG：**4.5–6**
   - 细节增强不要超过 **0.35**，否则会再次生成额外鞋带。
5. 最后缩回原图尺寸，并用轻微锐化：
   - Unsharp Mask 半径约 **0.5–0.8 px**
   - 强度 **50–80%**
   - 不要使用强烈的局部对比度增强。

### 预期效果

让单根鞋带在高分辨率中拥有足够像素宽度，减少交叉处粘连，同时避免把地毯纹理、裙摆褶皱一起锐化。它能解决“线太细、缩放后糊掉”的问题，但前提是鞋带路径本身已经大致正确；如果路径已经乱了，应先使用修法一或二。

---

### 推荐顺序

最稳妥的实际顺序是：

1. **先用修法二清掉明显重复线；**
2. **再用修法一低强度重绘鞋带和蝴蝶结；**
3. **最后用修法三做 2x 局部放大和轻微细节修复。**

不要直接对整只靴子做高强度重绘，否则鞋底、鞋跟、鞋面边缘也可能被重新解释，容易从“鞋带乱”变成“整个靴子结构变形”。

## 第 3 轮：tinkle 参考图的英文条款清单

- Ornate Japanese fantasy character illustration with delicate gothic-lolita elegance, luminous jewel-like colors, intricate costume design, and a polished storybook atmosphere.
- Use clean tapered linework with dark navy-blue or deep indigo contours instead of pure black.
- Keep primary silhouette lines at approximately 0.20–0.40% of the image width, secondary structural lines at 0.10–0.20%, and micro-detail lines at 0.04–0.10% of the image width, never exceeding roughly 1–2% of the eye height.
- Use thicker lines for the outer silhouette, overlapping foreground forms, facial framing, and major costume structures.
- Use thinner, lighter lines for hair strands, lace, fabric folds, embroidery, highlights, and distant background motifs.
- Taper every stroke toward its endpoint, with sharp elegant finishes and no blunt mechanical line caps.
- Vary line weight continuously according to depth, overlap, lighting, and material rather than using uniform outlines.
- Build the palette from saturated sapphire, midnight blue, turquoise, rose pink, pearl white, and cool lavender accents.
- Create translucency through layered saturated color, soft subsurface pinks, controlled cool shadows, and selective luminous highlights rather than by whitening the entire image.
- Preserve strong color chroma in midtones and shadows, using deep blue-violet anchors to prevent the image from becoming pale or gray.
- Use bright highlights sparingly on hair, eyes, satin, jewelry, and glossy accessories, with clear separation between highlight, local color, and shadow.
- Maintain a high-key luminous atmosphere while preserving distinct dark values, readable silhouettes, and rich chromatic contrast.
- Render with smooth cel-shaded foundations, softly blended transitions, delicate airbrushed light, and restrained painterly texture.
- Layer transparent glazes and cool reflected light over solid base colors to create glassy depth and crystal-clear illumination.
- Keep important contours crisp and controlled while allowing atmospheric background elements and distant fabric to use softer edges.
- Use selective bloom only around intense highlights, eyes, gemstones, and reflective satin, avoiding global haze or washed-out lighting.
- Paint hair as large flowing masses first, then divide it into grouped locks with tapered internal strands, sharp specular streaks, and varied strand spacing.
- Keep hair strand details directional and grouped along the form, avoiding evenly spaced parallel lines or tangled spaghetti-like strokes.
- Render lace as small connected scallops, floral motifs, and negative-space cutouts with consistent rhythm and clear spacing.
- Render ribbons, straps, and shoelaces as clean overlapping bands with visible thickness, tension, knots, loops, and cast shadows.
- Simplify tiny accessories into readable graphic shapes with a few high-contrast accents instead of filling them with noisy micro-lines.
- Draw fabric folds as selective long curves following tension and gravity, supported by deep occlusion shadows and narrow luminous ridges.
- Use crisp decorative edges, translucent frills, layered ruffles, and satin reflections while preserving clear material separation.
- Give the eyes large luminous irises with layered turquoise gradients, dark defined rims, multi-point reflections, and subtle inner glow.
- Keep facial features delicate and minimal, with soft blush, clean skin gradients, and no harsh gray modeling.
- Use sparkling floral ornaments, jewelry, and gemstones as concentrated focal accents with sharp highlights and saturated surrounding shadows.
- Separate foreground, character, costume, and background through value grouping, edge sharpness, atmospheric softness, and controlled overlap.
- Avoid muddy gray shadows, desaturated midtones, excessive white wash, flat lighting, and low-contrast color blending.
- Avoid pure black outlines, uniform line weight, blunt stroke endings, rough sketch residue, and heavy comic-style contouring.
- Avoid over-smoothed vector edges, plastic-looking skin, generic airbrushed gradients, and textureless flat fills.
- Avoid tangled hair lines, random lace noise, illegible straps, merged ribbons, broken shoelaces, and indistinguishable accessory details.
- Avoid excessive bloom, blown-out highlights, chromatic clipping, gray veils, dirty color mixing, and loss of deep navy value structure.
- Avoid overcrowded micro-details that compete with the face, eyes, silhouette, and main costume focal points.
