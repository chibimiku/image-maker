## 1. 实际差异

先定性：图 1 并不是以“透明水彩晕染”为主。它更接近：

> 清晰数字动漫线稿 + 平涂/柔和赛璐璐塑形 + 局部半透明水彩效果 + 少量纸面颗粒和喷溅装饰

把整体概括成 `transparent watercolor-like washes`，会让模型把本应承担结构的头发、五官和服装也水彩化，这是图 2 失控的主要原因之一。

### 明度层次

图 1 有明确的明度锚点：

- 头发内部有粉红中间调，发根、耳后、颈侧和重叠发束有更深的玫红/灰紫阴影。
- 眼睑、瞳孔、领口、袖口、裙子、鞋袜形成连续的深色节点。
- 白色上衣不是纯白平面，褶皱使用浅灰紫和浅蓝分面。
- 背景虽白，但人物外轮廓大多由线稿、色相差和局部暗部共同分离。

图 2 则把大部分像素压进了高光区：

- 皮肤、白裙、天空、海面都集中在接近白色的窄明度范围。
- 左侧头发尚有浅粉中间调，右侧逆光发丝和天空几乎完全融合。
- 白裙胸口、腰部、水面交界缺少稳定的灰蓝阴影，衣服结构只能依赖零碎蕾丝边。
- 唯一较强的暗部集中在眼睛、左耳后和少量发束下方，无法支撑整个人体结构。
- “黄昏”被表现成强曝光的粉白逆光，而不是保留蓝紫中间调的高调黄昏。

### 线条连贯性

图 1：

- 线条通常能连续描述一个形体，例如一整束头发、脸颊、手臂、衣领和裙褶。
- 轮廓线与内部结构线有粗细层级。
- 即使局部淡化，线条也是“有控制地消失”，不会在关键转折处同时丢失。

图 2：

- 右侧头发外轮廓由大量很淡、近似等权重的细线叠加，很多线从中段开始或无结构地结束。
- 脸颊到下颌的轮廓过淡；鼻、嘴主要是短小孤立笔画，没有与面部体积形成关系。
- 眼睫相对清晰，但虹膜内部纹理、上眼睑和发丝交叠处发生粘连。
- 胸前蕾丝、肩带、蝴蝶结和水面高光都使用类似频率的细碎线，局部层级混乱。
- 这不只是“上采样糊”。更像是模型先生成了很多低对比微线，再被高曝光和水彩边缘融合一起压掉。

### 发丝结构

图 1 的头发是“先大组、后小束、最后少量单丝”：

- 头顶、刘海、两侧垂发、后发是可辨认的大体块。
- 每个大体块内部只有有限数量的锥形发束。
- 发束从发旋或发根出发，沿头颅体积连续流动到发梢。
- 暗部主要放在发根、叠压处和内层，因此浅色头发仍有体积。

图 2 的问题不是完全没有发丝，而是缺乏分级：

- 左侧长发先被铺成一块低对比粉色，再覆盖大量同等细度的曲线。
- 许多细线没有明确发根归属，也没有汇入一个发束。
- 右侧出现大量透明飞丝，轮廓被拆成线团。
- 刘海根部、鬓发、后发之间的遮挡关系不够明确。
- 高光近似大面积漂白，没有沿发束成组布置。

### 细节分布

图 1 的细节集中在脸、眼睛、发束交界、衣服结构和少量水滴；大片皮肤与背景保持安静。图 2 则把细节预算分散给了发丝、蕾丝、水纹、云层和气泡，关键结构与装饰纹理争夺同一层级。

另外，`tiny floating droplets and bubbles` 被模型执行得很强。它既复制了图 1 的装饰符号，又进一步挤占了眼睛、发束和衣褶的清晰度预算。

## 2. 问题 A：恢复明度层次

建议直接替换相关字段：

```text
Palette: pale pastel colors with controlled value separation; a white-dominant background, softly colored skin, and clearly visible midtones in the hair, dress folds, and water. Use muted rose, cool blue-gray, and lavender accents, plus a few small dark anchors around the eyes, hair overlaps, and clothing seams.

Lighting: luminous high-key dusk lighting with protected midtones and restrained highlights. Keep the face and white dress below pure white except for a few small specular accents. Use soft cool shadows at form turns and overlap points so the subject remains clearly separated from the background.

Composition density: low, with broad quiet areas and generous light negative space; negative space may be near-white, but the subject must retain a complete readable silhouette.

Detail level: restrained in quiet areas, structurally precise around the face, hair groups, neckline, dress construction, hands, and waterline.

Avoid: clipped highlights, global overexposure, white-on-white loss of silhouette, uniformly pale values, glowing haze over the subject, and removal of all shadows.
```

这些修改的关键点：

- 用 `white-dominant background` 代替 `lots of clean white`。前者限定白色主要属于背景，后者容易被解释成整幅图都向纯白推。
- 用 `controlled value separation` 和 `protected midtones` 明确要求保留中间调。
- 不再笼统地写 `soft even light`。`even light` 很容易消除体积和遮挡阴影。
- 不写 `Avoid: heavy shading`。模型可能把它执行成“尽量不要阴影”。改成要求 `soft cool shadows at form turns and overlap points`，同时禁止的是硬重阴影，而不是所有阴影。
- 把高光限制为 `a few small specular accents`，避免皮肤、裙子和背景一起曝光封顶。
- `complete readable silhouette` 比“提高对比度”更精确，因为你要的是局部结构分离，不是把整张图变硬、变黑。

建议避免这些词组：

```text
lots of clean white
soft even light
extremely airy
ethereal glow
dreamy haze
almost no shadows
very low contrast
washed-out pastel
pure white dress under bright backlight
```

其中 `low saturation` 本身可以保留，但必须和 `controlled value separation` 同时出现。低饱和不等于低明度对比。

## 3. 问题 B：线稿和发丝结构

替换为：

```text
Brushwork: clean digital anime drawing with continuous, confident fine linework over softly painted color shapes. Build each form with a small number of intentional lines. Use translucent watercolor texture only in background washes, water reflections, and sparse decorative accents, not as the construction method for the face, hair, eyes, or dress.

Edges: crisp and continuous on the face, eyes, hair silhouette, major hair locks, neckline, dress seams, and waterline. Allow soft or lost edges only in distant clouds, background water, and a few nonessential outer accents. Preserve edge hierarchy: strongest at focal features and overlaps, lighter on secondary contours.

Hair structure: organize the hair into a clear hierarchy of large masses, medium tapered locks, and only a few individual flyaway strands. Every major lock should begin at the crown or a visible root region, follow the curvature of the head, remain connected through its length, overlap clearly, and taper to a defined tip. Use coherent grouped highlights that follow the locks, with cool rose-gray shadows at roots, underneath layers, and overlap points.

Detail level: use three scales of detail: large readable silhouette and body masses; medium hair groups, facial features, dress folds, and waterline; sparse fine accents in eyelashes, irises, selected hair strands, lace edges, and droplets. Keep micro-detail concentrated around the face and upper torso.
```

措辞上的取舍：

- `delicate fine lineart` 容易得到“很细、很淡、很多、很脆弱”的线，不保证连续性。改成 `continuous, confident fine linework`。
- 单独写 `fine lineart` 不够，要加 `a small number of intentional lines`，抑制无层级的碎线堆积。
- `transparent watercolor-like washes` 不应作为全局 Brushwork。把水彩限定在背景、水面反光和装饰层。
- `Edges: soft, occasionally lost` 的作用范围太大。模型会在脸、发梢、衣领同样丢边。应明确列出哪些边必须清晰，哪些区域才允许 lost edges。
- 不建议要求“每根头发都清晰”。这通常会增加线团和噪声。目标应是有限分组的发束，单根飞丝只作少量点缀。
- 对蕾丝也不要写 `intricate lace`。使用 `simplified lace with a continuous outer contour and a few readable internal motifs` 更稳定。

## 4. 参数层建议

### `quality`

改成 `high`。

`medium` 足以验证构图和大体风格，但这种浅色细线、睫毛、发束交叠、蕾丝和水纹高度依赖局部解析能力。`high` 不能自动修复错误的线条组织，但通常能减少局部粘连和不完整边缘。

先用 `medium` 做提示词消融测试，锁定字段后用 `high` 出最终图。不要只把当前提示词原样升到 high：它可能更精细地生成同样错误的碎发和蕾丝噪声。

### `size`

`1024x1536` 的纵向比例适合当前构图，没必要仅为清晰度改比例。主体是腰上构图时，应让脸和上身占据更大画面面积：

```text
waist-up portrait, the head and upper torso occupy most of the frame
```

实际改善往往比继续提高画布尺寸更明显。不要先生成过远的全身图再期待后期上采样恢复发丝连接关系。

### 原画 + 主体草稿

建议使用两张输入图，前提是 API 调用方式允许多图输入：

1. 图 1负责风格、线色、明度组织和装饰语言。
2. 主体草稿负责姿势、裁切、视角、头发大分组、服装轮廓和水线。

草稿最好是清晰线稿或三值结构稿，不要是低质量成图。提示词中明确输入职责：

```text
Use the first image only as the rendering-style reference.
Use the second image as the composition and structural guide.
Preserve the second image's pose, crop, silhouette, major hair groups, dress construction, and waterline, while rendering it with the first image's visual treatment.
```

代价是风格服从度可能略降，但结构稳定性通常明显提高。若草稿里已经有大量碎发，模型也会继承，所以草稿只画大组和中组发束。

### 是否改用无主体风格板

不建议完全弃用图 1，只用裁剪后的无主体风格板。你最需要学习的恰恰是人物线稿、眼睛、头发和衣褶，而纯背景风格板无法提供这些信息。

更合适的是制作“部件式风格板”：

- 头部与刘海局部
- 一段长发及其发束末端
- 眼睛局部
- 衣领或裙褶局部
- 水滴、水面和留白背景局部
- 同时保留少量完整人物，提供全局线条层级

避免只保留颜色和纸纹，也避免把整张原画直接裁成无法理解的碎片。风格板还应去掉明显的角色身份特征和构图线索，否则 edits 容易复制粉发、饰物、植物或漂浮水滴，而不是只迁移画法。

## 5. 单次生成的上限与补救路径

单次可以显著改善，但很难同时保证：

- 严格复刻图 1 的线稿逻辑
- 全新主体和构图
- 白裙、浅肤、浅发、亮背景仍清楚分层
- 每处发束与蕾丝都连贯
- 水彩氛围不侵蚀线稿

推荐两步走。

### 路径一：结构图 → 风格重绘

第一步生成或提供结构明确的动漫线稿/低饱和平涂图：

```text
clean connected anime linework, grouped hair locks, clear dress construction,
simple value blocking, no watercolor texture, no haze
```

第二步用 edits 输入结构图和图 1，只添加配色、柔和渲染、纸纹、水面效果与装饰。

优点：发束连接、轮廓和服装结构最稳定。  
代价：第二步仍可能改动五官和细线，需要强调保留结构；工作流多一次调用。

### 路径二：先完成主体，再局部重绘

先生成整体满意的图，然后裁出头发、眼睛和衣领分别做高分辨率 edit，要求保持构图，只修复连续线稿和分组。最后缩回原尺寸合成。

优点：最适合修复局部糊线，成本可控。  
代价：区域边界可能出现色温或线宽变化，需要重叠裁切并留足上下文，不能只裁一小撮发丝。

### 路径三：外部模型提线，再回到 gpt-image

可用 Gemini 或其他擅长图像重绘的模型做“清理并连接线稿”，但不要只说“增强清晰度”。应指定：

```text
Connect broken contours, merge stray hair lines into a limited number of tapered
locks, preserve the face and composition, and do not add new strands or lace.
```

优点：能快速整理已有结果。  
代价：可能改变脸、眼睛和角色身份；单纯锐化只会把碎线和噪声一起强化。更可靠的做法是先提取/重画结构线，再把线稿作为约束输入最终渲染。

传统超分辨率只能改善像素边缘，不能推断正确的发束拓扑，因此不应作为问题 B 的主要方案。

## 6. 完整修正版提示词

```text
Create a new illustration using the reference image only as a guide to the rendering language, line hierarchy, color handling, and balance of crisp structure with sparse watercolor accents. Do not copy its character, pose, clothing, plants, fish, butterflies, or composition.

Palette: pale low-saturation pastels with controlled value separation. Use a white-dominant background, softly colored skin, and clearly visible midtones in the hair, white dress folds, and water. Use muted rose, cool blue-gray, and lavender dusk accents, with a few small dark anchors around the eyes, hair overlaps, neckline, and clothing seams.

Lighting: luminous high-key dusk lighting with protected midtones and restrained highlights. Keep the face and white dress below pure white except for a few small specular accents. Use soft cool shadows at form turns, roots, folds, and overlap points so the subject remains clearly separated from the background. Preserve a readable dusk atmosphere without glowing haze over the subject.

Brushwork: clean digital anime drawing with continuous, confident fine linework over softly painted color shapes. Build each form with a small number of intentional lines. Use translucent watercolor texture only in distant clouds, background water reflections, and sparse decorative accents, not as the construction method for the face, hair, eyes, or dress.

Edges: crisp and continuous on the face, eyes, hair silhouette, major hair locks, neckline, dress seams, and waterline. Allow soft or lost edges only in distant background elements and a few nonessential outer accents. Preserve a clear edge hierarchy: strongest at focal features and overlaps, lighter on secondary contours.

Texture: subtle paper grain in quiet background areas, gentle watercolor granulation in the sky and water, and only a few small droplets or bubbles as accents. Keep texture away from the eyes, facial contours, major hair boundaries, and important dress construction.

Composition density: low, with broad quiet areas and generous light negative space. Negative space may be near-white, but the subject must retain a complete readable silhouette. Keep background details sparse and subordinate to the figure.

Detail level: organize detail at three scales: a large readable silhouette and body masses; medium-scale grouped hair locks, facial features, dress folds, and waterline; sparse fine accents in eyelashes, irises, selected hair strands, simplified lace edges, and droplets. Concentrate fine detail around the face and upper torso. Organize the hair into clear large masses, medium tapered locks, and only a few individual flyaway strands. Each major lock begins at the crown or a visible root region, follows the curvature of the head, remains connected through its length, overlaps clearly, and tapers to a defined tip. Highlights follow grouped locks; cool rose-gray shadows define roots, inner layers, and overlap points.

Avoid: clipped highlights, global overexposure, uniformly pale values, white-on-white loss of silhouette, glowing bloom over the subject, removal of all shadows, broken or sketchy contours, tangled equal-weight hair lines, excessive flyaway strands, hair rendered as an undivided blurry color mass, watercolor bleeding across facial features, fuzzy eyes, fragmented lace, dense micro-detail, thick outlines, harsh black shadows, saturated colors, and background clutter.

Scene: A girl stands in shallow water at dusk, shown waist-up, looking directly at the camera, wearing a white dress; her head and upper torso occupy most of the frame.
```