## 1. 图 3 / 图 4 的实际改善

### 过白

**图 3：解决了过曝式过白，但校正过头。**

- 人脸、头发、衣服终于有稳定的明暗分层，不再像图 2 那样大面积接近纯白。
- 白裙褶皱可以辨认，人物与背景也分开了。
- 新问题是整体偏灰紫，尤其头发暗面、裙子和前景水面。参考图虽然明亮，但依靠的是“高明度下仍有局部对比”，不是把全图压暗。
- 图 3 的白底占比从参考的 `0.667` 降到 `0.228`，说明它丢掉了参考图最重要的“大面积干净留白”。纸纹、水波和衣褶也铺得过满。
- 暗部仍然不足：`<60` 只有 `0.001`。它不是缺少中间调，而是缺少少量、明确的深色锚点。参考图的深色领口、裙子、袜子承担了视觉骨架。

**图 4：亮度比图 3 更接近参考，但局部又开始泛白。**

- 人脸和头发仍有层次，没有退回图 2 的全面漂白。
- 白裙胸口、手臂和右侧水面高光接近融合，局部边界变弱。
- 饱和度 `18.3` 偏低，综合色彩比参考更粉灰、更同质。
- 构图裁得太满：头发贴近顶边，手臂和裙子被底边截断，几乎没有参考图那种留白和呼吸空间。

结论：**明度问题从“失控过曝”变成了“可控但层次策略不准”**。下一步不是继续压暗，而是恢复高明度基调，同时保留少量深色锚点和主体局部对比。

### 线条和发丝

**图 3：基本解决。**

- 脸部轮廓、下颌、眼睛、肩部和衣服边界明显连续。
- 头发已形成大束、中束、少量游丝三级组织，比图 2 清楚很多。
- 问题是游丝仍偏多，左侧和肩部附近有不少相互穿插的长线。
- 裙子蕾丝、褶皱、水纹和发丝同时争夺边缘预算，导致边缘密度 `0.114`，比参考高约 72%。画面清楚，但不够克制。

**图 4：主轮廓连续，但内部发丝组织退化。**

- 外轮廓没有明显糊成团，主要发束也能读懂。
- 头顶出现较宽、较硬、近似平行的条带，发丝更像逐条描画，缺少图 3 的柔和分组。
- 面部、胸口和白裙边界因明度过近而变弱。
- medium 质量也会减少细线稳定性，所以不能把差异全部归因于精简提示词。

总体判断：**连续性已改善；当前主要问题已经不是“断线”，而是边缘过量、发丝层级不足，以及该弱化的纹理没有弱化。**

## 2. 图 3 与图 4 的差距

没有看到你的 2900 字符原文和精简前文本，无法严格判定具体删掉了哪一句；而且这次同时改变了 `quality`、随机采样和构图，不能做单变量归因。

从结果反推，最可能弱化的是以下三类约束：

1. **大、中、小三级细节的数量关系**
2. **高明度不等于无阴影，必须保留局部深色锚点**
3. **负空间和主体占画比例，而不只是“头与上身更大”**

建议保留的短写法：

```text
Hair: a few large locks, fewer inner strands, only sparse flyaways; no strand-by-strand striping.
```

```text
Keep a luminous high-key image, but preserve midtones and a few small dark anchors; never wash white fabric, skin, and sky into one value.
```

```text
Detail budget: strong silhouette, selective internal lines, sparse microtexture; simplify water, lace, and background.
```

```text
Waist-up with breathing room around the head and shoulders; do not crop the figure tightly.
```

其中最不能丢的是 **detail budget**。它同时控制发丝、衣褶、水纹和边缘密度，比单独写“high detail”或“clean line art”有效。

## 3. 统计口径是否适合作为目标

这些统计量**可以做故障报警，但不应直接作为优化目标**。

原因是：

- 全局亮度和白底比例高度受构图影响。参考图有大量纯白背景，而图 3、4 几乎全画幅是人物和海水，数值天然不可比。
- 边缘密度不区分有效轮廓、衣褶、纸纹、水纹和噪声。图 3 的边缘更多，但其中很多确实提高了可读性。
- 固定阈值 `>235`、`<60` 对色调映射、输出尺寸和压缩非常敏感。
- 参考图与生成图内容、姿势、背景不同，直接比较全图像素分布主要测到的是构图差异，不是画风差异。

建议改成以下自动指标：

1. **分区亮度层次**  
   先用人物分割和人脸/头发/服装解析，分别计算亮度 `P10/P50/P90`、高光裁切率及局部 RMS 对比度。不要比较全图均值。

2. **主体与背景分离度**  
   在人物轮廓两侧采样，计算边界内外的 `ΔL* + ΔE00`。它比全局边缘密度更接近“人物是否清楚”。

3. **多尺度边缘匹配**  
   对参考和结果统一尺寸，分别统计粗、中、细三个尺度的 edge energy 占比。目标是匹配比例，而不是最大化总边缘数。

4. **线条连续性**  
   对线稿响应图骨架化，统计平均连通段长度、短碎线比例、端点密度、分叉密度。短碎线越多，通常越接近“糊/断/乱”。

5. **负空间指标**  
   对低纹理高明度区域做连通域分析，计算最大干净区域面积、背景纹理熵和主体占画比。这比 `>235` 更能描述参考图的留白。

6. **感知风格距离**  
   使用 DINOv2/CLIP 特征和多层 Gram matrix；在人脸、头发、服装、背景四个区域分别比较。不要单独依赖全图 CLIP。

推荐最终评分：

```text
30% 分区明度与局部对比
25% 多尺度边缘比例
20% 线条连续性
15% 负空间与主体占比
10% 分区感知风格距离
```

## 4. 最终版提示词

以下保持 8 个字段，约 1500 个英文字符，低于 1800：

```text
1. Reference
Use the reference only for visual language: luminous anime illustration, clean elegant contours, restrained watercolor texture, airy white space, selective color accents. Do not copy its character, pose, clothes, or objects.

2. Scene
A girl stands waist-up in shallow water at dusk, looking directly at the camera, wearing a white dress.

3. Composition
Portrait orientation, waist-up framing. Make the head and upper body prominent but leave clear breathing room above the hair and beside both shoulders. Keep the pose readable; avoid tight cropping, oversized bust, or hidden arms.

4. Style
Polished Japanese anime illustration with graceful tapered linework and soft watercolor-like fills. Bright, delicate, and graphic rather than photorealistic. Keep the background spacious and simplified.

5. Light & color
High-key dusk light with pale pink, cool blue, and small warm accents. Preserve midtones and a few compact dark anchors around eyes, hair roots, dress folds, and water contact. White dress, skin, sky, and reflections must remain distinct; no blown highlights, gray veil, or uniform pastel wash.

6. Lines
Use continuous, confident contours for face, jaw, eyes, hair silhouette, shoulders, dress, and waterline. Lines may soften in light but must not dissolve, double, fray, or merge into texture. Prioritize silhouette and facial features over internal decoration.

7. Hair
Build hair as a few large flowing masses, then fewer medium locks, with only sparse fine flyaways. Each lock follows one clear root-to-tip flow. Avoid strand-by-strand striping, tangled loops, parallel grooves, and dense crossings over the face.

8. Detail budget
Three scales only: large silhouette and value masses; selective medium folds, locks, and ripples; very sparse microdetail and sparkles. Keep face and eyes crisp, hair moderately detailed, dress controlled, water and sky simple. Preserve clean negative space; no paper grain everywhere, excessive lace, foam, ripples, or edge clutter.
```

## 5. 取舍顺序

1800 字符内可以兼顾四项，不需要主动牺牲。若还要继续压缩，顺序应是：

1. **先保明度层次**：尤其“高明度 + 中间调 + 少量深色锚点”，否则会直接退回图 2。
2. **再保细节预算**：它决定边缘是否失控，也间接保护参考图的留白。
3. **再保发丝分组**：大束、中束、少量游丝是最短且最有效的表达。
4. **最后保线稿连续性**：可压缩成一句，因为模型通常已经能理解 clean continuous tapered contours。

优先牺牲背景色彩描述、材质形容词和装饰物枚举。不要牺牲构图留白；否则即使局部画风正确，统计和视觉上也永远无法接近参考图。