## ① 保留什么

### 保留

1. **短提示词 + 参考图**
   - 你现在的数据已经证明：600～900 字符附近明显优于 11k 长说明书。
   - 长说明不是“信息更多所以更好”，而是在和参考图、主体描述争夺控制权。
   - gpt-image 对风格更适合“少量高信息密度约束 + 视觉参考”。

2. **明确区分风格来源和主体来源**
   - 继续保留“风格来自 Image 1，主体和构图来自文字”的意图。
   - 但不要把这种区分写成很长的解释。

3. **明确要求生成新图**
   - 参考图在 `/images/edits` 中不是纯粹的 style token。它是输入图，模型天然可能编辑、复用、继承其中人物和构图。
   - “new image / new subject / new composition”仍然值得保留。

4. **保留参考图**
   - 你的 G/V 实验已经说明，短文本无图时只能复现部分风格概念，挂图后亮度、饱和度、留白和边缘密度更接近目标。
   - 参考图不是可选增强，而是当前方案的主要风格锚点。

5. **主体描述应该短而具体**
   - “少女站在浅水、黄昏、半身、看镜头、白裙”这种描述适合测试。
   - 不要把画风说明书和主体说明书混成一段叙事文本。

---

## ② 该改什么

### A. 直接替换为更短、更像图像任务指令的模板

建议先用英文模板做主实验。不是因为模型不理解中文，而是图像生成训练语料中这类图像编辑指令通常以英文结构出现。

```text
Create a new image.

Use Image 1 as a visual style reference only:
- watercolor-like transparent washes
- very light, low-saturation colors
- bright high-key lighting
- soft delicate edges
- abundant clean white negative space
- restrained fine linework
- airy, quiet, elegant atmosphere

Do not reproduce the person, pose, objects, camera angle, or composition from Image 1.

Scene:
<主体描述>

The scene description controls the subject, pose, clothing, framing, perspective, and composition.
The reference image controls only the visual style, color treatment, lighting quality, mark-making, and overall degree of detail.
```

这个版本比你当前模板更适合做两件事：

- 把风格拆成**可观察的视觉属性**；
- 把主体控制权单独绑定给 `Scene`。

你的原模板里 `Keep from Image 1` / `Change from Image 1` 逻辑是对的，但可以进一步压缩。`Change from Image 1` 仍然让模型先建立“要从图中改变什么”的编辑关系，可能增强输入图继承。建议改为：

```text
Image 1 provides visual style only.
The new scene below provides all content and composition.
```

### B. 推荐版本：避免过强的“复制/禁止”语气

```text
Create a new illustration in the visual style of Image 1.

Image 1 is a style reference, not the subject reference.
Match its:
- color palette and saturation
- brightness and amount of white space
- brushwork and edge softness
- line quality
- lighting mood
- level of simplification and detail

Use only the following scene for the content:
<主体描述>

Use the scene description for the character, action, clothing, setting, framing, camera angle, and composition.
Create a different character and a different composition from Image 1.
```

### C. 关于否定句

不要完全迷信“禁止否定句”。问题不是否定句本身，而是：

```text
Do not copy anything from the reference image.
```

太宽，会和“使用参考图风格”冲突。

更好的是：

```text
Image 1 is a style reference, not a subject or composition reference.
Create a different character and composition.
```

这是**角色分工声明**，比泛化的 `Do not copy` 更稳定。

仍然建议保留一条具体否定约束，尤其当参考图角色特征很明显时：

```text
Do not reuse Image 1's character identity, hair color, eye color, costume, or pose.
```

但不要连续堆十几条禁止项。

### D. 不建议保留的内容

从 11k 说明书里优先删除：

- 风格历史、流派来源、艺术家背景；
- 同义词堆叠；
- “必须、绝对、严格、不可偏离”等元指令；
- 结构性解释，例如“这一层风格会影响另一层风格”；
- 过多具体装饰名词；
- 与当前主体无关的示例场景；
- 负面提示词列表；
- 对模型行为的解释。

保留四类信息：

1. 色彩：亮度、饱和度、主色、白色占比；
2. 笔触：水彩、干湿、边缘、线稿；
3. 光感：高调、逆光、柔光、雾感；
4. 画面密度：留白、简化程度、细节密度。

你可以把风格描述压成 100～250 字，剩余字符给主体。

---

## 参考图怎么用

### 1. 整图 vs 裁剪风格区域

优先测试以下三种，而不是简单地“整图或裁一块”：

#### A. 原始整图

适合保留：

- 全局配色；
- 明暗分布；
- 留白比例；
- 构图密度；
- 画面整体气氛。

缺点是容易继承：

- 人物脸；
- 发色和眼睛；
- 姿势；
- 物件；
- 构图。

#### B. 只保留风格纹理的裁剪图

裁掉人物和可识别物体，只保留：

- 大面积笔触；
- 色块；
- 白底；
- 局部线条；
- 水彩晕染。

这通常是你最值得测试的方向。它可以减少“编辑原人物”的诱因，但会失去一部分全局构图和留白统计。

建议裁剪图保留原始比例附近，不要做成极端横条。可以制作：

- 中央 60%～80% 的无主体区域；
- 左上或右下的笔触区域；
- 2～4 个局部风格块拼成一张“style board”。

#### C. 风格板

把同一张图中的 3～4 个局部区域拼成一张新图，之间用白色间隔分开，并在提示词中说：

```text
Image 1 is a style board. Use its palette, brushwork, line quality, and lighting treatment.
Do not use the individual patches as objects or composition references.
```

这可能比单独裁一块更稳，因为模型同时看到：

- 颜色；
- 笔触；
- 线条；
- 留白。

但它不是官方保证的工作流，需要实测。

### 2. 不建议加白边

除非白边本身就是画风的一部分，否则加白边会把白色背景统计强化成图像内容，模型可能生成：

- 更大的白底；
- 类似海报边距；
- 不必要的边框；
- 居中构图。

你的目标本来就有大量留白，因此白边可能会提高 HSV 相似度，却不一定提高真实风格相似度。应分别测试“原图”和“无主体风格裁剪”，不要把加白边作为默认预处理。

### 3. 不建议主动降分辨率

降分辨率可能抹掉：

- 线条质量；
- 水彩边缘；
- 纸张纹理；
- 小面积颜色关系。

如果原图已经远高于接口需要，可以生成一个尺寸合理的副本，但要保持：

- 纵横比；
- 色彩；
- 细节；
- 不进行明显 JPEG 压缩。

1000×1442 本身不需要降到很低。

### 4. 多张参考图

多张图不是必然更好。

优势：

- 可减少单张图中人物身份、构图和偶然物件的影响；
- 可以覆盖同一画风的多个色调、笔触和构图；
- 对“风格定义不完整”的单张样图有帮助。

风险：

- 风格不一致时，模型平均化；
- 多个人物图会混合脸、发型、服装；
- 多张图中的主体共同变成一种“内容先验”；
- 模型无法严格知道哪张图的哪部分是风格、哪部分是主体。

如果试多图，建议使用：

- 2～3 张同一风格；
- 不同主体；
- 相近色调；
- 至少一张无人物或主体很弱的图；
- 提示词明确标注：

```text
Image 1, Image 2, and Image 3 are visual style references only.
Use their shared palette, brushwork, line quality, lighting, and level of detail.
Do not combine or reproduce their subjects, characters, poses, or compositions.
```

不要把“风格参考图”和“主体参考图”混在一起，除非你确实希望模型编辑主体。

---

## 参考图与 multipart 顺序

目前没有公开约束表明 `/images/edits` multipart 中：

- 先传图后传 prompt；
- 先传 prompt 后传图；
- multipart 字段排列顺序；

会改变模型的语义权重。

HTTP multipart 字段顺序通常只影响服务端解析顺序，不应当被当成 prompt priority 机制。官方也没有公开“先图后文更强”或“先文后图更强”的行为保证。

真正有意义的是：

1. `prompt` 中如何命名图像；
2. 是否明确 `Image 1`；
3. 输入图在请求中是否确实被映射为 Image 1；
4. 是否有多张图时稳定使用 Image 1 / Image 2；
5. `/images/edits` 的输入图是否天然被视为待编辑内容。

你的 B5 长出参考图人物，是最重要的证据：**即使你在 prompt 里称它为 style reference，编辑接口仍然可能把它当作视觉内容输入，而不是纯 style embedding。**

建议不要测试 multipart 字段排列，收益很低。若要验证，只做 2～3 个配对请求，并固定：

- 同一图；
- 同一 prompt；
- 同一 seed 条件（如果接口支持）；
- 同一参数。

但不要把它作为主要优化方向。

---

## 参数建议

### `quality=medium`

不要先假定 medium 是问题来源。

你当前差异主要是：

- 长提示词 vs 短提示词；
- 有图 vs 无图；
- 参考图主体泄漏；
- 文本与视觉风格竞争。

这些都比 `quality` 更可能决定结果。

`quality=high` 可能增加：

- 小线条；
- 纹理；
- 局部边缘；
- 复杂服装细节；
- 渲染时间。

但它未必提高“风格贴合”，甚至可能降低水彩风格中的：

- 简化；
- 留白；
- 透明度；
- 低细节感。

建议只做一个小型 A/B：

- medium/high；
- 同一短 prompt；
- 同一参考图；
- 同一主体；
- 至少 4 个主体。

### `size`

尺寸主要影响：

- 构图方向；
- 画面布局；
- 可表达的空间；
- 细节分配。

不应把尺寸当成主要画风参数。

但要注意：

- 参考图是竖图；
- 输出也是 1024×1536；
- 这会减少模型对构图的重新解释。

如果你换成 1536×1024，模型可能改变主体布局和留白分布，导致 HSV/白底指标变化，不一定是风格变差。

建议固定产品默认比例，不要为了风格专门换方向。除非用户主体天然适合横构图。

### `background`

如果接口允许透明背景，画风测试应固定为：

```text
background: opaque
```

透明背景会影响：

- 白底占比；
- 颜色统计；
- 边缘表现；
- 水彩晕染的背景扩散。

你当前目标有大面积白色留白，建议使用不透明背景，并在 prompt 里写：

```text
Use a clean, pale background with generous white negative space.
```

不要同时让 `background` 参数和文字对背景做互相冲突的要求。

### `output_format`

PNG 更适合评估：

- 水彩边缘；
- 低饱和区域；
- 细线；
- 透明/半透明感；
- 白色留白。

JPEG 会引入压缩伪影，尤其会影响边缘密度和颜色统计。产品若最终需要 JPEG，可以最后一步转码，不要用 JPEG 作为风格实验格式。

---

## 额外方向：优先级排序

### 第一优先级：风格裁剪/风格板

这是最直接解决“参考图人物被搬过来”的方案。先不要做两步生成，先改输入图。

### 第二优先级：风格说明书自动转写

可以做一次离线预处理，但不要让模型直接总结成散文。

要求转写结果输出固定字段：

```text
Palette:
Lighting:
Brushwork:
Linework:
Edges:
Texture:
Composition density:
Negative space:
Detail level:
Avoid:
```

然后再压缩成 100～250 字的风格段落。

关键是让转写模型只描述**视觉统计和技法**，不要描述：

- 图中人物身份；
- 发色；
- 服装；
- 姿势；
- 场景物件；
- 构图故事。

示例：

```text
Describe only the reusable visual style of this image.
Do not describe its subject, character identity, objects, pose, or composition.
Return fields for palette, brightness, saturation, brushwork, linework,
edge softness, lighting, texture, negative space, and detail density.
```

这不能替代参考图，但可以减少不同主体下的漂移。

### 第三优先级：两步走

“先生成主体，再用 edits 套风格”不应作为首选。

原因：

- 第二步仍然把第一张图当作编辑输入；
- 第二步可能保留主体但无法稳定改掉原始渲染语言；
- 两次生成增加身份漂移和构图漂移；
- 风格可能变成“在已有图上做滤镜”，而不是重绘。

两步走值得测，但应把第二步 prompt 写成：

```text
Re-render the input image as a new illustration.
Preserve the subject identity, pose, clothing, camera framing, and composition.
Replace the rendering completely with the visual style described below:
<style description>
Do not retain the original medium, texture, color treatment, or rendering style.
```

这更像重绘，不是简单“apply style”。

---

# ③ 下一批实验：建议做 12 个

固定：

- 同一模型；
- `quality=medium`；
- 1024×1536；
- PNG；
- opaque background；
- 4 个主体；
- 每个变体至少 2 个主体，最好 4 个主体；
- 不要混用不同尺寸或模型。

如果只能做 8 个，优先做前 8 个。

| 编号 | 变体 | 做法 | 预期 |
|---|---|---|---|
| 1 | 当前基线 | 原始整图 + 678 字模板 | 作为可比基线，预计风格稳定性中等 |
| 2 | 极短英文 | 原始整图 + 150～250 字固定字段风格描述 | 可能略低于完整短版，但主体泄漏更少 |
| 3 | 角色分工版 | 使用上面的 `Image 1 provides visual style only` 模板 | 预计比 Keep/Change 更少继承人物和构图 |
| 4 | 具体身份排除 | 变体 3 加 `hair color, eye color, costume, pose` 四项具体约束 | 预计减少粉发金瞳等身份迁移 |
| 5 | 无主体裁剪 | 裁掉人物，只留笔触、色块、白底区域 | 预计主体泄漏显著下降，局部风格贴合可能上升 |
| 6 | 风格板 | 3～4 个无主体局部拼图 + 变体 3 模板 | 预计颜色和笔触更稳，但全局留白不一定更稳 |
| 7 | 多图同风格 | 2 张不同主体的同风格图 + 明确共享风格描述 | 预计降低单图人物继承，测试是否出现风格平均化 |
| 8 | 多图风格板 | 2 张无主体风格板或纹理图 | 预计最少身份泄漏，但可能失去画面整体气氛 |
| 9 | medium/high | 变体 3，medium 与 high 各半 | 测试细节增加是否损害低饱和、透明水彩感 |
| 10 | 纯文本控制 | 无参考图，150～250 字风格字段描述 | 测量参考图本身的增益，不作为产品方案 |
| 11 | 风格转写 | 先把参考图转写成结构化风格字段，再生图 | 测试离线 caption 是否能稳定跨主体复用 |
| 12 | 两步重绘 | 第一步生成主体；第二步 edits 重绘为目标风格 | 预期主体保留较好，但耗时、成本和风格一致性风险更高 |

### 最有价值的 8 个组合

若只能做 8 个，我建议：

1. 当前基线；
2. 极短英文；
3. 角色分工版；
4. 角色分工版 + 具体身份排除；
5. 无主体裁剪；
6. 风格板；
7. 多图同风格；
8. high vs medium。

不要把实验名写成 `head / priority / interleave` 继续比较。对 gpt-image 来说，更关键的变量是：

- 输入图内容；
- prompt 长度；
- 风格与主体的字段分离；
- 单图或多图；
- 是否有可识别主体。

---

## 量化口径

你现在的 HSV 余弦可以保留，但不能作为唯一指标。它明显会奖励“生成白底”，不一定奖励真正的画风。

每张图至少记录以下指标：

### 1. 风格指标

- 亮度均值；
- 饱和度均值；
- 白/近白像素占比；
- 边缘密度；
- 颜色直方图距离；
- 参考图与输出的 CLIP/image embedding 相似度。

最好增加一个**风格裁剪相似度**：

- 从输出中取不含主体的背景/笔触区域；
- 与参考图对应区域比较；
- 避免主体颜色主导统计。

### 2. 主体指标

人工 1～5 分：

- 主体是否正确；
- 动作是否正确；
- 服装是否正确；
- 镜头和画幅是否正确；
- 是否出现参考图人物特征；
- 是否出现参考图物件或构图。

其中“参考图人物泄漏”应该单独计分：

```text
0 = 无可识别继承
1 = 轻微相似
2 = 明显继承一个特征
3 = 同一角色感
```

### 3. 风格人工评分

由不了解实验编号的人盲评：

```text
0 = 完全不像
1 = 只有色调相似
2 = 色调和部分笔触相似
3 = 整体画法相似
4 = 明显属于同一画风
5 = 高度一致
```

### 4. 产品效用分

不要只看风格分。建议用：

```text
utility = 风格评分 × 主体正确率 × (1 - 身份泄漏率)
```

或者更简单地单独报告三项：

- style score；
- subject score；
- leakage score。

不要把白底占比直接当作风格贴合度。你的数据里已经有可能出现“统计指标变好但主体和真实画法变差”的情况。

---

## ④ 明确风险与不确定性

1. **`/images/edits` 不是纯风格迁移接口**
   - 这是当前最大结构性限制。
   - 参考图会同时携带内容、人物、构图和风格。
   - Prompt 不能保证把它拆成纯 style embedding。

2. **`Image 1` 不是官方意义上的风格权重开关**
   - 点名 Image 1 有助于多图消歧，但不会强制模型只取风格。
   - 它是语义引用，不是控制参数。

3. **multipart 顺序大概率没有可利用的优先级效果**
   - 没有官方文档支持“先文字后图片”会增强文字。
   - 不值得作为主要研发方向。

4. **多张图可能提高风格稳定性，也可能增加内容混合**
   - 只有在多张图主体差异大、风格高度一致时才值得尝试。
   - 多张“同一个人”的样图很可能强化身份泄漏。

5. **HSV 指标容易被背景操纵**
   - 白底、低饱和并不等于水彩。
   - 必须加入人工风格评分、主体正确率和身份泄漏率。

6. **模型版本差异很大**
   - 你已经观察到 2、2.5-flare、2.5-sunburst 不同。
   - 不能把 gpt-image-2 的结论直接外推到所有 2.5 变体。
   - 产品默认应固定模型版本，不要用“gpt-image 系列”作为一个统一行为假设。

7. **两步走可能解决主体冲突，但不一定解决风格冲突**
   - 它更可能增加成本和延迟。
   - 只有当单步输入图持续强烈继承参考人物时，才值得进入第二轮。

### 产品默认建议

目前可以固化，但默认模板应改为：

- 150～700 字；
- 风格字段化；
- `Image 1 provides visual style only`；
- 主体单独放在 `Scene:`；
- 只保留少量具体身份排除；
- 参考图优先使用无主体裁剪或风格板；
- PNG、opaque、medium；
- 固定输出比例；
- 不使用 11k 全量说明书；
- 不把 multipart 顺序当作控制手段。

最值得押注的组合是：

> **无主体风格裁剪/风格板 + 150～300 字结构化英文风格描述 + 明确的 Image 1 角色分工 + 简短主体描述。**

