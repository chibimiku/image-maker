## 总判断

v4 已经明显避开了旧 cel 固件的主要失败模式：没有把水彩/厚涂转成平涂，肤色、背景冷暖和局部材质基本保住；但它仍有两个问题：

1. **清线收益主要来自“重新解释”边界，而不是严格修复原始边界。**
2. **仍会凭模型先验补全或改写内容。** 最明显的是 say-hana v4 在腹部新增了一只手。这个问题比线条是否连续更严重，说明“不要改内容”的约束还没有成为最高优先级的可执行检查。

---

## 1. 图 3 / 图 6：头部还有多大改进空间

### say-hana

图 3 中，v4 的整体保真已经较高，头部 MAE 3.5、配色 0.9690 也印证了这一点。剩余问题主要是局部结构和笔触，不是大范围色彩错误。

具体差异：

- **睫毛**：v4 的上眼睑和睫毛边缘更规整、更连续，但部分睫毛被合并成较粗的暗线；源图中的睫毛数量、方向和断续感更自然。
- **虹膜**：v4 保留了棕绿色和高光位置，颜色基本正确；但虹膜内部的细碎色彩变化被简化，瞳孔与虹膜边界略偏“干净”。
- **眼睑**：v4 的上下眼睑连续性更好，但眼角有轻微加重，视觉上比源图更像清晰的动画线稿。
- **面部明暗**：v4 没有明显改坏脸部大关系，但脸颊、鼻梁和下颌的柔和过渡略被压平，源图的暖色水彩混色更丰富。
- **肤色**：整体非常接近源图；v4 的脸部肤色略更均匀、更亮，局部粉橙色变化稍少。

结论：say-hana 头部还有约 **小幅局部优化空间**，主要是恢复睫毛断续、虹膜内部纹理和面部柔和明暗；不应再增强整体清晰度。

### noir

图 6 的 v4 也保持了主要身份和配色，但差异比 say-hana 更容易在眼睛和冷色肤色上暴露。

具体差异：

- **睫毛**：v4 的上眼线较连续，但源图的睫毛更细、更碎、更融入蓝紫色笔触；v4 有轻微“线稿化”。
- **虹膜**：蓝色主色、高光和眼睛位置基本正确；但 v4 虹膜内部的青蓝渐变和细小纹理减少，瞳孔结构更简化。
- **眼睑**：v4 眼睑边缘更明确，尤其眼角处比源图更硬；源图眼睑部分由冷色阴影和半透明笔触共同构成，不是单一轮廓线。
- **面部明暗**：v4 保住了高亮脸部和紫蓝环境，但面部阴影被轻微平滑，源图的冷紫色局部过渡更有层次。
- **肤色**：v4 的肤色仍然偏冷、偏白，未发生严重暖化；但局部粉紫、蓝紫反射略少，面部更接近均匀浅色填充。

结论：noir 头部仍有比 say-hana 稍大的改进空间，但方向不是“增加线条”，而是保留**彩色、半透明、断续的眼部笔触**。

---

## 2. 最多 3 条可直接替换的文本改动

下面的改动建议优先解决“内容被模型擅自补出”和“眼部线稿化”两个问题。

### 改动 1：加入不可擅自补全内容的硬约束

**改位置：** 放在开头 `The input image is the single source of truth for content.` 后面。

```diff
 The input image is the single source of truth for content.
+Do not infer, invent, mirror, complete or repair any body part, hand, finger, limb, garment edge or object that is not visibly present in the input. If a region is ambiguous, occluded or cropped, preserve its ambiguity and preserve the visible pixels rather than completing it from anatomy priors.
 Unless there is a clearly visible defect ...
```

**预期影响：**

- say-hana：降低新增手、手指、衣物褶皱等内容错误；头部 MAE 可能不变，整体内容一致性应提高。
- 骨架段长：可能略降，因为模型不再用新增结构连接断点。
- 配色：基本不变。

**不达标时的含义：**

如果仍然出现新增手或新增肢体，说明问题不是文本优先级，而是 Gemini 的图像重绘机制会自动完成遮挡区域；继续加文字的收益很低，应改为局部重绘、遮罩或结构线后处理。

---

### 改动 2：限制眼睛只能做“像素级/局部保留式修复”

**改位置：** 替换现有 `PRESERVE THE SOURCE'S EYE DRAWING` 段落。

```diff
-3. Preserve the source's eye drawing. Do not replace it with generic anime eyes or standard cel eyes. Keep the eyelid shape, iris shape, pupil structure, lash count and direction, highlight positions, sclera colour and the eye-to-skin relationship.
+3. Preserve the source's eye drawing exactly as a local painted structure. Do not redraw, regularise, enlarge, sharpen or symmetrise the eyes. Keep the original eyelid thickness and breaks, iris contour and internal colour variation, pupil structure, lash count and direction, highlight positions, sclera colour, blur, translucency and eye-to-skin relationship. Eye lines must use local source colours and may remain soft or discontinuous; never convert them into a single clean dark contour.
```

**预期影响：**

- 眼部 MAE：应下降或至少不升。
- 头部配色：say-hana 维持 0.969 附近；noir 有机会从 0.8589 小幅提升。
- 骨架段长：眼部局部连续性可能下降，但这是有意的；不应为了增加眼线连通而牺牲眼部材质。
- 端点密度：眼睛区域可能上升，这是合理结果。

**不达标时的含义：**

如果头部 MAE 上升或眼睛变得模糊，说明“preserve exactly”触发了过度保守的复制/模糊，而不是改善局部修复。此时应删掉 `exactly`，保留“不要规则化、不要单一深色轮廓”两条限制。

---

### 改动 3：把主结构线和次级笔触明确分级

**改位置：** 替换 `LINE STRUCTURE WITHIN THE SOURCE'S OWN MEDIUM` 开头到 `Express...` 之前的部分。

```diff
-LINE STRUCTURE WITHIN THE SOURCE'S OWN MEDIUM
-While keeping the source's character, composition, features, materials and colour relationships unchanged, preserve and rebuild the MAIN CONTINUOUS EDGE STRUCTURE that already exists in the source.
+LINE STRUCTURE WITHIN THE SOURCE'S OWN MEDIUM
+While keeping the source's character, composition, features, materials and colour relationships unchanged, repair only the MAIN structural boundaries that are visibly broken. Use continuity only for the outer hair silhouette, major lock boundaries, jaw, neck-shoulder boundary, clothing silhouette and clear foreground occlusion edges. Do not force continuity through painterly texture, facial shading, eyelashes, iris texture, watercolour strokes or ambiguous overlaps.
+Treat secondary strokes as intentionally allowed to remain broken, translucent, tapered and irregular. A missing connection is not a defect unless both endpoints and the intended path are clearly visible in the source.
```

**预期影响：**

- 长线占比：预计维持或小幅上升，不应追求 cel 固件的 0.93–0.94。
- 骨架段长：可能从 say-hana 120.3 小幅提升，noir 133.6 小幅提升。
- 端点密度：可能下降，但下降幅度应小于旧 cel 固件。
- 头部 MAE/配色：应基本稳定，尤其能减少眼睛、脸部阴影被硬连的问题。

**不达标时的含义：**

如果段长不升、端点密度不降，说明模型没有把“主结构边界”转成可执行的空间选择；继续增加线条描述意义不大，需要使用边缘图、骨架图或局部遮罩作为额外输入。

---

## 3. 是否值得引入两阶段

### 不建议直接采用“全图两次 Gemini 重绘”

原因是第二阶段不是图像编辑器意义上的“只改材质与色彩”。如果把第一阶段输出再次作为输入，Gemini 仍可能：

- 改变脸部结构；
- 重画眼睛；
- 补全手或衣物；
- 改变构图和局部遮挡；
- 把第一阶段形成的结构再次解释成新的内容。

因此，全图两阶段很可能把一次重绘误差叠加成两次重绘误差。

### 值得采用的形式：结构阶段 + 受保护的局部材质阶段

只有在第二阶段具备**遮罩或强约束**时才值得做。

#### 第一阶段：结构连通

职责边界：

- 只处理明确断裂的主轮廓；
- 不改变颜色、明暗、材质、纹理和构图；
- 不处理眼睛内部、脸部阴影、虹膜纹理和水彩颗粒；
- 不补全不可见肢体或物体。

关键句：

```text
Repair only clearly broken primary structural contours. Do not alter colour, value, texture, brushwork, facial features, eye interiors, materials or hidden/ambiguous anatomy. If a contour path is not visibly determined by the source, leave it unchanged.
```

#### 第二阶段：材质与色彩还原

职责边界：

- 只恢复第一阶段可能压平的局部色彩和笔触；
- 不重新定义轮廓；
- 不重新绘制眼睛和脸型；
- 最好只对脸部、头发、衣服等指定区域做局部重绘。

关键句：

```text
Restore the source's local colour mixing, translucency, value transitions, brush texture and material variation without changing any contour, feature position, anatomy, pose, occlusion or composition. Do not add or remove strokes; restore only the appearance of existing regions.
```

如果没有区域遮罩、图层合成或像素级锁定，我不建议上线全图两阶段。当前 v4 的单阶段风险小于两次自由重绘。

---

## 4. 参数层建议

### 分辨率

- **1K**：适合快速验证提示词；细睫毛、虹膜纹理和细线容易被吞掉。
- **2K**：建议作为默认。通常是保真、稳定性和成本的最佳平衡。
- **4K**：不一定提高语义保真。它可能增加细节，但也可能增加模型“凭先验补细节”的机会，导致更多发丝、睫毛、衣褶和手部幻觉。

建议：**先固定 2K 做提示词 A/B；只有局部细线确实被分辨率限制时，再测试 4K。**

### 输出格式

固定 JPEG 不利于评估，尤其会影响：

- 细线端点；
- 水彩颗粒；
- 高光边缘；
- 颜色偏差；
- 端点密度和长线占比。

建议：

- 中间产物和指标评估使用 **PNG**；
- 最终交付需要 JPEG 时，再单独压缩；
- JPEG 质量至少固定为 95，并确保所有版本使用相同编码参数。

PNG 不会让 Gemini 本身更“懂图”，但会避免把重绘后的差异污染到指标中。对当前任务，**值得改 PNG**。

### `aspect_ratio=auto`

不建议依赖 `auto`。

虽然 v4 文本已经要求保持比例，但 `auto` 可能由接口或模型重新决定输出尺寸，造成：

- 非整数缩放；
- 边缘裁切；
- 主体位置轻微漂移；
- 不同样本之间不可比。

建议显式传入源图比例，例如：

```text
aspect_ratio = source_width : source_height
```

并在输出后做尺寸、比例、裁切位置校验。

### 是否固定比例

应固定。比例、裁切和主体位置是保真指标的一部分，不能只靠文本约束。

建议流水线固定：

1. 输入先按目标比例准备；
2. API 显式指定相同比例；
3. 输出检查宽高比；
4. 禁止自动裁切；
5. 若接口无法保证原尺寸，最后只做无内容变化的像素级缩放。

### 参数优先级

对本项目的实际影响大致是：

1. **遮罩/局部重绘能力**：最大；
2. **PNG 和统一编码**：对评估可靠性影响很大；
3. **固定 aspect ratio**：对构图保真很重要；
4. **2K vs 4K**：主要影响细节稳定性，不会根本解决重绘幻觉；
5. **1K vs 2K**：对眼睛和细线有明显影响。

---

## 5. 一句话结论

**在必须使用 Gemini 且要求高保真的前提下，v4 已接近全图文本重绘的上限；剩余提升主要应靠“禁止不可见内容补全 + 眼部局部保护 + 遮罩式局部重绘或结构线后叠加”，而不是继续增强全图清线提示。**