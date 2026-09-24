## 1. tid：默认固件确实改写了画风

**图 3 不是单纯修复图 2，而是把它重新解释成了标准硬边赛璐璐插画。**

- **线**：人物外轮廓、发束、下颌、手臂和服装边缘被统一成较粗、较深、近似等线宽的连续描边。图 2 原本是细线、局部弱化甚至融入色面的处理；图 3 把“线条存在感”提升成主要视觉结构。
- **色**：大面积颜色被归并成更少的色阶。头发内部的粉色变化、皮肤暖冷过渡、白裙受环境光影响的细微色偏都被压平。配色相似度仍有 `0.952`，说明主色没大改，但不代表用色方式没改。
- **明暗**：图 2 的柔和渐变、透光感和环境反射，被替换成边界明确的块状阴影。尤其头发暗面、锁骨、胸口和裙褶，明显符合“一层硬边阴影”的指令。
- **质感**：图 2 尚有轻微纸感、柔和颗粒和空气透视；图 3 表面更干净、塑料化、矢量化，水面和云层也被归纳成平面色块。

所以默认版指标中平均骨架段长从 `140.4` 升到 `266.1`、端点密度降到 `8.90`，部分是因为**线被加粗、简化和合并**，不能全部解释成缺陷修复。

**图 4 保住画风的程度明显更好，但不是完全无损。**

它保住了：

- 较细、较浅的轮廓；
- 更柔和的肤色和头发渐变；
- 白裙受夕阳影响的粉、紫、蓝灰色变化；
- 背景与人物相近的柔和渲染语言。

还欠缺：

- 图 2 的微颗粒、细碎笔触和局部透明感仍被清理掉了一部分；
- 头发和裙褶仍有一定程度的色阶归并；
- 面部被“标准美型化”，眼睛、脸型、嘴部细节并非严格保真；
- 更严重的是，图 3、图 4 都相对图 2 发生了**重新取景和人物放大**。这已经违反“构图保持不变”，不能算纯渲染增强。

严格说，图 2 本身也不是图 1 那种典型“清透水彩细线”，而是偏柔和数字插画。中性固件保住的是**图 2 的当前渲染语言**，不是把它拉回图 1。

## 2. say-hana：默认版损失明显，中性版基本救回

图 7 损失了四项核心特征：

- **水彩边缘**：图 6 的湿画法扩散、不规则色边和局部颜料堆积被消除。
- **纸面与颗粒**：背景、皮肤、裙子上的纸纹和颜料颗粒大幅减少。
- **色彩关系**：橙、黄、青绿原本互相渗透，图 7 变成分区明确的平涂配色；饱和度从 `103.9` 降到 `86.5`，配色相似度只有 `0.711`。
- **笔触节奏**：海水、云、裙褶由破碎而有方向性的笔触，变成轮廓封闭、内部填色的动画背景和赛璐璐服装。

图 8 **大体救回了水彩媒介感和暖橙绿关系**。`0.972` 的配色相似度和 `101.4` 的饱和度与肉眼判断一致，不只是“颜色接近”，橙绿互染、浅色留白和局部松散边缘也回来了。

仍然存在的主要损失是：

1. **纸张颗粒和真实颜料扩散不足**。图 8 更像“数字水彩滤镜”，没有图 6 那种细密、随机的纸纤维与干湿变化。
2. **局部细节被概括**。发饰、碎发、裙边和水面小笔触比图 6 更整齐、更少。
3. **线条仍略被规整**。尤其脸部、手臂和主发束，轮廓比图 6 更连续光滑。
4. **人物身份和细节有轻微重生成**。眼形、刘海分束、发饰结构并非逐像素修复。

因此结论是：中性版把最重要的色彩和水彩语言救回来了，但仍牺牲了**材料级微纹理与随机笔触**。

## 3. 线稿修复效果

### 轮廓连续性

两组都有明确改善，指标也支持这一点。

- tid 默认版改善最强，但代价是粗线化和赛璐璐化。
- tid 中性版从 `140.4 / 18.90 / 0.913` 改到 `225.6 / 9.90 / 0.960`，已经足以消除大部分断线和短碎线，收益更均衡。
- say-hana 中性版也有实质改善，但控制得更保守：段长 `141.3`、端点密度 `14.98`、长线占比 `0.899`。
- 默认版的数值最好看，但它通过重画和简化获得了部分提升，不等价于“更忠实地修好”。

### 发丝分组

- 图 2 到图 3：分组最清楚，但发束数量减少，粗细趋同，细碎飘发被合并，属于“重新设计后变清楚”。
- 图 2 到图 4：主次发束更易读，长线连接更稳定，同时保留了更多细发，是 tid 中更合适的结果。
- 图 6 到图 7：发束高度图形化，黑色大块与硬边高光非常明确，但水彩发丝层次丢失。
- 图 6 到图 8：主发束和外轮廓得到整理，内部仍保留橄榄绿、金色反光和柔化边缘，综合最好。

### 手部结构

改善只有**中等程度**，不是可靠的解剖修复：

- 图 2 的双手本来已有可辨识的掌指关系，但指节、入水遮挡和部分手指长度不稳定。
- 图 3/4 把轮廓画得更完整，指尖和水面交界更干净；但可见手指数、遮挡逻辑、关节弯折仍有模型式简化。图 4 的手没有达到可作为解剖校正样本的程度。
- 图 6 到图 7/8 同样主要改善了“线是否闭合”和“轮廓是否顺”，并没有严格恢复每根手指的三维结构。图 7 的硬描边反而把不准确的手指结构强调出来；图 8 隐藏得更自然，但不代表结构更正确。

Gemini 仍难稳定修好的部位包括：被水面遮挡的手指、交叠发丝与肩带、发饰链条、小型褶边、发丝穿插关系，以及原图中没有充分视觉证据的关节。它会进行合理化重生成，而不是局部几何求解。

## 4. 固件定稿建议

选择 **（c）自动判断／两套可选**，但实现上应当是：

- **画风中性版作为默认路径**；
- 只有输入明确属于硬边赛璐璐，或用户明确要求 cel rendering 时，才追加 cel 专用后缀；
- 不建议让模型在同一段提示中自行决定是否套用 cel 指令。应由上游分类或用户选项决定，便于测试和追责；
- 分类置信度不足时，一律走中性版。

只用默认 cel 固件会系统性破坏水彩、厚涂、铅笔、粉彩等输入。只保留中性版虽然比现状安全，但对确实需要统一硬边动画成稿的项目，约束可能不够强。因此应保留两种模式，默认中性。

### 建议的中性固化文本

```text
You are performing a conservative restoration pass on the FIRST input image.

PRIMARY RULE: preserve the first image's existing visual language. Infer its
current medium and rendering method from the image itself, including line
weight, edge softness, brushwork, texture, granulation, colour transitions,
shadow style, and level of detail. Improve the image only within that same
visual language.

Do not convert between media or rendering styles. In particular, do not turn
watercolour, painterly, textured, soft-shaded, sketch-like, or mixed-media
rendering into cel shading, flat fills, vector-like shapes, or bold inked
outlines. Do not make outlines darker, thicker, more uniform, or more numerous
than they are in the first image.

LOCKED CONTENT:
- Preserve the exact canvas aspect ratio, framing, crop, camera distance,
  composition, pose, anatomy, silhouette, facial identity and expression.
- Preserve clothing design, accessories, hairstyle, background layout,
  lighting direction, palette, saturation, and all intentional details.
- Do not zoom, recrop, reposition, redesign, beautify, simplify, or add/remove
  objects or decorative elements.

ALLOWED CHANGES ONLY:
- Repair accidental broken, doubled, stray, banded, or colour-bleeding edges.
- Reconnect contours only where the intended continuation is unambiguous.
- Clarify existing hair groups and shape boundaries without merging,
  inventing, or thickening them.
- Correct clearly malformed local anatomy only when the surrounding image
  provides sufficient evidence, especially fingers and joints.
- Remove generation artefacts, compression artefacts, unintended smears,
  halos, and inconsistent micro-details.
- Improve local rendering consistency and finish while retaining the original
  brush texture, edge variation, translucency, grain, and colour variation.

When uncertain, preserve the source pixels and structure rather than
redesigning them. A smaller correction is preferable to a stylistic or
structural change.

Output only one restored image. Do not include text, borders, or commentary.
```

相对你现有中性版，我建议补强的差异是：

```diff
+ 明确 FIRST input image 是构图与内容基准
+ 锁定 exact framing / crop / camera distance，针对本轮明显的放大重取景
+ 锁定 facial identity、saturation、lighting direction
+ 禁止 beautify / simplify / add-remove details
+ 修线仅限 intended continuation is unambiguous
+ 手部仅在证据充分时修，避免借“修复”重造姿势
+ 明确保留 granulation、translucency、edge variation 和 grain
+ 不确定时少改，不允许模型为了整洁而统一线宽或色阶
```

cel 模式不应再维护一份完整系统提示，只需要在中性基底后追加覆盖项，例如：

```text
CEL MODE OVERRIDE:
The first image is confirmed to use hard-edged cel rendering. Preserve its
existing outline weight and palette while cleaning flat fills, maintaining
crisp shadow boundaries, and repairing inconsistent line continuity. Do not
introduce thicker outlines or reduce the number of shadow tones unless the
source already uses that treatment.
```

即使是 cel 模式，也不建议继续写死 `bold even-weight outlines` 和 `one hard-edged shadow tone per material`，因为这两条仍会把细线 cel、多阶 cel 和无外轮廓 cel 强制改成同一种风格。

## 5. 参数建议

### 分辨率：默认 2K

- **1K**：更快、更省，但发丝、发饰、手指和水面碎笔触缺少足够像素，模型更容易把它们概括掉。适合预览或低风险筛选，不适合作为最终修复。
- **2K**：当前最合理的默认值。细节容量、成本和生成稳定性比较平衡。
- **4K**：不应作为常规默认。成本和延迟明显增加，而且生成式 4K 不等于忠实修复，可能产生更多伪细节、纹理漂移和局部重设计。适合最终入选图的第二阶段放大，前提是先验证 2K 构图和结构正确。

建议：`2K 重绘 -> 通过结构/风格检查 -> 必要时再做独立 4K 放大`，不要用一次 4K 重绘同时承担修复和放大。

### 输出格式：优先 PNG

当前固定 JPEG 不理想：

- JPEG 会在线条、浅色渐变、水彩颗粒和高反差边缘周围产生振铃、色块与蚊噪；
- 后续如果还要裁切、放大或二次处理，会发生重复有损压缩；
- 你要修的“色溢、带状、边缘杂质”有一部分可能被 JPEG 再次引入。

建议保留模型返回的原始 MIME 和字节；若接口能返回 PNG，直接存 PNG。若模型只返回 JPEG，不要仅改扩展名或无意义转码成 PNG。存储成本是 PNG 的主要代价，水彩复杂纹理下文件可能明显增大。

### `aspect_ratio=auto`：不适合严格保构图任务

`auto` 对自由生成方便，但这里要求保持画布和取景。应从第一张输入图读取宽高比，并映射到模型支持的最接近比例，例如本组竖图明确指定 `2:3`。同时在提示中锁定 crop 和 camera distance。

需要注意：显式宽高比只能锁画布，**不能阻止人物放大和内部重取景**，所以还需要构图相似度检测，例如关键点位移、边缘/特征匹配或感知相似度检查。

### 是否同时喂参考原画：不要作为无条件默认

直接加入参考原画有潜在收益：

- 帮助恢复目标媒介、配色关系、线条特征和装饰密度；
- 当首图已经偏离目标风格时，比纯文字描述更有效。

但风险同样明显：

- Gemini 可能从参考图借用人物、服装、姿势、背景物体或构图；
- 多图主次关系可能不稳定；
- 本轮图 1 与图 2 的内容差异很大，若同时输入，可能把校服、植物、水池等语义带进海边白裙图；
- “保持首图不变”与“靠参考原画恢复风格”本身存在目标冲突。

更稳的策略是：

1. 第一张固定为待修复图，明确它是唯一的内容、构图、身份基准。
2. 第二张只在检测到首图风格偏移，或用户开启“参考风格校正”时加入。
3. 明确声明第二张只提供 medium、palette、brushwork、line character，不提供任何人物或场景内容。
4. 最好先从参考原画提取结构化风格描述或风格特征，再把描述用于重绘；这比直接双图输入更可控。
5. 若必须双图输入，应做 A/B 验证，重点检测构图位移、服装变化和参考图语义泄漏。

## 最终一句话

**当前瓶颈在 Gemini 重绘环：它能显著清线和提高完成度，但仍把“局部保真修复”当成整图再生成，导致画风、身份细节和取景发生不可控漂移。**