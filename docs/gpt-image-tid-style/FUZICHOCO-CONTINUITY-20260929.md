# fuzichoco-v2 单图色彩与线条连续性复核（2026-09-29）

仅评估 `data/20260928/1a6ecf6e_fuzichoco-v2-final-rescue-1_233546-4169ad-canvas.png` 这一张图及其同任务阶段产物。对照的 Gemini 直出 `2a47e58c_232424-c90b6f.jpg` 使用另一画风（lavender/rose/pearl），只能用来观察轮廓和色块组织，不能作为 fuzichoco 同风格分数目标。

## 阶段诊断

GPT 首图的伞是收拢状态，与内容锚的张开要求冲突。第一次 Gemini 重绘改善了线条连贯，但放大了青色发梢、粒子和局部高光。质量修订、身份修订、最终补救把伞修成张开，也逐轮柔化了衣褶、蕾丝和发丝。原图与最终图的 `local_rms_contrast` 为 17.97 → 13.59；平均线段长度为 72.7 → 95.4。指标只描述这张图的阶段变化，不能证明风格优劣。

## 同图实验

候选均保存在 `data/test-result/20260929/fuzichoco-single-image/`，正式图没有覆盖。

| 候选 | 输入 | 局部对比 | 长线平均长度 | 碎段比例 | 视觉判定 |
|---|---|---:|---:|---:|---|
| 正式最终图 | — | 13.59 | 95.4 | 1.2% | 伞、身份正确；粉裙高光将若干色块压平，衣褶与发丝偏柔 |
| `all-local-contrast-soft.png` | 最终图，本地亮度锐化 amount=0.20 | 16.41 | 90.0 | 1.3% | 轻微提高已有边缘；无法找回被重绘抹掉的笔触 |
| `colour-line-continuity_081305-9bcc46.jpg` | 最终图 + GPT 首图作为笔触参考 | 16.98 | 82.3 | 1.3% | 粉裙与棕发色块较完整；伞被错误改回收拢，脸也漂移，淘汰 |
| `colour-line-source-only_081657-8c24ab.jpg` | 仅最终图 | 15.20 | 90.6 | 1.8% | 保住张开的伞，中间调略成片；脸有细微重画，线条碎段增加，淘汰 |

提示词分别保存在 `prompts/gpt-image-optimize/fuzichoco-continuity-study-20260929.md` 与 `fuzichoco-continuity-source-only-20260929.md`。两次模型调用均为 `gemini-3-pro-image-preview`、2K、2:3，只用这一任务的图。测量使用 `tests/style_render_metrics.py`。指标对重绘引起的构图变化敏感，必须结合伞、脸、手和裙摆目视复核。

## 管线改进方向

1. 首图后、画风重绘前验证高价值物件状态（本例为伞张开/收拢）。尽早修正可避免末尾以身份修订重画整图。
2. 把“色彩连续性”列成明确审计项：发色主体是否连续、相邻裙层是否各有稳定中间调、环境色是否被零散高光切碎；和线条连通性分开判断。
3. 后续修订应以当前图为唯一内容源。历史首图即使标为笔触参考，仍可能把错误物件状态带回。单图全局重绘也会产生细微身份与线稿漂移，须逐次与输入图对照，未过门禁就回退。
4. 只靠最终本地锐化不能恢复丢失的绘画细节。本图的主要收益应来自减少后续全图修订和保护首轮笔触，而非把现有成品再锐化。

本次没有证据支持替换正式图或把实验提示词设为默认固件。泛化修改需另选同画风样本验证。

## 追加：仅用 Gemini 提示词处理反光（同图，new.aigc2d）

用户指出左下角前景蓝色框架沿同一长边出现间断亮点，因此又用正式最终图作唯一输入、通过现有 `generate_image_repaint` 调用 `gemini-3-pro-image-preview`（2K，2:3）测试两种提示词。没有修改或发布正式图。

| 提示词 | 候选文件 | 目视结果 |
|---|---|---|
| `fuzichoco-colour-flow-20260929.md`：整体彩色中间调与左下角反光连贯 | `colour-flow-source-only_223718-15fb64.jpg` | 左下角反光更连贯；框架形状与其他区域也被重画，裙摆更平滑，不能采用 |
| `fuzichoco-foreground-reflection-20260929.md`：只要求修左下角框架，其余不变 | `foreground-reflection-source-only_223954-a0628d.jpg` | 左下角变成过亮的青色带，右下角框架也变亮，人物和裙摆仍有漂移，不能采用 |

结论：提示词能够引导色彩连续性，但整图 Gemini 编辑无法可靠限制在框架局部；追加“其他区域完全不变”的文字仍不足以保证原图身份与材质细节。若继续试验，应使用真正的局部编辑/遮罩或谨慎的裁切贴回，并逐张检查拼接缝和画面漂移。当前不应把这两种提示词并入默认全图重绘固件。

## 追加：全局底色 prompt 与 GPT edits 对照

继续只使用同一正式最终图。`fuzichoco-colour-underpainting-20260929.md` 要求 Gemini 保留高光小点，只让低/中明度底色在物体内部延续，不把亮点连成霓虹线。产物 `colour-underpainting_230633-8affbf.jpg` 确实没有出现强青色光带，但左下角的断续感改善很小；脸、裙层和室内仍发生全图重绘。该候选不采用。

`fuzichoco-gpt-edit-colour-flow-20260929.md` 用已配置的 `new.aigc2d` `/v1/images/edits`，只挂最终图。`gpt-image-2` high 连续返回 429（上游组负载已满），没有图；随后项目已登记的 `gpt-image-2.5-flare` high 成功，产物为 `gpt25-flare-edit-colour-flow_output_231106_0_d543d7.png`，尺寸 1024×1536。其整体色块和轮廓较规整、张开的伞保留，但左下角亮点依然间断；脸、裙层、房间细节和画面尺度明显被重新解释，且分辨率低于原图 1696×2528。也不采用。

这三次新增对照都没有达到“色彩更连贯且身份/结构/细节稳定”的双重目标。单靠全图 prompt 或无遮罩 GPT edit 不适合作为本问题的默认后处理。左下角例子下一步需要将可编辑范围真正限制在该物体，同时比较边缘衔接、原有虚焦和整图身份；本轮没有修改生产管线，也没有替换或发布原图。

## 追加：GPT-image 提示词双方案实测

在上述 GPT 初探之后，固定输入为正式最终图，只使用已配置的 new.aigc2d `/v1/images/edits`、`gpt-image-2.5-flare`、`high`、PNG、1024×1536、单张输出，测试两种 GPT 编辑提示词。两次均成功返回；产物保存在同一 `data/test-result/20260929/fuzichoco-single-image/` 目录。提示词原文分别是 `prompts/gpt-image-optimize/fuzichoco-gpt-edit-frame-base-20260929.md` 和 `fuzichoco-gpt-edit-material-flow-20260929.md`。

| 方案 | 产物 | 同原图目视对照 |
|---|---|---|
| 仅补左下角前景框架的暗蓝色主体，保留零散亮点 | `gpt25-frame-base_output_231657_0_dd03bb.png` | 框架的暗蓝色略更连续，但亮点仍间断。张开的伞和大构图保住，脸、发丝、裙层、室内细节仍被全图重画；不适合局部修复。 |
| 给发丝、裙层、蕾丝、房间与框架各自补连续的低/中明度底色 | `gpt25-material-flow_output_231850_0_713fe7.png` | 深蓝房间、棕发与粉裙的色块层次更清晰，是本轮整体色彩组织较好的一张。左下角亮点仍未根治，脸眼与裙层细节发生变化，暗部和整体对比也明显加强；更像一次重新渲染。 |

两张候选都比正式图分辨率低（1024×1536 对 1696×2528）。本轮结果说明 GPT edit 可以用提示词改善全图色彩组织，但当前无遮罩编辑难以只改一件物体或严格锁定身份细节。以上是单图目视判断，没有以跨构图数值指标代替质量审查。两张候选均不发布，不改默认固件；若继续开发局部修复，应先解决真实可编辑区域及输出分辨率，再以原图对照审计。

## 追加：从 GPT 首图开始做全图渲染编辑

前面把左下角框架当成局部目标过于狭窄。本图的发丝、蕾丝、墙面、床品和前景框架同时出现密集等亮度闪点、局部青白色反光与主体底色割裂的问题；它是跨材质的光照层级和色块组织问题。以下三次均只输入本任务真正的 GPT 首图 `1a6ecf6e_output_232916_0_dba837.png`，使用 new.aigc2d `/v1/images/edits`、`gpt-image-2.5-flare` high、1024×1536、PNG。首图本身把内容要求的张开伞画成收拢伞；实验为隔离渲染变量而保留这个状态，候选不能直接发布。原首图由 `gpt-image-2` 生成，因此首图与编辑后的变化同时包含模型差异，不能全归功于提示词。三个编辑候选使用同一模型与输入，彼此可以做提示词方向对照。

| 全图策略 | 提示词与产物（均在测试目录） | 目视判断 |
|---|---|---|
| 重建光照层级：连续材质底色、减少零散高光和 3D 镜面感 | `fuzichoco-gpt-first-light-hierarchy-20260929.md` → `gpt-first-light-hierarchy_output_233023_0_c3aacc.png` | 发丝、裙层和房间更容易读成整体；却几乎抹掉星形光源、青蓝色奇幻反射和原有冷暖戏剧性，变成普通室内暖光。力度过大。 |
| 限制主色族：蓝房间、棕发、粉白服装，按材质分少量明度层 | `fuzichoco-gpt-first-limited-palette-20260929.md` → `gpt-first-limited-palette_output_233233_0_a31eaf.png` | 色块最干净，层级清楚；画法偏硬边、近乎平面插画，原画风的透明色层和幻想氛围流失。直接限制颜色种类不是合适的通用解。 |
| 选择性保留星光与蓝窗，只合并物体内部的随机微闪点 | `fuzichoco-gpt-first-selective-sparkle-20260929.md` → `gpt-first-selective-sparkle_output_233554_0_67e360.png` | 三者中最接近原画风：星形光点、青蓝窗光与粉蓝对比仍在，深棕发和深蓝室内更连贯。裙摆仍有不少细碎亮斑，人物五官与衣饰细节有轻微重画，伞仍收拢。 |

为检查额外 GPT 工序能否改善全链路，将第三张候选按该任务断点中的原始 `repaint` 设置继续跑一次 Gemini：相同 v5 固件、完整画风参考图、`reference_mode=style`、`scope=full`、2K。产物为 `gemini-after-gpt/gpt-first-selective-sparkle_output_233554_0_67e360-233926-fbc285-final-rp_234016-3615c8.jpg`。对照原本的第一次 Gemini 重绘，新增链路的房间更蓝、部分大色块较清楚，但 Gemini 再次加强青色光效和亮点，并在裙侧画出原图没有的半透明拖尾；脸和服装又经历一轮重画，收拢伞错误未修。这个单次下游对照没有证明额外 GPT edit 对最终图有稳定收益，且出现明确设计漂移。

**阶段结论**：改善方向是全画面建立「主体底色→成组反射→少量焦点亮光」的层级，而不是简单降低全图颜色数量或删除幻想光源。对生产管线，优先在 GPT 首图生成条款和 Gemini 首次重绘提示里规定这种层级，并在审计中分辨“有意的星光/装饰”和“随机微闪点”。目前仅单图三种 GPT 编辑和一次下游重绘，不能据此把额外 GPT edit 插入默认链路，也不能宣称泛化有效。需要其他同画风与不同画风图验证，且任何新增工序仍须通过内容、身份、画风及最终图审计。

这张图的首图请求本身同时使用了 `luminous highlights`、`glitter`、`transparent pigment` 和 `intricate ornament`；画风重绘条款又要求 `glitter at focal areas`。这些词没有说明亮光必须服从材质底色与主光源层级，可能给模型留下把各处纹理都画成同等显眼闪点的空间；这是由提示词和图像结果推断的待验证原因，不能当成已证实的模型机制。下一轮可先在首图与首次重绘的画风条款中补入通用原则：*Each material must remain readable as a connected base colour and a few form-following shadow/light masses; reserve the strongest sparkle for intentional focal motifs, and keep scattered texture marks distinctly lower in value and contrast.* 不建议写死“每种材质三阶明度”或全图固定颜色数，第二种测试已显示这会损伤原画风。

## 追加：两种全局提示词的 GPT 首图 → Gemini 重绘对照

按用户要求进一步实测两种完整链路。固定同一份任务断点的内容、`fuzichoco-v2` 画风说明与参考图、1024×1536/high/PNG，GPT 首图均走 new.aigc2d `/v1/images/generations`，模型固定 `gpt-image-2.5-flare`；原任务首图为 `gpt-image-2`，因此只把本轮两臂互相比，不将其与旧首图差异全部归因于提示词。Gemini 均使用断点原 v5 固件、同一画风图、`reference_mode=style`、`scope=full`、2K；各臂仅追加对应的全局渲染条款。四段条款原文在 `prompts/gpt-image-optimize/global-*-experiment-20260929.md`。所有结果保存在 `data/test-result/20260929/fuzichoco-global-prompt-ab/`，未进入发布目录。

| 策略 | GPT 首图 | 首次 Gemini 重绘 | 观察 |
|---|---|---|---|
| 材质底色→顺形反射→少数焦点高光 | `gpt-first/hierarchy-first_output_234815_0_76bed1.png` | `gemini-hierarchy/hierarchy-first_output_234815_0_76bed1-235030-b73558-final-rp_235111-9a1869.jpg` | 仍有较丰富的蓝粉氛围和细节，发与裙的主色块较易辨认；Gemini 又在挂裙、墙面和家具附近增添闪点。相对限色版更接近目标画风，但高光层级未稳定锁住。 |
| 全图约 8–10 个主色族、每材质底/暗/亮三组 | `gpt-first/limited-first_output_234930_0_5cb811.png` | `gemini-limited/limited-first_output_234930_0_5cb811-235144-0dd24b-final-rp_235229-43f40b.jpg` | 色块安静、蓝色家具和粉裙更整齐，但画面显得普通、层次较硬，透明色与幻想氛围减少；Gemini 继续沿这一方向。 |

两条 GPT 首图随机生成了不同的房间布置、脸与服装细节，且都把内容要求的张开伞画成收拢伞。因此上表是单次完整链路的视觉对照，不能视为同构图下的严格因果实验，也不能通过后续重绘修复伞状态。

为单独检查 **Gemini 条款**，又固定原任务同一张 `gpt-image-2` 首图，各跑一次 Gemini；除了末尾追加的条款外，其余固件、画风参考图和参数相同，并可同原来的首次 Gemini 重绘并排看：

| 同源重绘 | 产物 | 观察 |
|---|---|---|
| 层级条款 | `same-first-hierarchy/1a6ecf6e_output_232916_0_dba837-235344-9f84a3-final-rp_235423-c072ad.jpg` | 原房间布局与星形光点大体保留；比原首次 Gemini 重绘少一些散布的青色粒子，发色与裙层略更成片。脸、蕾丝和伞仍被重画或继承错误，不能直接采用。 |
| 限色条款 | `same-first-limited/1a6ecf6e_output_232916_0_dba837-235507-f38486-final-rp_235547-c9ef0e.jpg` | 色块更安静，但新增原图没有的巨大画框和白鸟，背景结构严重漂移；本次样本明确失败。 |

**本轮判断**：层级提示词方向在同源重绘中有可见收益，但一次样本尚不足以验证泛化；硬性限制主色族虽能压住部分色彩碎片，却明显削弱画风，而且同源 Gemini 对照出现重大背景漂移。两臂都未通过完整内容/身份/终审，不发布、不改默认固件。下一步若改生产提示词，应先只提炼“主体底色优先、反射成组、焦点高光少量”这一原则，并在其他题材和画风复测，保留最终图门禁。
