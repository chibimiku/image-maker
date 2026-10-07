# Gemini 首轮重绘：画风与结构碎线

## 实盘根因

源图：`data/20261007/gpt-image-2/gptimage2_output_154033_0_e4052e.png`。
旧首轮重绘：`data/20261007/gpt_image_repaint/repaint_gptimage2_output_154033_0_e4052e_154127-80c7d9.jpg`。
所选画风：`cccccanh`，附件 `submodules/image-maker-artstyle/references/cccccanh.png`。

`log/2026-10-07.log` 15:40:33 的首轮 Gemini 请求只有一个图像 part，未发送画风附件；提示词明确包含 `flat cel shading`、`no painterly blending`、`bold even-weight continuous outlines`。源图的笔触、半透明色层与软边因此变为均匀墨线和硬边平涂。15:41:27 后的双参考请求是衣装工序，不是此次被比较的首轮重绘。

现有 Tab 的 `build_repaint_request` 在生图/编辑模式误把生图正文作为重绘 prompt 覆盖固件；且衣装分支只重新读取固件，未传画风图。不能仅更新磁盘固件就认定请求修好了，必须核验最终 prompt 和传输附件。

## 已落地的接线修复

- 自动重绘与生成模式下「重绘结果列表的图」使用当前磁盘修复固件，不再把生图正文送给 Gemini。
- 自动重绘将所选画风图传入 `extra_reference_paths`：每张请求依次为 Image 1 源图、Image 2 画风图。画风图不作为独立待重绘源图，不混用 GPT 上游的用户附件编号。
- 默认保留现有 `repaint-system-conservative-v5.md`，其 OPTIONAL MULTI-IMAGE MODE 已明确源图负责内容/身份，第二张负责画法，禁止拷贝角色、饰物和构图；不激活下面的实验固件。
- 独立重绘模式继续尊重用户在编辑框中写的完整修复 prompt；本次不自动追加画风附件或替换用户指令。
- 衣装连续性条款追加到已经选好的修复指令，不再覆盖它。自动重绘禁止后端再次追加通用细节后缀。日志和重绘开关 tooltip 显示两图分工。
- 保留此前逐源图显式比例与输出比例校验。

## 两次已授权实验

Google 的[官方图像编辑指南](https://ai.google.dev/gemini-api/docs/image-generation)提供了保留构图并改变绘画技法的提示词方式；[DeepMind 提示指南](https://deepmind.google/models/gemini-image/prompt-guide/)建议明确区分参考图中的对象。这支持显式分配两张图的用途，但不能保证模型不串入内容。

实验候选：`prompts/gpt-image-optimize/repaint-style-structure-v1.md`。它允许参考图影响眼睛/头发的绘画技法，要求只把主要发束、衣装边界、遮挡线的冗余碎笔整理成连贯的渐细线，保留笔触、软边、失边与次级纹理。

两次使用完全相同的源图、参考图、2570 字符 prompt、`gemini-3-pro-image-preview` / 2K / 3:2，各调用一次。关闭自动重试，持久账本硬限制 2 次；不接衣装、色调、加墨或文本审计。旧首轮结果是历史对照，本次没有重跑旧版。两次只检查候选的重复表现，并非隔离单个变量的 A/B 因果实验。

结果目录：`data/test-result/20261007/style-structure-154033/`，包含 `prompt.txt`、`outputs.json`、`budget/images.jsonl`、两次原始请求图像摘要及返回图。

| 候选输出 | 肉眼检查 | 判定 |
| --- | --- | --- |
| `style-structure-1_155458-6c7c1b.jpg` | 发束、辫子和盔甲边缘更连贯；背景与盔甲保留笔触和软边，未出现旧版均匀黑线/硬边平涂；表情与眼睛已有重绘变化 | 有用的方向，仍有局部漂移 |
| `style-structure-2_155525-b2bb58.jpg` | 绘画质感仍在、发束更顺，但源图胸前和腰侧新增参考图类似的蓝色花饰；脸更光滑、发束更规整 | 内容保真失败，不可作为默认 |

两张均为 2528×1696。不能以“线更顺”掩盖复制饰物，也不能声称已经稳定兼得画风和内容。候选仅留作研究，未接入自动运行路径。未新增第三次请求。

## 验证边界与后续方向

四组离线回归 `test_gpt_image2_api` / `test_wardrobe_repaint_step` / `test_post_process` / `test_gpt_image2_cli`：217 passed；模块编译与 diff whitespace 检查通过。新增回归核验真正进入后端的 `[source, style]` 顺序、prompt 不含生图旧平涂正文、3:2 比例、手动重绘不被覆盖、衣装条款不抹掉修复分工。

**线上实测的是实验候选，不是最终默认的 v5＋画风附件组合**。后者的代码接线已离线验证，但视觉收益尚未在本次预算内复验。下一轮应优先验证保守组合；若迁移不足，再比较“不动脸部绘画，只整理主轮廓”和“保留脸部几何但允许画法迁移”。每份关键方案至少两次同源复跑，单独检查饰物复制与身份漂移，不只测整图锐度。只有用户另行授权后才增加收费测试。

需要重启 App 才加载代码。现有原图、旧重绘、最终图与日志全部保留。
