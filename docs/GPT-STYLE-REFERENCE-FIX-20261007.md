# GPT 首图画风参考指令修复（2026-10-07）

## 多图分工与 UI 补充

官方 [GPT Image 提示词指南](https://developers.openai.com/cookbook/examples/multimodal/image-gen-models-prompting-guide) 的多图建议是按编号和用途指认每张图片，并描述它们如何配合。该指南直接给出 `gpt-image-2` 的编辑与画风迁移示例。[Image prompting](https://developers.openai.com/api/docs/guides/image-prompting) 也建议复用角色参考并重复关键外观约束，允许跨动作和场景保持角色一致性。

本应用顺序是：手动素材按缩略图顺序为图1～图N，系统画风图追加为图N+1。已检查 `ordered_reference_images` 和 multipart `image[]` 的构造，并用模拟上传验证实际提交顺序一致。官方没有要求画风图必须在前或在后；画风放最后是本应用约定，提示词编号和 UI 必须与该约定一致。

原来的 `STYLE_REF_ROLE_INSTRUCTION` 写用户图决定姿势/场景/构图且图像内容胜出，会压过“让人设跑步”的文字要求，也把“保持面部画法”与“迁移画法”混在一起。现在模板 `gpt-content-style-image-roles.md` 明确：各素材按用户指定用途使用；文字中明确要求改变的动作、姿势、表情、场景和构图优先；保留角色身份/脸部结构/比例/固有颜色/衣装，除非用户明确要求改变；系统画风图只决定作画技法。分析链兼容入口也用同一编号生成函数，避免遗漏占位符。

一个角色图 + 系统画风图时，建议写：

> 图1是角色人设参考。保留TA的身份、脸部特征、发型、发色、瞳色和服装设计，让TA在公园跑步，全身入镜。动作和背景按这段文字改变，不沿用人设图的站姿或排版。图2是系统画风参考，用它的线条、笔触和上色绘制图1的角色，不借用它的人物、服装或背景。

两张素材 + 系统画风时，可写“图1提供角色，图2只参考跑步姿势，图3只参考画风”；若两张素材是同一人设的不同角度，则写“图1和图2是同一角色的不同视角”。系统不猜测素材用途。后续用户要求已实现点击插入 `Image N`、排序自动更新正文图号和显示系统画风卡片，详见 `GPT-IMAGE-REFERENCE-INTERACTION-20261007.md`。

UI 在提示词框下常驻当前图号分工，悬浮提示列出图号与文件名，`插入引用示例` 按当前附件数量生成并追加示例（不覆盖已有文字）。添加/删除/重排/切画风都会更新；重绘模式隐藏 GPT 多图示例。模板放在 `prompts/gpt-reference-example-*.md`。请求日志中的图片也标图号。

额外修复附件顺序边界：有系统画风时素材上限15张，为最后的画风预留一张；已存在16张素材再选画风时拒绝提交并提示调整，保留所有素材。若素材文件已被外部删除，也拒绝提交，防后端过滤文件导致图号错位。没有画风时仍可放16张素材。

最终六组离线回归 **250 passed**，涵盖角色跑步优先级、实际模拟 multipart 顺序、重排后图号与文件对应、追加示例、模式切换、容量边界、失效路径。Qt 离线截图人工检查通过：`data/test-result/20261007/gpt-reference-ui/character-style.png`。未新发收费生图；第三方网关的真实模型行为尚需下一次生成验证。需要重启已有 GUI 加载修改。

用户报告首图 `data/20261007/gpt-image-2/gptimage2_output_122551_0_a2639b.png` 没有体现所选参考画风。

## 日志证据

`data/20261007/gpt-image-2/gptlog.txt` 的 12:23:19 请求走 `/v1/images/edits`，挂了 `submodules/image-maker-artstyle/references/cccccanh.png`。返回 usage 的 `input_tokens_details.image_tokens=1024`，说明请求有图片输入；这不能证明模型执行了画风迁移。

1900 字符正文由字段式画风说明、内容、末尾排除句组成。旧末尾已经有 `borrow only its palette, lighting, brushwork and rendering grammar`，因此不能说“完全没有引用指令”。问题是只有弱化的末尾借用句，没有开头明确的视觉画法权威与内容渲染词冲突处理；日志不足以证明这就是模型没迁移成功的唯一原因。

后续 12:25:51 的重绘正文要求 `flat cel shading`、`no painterly blending`、`bold even-weight continuous outlines`，与该画风字段 `layered glazing, watery washes` 冲突。它属于旧 `repaint-system.md`；当前磁盘配置已经是保留画风的 v5。窗口缓存启动时配置，且代码缺配置时的兜底仍是旧固件，两处都能延续旧画法；不能仅凭日志确定当时是哪一个触发。

## 修复

- `prompts/gpt-style-authority.md`：明确最后一张附件是画风视觉依据，匹配笔触、轮廓、边缘、上色、明暗与五官/头发画法。身份、物体固有配色和构图仍来自内容，禁止复制参考角色。内容中的渲染/镜头措辞不能覆盖画法。
- `utils/styles.compose_style_prompt`：有画风附件时先加入上述模板，再拼原有角色分工、画风说明、内容和防复制条款。GUI 和 CLI 共用此函数；画风测试页的共享组装也获得同一声明。没有画风附件时不添加声明。
- `GptImage2Widget.current_repaint_config/save_repaint_defaults`：读取磁盘当前固件，覆盖界面可调参数，防止窗口旧缓存继续发送旧固件或把旧固件写回配置。
- `utils/gpt_image_optimize.DEFAULTS`：兜底改为 `repaint-system-conservative-v5.md`，默认关闭细节后缀，与磁盘配置一致；显式指定其他固件仍有效。

## 验证与限制

离线回归覆盖首图实际 GUI 组装、CLI、画风文字、分析链兼容、不同附件数量及窗口启动后固件更新：`test_style_gpt_prompt / test_gpt_image2_api / test_gpt_image2_cli / test_analysis_gen / test_style_analyzer_prompt_pack / test_style_merge_validation` 共 **245 passed**。没有重新发起收费生图，也没有修改原产图或原日志；实际画风迁移效果仍需新请求观察。现有 GUI 需要重启才会加载 Python 修复。
