# 模型侧区域控制研究：不做本地拼接

## 用户约束与历史证据

2026-10-05 用户明确要求：历史 Python 蒙版羽化叠加效果差；直接向生图接口传蒙版效果未知，可以研究模型侧控制，**不要尝试本地图片叠加**。

本研究禁止对候选裁切后贴回、alpha 混合、羽化融合、源图保护区复制回候选、蒙版内本地调色，也不调用 `local_repaint_composite.py` 或既有 local/feather 工序。最终候选必须是服务端返回的完整图，原始字节保存；越界失败保留，不以本地修补把它做成通过。

仓库 `docs/gpt-image-tid-style/BEST-PIPELINE.md` §7.7 / `HANDOFF.md` §8.3 记录了裁切重绘重新构图与贴回错位，调整羽化仍存在相同矩形边界。本次不重复这条路线，不修改其历史文档或既有生产功能。

## 当前接口能力核查

|路线|文档/本地证据|本研究判断|
|---|---|---|
|当前 Gemini generateContent，文字描述局部|官方称 semantic masking；项目发送 text + inline_data 图片|已具备语义编辑入口，但旧实验已显示文字保护不可靠|
|Gemini 第二张黑白蒙版或位置图|项目支持多个普通图片附件，没有 mask_path/mask 字段|只能作为普通视觉指导；效果待测，不能宣称硬蒙版支持|
|Imagen 原生蒙版|官方有 REFERENCE_TYPE_MASK / MASK_MODE_USER_PROVIDED|不同模型、接口、鉴权与编辑模式；当前项目 Gemini 入口没有接入。暂不切换或假定中转站支持|
|本地 SD/WebUI 服务端 inpaint|diff_cg_tab.py 把 mask 放进 WebUI payload|这是另一条已有服务端蒙版路线，不能证明 Gemini 中转站支持同一字段，也不与当前 Gemini 实验混比|

依据：[Gemini 编辑说明](https://ai.google.dev/gemini-api/docs/image-generation)、[generateContent 参考](https://ai.google.dev/api/generate-content)、[Imagen 蒙版编辑请求](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/image/edit-insert-objects)。Gemini 文档将局部范围描述为对话语义；Imagen 请求则有专门蒙版类型。上述区别不保证本机中转端点的实现或效果。

本地核查：`modules/others/api_backend.py::generate_image_aigc2d` 只把所有附件序列化为 `inline_data`；`_post_images_edits_request` 只追加 `image[]`；不能偷偷增加未知 mask 字段并称成功。检索未找到 new.aigc2d.com 可核验的原生 mask 契约，状态记 **未确认**。本次没有携带凭据或图片进行在线能力探测。

## 先做位置图引导，暂不冒充分割蒙版

首个候选实验应使用独立的黑白**位置图**，帮助模型区分容易被染色的两处枝叶：右下近景植物与右上伸入画幅的枝叶。位置图在空白同尺寸画布上画框，不把源图像素复制进去，不生成羽化层，不合成任何候选。

位置框只用于定位物体，**框内地面/水面不自动变成保护区**；框外也不自动获准改色。它不是逐像素分割，不能据此做面积测量或 mask 外锁定承诺。右上枝叶在新实验的两分支都明确保护，避免旧报告中深度类别不清的问题。

草案产物：`data/test-result/20261005/book-spatial-guide-design-v1/protected-region-locator.png` 与 `locator-manifest.json`；完整原图不修改。局部边界未分割，位置图没有“准确保护全部叶片”的保证。

生成说明：`prompts/color-knowledge/book-spatial-guide-v1.md`；评审补充：`book-spatial-guide-review-v1.md`。它们是工程实验规则，不是新书籍色彩知识。

## 下一轮实验设计

独立 ID：`book-spatial-guide-v1`，当前 **design_only / not_frozen / not_sent**，不直接拿旧 book 协议强塞第二张图。

固定同一源图、C16 色板、Flash/1K/3:2、零自动重试、无画风图。两个分支各两轮，共 4 次生成、2 次分组评审：

1. source：源图 + 统一新合同/枝叶保护说明。
2. source_locator：同样正文及源图，再加明确角色的独立位置图。

只有第二张指导附件及其角色说明不同，记录这种整体干预，不声称只测图像。沿用完整画幅，保持授权蓝/黄/橙红与人物；不混入色调、材质或本地后处理。

首轮仍要求逐张全图查看：右下叶片粉紫保留、右上枝叶保留、对岸保留；授权环境与饰带执行；人数、持竿结构、服装、单画幅、无指导框/图例复制。指导框入画即内容失败。位置图不是第二个人、画风图或配色参考。

成功只表示有继续多源验证的价值。若位置图漏保护、被画进产物或导致内容变化，停止扩量，不以复制源区/本地叠加兜底。随后才研究独立原生 mask 端点是否具备可用契约；端点未确认前不发未知字段请求。

## 接入现有 CLI 前必须完成的事项

现有 `book_ablation_plan` 只有 content_reference 角色，run 账本把所有附件写成 content，review 也只带一个源图。因此现在还不能把位置图放入 image_paths 然后声称角色隔离完成。

后续在现有 CLI/工具内新增显式 spatial_guide 角色，分别冻结源/位置图 hash、尺寸、次序，实际 body 与预览一致；发送前发现缺失或变更必须停止。额外元数据不得泄漏给 backend kwargs。评审真实带源/位置图并说明候选全图/裁剪/灰度顺序；明确检查位置符号泄漏。旧协议 hash 与两种附件分支回归通过后，再冻结运行协议。不得新建实验目录内可执行脚本来绕开共享工具。

本次交付完成能力研究、独立位置图草案与生成/评审提示词；**未实现新的 CLI 接线，未冻结收费实验，未出图**。原生 mask 与视觉指导的区别保留为未知能力边界。
