# 整体修复＋头发重绘＋光照重渲染接入

## 当前流程：条件针对性修复＋固定整体修复＋材质指导

2026-10-11 后续用户要求已接入：`源图诊断（含服装材质）→ [明确缺陷：针对性修复＋审计] → 固定整体修复（追加材质指导）＋审计 → 本地后处理`。下文v6记录、预算和实验为先前接入历史。

针对性分支仅接受高置信度诊断的明确缺陷及完整 region/defect/evidence/repair/preserve 字段，不将一般保留要求视为缺陷。分支审计还须有 targeted_repair_completed=true 和非空可见证据。未通过、无图、比例错误或请求/审计失败保留输入，随后仍执行一次固定整体修复；取消及源人体诊断不可靠时停止。整体未通过则保留其输入（可能是已通过的针对性修复图）。均无自动重试。

诊断新增 garment_materials，逐项记录区域、英文材质分类、可见证据与置信度。分类覆盖皮革、丝绸、缎面、棉麻、毛料、针织、蕾丝、半透明布、牛仔、天鹅绒、毛皮及金属衣甲；无法确认时 unknown 或低置信度。只有源诊断和该项均≥0.8、角色服装可见且证据完整的已支持分类参与指导，不从艺术高光推断材料、不以首饰替代衣装材料。

固定 v3 整体模板保持，后追加最多两句英文：第一句组合已识别材质的质感/光学响应，第二句要求符合现有照明和画风、保持固有色/位置/设计、避免过度光泽及替换材质。规则存于 refine-materials.json，区域和证据随记录保留，不将自由文本直接拼入固定指导句。针对性步骤不追加此后缀。完整实际prompt送入整体审计，诊断标签不能覆盖源图实际证据。状态视图显示材质，后处理清单记录规则文件。

新 v7 workflow=targeted_whole；最大2图3文本、网络预算1260秒，无缺陷为1图2文本。材质识别复用同一次诊断，无额外调用。显式整体重试复用原诊断，旧v6仅whole、v5及更早保留原分步断点，均不自动重跑。

本轮离线验证见 data/test-result/20261011/targeted-material-refine-adoption/offline-check.json；没有新增收费调用、配置修改、commit或重启，尚无本轮整链线上视觉验证。重启app后新任务生效。

用户在中文整体优化对照后明确要求接入当前工作流程。新自动生图/编辑与分析 GPT refine 入口采用：

`GPT 首图 → 源图诊断 → 整体优化一次 → 三图保真审计 → 已勾选的本地后处理`

图片优化使用 `prompts/gpt-image-optimize/refine-whole.md`，初次接入逐字采用用户试过的80字中文原文，随后按用户要求改为等义英文（见下文）。不追加人物/背景任务、身份描述、画风附件或 detail suffix。只有当前输入一张图，沿用配置的 Gemini Flash Image、2K和逐图比例；没有切换默认模型到Pro。

新任务不再自动执行人物、条件头发、背景、配饰、光照五段分步修复；整体任务已覆盖衣装修复，不重复额外衣装重绘。色调、加墨等本地工序仍按当前勾选执行。整体优化的输出保留源图可识别艺术风格由审计检查，不声称中文“像素级精度”能保证逐像素不变。

诊断仍要求可核实的人体/身份信息。整体审计使用原图、当前输入、候选三张图及真实尺寸，明确允许整体连贯化、头发重绘与光照变化，不能套用分步阶段的冻结背景/光照条件。人体、重大内容/构图、明显固有色和画法变更仍检查；头发明确仍碎或判断不明不能自动采用。没有降低0.8置信度及全保真字段门槛。

新优化上限1图2文本（360秒图片、180秒每个文本，网络预算720秒）。审计失败、请求失败、无产物或比例错误回退输入，不自动重试，完整请求/候选/错误记录保留。v6记录新增 `workflow=whole` 和 `whole/whole-audit`。

手动重试从整体步骤原输入发一次图片＋审计，复用源图诊断；人工采用已返回候选不会新增请求或把历史失败改写为pass，也不重复本地后处理。源图hash改变仍拒绝续跑。

历史v5及更早分步记录按原阶段继续，旧person/hair/background/ornaments/lighting模板保留用于显式历史重试。状态查询、阶段下拉按记录的workflow选择，避免新记录显示未执行的旧阶段，或者旧记录被静默迁移到新流程。

本轮只离线接入，没有新增收费图片或文本调用；此前单样本中文Flash对照不等于新生产“诊断＋审计”整链在线通过。没有修改运行配置、commit或重启。**重启app后新任务采用新流程；历史记录不自动重跑。**

验证：核心refine、状态/UI、分析通道三套131 passed；公共后处理、费用、分析配方、CLI、GPT API五套250 passed，共381 passed。覆盖单次完整链、实际三图尺寸、失败/超时/无图回退、头发仍碎停、人工重试/采用、历史断点与UI及新预算。八个修改Python文件编译与diff-check通过。证据 `data/test-result/20261011/whole-refine-adoption/offline-check.json`。

## 等义英文更新

用户要求尽可能使用英文，初版英文模板曾更新为（当前使用末节v3）：

> Repair the image with pixel-level precision while keeping its artistic style unchanged. Resolve visual fragmentation and discontinuous lines. Completely redraw the character's hair with clear, continuous lines, removing especially tangled or chaotic hair strokes. Re-render the lighting in the image.

保留四项力度与范围：像素级精度修复、艺术风格不变、完全重绘头发、重新渲染光照；不改成仅修补少量发丝或只调整不一致的光照。“特别杂乱的部分”译为杂乱的头发笔画，避免被理解为删除整片头发或缩短发型。没有追加新的构图/颜色冻结条款。

仅语义核对与离线请求验证，未付费生成英文对照；不能声称模型输出与中文完全一致。之前中文实验及接入证据保留为历史快照，不改写其prompt hash。

## 随后英文Flash单图复验

用户随后明确要求出一张Flash与此前比较。同tone+ink输入、Flash Image/2K/3:2/360s/单附件/无suffix和附加后处理，英文一次28.9秒返回。源图、公共配置及重绘配置与中文轮hash一致，传输仍为同后端JPEG编码。1图1预留全结算，无文本调用、失败、重试或在途。

全图与相同归一化头部区域观察：英文同样改善原图连贯性，但比中文候选保留更多左侧细碎绕线、明暗对比更强，并意外新增左下角 `Original character: [Creator's handle]` 文本。英文图不采用为成品；两图均有脸部、花饰、蕾丝重绘。没有修改英文生产模板或额外请求；单样本差异不能全部归因语言。没有付费审计或新生产整链验证。

证据 `data/test-result/20261011/english-optimize-comparison/`：冻结plan/英文原文/请求结果/预算/summary、自包含comparison.html、head-comparison.png及unexpected-caption.png。新图 `english-flash_021240-1a4cf8.jpg`。

## 英文修订v2/v3与当前采用模板

用户要求修改并再测试。v2强化清晰发束、使用hair ribbons并禁止新文字；同输入Flash单次33.8秒返回，无可见新增文字，但短发明显变成长发并反向流动，代理判断不采用。随后v3去掉丝带隐喻，明确重绘只限原头发轮廓及范围，不延长/增量/反转风向，减少内部乱线；API响应28.7秒，含本地收尾总39.5秒返回。

当前生产英文为95词v3，见 `prompts/gpt-image-optimize/refine-whole.md`。删除模糊的pixel-level precision措辞，保留全图碎纹修复、完全重绘既有头发、统一重渲染光照；新增无文字及头发范围约束。v3保住短发与向左风向，较v1发束更清楚，没有可见新增文字；仍有蕾丝/花饰/脸部细节重绘。单样本视觉可用方向，不是收费审计通过、用户成品验收或稳定性证明。

本次两个独立版本各一次图片请求，2图全返回/预留全结算，无请求失败/自动重试/文本/在途；没有强制续跑原链。v3完成时冻结hash一致，未修改API配置、模型、流程代码或重启/commit。生产请求单测1 passed、diff-check通过。

证据 `data/test-result/20261011/english-optimize-comparison-v2/` 与 `english-optimize-comparison-v3/`；v3含四版自包含comparison.html、保持比例的头部对照、完整原文/计划/请求/结果/预算/summary。最终新图 `english-v3-flash_021634-66a17d.jpg`。
