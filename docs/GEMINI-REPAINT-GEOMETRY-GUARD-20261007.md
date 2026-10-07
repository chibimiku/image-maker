# 强化构图约束：154859 第二轮实验

## 目标与方法

用户确认上一轮已不是同一张图后，授权进一步实验“prompt明确不要改变构图”等保障措施。原v1已经有preserve composition/pose/framing，所以本轮验证更具体的约束，而非仅再加同一句。

固定原图 `data/20261007/gpt-image-2/gptimage2_output_154859_0_ad9665.png`、模型 `gemini-3-pro-image-preview`、2K、3:2，各调用一次，关闭自动重试。两组：

1. `style`：原图＋cccccanh画风图；画风图只提供少量修复边界的笔触技法，不提供脸、眼睛、衣装或场景。
2. `source-only`：只发原图，在源图既有绘画介质内修主要结构碎线，排除第二张图的内容干扰。

两组共享 `repaint-source-geometry-v2.md` 与 `repaint-content-contract-154859.md`。正面锁定单一全身坐姿画面、人物左侧/右侧留白、手脚接触、椅子、地面两只鞋、左窗/室内背景；禁止重新取景、移物、换姿势、换脸或生成多栏。编辑范围缩到少量发束/衣装边界，不重绘五官，内容保真优先于风格相似。两组画风角色文字不同，不能宣称是纯附件消融或单变量A/B。

[Google官方指南](https://ai.google.dev/gemini-api/docs/image-generation)的风格迁移模板明确要求保留原构图，最佳实践也建议以正面场景说明和具体相机要求表达目标。本实验把这些要求进一步变成可核对清单，但它们仍是提示词，不能当作几何约束参数或保证。

## 结果

`style` 输出 `geometry-v2-style_162119-4b3445.jpg`，2528×1696，响应为fileData URL，由已有后端正常保存。肉眼：变成不同的长发写实角色、不同坐姿和鞋子关系，且输出双栏，未保住原图内容。

用新增 `repaint-global-content-audit.md` 发送一次原图/结果双图视觉审计。结果 `style-audit.json`：`verdict=fail`、`confidence=0.99`、`same_artwork=false`、身份/姿势/构图/主要物件/衣装均不保真、`global_replacement=true`、`eligible_for_local_cleanup=false`。本次不再对它发局部清除请求。审计把“更硬的轮廓”也列入added_elements，这不是独立逻辑物件，后续局部清除仍需过滤；但整体判定与人工检查一致。

`source-only` 首次在约125秒后未返回图片。无log_callback时后端只把具体请求异常交给回调，没有写到公共日志，因此首次具体异常原因未保存，不能自行补称为超时。随后在已授权研究范围内手动补一次同prompt/参数，并保存回调日志：第二次明确 `Read timed out (read timeout=120)`，未返回图片。两次不能算视觉失败或成功，也不能推断被审核拒绝。

共3次图片尝试＋1次文本视觉审计；2次无产物请求的扣费未知。每次都关闭底层自动重试，补跑是独立受控run，持久图片额度由2调整为3，文本上限2但实际只用了1。未继续补跑或延长全局超时设置。

## 证据与状态

完整产物目录：`data/test-result/20261007/style-structure-154859-geometry-v2/`。

- `style-prompt.txt` / `source-only-prompt.txt`：实际prompt。
- `*-request.json`：两组附件/模型/比例/完整prompt，不含密钥。
- `outputs.json`：style产物、source-only及repeat为空。
- `style-audit-request.json` / `style-audit-raw.txt` / `style-audit.json`：实际审计。
- `source-only-repeat-log.txt`：第二次请求明确超时。
- `budget/`：3次图片尝试与1次文本调用的发送前/结算账本。

新增模板均放 `prompts/gpt-image-optimize/`，通过真实prompt_loader检查占位替换，最终长度3570/3345字符，没有未替换字段。未修改运行配置或默认固件，未接入App自动流水线，未重启App，原图和历史产物不动。

## 结论与后续门槛

在当前模型/网关与这一张图上，更详细的构图/内容prompt仍未保证保真。带画风的加强版失败；源图单参考方案因请求异常尚无视觉结果，不可下结论。

可靠的交付措施应该是“生成＋判定＋失败回退”，而不是声称prompt能保证：源图始终保留；全局内容被替换直接拒绝重绘/回退原图，禁止局部清除；只有内容已保真且存在明确局部新增物件才清除一次。审计不完整/异常必须独立记为无法判定，不能作为通过。修复后也应核验内容保真，且调用次数需预算上限。此处为实验设计结论，未实现这些默认分支。

下一次实验应先获得源图单参考的有效结果（使用受控进程级更长超时、完整回调日志，不改公共配置、不盲目自动重试），或比较更短的纯编辑prompt与此内容清单版；任何关键成功方案至少同源复跑两次再替换默认。当前没有足够成功证据触发用户此前“如果ok就接自动流程”的条件。

## 用户授权360秒超时补跑：返回成功、内容通过，修线收益不明确

用户明确授权重跑超时的仅源图实验并延长超时。16:37:57 追加一次独立请求，实际回调显示timeout=360秒、max_retries=0、repeat=1。只在该Python进程内包装 `get_api_config` 的返回副本，设置timeout，不修改任何公共配置。记录公共 `conf/config.json` 前后SHA-256一致。输入图片、提示词、模型、2K、3:2均与此前仅源图请求一致；prompt SHA-256仍为 `e9e4c6b4de9a4c7e0cd2a5da3c082002a30f77b72f863e23630819d636c26de5`。

16:39:00 成功返回 `geometry-v2-source-only-long360_163900-f2d946.jpg`（2528×1696），HTTP响应约61.9秒、总耗时62.36秒。**这次在旧120秒窗口内已返回，因此不能把成功因果归给延长超时**；仅确认本次延长了允许等待窗口，并拿到了有效产物。

追加一次已说明的原图/结果双图审计，使用本轮此前未用的第二个文本额度。`source-only-long360-audit.json`：pass0.99，same_artwork、identity/pose/composition/major_objects/garment_design_preserved均true，global_replacement=false，added_elements与missing_or_replaced_elements均空，eligible_for_local_cleanup=false（无明确新增，不需要清除）。人工查看也确认坐姿、手脚接触、椅子、两只鞋、留白、脸和衣装基本保留，原有软边/色层/绘画质感未变成硬边平涂。

但 `line_repair.judgment=unchanged`，审计未确认局部线修收益；它认为变化可能主要来自放大。不能因此把“内容通过”宣传为“线条问题已解决”，也不能把模型的放大判断当作已证明的像素来源。将结果Lanczos缩到原图1536×1024后的全图平均RGB绝对差为5.867/255、RMSE11.487/255；未做几何配准，这些像素差不能评价语义或线条连贯，也不能证明只是放大。数值与边界保存在 `source-only-long360-pixel-comparison.json`。

新增记录：`source-only-long360-request.json` / `source-only-long360-log.txt` / `source-only-long360-result.json` / 双图审计请求、原文和JSON，全部同目录。新增1次图片＋1次文本请求，本实验目录累计4次图片尝试（2成功、2无图扣费未知）＋2次文本审计。原图、所有失败与成功产物均保留；未清除、未再生图、未启用默认固件/自动流程、未重启App。

当前更新结论：仅源图方案在这一张上成功保真，修线改善仍无明确证据，且只有一次有效视觉样本。下一步若继续需比较更明确但限定区域/轮廓的修复要求，并复跑确认保真与修线同时成立；无需再清除不存在的新增元素。


## 自动工作流接入（恢复损坏会话最后工作项）

用户随后明确要求将当前方案接入现有工作流。本次完成此前中断的接线：

- GPT 生图/编辑 Tab 的自动重绘改用通用源图保真固件；不再附第二张画风图，不使用生图正文和旧平涂渲染条款。手动重绘仍按用户编辑框指令执行。
- 图片分析 GPT 链的 `pipeline_steps_from_flags` 默认启用 `content_guard`，有效请求组装与实际发送均使用同一份仅源图请求。`content_guard=False` 保留旧参考模式供显式实验。
- 每张首轮重绘一次，单次请求360秒、图片自动重试0；超时/异常/无图/比例错误保留候选并逐张回退输入图。仅为本次调用传参数，公共API配置不改。
- 双图审计固定 `[原图, 重绘结果]`，180秒文本请求。人物、姿势、构图、主要物件、衣装及源图画法须全部通过；缺失、全局替换、低置信、无效报告、审计异常均回退输入图，不清除。`line_repair=unchanged` 仍可通过内容审计，不代表修线有效。
- 仅明确局部新增允许一次清除：Image 1=原图，Image 2=重绘底图；清除请求360秒/重试0/重复1，不递归启用内容保护。再与原图审计一次，只有无新增且完整通过才采用，否则回退输入图。
- 原图、首轮候选、清除候选均保留。候选旁写 `*-content-audit.json`，含审计prompt、原始回复、解析报告、选择路径及清除prompt/产物。清除拥有独立预算run，避免和首轮预算冲突。
- 后续既有衣装、色调、加墨、质量/身份/人体门禁继续按原勾选运行；回退的是重绘工序输入，并不绕过这些既有工序。
- UI说明同步；自动重绘估算包含最多2次文本审计和1次条件清除，按上限展示。图片分析总等待窗口另预留1080秒覆盖两次360秒图片请求和两次180秒审计。

固件将原实验中具名人物、椅子、鞋子的清单泛化为从实际输入图检查内容，不把测试图场景硬编码到其他任务。最新实现主要在 `utils/repaint_content_guard.py`、`utils/post_process.py`、`utils/analysis_gen.py` 与两个Tab接线。未发新增收费生成/审计请求，未重启App。**重启后新任务生效；已有成功断点按原有续跑机制复用，需显式从GPT首图重绘才能重新应用。** 批量视觉稳定性与修线收益仍需后续实际图片判断。

验证：新增 `tests/test_repaint_content_guard.py` 覆盖全局替换拒绝清除、严格报告与介质判定、清除一次/复核、取消、超时与异常、多图回退、真实请求超时/重试参数及公共配置不变、有效请求与估算。连同 analysis_gen、GPT UI/API、post_process、wardrobe、analysis_channel、style_merge_validation、style_gpt_prompt、output_isolation 共9组 **396 passed**；7模块编译通过。Windows沙箱TEMP的原子重命名曾拒绝访问，换工作区内TEMP和pytest临时目录后全部通过；产物均在 `data/test-result/20261007/`。
