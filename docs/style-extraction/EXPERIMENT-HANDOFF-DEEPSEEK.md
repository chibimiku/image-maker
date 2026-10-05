# 给 DeepSeek 的任务书：独立 Lolita 衣装 pilot

日期：2026-10-05。状态：**提示词已整理，实验未执行、未自动发送给 DeepSeek**。沿用项目现有“文件任务书＋复制提示词”的交接方式。DeepSeek 作为执行助手，调用项目既有生图接口；不能把向纯文本 DeepSeek API 提问当成实际生图测试。

## 目标与材料

检验独立衣装控制能否在没写衣服时补全 Lolita、把原服装改款、保留明确指定的颜色配饰，同时尊重保留原衣服的要求。本轮不训练模型，也不实现多图衣装提取。

依次读取 PROJECT_REQUIREMENTS.md、AGENTS.md、`LOLITA-WARDROBE-DESIGN-20261004.md`、`WARDROBE-IMPLEMENTATION-20261005.md`（后两者在本目录），然后读：

- `prompts/wardrobe/` 下的 presets、application、三策略、lolita、continuity、identity-audit、audit。
- `prompts/wardrobe/test-plan.json`：六个预登记用例和预期。
- `prompts/wardrobe/deepseek-review.md`：文本语义审查。
- `prompts/wardrobe/test-score.md`：匿名视觉评分。
- `utils/wardrobe.py`、`utils/analysis_gen.py`、`tests/test_wardrobe.py`、`tools/gpt_image2_gen.py`、`tools/analysis_gpt_run.py`；需要 Gemini 直出时读 `modules/others/api_backend.py`。

当前实现已通过19个衣装离线用例和7个既有CLI用例；尚未在线验证。衣装生成预设是手写规则。GPT分析完整链有衣装终审，Gemini直出和gpt-image专用页没有新增自动衣装终审。衣装关闭默认沿用旧流程。

## 阶段0：先审查再花费生图调用

按 deepseek-review.md 对实际规则做审查，保存原始响应。重点识别 application 的“explicit garment requirements”是否会把 C02 的旧校服/C05 的旧西装锁死。原服装事实不等于“必须保留”；明确 keep-original 则优先。不要因为两者都写在同一条最终 prompt 就忽略冲突。

有阻塞项：先写 v2 候选到实验目录，记录原版/候选的逐处差异并做离线检查；不覆盖生产模板。只有一个有依据的候选时按该版本执行本轮，原版不额外生图；若候选之间需要在线筛选，停止并提出预算替代方案。不能把全局关闭身份门禁作为解决办法。

## 阶段1：六例 × off/on × GPT/Gemini = 24个逻辑生图样本

每组一次，这是筛选试验，不能据此宣称稳定泛化。C01补全，C02原校服改款，C03明确保留，C04只补全保留，C05替换且保黑色/月牙胸针，C06半身裁剪与可见性。C03/C04是负向控制，C06不证明整套衣装符合。

统一画法文字取 spec 的 rendering；实际内容取各 case 的 content。**off = rendering + 两个换行 + content；on = apply_wardrobe(off, build_wardrobe_spec('lolita', case.policy))**。默认不另加通用换装、画风保护或年龄模板，不叠加旧 cute-lingerie-wardrobe。若阶段0采用候选，使用有 hash 的候选 spec，并记录与运行时原版不同。

预展开文本与 manifest 在 `prompts/wardrobe/test-pack-v1/`；它们是原版运行模板的准确快照，尚未执行。调用前重新核对 hash，避免把后续模板改动混入同一轮。不得把 expected_on/protected/评分标准拼进生图请求，它们只用于评价。

模型从现有配置与可用列表核验，固定一个 GPT image 模型和一个 Gemini image 模型，记录具体 ID。不要硬编码或悄悄替换不可用模型。无可用端点/凭据时交付预检报告，不能编造已执行。GPT用1024x1536、n=1；Gemini用2:3；画质/分辨率从现有配置读取并在全组固定。无画法参考图、衣装参考图、重绘、tone、线条收尾或发布。

复用 `tools/gpt_image2_gen.py --prompt ... --dry-run` 预检 GPT；在线仍用该 CLI 或项目后端。**此CLI的 --prompt-file 是“每行一条任务”，不能直接读本包的多段单条 prompt，否则会拆成多次生图。** PowerShell单例预检：

```powershell
$wardrobePrompt = Get-Content -LiteralPath 'prompts/wardrobe/test-pack-v1/C01-lolita.txt' -Raw -Encoding UTF8
& 'C:\Program Files\Python310\python.exe' tools/gpt_image2_gen.py --prompt $wardrobePrompt --size 1024x1536 --dry-run
```

此命令未在线生图。不要把Python命令套管道或重定向。Gemini复用 `generate_image_aigc2d`，关闭自动face_quality_boost，不要混入细节增强固件。如果需要批次编排，扩展现有同族CLI，禁止新建单次实验脚本。不要使用 analysis_gpt_run 的完整链来假装比较首图：那会带入修订、门禁和额外调用。

设置 `IMAGE_MAKER_TEST_OUTPUT=1`，按 `utils/output_isolation.py` 的真实行为预检目录，所有样本落 `data/test-result/<执行日期>/wardrobe-pilot-<run-id>/`；每个样本单独ID、请求和结果，避免反复覆盖输出文件。保留失败，不额外补图挑赢家。成功联网请求缓存；底层重试和逻辑调用分别记录。发出最多24个逻辑生图请求，不执行扩展或重绘。

## 阶段2：匿名评价与交付

每 case/channel 一对，匿名随机A/B后按 test-score.md 评价，共12次视觉调用；反转位置复核C02和C05的两通道，共4次，**视觉调用上限16次**。保留匿名映射与原始响应，评分后再揭盲。DeepSeek端点若仅支持文本，不把base64当作已看图；视觉评价使用已配置、确认支持图片的分析端点。没有可信视觉能力则保留图片和待人工评价清单，不编造分数。

与 off 比较目标符合度及非衣装漂移，允许平局；保留不符合、请求失败、不确定和不可见项。C03/C04不能按缺少Lolita判失败。C06局部得分与“完整衣装不可验证”同时报告。现有 wardrobe/audit.md 可以审计 policy，但不能代替匿名对照，也不在本轮额外请求预算里自动增加调用。

交付准确模型/配置摘要（不含key）、模板hash和长度、24条实际请求/尝试/产物、12组可离线浏览的并排图、全部原始评分及位置复核、分通道描述统计，以及 `docs/style-extraction/LOLITA-WARDROBE-PILOT-RESULTS-<日期>.md`。报告明确哪些有效、哪些失败、是否改变了人物/颜色/场景，以及下一轮最小建议。

不在线执行源图编辑、有图画风兼容、重绘连续性、多人或自动提取。本轮首图符合并不能证明这些流程安全或有效。完成后停止，不修改全局配置、生产提示词、发布图片或覆盖工作区其他人的修改。

## 直接复制给 DeepSeek

```text
请在 D:\code\image-maker 中执行独立 Lolita 衣装测试。先完整阅读 PROJECT_REQUIREMENTS.md、AGENTS.md 和 docs\style-extraction\EXPERIMENT-HANDOFF-DEEPSEEK.md，按任务书执行阶段0提示词审查、阶段1的24个逻辑生图样本（六例×off/on×GPT/Gemini，各一次）和阶段2至多16次匿名视觉评价。

使用 prompts\wardrobe\test-plan.json 和现有 utils.wardrobe 的真实提示词组装，核对 prompts\wardrobe\test-pack-v1\ 中的原版快照。重点检查没写衣服时补全、旧校服改款、明确保留原衣服、只补全策略、Gothic黑色与配饰、半身裁剪。先解决“原服装描述”与“显式保留衣物要求”的语义冲突，禁止用关闭身份门禁掩盖问题。

核验现有模型和凭据后冻结模型ID；不得自动换模型、追加生图、补图选赢家或扩展到重绘。测试产物隔离到data/test-result，成功调用缓存，失败原样保留。DeepSeek纯文本审查不等于生图/视觉测试；视觉端点不可用时明确留待人工评分。

不发布、不改全局配置、不覆盖生产模板或已有工作区修改。交付真实请求、产物、匿名评分、对比图库和结果报告，允许无效或不确定的结论。源图编辑、有图画风和重绘连续性留作下一轮，不宣称本轮已经验证。
```
