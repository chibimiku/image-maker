# 独立穿衣风格（当前入口与研究索引）

当前版：2026-10-06。衣装默认关闭；选中预设后默认完整转化。衣装只提供服装设计，不附带画风图、艺术八字段、画幅或调色目标。艺术画风是用户另外选择的可选功能，不是衣装功能的依赖。Round5 单独衣装主线效果在该批样本中成立，艺术组合的肤色/背景漂移是独立兼容限制。

## UI 与逻辑链

设置 → 穿衣风格：查看有效规则。图片分析与 gpt-image 专用页：各自选择并记忆衣装，支持 Lolita 总类、古典/甜美/哥特/中式/日式与成人内衣。分析页“分析服装覆盖”是旧分析选项，与生成衣装不同。专用页生成/编辑可应用衣装，独立重绘置灰。

`replace`（完整转化）：换服装类别，但尊重明确保护；`reinterpret`（改款）：保留原服装类别与层次，只用兼容剪裁和装饰；`fill_missing`（只补全）：衣装未写且源图未给出时补全，否则保留。显式保留原衣装优先于目标词汇。成人内衣保持旧规则“正文明确至少 21 岁”的条件；不符合时不改变年龄或改成内衣。

模板 `prompts/wardrobe/` → `utils/wardrobe.py` 生成正文/hash 快照 → 提交时冻结 → 分析 GPT/Gemini 或专用 GPT 生图。源分析事实不被改写，衣装不发送用户训练目录图片。艺术画风不能豁免所有身份门禁；修订/重绘只保住当前衣装，不重新执行旧服装改换。关闭衣装保留原请求路径。重复注入同一快照不再追加第二份。

分析 GPT 全流程：最终衣装只读审计，失败保存候选并禁止正常自动发布，成功调用由断点复用。分析 CLI 全流程也有终审，`--first-pass-only` 按原含义只出首图。Gemini、专用 GPT 与直接 CLI 生成没有自动衣装符合度终审，成功返回不是审核通过。专用 GPT 和直接 CLI 在产物旁写 `.wardrobe.json`（请求正文/hash、衣装快照、图片 hash、阶段、effect_reviewed=false）；分析流程使用原任务/生成清单/断点记录。

历史：旧任务沿用原正文/hash，不自动改成新模板。旧 `cute-lingerie-wardrobe` 配置保留给历史，从新任务艺术列表隐藏；新任务选独立成人内衣。旧艺术衣装条目与新衣装目标同时使用会被拒绝。

## CLI

系统 Python `C:\Program Files\Python310\python.exe`；以下命令从项目根目录运行，不套 shell 管道或重定向。

```powershell
python tools/gpt_image2_gen.py --list-wardrobes
python tools/gpt_image2_gen.py --prompt "A long-haired woman." --wardrobe lolita-classical --dry-run
python tools/gpt_image2_gen.py --site autodl --prompt "A 32-year-old adult woman." --wardrobe cute-lingerie --wardrobe-policy replace --dry-run
python tools/gpt_image2_gen.py --mode edit --image <源图> --prompt "Keep the original outfit." --wardrobe lolita-sweet --wardrobe-policy fill_missing --dry-run
python tools/analysis_gpt_run.py --json <分析JSON> --wardrobe lolita-sweet --wardrobe-policy replace --dry-run
```

直接 CLI 默认 `--wardrobe off`，支持两站点、生成/编辑、重复 prompt 与 prompt-file。分析 CLI 未指定 `--wardrobe` 时沿用分析 JSON 的 generation_wardrobe 快照，明确 off 才关闭。新预设默认 replace，与 UI 一致。独立 `--repaint` 不接受开启的 wardrobe，避免重绘意外再次改衣装。直接 CLI 没有新增自动重绘链或自动审计承诺。

实验 CLI：`tools/wardrobe_experiment.py --frozen-pack <包> --run-dir <目录> --prepare/--generate/--status`。prepare 无 HTTP；generate 只执行冻结首图；成功复用，已尝试失败默认不重发。用户追加授权的补跑可记录 --grant-frozen-budget generation / --grant-limit / --authorized-by / --grant-reason / --grant-source / --retry-slots，授权是有效授权。槽位账本不能代替实际 HTTP 次数，传输端点回退需单独报告。旧 Round4 spec/requests 是规划草案，不得作为已实现提取链运行。

## 研究与证据

| 阶段 | 材料 |
|---|---|
| 设计边界与可复用提取框架 | [设计研究](LOLITA-WARDROBE-DESIGN-20261004.md) |
| 早期试验 | LOLITA-WARDROBE-PILOT-*、ROUND2-*、ROUND3-* 文档及 prompts/wardrobe 对应冻结素材 |
| 用户数据集视觉参考 | [参考数据集研究](LOLITA-REFERENCE-DATASET-STUDY-20261005.md)、prompts/wardrobe/reference-study/evidence-review.json |
| Round4 完整转化、五系列 | [效果评审](LOLITA-WARDROBE-ROUND4-EFFECT-REVIEW-20261005.md)，40 条冻结请求、39 张产物；用户授权补跑记录保留 |
| Round5 保留/改款/内衣 | [效果评审](WARDROBE-ROUND5-EFFECT-REVIEW-20261006.md)，28 条冻结请求、25 张产物 |
| 便携研究档案 | [归档清单](wardrobe-research/archive-manifest.json)：去凭据元数据、哈希、计数、结果摘要与效果拼图 |

源训练素材在用户提供的 `C:\data\train\train\lo_v2_train\train\source\img`，完整生图在本地 `data/test-result/20261005/wardrobe-*`，不提交大图原件、密钥、服务原始请求日志或完整训练集。便携档案不会代替原始数据；清单记录原文件路径与哈希，复算需取本地原件，缺图片时 CLI preflight 应拒绝执行。unit tests 只验证历史请求元数据契约，不要求另一台机器具备本机艺术图。

规则是人工视觉参考整理，不是 LoRA 训练产物或自动衣装提取。多图艺术提取框架可复用，但衣装 schema、局部区域与评估维度仍须独立实现。本轮未上线自动提取。历史文档记录当时状态，以本索引和最新评审为准。

## 验证与限制

```powershell
python -m unittest discover -s tests -p 'test_wardrobe*.py'
python -m unittest discover -s tests -p 'test_analysis_cli_unittest.py'
```

以最终暂存快照独立跑回归和 UI 导入冒烟，防止未提交的配色/氛围改动成为隐含依赖。本机未装 pytest，使用 unittest，不声称 pytest 全套通过。本轮不新跑付费图，不中断现有 app；正常统一重启后新模块生效。

2026-10-06 交付复核：暂存快照导出后衣装族 119 条、既有分析 CLI/UI 7 条通过；独立主窗口初始化通过（网络请求被测试替身拦截）。衣装测试用固定本地配置替身与版本化历史档案，CLI/UI 既有冒烟使用本地艺术配置作测试夹具，不包含配色/氛围代码。历史原始指纹按字节保留；为适应 Git 行尾转换，另记 LF 投影指纹用于便携回归，不改运行期 hash。Python 编译与暂存差异检查完成。

线上衣装证据限于首图、单人物和所测条件；真实源图编辑、重绘后衣装审计和多人物泛化未在这些线上轮次完整覆盖。艺术组合仍有肤色/背景漂移，且 GPT 图文对照存在实际端点差异，不能归因于附件单一因素。以上限制不意味着独立衣装需要启用艺术画风。
