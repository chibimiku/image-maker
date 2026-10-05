# DeepSeek 任务：续跑 Round3 在线实验

请在 `D:\code\image-maker` 继续执行已经准备好的 Round3 衣装实验，完成在线审查、校准、对照生图、评分和报告。不要停留在计划或离线验证。

## 用户授权与范围

用户在本次对话中明确答复「允许」，允许向已配置的 `new.aigc2d.com` 发送本轮人物设定、提示词、实验计划及校准图片，并在审查、校准通过后按既定预算生图。用户随后要求把执行交给 DeepSeek。这份任务由用户转发给你，作为该授权的上下文。

此前 Codex 的提权联网请求被自动审批拒绝，原因是缺少对该具体目的地及材料发送的明确授权；上述用户答复已补充授权。若你的执行环境仍拒绝，按正常权限流程说明此授权；不得绕过安全拒绝。授权不允许跳过实验质量门禁或扩张预算。使用现有配置和密钥解析，不打印或上传密钥文件，不更换外部服务。

## 先读取的材料

1. `PROJECT_REQUIREMENTS.md` → `AGENTS.md`。
2. `docs/style-extraction/LOLITA-WARDROBE-ROUND3-START-CHECK-20261005.md`。
3. `docs/style-extraction/LOLITA-WARDROBE-ROUND2-REVIEW-20261005.md`。
4. `prompts/wardrobe/round3/spec/plan.json`、`review.md`、`calibration-cases.json`、`calibrate.md`、`score.md`。
5. `tools/wardrobe_experiment.py`、`utils/wardrobe_experiment.py`、`tools/wardrobe_report_v2.py`，以及现有实验目录的 state、ledger、prepare-report。

旧 PREP-READINESS 报告早于 Codex 修正，不得按它恢复错误的控制臂。当前 spec SHA256 为 `9f30cc71d252848f4cd55728c8e46898fdb204e1c8fe3fe862a98d99c11c3f17`；实际文件如有后续变化，先查明来源并重做相应离线检查，不能沿用旧就绪结论。

## 当前真实状态

- spec：`prompts/wardrobe/round3/spec`；完整生图快照：`prompts/wardrobe/round3/pack`。base-pack 只是正文底稿，不可直接当完整请求。
- 唯一 run-id：`wardrobe-round3-prep`，目录：`data/test-result/20261005/wardrobe-round3-prep`。保留所有原记录。
- 12 份最终提示词、32 个生成槽位、16 对比较，A 为控制/候选各 8 对；离线 prepare 通过。
- 最近验证：实验工具 36 项 unittest + 报告工具 8 项 unittest，共 44 项通过；修改 Python 已编译检查。
- 一次文本审查连接失败（WinError 10013），未收到模型回复，已保守计入账本。后续提权被拒没有发 HTTP。不代表文本审查通过。
- 在线审查、看图校准和生图均未成功执行。未登记视觉模型。

控制臂是真实 Round2 v3 框架：保留分支不携带目标设计块；近景已经有明确上胸裁剪。不能再声称 v3 有 v2 的“保留仍发整套设计、近景无下缘约束”缺陷。T06 是新增可见区域改款用例，控制规则有最小授权适配，不能声称已在 Round2 验证。CAL-CROP 的本地预期是 A 较宽、长衣身向腰部延伸，B 较紧上胸；两张图是独立生成，不能称同一身份。

## 预算与计数

以现有 ledger 为唯一账本；启动前核对，发现已有新增操作时按实际扣减。

| 操作 | 全轮上限 | 当前已用 | 当前剩余 |
|---|---:|---:|---:|
| 文本 HTTP | 2 | 1 | 1 |
| 生图 HTTP | 32 | 0 | 32 |
| 视觉 HTTP | 20 | 0 | 20 |
| 模型列表 HTTP | 1 | 0 | 1 |

视觉分配：3 次校准 + 16 次双图评分 + 1 次预留。文本剩余 1 次用于审查，不再跑 resolver。模型列表只在确有必要时调用，不能另加收费探针。每次发送前扣减；失败、超时、重试、备用及未知结果都占预算。禁止换 run-id、删 ledger、改上限或开启隐式重试来补齐结果。生成失败时保留缺失槽位，不保证能得到 32 张成功图片。

## 执行顺序

系统 Python 唯一入口是 `C:\Program Files\Python310\python.exe`，长任务加 `-u`，不要给 Python 套管道或重定向。每次只执行一个阶段，检查结果再继续；禁止 `--run`，它会重复调用 resolver/review/calibrate。

PowerShell 通用参数（2026-10-05 当天）：

```powershell
$wardrobePy = 'C:\Program Files\Python310\python.exe'
$wardrobeArgs = @('--spec','prompts/wardrobe/round3/spec','--pack','prompts/wardrobe/round3/pack','--run-id','wardrobe-round3-prep')
& $wardrobePy -u tools/wardrobe_experiment.py @wardrobeArgs --status
```

注意：CLI 输出目录用当天日期。若跨天执行，同一 run-id 也会创建新目录！先在现有同族工具中做最小、可测试的“指定已有 run 目录”恢复支持，再执行；所有阶段必须读写上述 20261005 原目录及原 ledger，不能迁移成一份新预算。

1. 核对 prepare、hash、快照、账本。现有材料正确时不要无意义重组 prompt。必要修复只能在现有实验工具内完成，留 diff 与测试记录。
2. 运行一次 `--review`，检查原始模型回复、严格布尔 `ready=true`、无 blocking、无 HTTP/解析错误。审查不通过或文本额度耗尽就停，不得手写 ready 或把用户联网授权解释为跳过门禁。
3. `--pick-vision-model` 本地登记配置中的视觉评价模型；沿用前轮已确认支持读图的模型，记录实际名称与来源，不做额外探针。
4. 运行 `--calibrate` 三组实际图片。除脚本返回外，检查原始观察、A/B 顺序、源图 hash、衣物类别、裁剪方向及银色月牙配饰。CAL-CROP 必须读出 A 比 B 更宽；仅“两个下缘字符串不同”不充分。不得将本地 gold 预期发给评价模型。
5. 校准任一失败就停。只有确有必要且总视觉额度允许时，才能用剩余 1 次预留对失败项单独重试；不得把整个三项校准重跑当成一次。成功项缓存复用。若工具尚不支持单项恢复，先在原工具实现并验证，不另建实验脚本。
6. `--gate` 必须 ready=true，且审查/校准针对当前材料和登记模型。随后 `--generate` 执行预登记 32 个槽位，冻结 GPT/Gemini 参数，不加参考图、编辑、重绘、色调校准或后处理。
7. `--pairs`：核对 16 对及 A/B 8:8 平衡，失败槽位如实缺失。`--score`：逐项检查请求包含人物身份、源衣物、用户要求、protected 和裁剪条件；必须附实际两张图，不得仅凭文字评分。每张独立对条件判断，不能从配对图借衣物或颜色。
8. 离线生成汇总与可查看图集：

```powershell
& $wardrobePy -u tools/wardrobe_report_v2.py --run-dir data/test-result/20261005/wardrobe-round3-prep --spec prompts/wardrobe/round3/spec
```

各阶段调用形式为 `& $wardrobePy -u tools/wardrobe_experiment.py @wardrobeArgs --review`，然后分别替换为对应单个阶段参数。不要拼成无条件自动继续的命令串；必须检查退出码与实际结果。

## 交付要求与停止条件

保留请求快照、原始响应、图片、源图及产物 hash、A/B 映射、校准证据、评分 JSON、预算与失败记录。完成后写 `docs/style-extraction/LOLITA-WARDROBE-ROUND3-ONLINE-RESULT-20261005.md`，给出图集及 summary-v2.json 路径。

报告按 GPT/Gemini 和 T01～T06 分开比较控制/候选：衣物保留、未指定衣装补全、Gothic 改款、肤色/成熟年龄/体型/眼镜、配饰位置、全身与上胸裁剪。区分 pass、failed、unproven、missing，不把身份不可验证算通过。说明授权用预登记 gold 路由，不能宣称自动解析端到端成功；不能称手写规则是训练或多图衣装提取产物。

任何阻塞都必须记录具体阶段、错误、预算和可恢复点。预算耗尽、审查不通过、校准不通过或权限仍被拒就停止依赖阶段，交付现有证据；不要绕过、扩预算或编造完成结果。所有联网成功操作应缓存复用，未知请求结果不得盲目再发。

不修改生产衣装模板、GUI、全局配置或其他人的工作，不发布图片、不部署候选。当前任务止于实验与报告；不要擅自把 v4 推进生产。
