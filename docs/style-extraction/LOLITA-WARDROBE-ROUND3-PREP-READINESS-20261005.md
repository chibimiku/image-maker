# Round3 准备就绪报告（2026-10-05）

状态：**材料与离线工具已就绪，在线试验未启动**。本报告只覆盖准备阶段；32 次生图、16 组评价、
2 次文字审查与 3 次看图校准都**没有执行**，因此没有任何在线结论。

- 计划与人物安排：[WARDROBE-ROUND3-CHARACTERS-AND-PLAN-20261005.md](WARDROBE-ROUND3-CHARACTERS-AND-PLAN-20261005.md)
- 上一轮复核：[LOLITA-WARDROBE-ROUND2-REVIEW-20261005.md](LOLITA-WARDROBE-ROUND2-REVIEW-20261005.md)
- 运行目录：`data/test-result/20261005/wardrobe-round3-prep/`
- **spec 指纹（就绪结论绑定它）**：`a03a619868437fdfe157e8ed1ad63ceac5a48bb9c58972739cb1763f59dc682a`

## 1. 交付物

| 类别 | 路径 | 说明 |
|---|---|---|
| 参数化 spec | `prompts/wardrobe/round3/spec/plan.json` | 用例/枚举/预算/配对要求/冻结生图参数；用例条件从 `characters.json` 读，不硬编码 |
| 人物卡（未改） | `prompts/wardrobe/round3/characters.json` | 6 名原创成年女性，含 identity / source_clothing / request / protected |
| 模板 | `prompts/wardrobe/round3/variants/*.md` | preserve、redesign-v3/-v4、visible-v3/-v4/-v4-redesign、design、3×policy、3×frame |
| v4 定向修订 | `prompts/wardrobe/round3/characters/T01..T06.md` | 具体衣物/配饰位置 + 近景构图，不动画法正文 |
| **完整两臂快照** | `prompts/wardrobe/round3/pack/` + `manifest.json` | 12 份完整请求（base + 分支 + tailoring + 画幅），带 hash/字符数/32 槽位顺序/8-8 配对 |
| 逐处差异 | `.../arm-diff.json` | 每用例 control/candidate 的段落级差异与字符差 |
| 离线自检 | `.../offline-suite.json`、`.../prepare-report.json` | spec 校验、确定性检查、配对平衡 |
| 候选门禁 | `.../candidate-gate.json` | 当前 `ready=false`，生成入口已被拦 |

## 2. 核对结果（全部离线）

- **base-pack 未改动**：`base-pack/manifest.json` 的 6 份 hash 与实际文件一致；
  base-pack 只是共同基座，**不是**最终生图请求（`not_complete_generation_requests=true`）。
- **两臂共享基座**：12 份完整请求都以同一份 `T0x-base.txt` 开头，差异只在分支块、v4 的 tailoring 段与画幅收尾。
- **v4 候选 6/6 strict 通过**确定性检查：保留分支不含目标衣装设计块；近景只写可见领口/局部衣片并带明确
  裁剪下缘（“above the bust midpoint”）；无评价字段泄漏；无占位符。
- **v3_control 的 T05/T06 报 `crop_boundary_explicit`**：这是预登记缺陷，按 spec 记为
  `severity=defect_expected`，只报告、不阻塞（这正是本轮要对照的缺陷）。
- **A/B 精确平衡 8/8**（16 对；seed 20261005），32 个槽位 ID 唯一。
- **预算账本为零**：文字 0/2、生图 0/32、带图 0/20、模型清单 0/1；
  `--prepare`/`--compose-pack` 不发请求，`--generate` 实测被门禁拦下（退出码 1，未扣预算）。

## 3. 按任务书修的四处

1. **评价输入**：评分请求现在携带 identity / source_clothing / user_request / protected 与用例条件；
   缺任一条件 → `not_scored_missing_conditions` 且不发请求。评分提示词新增
   person_count / identity_anchors / specified_items / framing / scene 逐字段判定，并明确
   `uncertain 不是通过`、近景 `full_outfit_verifiable=false`、不得向另一张图借衣物或肤色。
2. **候选门禁**：`candidate_readiness()` 是唯一判据，`--generate/--score/--run/--calibrate` 全部连接它；
   就绪状态与 spec 指纹绑定，素材变更即失效。审查提示词写清 `v3_control` 是冻结控制，
   其已知缺陷只报告 —— 不再出现「多个竞争候选」式误判。
3. **参数化汇总**：`tools/wardrobe_report_v2.py` 按 spec 读用例条件与条数；评分口径按该轮
   score 模板自动选择（round3 六字段 / 历史三字段）；`pass/failed/unproven/missing` 分开，
   缺字段与 uncertain 都记 `unproven`。用 round2 数据回归得到 12/2/2，与复核数字一致。
4. **校准规则化**：`spec/calibration-cases.json` 3 例（类别、裁剪宽度、颜色与配饰），判据写在用例里；
   **缺规则判失败**，不允许默认通过。

离线回归：`tests/test_wardrobe_experiment.py` 34 条 + `tests/test_wardrobe_report_v2.py` 9 条，共 43 条全通过
（含缺字段、错枚举、槽位与预算不符、A/B 不平衡、门禁四级、缺规则校准、缺评分条件等失败路径）。

生产侧未受影响：`prompts/wardrobe/*.md`（application / lolita / 三个 policy / continuity / audit /
identity-audit）与 `presets.json` 的 hash 与之前冻结记录一致；本次新增的是 `utils/wardrobe_experiment.py`
（只被实验 CLI 与测试引用，App 不导入，因此不需要重启 app）。

## 4. 在线执行前还要做的（未授权，本准备指令不包含）

1. 登记视觉模型（`--pick-vision-model`，登记不消耗预算）→ `--review`（1 次文字，只剩 2 次额度）
   → `--calibrate`（3 次带图，用已有 round2 图片）→ 门禁变为 ready 后才允许 `--generate`（32 次）
   → `--pairs` → `--score`（16 次带图）。
2. 若候选审查 `ready=false` 或校准失败：**停止，不生成**；第二次审查仍失败也停止（任务书 §4）。
3. 生图参数沿用冻结值（`gpt-image-2` 1024×1536 high png n=1；`gemini-3-pro-image-preview` 2:3 2K，
   face_quality_boost=false），无参考图、无重绘、无色调/线条后处理、无发布。
4. 生图使用预登记 gold 授权路由：**不能**据此声称自动解析到生图端到端可用。

## 5. 明确没做的事

- 没有发任何在线请求（文本 0、生图 0、带图 0、模型清单 0）。
- 没有改生产模板、GUI、全局配置，没有发布，没有新增人物或预算。
- 没有把 base-pack 当成最终生图提示词：`pack/` 里的 12 份才是本轮完整请求。
- 没有自行开始 round4 或任何后续轮次。
