# Round3 人物与试验计划

## 准备完成情况（2026-10-05，DeepSeek 交付，**未启动在线试验**）

材料与工具已按本文件 §4/§5/§6 的要求做完，**在线试验仍未开始**（本轮请求即为此准备任务）。
就绪结论绑定 spec 指纹 `a03a619868437fdfe157e8ed1ad63ceac5a48bb9c58972739cb1763f59dc682a`，
素材任何改动都会让它失效。

| 交付 | 位置 | 状态 |
|---|---|---|
| 参数化 spec | `prompts/wardrobe/round3/spec/plan.json` | 6 用例 × 2 版本 × 2 通道，32 槽位；含 allowed 枚举、预算、配对要求、冻结生图参数 |
| 人物卡 | `prompts/wardrobe/round3/characters.json` | 6 名原创成年女性（未改） |
| 分支/画幅/规则模板 | `prompts/wardrobe/round3/variants/` | preserve / redesign-v3 / redesign-v4 / visible-v3 / visible-v4 / visible-v4-redesign + design + 3 个 policy + frame-full(-v4)/frame-bust(-v3/-v4) |
| v4 定向修订 | `prompts/wardrobe/round3/characters/T01..T06.md` | 只加强指定衣物/配饰位置与近景构图，不改画法正文 |
| **完整两臂快照** | `prompts/wardrobe/round3/pack/*.txt` + `manifest.json` | 12 份（6 用例 × v3_control/v4_candidate），base-pack 只是共同基座、**不是**最终请求 |
| 逐处差异 | `data/test-result/20261005/wardrobe-round3-prep/arm-diff.json` | 段落级 only_in_control / only_in_candidate + 字符差 |
| 离线自检 | `.../offline-suite.json`、`.../prepare-report.json` | spec 校验、确定性检查、配对平衡、候选失败项 |
| 候选门禁 | `.../candidate-gate.json` | 现状 `ready=false`（审查与校准尚未执行），**所有生成入口已被拦住** |

`base-pack/manifest.json` 声明的 6 份 hash 与实际文件逐份一致（未改动）；两臂都以同一份
`T0x-base.txt` 开头，差异只在分支块、v4 的 tailoring 段与画幅收尾。

| 用例 | v3_control 字符/哈希前 12 位 | v4_candidate 字符/哈希前 12 位 | 字符差 |
|---|---|---|---|
| T01 | 1393 / `0df40133869c` | 2061 / `93cbed77608d` | +668 |
| T02 | 1336 / `5f03bb131146` | 1923 / `804cc217aa8b` | +587 |
| T03 | 2092 / `7abec7ba5b15` | 1598 / `ac1987d26499` | −494 |
| T04 | 2369 / `8d7e9cfb9ef4` | 1843 / `227de54efae4` | −526 |
| T05 | 1088 / `f306f9cf328b` | 1325 / `9dadb82d5c32` | +237 |
| T06 | 1184 / `3c3dcebed27a` | 1581 / `eed5a6eeda79` | +397 |

确定性检查：**v4 全部 6 例 strict 通过**（保留分支无目标衣装设计块、近景只写可见区域且带明确裁剪下缘、
无评价字段泄漏、无占位符）；v3_control 在 T05/T06 报 `crop_boundary_explicit`——这是预登记缺陷，
按 spec 只报告不阻塞（`severity=defect_expected`）。

### 工具侧修了什么（对应 ROUND2 复核 §1–§5）

1. **评价输入**：`score.md` 改为逐字段输出，`cmd_score` 现在把 identity / source_clothing / user_request /
   protected 原样发进请求，并按 spec 的 case 条件组装；**缺任何一项就记 `not_scored_missing_conditions`
   并拒绝发送**，杜绝「借对照图发明锚点」。
2. **候选门禁**：`candidate_readiness()` 统一判定，`--generate`、`--score`、`--run`、`--calibrate`
   任一入口在 `prepare_ok` / spec 指纹 / 候选审查 / 校准任一不满足时直接返回 1 且**不占预算**；
   审查提示词显式声明 `v3_control` 是冻结控制，其已知缺陷只报告，不再被误判成「多个候选」。
3. **参数化汇总**：`tools/wardrobe_report_v2.py` 从 spec 读用例条件与条数，不硬编码 C01..C06；
   评分口径按该轮实际 score 模板选择（round3 六字段 / 历史轮次三字段）；
   `pass / failed / unproven / missing` 分开统计，缺字段与 uncertain 一律 `unproven`。
   用 round2 运行目录回归：12 pass / 2 failed / 2 unproven，与 ROUND2 复核的数字一致。
4. **A/B 精确平衡**：配对不再随机抽，改为每对 control/candidate 各占 A 一半 → 16 对里 **8/8**，
   `--pairs` 与评分入口都在不平衡时拒绝运行。
5. **校准规则化**：校准判据写在 `spec/calibration-cases.json` 的 `validation` 段（3 例：类别、
   裁剪宽度、颜色与配饰），**没有规则判失败**，不再用关键词粗糙放行。
6. **离线回归**：`tests/test_wardrobe_experiment.py`（34 条）+ `tests/test_wardrobe_report_v2.py`（9 条），
   覆盖缺字段/错枚举/槽位数不符/A-B 不平衡/门禁四级/缺规则校准/缺少评分条件等失败路径；43 条全通过。

详细就绪报告见 [LOLITA-WARDROBE-ROUND3-PREP-READINESS-20261005.md](LOLITA-WARDROBE-ROUND3-PREP-READINESS-20261005.md)。

### 仍需在线完成后才能判定的部分

候选 `ready` 目前为 **false**：本轮**未**执行（也不被本准备指令授权执行）文本审查与看图校准，
因此 §8 的 32 次生图、16 组评价全部未运行，预算账本为零：文字 0/2、生图 0/32、带图 0/20、
模型清单 0/1。`--prepare` 与 `--compose-pack` 都不发请求；`--generate` 已实测被门禁拦下（退出码 1、未扣预算）。

## 人物安排

日期：2026-10-05。**计划阶段，未启动在线试验。** 依据LOLITA-WARDROBE-ROUND2-REVIEW-20261005.md，生产模板不改。

| 人物 | 固定身份 | 本轮任务 |
|---|---|---|
| P01桥接 | 24岁、棕长发绿眼 | 保红校服；明确白衬衫、红裙、胸针位置，不让评分另加条件 |
| P02短银发眼镜 | 29岁、暖棕肤色、短银发、黑框眼镜、结实体型 | fill_missing保工装；检查是否改长头发/去眼镜 |
| P03深肤卷发 | 30岁、深棕肤色、赤褐卷发、琥珀眼、宽肩丰满体型 | 未写衣服→补全Lolita；保肤色、卷发和体型 |
| P04成熟黑发 | 38岁、波浪黑长发、蓝眼、成熟面孔 | 黑西装→Gothic；保月牙胸针、右手书，不幼态化 |
| P05蓝短发眼镜 | 26岁、蓝色短发、灰眼、圆铜框眼镜 | 严格上胸以上补全；不扩图、不去眼镜 |
| P06紫色束发 | 28岁、紫色高马尾、紫眼 | 橙卫衣→局部改款；保橙色和左上胸蓝鸟胸针，不扩近景 |

准确英文身份/衣物/用户要求与protected列表在prompts/wardrobe/round3/characters.json；base-pack提供六份共同基础文本，**还不是完整v3/v4生成请求**。均为原创成年女性，无现实人物或作品角色。年龄数字不能凭脸精确验证，只检查适用的成年/成熟表达，不能把“看着25而不是26”判fail。

每名人物一类任务，不覆盖所有人物×所有策略；多属性同时变化是人物组合探索，不能归因于某单一属性。没有源图，不验证同脸身份和原剪裁逐细节一致。

## 先修评价和门禁

1. 评分输入必须带identity/source_clothing/request/protected准确条件，隐藏variant与期待赢家。每图独立检查；不得向另一图借白衬衫、格纹或肤色要求。
2. spec显式区分v3_control与v4_candidate。控制已知缺陷仅报告；候选ready=false、prepare失败、校准失败必须拦住--generate与--run所有入口，不靠分阶段命令绕过。就绪结果绑定spec和prompt hash，变更后失效。
3. wardrobe_report从spec读取case条件/条数，不硬编码旧C01..C06。缺字段、错枚举、unknown不能默认pass。A/B配对预先精确平衡8/8，统计pass/fail/uncertain/missing分开。

P01/P02明确衣物细节是**本轮新要求**，不能倒推round2已指定白衬衫/格纹。两臂基础输入相同，v3_control用round2真实模板按gold分支组装；v4只加强具体衣物/配饰位置和近景构图，不改画法正文。全身改款继续相同Lolita设计块；preserve分支不送该块；近景只写可见领口/局部衣片，不发裙鞋指令。

先由DeepSeek制作参数化spec、完整两臂快照与hash、差异和离线测试，再进入试验。**本次规划材料不代表候选v4已经实现或审查通过。** 生图采用预登记gold分支，不称自动解析端到端验证。

## 建议规模

六用例×两版本×GPT/Gemini各一次=24张；T01保留桥接与T05严格裁剪各多一次=8张；总32次生图，16组配对。其他人物每通道每版本仅一次，是问题筛选，不宣称稳定泛化。

建议文字最多2次（候选审查1＋修订后审查预留1），带图最多20次（已有图校准2＋16配对＋2备用），模型清单最多1次。所有失败/超时/重试/备用/探针发送前计数并持久保存；未知结果也消耗额度，不重启run-id重置。候选第二次审查仍失败则停止，不生成。隐藏重试须禁用或同一账本计入。

冻结同站点/模型/画质/尺寸，参数从round2记录读取；模型不可用停止。无画风参考图、源图编辑、重绘/人体修订、色调收尾或发布。旧图只作历史参照和校准，不冒充新人物的同场控制。

## 评价标准与交付

评价逐字段观察：人数、成人/成熟表达、发型发色、眼色眼镜、肤色与可见体型、衣装行动、指定衣物/颜色/配饰位置、持物、场景、实际下边缘。未知和not_visible分开，不推断藏住的身体；close图完整衣装不可验证，不因缺裙鞋扩图。年龄精确数字与无源图同脸一致不作为可观察通过条件。

保护fail不能被漂亮衣装抵消；uncertain不算通过，但也不改成生图失败。允许两臂都通过、无改进或候选退化。每个人物条件少，不做群体层面结论。任何生产部署仍需另行评估，本轮不部署。

交付32次请求/产物与hash、16组匿名评价/标签/内容块顺序、实际全条件输入、预算账本、离线图库和LOLITA-WARDROBE-ROUND3-RESULTS-<日期>.md。生产模板/GUI/全局配置不变，不自动开始round4。

## 给 DeepSeek 的准备 Prompt

```text
请先读D:\code\image-maker的PROJECT_REQUIREMENTS.md、最新AGENTS.md，以及docs\style-extraction\LOLITA-WARDROBE-ROUND2-REVIEW-20261005.md和WARDROBE-ROUND3-CHARACTERS-AND-PLAN-20261005.md。

先准备Round3，人物使用prompts\wardrobe\round3\characters.json的六名原创成年女性。保留年龄表达、肤色、发型、眼镜、体型和标志配饰。制作参数化spec、v3_control/v4_candidate完整prompt快照、hash与差异，不把base-pack当成最终生图提示词。

先修实验工具的评价输入、候选门禁和参数化汇总：评分必须收到真实人物/服装描述及protected清单，不能借对照图发明锚点；候选ready=false必须拦所有生成入口；冻结控制身份明确，不因已知缺陷误判多个候选；按spec验证case字段/数量，A/B精确平衡8/8。加入有意义的离线回归，禁止默认缺字段通过。

本轮计划是32次生图、20次带图、2次文字、1次模型清单HTTP上限，失败、重试、探针全计；使用gold授权，不证明自动解析端到端。先完成材料与离线测试、报告就绪情况，此准备指令不启动在线试验。生产模板/GUI/全局配置不改，不发布，不自行增加人物或预算。
```
