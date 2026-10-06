# 固定色板 → 画风重绘保留验证（修复版 v2）：执行报告

日期：2026-10-04。目录：`data/test-result/20261004/color-palette-retention-v2/`（独立新建，未覆盖 v1 任何文件）。
纠错说明见 [REPAINT-RETENTION-V1-CORRECTIONS-20261004.md](REPAINT-RETENTION-V1-CORRECTIONS-20261004.md)。

本轮只回答两件事：**① 首图的冷蓝环境基调与红色点缀（腰带 + 两只鞋蝴蝶结）在画风重绘后是否保留；② 候选能否穿过现有身份/质量/人体/终审门禁**。不评画质、不排美感、不用全图冷色占比代替区域落点验收。

## 一、结论摘要

- **图片请求实际只发了 2 次**（只生成真正带合同的 B-r1/B-r2），两发两中；旧 A 两张靠**逐项 hash 一致**沿用，没有重新生成。
- **合同确实发出去了**：A 正文 11541 字符、B 正文 13493 字符，B 的正文里含 `SOURCE COLOUR CONTRACT`；后端日志里能直接看到发送的 payload 包含合同全文。四条请求的顶层与发送正文 hash 完全一致。
- **配色保留**：B 两张都是 `conform`（主色 conform、腰带与左右两只鞋蝴蝶结都 present、四项保护色 retained、无新增显著色块、无内容变化）；A 两张里 A-r2 `conform`、**A-r1 是 `partial`**（环境岸地与植被大面积转暖黄绿，且画幅由横变竖）。
- **门禁**：四张都没有达到「全部门禁通过」。A 两张是 `identity pass / quality needs_refine / hands pass / final_review pass`；B 两张是 `identity pass / quality needs_refine / hands pass / final_review fail`（B-r1：人物放大到鞋被画幅裁切；B-r2：线条过重、画风不够通透）。
- 本轮**只生成 B 初始候选并审计，没有做任何门禁修订**；需要修订的地方按约定记「需修订」并停止。

## 二、执行参数与冻结核对

| 项 | 值 |
|---|---|
| 首图 | `P1-knowledge-r1` / `r2`（hash 与冻结值一致：`e25e19c8…`、`05321e89…`） |
| 画风 | `tid` 实际参考图 `data/style-ref/tid-fullbody-background-candidate.jpg`（`188f0312…`）；渲染条款来源 `derived`（10 条）；未用 motif_clauses |
| 模型 / 参数 | `gemini-3-pro-image-preview` / 2K / repeat=1 / 画幅跟随首图 / 无 detail suffix；服务器 `modelVersion` 与请求一致 |
| 底层重试 | 运行前 `IMAGE_MAKER_IMAGE_MAX_RETRIES=0`；发送前预算目录 `send-budget/` |
| A 基线核对 | `retention-a-baseline-check.json`：**status ok**，逐项比对顶层与发送 prompt hash、源图 hash、参考图顺序/mime/传输字节 hash、模型、节点、分辨率、比例、repeat、detail suffix —— 全部一致，因此沿用旧 A 两张 |
| 冻结 | `freeze_hash = e08dff7e641d3cb4…`（含 spec hash、合同正文 hash、画风参考与条款 hash、逐条请求 hash 与发送 hash） |
| 画法例外 | 只适用于新合同；**B 没有继承 M 的「禁止改变画法」**，合同里明确写「首图协议里不涉及颜色的条款一律不继承」。旧 M 未改 |

发送正文 hash（冻结文件里的 `request_sent_hashes`）：

| 请求 | 发送正文字符 | 发送 hash | 含合同 |
|---|---|---|---|
| A-r1 / A-r2 | 11541 | `f3deba1a…dffbe` | 否 |
| B-r1 / B-r2 | 13493 | `08fd2f15…4104c` | **是** |

## 三、逐张结果（配色看区域与落点）

| 候选 | 画幅 | 主色 | 腰带 | 左鞋蝴蝶结 | 右鞋蝴蝶结 | 保护色 | 新增显著色块 | 内容变化 | 疑似泄漏 | 配色总判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| A-r1（沿用） | 1920×2240 竖 | **partial**（岸地/植被转暖黄绿） | present | present | present | 4/4 retained | 2（暖黄绿草地、大面积白色水面高光） | 2（横→竖、环境变亮闪光） | suspected | **partial** |
| A-r2（沿用） | 2528×1696 横 | conform | present | present | present | 4/4 retained | 0 | 0 | unverifiable | **conform** |
| **B-r1（新生成）** | 2528×1696 横 | conform | present | present | present | 4/4 retained | 0 | 0 | none | **conform** |
| **B-r2（新生成）** | 2528×1696 横 | conform | present | present | present | 4/4 retained | 0 | 0 | none | **conform** |

**配色保留率**：`conform` 3/4、`partial` 1/4；**红色点缀与落点 4/4 全部到位**（腰带与两只鞋蝴蝶结都 present）；
**受保护固有颜色 4/4 全部 retained**。

**逐张证据摘录**（完整原文见 `review/colour-A.json`、`review/colour-B.json` 与 `retention-per-image.json`）：

- A-r1：评审写「河流和天空仍保留蓝色，但候选的岸地、远景植被和部分阴影明显转为暖黄绿色……冷蓝环境基调被削弱」；
  泄漏记 `suspected`，归因写「只能确认与疑似参考图的暖绿和浅色倾向相近，无法仅凭颜色相近断定是参考图泄漏」。
- B-r1：评审写「河流、天空、岸地、远景植被与阴影仍以冷蓝为主，未见整体转暖黄绿、紫色或高饱和青绿色替换」，
  并确认「红色仍集中在既有宽腰带及其垂下的蝴蝶结、两只鞋的鞋面蝴蝶结上，未扩散到背景或新增道具」。
- B-r2：评审写「继续维持首图的冷蓝基调」，保护色 4/4 retained，泄漏 `none`。
- 我另外目视核对了两张新 B 产物：冷蓝环境、腰带红结、两只鞋红蝴蝶结、银灰发、象牙白裙、深中性色厚底鞋都在；
  **B-r1 与 B-r2 都是横幅，与首图同为横幅**。A-r1 仍是竖幅、A-r2 仍是横幅。

## 四、门禁结果（现有判据，未关闭、未放宽）

| 候选 | identity | quality | hands | final_review | 门禁后可用 | 审计来源 |
|---|---|---|---|---|---|---|
| A-r1 | pass | **needs_refine**（major：横幅变竖幅、背景漂移、水面高亮青蓝） | pass | pass | 否 | 4 项全部复用 v1 的成功审计 |
| A-r2 | pass | **needs_refine**（minor：轮廓过脆、画风差异） | pass | pass | 否 | 4 项全部复用 v1 的成功审计 |
| B-r1 | pass | **needs_refine**（major：人物放大导致右脚与鞋被画幅裁切、下前景丢失） | pass | **fail**（同一结构问题） | 否 | 4 项全部新调用 |
| B-r2 | pass | **needs_refine**（minor：线条过重、画风不够通透） | pass | **fail**（画风差异未达终审） | 否 | 4 项全部新调用 |

- **人体门禁按现有生产判据**：明确结构缺陷 / 结论无效 / 归属不确定三者任一都不算通过。
  旧记录没有 `hands_clear` 字段时按同一判据**重算**（这条在修复过程中踩过一次：复用来的旧审计缺字段曾被误判成 fail，
  现已修正并有回归用例 `test_legacy_hand_audit_without_hands_clear_is_recomputed_not_failed`）。
- **门禁后可用 0/4**：卡住的是构图/线条/画风，不是配色。四张的配色与保护色都没有违约，
  按约定**不用画质或终审通过抵消配色违约**，也不因为配色好就放过门禁失败。
- **本轮没有任何门禁修订**：需要修订的全部记「需修订」并停在原地，没有生成任何修订产物。

## 五、调用次数与费用（估算与真实账单分开）

| 项 | 数值 |
|---|---|
| 实际发出的图片请求 | **2**（B-r1、B-r2 各 1 次；均成功返回 1 张） |
| 图片预算上限 | 3（余 1 个只用于「发送被本地预算层拒绝且没发 HTTP」的极端情况，未使用） |
| 实际发出的文本审计请求 | **8**（B 两张 × 4 项；A 的 8 项按候选 hash 复用 v1，**未重复收费**） |
| 配色保留评审 | **2**（A 两张一次、B 两张一次，含对应首图 + 画风参考 + 下肢细节裁剪） |
| 文本预算上限 | 20（门禁 16 + 配色评审 2 + 失败恢复 2；实际用 10） |
| 本地被拒绝的预留 / 预算阻止的审计 / 审计错误 | 0 / 0 / 0 |
| 估算费用 | 图片 2 × 0.036397 ≈ 0.0728 USD；预检时按「2 图 + 10 文本」估 0.1185 USD（`utils/cost_estimate`，分组 Discounted-Banana-1） |
| 真实账单 | **未获取**；以服务商账单为准，本报告不把估算写成账单 |

账本：`send-budget/images.jsonl`、`send-budget/audit-calls.jsonl`、`retention-ledger.json`。
底层重试与备用端点：图片通道 `IMAGE_MAKER_IMAGE_MAX_RETRIES=0` 且预算层在重试未关闭时**拒绝发送**；
文本审计走 `call_text_model` 单次 POST，无隐式重试与端点切换。

## 六、离线修复内容（本轮真正改了的东西）

1. **合同写进真正发送的参数**：`retention_requests()` 现在同时写顶层 `prompt` 与 `call_kwargs.prompt`，
   并断言两者相等（不等就拒绝继续）；另记 `sent_prompt_sha256` 供冻结与核对。
2. **合同正文单一来源**：`RETENTION_CONTRACT_SECTIONS` + `retention_contract_text()`（当前 1949 字符，
   `text_sha256 = 2365d7e6…`），来源文档 `prompts/color-knowledge/repaint-colour-contract-v1-source.md` 说明每一条来自哪里。
3. **mock 捕获后端参数**：测试用假 `dispatch_repaint_request` 记录真实传参，验证 A 无合同、B 含完整合同，
   其余输入与参数逐项一致（见 `tests/test_color_retention_fix.py`）。
4. **冻结核对不可覆盖**：`_retention_freeze_check()` 在续跑前比对 spec/合同/画风条款/参考图/逐条请求 hash，
   任何一项变化就抛错要求新协议或新目录。
5. **按 hash 复用 + 只补审计**：`retention_audit_only()` 与 `_retention_reuse_map()` 只复用
   「候选内容 hash 匹配且审计成功」的结论；`audit-only` 入口不碰 `assemble_repaint_request` / `dispatch_repaint_request`。
6. **预算分开计数**：`retention_ledger()` 把「实际发送成功 / 发送失败或未知 / 本地被拒」分开记，
   文本侧分开记「新调用 / 复用 / 预算阻止 / 审计错误 / 配色评审」。
7. **测试隔离**：新增 `IMAGE_MAKER_COLOUR_TEST_OUTPUT_ROOT`，只在 pytest 显式设置时才允许实验写到临时根，
   生产路径仍受 `data/test-result` 约束（回归用例全部不写交付目录）。
8. 仍然保留的旧原则：实验层**不自做外层预留**（后端内部已预留一次，重复预留会被「同一 run_id 只派发一次」拒绝）。

## 七、交付物

- 纠错说明：[REPAINT-RETENTION-V1-CORRECTIONS-20261004.md](REPAINT-RETENTION-V1-CORRECTIONS-20261004.md)
- 本报告：[REPAINT-RETENTION-V2-RUN-20261004.md](REPAINT-RETENTION-V2-RUN-20261004.md)
- 冻结与预检：`retention-frozen.json`（含 `freeze_hash` 与逐条请求/发送 hash）、`retention-preflight.json`、`retention-a-baseline-check.json`
- 逐张 / 汇总：`retention-per-image.json`、`retention-summary.json`、`retention-gate-outcomes.json`
- 门禁原文：`stages/*-gates/{identity,quality,hands,final-review,gates}.json`（每项都带 `candidate_sha256` 与 `audit_sources`）
- 配色评审：`review/colour-{A,B}.{request.json,raw.txt,json}`（含图片顺序、hash、逐张字段）
- 离线对照页（内嵌 2 首图 + 4 候选）：`retention-review.html`
- 调用账本：`send-budget/{images.jsonl,audit-calls.jsonl}`、`retention-ledger.json`、`review-calls.jsonl`
- 协议：`prompts/color-knowledge/fixed-palette-retention-v2.json`、`prompts/color-knowledge/repaint-colour-contract-v1-source.md`、`prompts/color-knowledge/retention-colour-review-system.md`
- 测试：`tests/test_color_retention_fix.py`（16 条，全部离线、mock 后端）

## 八、没做完 / 不能宣称的部分

1. **没有做任何门禁修订**（按约定本轮只生成 B 初始候选并审计）：B 两张要过终审都需要修订，修订链未跑，也不能说「修订后能保留配色」。
2. **A/B 只有两对**，不宣称普遍可靠，也不做统计显著性声明。
3. **A 不是纯无保护基线**：`tid` 的派生条款本身就含「保留主体配色」的句子，如实记录这条既有张力，不修改 A。
4. **首图是 Flash、重绘是 Pro**：同批公平，但只能称「Pro 重绘保留实验」，不是 Flash 全链实验。
5. **配色评审是模型评审 + 人工目视**，不是人工终审；结论按区域与落点记录，未做像素级区域分割测量。
6. **不发布、不接生产入口**：后续配色功能仍只允许在没有画风参考图时选择；本次双参考是受控实验。
