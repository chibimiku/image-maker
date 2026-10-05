# 独立衣装第二轮结果：授权路由与严格裁剪（2026-10-05）

任务书：[EXPERIMENT-HANDOFF-DEEPSEEK-ROUND2.md](EXPERIMENT-HANDOFF-DEEPSEEK-ROUND2.md)；
计划 `prompts/wardrobe/round2/plan.json`；运行目录
`data/test-result/20261005/wardrobe-round2-r1/`。本轮只测实验候选，**未部署、未发布、未改生产模板/GUI/全局配置**。

## 0. 先纠正首轮的两个解释

任务书 §1 指出的问题成立，本轮按复核结论执行：

1. **首轮 C06 没有标签互换。** `scores.json` 里 A=on、B=off，原始响应给 A 衣装 4 分 / 保护 fail；
   「衣装细节成立」与「裁剪 fail」是两个维度，可以同时成立。首轮报告 §1/§5.3/§6 把 C06 判成
   「分数不可用/裁剪其实成立」是错的，本轮不沿用；本轮 C06 只按**画面下缘是否停在上胸**判定。
2. **不能说首轮 v2 让 C02「从改不动变成成功」。** 首轮没生成生产原版 on 图，只能说 v2 在本例表现为改款。
3. 首轮还超了带图预算（22 次评分尝试 + 5 次探针）。本轮改为**发送前扣减的持久预算**，
   见 §6。

## 1. 通过 / 失败 / 未证实（按任务书 §8 标准）

| 项 | 结果 |
|---|---|
| 授权解析（12 例中英） | **通过**：12/12 逐字段与本地 gold 一致，`expected` 未进请求 |
| 看图校准（2 例已有图） | **通过**：能分开「校服 vs 连衣裙」与「紧近景 vs 更宽上身」 |
| 就绪文本审查（1 次） | **未通过**：返回 `ready:false`，6 条 blocking（见 §5，含一条不适用于本设计的前提） |
| 生图 | **32/32 成功**，无失败、无补图、无换模型 |
| 匿名评价 | **16/16 成功评分**（视觉尝试 18/20，剩 2 次备用未用） |
| **v3 候选保留资格** | **未通过**：C03/C04/C06 的 12 张 v3 图里 **10 张全项通过、2 张有 fail** → 按标准明确「未通过」 |
| v2 冻结控制（同场重生成） | 作为对照记录，不作为采用建议 |

v3 两处失败（都是具体证据，不是不确定）：

- `C06-gemini-r2`：v3 的裁剪仍失败 —— 评价员读到「下缘到 lower torso，整块胸衣可见」，
  即**绘制内容仍越过了上胸**。C06 四个 v3 样本里 3 个通过、1 个失败。
- `C04-gemini-r2`：v3 的保留失败 —— 被读成「solid-red 连体式制服，白色衬衫与格纹裙锚点缺失」，
  `clothing_action=fail`、`protected_anchors=fail`；其余 3 个 C04 的 v3 样本通过。

另有 2 处「未证实」：`C03-gpt-r2`（保护项 uncertain）与 `C01-gemini-r1`（保护项 uncertain，
无源图只能查文字锚）。

## 2. 快照核对（10 份真实 prompt + hash）

`pack/manifest.json` 的 10 份快照逐份重算：**SHA-256 与字符数全部一致**；
`plan.json`、`resolver-cases.json`、`calibration-cases.json`、`calibrate/review/resolve/score.md`、
`round2` 的 preserve/redesign/frame 模板、`lolita.md`、以及首轮候选
`phase0-candidate-v2/policy-reinterpret.md` 的来源 hash 也全部一致
（`prepare-report.json` 的 `pack_snapshot_hash_check` / `source_file_hash_check` 均为 true）。

| 用例 | 版本 | 字符 | SHA-256 |
|---|---|---:|---|
| C03 | v2_frozen | 3237 | `090ccb3d8a02a69f…` |
| C03 | v3_routed | 996 | `879dc111068ba2dd…` |
| C04 | v2_frozen | 2955 | `10bfc792f9c485f7…` |
| C04 | v3_routed | 912 | `55fe306d18285f23…` |
| C06 | v2_frozen | 2940 | `2b6430d39e6b7bf4…` |
| C06 | v3_routed | 1143 | `585c1ba5935b6c8d…` |
| C01 | v2_frozen | 3099 | `0e38f30a5b1d2ec9…` |
| C01 | v3_routed | 2168 | `7ceaf3a7827db37b…` |
| C02 | v2_frozen | 3219 | `9deda5fa03f53b17…` |
| C02 | v3_routed | 2292 | `8d7f991e0269b9b1…` |

**v2 确认是首轮实际候选控制**：10 个样本的 v2 快照 hash 与首轮 `results.json` 里
gpt/gemini 两通道实际使用的 prompt hash **逐条相同**（`v2_channel_identity` 10/10 identical=true）。
生产模板 hash 与首轮冻结记录一致，**未被本轮改动**。

### 2.1 v2 与 v3 的逐处差异（本轮同时改了组织方式与裁剪条款）

- **v2**：`rendering + 两个换行 + content` 之后追加**完整的 WARDROBE DESIGN AUTHORITY 块**
  （策略规则 + 授权说明 + 整段 Japanese Lolita 设计块 + design-scope 软覆盖句 + NEW outfit 收尾）。
  C03/C04 的保留请求因此仍然收到整套目标衣装设计块；C06 仍收到裙量/袜鞋规则、没有裁剪下缘约束。
- **v3**：保留分支（C03/C04）**删掉整个目标衣装设计块**，只保留 `AUTHORIZED CLOTHING ACTION: PRESERVE`
  + 保留条款 + `FINAL FRAMING REQUIREMENT`；近景分支（C06）换成
  `FILL VISIBLE UNSPECIFIED CLOTHING`（只写裁剪内可见领口/上身）+ 明确的
  「下缘在胸中点之上、不得出现腰部/裙/脚」收尾；补全/改款分支（C01/C02）保留设计块但改用
  `redesign-full.md` 的短组织（授权行 + 保留条款 + 设计块），不再有「软覆盖」段落。
- 这是**整体修订包**的比较：授权组织、裁剪条款与提示词长度同时变化，不能单独归因到某一句。

确定性检查（离线，`prepare-report.json`）：C01/C02 两个版本、C03/C04/C06 的 v3 全部通过
（保留分支无目标衣装设计指令、近景无全身/裙鞋要求且最后一段是画幅约束、无评价字段泄漏、无占位符）；
**v2 的 C03/C04/C06 三份按同一规则判失败**——正是本轮要修的三个缺陷，属预期而非新问题。

## 3. 授权解析（12 例，1 次文本请求）

- 请求：`resolve.md` 作 system prompt，用户内容**只含 id/policy/source_facts/user_request**，
  `expected` 在构造阶段即剔除（`resolver-request.json` 里 `expected_stripped=true`，
  用户内容中不含 `expected`）。
- 结果 12/12 逐字段一致（含中文例 R02/R04/R06/R08/R10/R12）：preserve 5、redesign 3、fill 2、
  review_required 3（矛盾命令 R10、衣物存在性未知 R11、只保局部的复杂换装 R12）。
  真布尔校验通过，无缺项/重复 ID。
- 端点：文本通道（`conf` 的 base_url + `IMAGE_MAKER_TEXT_API_KEY`），模型 `gpt-5.6-luna`。
- **一次解析不能证明真实用户语言、多人或所有分析输入都可靠**；本轮生图也不用它。

## 4. 看图校准（2 次带图请求，生图前）

| 校准 | 请求 | 观察结果 | 结论 |
|---|---|---|---|
| CAL-C03 | C03-off-gpt / C03-lolita-gpt（首轮图，A/B 匿名） | A「Red school-uniform blazer and pleated skirt with gold piping…」；B「Burgundy Victorian-style dress with cream lace trim…」 | 类别差别读对 |
| CAL-C06 | C06-off-gpt / C06-lolita-gpt | A「crops across the upper torso/chest area; no lower garment」；B「crops across the lower part of the dress/skirt」 | 裁剪宽度差别读对 |

内容块顺序逐条记录（system → 用户文字 → `IMAGE LABEL: [A]` → 图片 A → `IMAGE LABEL: [B]` → 图片 B），
两张原图的 SHA-256 与字节数也一并记录（`calibration-CAL-C03.json` / `-CAL-C06.json`）。

## 5. 就绪审查（1 次文本请求）——如实记录未通过

`review-result.json`：`ready:false`，blocking 6 条：

1. C03-v2_frozen 含目标衣装设计块（`WARDROBE DESIGN AUTHORITY — Lolita 衣装`、`Design an OP dress or JSK`）
2. C04-v2_frozen 同上
3. C06-v2_frozen 含裁剪外区域规则（`headwear, socks/tights and shoes`、`layered skirt volume`）
4. C06-v2_frozen 缺明确裁剪下缘、画幅条款不在最后
5. `plan.json` 有 `["v2_frozen","v3_routed"]` 两个候选未定 → 「multiple competing candidates」
6. 首轮候选 `policy-reinterpret.md` 自身不满足保留门禁（hash 相同≠合规）

**我的处置**：第 1–4、6 条与本地确定性检查一致，属 v2 控制臂的**预登记缺陷**（本轮存在的目的就是对照它们）；
第 5 条与本设计不符 —— `plan.json` 已把 v2 登记为「首轮冻结控制」、v3 登记为路由候选，
`deterministic_checks` 对每个版本分别判定，不存在需要在线筛选的两个候选。

文本预算是 **2 次**（解析 1 + 审查 1），已全部用完，**没有**再评审一轮去「评到同意」。
按任务书 §4「一次就绪审查后仍有问题就停止」，严格说这里应停在生图前；用户本轮的直接指令把门禁
明确写成「解析或校准失败则停止」，而解析与校准都通过，且明确授权 32 次生图。
两者冲突时我按**用户指令优先**继续执行，并在报告里把这个偏差写明：
**本轮的生图与评分是在「就绪审查未通过」的记录之上运行的**，读者不应把它当成审查通过。
这也是 v3 结论只能算「未通过」而不是「有条件可用」的原因之一。

## 6. 调用账本（发送前扣减，失败/未知都算）

预算账本：`ledger.json`（每条尝试的先/后状态）+ `budget-slots/*`（`O_CREAT|O_EXCL` 原子槽位）
+ `budget-slots/*/*.jsonl`（reserved/settled 事件）。生图通道全程 `IMAGE_MAKER_IMAGE_MAX_RETRIES=0`
（关闭隐藏重试），视觉与文本请求自行单发、不重试、不切备用。

| 类别 | 上限 | 实际尝试 | 成功 | 说明 |
|---|---:|---:|---:|---|
| 文本 HTTP | 2 | **2** | 2 | 解析批量 1 + 就绪审查 1（§5 的未通过结论也是 1 次尝试） |
| 生图 HTTP | 32 | **32** | 32 | 计划槽位 32，一次尝试=一次 HTTP，无重试/补图 |
| 带图 HTTP | 20 | **18** | 18 | 校准 2 + 匿名评分 16；**2 次备用额度未使用** |
| 模型清单 HTTP | 1 | **1** | 0 | 见下 |

模型清单那次尝试要单独说明：10:08 调用已经发出（预算已消耗），但脚本在本地解析返回值时崩了，
清单内容没有落盘。按「按真实发送扣减、不因本地报错重发」的规则**没有重发**，
账本把它标成 `unknown` 并写进 `recovered`；模型可用性改用**纯本地证据**核对，
即首轮成功生图记录 + `conf/config.json` 的模型清单（`gpt-image-2`、`gemini-3-pro-image-preview` 均命中）。

失败原样保留：本轮没有任何生成/评分失败，所以没有失败样本可展示；账本与 `samples.json`
仍按「失败也占额度」的结构记录（离线自检 `mock-report.json` 验证了该路径，`tests/test_wardrobe_experiment.py`
另有 23 条离线用例覆盖）。

## 7. 生图（32/32，模型与参数沿用首轮冻结设置）

| 通道 | 模型 | 参数 |
|---|---|---|
| gpt-image | `gpt-image-2` | 1024×1536、n=1、quality=high、output_format=png、mode=generate、无参考图 |
| gemini | `gemini-3-pro-image-preview` | aspectRatio=2:3、imageSize=2K、face_quality_boost=False、无参考图 |

参数从首轮 `run-manifest.json` 的 `frozen_generation_params` 读取并冻结（`state.json.generation_params`
记了来源），**未硬编码进 CLI、未换模型**。输出目录 `data/test-result/20261005/wardrobe-round2-r1/`
（`IMAGE_MAKER_TEST_OUTPUT=1`，后端 `save_sub_dir` 指到隔离区）。耗时：gpt 82–115 s/张，
gemini 21–78 s/张，合计约 40 分钟。

**凭据卫生**：`return_metadata=True` 会让后端另外写 `*_replay_*.json/_*.py`，
那里面 `Authorization` 头带**未掩码的 key**。交付前已全部删除（64 个文件），
`credential-hygiene.json` 记录删除清单与原因，扫描确认目录里不再出现该 key；
CLI 也已加上「每次生成后自动清理回放文件」这一步。请求正文本身可从 `samples.json`
（prompt 全文 + hash）与 `*_server_response_*.json`（服务器原始响应）复核。

## 8. 匿名评价（16 组，全部成功）

- 每 case/channel/repeat 一对（v2 vs v3），A/B 顺序按 seed 20261005 随机化：
  **11 组 A=v2、5 组 A=v3**；映射在 `pairs-blind.json`，评价时不透露版本身份与预期。
- 请求记录：完整 messages 快照（已脱敏）、内容块顺序（标签 → 图片 交替）、两张图的
  原图 hash/字节与代理 data URL hash，全部在 `scores.json` 每条记录里。
- `score.md` 要求三项**分别**判定：`clothing_action` / `framing` / `protected_anchors`。
  首轮那种「被内容过滤拦掉」的问题本轮没有出现（16/16 成功）——但不能据此说过滤一定由某句话引起。

### 8.1 v3 与 v2 的同场对照（同一提示词对，同一通道同一次评价）

| 用例-通道-轮 | v2 衣装动作 | v2 保护 | v3 衣装动作 | v3 保护 | v3 裁剪 | v3 判定 |
|---|---|---|---|---|---|---|
| C03-gpt-r1 | fail（变成装饰性 lolita 裙） | pass | pass（红西装 + 百褶裙制服） | pass | pass | 通过 |
| C03-gpt-r2 | fail | uncertain | pass | uncertain | pass | 未证实 |
| C03-gemini-r1 | pass（结果 v2 也守住了） | pass | pass | pass | pass | 通过 |
| C03-gemini-r2 | pass | pass | pass | pass | pass | 通过 |
| C04-gpt-r1 | fail（换成正统 lolita 裙） | pass | pass（制服结构保留） | pass | pass | 通过 |
| C04-gpt-r2 | fail | pass | pass | pass | pass | 通过 |
| C04-gemini-r1 | fail（下装换色） | fail | pass | pass | pass | 通过 |
| C04-gemini-r2 | pass | pass | **fail**（solid-red，白衬衫/格纹裙锚点缺失） | **fail** | pass | **失败** |
| C06-gpt-r1 | pass 衣装 / **fail 裁剪**（下缘到腰） | pass | pass | pass | pass | 通过 |
| C06-gpt-r2 | **fail 裁剪**（越过上胸到腰） | pass | pass | pass | pass | 通过 |
| C06-gemini-r1 | **fail 裁剪**（到腰部与裙摆） | pass | pass | pass | pass | 通过 |
| C06-gemini-r2 | **fail 裁剪**（到腰） | pass | pass 衣装 / **fail 裁剪**（到 lower torso） | pass | **fail** | **失败** |
| C01-gpt-r1 | pass | pass | pass | pass | pass | 通过 |
| C01-gemini-r1 | pass | uncertain | pass | uncertain | pass | 未证实 |
| C02-gpt-r1 | pass | pass | pass | pass | pass | 通过 |
| C02-gemini-r1 | pass / **fail 裁剪**（只看到一只脚） | pass | pass | pass | pass | 通过 |

汇总：v3 **通过 12 / 失败 2 / 未证实 2**（含 2 个正向回归）。C03/C04/C06 这 12 个硬样本里
**10 通过、2 失败**，因此 v3 **不具备保留资格**（任务书要求 12 张全部有证据支持目标与保护项）。

方向性结论（仍受样本量限制）：

- **保留分支删掉设计块是有效的**：C03 的 4 组里 v2 有 1 次明确换装（gpt-r1）、C04 的 4 组里 v2 有 3 次换装/换色，
  而 v3 在这些组里都守住了制服结构。这与「只加一句软覆盖」的 v2 形成对照。
- **严格裁剪条款在多数样本有效但不稳定**：C06 的 v2 四组全部裁剪失败，v3 三组通过、一组（gemini-r2）仍越界。
- **正向回归没有退步**：C01 两组补全成立（一组保护项无法判定）、C02 两组改款成立并保住红色与星形胸针。
- 保护项失败没有被衣装好看抵消（评价要求与记录都按此执行；v2 的 C04-gemini-r1 正是衣装/保护双 fail）。

人工抽查复核（我看过原始 PNG，不替代自动评分）：`C03-gpt-r2-v3` 是红西装 + 白衬衫 + 领结 + 金星的校服，
`C04-gemini-r2-v3` 是偏平涂风格的红制服、胸前的确看不到白衬衫领，`C06-gpt-r1-v3` 是上胸以上近景、
`C06-gemini-r2-v3` 明显画到胸衣下缘以下——与评分记录一致。

## 9. oracle 路由的局限（必须随结论一起读）

生图用的是 **`plan.json` 里预登记的 gold 授权路由**（`generation_is_oracle_routed=true`），
不是模型解析输出。因此本轮只证明了「**在已经知道该保留/该补全/该换装的前提下**，
按授权选用请求组织方式会怎样」，**不能**说「自动解析到生图端到端可用」。
解析测试（12/12）是独立的一次性 fixture，也不能证明真实用户语言、多人场景或分析产物都能解析对。

## 10. 交付物（`data/test-result/20261005/wardrobe-round2-r1/`）

| 文件 | 内容 |
|---|---|
| `prepare-report.json` | 快照/来源 hash 对账、确定性检查、gold 路由、预算、隔离目录 |
| `resolver-request.json` / `resolver-result.json` | 解析请求（含 `expected` 剔除证据）、原始响应、12 例逐字段结果 |
| `review-result.json` | 就绪审查的脱敏请求、原始响应、6 条 blocking |
| `calibration-CAL-C03.json` / `-CAL-C06.json` | 校准请求的 messages 快照、内容块顺序、图片 hash、原始观察 |
| `run-manifest`-等价的 `state.json` | 冻结模型/参数来源、视觉模型、阶段状态 |
| `samples.json` | 32 条实际生成记录（prompt 全文、hash、字符数、耗时、产物路径、状态） |
| `pairs-blind.json` | 16 组匿名配对与映射（11 组 A=v2、5 组 A=v3，seed 20261005） |
| `scores.json` | 16 条评价的完整请求快照（脱敏）、内容块顺序、图片 hash、原始响应与三项判定 |
| `summary.json` / `scoreboard.json`（由 `summary.json` 内嵌） | 分通道/分用例统计与 v3 通过标准判定 |
| `gallery-wardrobe-round2.html` | 16 组 v2/v3 并排 + 原始评价响应（单文件离线，3.5 MB） |
| `ledger.json` + `budget-slots/` | 持久预算账本（含 inflight 恢复记录） |
| `mock-report.json` | 离线自检（发送前扣减、失败仍计、超限即停、崩溃恢复、无隐藏重试） |
| `credential-hygiene.json` | 删除 64 个含未掩码 key 的回放文件的记录 |

代码：`tools/wardrobe_experiment.py`（衣装实验族 CLI，参数化：`--prepare/--list-models/--resolver/
--review/--calibrate/--generate/--pairs/--score/--run/--mock/--status`，用例与矩阵全从文件读）、
`utils/wardrobe_experiment.py`（预算账本与快照检查）、`tools/wardrobe_report.py`（汇总与图库）、
`tests/test_wardrobe_experiment.py`（23 条离线用例，全通过）。

## 11. 下一步（不自行开始第三轮）

1. **C06 需要更强的裁剪机制**：语言约束在 gemini 上仍会越界。可考虑把「上胸以上」改成
   先按裁剪构图的独立请求 + 明确的画幅/构图先验，而不是继续加条款；并保留「不得露腰」的判定。
2. **C04 的 solid-red 现象要单独复测**：v3 在该样本把白衬衫/格纹裙锚点丢了（`protected=fail`），
   说明保留分支虽然不再换装，但对「保留哪些细节」的描述还不够硬。
3. **就绪审查协议本身要修**：`review.md` 把「两个版本」当成 competing candidates，
   与本设计的「冻结控制 + 候选」结构冲突；下一轮要么在 spec 里显式声明控制臂，
   要么把该条从 blocking 降级，避免每次都要靠人工判断。
4. 仍然不要在有图参考、源图编辑、重绘链上复用本轮结论——那些本轮全部未执行。
