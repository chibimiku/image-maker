# 固定色板执行验证：执行报告（21 张）

日期：2026-10-04。范围：`data/test-result/20261004/color-fixed-palette-v1/`（独立 test-result 目录，未覆盖任何旧实验）。
本轮目的只有三件事：**指定的颜色组合是否出现、是否落到指定区域、重复执行是否可靠**。不做画质打分、不排美感名次、不与基线比好看。配色符合度与主体内容/保护色**分开统计**。

一句话结论：**21/21 张生成成功，21/21 张完成评审；配色执行 19 符合 + 2 部分符合、0 失败、0 不可验证；区域分配 18 符合 + 3 部分符合；受保护颜色 21/21 保留；内容 20/21（1 张双人失败）。按事先约定，P1-knowledge、P2-simple、P3-knowledge 三组达到 3/3 该版本配色符合且无明确保护色/内容违约，可进入重绘保留验证；B0 是对照组（无配色条款），不作为配色方案晋级。**

## 1. 执行参数与冻结核对

| 项 | 值 |
|---|---|
| 规模 | full：7 组 × 3 轮 = 21 张（P1/P2/P3 各两版 + B0 对照） |
| 模型（请求与服务器回报） | `gemini-3.1-flash-image-preview`（服务器 `modelVersion` 21/21 一致；未切换 pro） |
| 参数 | 1K / 3:2 / 无参考图 / 不重绘 / 无色调校准 / 无本地后处理 |
| 节点 | `new.aigc2d.com`（授权主机），key 来源 `env:IMAGE_MAKER_AIGC2D_API_KEY` |
| 底层 HTTP 重试 | 运行前显式设 `IMAGE_MAKER_IMAGE_MAX_RETRIES=0`（配置原值 1），使收费调用数 = 产物数 |
| 协议冻结 | spec sha256 `763dca46…2c39`；逐组请求 hash 冻结并逐条比对通过（见 `protocol-freeze-audit.txt`） |
| 预检 | 0 问题；两版条款条数一致（各 7 条）、同等点名主色/点缀色/落点区域；保护块 M 只授权「既有腰带与鞋蝴蝶结」改色，无保护色冲突 |

生成后逐张核对：21/21 的 `request_hash` 与冻结值一致、`frozen=True`、产物文件存在且 `PIL.verify()` 通过；`previous_attempts = 0`（没有需要替换的样本）。

## 2. 完整状态清单（21 张计划样本，失败样本不从分母消失）

| # | 样本 | 组 | 轮 | 配色 | 区域 | 保护色 | 内容 | 产物 |
|---|---|---|---|---|---|---|---|---|
| 1 | gemini-flash-B0-r1 | B0 | r1 | conform | conform | True | True | test-gemini-flash-B0-r1_144414-5f4f65.png |
| 2 | gemini-flash-B0-r2 | B0 | r2 | conform | conform | True | True | test-gemini-flash-B0-r2_144605-94911c.png |
| 3 | gemini-flash-B0-r3 | B0 | r3 | conform | conform | True | True | test-gemini-flash-B0-r3_144741-2538ae.png |
| 4 | gemini-flash-P1-knowledge-r1 | P1-knowledge | r1 | conform | conform | True | True | test-gemini-flash-P1-knowledge-r1_144341-6bef63.png |
| 5 | gemini-flash-P1-knowledge-r2 | P1-knowledge | r2 | conform | conform | True | True | test-gemini-flash-P1-knowledge-r2_144624-70b879.png |
| 6 | gemini-flash-P1-knowledge-r3 | P1-knowledge | r3 | conform | conform | True | True | test-gemini-flash-P1-knowledge-r3_144900-53fa5e.png |
| 7 | gemini-flash-P1-simple-r1 | P1-simple | r1 | conform | conform | True | True | test-gemini-flash-P1-simple-r1_144518-103227.png |
| 8 | gemini-flash-P1-simple-r2 | P1-simple | r2 | conform | conform | True | True | test-gemini-flash-P1-simple-r2_144532-5e8049.jpg |
| 9 | gemini-flash-P1-simple-r3 | P1-simple | r3 | **partial** | conform | True | True | test-gemini-flash-P1-simple-r3_144845-088bd0.png |
| 10 | gemini-flash-P2-knowledge-r1 | P2-knowledge | r1 | conform | **partial** | True | True | test-gemini-flash-P2-knowledge-r1_144431-ed7cf4.png |
| 11 | gemini-flash-P2-knowledge-r2 | P2-knowledge | r2 | conform | **partial** | True | True | test-gemini-flash-P2-knowledge-r2_144709-89a300.png |
| 12 | gemini-flash-P2-knowledge-r3 | P2-knowledge | r3 | conform | conform | True | **False** | test-gemini-flash-P2-knowledge-r3_144727-c5eaa5.png |
| 13 | gemini-flash-P2-simple-r1 | P2-simple | r1 | conform | conform | True | True | test-gemini-flash-P2-simple-r1_144357-5b77d6.png |
| 14 | gemini-flash-P2-simple-r2 | P2-simple | r2 | conform | conform | True | True | test-gemini-flash-P2-simple-r2_144635-f1fb33.jpg |
| 15 | gemini-flash-P2-simple-r3 | P2-simple | r3 | conform | conform | True | True | test-gemini-flash-P2-simple-r3_144813-e894e7.png |
| 16 | gemini-flash-P3-knowledge-r1 | P3-knowledge | r1 | conform | conform | True | True | test-gemini-flash-P3-knowledge-r1_144504-e88860.png |
| 17 | gemini-flash-P3-knowledge-r2 | P3-knowledge | r2 | conform | conform | True | True | test-gemini-flash-P3-knowledge-r2_144549-fcb81c.png |
| 18 | gemini-flash-P3-knowledge-r3 | P3-knowledge | r3 | conform | conform | True | True | test-gemini-flash-P3-knowledge-r3_144829-e67bc4.png |
| 19 | gemini-flash-P3-simple-r1 | P3-simple | r1 | conform | conform | True | True | test-gemini-flash-P3-simple-r1_144448-a30ef2.png |
| 20 | gemini-flash-P3-simple-r2 | P3-simple | r2 | conform | conform | True | True | test-gemini-flash-P3-simple-r2_144652-51789f.png |
| 21 | gemini-flash-P3-simple-r3 | P3-simple | r3 | **partial** | **partial** | True | True | test-gemini-flash-P3-simple-r3_144759-7a9c49.png |

汇总：生成 21 次请求 / 21 张成功 / 0 失败 / 0 拒绝 / 0 需替换；评审 7 次调用（7 组各 1 次），**额外 2 次未使用**。

## 3. 逐组结果

| 组 | 版本 | 覆核 | 指标①配色执行 | 指标②可用候选 | 晋级 |
|---|---|---|---|---|---|
| P1-simple | 简洁 | 3/3 | 符合 2 / 部分 1 | 2/3 | 否（r3 部分符合） |
| P1-knowledge | 知识编译 | 3/3 | 符合 3 / 部分 0 | **3/3** | **是** |
| P2-simple | 简洁 | 3/3 | 符合 3 / 部分 0 | **3/3** | **是** |
| P2-knowledge | 知识编译 | 3/3 | 符合 3 / 部分 0 | 2/3 | 否（r3 双人内容失败） |
| P3-simple | 简洁 | 3/3 | 符合 2 / 部分 1 | 2/3 | 否（r3 鞋蝴蝶结未上色） |
| P3-knowledge | 知识编译 | 3/3 | 符合 3 / 部分 0 | **3/3** | **是** |
| B0 | 无条款对照 | 3/3 | 「符合」3（无方案可违背） | 3/3 | **不适用**（无配色条款） |

**总计**：配色执行 19 符合 / 2 部分符合 / 0 失败 / 0 不可验证；区域分配 18 符合 / 3 部分符合；受保护颜色 21/21 保留；内容 20/21 通过（1 张明确双人失败）。

### 3.1 偏差逐条（这是本轮最有信息量的部分）

- **P1-simple-r3（配色部分符合）**：环境整体是冷蓝，但**右侧岸地有较大灰褐色区域**（另两张只出现「少量浅灰褐」或「钓竿棕褐」），被记为未列入方案的大面积色块。两张同组仍判符合，所以是局部不稳定而非系统性失败。
- **P3-simple-r3（配色 + 区域都部分符合）**：腰带成功变成金色，但**两只鞋蝴蝶结仍是深灰，没有执行金色点缀**，因此点缀色只落在两个落点区域中的一个。这是本批次最典型的「区域分配没做完」。
- **P2-knowledge-r1 / r2（配色符合、区域部分符合）**：玫红点缀在腰带与鞋蝴蝶结上成立，但**岸边既有小花仍是粉紫/粉色**，与「玫红不得等同粉」的落点收敛要求只部分一致。这是环境里本就存在的花，属于「现有元素没有跟着色板走」，不是新增道具。
- **P2-knowledge-r3（内容失败，配色仍判符合）**：画面**并列出现两个人物**，左侧持竿钓鱼、右侧动作不完整，违反单人合同 → 该张不计入可用候选（可用候选 2/3）。配色本身仍判符合，**没有拿配色符合抵消内容失败**。
- **受保护色**：28 项保护判定（银灰发 / 象牙白洋装 / 深中性色厚底鞋 × 21 张主体判定，逐组汇总为 21 条 protected 结论）**全部为 True，0 项改变**；没有出现「把点缀色改到头发/裙子/鞋体上」的情况。

### 3.2 两种条款版本的对照（探索性，不作定论）

| 色板 | 简洁版可用候选 | 知识编译版可用候选 | 差异 |
|---|---|---|---|
| P1 冷蓝 + 红 | 2 | 3 | 知识版更多 |
| P2 深绿 + 玫红 | 3 | 2 | 简洁版更多（知识版差在内容，不是配色） |
| P3 蓝紫 + 金 | 2 | 3 | 知识版更多 |

三组方向不一致，**不能得出「知识编译版更可靠」的结论**：P2 的差距来自知识版的 r3 内容失败（双人），而内容失败与条款写法没有已建立的因果关系；每组只有 3 张、单批单模型，不构成显著性。

## 4. 调用次数与费用记录（可核实口径）

| 项 | 次数 | 说明 |
|---|---|---|
| 生成请求 | **21** | 上限 21，全部落在授权内；成功 21，失败 0 |
| 内部自动重试 | **0** | 运行前已把 `IMAGE_MAKER_IMAGE_MAX_RETRIES` 设为 0，请求次数 = 产物数 |
| 评审请求 | **7** | 每组 1 次（每组 3 张图在同一调用里评审）；计划 7 |
| 评审额外额度 | **已用 0 / 可用 2** | 未因「想改评价/想提高成功率」而重复评审 |
| 生成耗时 | 340.1 s 合计 | 单张 11.8 ~ 21.1 s |
| 费用（估算） | **≈0.41 USD** | 生成 21 × 0.0183 ≈ 0.384 + 评审 7 × 0.0035 ≈ 0.025；按本地定价接口（分组 `Discounted-Banana-1`）估算，实际账单以服务商为准。见 `call-ledger.json` |

**一次本地判据缺陷（不影响结论，已如实记录）**：首轮 7 次评审里，`P2-simple`、`P3-simple`、`P3-knowledge` 三组的响应被本地校验器误判为「条款编号不是 1 起」——模型返回的是 `[0,1,…,6]` 连号，而当时的判据写的是「有 0 且没有 1」才移位，连号里也存在 1，于是漏掉了。修正判据后**用离线回收重跑本地校验**（`palette review-recover`，**0 次新调用、0 额外费用**）把三组找回，原始响应、`.failed.json` 与回收记录全部保留；没有靠追加调用掩盖这个缺陷。

## 5. 交付物

- 逐张评价页（离线可浏览、内嵌 21 张实际产图）：`data/test-result/20261004/color-fixed-palette-v1/palette-review.html`
- 结构化：`palette-summary.json`（逐组 + 两个指标 + 逐张行）、`palette-per-image.json`（逐张）、`suite.json`（21 条样本的完整状态与请求 hash）、`schedule.json`
- 协议与冻结：`palette-preflight.json`、`palette-plan.md`、`protocol-freeze-audit.txt`（spec sha256 + 逐组请求 hash）
- 请求与原始响应：`samples/gemini-flash/<样本>/{prompt.txt,request.json,result.json,<产物>.png,_aigc2d_replay_*.json}`；评审侧 `review/<组>.{request.json,raw.txt,json}` 与三份 `*.failed.json`
- 调用与费用：`call-ledger.json`、`palette-review-calls.jsonl`、`palette-review-recovery.json`
- 协议与提示词：`prompts/color-knowledge/fixed-palette-v1.json`、`fixed-palette-v1-simple.md`、`fixed-palette-v1-knowledge.md`、`fixed-palette-review-system.md`

脱敏：请求快照只含 prompt/参数/图片 hash，不含任何 key 或鉴权头（已逐文件检查）。

## 6. 可进入重绘保留验证的组（仅列出，本轮不执行）

按事先约定的晋级条件（**该版本 3/3 配色符合，且没有明确的保护色或内容违约**）：

| 组 | 晋级依据 | 备注 |
|---|---|---|
| **P1-knowledge** | 3/3 配色符合、区域全 conform、保护色全保留、内容全通过 | 用户点名的「冷蓝 + 红」组合，知识编译版 |
| **P2-simple** | 3/3 配色符合、区域全 conform、保护色全保留、内容全通过 | 诊断性色板 |
| **P3-knowledge** | 3/3 配色符合、区域全 conform、保护色全保留、内容全通过 | 诊断性色板 |

不晋级：**P1-simple**（r3 配色部分符合）、**P2-knowledge**（r3 双人内容失败）、**P3-simple**（r3 鞋蝴蝶结未上色）。**B0 不适用**（没有配色条款，不能作为配色方案晋级）。

建议的下一阶段取图顺序（若获确认）：从晋级的组里**按既定的轮次顺序取最早两张可用图**（P1-knowledge-r1/r2），分「现有重绘」与「显式传递配色合同的重绘」两支，共建议 4 张，另开预检与预算。**本轮到此停止，不自动进入该阶段。**

## 7. 本轮没有验证什么

1. **跨主题/跨模型的普遍可靠性**：3/3 只是本批次、本模型、这一套中性色主体的结果。
2. **重绘/身份修订/后处理是否保留配色**：没有任何后续工序产物。
3. **画风兼容**：未挂画风参考图，`Palette`/`Lighting` 条款与配色条款是否冲突未检验。
4. **严格排他色板**：本协议没有任何组属于「只允许某几个颜色」，因此没有验收排他性；「主导色」与「指定区域色」按原文验收（例如 T3 那类「暖色主导」不等于禁止局部冷色）。
5. **画法变化**：评审只记录同组内的画法观察，**同组相似不能证明相对基线或其他组画法未改变**。
6. **美感与画质**：本轮不做任何此类判断，也没有「优于基线」的结论。

## 8. 环境与回归

- 解释器 `C:\Program Files\Python310\python.exe`（3.10.11）；pytest 9.0.3；`py_compile` 通过。
- 本轮为支持执行新增/修正的代码：`utils/color_experiment.py`（固定色板消费层、评审校验、离线回收、评价页）、`tools/color_knowledge.py`（`palette` 子命令：`preflight/preview/generate/review-plan/review/review-recover/summary/status`）、`tests/test_color_fixed_palette.py`（新增 0 起连号、`clauses/items` 别名、离线回收不发调用等用例）。未新增一次性脚本，未修改生产 GUI、画风固件、质量或身份门禁。
- 相关测试：`tests/test_color_fixed_palette.py`、`tests/test_color_reassessment.py`、`tests/test_color_experiment_stage2.py` → **65 passed**。
