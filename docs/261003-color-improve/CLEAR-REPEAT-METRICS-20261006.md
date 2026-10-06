# 清色重复验证与只读指标（2026-10-06）

本轮按 `prompts/color-knowledge/DEEPSEEK-STAGE2-CLEAR-WITH-METRICS-20261006.md` 执行：先核对是否已执行，确认未执行后离线建立冻结包，再按 verify → preflight → run → review → evidence → report → status 走完，最后跑一次只读深度指标。没有改生产模板、共享配置或 App 评分公式，没有重启或停止 app。

## 1. 是否已经执行（先核对，避免重复计费）

- `data/test-result/20261006/color-clear-repeat-validation-v3/` 之前**不存在**；`prompts/color-knowledge/` 下也没有任何 clear-repeat/v3 冻结包，全仓库搜索 `*clear-repeat*` 无命中。结论：这份交接的 9＋3 从未执行，也不存在同内容的其他目录，因此不存在可复用的成功槽位。
- 上一轮 v2（`color-chroma-app-v2-validation-v2`）的 6＋2 已用满，与本轮是两笔独立预算，本轮**没有**复用它的图，也没有对它做任何改动。
- 所以本轮按「尚未准备」分支处理：离线建立新冻结包，沿用文档指定的输出目录。

## 2. 执行状态与发送次数

| 项目 | 实际 | 预算 | 说明 |
|---|---:|---:|---|
| 图片请求 | **10** | 9（＋1 用户授权） | A1 首次发送读取超时未出图 → 记为 `unknown_after_send` 并暂停；用户明确授权后重发一次（attempt 2），其余 8 槽位各 1 次 |
| 视觉评审请求 | **3** | 3 | R1/R2/R3 各 1 次，`attempts=1` |
| 授权额外发送 | 1 | — | `counters.authorized_extra_sends=1`，有效图像预算 9＋1=10，报告里单独列出预算偏差 |

- 冻结包 hash：`0c3178c7795a3456c553196cccced84ae449115974208dd52f1ba852fa789298`
- 运行参数 hash：`319049ee016c5521e71fbbafd3b5cebe8bdfadcb742443be825f3cd3ff53587b`
- 生成：`gemini-3-pro-image-preview` @ `new.aigc2d.com`，2K，1:1，超时 120s，`IMAGE_MAKER_IMAGE_MAX_RETRIES=0`；评审：`gpt-5.6-luna` @ 同主机，超时 420s。9 个槽位参数完全一致。
- 最终状态：未暂停；9/9 槽位 `generated`（每槽 1 张 2048×2048），3/3 评审 `reviewed`，`missing_slots` 全空。

**过程事件（不是本任务的成果，如实记录）**：10:26:36 首次发送 A1 后 120 秒读取超时（`transport_error`），运行器按约定暂停并退出码 3，未自动重发。当时只发出 1 次请求，A2–C3 全部仍是 `planned`。10:38:58 用户明确授权后 `authorize-resend`（旧记录进 `previous_attempts`，attempt 计数保留），10:39–10:42 补完 A1 与其余 8 槽位。原文见 `run-notes/incidents.json`。

## 3. 冻结包（离线编译）

- 规格：`prompts/color-knowledge/theme-color-clear-repeat-v3-spec.json`
- 冻结包：`prompts/color-knowledge/theme-color-clear-repeat-v3-pack.json`；场景：`...-v3-scenes.json`；盲评模板：`...-v3-review.md`；候选正文：`theme-color-clear-candidate-v3.md`
- 场景：一个简化桌面静物（原 S2 的对象与绑定：基底 `back wall` 冷蓝／蓝灰，强调 `closed notebook cover, mug` 红而非粉），固定正面视点、固定物件位置、温和单侧光；无人物、无天空、无玻璃、无画风参考、无衣装、无氛围、无重绘。
- 三条件各三次：A＝第一阶段固定色板基准；B＝当前生产清色规则；C＝**与 B 逐字相同**，仅追加候选正文。
- 关键等价性（离线复算，全部通过）：B 与 C 的 `color_plan` 与 `plan_hash` 完全相同（`6691861b985c…`），`selection` 相同；`C.prompt == B.prompt + "\n\n" + 候选正文`，逐字节。A 的 plan hash 为 `5831ee222e14…`（与上一轮 S2 基准同源）。
- 候选正文只登记在 `pack.candidate_texts`，并有 `candidate_text_sources` 指向同名 md 文件；verify 会核对「来源文件 ↔ pack 内文本」「槽位追加 ↔ pack 登记文本」「只有声明条件带追加」三项，任何一处不一致都会报错。**候选措辞没有写回任何生产模板或共享配置。**

### 为支持本轮新增的最小通用协议（`utils/color_direction_pack.py`、`tools/color_knowledge.py`）

旧 runner 只支持「每场景 A–E 各一张、正文=编译器输出」，不支持重复条件、候选追加和预先平衡的展示顺序，也没有除 S1/S2 命名约定以外的解码路径。新增能力全部是通用、数据驱动的，不含任何本轮写死的结果：

1. `compose_slot_prompt()` + `candidate_text_id` / `candidate_append` / `candidate_append_sha256`：正文 = 编译器输出 ＋（可选）唯一候选追加；verify 逐字复算并把追加正文单独 hash。
2. `build-pack`：离线按 spec 编译冻结包与场景文件（任意场景 × 条件 × 重复次数），只调用 App 既有 `freeze_theme_color` / `apply_theme_color`。
3. `review_groups` 的 `neutral_order`：按冻结包声明的顺序固定中性 ID 展示位（未声明时仍随机打乱），并把计划映射写进发送前的 `planned-request.json`；本轮 R1=A/B/C、R2=B/C/A、R3=C/A/B。
4. `comparisons`：解码改为「显式（基准槽位, 候选槽位）」表驱动，不再依赖 `S1-`/`S2-` 命名；旧包没有 `comparisons` 时完全走原路径（回归用例覆盖）。
5. 报告措辞：`verdict_texts` 由包声明；样本数为 0 时不再输出「不支持推广」这类会被读成负面结论的标题。
6. 恢复动作（`authorize-resend` / `settle`）现在接收当前冻结包并校验 ledger 归属，避免在别的轮次目录里改状态。

回归：`tests/test_color_direction_repeat.py`（18 条）＋既有 `tests/test_color_direction_pack.py`、`tests/test_color_direction_fixes.py`、`tests/test_theme_color*.py`，共 64 条通过；改动文件 `py_compile` 通过。离线预检显示 `planned_slots=9`、`planned_reviews=3`、`image_sends=0`、`review_sends=0`。

## 4. 盲评与解码

- 三组都只含本重复的三张图，中性 ID 顺序预先冻结：R1 `N1=A1,N2=B1,N3=C1`；R2 `N1=B2,N2=C2,N3=A2`；R3 `N1=C3,N2=A3,N3=B3`。评审输入不含 A/B/C、不含候选措辞、不含预期方向；`more_distinct` 被明确限定为「哪个图的绑定蓝／红表面灰混杂更少」，并要求不得用亮度或鲜艳度替代。每次比较另外单独记录 `vividness` 与 `brightness`，`confounded` + `confound_kind` 单独记录。
- 每组 3 张 → 每组 3 对，共 9 对；模型全部返回，无缺项、无 ID 不匹配。

### 三个独立统计（冻结解码，自动得出）

| 比较 | 方向观察 | 候选合规成功 | 完整可比对照成功 |
|---|---:|---:|---:|
| B（当前生产清色）相对 A 基准 | **0/3** | 0/3 | 0/3 |
| C（清色追加正文）相对 A 基准 | **2/3** | 1/3 | 1/3 |
| C 相对 B（当前生产清色） | **3/3** | 2/3 | 0/3 |

逐重复的「绑定面更少灰混杂（`more_distinct`）」方向：

| 重复 | B vs A | C vs A | C vs B |
|---|---|---|---|
| 1 | A1 更清楚 | **C1 更清楚** | **C1 更清楚** |
| 2 | A2 更清楚 | A2 更清楚 | **C2 更清楚** |
| 3 | A3 更清楚 | **C3 更清楚** | **C3 更清楚** |

也就是：**当前生产清色 B 在三个重复里都比基准 A 更灰**（0/3，三组同向），而候选 C 对 B 是 3/3 同向、对 A 是 2/3 同向。按交接里事先写好的实用条件「至少两组同向且候选均未新增保护色失败」，C 达到该条件（3/3）；本轮**不是**生产硬门禁，也不能当统计显著性。

### 红色强调保留（逐图，来自盲评原文）

| 槽位 | 笔记本封面 | 马克杯 | 判定 |
|---|---|---|---|
| A1 | 偏浅珊瑚色、接近鲑粉 | 浅红粉、粉感明显 | **粉色问题** |
| A2 | 深红／酒红 | 深红 | 保留 |
| A3 | 中等红、偏暗红 | 深红／酒红 | 保留 |
| B1 | 明亮珊瑚红粉 | 明亮粉红红 | **粉色问题** |
| B2 | 明亮珊瑚红、接近鲑红 | 偏亮红、高光呈粉珊瑚 | **粉色问题** |
| B3 | 明显鲑粉色（判 fail） | 浅红至珊瑚红 | **粉色问题** |
| C1 | 正红／深红 | 正红／暗红阴影 | 保留 |
| C2 | 红色 | 红色 | 保留 |
| C3 | 深红／酒红 | 红色 | 保留 |

- 候选 C **3/3 次**都没有粉化，也没有强调区消失或串色；当前生产清色 B **3/3 次**出现了鲑粉／珊瑚粉倾向；基准 A 出现 1/3 次。这正是上一轮遗留的「红色变粉」问题在本轮被复现并定位到生产清色措辞上。
- 受保护色：9 张图 `protected_hues_status` 全部 pass，`unauthorized_additions` 全部为空（绿叶、棕盆、暖灰桌布、物件数与位置都在）——即「候选均未新增保护色失败」。

### 内容合规与混杂（必须与方向分开看）

- 5 张被判 `content_status=fail`（A1、B1、C1、B2、B3），理由**全部是同一类**：与同组另外两张相比物件／桌面尺度与取景不同，未满足正文里声明的「固定物体尺度」；B3 另有背墙竖向分界（这一条确实是图内问题）。这直接导致 `clear_b_vs_a` 的候选合规与有效样本被清零。
- 这是**评审模板措辞的歧义**，不是新证据：正文的 `same viewpoint, object scale and lighting direction throughout` 本意是「同一张图内保持一致」，但每组事实被声明为「对每张图都相同」，评审就把它当成了跨图要求。冻结结果**不改**、不重解码、不调门禁。
- 更要紧的一点：我对 R1 三张原图的独立查看与评审对 A1 的尺度描述**相反**（A1 的物体在组内最小、离得最远，评审却写它「明显更大、取景更近」）。因此本轮的自动 content 门禁在这类判据上噪声较大，`content_status` 不适合当作颜色结论的证据；颜色轴（`more_distinct` 与粉色问题）在 9 对里自洽得多。
- 混杂分级（评审自报）：R1 三对全是 `outline_only`（不影响绑定面颜色判断）；R2 有两对 `substantive`（B2 物体放大、取景更紧）；R3 有两对 `substantive`（C3 之外的 B3 鲑粉化与背墙分界）。因此 C vs B 的方向 3/3 里有两对带实质混杂，才出现「方向观察 3、完整对照 0」。几何／取景差异是独立随机生成的常见后果，不能用它反推颜色无效。

## 5. 免费的本地只读指标（全部四项可用）

- 命令：`python tools/style_metrics_verify.py --device cpu --precision fp32 --metrics gram,adain,lpips,csd --no-compare-cpu --pair …×9 --out …`，**一次进程**传入 9 对，权重只加载一次；退出码 0。
- 产物：`data/test-result/20261006/color-clear-repeat-validation-v3/local-metrics/pairs-gram-v2-cpu-fp32.json`（36 行 = 9 对 × 4 指标，`status` 全 `ok`）。
- 校验：所有值有限；实际设备 `cpu`、后端 `torch-cpu`、精度 `fp32`（`fallback=false`）；Gram 版本 `gatys-layer-sum/v2`；同图自检 0.0（Gram/AdaIN/LPIPS）与 1.0（CSD，相似度指标）；**9 对 18 张原图的计算前后 SHA-256 与 ledger 记录逐一相同**，没有改动图片。

| 对（候选 vs 基准） | Gram ↓ | AdaIN ↓ | LPIPS ↓ | CSD ↑ |
|---|---:|---:|---:|---:|
| R1 B1 vs A1 | 0.4103 | 37.13 | 0.5608 | 0.7799 |
| R1 C1 vs A1 | 0.8403 | 62.98 | 0.6662 | 0.7415 |
| R1 C1 vs B1 | 0.5616 | 44.65 | 0.4511 | 0.8952 |
| R2 B2 vs A2 | 0.1554 | 25.90 | 0.2321 | 0.9333 |
| R2 C2 vs A2 | 0.1092 | 18.84 | 0.2658 | 0.9392 |
| R2 C2 vs B2 | 0.1440 | 19.71 | 0.2922 | 0.9538 |
| R3 B3 vs A3 | 0.2257 | 27.85 | 0.3392 | 0.9272 |
| R3 C3 vs A3 | 0.4807 | 36.56 | 0.4939 | 0.8193 |
| R3 C3 vs B3 | 0.6257 | 42.59 | 0.4978 | 0.7917 |

**这些数字只描述整图变化，不能归因于局部去灰。** 距离最大的三对（R1 的三对）同时也是取景差异最大的一组，R2 的对（物体放大、画面更接近）距离最小。构图、尺度、笔触差异与颜色差异在整图特征里混在一起，因此：不用 Gram「更像」推翻任何保护色或主体结论，不新增总分、阈值或百分比，也不把深度距离换成配色成功率。

### 更正上一轮的一个误判：lpips / clip 并非缺失

`docs/style-extraction/STYLE-METRICS-GRAM-V2-20261006.md` 记录「当前 Python 的 lpips/clip 缺失，两个指标记 unavailable」。本轮按要求先实测：系统 Python 3.10.11 里 `lpips 0.1.4`、`clip 1.0`、`open_clip_torch 3.3.0` 都能导入（安装时间 2026-10-04 19:28，早于那份报告），LPIPS 校准权重与 AlexNet 骨架、CSD 权重、CLIP ViT-L/14 缓存也都在位。四指标端到端试算（`data/test-result/style-metrics-avail-probe-20261006.json`）全部 `status=ok`，本轮 36 行也全部可算。**未安装任何包、未下载任何权重、未改驱动**；旧的 unavailable 结论不应再被引用。

原因是环境可见性而不是依赖真的缺失：这几个包装在**用户级 site-packages**（`C:\Users\ashsu\AppData\Roaming\Python\Python310\site-packages`），受限沙箱下那次探测读不到该路径，于是把「读不到」误报成「没装」。同一台机器、同一个系统 Python，在沙箱外导入正常——同日另一处工作也已记录同一条环境纠正。因此：以后判断依赖可用性，要在**实际执行指标的那个进程**里做一次真实导入＋试算，不要沿用受限沙箱里的结论。

## 6. 缺项、未验证与边界

- 缺项：0 个漏项（9 张图、3 组评审、9 对比较、21 条绑定观察全部返回；`region_id` 逐字英文，无漏项）。A1 的第一次发送无图但已按 `unknown_after_send` 结算并留在 `previous_attempts`。
- 只有**一个场景、一套色板、一个方向**（clear）的三个重复；没有第二场景、第二色板、人物场景。
- 每条件 3 次独立随机生成，无 seed 保证：只能说明「随机波动 vs 措辞作用」的相对大小，**不能**给统计显著性、生产门禁或稳定普遍效果。
- 未验证：明亮／深沉、浊色、主次面积、辅助层层级、情绪／季节、既有材质与光线、色彩知觉突出后退、画风迁移、重绘保色、GPT-image 通道、参考图、App GUI 内实际生效。
- 本地指标只做整图描述；内容门禁本轮噪声较大（见 §4），不作为颜色结论。
- 全程没有本地蒙版、羽化、叠加、上色或局部补丁贴回；只读特征计算未改动任何图片。没有执行任何 replay 脚本；回放 JSON 的认证头已完全脱敏。
- 现有 GUI 仍是旧模块、不会热更新；本轮结论只出自本次新启动的 CLI 进程。
- 回归套件里 `tests/test_style_metrics.py::test_cli_inventory_runs_and_lists_weights` 在**默认非 UTF-8 控制台**下会失败：它用 `subprocess(..., text=True)` 抓子进程 stdout，而子进程在这台中文 Windows 上按 cp936 输出中文，父进程按 utf-8 解码 → 该测试拿到 `None`。加 `PYTHONUTF8=1` 后 36 条全过。这是既有的、与依赖是否安装无关的环境问题（也不在本轮改动范围内：本轮没有触碰 `tools/style_metrics_verify.py` 与 `utils/style_metrics/`），但值得单独修掉，免得以后被误读成「指标不可用」。

## 7. 证据路径

目录：`data/test-result/20261006/color-clear-repeat-validation-v3/`

- `FROZEN-PACK.json`、`RUNTIME-FROZEN.json`、`offline-verification*.json`、`ledger.json`
- `requests/<槽位>/{planned-request,request,prompt.txt}`、`images/*.color-plan.json`、`images/raw/<槽位>/*replay*.json`
- `images/{A1..C3}-1.jpg`（9 张原图，未做任何本地处理）、`gallery.html`、`RESULTS.json`、`RUN-REPORT.md`
- `reviews/{R1,R2,R3}/{mapping.json,request.json,system-prompt.md,response.raw.txt,review.json}`
- `local-metrics/pairs-gram-v2-cpu-fp32.json`
- `run-notes/incidents.json`、`run-notes/DS-SUMMARY.md`

## 8. 下一步建议

1. 清色**有信号**：候选 C 对当前生产清色 B 是 3/3 同向，且 C 三次都没有把红推成粉。建议下一步在**不改生产模板**的前提下，跨场景／跨色板各补一组小重复（同一 A/B/C 三条件结构，复用一个精简静物 + 一个人物场景），检验迁移性。
2. 生产清色措辞本身该被审视：B 在 3/3 次里让绑定面更灰、并让红色偏向鲑粉／珊瑚粉。若跨场景复现，应把候选的「只在授权蓝红表面减少灰混杂、不得靠提亮或压暗实现、红必须仍是红」并入清色措辞的候选修订，再按流程版本化，不覆盖历史快照。
3. 评审模板要修第 4 节那条歧义：把「固定视点／物件尺度」明确为**图内**要求，跨图的构图差异只进 `confounds.outline_only`，不再触发 `content_status=fail`；否则重复实验的有效样本会被随机构图吃掉。
4. 二阶段主次关系仍未被清色研究覆盖，下一轮应与清色并行推进（用本来就有可比大面积表面的场景，不放大物件）。
5. 深度指标只作辅助：本轮四指标可用，但它无法把局部去灰从构图/画法差异里分离出来，不要给它设阈值或当成验收门禁。
