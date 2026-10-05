# 独立 Lolita 衣装 pilot 结果（2026-10-05）

> **2026-10-05 事后更正（第二轮前）**：本报告 §1、§5.3、§6 关于 **C06「标签互换 / 分数不可用 /
> 裁剪其实成立」的判断是错的**，已由 [LOLITA-WARDROBE-PILOT-REVIEW-20261005.md](LOLITA-WARDROBE-PILOT-REVIEW-20261005.md)
> 复核原始记录后纠正：`scores.json` 里 C06 两通道都是 A=on、B=off，原始响应给 A 衣装 4 分、
> 保护项 fail；「衣装细节成立」与「裁剪 fail」是两个维度，可以同时成立，不存在证据对调。
> 同时：§7 不建议把「关闭身份门禁」当方案的前提仍成立，但我**不能**用「v2 使 C02 从改不动变成功」
> 这种说法（本轮没有生产原版 on 的实测）；也不能断言内容过滤只由评分提示词文本导致。
> 原评分、原图与原始响应一律保留，未改写任何失败为成功。第二轮结果见
> [LOLITA-WARDROBE-ROUND2-RESULTS-20261005.md](LOLITA-WARDROBE-ROUND2-RESULTS-20261005.md)。

任务书：[EXPERIMENT-HANDOFF-DEEPSEEK.md](EXPERIMENT-HANDOFF-DEEPSEEK.md)。
运行标识 `wardrobe-pilot-v1`，全部产物在 `data/test-result/20261005/wardrobe-pilot-v1/`。
本轮**只做首图筛选试验**：源图编辑、有图（画风）参考、重绘/身份修订连续性、多人、多图衣装提取
一律未执行，不能据此宣称已验证。

## 1. 结论摘要（允许无效与不确定）

| 用例 | 目标 | 结果 | 证据 |
|---|---|---|---|
| C01 未写衣服 | 补全 Lolita | **未获得自动评分**（视觉端点内容过滤 ×2） | 图已生成，待人工评分 |
| C02 旧校服改款 | reinterpret 改成 Lolita + 保红色/金胸针 | **通过（两个通道都通过）** | off 1/0 → on 4；往返位置复核一致 |
| C03 明确保留原校服 | 保留原校服及其剪裁 | **失败** | on 被判 protect=fail、target=1（gpt 通道），on 已变 Lolita 连衣裙 |
| C04 只补全+已有衣服 | 保留红校服 | **失败** | on 被判 protect=fail（gpt 通道 target=1、gemini 下装变白） |
| C05 替换为 Gothic、保黑+银月牙 | 改成 Gothic Lolita、保黑 | **部分**（gpt 通道 1 次成功：off 0 → on 4、protect=pass；gemini 通道待人工） | 位置复核只有单侧成功 |
| C06 半身裁剪 | 只补可见领口/上身 | **自动分不可用**（分数与证据自相矛盾、疑似标签互换）；人工看图：裁剪与可见细节其实成立 | 需人工重评；见 §5.3 与 §6 |

**最重要的一条**：v2 解决了「原服装事实被读成保留要求」的冲突（C02 从"改不动"变成 4 分改款成功），
但**反向冲突没有解决**——C03 的显式保留要求、C04 的 fill_missing 保留语义都被 Lolita 设计块压过。
因此本轮不能宣称「独立衣装控制可安全保留用户明确要求保留的原衣物」。这不是靠关闭身份门禁能掩盖的，
而是模板层的设计块强度问题（见 §7 下一轮最小建议）。

## 2. 阶段0：提示词审查与原服装/保留要求冲突

### 2.1 发现的 blocking（都在**生产原版**模板）

1. `prompts/wardrobe/application.md` 原句
   `Explicit keep-original, garment, colour and signature-accessory requirements take precedence.`
   把**任何**服装要求都抬到策略之上。C02 的 `Her original outfit is a red school uniform…`
   与 C05 的 `Her original outfit is a black trouser suit…` 只是**原服装事实**，
   却被这句读成"必须保留"，和 reinterpret/replace 的改款授权直接冲突。
2. 原版没有任何"设计块只在策略授权范围内生效"的限定：
   C03（显式保留）与 C04（fill_missing 且已写衣服）仍然会收到完整 Lolita 设计块。

审查原文与逐轮结论见 `phase0-review.json`、`phase0-review-rounds12-archive.json`。

### 2.2 采用版本：`candidate_v2`（唯一候选，未并行跑原版）

候选文件只存于实验目录 `phase0-candidate-v2/`，**生产模板一个字节都没改**（hash 逐项核对仍等于
test-pack-v1 快照）。改动共 5 处，均属语义条件化，没有关闭任何身份门禁：

| 文件 | 改动 |
|---|---|
| `application.md` | 把"原衣优先"换成：reinterpret/replace 下未带明确保留指令的原衣描述＝**改款授权**；fill_missing 下显式写出的衣物**保留**；明确 keep-original / 颜色 / 标志配饰仍优先。新增 person count 到 preserve 列表。 |
| `policy-reinterpret.md` | 明确「保留的是明确要求的衣色与点名配饰，未要求的原类别与剪裁不保留」。 |
| `policy-fill_missing.md` | 未改（与快照一致）。 |
| `policy-replace.md` | 未改语义（仅换行口径）。 |
| `design-scope.md`（新增，填入 `{design_scope}`） | 「设计块只在策略授权的范围内适用；策略保留的衣物或显式 keep-original 覆盖设计块；裁剪时只作用于可见衣物，不补不可见的裙型/鞋。」 |

hash 口径 = UTF-8 文本的 SHA-256（与 test-pack-v1 manifest 相同）：

- 生产模板 vs 快照：`production_matches_snapshot = true`（10/10 项一致）
- 候选 spec：reinterpret `c239f180…`、fill_missing `d7fe16de…`、replace `757ee763…`
- 候选文件：application `fadc873d…`、policy-reinterpret `06542c8e…`、policy-fill_missing `5c93ac6c…`、
  policy-replace `e9a44214…`、design-scope `4aaab8c6…`

确定性自检（`phase0-selfcheck.json`，failures=[]）：6 例 on 提示词里策略原文恰好出现 1 次、
design_scope 恰好 1 次、无残留占位符；`design-scope.md` 正文与展开文本 token 级一致。

### 2.3 审查过程要交代的两件事（不掩盖）

- **文本端点的内容过滤**：第一选择 `gpt-5.6-luna` 对审查请求直接返回
  `HTTP 400 content_filter`（同一份请求打到备用端点才成功）。审查因此由备用文本端点
  （`conf` 的 `fallback_*`，本机为 deepseek 通道）完成，全部原始响应与每次尝试的端点都在
  `phase0-review.json`。
- **第一版装载器缺陷**：前两轮评审时我的候选装载器只替换了 `utils.wardrobe` 命名空间里的
  `read_prompt_file`，而 `render_prompt_file` 是 `from utils.prompt_loader import read_prompt_file`
  的早期绑定 → 候选文本从未进入组装，那两轮评审的其实是**原版**。这两轮的原始响应完整保留在
  `phase0-review-rounds12-archive.json`，修正后的 4 轮（round_1..4）才是候选的真实审查结果，
  最终 `ready_for_pilot=true`、无 blocking。原始数据没有删除或改写。

## 3. 模型与凭据冻结

`run-manifest.json` 记录：`python 3.10.11`、`cwd=D:\code\image-maker`、凭据来源
`env:IMAGE_MAKER_AIGC2D_API_KEY`（两个图片节点共用，界面/日志只出现变量名与掩码）。

| 通道 | 模型 ID（冻结） | 参数 |
|---|---|---|
| gpt-image | `gpt-image-2` | `size=1024x1536`、`n=1`、`quality=high`、`output_format=png`、`mode=generate`、无参考图 |
| gemini | `gemini-3-pro-image-preview` | `aspectRatio=2:3`、`imageSize=2K`、`face_quality_boost=False`、无参考图 |

冻结前核验：`GET https://new.aigc2d.com/v1/models` 返回 419 个模型，gpt-image 系列 10 个
（含 `gpt-image-2`），gemini 图片系列 6 个（含 `gemini-3-pro-image-preview`）；两个 ID 都在实时清单里。
`tools/gpt_image2_gen.py --list-models` 与任务书给的 PowerShell 单例 `--dry-run`
（`C01-lolita.txt`，请求体 = `gpt-image-2 / 1024x1536 / n=1`）都跑过，未发在线生图。
**全程没有换模型、没有替失败样本补图、没有扩展到重绘。**

## 4. 阶段1：24 个逻辑生图样本

- 提示词：`off = rendering + 两个换行 + content`；`on = apply_wardrobe(off, build_wardrobe_spec('lolita', case.policy))`，
  全部走 `utils.wardrobe` 真实组装路径，候选 spec 带 hash。
- `off` 与 `on` 的 hash 与 `prompts/wardrobe/test-pack-v1/manifest.json` 逐条对账：
  **off 12/12 与快照一致**；**原版 on 12/12 与快照一致**；本轮采用的候选 on 与快照不同（已在
  `phase0-review.json` 的 `candidate_deviation_from_pack_recorded` 逐条记录，未覆盖快照）。
- 产物隔离：`IMAGE_MAKER_TEST_OUTPUT=1`，并按 `utils/output_isolation.resolve_output_target`
  的真实行为预检；后端 `save_sub_dir` 用相对回退路径 `../test-result/20261005/wardrobe-pilot-v1`
  指到隔离区（`run-manifest.json` 里记了解析结果），**没有文件落进 `data/20261005/`**。
- 结果：**24/24 成功**（1 次调用 1 张图，无重试、无换模型、无补图）。
  实际请求体、服务器返回 JSON 与产物文件名逐条保留（`results.json`、`artifacts-summary.json`
  及同目录 `*-_-server_response_*.json`）。耗时：gpt 21.9–115.9 s，gemini 20.7–51.1 s。
- 尺寸：gpt 全部 1024×1536；gemini 全部 1696×2528（2:3，2K）。

## 5. 阶段2：匿名视觉评价

- 匿名：每对随机 A/B 顺序（seed 20261005），评分提示词按 `prompts/wardrobe/test-score.md` 组装，
  不向模型透露 off/on 身份，也不告知预期胜负。映射在 `scoring-blind.json`，**评分落盘后才在
  `scores-unblinded.json` 揭盲**。
- 视觉端点：先做看图探针（`vision-probe.json`）确认 `gpt-5.6-luna` 真能读到图
  （它对 C01-off 如实回了"奶油色毛衣 + 橄榄绿长裙"，并判 `is_lolita_style=no`，说明看图是真的）。
- 调用统计：**计划 16 次（12 组 + C02/C05 位置复核 4 次）→ 实际 22 次**。
  第一轮 16 次：10 次成功、6 次被端点 `content_filter` 拦截（C01 两通道、C05 两通道 ×2）。
  第二次尝试改用 `gpt-4o` 重跑这 6 对：**1 次成功、5 次仍被过滤**；用最小提示词做的看图探针
  （`vision-probe-filtered.json`）证明 `gpt-4o` 能看这些图，被拦的是**评分提示词文本**。
  超预算的 6 次失败调用如实记录在 `scores.json`（`attempts` 字段保留两轮尝试），未把失败当 0 分。
- 有效评分：**11 次**，覆盖 9/12 配对。评分模型构成：`gpt-5.6-luna` 10 次、`gpt-4o` 1 次。
- 未评分配对：`C01-gpt`、`C01-gemini`、`C05-gemini` → 交付
  `pending-human-scoring.html` / `.json`（匿名顺序 + 揭盲映射 + 人工评分槽），**不编造分数**。

### 5.1 分通道描述统计（0–4 分，仅统计有效评分）

| 用例-通道 | off | on | 偏好 | 保护项(off/on) |
|---|---|---|---|---|
| C02-gpt | 1 | 4 | on | pass / pass |
| C02-gemini | 0, 1 | 4, 4 | on | pass / pass |
| C03-gpt | 4 | 1 | off | pass / **fail** |
| C03-gemini | 2 | 3 | on | uncertain / uncertain |
| C04-gpt | 4 | 1 | off | pass / **fail** |
| C04-gemini | 4 | 3 | off | uncertain / **fail** |
| C05-gpt | 0 | 4 | on | pass / pass |
| C06-gpt | 0 | 4 | on | pass / **fail** |
| C06-gemini | 0 | 4 | on | **fail** / **fail** |

### 5.2 位置复核（C02、C05）

- C02-gpt primary vs reverse：off 1/1、on 4/4，偏好一致（**位置不变**）。
- C02-gemini primary vs reverse：off 0/1、on 4/4，偏好一致。
- C05-gpt：只有 reverse 成功（off 0 / on 4）；primary 被过滤 → 单侧证据。
- C05-gemini：两侧都被过滤 → 无评分。

### 5.3 评价可靠性警告（必须随分数一起读）

- **C06 两次评价的分数与证据自相矛盾**：被标 `on` 的那张，证据写的是"肩部和上胸皮肤、未见可评价的
  领口或衣身结构"，却给了 4 分；被标 `off` 的那张，证据写的是"高领褶边、方胸衣片、连续蕾丝、蝴蝶结"，
  却给了 0 分。也就是说 **C06 的分数疑似标签互换**，只能当"不可用"。
- **C06 的 protect=fail 经人工看图复核为误判**：`C06-lolita-gpt` 确实是胸像（上胸以上、灰底），
  并没有露出脚或全身（详见 §6 C06 一行）。模型给的分与保护结论都不可用。
- 抽查未发现其他配对有明显的标签互换（C01-off 的看图描述与 off 语义一致，C02/C03/C04 的
  on 证据都指向"真的变成了 Lolita 连衣裙"，与分数方向一致）。
- 视觉模型只看得到两张图与文字锚，**没有源图**，因此脸部身份保持、年龄表达只能算"文字锚层面"的检查，
  不能证明身份没变。

## 6. 逐用例说明

> 下面每条都附**人工看图复核**（我看过原始 PNG），与自动评分并列；两者不一致时以人工观察为准并写明。

- **C01（未写衣服 → 补全）**：图已出（`C01-lolita-*`），但两通道都被内容过滤，**无自动评分**。
  人工复核：`C01-lolita-gpt` 是一件结构完整的古典向 Lolita（绿底奶油蕾丝、JSK＋衬衫、头饰、
  及膝伞裙、袜与mary jane 鞋），公园、全身含脚、深棕长发与绿眼都在。轮廓与层次成立，
  但不能据此给"通过"的自动结论——分数留待人工补。
- **C02（旧校服 → 改款）**：本轮**最有把握的正结果**。两个通道、两个位置都判 off 低分 / on 4 分，
  红色与金色星形胸针在 on 上都被确认保留，保护项 pass。人工复核：off 是标准红色校服套装
  （西装外套＋百褶裙＋领结＋星形胸针），on 是同一人物、同姿势同场景的红色 Lolita 连衣裙
  （钟形裙、多层蕾丝、头饰、袜鞋协调，星形胸针仍在领口正中）。说明 v2 的"原衣事实≠保留要求"确实生效。
- **C03（明确保留）**：**失败**，且经人工确认。gpt 通道 on 变成"大面积蕾丝 + 荷叶边 + 束腰裙 + 头饰的
  Lolita 连衣裙"，保护项明确 fail；gemini 通道略好（仍被读出不是原剪裁，protect=uncertain）。
  **显式写着的 `do not redesign or replace it` 没有压住设计块**，这是本轮最重要的负结果。
- **C04（只补全 + 已有衣服）**：**失败**，人工确认。gpt 通道 on 被改成酒红色 Lolita 连衣裙；
  gemini 通道把下装改成白色百褶裙（红色主色丢失）。fill_missing 的"已有衣服就保留"没有生效。
- **C05（替换为 Gothic、保黑/银月牙）**：gpt 通道单侧评分 off 0 → on 4、protect=pass。
  人工复核 `C05-lolita-gpt`：黑色 Gothic Lolita（高领蕾丝、束腰、伞裙、黑色袜鞋），
  黑色保持、领口**银色月牙胸针可见**，石庭场景保留，**没有出现粉色或茶会强制**（模板显式禁止项）。
  限制：同一提示词里的"黑色裤装"在视觉上无从对照，且 gemini 通道无评分。
- **C06（半身裁剪）**：分数不可用（见 5.3），保护项 fail 是误判。人工复核 `C06-lolita-gpt`：
  画面确实是**上胸以上的胸像 + 纯灰底**，只看到高领褶边、胸衣片、蕾丝与蝴蝶结；没有露出脚，
  也没有补出不可见的裙型/鞋。也就是说"只补可见细节"与"保持裁剪"这两个目标**在画面上是成立的**，
  只是自动评分端给不出可用结论；完整衣装符合度仍不可验证（`full_outfit_verifiable` 必须为 false）。

## 7. 下一轮最小建议

1. **把 keep-original / fill_missing 的保留做成硬锁**：命中保留条件时，不再把 Lolita 设计块原文
   发给模型（本轮 `design-scope` 的"软覆盖"句不够），改为只发"保留原衣物 + 允许的微调"条款；
   这是 C03/C04 失败的直接原因，也是本轮最值得先修的一件事。
2. **裁剪用例单独处理**：C06 需要在模板层强制"只画裁剪内可见衣物"，并让评分端对这种图使用
   更短、更中性的提示词；同时给评分端加"标签-证据一致性"自检（本轮 C06 就是靠人工比对才发现的）。
3. **评分端准备无过滤的备用视觉端点**：本轮 5/16 计划评分被 `content_filter` 拦住，且用的是
   评分提示词文本而非图片。下一轮要么把量表拆成更中性的短问句，要么固定一个不被过滤的视觉端点。
4. **仍然不要在本轮结论上加码**：源图编辑、有图画风兼容、重绘连续性、多人、自动衣装提取全部未验证，
   下一轮应先做"显式保留 + 源图"这一条最小闭环，再谈组合工序。

## 8. 产物清单（`data/test-result/20261005/wardrobe-pilot-v1/`）

| 文件 | 内容 |
|---|---|
| `preflight.json` | 凭据来源、隔离目录解析、模板 hash 对账、实时模型清单 |
| `phase0-review.json` | 候选 diff、离线检查、4 轮文本审查原始响应（含端点与失败原因） |
| `phase0-review-rounds12-archive.json` | 装载器修正前的两轮原始响应（评审的是原版） |
| `phase0-selfcheck.json` | 确定性字符串检查（failures=[]）与冻结的候选 spec 正文 |
| `phase0-candidate-v2/` | 5 个候选模板原文（生产模板未改） |
| `run-manifest.json` | 模型 ID、冻结参数、模板 hash、组装口径、凭据来源（不含密钥） |
| `results.json` | 24 条实际请求（prompt 全文、hash、耗时、产物路径、状态） |
| `artifacts-summary.json` | 24 张产物的文件名/字节/像素尺寸 |
| `scoring-blind.json` | 匿名 A/B 顺序与揭盲映射 |
| `scores.json` | 22 次视觉调用的原始响应、失败原因、尝试链 |
| `scores-unblinded.json` | 揭盲后的分数、保护项、偏好与统计 |
| `vision-probe.json` / `vision-probe-filtered.json` | 看图能力探针（主/备选视觉模型） |
| `gallery-wardrobe-pilot-v1.html` | 12 组 off/on 并排 + 评分原文（单文件离线） |
| `pending-human-scoring.html` / `.json` | 未评分配对的人工评分清单与匿名图 |

## 9. 本轮明确没有做的事

- 没有发布任何图片、没有改全局配置、没有覆盖生产模板或工作区里他人的修改（只新增实验目录与本报告）。
- 没有执行源图编辑、有图画风参考、重绘/身份修订连续性、多人、多图衣装自动提取。
- 没有把文本审查当成生图或视觉验证；DeepSeek 端只做文本审查，看图的是 `gpt-5.6-luna` / `gpt-4o`。
- 没有为失败或未评分的样本补图、重抽或换模型；失败原样保留。
- 本轮首图符合**不能**证明上述任何流程安全或有效。
