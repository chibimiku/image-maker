# cccccanh 人工优化候选（opt1）实际生图与对照验证（2026-10-05）

验证对象：`data/20261005/style-extraction/cccccanh/run-20261005-195523-094029/manual-opt1/cccccanh-round5-opt1_style_iter_result.json`
对照对象：`data/20261005/style-extraction/cccccanh/run-20261005-195523-094029/cccccanh_style_iter_result.json` 的 **round_5** 包
本次结论依据：[CCCCCANH-REVIEW-20261005.md](CCCCCANH-REVIEW-20261005.md) 提出的待验证特征，逐条用**本轮实际产物**核对。

> **状态区分是本报告的第一要求**：
> - **已出图**：18 张全部落盘（A 组 6 张 + B 组 6 张 + C 组 6 张），产物 SHA-256 与请求快照齐全。
> - **已通过复核**：**没有**。本轮只完成人工视觉复核与本地渲染数值；**视觉量表评分与深度指标（CSD/Gram/AdaIN/LPIPS）尚未运行**（见 §6）。
> - 本报告不构成立项、入库或发布结论。

## 0. 本轮决策与边界（执行前与用户确认，写进报告）

| 项 | 决定 | 影响 |
|---|---|---|
| 验证范围 | **先只出图**，比较链路（视觉量表 + 深度指标）后置到同一状态文件上继续 | 本轮不产出 12 项量表分数与 CSD/Gram/AdaIN/LPIPS 数值；结论只基于人工看图与本地渲染数值 |
| A 组测试主体 | **沿用原运行主体原文** `a girl sitting beside lake, she has pink hair and brown eyes, spring,` | A 组与原第 5 轮同条件，可作历史对照 |
| 原候选来源 | 取 `test_images.round_5.prompt_variants`（**不是**原运行的最终 `prompt_variants`） | 与原运行的最终包区分开，避免把终审包当成第 5 轮包 |
| 重绘模式 | `reference_mode=none`、`clauses_without_image=true`、`scope=full`，只发送各自 GPT 首图 | 与原运行第 5 轮的重绘模式一致，不附完整画风图 |
| 可选母题 | **两边统一关闭** | 原运行当年 `motif_enabled=true`（花瓣/光点可选条款进了 GPT 首图）；若只让一边带母题，对照不成立。因此本次原包首图是**同条件重生成，不是逐字复现** |
| 每场景次数 | 每包每通道 **1 次** | 满足 14 次预算（实到 18 张成功）；单张波动不能当整包结论 |
| 未做 | 不重新筛图、不换参考图、不改候选 JSON、不写 `conf/config-styles.json`、不发布图片 | 原运行、原图、原报告全部保留未覆盖 |

任务书列出的 5 条待验证特征原文对照位置见 §4。

## 1. 输入核验（不改写候选字段）

| 候选包 | 文件 SHA-256 | 形态 | 取包位置 | 校验 |
|---|---|---|---|---|
| `cccccanh-r5-original`（原第 5 轮） | `2e451edd4ab5bd097ce3a53d80c2a124ad4a0e64b27f73055964eeb0051db80c` | v2 完整训练结果 | `test_images.round_5.prompt_variants` | `gpt_image_prompt_valid=true`，0 错误 |
| `cccccanh-r5-opt1`（人工优化） | `400db699a1dc1f8b5fd0ce8b0d740f322d1a2ddedfaa58cf14b1eb120f57110b` | light 精简候选 | 根级 `prompt_variants` | `gpt_image_prompt_valid=true`，0 错误 |

字段级指纹（进 `precheck.json` / `request-audit`）：

| 字段 | 原第 5 轮 | opt1 |
|---|---|---|
| `gemini_full_prompt` | 5250 字 `5ad98fd7c57068cf` | 2885 字 `cd5adf8e9860bc68` |
| `gpt_image_prompt` | 592 字 `91c2ca32eb27d454` | 568 字 `62fed93538faf65e` |
| `gemini_repaint_clauses` | 6 条 1118 字 `07b94ccfac40ffa5` | 9 条 2143 字 `97433b13cc208f9f` |

两边都走 `normalize_style_prompt_package` 重新校验，`copy_checks` 四项（三个字段逐字未变 + 母题为空）全部为真，否则直接停止。字段**未拼接、未压缩、未改写**。

源图：候选 JSON 自己记录的 **15 张** `dataset.images` 全部存在，逐张哈希写入预检；张数不再写死 12。
测试画风参考图：`data\style-datasets\cccccanh-ep-730ebc9f91af\images\1024x1024\prejpg-2019-04-26 14_28_08_1_fixed_2025-04-05_23-25-45.png`（1024×1024 → 比例吸附 `1:1`）。

## 2. 实际参数（不输出凭据）

| 通道 | 模型 | 关键参数 |
|---|---|---|
| Gemini 直出 | `gemini-3-pro-image-preview` | `aspectRatio=1:1`、`imageSize=2K`、`face_quality_boost=False`、只附测试画风参考图 1 张 |
| GPT 首图 | `gpt-image-2` | `mode=generate`、`size=1024x1024`、`n=1`、quality 未显式传（取节点配置）、只附画风参考图、不做 edits |
| GPT→Gemini 重绘 | `gemini-3-pro-image-preview` | `2K`、`repeat=1`、固件 `repaint-system-conservative-v5.md`、`reference_mode=none`、`clauses_without_image=true`、`scope=full`、只发送本次 GPT 首图 |

密钥只以来源形式记录（`env:IMAGE_MAKER_AIGC2D_API_KEY`），**文件与报告不含密钥值**；服务端日志里的 key 由后端自身掩码（`sk-btNsN...aAp4`）。

逐主题请求摘要（`request_digest`，由「实际传输内容」折算：提示词哈希 + 输入图哈希 + 模型/尺寸/质量/模式/参考模式/条款）：

| 场景 | 原第 5 轮 | opt1 |
|---|---|---|
| A-lake-pink | `d733614d3974d315` | `07997b222e1612bf` |
| B-fullbody-soft | `f9ed2f65567971e9` | `ee018df972eae10e` |
| C-dark-contrast | `f9709a1a08e67a85` | `e2df0b5990b9b26b` |

每包每通道的生效提示词全文、字符数、SHA-256、输入图顺序与哈希都在各 `precheck.json` 与 `generated/<包>/attempt-1/attempt.json` 里。**预检与实际发送共用同一段组装代码**（`StyleIterativeWorkerThread._assemble_test_generation_requests`；重绘走 `utils.post_process.assemble_repaint_request`），不是另写一份拼装。

## 3. 失败与修复（如实记录，不掩盖）

本轮**不是一次跑通**。前两次执行被我自己引入的缺陷打断，产物已弃用重跑：

1. **`api_backend.resolve_save_dir` 绝对路径缺陷（我引入的）**：`save_sub_dir` 传绝对路径时会被拼成 `data/<日期>/D:\code\...` 的畸形路径。我为受控实验加 `resolve_save_dir` 收口时，误删了 `generate_image_aigc2d` 里 `save_dir` 的赋值，导致 Gemini 直出在**服务端已成功返回图片之后**抛 `NameError: name 'save_dir' is not defined`。
   - 代价：22:06 那轮 2 次 Gemini 直出调用被浪费（图已生成但落盘失败）；已修复并复跑。
   - 影响范围：只影响传绝对路径的新代码路径；GUI 常规用法传相对子目录，行为与改动前一致。
2. **失败的 partial 产物被当成重绘成功（既有缺陷，已修）**：`run_pipeline` 工序失败时会写一个带 `-partial` 的 final 回退件，那是**源图的重编码副本、不是重绘结果**。复核链路原先只看文件名里的 `-final-rp`，于是把 `gpt_repainted` 记为成功，**产物哈希与 `gpt_first_pass` 完全相同**。
   - 若不拦住，视觉比较会把「根本没重绘过」的图当成重绘候选评分 —— 这会直接伪造对照结论。
   - 现在：`_final_repaint` 排除 `partial`，`_run_missing_channels` 额外读 manifest 判 `status=failed`，复用路径也过滤 `partial`。22:11 的产物因不可信已删除重跑。
3. **上游偶发拒绝**：`Unknown parameter: 'image'.`（A 组原包 1 次、B 组原包 1 次），后端自带 1 次重发后成功。已记录在 `attempt.json` 的 `errors`/服务端响应文件里。
4. **未做无上限重试**：`IMAGE_MAKER_IMAGE_MAX_RETRIES=0`；重绘失败一次即按失败记录（22:08 那轮 opt1 重绘 `RuntimeError: 重绘没有返回图片` 就保留为 failed，复跑后成功）。

最终有效产物：**18 张，全部 3 通道 success、零复用**（每张都在本轮由对应包真实生成，`复用 0 条通道`）。

## 4. 逐项验证任务书 5 条特征（基于实际产物）

**依据图像**：`_review/source-sheet.jpg`（15 张源图）、三组 `_review/contact-sheet.jpg`（每组 6 张候选）、以及 `_review/*-face.jpg` / `*-hair.jpg` / `*-upper.jpg` 放大裁剪。

### 4.1 平滑简化的脸 vs 有块面的头发/材质 —— **opt1 变强**

- opt1 重绘放大图（A 组）：额头与脸颊是**低频平面**，几乎无结构线；头发则是**清晰分组的宽发片**，片与片之间有暗色分隔槽，头顶高光是一条**窄的、成束的**断续亮带而不是整片玻璃带。头发与脸的处理强度形成可见反差。
- 原第 5 轮重绘（A 组同位置）：脸同样平滑，但头发是**更多细分股 + 连续高光带**，更接近「丝滑分股」，脸与发的反差没有拉开。
- 数值侧：A 组 opt1 重绘碎线比 0.013、段均长 74.0；原包重绘 0.007 / 110.7。原包更「长线连通」，opt1 更「分片」，与块面化方向一致。

### 4.2 大发束成束细笔触、透明叠色、不连续片状高光 —— **opt1 变强**

- opt1 的 repaint 条款第 5 条明确要求把头发反光「break into selective slivers, angular chips and uneven dashes with matte gaps」，实际产物确实出现了**哑光间隔 + 成束细笔触**并存的表面。
- 原包对应条款只说「tapered highlight bands」，产物是较连续的亮带，哑光间隔少。
- B 组 opt1 重绘的裤装出现**宽折面 + 楔形暗部**，原包同位置偏均匀渐变。

### 4.3 三档边缘层级 + 人体结构连通 —— **部分变强，但有明确结构缺陷**

- 三档层级：opt1 外轮廓/重叠边有局部深色强调（重绘放大图里发片边缘可见偏冷的深色线），内部线更细更断续，非焦点区边缘发虚 —— 三档能看出来。原包更接近**两档**（轮廓 + 均匀细线）。
- 数值侧：A 组 opt1 直出粗边占比 0.126、长线占比 0.841（源图 0.111 / 0.773），是 18 张里最接近「强轮廓 + 长线连通」的一组。
- **结构缺陷（内容侧扣分，不因画风分抵消）**：
  - **A 组 opt1 直出的手部不可读** —— 左手与膝盖/衣摆糊在一起，看不出指分。
  - **C 组 opt1 的直出**：头部比例与姿势与指定主体不符（详见 §5 C 组）。
  - B 组 opt1 首图与重绘的手（托腮那只）可读，手指分组清楚。

### 4.4 保留中间调和深色锚点，避免全图浅色柔雾 —— **opt1 明显变强（暗色场景出现过度修正）**

| 场景/通道 | 亮度 | 高光占比>240 | 中间调占比 | 对照 |
|---|---|---|---|---|
| A 原包直出 | **190.3** | **0.108** | 0.419 | 全图浅色柔雾最典型的一张 |
| A opt1 直出 | 170.2 | 0.032 | 0.630 | 高光裁切降 7.6 个百分点，中间调 +0.21 |
| B 原包直出 | 167.6 | 0.092 | 0.494 | 仍然偏亮 |
| B opt1 重绘 | 149.0 | 0.029 | 0.614 | 更接近源图亮色子集（142.5） |

源图基线：亮色子集（6 张）亮度 142.5、高光 0.069、中间调 0.487；**两个包都仍比源图偏亮**，opt1 只是接近得多。
**过修正**：C 组 opt1 首图/重绘亮度 32.9 / 37.0，暗部占比 0.87 / 0.85，中间调只剩 0.06 —— 已低于源图暗色子集（亮度 80.0、暗部 0.429、中间调 0.338）。见 §5 C 组。

### 4.5 不强制服装/刘海/Q 版比例/肖像裁切/新增装饰 —— **条款变强，但产物仍出装饰**

- 条款侧 opt1 明确写了「do not crop to match a style portrait」「do not impose the reference's fringe」「add no collars, lace, bows, head ornaments, flowers, particles or new background objects」，原包没有这几条。
- 产物侧**部分生效**：
  - A 组 opt1 直出是**全身坐姿 + 完整石台**，没有裁成肖像；B 组三路都是完整全身 + 双脚 + 石凳接触关系（这符合 B 组主体的明确要求）。
  - **但 A/B 组两组在 GPT 首图与重绘上都出现了白色蝴蝶结/发带**（见 contact sheet 右上与中下）。原包与 opt1 都有，说明这是首图阶段带进去的装饰，未随画风包区分 —— 属于**内容侧新增装饰**，记为缺陷，不算画风契合加分。
  - 服装：两边都没有强制成参考图的紫发金领，但都自由发挥成浅色连衣裙/衬衫，属测试主体未指定服装的自由度，不作为违反。

## 5. 逐场景结论

### A. 同主体三路对照（`a girl sitting beside lake...spring`）

- **opt1 路线**：直出为全身坐姿、结构清楚、颜色偏暖灰、高光不炸；首图偏「清爽动漫」、发丝偏滑；重绘把头发改成成束片状、脸部更平，是三条里最接近源图画法的。
- **原包路线**：直出是三者里最「弥漫淡彩」的一张（亮度 190.3、高光裁切 0.108），线条长、细节少；首图与重绘的整体偏亮偏雾，重绘的头发出现连续玻璃高光带。
- **内容问题**：两组的眼睛颜色都没守住原主体 —— 原包重绘把棕眼画成了**琥珀金眼**，opt1 重绘保住了棕色。两者都在头发上加了白蝴蝶结。
- **净判断**：本场景 **opt1 更好**（暗部与中间调更接近源图、头发块面化有效、眼睛颜色守恒），但**不足以判为通过**：蝴蝶结装饰为两组共有的内容偏离，手部在 opt1 直出上不可读。

### B. 泛化：柔亮全身

- **两组都守住了客观约束**：完整身体、双脚、石凳与坐姿接触关系可读，无发饰/蕾丝/蝴蝶结（**注意：A 组的蝴蝶结在这里没有出现**），衣装为米色衬衫 + 海军蓝长裤 + 棕色平底鞋。
- **opt1**：重绘亮度 149.0（最接近源图 142.5），中间调 0.614，碎线 0.019；裤装折面与楔形暗部可见，手（托腮）可读。
- **原包**：直出亮度 167.6、高光 0.092，明显偏亮偏白；首图/重绘亮度 155.7 / 151.8，比 opt1 更亮更均匀。
- **净判断**：**opt1 更好**，且是本轮最干净的一组（没有明显结构缺陷）。

### C. 泛化：暗色强对比

- **原包**：直出亮度 62.3、暗部 0.570、饱和 82.5；首图/重绘 85.5 / 86.7、暗部 0.32 / 0.30。深色留住了，但首图/重绘亮度高于源图暗色子集（80.0）的边缘，**没有漂白**。粗边占比只有 0.052 / 0.065，轮廓层级偏弱。
- **opt1**：
  - 首图 / 重绘：亮度 32.9 / 37.0，暗部 0.87 / 0.85，中间调 0.06 —— **没有被漂白，而是压得过头**，接近一张以近黑为底的图，路面反光带成为唯一亮部；脸部平面在 2K 放大下仍可读，衣料折面可见。
  - **直出：明确失败**。产物是**紫发少女近景肖像**，白色/浅色背景、蓝金领口 —— 这既是**参考角色特征泄露**（源集第 0 张的紫发 + 蓝金领），也是**年龄与构图改变**（指定「成年短银发琥珀眼女性」「完整身体与脚部接地可见」），并且**画面被漂白**（亮度 142.4、中间调 0.483、暗部仅 0.082，与该场景要求完全相反）。
- **净判断**：C 组 **不能用高画风分抵消**。opt1 的 GPT 两路比原包更接近深色强对比方向但已过度压暗；opt1 的直出存在角色泄露 + 年龄/构图偏离 + 漂白，属于**内容门禁级失败**（若跑量表，`reference_content_copied` 与 `explicit_subject_mismatch` 应置真）。原包在本场景反而是**内容更稳的一组**。

## 6. 比较能力的实际状态（本轮未运行的部分）

| 能力 | 状态 | 说明 |
|---|---|---|
| 人工视觉复核 | ✅ 已完成 | 联系表 + 脸/发/上半身放大裁剪，对照 15 张源图 |
| 本地渲染数值 | ✅ 已完成 | 复用 `tests/style_render_metrics.py` 口径（明度层次 / 多尺度边缘 / 线稿连通 / 负空间），见 §4.4 与 `_review/metrics-table.md` |
| 12 项视觉量表 v2 + 四门禁 | ⛔ **未运行** | 本轮范围外（用户选择先只出图）。三个状态文件已可直接喂给现有比较链路 |
| CSD / Gram / AdaIN / LPIPS | ⛔ **未运行** | 同上；NPU 优先入口为 `python tools/style_metrics_verify.py --comparison-state <状态文件> --device auto-npu` |

**因此本报告不给出任何「相似度百分比」**，也不合成跨指标总分。上游复核记录的深度指标（第 5 轮重绘 CSD 0.5195 / LPIPS 0.6009，最终重绘 CSD 0.5373 / LPIPS 0.6152）是**旧运行**数据，两项偏好不一致，**不能用来支持本轮结论**。

## 7. 建议：**只采用部分改动**

不建议整包替换或整包保留，逐条列：

**建议采纳（有本轮实际产物支撑）**
1. 「保持中间调与深色锚点、不做全图浅色淡彩」这条 —— A/B 组高光裁切降 7.6 / 6.3 个百分点、中间调升 0.21 / 0.12，直接对治原包最刺眼的问题。
2. 头发「宽片 + 成束细笔触 + 哑光间隔 + 不连续片状高光」的条款 —— 产物层面确实把连续玻璃高光带改掉了。
3. 三档边缘分层与「不得为匹配画风肖像而裁切」的条款 —— A/B 组保住了全身与接触关系。
4. 保留 `gpt_image_prompt` 的 8 字段结构（568 字，本地校验通过），比原包的象牙雾取向更中性。

**建议保留原版（opt1 有明显退步或未验证）**
5. **C 类暗色强对比场景的 Gemini 直出入口**：opt1 出现参考角色泄露 + 年龄/构图偏离 + 漂白，不能上线；同时 opt1 的整体色调阈值在深色场景**过度压暗**（亮度 32.9 已低于源图暗色子集 80.0），需要单独的暗色档位而不是同一套阈值。原包在该场景内容更稳。
6. 未验证项一律不动：本轮的 Gemini 直出只跑了 3 个主体各 1 次，**不足以**推广到「所有主体」。

**必须先修再谈采纳**
7. **裁掉组内无法守住的装饰与结构**：A 组两组 GPT 路都加了白蝴蝶结，opt1 的直出还出现手部不可读 —— 这两条是**内容侧**缺陷，不能靠画风分抵消。
8. 建议下一步在同一批状态文件上补跑量表与深度指标，再用 C 组直出这类**已确认的内容失败**做门禁负样本校准；本轮不修改任何候选 JSON 或画风配置。

## 8. 产物路径（绝对路径）

**验证根目录**：`D:\code\image-maker\data\20261005\style-extraction\cccccanh\verify-20261005-2205`

| 内容 | 路径 |
|---|---|
| A 组状态文件 | `...\verify-20261005-2205\A-lake-pink\ccccanh-review_style_iter_result.json` |
| B 组状态文件 | `...\verify-20261005-2205\B-fullbody-soft\ccccanh-review_style_iter_result.json` |
| C 组状态文件 | `...\verify-20261005-2205\C-dark-contrast\ccccanh-review_style_iter_result.json` |
| 逐主题预检（未联网） | `...\<场景>\precheck.json` |
| 逐包逐通道请求与产物记录 | `...\<场景>\generated\<包>\attempt-1\attempt.json` |
| 重绘工序清单 | `...\<场景>\generated\<包>\attempt-1\gpt-repaint-steps\pipeline-manifest.json` |
| 联系表（3 组） | `...\<场景>\_review\contact-sheet.jpg` |
| 源图联系表（15 张） | `...\verify-20261005-2205\_review\source-sheet.jpg` |
| 脸/发/上半身放大裁剪 | `...\<场景>\_review\<包>-attempt1-<通道>-{face,hair,upper}.jpg` |
| 渲染数值原始数据 | `...\verify-20261005-2205\_review\metrics-raw.json` |
| 数值对照表 | `...\verify-20261005-2205\_review\metrics-table.md` |

原运行与优化候选**均未覆盖**：
- `data\20261005\style-extraction\cccccanh\run-20261005-195523-094029\cccccanh_style_iter_result.json`（SHA-256 `2E451EDD...`）
- `...\manual-opt1\cccccanh-round5-opt1_style_iter_result.json`（SHA-256 `400DB699...`）

## 9. 代码改动（不新建一次性脚本，不改通用生图固件）

| 文件 | 改动 | 是否影响既有行为 |
|---|---|---|
| `modules/others/api_backend.py` | 新增 `resolve_save_dir`，12 处 `save_sub_dir` 目录拼接改走它；修复误删的 `save_dir` 赋值 | 相对子目录行为不变；绝对路径不再被污染 |
| `modules/image_analysis/style_merge_validation.py` | 新增「按运行记录候选 JSON 取包」模式（`ReviewCandidateTask` / `build_review_package` / `build_precheck_document` / `build_review_state`）；重绘 partial 回退件不再算成功 | 原 myc0t0xin A/B/C 合并路径未改 |
| `tools/style_merge_validate.py` | 新增 `--candidate` / `--reference` / `--candidate-keys` / `--state-name`，`--packages` 默认值按模式推导 | 原合并用法默认值不变 |

**未修改**：`utils/post_process.py`（通用工序固件）、`utils/analysis_gen.py`、`modules/image_analysis/style_analyzer.py`、`conf/config-styles.json`、候选 JSON、源数据集。
本轮未新建生图脚本；`_review/` 下的 3 个 `.py` 只做「拼联系表 / 收集数值 / 汇总表格」，不参与生图链路。

### 9.1 回归验证

| 检查 | 结果 |
|---|---|
| `py_compile` 三个改动文件 | 通过（exit 0） |
| `tests/test_style_merge_validation.py` + `test_style_analyzer_prompt_pack.py` | **全部通过**（含复用判定、字段哈希、状态文件结构） |
| `utils/output_isolation.py` 隔离效果 | `data/20261005/` 顶层与仓库根目录**零新增文件**；原运行与候选 JSON 的 SHA-256 未变 |
| `tests/test_analysis_gen.py` 与上一批同跑 | 4 项失败，**已证实与本轮改动无关** |

**关于那 4 项失败（如实记录，未修）**：把 `modules/others/api_backend.py` 的改动 `git stash` 撤销后，同一组命令得到**完全相同的 4 项失败** —— 属**既有用例隔离缺陷**，不是我引入的回归。
根因指向 `tests/test_style_merge_validation.py` 的打桩泄漏：假产物生产者 `_make_output` 写进 `tempfile.TemporaryDirectory()` 的临时目录，而该目录可能已被同一测试类另一个用例（`test_reuse_only_when_transmitted_content_matches` 也用 `self.root / "run"`）提前清掉；一旦假生产者抛 `FileNotFoundError`，`patch.object` 的还原被打断，`utils/analysis_gen.run_gpt_image_pipeline` 与 `api_backend.generate_image_repaint` 的桩会**泄漏到后续用例**，于是那 4 项看到的是假函数而不是真实现（失败信息里的 `out-N-final-rp.jpg`、缺 `use_detail_suffix` 就是它的指纹）。
- 单独跑 `tests/test_analysis_gen.py`：**37 passed**。
- 本轮只在该测试文件里加了一行 `path.parent.mkdir(parents=True, exist_ok=True)`（防御性，不改断言语义）；**根因仍在**，需要单独一轮修测试隔离，不在本轮范围。**没有**为了让用例变绿而改任何生产代码或断言。

## 10. 产物清单（图像类）

验证目录下共 76 个图像文件：**18 张真实生图产物** + 3 张候选联系表 + 1 张源图联系表 + 54 张脸/发/上半身放大裁剪。

18 张产物分布（每包每通道各 1 张）：

| 场景 | 原第 5 轮 | opt1 |
|---|---|---|
| A-lake-pink | 2048×2048 直出 / 1254×1254 首图 / 2048×2048 重绘 | 同上三路 |
| B-fullbody-soft | 同上三路 | 同上三路 |
| C-dark-contrast | 同上三路 | 同上三路 |

每张的 SHA-256 记录在对应 `attempt.json` 的 `output_sha256`，与 `generated_files` 双向可校验；重绘产物另记 `repaint_source_sha256`（= 该次 GPT 首图哈希）。**已核对三路产物哈希互不相同**（早期「重绘=首图同一文件」的问题已修复并复核）。

## 10. 限制

- 每包每通道**只跑了 1 次**，单张波动无法与整包差异分离；不得凭单张好图断言整包更优。
- A 组是**同条件重生成**（母题统一关闭），不是原运行第 5 轮的逐字复现；两者不可混为一谈。
- B/C 主体为中文书写，与原运行主体（英文）不同书写；B/C 只用于泛化观察，不与 A 组跨场景比较数值。
- 人工复核基于联系表 + 三处放大裁剪，**未逐张像素级审查全部 18 张**；源图侧也只看了整图联系表与参考图放大，未逐张放大 15 张源图。
- 本地渲染数值受构图与内容影响，只作描述统计，**不是相似度百分比**，也不参与任何总分。
- 本轮**没有**得出「已在内容上通过复核」的候选：C 组 opt1 直出为确认的内容失败，A 组有共有的装饰偏离与手部缺陷。
