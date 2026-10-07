# Gemini 直出「首图审计 + 重画」机制说明（2026-10-06）

对象：图片分析 Tab 的 **Gemini 直出**通道（`gen_channel_gemini`）。
状态：**已实现、未在线验证**。离线回归 `tests/test_first_image_audit.py` 实测 **34 项全绿**（"发布名提升"那条先暴露过一次真实缺陷，已修并留了回归，见 §八 末条）；全量 `python -m pytest tests/ -q --ignore=tests/test_style_ref.py` = 1301 passed / 9 failed，9 项在本次改动前的 HEAD（583bba7）上同样失败（都是需要真实文本端点或直接读工作区配置的既有用例）→ **零新增回归**。审计提示词的实际命中率、重画后是否真的纠正了漂移，都没有跑过真实生成 —— **不能用本文宣称"抽卡/参考图内容复制已修复"**。
本轮不改 `conf/config-styles.json` 里的画风数据，不发新增在线请求，**改完必须重启 app 才生效**（GUI 进程里是旧模块）。

---

## 一、需求来源与诊断关系

用户 2026-10-06 的两个反馈合起来提出了这次改动：

1. **Gemini 直出有抽卡性** —— 同样的输入，出图质量/贴合度不稳定，偶尔第一张就明显不合要求；
2. **「参考优先 + 短内容锚」会把参考图的内容复刻进新图** —— 该次输出抄了画风参考图（`myc0t0xin.jpg`）的角色、服装、道具与月夜城堡/玫瑰框。

用户的要求是「一次审计 + 重试」：**生成后先审计，不通过（内容偏离 / 抄了参考图内容 / 没跟上参考画风）就带着失败原因重画一次，上限 2 张（可配 3）**。

与既有诊断文档的关系：

| 文档 | 关系 |
|---|---|
| `docs/ANALYSIS-B51AA600-REVIEW-20261006.md` | **上游诊断**。它定位了两个失效点：**A** 内容锚只有 57 词且源图没有附给 Gemini，参考图成了模型手上唯一的图像线索（主因）；**B** Gemini 直出没有内容一致性门禁，严重复制参考主体的图仍被当正常产物保存并发布（放大器）。本文实现的正是**失效点 B 的拦截器**：把"出图后没人看"补成"出图后必审、不合格重画"。 |
| `docs/ANALYSIS-TWO-STEP-IMPROVEMENT-20261006.md` | **方案来源**。该文把改进拆成 M1~M6，其中 **M4 =「首图内容一致性门禁」+ 独立 `utils/first_image_audit.py`**。本次是该方案的**首版落地**，同时吸收了 M6 的验收纪律（不拿"改了提示词"当已修复）。 |
| 与 M4 草案的差异 | 草案写的是「审计不通过 → 标 `needs_review`、保留候选、**不自动重生成**」；用户随后明确要求"不通过就重画一次"。落地版因此是**审计 → 重画（上限内）→ 仍然全部不合格才人工复核**，重试与人工复核两段都有。 |

M1（多来源内容权威块）、M2（源图随参考图发给 Gemini）、M3（参考图内容禁区清单的数据生成）、M5（两步走本体收紧）**本次都没有做**；M3 只落地了它的**字段与消费侧**（画风条目 `reference_content_exclusions` + 重画指令里的具名禁区），清单本身仍需离线生成或手写。

---

## 二、机制流程

```
[图片分析 Tab · 生图通道 = Gemini 直出]
   │
   │ ① 通道行勾选「首图审计」（默认勾选）＋「最多 2 张 / 最多 3 张」下拉
   │ ② 组装审计回调 build_first_image_audit_callback(...)
   │      内容规格 = 本次提示词 ＋ 分析产物完整锚 gpt_image_prompt（短锚档用短锚）＋ 授权服装正文
   │      参考图禁区 = 画风条目 reference_content_exclusions（缺 → DEFAULT_FORBIDDEN_ITEMS）
   │      审计图序 = 生成图 → 画风样本（若有）→ 源图（若有）
   ▼
第 1 次出图 ──► 候选 <发布名>-a1.png（落在发布目录）
   ▼
审计（视觉文本通道 call_text_model，主文本 API）
   三类判定：① 内容保真  ② 参考图内容复制  ③ 画风覆盖
   返回严格 JSON：content_ok / style_ok / copied_from_reference / verdict / issues…
   ▼
结论有效吗？（缺字段、JSON 取不到、调用抛异常都算无效）
   │
   ├─ 无效 ──► 视为「不通过」，补画一次（缺字段绝不等于通过）
   ▼ 有效
should_retry?
   │
   ├─ 不重画 ──► ✅ 通过：选中这张，另存**不带 -aN** 的发布名
   │              stopped_reason = passed
   │
   └─ 重画（copied_from_reference=true / major 且 confidence≥0.65 的内容或画风不符 / verdict=fail）
          │
          ├─ 还有额度 ──► build_retry_prompt(...) 渲染 prompts/first-image-audit-retry.md
          │                 （第几次尝试 ＋ 打回原因 ＋ 内容权威 ＋ 具名禁区）
          │                 → 带着这份指令回到「出图」（第 2 张候选 <发布名>-a2.png）
          │
          └─ 额度用尽 ──► 🛑 stopped_reason = exhausted
                          按 attempt_score 选"最可用"的一张：
                          审计通过 > 未审计 > 明确不合格；
                          同级比 内容 > 画风 > 未复制 > 更新的尝试
                          产物保留，队列写 pipeline_error + first_image_audit 摘要
                          → 需要人工复核（不自动当正常产物）
```

其它出口（不常见但都有明确记录）：`cancelled`（用户中止，不再重画）、`no_output`（这一次没出图，继续用额度）、`audit_disabled`（没给审计回调 → 只出一张就停）、`audit_unavailable`（回调不可调用 → 停）。

---

## 三、判定规则表

### 会重画

| 审计结论 | 重画原因（写进日志与重画指令） |
|---|---|
| `copied_from_reference = true` | 「复制了画风样本的内容元素」——画得再好看也算失败，这是本次要治的主症 |
| `content_ok = false` 且存在 **major** 问题（`confidence ≥ 0.65`） | 「内容与本次要求明显不符」 |
| `style_ok = false` 且存在 **major** 问题（`confidence ≥ 0.65`） | 「没有跟上画风样本的画法」 |
| `verdict = "fail"`（即使没有 major 明细） | 「审计判为不合格」 |
| 结论无效：缺 `content_ok` / `style_ok` / `verdict` / `copied_from_reference` 任一，或 JSON 抽不出来 | 「审计未给出有效结论（缺字段或格式非法），补画一次」 |
| 审计调用抛异常（端点报错、超时、内容规格缺失） | 同上：异常被折成「结论无效」，**不静默当通过** |

### 不会重画

| 情况 | 为什么不重画 |
|---|---|
| 只有 `uncertain` 项 | 抽卡解决不了不确定；不确定只记录在 `audit.uncertain` 里，留给人工看 |
| 只有 `minor` 问题 | minor 不构成重画理由（与身份审计同口径） |
| `major` 但 `confidence < 0.65` | 置信度不足的 major 不触发重画（阈值与 `utils/identity_audit` 一致） |
| 审计整体通过 | 直接用这一张，`stopped_reason = passed` |
| 上游没有可审计的条件（剪贴板输入、没分析产物也没画风图） | 回调返回 `None`，审计整体跳过并打印日志，走老路径 |

### 为什么「缺字段」不算通过

「模型没给出结论」和「模型给了通过的结论」是两件事。首版一律按**最不利**解释：缺字段 → `conclusion_valid=False` → `should_retry` 返回 True（还有额度就补画一次）。另外两条防漏：

- 结论自相矛盾（`verdict=pass` 却报了 major 问题）按 **fail** 处理，并在摘要里写明冲突；
- 结论无效时**不允许保留 pass**：`verdict` 强制改回 `unknown`（缺字段的回复里写着 pass 只是模型漏答）。

---

## 四、花费模型

| 项目 | 计费次数 |
|---|---|
| 每张候选图 | **1 次 Gemini 生图计费**（按当前配置的分辨率/模型计价） |
| 每张候选图的审计 | **1 次文本调用**（主文本 API 视觉通道，本机 `gpt-5.6-luna`） |
| 默认上限 2 张（= 首图 + 1 次重画） | 最坏 **2 次生图 + 2 次审计** |
| 选「最多 3 张」 | 最坏 **3 次生图 + 3 次审计** |

- **唯一的花费闸门是 `max_attempts`**：上限内不通过才重画，到顶就停；不会"重画到满意"。
- 审计关掉（没回调）时只出 1 张（`audit_disabled`），不产生审计调用。
- 倒计时预算同步放大：开了审计后按 `单张超时 × 最多张数` 计时，否则第 2 张还在画就会被判超时。

**缓存/复用边界**：

- 审计记录 `<prefix>-first-image-audit.json` **会落盘供复核，但机制本身不复用旧候选** —— 每次任务重新出图，历史候选不会被送回审计、也不会被直接提升为发布图。
- 与 gpt 链路的"成功请求缓存复用"不同：这里的重画是**新的计费请求**，不存在"复用上次那张"的省费路径。

**`send_budget` 实验额度（仅在启用时生效）**：

- 每次审计都包在 `utils.send_budget.guarded_text_call("first-image-audit", …)` 里。
- **未设置 `IMAGE_MAKER_SEND_BUDGET_DIR` 时这一层完全惰性**（正常 GUI 不受影响，`guarded_text_call` 直接透传）。
- 启用后：同一 `run_id` 的 `first-image-audit` 阶段**最多 2 次（含失败）**，全局文本审计上限 64 次；额度用尽或同一 run 重复执行会**直接拒绝发送**（`SendBudgetExhausted`），不会偷偷多花钱。
- 注意：这套额度只管**审计文本调用**；生图张数仍由 `max_attempts` 决定，两者不互相顶替。

---

## 五、产物与可复核性

| 产物 | 命名/位置 | 说明 |
|---|---|---|
| 候选图 | `<发布名主干>-a1.png` / `-a2.png` / `-a3.png` | `-aN` 表示**第几次尝试**；被审计打回的候选**保留**，不删不覆盖 |
| 发布图 | `<hash>_<HHMMSS>-first-<token>.png`（**不带 `-aN`**） | 设计契约：选中的那张另存一份发布名；**8 位 task hash 仍在第一个下划线字段**，`publish_server.py` 靠它关联同目录的投稿 JSON。提升失败（候选被删/不可写）时回落候选原文件并打日志，不静默丢交付 |
| 发布目录 | 分析产物所在目录（通常是 `data/<日期>/`）；拿不到时当天 `data/<日期>/` | 设了 `IMAGE_MAKER_TEST_OUTPUT` 时走 `resolve_output_target` 改道 `data/test-result/…`，不污染真实产出 |
| 审计记录 | `<发布名主干>-first-image-audit.json` | 见下 |
| 队列记录 | `first_image_audit` 摘要 + （全不合格时）`pipeline_error` | 见下 |
| 日志 | `🔍/⚠️/✅/🛑` 每轮一行 + `=== 首图审计结果 ===` 块 | 块里是 `stopped_reason` / `suspended` / `selected` / 每张的 `verdict` + `summary` |

**`<prefix>-first-image-audit.json` 内容**：`max_attempts`、`stopped_reason`、`selected_file`、`selected_round`、`candidates[]`；每个候选记 `round`、`file`、**该次实际使用的 `prompt`（含重画指令）**、规范化后的 `audit`（内容/画风问题、复制元素、不确定项、`conclusion_valid`、major 明细、模型原始回复 `raw` 与模型名）。**不含任何 API key/端点等凭据**，可以直接拿去复核"当时到底判了什么、为什么选这张"。

**队列摘要**：`first_image_audit = {stopped_reason, suspended, attempts, selected}`。当**所有候选都被判不合格**（`suspended=True`）时，额外写

```
pipeline_error = "首图审计未通过（N 张候选），需要人工复核：<选中文件名>"
```

即：产物保留（人工复核要用），但这条任务**不自动当正常产物**，也不进自动发布。

---

## 六、文件清单与配置项

### 文件清单

| 文件 | 角色 |
|---|---|
| `utils/first_image_audit.py` | **判定层**（不发请求）：`extract_json_object` / `normalize_audit` / `should_retry` / `audit_reasons` / `build_retry_prompt` / `attempt_score` / `choose_best_candidate` / `promote_candidate` / `generate_with_audit` / `audit_settings` / `resolve_audit_content` / `audit_first_image` |
| `prompts/first-image-audit-system.md` | 审计固件：三类判定（内容保真 / 参考图内容复制 / 画风覆盖）+ 严格 JSON schema + severity 口径（`fail` 必须来自 confidence ≥ 0.65 的 major 或 `copied_from_reference=true`） |
| `prompts/first-image-audit-retry.md` | 重画指令模板：`{attempt}` / `{max_attempts}` / `{reasons}` / `{content}` / `{forbidden}`，明文要求"重画、不要在原图上小改" |
| `utils/styles.py` | `normalize_style_entry` 保留并透出 `reference_content_exclusions`；新增 `reference_content_items(styles, name)` 取具名禁区 |
| `modules/image_analysis/single_analyzer.py` | 接线：`ImageGenWorkerThread(audit_retry / audit_callback / audit_max_attempts / publish_dir / publish_name)`、`_run_with_audit()`、模块级 `build_first_image_audit_callback()` / `first_image_audit_settings()` / `first_image_publish_target()` / `build_publish_name()`、通道行勾选框与数量下拉、队列摘要与 `pipeline_error` |
| `tests/test_first_image_audit.py` | 离线回归（判定层 + 接线层，全部假函数，不发请求） |

### 配置项

| 位置 | 键 / 控件 | 默认 | 行为与边界 |
|---|---|---|---|
| `conf/config.json` | `gemini_audit_retry_enabled` | **true**（缺键即开） | 置 `false` 后 GUI 直接跳过审计并打印「首图审计已在配置里关闭」 |
| `conf/config.json` | `gemini_audit_retry_max_attempts` | **2** | 判定层默认上限，读取时硬夹到 `[1, 3]`；GUI 上实际张数由通道行下拉决定 |
| 画风条目（`conf/config-styles.json`） | `reference_content_exclusions: [str]` | 空 | 具名"参考图内容禁区"（如 `antlers`、`rose frame`、`purple moonlit castle`）；进重画指令；最多取 12 条；**缺失时用 `utils.first_image_audit.DEFAULT_FORBIDDEN_ITEMS` 兜底**。字段必须在 `normalize_style_entry` 里保留，否则会被归一化静默吃掉（同 `gemini_content_field` 的教训） |
| 通道行控件 | 「首图审计」勾选框 | **勾选** | 仅 Gemini 直出有效；走 gpt 通道时**置灰但仍可见**（不是隐藏） |
| 通道行控件 | 「最多 2 张 / 最多 3 张」下拉 | 2 张 | 与 `gpt_ui_state` 一起持久化（`gemini_audit_retry` / `gemini_audit_attempts`），重启后仍然生效 |

---

## 七、明确不做

1. **不动 gpt-image 通道的门禁**：gpt 链路已有身份审计 / 人体门禁 / 最终质量审计与断点缓存；首图审计只在 Gemini 直出分支生效，走 gpt 时勾选框置灰。也不把 Gemini 图塞进 GPT 链路的门禁（坐标系与语义不同）。
2. **不把"最可用"解释成"最好"之外的东西**：上限用尽时只按固定排序（通过 > 未审计 > 不合格；同级比内容 > 画风 > 未复制 > 更新）选一张，不继续抽卡、不悄悄提高上限、不换模型、不静默删掉失败的候选。
3. **不在不确定时反复抽卡**：`uncertain` 只记录不重画 —— 重试解决不了不确定，只会白花钱。
4. **不静默吞掉审计失败**：审计异常折成"结论无效"并落盘；全部候选都不合格时显式写 `pipeline_error` 与 `suspended=True`，交给人工复核，而不是当正常产物发布。
5. **不改本轮之外的既有行为**：不动分析两步走（`observe-then-secondary-edit/v1` 不回滚）、不动画风配置数据、不改 Gemini 请求里图前的组装顺序（重画指令只加在**图后**的正文里）。

---

## 八、未验证 / 边界

- **没有在线跑过**：审计提示词的实际命中率、重画后是否真的纠正了漂移，一次真实生成都没有验证过。**不能声称"抽卡问题已解决""参考图内容复制已修复"。**
- **审计模型自己也会误判**：首版只拦 `copied_from_reference` + confidence ≥ 0.65 的 major + 无效结论；误判有两个方向 —— 把合理的重构判成漂移（白画一张、多花一次钱），或把漂移判成通过（仍然出问题图）。`uncertain` 与 minor 一律只记录。
- **`suspended=True` 的图仍要人工看**：它只表示"这一组候选没有一张拿到通过结论"，可能是真的都不行，也可能是审计误判 —— 判定层不做最终裁决。
- **可用性依赖分析产物**：剪贴板输入没有分析 JSON，也没有源图路径 → 回调为 `None`，审计整体跳过（日志写明），行为等同改动前。
- **审计通道不可用时**：结论 → 无效 → 补画一次，最多把额度用满，绝对不会把"审不出来"当通过。
- **重画指令的作用面有限**：它重申内容权威与具名禁区、给出打回原因，但**不改图前的参考优先声明**；这条路径的效果同样没有在线数据。
- 本机可复核的只有离线回归：`$env:PYTHONPATH="D:\code\image-maker"; python tests/test_first_image_audit.py`（判定层 + 接线层，全部假函数，不发任何请求）。
- **发布名提升曾经失效过一次（已修，留有回归）**：首版 `_run_with_audit()` 的 `publish_name` 回调返回的是**候选名**（`…-aN.png`），`promote_candidate` 见"目标与源是同一个文件"就直接返回、不再另存 → 发布图会带上 `-aN`。现在该回调恒定返回基础发布名，`tests/test_first_image_audit.py::WorkerThreadAuditTests::test_second_round_is_drawn_with_the_retry_directive_and_only_one_publish_file` 断言"目录里有两张 `-aN` 候选 + 一张不带 `-aN` 的发布图"，再次踩到会立刻变红。

---

## 九、回滚方式

本次改动是**独立提交范围**（新文件 + 单图分析通道接线 + 画风字段保留），回滚请用 `git revert <commit>`，**不要用 `git reset`** —— 工作区里还有其它在途改动，reset 会连带回滚不该动的东西。

- 需要还原的文件：`utils/first_image_audit.py`、`prompts/first-image-audit-system.md`、`prompts/first-image-audit-retry.md`、`tests/test_first_image_audit.py`（新增）；`utils/styles.py`（`reference_content_exclusions` 保留 + `reference_content_items`）与 `modules/image_analysis/single_analyzer.py`（回调、线程参数、UI 控件、队列摘要）为改动点。
- 画风条目里如果已经写了 `reference_content_exclusions`，那是**数据不是代码**：旧版本没有这个字段的保留逻辑，会被 `normalize_style_entry` 静默忽略，回滚代码后不需要单独清数据（想彻底清理可手工删除字段）。

---

## 十、本次误发的在线生图（必须记账，别当成实验数据）

实现期间为了"跑相关回归"，执行过 `tests/test_style_ref.py` —— 该文件**不是离线单元测试**，而是会真调付费生图接口的五组对照脚本。结果：**2026-10-06 21:20:06 – 21:23:26 实际发出 5 次 `gemini-3-pro-image-preview:generateContent`**（A/B/C/D/E 各一张，画风 `puracotte-style-v2`，同一测试主体），全部成功。

- 产物全部落在测试隔离目录：`data/test-result/20261006/style-ref-horizontal/puracotte-style-v2/`，**没有污染 `data/20261006/` 的真实产出**。
- 这 5 张与本机制无关，**不能当作审计/重画的验证数据**（它们甚至没走审计）。真实文件见 `log/2026-10-06.log` 的 21:20–21:23 段。
- 教训：跑"相关回归"前先确认目标不是在线脚本。以后跑测试一律 **`python -m pytest tests/ -q --ignore=tests/test_style_ref.py`**；`tests/test_style_ref.py` 只在明确要花这一笔钱时手动执行。
- 回滚后**必须重启 app**（GUI 进程里是旧模块），否则界面上还会带着旧的勾选框状态。
