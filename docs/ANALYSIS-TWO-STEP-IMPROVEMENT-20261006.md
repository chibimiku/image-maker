# 分析两步走（GPT-Luna 观察 + DeepSeek 改写）改进方案

对象：任务 `b51aa600`，输出 `data/20261006/b51aa600_180553-2f1979.jpg`。
输入诊断：`docs/ANALYSIS-B51AA600-REVIEW-20261006.md`（GPT 独立复核）。
本文件只做诊断与方案，不改生产代码、不改画风配置、不重启、不发新增生图请求。

## 一、对 GPT 诊断的复核结论

诊断成立，并补两条本地证据：

1. **今天对分析逻辑的改动不是主因，但确实在两条支线上放大了漂移。**
   `583bba7`（10-06 17:34，`observe-then-secondary-edit/v1`）做的是把四类生成改写从视觉提示里搬到 DeepSeek，Step 1 改为纯事实观察。本任务的 `source_analysis` 与 `english_description` 对比显示：主体、服装、姿态、视角、遮挡、环境全部保留，只多了「照片→插画」和语序整理。按要素计，该改动解释不了任何一项整图替换。
2. **但改动把「观察到的全部内容」变成了只读快照，最终生图并不读它。** 18:05:20 那次 Gemini 参考优先请求的内容锚是 `gpt_image_prompt_short`（57 词，配置项 `gemini_content_field`，本画风条目显式指定），附件只有 `submodules/image-maker-artstyle/references/myc0t0xin.jpg`，**源图未附给 Gemini**。也就是说，源图在整条链路里只贡献了文字，而文字被压缩到 57 词；生成器手上唯一的图像线索就是参考图本身。

**结论：这不是「分析两步走写错了」，而是「内容权威方（生成器）手里没有内容真相」。** 参考优先把参考图声明为权威，模型在内容含糊时把权威当成模板复刻，是当前配置下的合理输出。GPT 关于「不回滚 DeepSeek 策略、优先处理参考优先＋短锚的参考内容复制」的判断，本地证据支持。

### 今天改动带来的两处小漂移（真实存在，需要收口）

| 漂移 | 产生点 | 为什么是缺陷 |
|---|---|---|
| 不确定的深色眼睛 → 确定的 `dark eyes` | Step 2 `refine-desc.md`（放宽到「精确措辞与适度扩展」后，句 5 只管“不能把不确定改成确定”，但触发面太弱，实测未拦住） | 把观察的不确定性升级成事实，后续 `gpt_image_prompt_short` 直接继承 |
| 新增 `opaque black panels` | Step 3 服装检查 | 源文只确认「黑色遮盖形状」，不能确认服装结构；而 Step 3 跑在纯文字上（无图），本来就不该新增织物结构断言 |

两条都在 `docs/ANALYSIS-B51AA600-REVIEW-20261006.md` 里识别出来了，属于「收紧」而非「回滚」。

## 二、当前两步走的实际数据流（失效点已标注）

```
源图/剪贴板图
  └─ Step 1（视觉通道，本机 gpt-5.6-luna 类）：忠实观察 → source_analysis（事实快照，保留不确定与遮挡）
       └─ Step 2（DeepSeek 文本）：生成措辞改写 → english_description / short_description / 标签
            ├─ Step 3 服装检查（文本，无图）        ← 会新增未观察到的结构断言
            ├─ Step 4 去照片风格（文本，无图）
            └─ 生成 gpt_image_prompt / gpt_image_prompt_short     ← 短锚是最终唯一内容载体
                 └─ Gemini 参考优先（参考图 + 锚）        ← 唯一图像输入是画风参考图 ✗ 失效点 A
                      └─ 直出即发布（无内容一致性门禁）  ← 无拦截 ✗ 失效点 B
                 └─ 或 gpt-image 通道（有身份/人体/最终审计）  （本例未走）
```

- **失效点 A（主因）**：内容锚过短（57 词）且生成器看不到源图。参考图对模型而言既是风格又是唯一可见的"内容样本"。
- **失效点 B（放大器）**：Gemini 直出没有该任务的内容一致性自动拦截；严重复制参考主体的图仍被当正常产物保存并发布。GPT 链路有 `identity-audit` / `hand-audit` / `final-quality-audit`，Gemini 通道没有对等物。

## 三、改进方案（按性价比排序）

### M1 内容锚不再是"短锚"，而是多来源内容权威块（不改调用形态，先落地）

- `utils/analysis_gen.resolve_gemini_reference_content`：
  - 取 `source_analysis.short_description` + `short_description` + 标签，拼成 `CONTENT FACTS`（保留 `appears dark, colour not fully certain` 这类原句不确定性）；
  - 参考优先模式对 `gemini_content_field == "gpt_image_prompt_short"` 给一次显式告警日志（`[Gemini 画风] 短锚+参考优先：内容约束偏弱`），并把 `gpt_image_prompt`（1400 档）作为并发内容块追加，而不是二选一。
- 放宽长度预算的理由：`REF_PRIORITY_POSTAMBLE`（图后）与 `gemini-analysis-content-lock.md`（图后）都在图**之后**，内容后置不参与"压过画风图"的竞争；真正的风险是图前 style 段被顶长，所以只加图后，不动图前。
- 成本：0 次新增 API 调用。

### M2 源图作为"内容来源"随参考图一起发给 Gemini（直击主因）

- 数据流补齐：`prompt_bundle` 里已有 `source_image_path`，但 `single_analyzer._start_generation` 构造的 `prompt_context` 丢了它 → 补 `"source_image_path": ...`；再传给 `ImageGenWorkerThread(image_paths=[style_ref, source])`。
- 请求结构（`generate_image_aigc2d` 已支持多 `inline_data`，无需改后端）：
  1. 图前：参考优先声明 + 精简画风精简说明；
  2. 图 A：画风参考图（label：`STYLE REFERENCE — technique only`）；
  3. 图 B：源图（label：`CONTENT SOURCE — identity, clothing, pose, props, setting`）；
  4. 图后：内容权威块 + 参考图内容禁区 + `gemini-analysis-content-lock.md`。
- 新模板 `prompts/gemini-analysis-content-authority.md`，明确优先级：`CONTENT SOURCE image + CONTENT text` 决定一切内容；`STYLE REFERENCE image` 只决定画法；两者冲突时以内容为准。
- 复刻防御：把"内容来源图是待重构素材、不是待复制像素"写进图前声明（照片→插画本来就要重画，不需要复制像素），避免模型退化成 img2img。
- 剪贴板输入没有路径 → 只走 M1（现状保持）。
- 成本：每次生成请求多一张 1K 附件（`to_base64_compressed` 已降到 1448x2048 级），按输入图 token 计费；**不新增请求次数**，只是同一次请求的输入变大。

### M3 参考图"内容禁区"清单（离线生成一次、可复用、可人工改）

- 画风条目新增字段 `reference_content_exclusions: [str]`（`utils/styles.py::normalize_style_entry` 需保留该字段，否则会被静默吃掉——已有 `gemini_content_field` 的前车之鉴）。
- 生成方式：离线一次性对参考图做"签名字符素"提取（鹿角、下垂动物耳、紫红层叠裙、红长袜与靴、兔子、城堡、月夜、玫瑰框…），写入 `conf/config-styles.json`（子模块同步工具 `tools/sync_styles_to_submodule.py` 的字段白名单也要加），支持人工手改。
- 注入位置：图后，与 M2 的 CONTENT AUTHORITY 同段，形如 `THESE ELEMENTS BELONG TO THE STYLE SAMPLE AND MUST NOT APPEAR: ...`。
- 为什么需要它：`gemini-analysis-content-lock.md` 已有"不要导入样例角色/服装/背景"，实测未生效——抽象禁令对模型太弱，具名清单才可执行。
- 成本：一次性文本调用/画风（缓存）；生成时 0 额外调用。

### M4 首图内容一致性门禁（补上失效点 B）

- 新固件 `prompts/first-image-content-audit-system.md` + `utils/first_image_audit.py`：
  - 输入：源图（或 `source_analysis` 摘要 + 内容锚）+ 首图 → 输出 `{verdict, matched[], drifted[], copied_from_reference[], uncertain[]}`；
  - 判定：`matched` 覆盖身份/服装/姿态/道具/环境/画幅；`copied_from_reference` 命中 M3 清单项即算复制；输出 `drifted + copied` 分级。
- 接线：Gemini 直出保存后先写候选，审计不通过 → 记录标 `needs_review`（保留候选、禁止自动发布），不静默替换、不自动重生成。
- 与现有门禁的关系：与 GPT 链路的 `identity-audit` 同源思路，但独立实现，**不把 Gemini 图塞进 GPT 链路的门禁**（那是编辑链，坐标系与语义不同）。
- 成本：每次直出 +1 次文本/视觉调用；成功缓存复用；审计失败保留候选。
- **落地情况（2026-10-06）**：已落地为独立机制，见 `docs/ANALYSIS-FIRST-IMAGE-AUDIT-20261006.md`（默认开启、上限 2 张可配 3；审计不通过就带着失败原因重画，只有上限用尽仍不合格才标人工复核——重试这一段是用户后续明确追加的要求）。

### M5 两步走本体的收紧（小改动，防同类事实漂移）

1. `prompts/refine-desc.md` 第 5 条补硬规则：源文出现 `appears / seems / not fully certain / cannot be determined` 等不确定表述时，改写句**必须保留等价的 hedge**，禁止升级为断言（颜色、身份、结构、数量都适用）。`refine-desc-system.md` 同步一句。
2. `prompts/single-analyzer-outfit-check.md`：现状第 3~5 条与第 12 条是**开放式授权**（"把服装改到内部协调 / 改成目标风格"），第 9 条只禁内衣类改写，没有禁"补结构"，而本步又跑在无图文本上——本任务的 `opaque black panels` 正出自这里。改为：允许改的只有**颜色/配饰/风格归属**，禁止新增结构断言（panel / lining / layering / fastening / corset 这类），被遮盖区域只允许复述已观察到的覆盖情况；无依据时 `modified=false` 必须是可接受的输出（现在这个出口存在，但没有被提示鼓励）。
3. 新增回归 `tests/test_analysis_two_step_guardrails.py`：
   - 用本任务真实 `source_analysis` + Step 2 真实输出做 fixture，断言不确定表述被保留；
   - 断言 outfit check 不得引入 panel/lining 类新词；
   - 断言短锚+参考优先时会产生告警且内容块非空（M1）。
4. 产物快照补一个字段：生成时把实际使用的 `content field + 字符数 + 附件列表（roles）` 写进记录 `generation_params`（本例诊断时只能靠短锚文本文件反推，下次应可直读）。

### M6 验证方式（不拿"改了提示词"当已修复）

- 用受控对照卡（同一源图 + 同一画风）跑四格并用现有指标（`style_similarity` 深度指标 + 人工查看）比较：
  | 组 | 配置 |
  |---|---|
  | 对照 | 现状（参考优先 + 57 词短锚） |
  | A | M1（内容权威块，无源图） |
  | B | M1+M2（附源图，roles 标注） |
  | C | M2+M3（附源图 + 禁区清单） |
- 判定最小集：①「坐椅子/书/鹿角/月亮/城堡/兔子/玫瑰框」是否消失；②姿态与服装是否回到蹲跪＋粉黑动物纹；③画风是否仍在（不能靠牺牲画风换内容），三档用深度指标与人工并看。
- 全部四格记录提示词 hash、附件 roles 与实际请求快照；未通过的组合保留，不删。**新的生成配额先申请，不默认授权。**

## 四、明确不做

- 不回滚 `observe-then-secondary-edit/v1`；不改 Step 1 为"边观察边改写"（那正是本次要保留的改进）。
- 不把参考图从参考优先里删掉（画风收益有实测支撑）；改为"参考管画法、源图管内容"的分工。
- 不用"加一句更长的禁令"替代 M2/M3（本次已证明抽象护栏不生效）。
- Gemini 直出审计不取代 GPT 链路的既有门禁，不合并两条链路。

## 五、边界与未验证项

- 未做在线复现；本文所有结论来自本任务留存的 JSON、Step 2 快照、日志与配置，没有新增 API 请求，没有重启 app。
- M1/M5 属提示词与组装改动，纯离线可回归；M2/M3/M4 涉及生成链路行为与配额，需用户授权后执行并登记请求快照。
- 审计门禁本身也会误判（把合理重构判成漂移）——首版只拦 `verdict=fail` 且 `copied_from_reference` 非空的情形，`uncertain` 只标记不拦。

## 六、实施进度（2026-10-06）

- **M4 首版已完成**（审计 + 重试）：`utils/first_image_audit.py`、`prompts/first-image-audit-system.md`、`prompts/first-image-audit-retry.md`、单图分析通道行的「首图审计」勾选框与「最多 2/3 张」下拉、画风条目可选字段 `reference_content_exclusions`。机制、判定规则、花费模型与边界见 `docs/ANALYSIS-FIRST-IMAGE-AUDIT-20261006.md`。**未在线验证，不宣称已修复**（M6 的纪律不变）。
- **M1 / M2 / M3 / M5 仍未做**：多来源内容权威块（M1）、源图随参考图一起发给 Gemini（M2）、参考图内容禁区清单的离线生成（M3 —— 字段与消费侧已就绪，清单本身还是空的，缺省走兜底禁令）、两步走本体的收紧（M5）都还没落地。
