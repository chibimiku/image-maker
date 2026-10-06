# retention-v2 离线交付修正

日期：2026-10-04。性质：**纯离线**。本轮没有收费 API 调用、没有新图片、没有视觉重评、没有修订生图。
修正交付目录：`data/test-result/20261004/color-palette-retention-v2-corrections/`。
旧证据（旧图、原始请求/响应、冻结文件、账本、已交付报告）**全部保留**在
`data/test-result/20261004/color-palette-retention-v2/`，本文件与修正交付只引用它们，不覆盖。

## 一、B-r1「鞋被裁切」的事实分歧（逐条核对）

用户观察：**实际产图里两只鞋及鞋底完整可见**，而 quality / final_review 声称鞋被底边裁切。

### 1.1 审计绑定的候选 hash 与实际产图：一致

| 项 | 值 |
|---|---|
| 实际产图 | `.../color-palette-retention-v2/stages/B-gemini-flash-P1-knowledge-r1-initial-repaint/retention-B-gemini-flash-P1-knowledge-r1_214833-cb6415.jpg` |
| 产物记录 hash | `d70137487fef772a5a3bebd16633127cfcfe292357eea7fd8d43340848fc05dc` |
| quality.json `candidate_sha256` | 同上（一致） |
| final-review.json `candidate_sha256` | 同上（一致） |
| 该次审计提交的图片 hash 列表 `audit_image_sha256` | 首图 `e25e19c8…`、候选 `d7013748…`、画风参考 `188f0312…`（顺序与声明一致） |

结论：**不存在错图或错位绑定**，审计审的就是这张候选。

### 1.2 真正提交的全图代理、细节拼图与顺序：已重建（并明确标注不是历史输入）

**本批审计没有保存实际发送的图片字节**——当时只落了审计结论与输入路径/hash。因此本轮的证据是
**按当前代码重建**：`input-evidence/<支>-<样本>/{candidate_full_proxy.jpg, candidate_detail_sheet.jpg}`
（重建件在原图上只做缩放/裁剪，**不修改原图**）。

提交顺序（`stages/*-gates/quality.json` 的 `audit_images` + `utils/refine_quality.audit_refine_quality` 代码）：

| 顺序 | 角色 | 内容 |
|---|---|---|
| Image 1 | 首图全图 | 期望来源 |
| Image 2 | 候选全图 | `_proxy(candidate)`：`thumbnail((1536,1536))` → 2528×1696 缩放到 1536×1030 |
| Image 3 | 画风参考图全图 | 画法参照 |
| Image 4 | **候选九宫格细节拼图** | `_detail_sheet(candidate)`：3×3 共 9 格、每格 512px 贴进 1536×1536 白底 |

### 1.3 是否有裁剪、缩放、错图、细节图被误当全图

- **没有错图、没有额外裁剪**：只有一次等比缩放到 1536 长边。
- **细节拼图是独立的一张（Image 4）**，与全图（Image 2）分开提交；系统提示词里也写明了图片顺序。
  因此「细节图被误当全图」在输入层面不成立。
- **但帧边界在细节拼图里出现了 9 次**：放大后的九宫格把「每一格的底边」都变成了视觉上的画幅边界。
  重建的 `candidate_detail_sheet.jpg` 里能看到左鞋鞋底在单元格底边处被切断的样子 —— 这是最容易让评审把
  「第三行单元格的底边」读成「图像底边」的地方。

### 1.4 逐像素与目视证据

- **全图目视（放大底部 24% 条带）**：两只鞋与鞋底**完整可见**，鞋跟、鞋底厚度、两只红色蝴蝶结都能看清。
- **像素测量（可复算）**：底部 12% 条带里最大的深色低饱和连通域（鞋体主体）距画幅底边 **75 px**（候选高 1696），
  该连通域**没有贴到画幅底边**（记录在 `corrected-per-image.json → shoe_region_evidence`）。
  说明：这个测量只说明「那一块深色区域」的位置，不等于对整只鞋做了像素分类。
- **首图对照**：首图（1264×848）底边同样有深色像素（`y=847` 那一行仍有 90 个深色像素），
  也就是说**首图本身也是一张主体贴近下边缘的构图**；候选延续了这个构图。
- **一处真实差异**：在九宫格细节拼图的放大视角下，**左鞋**的鞋底在画面最下缘处是贴边的；
  **右脚（含鞋底）距底边仍有明显空隙**。模型说的是「right platform heel 被裁切」，
  与实测（贴近下缘的是左脚一侧）**在指认对象上不一致**。

### 1.5 分歧状态：根因**未确定**（不是「模型错」也不是「用户错」）

保留原始模型判断不动，另记目视观察与分歧状态：

| 项 | 内容 |
|---|---|
| 原始模型判断（原样保留） | quality：`structural_issues` 含「right platform heel … visibly cropped」「figure 放大导致下前景丢失」，严重度 major；final_review：`repair`（结构、背景或非轻微质量缺陷） |
| 目视观察（本轮） | 两只鞋与鞋底在全图里完整可读；左鞋一侧贴近画幅最下缘，右脚一侧离底边更远 |
| 分歧状态 | **事实分歧已定位**：模型把「贴近下缘」描述成了「右脚被裁切」；但「人物尺度变大、下前景变窄」这一构图观察本身与实测相符，且首图也有同样的贴边构图 |
| 根因 | **未确定**。可能的来源：① 放大后的九宫格细节拼图让评审把单元格底边误读为画幅底边；② 模型对左右脚的指认错误。现有证据无法区分这两者，因为**历史发送字节没有保存** |
| 处理 | **不删掉裁切判定、不据目视缩小人物、不补背景、不宣布门禁通过**；该候选最终状态仍由离线重算的门禁结论决定 |

## 二、统一门禁交付（消除文件之间的矛盾）

矛盾事实：两张 A 的 `stages/*-gates/gates.json` 里 `outcomes.hands = "fail"`（旧口径写的），
而 `retention-summary.json` 与执行报告用的是重算后的 `pass` —— 同一批文件两个口径。

按**现有同一判据**离线重算（`utils/refine_quality.normalize_quality_audit(schema="anatomy")` +
`should_refine_quality` + 归属不确定/结论有效性 + `final_quality_decision`，**不关闭、不放宽**），
修正交付里每个候选的每一项都分别保存四种状态：

| 候选 | identity | quality | hands | final_review | 最终处理状态 |
|---|---|---|---|---|---|
| P1-knowledge-r1 · A | pass | **needs_refine** | pass | pass | revision_required |
| P1-knowledge-r2 · A | pass | **needs_refine** | pass | pass | revision_required |
| P1-knowledge-r1 · B | pass | **needs_refine** | pass | **fail** | revision_required |
| P1-knowledge-r2 · B | pass | **needs_refine** | pass | **fail** | revision_required |

- `原始模型判断`（记录里原本写的东西）、`离线重算`（同一判据重算）、`目视观察`、`最终处理状态` 四栏分开存：
  `corrected-gates.json` 与 `corrected-per-image.json` 同源，`corrections-review.html` 由它们渲染。
- 状态分类**不混为一类**：`not_executed`（预算阻止未执行）、`audit_error`（审计报错）、
  `fail`（明确不通过）、`pass` 各自独立；本批四张都没有未执行/审计错误项。
- A 两张的 hands 从 fail 改成 pass 的依据：旧记录缺 `hands_clear` 字段时**按同一判据重算**
  （明确结构缺陷 / 结论无效 / 归属不确定三者任一都不通过），而不是把「没有这个字段」当成不通过。
  旧文件里的 `fail` 原样保留在旧目录，没有改写。
- **自动一致性核对**：`python tools/color_knowledge.py retention verify-corrections` →
  `verification.json`（`status: passed`）。它检查：gates 与逐张文件的最终状态一致、逐项重算一致、
  候选 hash 绑定一致、判据取值合法、预算阻止/审计错误单独标注、汇总计数一致、
  「标记可用但存在未通过项」、以及「未附参考图却声称无泄漏」。回归用例
  `tests/test_color_retention_corrections.py::test_verification_catches_an_injected_inconsistency`
  会故意注入一处矛盾，确认能被拦住。

## 三、缓存指纹规范（审计与配色评审）

### 3.1 复用键的组成（`RETENTION_FINGERPRINT_FIELDS`）

`protocol_version`、`audit_kind`、`candidate_sha256`、`source_image_sha256`、`style_reference_sha256`、
`expected_spec_sha256`（审计期望规格文本）、`prompt_sha256`（审计 user 载荷）、`model`、
`params_sha256`（分辨率/比例/repeat/detail suffix/reference_mode/scope），另加
`system_prompt_sha256`（该审计的 system prompt 文件 hash）。

判定结果只有三种：`verified_match`（逐项一致才允许复用）、`mismatch`（任一变化即拒绝）、
`unverified_fingerprint`（旧记录缺必要指纹 → **复用资格未验证**）。

### 3.2 本批实际状态（如实记录，不补造）

- **A 支四项审计来自 v1**：`candidate_sha256`、`audit_kind` 有（以及 quality/final_review 的
  `audit_image_sha256` 前两项可读），但 **`protocol_version` / `expected_spec_sha256` / `prompt_sha256` /
  `model` / `params_sha256` 都没有保存**，hands 审计连提交图片 hash 列表都没有。
  → 全部记为 `unverified_fingerprint`；**没有补造指纹，也没有联网补审计**。
- **B 支四项审计是本批新调用**：同样没有保存完整指纹（当时代码还没有指纹功能），
  → `reuse_eligibility.state = not_applicable`（本批新调用，不涉及复用）+ `fingerprint_completeness.missing` 如实列出。
- 因此**下一次续跑不能据本批记录复用任何审计**；要复用就得先有新格式的完整指纹。

### 3.3 配色评审缓存

只看「结果文件在不在」**不算复用资格**：`retention_review_cache_state()` 会重算
`retention_colour_review_signature()`（协议版本 + system 提示词 hash + user 载荷 hash + 模型 +
图片路径/角色/hash + 逐项样本映射）并与记录里的 `request_signature` 比对；
没有签名 → `unverified_fingerprint`；签名不同 → `mismatch`。
本批旧评审记录**没有保存签名**，所以它们一律记「复用资格未验证」。

回归用例（全部 mock + tmp_path，零联网、不写交付目录）：
`tests/test_color_retention_corrections.py` 13 条 —— 指纹完整性、缺指纹不补造、
逐项变化拒绝复用、完全匹配允许复用、旧实验扫描一律 `unverified_fingerprint`、
配色评审签名缺失/不匹配/匹配三种状态、修正交付不改旧文件、A 的 hands 重算为 pass 且旧记录保持 fail、
泄漏一律 unverifiable、注入矛盾能被一致性测试拦住、鞋部证据测量不改原图、复核预览不调用。
（另有 `tests/test_color_retention_fix.py` 16 条保护 v2 执行链路本身。）

## 四、颜色泄漏口径修正

**事实**：本批配色评审的请求里，画风参考图**只有路径文字，没有随请求发送图片**
（`review/colour-*.request.json` 的 `images` 只含首图与候选及其细节裁剪）。

因此：

- **可引用的结论**（证据充分）：环境主色对首图、红色点缀与腰带/左鞋/右鞋落点、受保护固有颜色、
  新增显著色块、内容变化（含画幅）—— 这些都有对应首图与候选作为输入。
- **不可判定的结论**：**参考颜色泄漏**。旧评审给出的 `leak: none`（B 支两张）**不能当作「已证明没有参考颜色泄漏」**；
  交付里统一记 `leak = unverifiable`，并保留 `original_model_leak_claim`（原始模型声明）与不可验证原因。
- 口径一句话：**「未观察到偏色」不等于「证明没有参考颜色泄漏」**。

**为未来复核准备的请求预览**（本轮只准备、**不调用**）：
`recheck-request-preview.json` —— 明确图片角色与顺序（首图 → 候选全图 → **真正附上的画风参考图** →
候选局部裁剪，并注明局部裁剪不用于判断画幅、也不作为泄漏证据的唯一依据），
列出每张图的路径与 hash，以及复核的硬要求（必须真正附参考图、逐张分开判定、颜色相近只能支持「疑似」）。

## 五、复用的 CLI 与代码位置（没有新增一次性脚本）

- CLI：`python tools/color_knowledge.py retention {correct, verify-corrections, preview, freeze, run, audit-only, outcomes, ...}`
- 实验消费层：`utils/color_experiment.py`（`retention_corrections` / `retention_verify_corrections` /
  `write_retention_corrections_page` / `retention_recheck_preview` / `retention_audit_fingerprint` /
  `retention_reuse_eligibility` / `retention_review_cache_state` / `retention_shoe_region_evidence`）
- 提示词仍在 `prompts/`：`color-knowledge/retention-colour-review-system.md`、
  `color-knowledge/repaint-colour-contract-v1-source.md`、`color-knowledge/fixed-palette-retention-v2.json`
- 未改：生产 GUI、`conf/` 全局配置、画风固件、身份/质量/人体/终审门禁的判据与提示词。

## 六、仍未解决的问题

1. **历史发送字节没有保存**：所有「审计输入」的证据都是重建件，只能说明形态与顺序，不能证明当时就是它。
2. **B-r1 分歧的根因未确定**：无法区分「九宫格帧边界误读」与「模型指认错误」。
3. **指纹缺口无法回补**：本批所有审计都没有完整指纹，续跑不能据本批记录复用；要复用必须先重新审计。
4. **参考颜色泄漏不可判定**：需要一次真正附上参考图的复核。
5. **门禁后可用仍为 0/4**：A 卡质量（needs_refine），B 卡终审；本轮**没有做任何门禁修订**。
6. 首图与候选都倾向于主体贴近下边缘，因此这类「贴边」判定的稳定性本身偏弱——也是分歧反复出现的土壤。

## 七、最小收费复核方案（等用户确认后再执行，本轮不调用）

| # | 要解决的问题 | 建议调用 | 每次输入 | 为什么旧证据不足 | 次数 |
|---|---|---|---|---|---|
| 1 | 参考颜色泄漏到底有没有 | 配色保留评审（真正附参考图） | 每支一次：对应首图 + 本支两张候选全图 + **画风参考图** + 候选局部裁剪（标注不用于画幅） | 旧评审只给了参考图路径，没有附图片，泄漏结论不可判定 | **2**（A 一次、B 一次） |
| 2 | B 的终审 fail 是否只是「画风不够通透」、是否值得修订 | 终审复核（现有 final_review 审计，同一判据） | B 两张：首图 + 候选 + 参考图 + 本次契约说明 | 旧终审的两张候选**各有一次成功终审**，指纹不全但结论可读；若只想知道「质量差异有多大」，可用一次成对复核代替逐张 | **1 次成对**（两张候选一次）或 **0**（沿用旧终审结论，仅补指纹说明） |
| 3 | A 的 quality `needs_refine` 是否仍是当前判据下的结论 | 质量复审（现有 quality 审计） | A 两张 + 首图 + 参考图 | 同 2：旧结论可读但指纹不全 | **1 次成对** 或 **0** |

- **最少方案**：只做 #1（**2 次**文本调用，约 `2 × 0.0046 ≈ 0.009 USD`，按多图输入口径可能更高，以真实账单为准）。
- **推荐方案**：#1 + #2 成对复核（**3 次**文本调用，估算 `≈ 0.014 USD`）。
- **仍然不做**：不重画、不修订生图、不重跑整批、不换模型、不扩样本。
- 复核清单与输入顺序已在本目录 `recheck-request-preview.json` 里备好；确认后按它执行即可，
  并按新指纹规范记录（这样这批复核结论**以后是可复用的**）。
