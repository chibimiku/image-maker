# 无画风参考图的固定配色 · 跨主题执行验证（cross-theme-v1）

日期：2026-10-05。目录：`data/test-result/20261005/color-palette-cross-theme-v1/`。
用途：**指定颜色组合与区域落点**，不改善画质。只判配色、落点、保护色与内容，不评画质美感。

> 2026-10-05 Codex 离线修正：已目视查看全部 16 张实际产图，并按原始逐组 JSON 重算配色与落点计数。本文修正报告解释，不改原始评审响应、JSON、产图或冻结协议。原 `cross-summary.json` / `cross-per-image.json` / `cross-review.html` 仍展示原评审口径；其中漏判与混合字段以本文的修正说明为补充，不冒充模型的新评价。

## 一、合同与接线设计

合同（结构化）在 `prompts/color-knowledge/cross-theme-palette-v1.json` 的 `contract` 段，
说明文档 `prompts/color-knowledge/cross-theme-contract-v1.md`，`contract_hash = d7765328c5496114…`（随冻结记录保存）。

| 合同字段 | 内容 |
|---|---|
| **主色** | 每组方案一个（P1 冷蓝 `#4A6E8A` / P2 深绿 `#2F4F3A` / P3 蓝紫 `#4B4A7A`），声明为环境主导色 |
| **点缀色** | P1 红 `#B23A3A` / P2 玫红 `#B7417A` / P3 金 `#C9A227`，声明为小面积高彩度色且「不得替换成近似色」 |
| **允许中性色** | 近白（受保护服装本体色）、低彩度蓝灰（主色内的明度层次）、低彩度暖灰（主题既有的石面）、深中性色（受保护鞋体色） |
| **区域落点** | 三处**既有**区域：`waist_band_region`、`left_shoe_bow_region`、`right_shoe_bow_region`；环境主色映射到各主题既有的 `environment_dominant_region` |
| **保护色** | 发色、肤色、服装本体色、鞋体色 |
| **内容保护** | 服装设计、单人、全身双鞋与单幅完整构图；原合同和评审字段把这些与保护色合并，本报告分开解释，不修改冻结合同 |
| **授权改色区域** | 只有上面三处落点；每个主题给出 `protected_regions` 供机器核对重叠 |
| **版本与 hash** | `contract_id` + `version` + 规范化 JSON 的 `contract_hash` |

**落点映射（不靠新增物件满足点缀）**：两个主题都把三处落点映射到**已经存在**的区域——
石橋主题映射到「象牙白大衣前方的宽束带 + 左脚鞋既有蝴蝶结 + 右脚鞋既有蝴蝶结」，
石階主题映射到「象牙白连衣裙前方的宽束带 + 双脚既有蝴蝶结」；环境主色分别映射到
「天空/運河水面/石墙/阴影路面」与「天空/远山/梯田/石階阴影」。映射是 `{maps_to, describes}` 结构，
`maps_to` 必须是该主题 `existing_regions` 里的键，**校验不通过就拒绝运行**。
绑定条款里还写死一句 `Do not create any new region or object to satisfy the colour plan.`

**首版范围与限制**：只用于 Gemini 无参考图直出（`image_paths` 为空）；**有有效画风参考图时禁止选择或注入配色**；
未启用配色时原请求逐字不变且不额外调用模型；只扩展实验工具与测试，**没有接生产 GUI**。

## 二、两个新主题与选择理由

| 主题 | 内容 | 为什么选它 |
|---|---|---|
| `canal-town`（白日運河小鎮·石橋水巷） | 少女站在石桥上，象牙白长大衣 + 宽束带 + 深色厚底鞋 | 与旧钓鱼场景不同：环境由「天空 + 運河水面 + 石砌建筑 + 阴影」大面积主导，主色有清晰承载面；点缀仍落在既有束带与两只鞋蝴蝶结上，不需要新增任何物件 |
| `mountain-steps`（白日山道石階·梯田遠山） | 少女走在石階山道，象牙白长裙 + 宽束带 + 深色厚底鞋 | 与旧场景不同，且**承载面比水景更分散**（天空 + 远山 + 梯田 + 石階阴影），正好检验主色条款是不是只对水景有效 |

两个主题都是**白日、单人、全身、点缀三处既有**，同主题内 subject / rendering / 模型 / 分辨率 / 比例逐字相同，
只有 COLOUR PLAN 段不同（预检已断言「同主题内 subject/rendering 前缀一致」）。

## 三、执行与调用

| 项 | 值 |
|---|---|
| 图片请求 | **16 / 16 全部成功**（3 方案 × 2 主题 × 2 次 + 无配色对照 × 2 主题 × 2 次） |
| 文本评审 | **8 / 8 全部成功**（每组两张一次调用），无失败、无重试 |
| 自动重试 | **0**（`IMAGE_MAKER_IMAGE_MAX_RETRIES=0`，预检与执行共用同一开关） |
| 产图 | 全部 2816×1536 |
| 冻结 | `freeze_hash = 4cd45c9380fae7e0…`（含合同 hash、条款 hash、主题文本 hash、保护块 hash、逐条请求 hash） |
| 账本 | `cross-ledger.json` / `send-budget/`：图片 16 预留 16 成功 0 失败；文本 8 预留 8 成功 |
| 失败/拒绝 | 无。分母没有缩小 |

**一处必须如实说明的偏差**：这条链路真正发送的模型是 `generate_image_aigc2d` 的默认值
**`gemini-3.1-flash-image-preview`**（服务端 `modelVersion` 也回报同名），**不是** repaint 固件里的
`gemini-3-pro-image-preview`。也就是说我最初的预检把模型报成了 Pro，那是错的（预检读的是配置/固件的模型字段，
而这条文本→图片链路不读配置里的 `model`）。已修正预检：现在按**实际调用路径**记录，并把
`model_source` / `repaint_firmware_model` / `sent_model_note` 一起落盘。
**本轮 16 张全部同一模型同一参数，所以主题内与主题间的比较仍然成立**；但「Flash 的结果」不能当成「Pro 的结果」。

## 四、逐组结果（8 组）

| 组 | 类型 | 主色 | 点缀与三落点 | 原混合保护字段 | 配色判定 / 内容观察 |
|---|---|---|---|---|---|
| canal-town · P1-knowledge | 配色 | conform / conform | conform / conform（束带 + 左右鞋全 present） | true / true | **2/2 conform** |
| canal-town · P2-simple | 配色 | conform / conform | conform / conform（三落点全 present） | true / true | **2/2 conform** |
| canal-town · P3-knowledge | 配色 | conform / conform | conform / **partial**（Q02 左鞋落点 absent） | true / **false** | 1 conform + 1 partial；Q02 双人 |
| mountain-steps · P1-knowledge | 配色 | conform / conform | conform / conform | true / true | **2/2 conform** |
| mountain-steps · P2-simple | 配色 | conform / **partial** | conform / conform | true / true | 1 conform + 1 partial |
| mountain-steps · P3-knowledge | 配色 | conform / conform | conform / **partial** | true / **false** | 1 conform + 1 partial；Q02 分屏且左侧人物缺头 |
| canal-town · B0（无配色） | 对照 | 不适用（原模型记 unverifiable） | 不适用 | **true**（原模型值） | Q01 为全身图加服装/鞋部放大的拼接，原评审漏记；未见明确固有色违规 |
| mountain-steps · B0（无配色） | 对照 | 不适用（原模型记 unverifiable） | 不适用 | **true**（原模型值） | 未见明确固有色违规；Q02 新增木质栏杆 |

**按指标汇总（12 张配色图）**：

- **环境主色**：`conform 11 / partial 1 / fail 0`（唯一的 partial 是 mountain-steps·P2-simple 第 2 次，
  梯田与部分地面出现偏亮偏黄的绿色，没有完全维持「深绿 + 低彩度」）。
- **点缀色与三处落点**：按原始 JSON，**11/12 张的三个字段均为 present**；canal-town·P3-r2 的 `left_shoe_bow_region` 为 `absent`。这是模型字段计数，不是 11 张均满足单人区域绑定：双人或分屏图不能因为局部颜色存在就算完整落点成功。
  点缀色 `conform 10 / partial 2`（两次 partial 都是 P3·knowledge 的第二张）。
- **保护色与内容分开**：原字段 `protected_colours_ok` 为 `true 10 / false 2`，但它同时检查颜色、服装、人数和全身构图，不能解释为“两张保护色失败”。目视及原评价未指出明确的发色、肤色、服装本体色或鞋体色违规；P3 两张异常为明确内容违约，缺失或无法对应的部位不能算颜色验证通过。
- **总体配色判定**：原逐张 `colour_status` 为 **conform 9 / partial 3 / fail 0**。其中 P3 山道分屏图的 partial 混入了内容/区域对应问题，不应被解释为环境主色失败。
- **本轮候选计数**：原配色 conform 且混合保护字段 true 为 **9/12**（P1 4/4、P2 3/4、P3 2/4）。这里只是实验筛选，不是生产质量或身份门禁通过率。
- **无配色对照 4 张**：配色要求为**不适用**，保留原模型 `unverifiable` 作为历史字段，不计入配色成功率或证据不足。未见明确固有色违规，但 **canal-town·B0-r1 是全身图加服装/鞋部特写的拼接**，原评审未记录此构图偏差；**mountain-steps·B0-r2 新增了木质栏杆**。两项均为内容层面的偏差。

**失败区域（值得处理的具体问题）**：

1. **`P3-knowledge` 在主题上都出现了内容违约**：canal-town 第 2 次画出**两个人**（我目视核对确认：
   画面左右各一名同款少女，左侧那位鞋蝴蝶结是浅粉灰、不是金色）；mountain-steps 第 2 次变成**左右分割双画面，左侧人物头部缺失**。
   这是明确内容违约，不是主色失败——两张的蓝紫主色都 `conform`。原评审未单独列出缺头，此项为 Codex 目视补记。
2. **`P2-simple` 在 mountain-steps 上有 1/2 主色 partial**：深绿基调被梯田区域的亮黄绿削弱。
3. **无配色对照在 mountain-steps 上新增了木质栏杆**：说明「不加物件」这条约束在这个主题上对**空配色方案**也不稳。
4. **canal-town·B0-r1 的拼接漏判**：实际图右侧有独立放大的服装/鞋部视图，不是背景中的第二个人物。原模型给混合保护字段 true 且 `content_changes` 为空，不能据此宣称单幅构图符合。对照也会拼接，因此本轮不能把拼接归因于 P3 色板或知识版条款。

**注意**：同方案同主题两次重复只有 2 个样本，**只能作为探索证据**，不能据此宣称方案稳定或不可用。

## 五、离线回归（全部 mock + tmp_path，不联网、不写交付目录）

`tests/test_color_cross_theme.py`（**16 条全通过**）：

| 用户要求 | 对应用例 |
|---|---|
| 实际发送的 prompt 含合同 | `test_sent_prompt_contains_contract_and_region_binding`、`test_mock_capture_shows_prompt_actually_sent`（用假生图函数抓真实传参，断言 16 条正文都含 `PRESERVE:`、4 条对照不含 `COLOUR PLAN:`） |
| 保护色冲突拦截 | `test_conflict_between_protected_and_authorized_stops_the_run`、`test_conflict_protected_overlap_blocks_execution`（断言冲突时 `stopped = preflight_problems` 且**零请求发出**） |
| 合同变化使相关缓存失效 | `test_contract_or_clause_change_invalidates_cache`（条款/合同/主题文本任一项变化 → `invalidated`）、`test_frozen_run_refuses_a_changed_protocol`（保护块变化 → 抛错且零请求） |
| 任务快照可恢复 | `test_snapshot_marks_pending_and_resumes_without_resending`（先全 pending；跑完 16/16；**再跑一次零调用**） |
| 有参考图时不注入配色 | `test_colour_injection_is_refused_when_a_style_reference_exists`（有效参考图 / 渲染条款 / 请求里已挂参考图，三种都拒绝） |
| 未启用时原请求不变 | `test_disabled_colour_plan_keeps_the_original_request_byte_identical`（逐字节比对 + `extra_calls = 0`） |

另含：区域映射必须落在既有区域且是 `{maps_to, describes}` 结构、16 条请求矩阵、对照正文不得出现 COLOUR PLAN、
评审校验必须逐项给出三个落点且**不得输出美学评分**。

**DeepSeek 报告的全量 pytest 现状**：`930 passed, 5 skipped, 19 subtests passed, 7 failed`。
那 7 条失败全部在 `tests/test_save_to_source_dir.py` 与 `tests/test_single_analyzer_history_rerun.py`，
DeepSeek 将原因归于 `modules/image_analysis/single_analyzer.py` 里**另一个任务（衣装功能）**刚加的 `wardrobe` 关键字参数
与 `generation_wardrobe` 字段（该文件 mtime `2026-10-05 00:18:50`），是它们自己的 mock 还没跟上新签名，
并报告本轮没有改过那个文件。此次离线报告修正未独立复跑全量测试或核实失败根因，不把上述归因当成新增验证结论。
DeepSeek 报告本轮的配色模块（`test_color_cross_theme.py` 16 条 + 既有 7 个配色模块）**103 条全部通过**。

## 六、适用限制

1. **模型是 Flash 不是 Pro**（见第三节），这一轮的结论只对这条链路成立。
2. 两个主题各两次重复，**只是探索证据**，不能宣称普遍稳定。
3. 本轮**完全没有画风参考图**；带参考图时禁止注入配色是**校验函数**保证的，不是本轮的实验结论。
4. `hex` 只是设计师近似值，不是测量阈值；验收看区域与色相关系。
5. **无配色对照不看配色条款**（没有指定颜色），内容仍需核对；原评审已漏记一张拼接，不能把模型“未见变化”直接当成完整内容通过。
6. 只验证首图直出，不涉及重绘、身份修订或工序保留；**没有接生产 GUI，生产配色选择仍仅限没有画风参考图的场景**。

## 七、交付物

- 合同与协议：`prompts/color-knowledge/cross-theme-contract-v1.md`、`cross-theme-palette-v1.json`、
  评审提示词 `cross-theme-colour-review-system.md`
- 复核与冻结：`cross-preflight.json`（含合同 hash、冲突检查、实际解析出的模型与参数）、`cross-frozen.json`。旧交付列表所列 `cross-plan.json` 不在本目录，不作为已交付文件。
- 逐条请求与产物：`samples/<主题>/<方案>-r<次>/{request.json,result.json,*.jpg}`
- 逐张与汇总：`cross-per-image.json`、`cross-summary.json`、`cross-snapshot.json`、`cross-ledger.json`
- 评审：`review/group-<主题>-<方案>.{request.json,raw.txt,json}`
- 离线评价页（内嵌 16 张实际产图，零外链）：`cross-review.html`
- 测试：`tests/test_color_cross_theme.py`

## 八、下一步

先准备 Gemini 无画风参考图直出的最小功能接线，默认关闭，人工选配色，不新增自动选色或评审调用。P1 冷蓝+红作为优先候选，P2 深绿+玫红作为可选候选，P3 蓝紫+金暂列实验方案；小样本不构成可靠性保证。

生产接线之前要解决区域泛化：现有合同只验证了腰带与鞋蝴蝶结，不能强制任意素材增加这些物件。应从既有内容中确定可改色区域；无区域或与固有色冲突时阻止注入并明确提示，不能悄悄改设计。有效画风参考图存在时禁用配色且后端再次检查；未启用时保持原请求不变。合同、区域映射和版本/hash 写入任务快照与历史恢复依据，并验证缓存失效。

先完成入口梳理、接线设计及离线回归，再安排实际入口的小规模验收，不自动重跑整批或修复旧失败图，不关闭任何生产门禁。此次只修正文档，没有修改 GUI 或收费调用。
