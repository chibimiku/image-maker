# Image Maker Agent Guide

> **2026-09-26 当前覆盖说明**：单图分析 GPT GUI 的配方版本为 7，默认使用 gpt-image 通道：Gemini 第一次重绘接收 GPT 首图和完整画风图（`reference_mode=style`、`scope=full`、v5 固件）；重绘提示允许迁移参考图的五官/头发抽象画法，同时锁定身份和固有配色。个别画风可用 `face_hair_refine=true` 再做一次仅五官与头发的画法修订，之后仍走身份审计与最多两轮定点修订；`skip_quality_refine=true` 可保留第一次完整画风重绘。画风条目还可用 `generation_clauses`（只进 GPT 首图）、`identity_correction_clauses`（只进身份修订）和 `post_adjustment`（模型闭环后的确定性色偏/曝光/粗结构线收尾）。修订阶段不再发送画风图，且显式锁定当前图宽高比。日期根目录只发布最终选中图和分析投稿文件；首图、重绘、审计与修订过程放在 `analysis-gpt-image/<单次任务>/`。最终文件名把 8 位 task hash 保留在第一个下划线字段，供 `publish_server.py` 关联投稿 JSON；发布器启动时也会修复旧式文件名的空关联。额外本地工序默认关闭，高级选项默认折叠。画风条目的 `enabled=false` 会保留在管理界面，但从生图与测试列表隐藏。实测见 `docs/gpt-image-tid-style/REFINE-QUALITY-20260925.md`、`STYLE-SWEEP-20260925.md`、`STYLE-FACE-REFINE-20260926.md` 与 `STYLE-SECOND-REFINE-20260926.md`；不能宣称已彻底解决断线或所有身份偏离。
> **2026-09-25 下午补充**：GPT 首图的内容默认改为 `gpt_image_prompt`（约 1400 字符身份完整锚）；全文会压弱画风图，500 字符短锚可能漏发色/瞳色。画风图再次送 Gemini 即使配“忽略配色”文字仍会泄露角色颜色，因此只用于受控实验。身份审计/修订/重大漂移回退已接入无头实验 CLI，结论见 `docs/gpt-image-tid-style/E2E-GENERALIZATION-20260925.md`。

本文件用于给 Trae/DSH/AI 助手提供项目快速索引。
**每次开始改动前，先读 `PROJECT_REQUIREMENTS.md`（硬性要求）→ 本文档 → 具体代码文件。**

## 1. 项目目录总览（当前结构）

- 主界面入口：`app.py`
- 业务模块目录：`modules/`
- 工具与运行时目录：`utils/`
- 无头 CLI 工具目录：`tools/`
- 配置目录：`conf/`
- 模型目录：`models/`

### 根目录核心文件（只允许这些）

- `app.py`: 主窗口、Tab 组织、跨模块联动（唯一主入口）
- `make-pic.py`: 「赛博暖暖」生图独立入口（`modules/fashion_collection/make_pic_bridge.py` 会写状态并子进程拉起它）
- `sd-make-pic.py`: SD 工作流单窗入口
- `publish_server.py`: 图片发布 Server（Flask+拖拽 GUI，默认端口 18765，API 文档 `docs/publish_server_api.md`）
- `PROJECT_REQUIREMENTS.md`: 项目硬性要求（必读）
- `AGENTS.md` / `readme.md` / `docs/`

> 根目录已禁止堆放脚本：历史 `sd_gen_*.py`（30 个）→ `useless/scripts/sd_gen/`；
> `fashion-generate.py` / `fashion_pipeline.py` / `analyze_fashion.py` / `test*.py` →
> `useless/scripts/`；`doujin_translator.py` / `translate_booru_tags.py` / `web-probe.py` → `tools/`。

## 2. 模块目录索引

### `modules/image_analysis/`（图片分析）

- `single_analyzer.py`: 单图分析（Step 1~5 模块级函数 + Qt WorkerThread）。**改这个 Tab 前先看这三条契约**：
  - **生图通道两条链路**：`gen_channel_gemini`（老逻辑：`--ar + 画风完整指令 + 分析描述` → `generate_image_aigc2d`）与
    `gen_channel_gpt`（`utils.analysis_gen.build_first_pass_request` 组装 → `generate_image_aigc2d_gpt` → `run_gpt_image_pipeline` 跑工序）。
    两者**共用同一份分析产物**，只在「分析完成之后」分叉；完整对照表见 `docs/gpt-image-tid-style/BEST-PIPELINE.md` §八 末尾。
  - **队列状态机**：绿色 `[已完成]` == 该任务**所有工序**（分析 + 生图 + 后处理）都跑完。
    收尾点是 `_on_image_thread_stopped`（生图线程退出，补调 `_finalize_task_pipeline`）与
    `_cleanup_post_thread`（后处理线程退出）；分析线程退出时的「兜底更新」必须在
    `_pipeline_pending_for_hash()` 非空时跳过，否则会抢跑标绿、还把标题写成「已完成（兜底更新）」。
    生图无产物（失败/取消）→ 记录标红 `error`（`pipeline_error`），不再停在「等待最终产物」。
    回归用例在 `tests/test_analysis_channel.py`（`test_queue_*` 一族）。
  - **布局契约（窗口默认 1100x750，别再加常驻行）**：固定区 = 图片预览（拖拽落点）+ ② 生成按钮 + 队列 + 日志；
    选项区（① 按钮 / 自动生图 / 画风 / 生图通道 / gpt 工序与参数…）在 `controls_scroll` 里滚动，且滚动容器
    **不接受拖拽**（否则拖图落点会变）。gpt 通道的选项**只有两行**（`gpt_pp_row` 工序 + `gpt_param_row` 参数）——
    加行会把队列/日志挤没，`test_gpt_option_rows_are_compact_and_log_stays_visible` 会拦住。
    **拉伸方向**：`controls_scroll` stretch=0（按内容取高）+ `bottom_panel` stretch=1 + 选项区末尾一个
    `addStretch(1)` —— 窗口拉高/最大化时多出来的高度归队列与日志，**不能**分给每一行（否则按钮之间被撑出大片空白，
    用户 2026-09-24 反馈「最大化之后界面不正常」）。回归用例：`test_window_growth_goes_to_queue_and_log_not_option_rows`。
- `analysis_pipeline.py`: **无头全链路分析**（Step 1~5 编排 + 投稿格式落地 `save_result_to_source`，CLI 见 `tools/analyze_fashion.py`；`analyze_image_step1` 供批量出图后单步分析）
- `batch_analyzer.py`: 批量分析
- `style_analyzer.py`: 多图画风提取；最终审查后额外生成多用途 prompt 包（Gemini 完整/重绘、五官头发、GPT-image 680 字符八字段、不同身份保持与构图场景、负面规则、证据摘要），输出 JSON 的 `prompt_variants`
- `json_dataset_tab.py`: JSON 数据集导出
- `pic_cate_tab.py`: 图片分类切分

### `modules/fashion_collection/`（服饰采集与生图）

- `collector_tab.py`: GUI 采集/批量生图/自动分析 Tab（app.py 的「服饰素材采集」）
- `collection_service.py`: 采集服务（`collect_bundle` / `load_bundle_from_dir` 复用已有素材 / LLM 颜色协调）
- `generation_plan.py`: prompt 组装（`resolve_style_bundle` / `build_reference_prompt` / `generate_random_composition` 随机构图 / `generate_llm_composition` LLM 模板构图 / `build_character_consistency_sections` 角色一致性注入）
- `models.py` / `theme_profiles.py` / `brand_scraper.py` / `lolibrary_adapter.py` / `wear_adapter.py` / `mayla_adapter.py` / `make_pic_bridge.py` / `networking.py`

### `modules/image_generation/`（图片生成与编辑）

- `prompt_generator.py`: 批量提示词与生图
- `image_edit.py`: 批量图片编辑
- `char_design.py`: 角色设计生成
- `single_gen_debug_tab.py`: 单图调试生图
- `webp_compressor.py`: PNG/WebP 压缩
- `upscaler_tab.py`: 图片 Upscaler
- `flux2_client_tab.py`: WebUI Img2Img 客户端
- `diff_cg_tab.py`: 差分 CG 生成
- `gpt_image2_tab.py`: gpt-image-2 专用 Tab（「站点」下拉切 `new.aigc2d`=`apis.aigc-2d-gpt` / `autodl`=`apis.autodl`，「模式」下拉切生图/编辑/**重绘**；独立于全局 API 类型；aigc2d 尺寸仅 3 档、autodl 含 auto/1792x1024；两站点都支持参考图垫图与编辑）
  - **产物优化（重绘提线）**：模式 `重绘(Gemini优化产物)` 用 `gemini-3-pro-image-preview` 对 gpt-image 产物做整体重绘——修手、连通发丝、保住蕾丝/褶皱结构；**生图与编辑模式都可勾选「出图后立即重绘优化（以重绘结果作为产物）」**（gpt-image 两种模式出的分辨率都偏低，重绘顺带提分辨率）；结果列表旁有「重绘结果列表的图」「把结果加入参考图」（迭代重绘）
    - **输出宽高比默认 `auto`**：不下发 `aspectRatio`，由模型匹配输入图尺寸（官方默认行为）；写死比例会让非该比例的源图被拉伸/裁切。只有确实要强制时才选具体值（`1:1/3:2/2:3/3:4/4:3/4:5/5:4/9:16/16:9/21:9`）
    - 固件 prompt：`prompts/gpt-image-optimize/`（`repaint-system.md` 主 prompt + `repaint-detail-suffix.md` 细节后缀 + `config.json` 模型/分辨率/比例/次数/输出目录）
    - 运行时：`utils/gpt_image_optimize.py`（装载/拼接/落盘参数）、`modules/others/api_backend.generate_image_repaint`（逐张调 Gemini 通道，内部 `face_quality_boost=False`）
    - 理论与全部实验数据：`docs/gpt-image-optimize/`（含指标口径与已知偏差、四个批次实验）
    - CLI 同能力：`python tools/gpt_image2_gen.py --repaint --image <产物.png>`（`--repaint-show-prompt` 只看固件）
  - 旧的独立 Tab `autodl_image_edit_tab.py` 已合并进本 Tab，文件归档在 `useless/junk/autodl_image_edit_tab.py`
  - 「模型」是可编辑下拉：按站点列 gpt-image 家族（aigc2d 含 `gpt-image-2.5-flare/sunburst` 及 `-c` 计费版，autodl 只有 `gpt-image-2`），也能手输；「刷新模型列表」按 `GET {base}/v1/models` 现拉（`ModelListWorker` + `list_available_models`），刷新结果按站点各自缓存
- `ref_image_grid.py`: 参考图缩略图网格控件（`RefImageGrid` / `RefImageCard`）：缩略图卡片代替裸路径，卡片右上角 × 删除、拖动缩略图重排（顺序即 `image[]` 提交顺序）、单击预览、双击打开所在目录、支持目录拖入；`collect_image_paths` / `load_thumbnail` 可被其它 Tab 复用
  - `compact_when_empty=True`（gpt-image-2 Tab 在用）：没有图片时网格只占一行占位提示的高度（约 44px），有缩略图才恢复到 `thumb_size+76`；默认 `False`，其它 Tab 行为不变。**该 Tab 上方控件多**：网格一旦无条件占 172px，Qt 就会把可拉伸的「提示词」编辑框压到最小高度（这就是输入框看起来过小的原因）
  - 同 Tab 的三条 UI 约定（**改 UI 前必看**）：
    1. **功能开关常驻可见，只有参数可折叠**——「出图后立即重绘优化（以重绘结果作为产物）」勾选框在顶层，`▸ 重绘参数（模型 / 分辨率 / 比例 / 次数）` 是可折叠区（默认收起）。把开关塞进折叠区会让用户以为"只给了文字没有功能"。
       - **按模式启停**：勾选框在**生图与编辑模式都可用**（gpt-image 两种模式出的图分辨率都偏低，都要能挂重绘）；只有**重绘模式**下它没有意义（恒定走重绘）→ 置灰但仍可见。重绘参数在三种模式下始终可用。
    2. **界面控件已表达的状态不要再写一遍文字**——`repaint_notice` 只在三种情况出现：提示词装载失败 / 重绘 API key 未配置 / 输出比例被强制成非 auto。改提示文案请保持这条，别加回"当前：……"式复述。
       - **重绘模式会把固件正文填进提示词框**（`on_mode_changed`）：用户必须看得到实际在用的重绘提示词，否则他自己写一句描述发出去，固件里的修手/保蕾丝约束就没了、也体现不出修复效果。离开重绘模式会还原生图/编辑那边的原文；清空则后端回退用固件原文。
    3. 提示词编辑框最小高度 140（它是布局里唯一可拉伸的控件，下限太小就会被压成一条）；`RefImageGrid(compact_when_empty=True)` 空图时只占一行占位。
    - 回归用例：`tests/test_gpt_image2_api.py` 的 `test_repaint_switch_is_visible_without_expanding`、`test_repaint_checkbox_enabled_in_generate_and_edit_mode`、`test_edit_mode_with_repaint_chains_repaint`、`test_repaint_options_always_usable`、`test_repaint_notice_only_shows_when_action_needed`、`test_reference_grid_is_compact_when_empty`、`test_repaint_mode_fills_firmware_into_prompt_box_and_restores_on_leave`、`test_repaint_mode_allows_empty_prompt_meaning_use_firmware`
  - **画风选择（2026-09-22 新增）**：Tab 顶部「画风」下拉（首项 `默认(无附加)`）把所选画风的 `prompt_gpt` 拼进提示词，并把画风的 `ref_image` 追加到参考图列表**最后一张**（用户内容图在前），同时在提示词里写明 IMAGE ROLES 分工（`utils/styles.py` 的 `compose_style_prompt` / `ordered_reference_images`）。画风信息行显示字符数/来源/参考图，缺 `prompt_gpt` 会橙色提醒。回归用例在 `tests/test_gpt_image2_api.py`（`test_style_*`）
  - **后处理流水线（勾选框）**：出图后可选「重绘提线 / 结构线叠加 / 局部重绘+羽化贴回（区域含头发/脸部/头部/裙子/上半身/整个人物/人物不含面部/整张）」，实现在 `utils/post_process.py`；见 `BEST-PIPELINE.md`
  - **不要再加「重绘用双参考（源图+线锚图）」勾选框**（2026-09-24 删的死开关）：分析 Tab 的 `gpt_pp_dual` 连布局都没进、且 `_build_gpt_image_steps()` 从不读它；本 Tab 的 `post_dual_check` 也从不进 `post_pipeline_steps()`（repaint 步骤恒 disabled）。重绘第二张参考**恒为画风图**（`repaint_ref_mode="style"`），线锚图只留给无头 CLI `--repaint-ref line_anchor|both`（§三十一 实测会塌画面）。后端旧键 `dual_reference` 只在 `reference_mode` 缺失时兜底，见 `post_process.default_pipeline()` 注释。
- `z_image_edit_tab.py`: z-image 编辑（代码保留，默认不在主 UI Tab 显示）
- `conf/config.json`: 图片生成相关配置模板（模块内）

### `modules/others/`（通用与辅助能力）

- `api_backend.py`: 各类 API 后端封装（多模块共享）
- `tag_completer.py`: 标签补全（SD 生图相关）
- `booru_tag_generator.py`: booru tag 生成器

## 3. 工具与模型目录索引

### `utils/`（工具与运行时）

- `styles.py`: 画风预设统一解析与「艺术风格参考图」模式组装（config-styles.json 新格式）
- `style_ref_widget.py`: 跨 Tab 共享的「风格参考模式」下拉控件 + 全局配置持久化
- `wd14_tagger.py`: WD14 打标逻辑
- `booru_tags.py`: booru tags 处理
- `pic_cate.py`: 分类切分逻辑
- `task_runtime.py`: 任务运行时工具
- `image_upscale_runtime.py`: Upscale 运行时
- `webui_img2img_client.py`: WebUI Img2Img API 客户端
- `upscaler_arch.py`: Upscaler 架构定义
- `upscaler_arch_match.py`: Upscaler 架构匹配
- `upscaler_real_cugan.py`: Real-CUGAN 相关实现
- `upscaler/`: Upscaler 子包（`core.py`、`webui_provider.py`、`extras_pipeline.py` 等）
- `gpt_image_optimize.py`: **gpt-image 产物优化（Gemini 重绘提线）运行时**：`load_config` / `save_config`（读写 `prompts/gpt-image-optimize/config.json`）、`build_repaint_prompt`（主 prompt + 细节后缀拼接）、`plan_output`（按源文件名派生输出前缀与子目录）；被 GUI Tab、CLI、后端共用。**默认固件自 2026-09-22 起是 `repaint-system-conservative-v5.md`**（= 保守修复 + 色彩匹配 + 线条连通 + 禁止凭先验补全不可见肢体 + 眼部保留式修复 + 主结构/次级笔触分级）；`-v4.md` 保留为"最大清线力度"备选，旧 `repaint-system.md`（写死 cel shading）只在明确要 cel 成稿时用。要改重绘 prompt 先看 `docs/gpt-image-tid-style/README.md` §4.14/§4.17
- `analysis_gpt_prompt.py`: **分析产物的 gpt-image 专用短提示词字段**（两档）：`gpt_image_prompt`（≤1400，内容优先）与 `gpt_image_prompt_short`（≤500，**要挂画风参考图时用**——文本越短参考图越能起作用）。含 `call_text_model` / `load_text_api_config`、`build_gpt_image_prompt(..., tier="full"|"short")`、`compose_gpt_image_prompt(style, field, style_ref_exclusion=True)`（挂参考图时自动追加"只借色调/渲染、不得搬参考图角色/发色/瞳色/服装/姿势/构图"的排除句）、`resolve_content_field`、`composed_length_warning`（>2500 字符告警）。为什么单独出字段：给 Gemini 用的 `--ar + 11k 画风 + 长描述` 约 15k 字符，gpt-image 通道会断连、且画风参考图会被长文本压住失效（见 `docs/gpt-image-tid-style/README.md` §4.16/§4.18/§4.19）。开关 `conf/config.json` 的 `enable_gpt_image_prompt_single` / `enable_gpt_image_prompt_short_single`（默认都 True），落盘为 `-gpt-image-prompts.txt` / `-gpt-image-short-prompts.txt`；CLI 见 `tools/build_gpt_image_prompt.py`
- `cost_estimate.py`: **提示词长度护栏 + 单张成本估算**（GUI 常驻显示，避免用户写出会爆的提示词）：`prompt_budget` / `format_budget`（软上限 2000 字符 = 画风参考图失效线，硬上限 15000 = 中转站断连线，官网 32000）、`estimate_gpt_image_cost`（token 计费按 new-api 公式：`(文本token + 输入图token×image_ratio)×model_ratio + 输出token×completion_ratio×model_ratio`，×分组倍率 ÷500000）、`estimate_gemini_repaint_cost`（按次 `model_price × 分组倍率`）、`estimate_pipeline`（首图 + 重绘 + 结构线(0) + 局部重绘 + 可选分析链路）、`fetch_pricing`（拉 `new.aigc2d.com/api/pricing`，缓存 `cache/pricing_cache.json`，12h TTL，失败回退内置快照）。实测路由分组：gpt-image/文本 → `Openai-Gpt-1`(×0.88236)，Gemini 图片 → `Discounted-Banana-1`(×0.110295)（响应头 `X-Routing-Group`）。实测价：1024x1536 medium $0.039 / high $0.155、Gemini 重绘 2K $0.036、全工序 ≈$0.13/张
- `analysis_gen.py`: **「分析图片并生图」在 gpt-image 通道上的请求组装 + 工序流水线**：`resolve_content_text`（老路径兜底）、`build_first_pass_request`（**首图请求的唯一组装点**：画风说明 + 内容锚 + 参考图排除句 + 渲染语言条款，GUI 与 `tools/analysis_gpt_run.py` 共用）、`build_gpt_image_request`（底层拼装：**画风 `prompt_gpt` 拼在最前** + 内容描述 + 挂画风参考图时的参考图排除句 + `RENDERING LANGUAGE` 段；**不挂分析素材图**）、`pipeline_steps_from_flags`（支持 `local_regions` 多区域 + `tone` / `ink` 两个**本地**工序）、`run_gpt_image_pipeline`（重绘 → 结构线叠加 → 局部重绘 → 色调校准 → 加墨，重绘传 `use_detail_suffix=False`；**色调校准的目标参考图取画风参考图**）。GUI 接入点：`single_analyzer` 的「生图通道」单选框 + `GptImageGenWorkerThread`；见 `docs/gpt-image-tid-style/BEST-PIPELINE.md`
  - **GUI 默认配方（2026-09-24，`recipe_version=3`；§三十/§三十一 实测）**：重绘 ✓ / 结构线 ✓ / 色调校准（目标=画风参考图）✓ / 线条加墨 ✓ / **重绘范围 = `lines_only`**（`repaint.scope`）/ **局部重绘默认关闭**。
    - **重绘范围（`post_process.REPAINT_SCOPE_CLAUSES`）**：`full` / `person_only` / `person_noface` / `details` / `lines_only` —— **不裁切、不贴回**，只在一整张新图上用提示词要求模型保留不该动的部分；每档后面都追加 `IDENTITY_LOCK_CLAUSE`（源图是身份/设计/内容的唯一真值，不许从画风参考图搬角色/发色/衣服/道具/背景）。实测 `lines_only` 线条最好（tid 段长 74.7→185.0），`person_only`/`details` 背景最稳。
    - **局部重绘（裁切→重绘→贴回）为什么默认关**：模型会在裁切里重新构图，贴回原坐标就是一块错位内容 —— 5 画风 × 4 区域里报废 2 张（§三十，证据页 `DIAG-local-chain.html`）。要开只能显式勾，**优先改用上面的「重绘范围」**。
    - **超时预算 = 每份联网调用 ×（首图 1 + 重绘 1 + 每个局部区域 1）**，本地工序不占份额：默认配方 = 2 份 = 240 秒（旧的四区链是 6 份）。
  - **改完 `.py` 必须重启 app**：GUI 进程里是旧模块，不重启就还是旧提示词（用户 2026-09-24 踩过）。
  - **2026-09-24 修的 bug**：`single_analyzer._start_gpt_image_thread` 以前自己拼首图请求、**漏传 `extra_clauses`**，于是 GUI 首图永远没有 `RENDERING LANGUAGE` 段（CLI 有）→ 实测首图偏白偏灰、线条碎。现在改走 `build_first_pass_request`，`[gpt 通道]` 日志会打印 `渲染条款 N 条（画风自带 / 按 prompt_gpt 派生）`。回归用例：`tests/test_analysis_channel.py` 的 `test_gpt_first_pass_includes_derived_render_clauses` / `test_gpt_first_pass_prefers_handwritten_clauses`。
  - **当前工序契约（2026-09-24 用户指定，§二十六）**：① 内容锚与 **Gemini 通道同源**（分析产物里描述素材特征/构图/服装/道具的文本，`build_gpt_image_request(content_text=...)`），**已删除「重新构图」开关**与 `RECOMPOSE_CLAUSE`/`composition`/`content_tier`；② 首图走**「新建图片」**：`generate_image_aigc2d_gpt(mode="generate")` → `POST /v1/images/generations`，画风参考图放 JSON 的 `image` 字段（base64），不再走 `/images/edits`（edits 会把参考图当原图编辑）；③ 尺寸跟随输入比例；④ 局部重绘统一 **2K**（`detail_boost=False`，细节区不再升 4K）；⑤ **超时按工序计**：`timeout_budget = 每道工序秒数 × (首图 1 + 工序数)`，不再用 120s 掐整条链。仍未处理：无（2026-09-24 已修：GUI 重绘第二张参考改为**画风图**，`pipeline_steps_from_flags(repaint_ref_mode="style")` 默认，`single_analyzer` 把画风图 + `repaint_clauses` 传给 `run_pipeline`；线锚图要显式 `line_anchor`）。
- `output_isolation.py`: **测试产出隔离**：`resolve_output_target(save_dir, base_filename)` 在 `IMAGE_MAKER_TEST_OUTPUT=1` 时把 `data/<日期>/…` 改道到 `data/test-result/<日期>/…` 并加 `test-` 前缀（未设该环境变量时行为与改动前一致）；`find_legacy_test_artifacts` / `relocate_legacy_test_artifacts` 负责搬迁历史污染（识别范围含分析落盘的 `…-hash_0-title_0.json` 一族与流水线产物 `img-233500-a35670-final-sline50.png` 这类，见 `PIPELINE_TEST_ARTIFACT_RE`），命令行入口 `python -m utils.output_isolation [--apply]`。接入点：`single_analyzer.on_process_finished`、`batch_analyzer.save_result`、`analysis_pipeline.save_result_to_source`、**`post_process.run_pipeline`（`final_output_path()`，2026-09-23 加，之前它不传 final_dir 时会往真实日期目录扔废文件）**；`tests/conftest.py` 自动开启并兜底搬迁

### `tools/`（无头 CLI 工具）

- `compress_styles.py`: 批量调用 LLM 把 config-styles.json 中每个画风的完整指令压缩为 `prompt_compressed` 并固化（支持 `--only <样式名>` 单独处理）
- `convert_styles_gpt.py`: **把每个画风转成 gpt-image 通道用的短版字段式说明 `prompt_gpt` 并固化**（读 `prompt` 全文 + 参考图客观统计 → 文本模型 → 8 字段校验/重试 → 写回 `conf/config-styles.json`）。`--force`（全量重转）/ `--only`（指定画风）/ `--check`（只校验）/ `--dry-run` / `--no-image`。规则见 `docs/gpt-image-tid-style/prompt-gpt-rules.md`。**`--check` 同时报告渲染语言条款覆盖**（手写 / 按 `prompt_gpt` 派生 / 无），`--force` 重转一个带手写 `repaint_clauses` 的画风时会提醒条款可能过时
- `style_ref_stats.py`: 用 OpenCV 算画风参考图的渲染统计（亮度/暗部/高光/留白/饱和/色相分布/主色 k-means/边缘密度），供上面的转换器做客观依据；`--style <名>` 或 `--all`
- `make_round_gallery.py`: **把某一轮 `data/<日期>/` 下的所有比较组生成为单文件 HTML 画廊**（图片内嵌 data URI、可离线看，附每张的变体名、token 用量、实际提示词）；`python tools/make_round_gallery.py` → `docs/gpt-image-tid-style/round-20260921.html`
- `sd_batch_gen.py`: **本机 SD-WebUI 批量生图统一引擎**（无 PyQt；读 `conf/config-sd.json` + `prompts/sd-batch/*.txt` 主题文件；`--dry-run` 预检）
- `fashion_batch.py`: **Fashion 批量采集-生图无头 CLI**（`--profile <名称>`，画像在 `prompts/fashion-batch-themes.json`）
- `analyze_fashion.py`: 头图全链路分析 CLI（落投稿格式，功能在 `modules/image_analysis/analysis_pipeline.py`）
- `gpt_image2_gen.py`: **gpt-image-2 生图/编辑无头 CLI**（站点 `new.aigc2d` / `autodl` 二选一；支持参考图走 `/images/edits`、`--prompt-file` 批量、`--dry-run`、`--list-sites`（含各站点常用模型）、`--list-models`（实时 GET /v1/models 只列 gpt-image/dall-e 系列）、`--timeout` 临时覆盖；纯 python 无 PyQt，适用于只能跑 python 出网的机器）
  - 产物优化：`--repaint --image <产物.png>` 走 Gemini 重绘提线（同 GUI「重绘」模式），`--repaint-model/--repaint-resolution/--repaint-repeat/--repaint-prompt-file` 可调，`--repaint-show-prompt` 只打印最终固件不调接口
- `build_gpt_image_prompt.py`: **给分析产物生成 gpt-image 专用短提示词字段**：`--json <分析产物.json> [--tier full|short] [--max-chars N] [--from-field short_description] [--show]`，写回 JSON 的 `gpt_image_prompt`（≤1400，内容优先）/ `gpt_image_prompt_short`（≤500，**要挂画风参考图时用**）并另存 `-gpt-image-prompts.txt` / `-gpt-image-short-prompts.txt`（实现见 `utils/analysis_gpt_prompt.py`）
- `overlay_structure_lines.py`: **结构线叠加**（纯本地后处理，不调模型）：抽「长结构边」→ 按局部色调整色 → 低不透明度叠回。`--base <图> [--edge-source <图>] --strength 0.5 --min-len 120 --darken 0.45 --out <图>`。实测三族段长 +6~22%、端点密度最多 −16%，配色/饱和度几乎不动（见 `docs/gpt-image-tid-style/README.md` §4.20）
- `local_repaint_composite.py`: **局部重绘 + 羽化贴回**（近似遮罩式局部重绘，Gemini 无 mask）：`--image <图> --crop 0.0,0.0,1.0,0.62 --feather 48 --scale 1.5 --firmware <重绘固件> --out <图>`；`--patch <已有局部重绘结果>` 可跳过 API 只做合成。实测段长 +4~7%、端点密度下降，可与结构线叠加串联
- `sync_styles_to_submodule.py`: **把本地 `conf/config-styles.json` 的字段（prompt_gpt / prompt_compressed / ref_image / prompt / **repaint_clauses**）同步进子模块版本化文件 `submodules/image-maker-artstyle/config-styles.json`**（本地 conf 是 gitignore 的，只改本地等于没进版本库）；备份留在 `config-styles.json.bak`
- `make_tonight_tests.py`: **生成「本轮五项测试」的单文件 HTML 集合**（图内嵌、含目标达成判定表 + 各画风 prompt_gpt 覆盖表 + 验证产图），输出 `docs/gpt-image-tid-style/tonight-tests.html`
- `doujin_translator.py`: 同人本批量翻译 GUI（独立入口）
- `translate_booru_tags.py`: booru tags 翻译 CLI
- `web-probe.py`: 网页抓取 CLI（`utils/web_probe_cli.py` 的薄壳入口）

### `tests/`（测试与对比脚本）

- `calc_style_similarity.py`: CLIP 向量 + HSV 直方图 + 线条统计的「画风贴近度」量化对比
- `style_render_metrics.py`: **「画面可读性 / 线稿质量」自动指标**（§4.8 新口径）：分区明度层次（P10/P50/P90、局部 RMS 对比、中灰带占比、高光裁切率）、多尺度边缘占比（粗/中/细，**比例匹配**而非越多越好）、线稿连通性（骨架化后的平均段长/长线占比/碎线占比/端点密度）、负空间连通域；输出 `render_score`。**注意：全局亮度/白底占比/边缘密度受构图影响，不能当优化目标，旧的 `fingerprint.json` 口径只作故障报警**
- `test_style_ref.py`: 艺术风格参考图模式下的生图对比实验脚本

### `docs/`（文档与实验记录）

- `fashion-pipeline-strategy.md` / `fashion-theme-spec-template.md` / `publish_server_api.md` / `pyqt6-migration-checklist.md`: 既有文档
- `gpt-image-optimize/`: **gpt-image 产物优化的理论与全部实验数据**（`README.md` 总纲 + `metrics.md` 指标定义与已知偏差 + 四个批次实验 `experiments-A/B/C/D-*.md` + `FINAL-PROMPT.md` 固件快照 + `data/` 指标原始表与各版 prompt 全文 + `tools/` 分析脚本快照）。**要改重绘 prompt 或调指标口径，先读这里。**
- `gpt-image-tid-style/`: **gpt-image 系列上的「画风生图」可行性实验（tid 画风；22 + 12 + 8 次无头 CLI 实测 + 44 张产物 + CLIP/HSV 指标 + 官方文档快照 + 一次 `gpt-5.6-luna` 审阅并逐条复验）**。
  - **要用法先看 `BEST-PROMPT.md`：当前最优 = 结构化风格字段（Palette / Lighting / Brushwork / Edges / Texture / Composition density / Detail level / Avoid，约 250 字）+ tid 参考图 + `medium`**。实测 HSV 0.644、亮度 238.8、饱和 13.3、白底 0.686、边缘密度 0.0364（参考图 221.5 / 25.0 / 0.667 / 0.0663），且不复制参考图角色特征。备选是「角色分工版」（弱一些但不搬主体）。
  - 结论：`prompt` 上限是官方文档的 **32000 字符**，实测 11k~40k 全都能出图（**「前置词太长导致无法生图」不成立**）；长提示词真正的代价是**参考图失效**（真实提示词验证均值：短说明 0.390 / 全量+图 0.247 / 全量纯文本 0.119 的配色贴合度）；主体崩不崩取决于主体描述自身够不够硬——「少女站在浅水」这种稀疏主体在 11k 全量指令下被画成水下精灵，而自带大量细节的洛可可肖像不崩。
  - **`quality` 别默认 `high`**：同 prompt 下 `high` 把白底占比从 0.439 压到 0.222、边缘密度 0.0722→0.0934，细节变多留白变少，水彩感反而变差；默认 `medium`、`output_format=png`、`background` 保持不透明。
  - gpt-image-2 的 `input_fidelity` 不可调、输入图一律高保真，所以它**把参考图当"要编辑的原图"**（会连参考图角色的发色/瞳色一起搬），必须显式约束；不带参考图时文字再好也退回半写实 2.5D。参考图想更"纯风格"，可用无主体裁剪/风格板（`luna-cases/` 有素材，实测有效但会把留白放大）。
  - 「图文交错」在 gpt-image 通道里只是拼成一条 prompt，**交错结构不存在**；该 Tab 本身也不挂「艺术风格参考图」4 模式（`build_backend_request` 只发提示词框原文），所以在那上面直接用 `BEST-PROMPT.md` 的模板即可。
  - 项目里还有文本通道可用 `gpt-5.6-luna` 之类模型做技术审阅：`GET https://new.aigc2d.com/v1/models`（**注意 `aigc2d` 节点的 base 是 `.../v1beta/models/`，文本模型列表/对话走 `.../v1/models`、`.../v1/chat/completions`**，OpenAI 兼容）；`gpt-5.x` 是推理模型，要留 `max_completion_tokens` 余量（首次 4000 全是 reasoning、content 为空）。
  - **给 gpt-image 通道调画风注入逻辑前先读这里。**

### `models/`（模型资源）

- `models/wd14/`: WD14 模型与标签资源
- `models/upscaler/`: Upscaler 模型目录（`OmniSR`、`Real-CUGAN`、`SRFormer-Light`）
- `models/ESRGAN/`: ESRGAN 模型目录
- `models/segmentation/`: 分割相关模型目录（`GroundingDINO`、`sam2`、`hf-cache`）

## 4. 配置文件索引（按实际读取路径）

- `conf/config.json`: **统一配置**——文本分析/通用开关 + 图片生成 API（`current_api` / `apis.*` / 顶层 `style_ref_mode`）、图片相关界面记忆（`gpt_image2` / `webui_img2img` / `diff_cg` / `cached_image_models`）。
  （原先图片 API 单独放在 `conf/config-image.json`，2026-09 已合并到本文件；`api_backend.load_config` 仍保留旧文件回退读取，仅为兼容没做迁移的老机器。）
  - **密钥来源**：`apis.<节点>.api_key` 留空，改用 `IMAGE_MAKER_<节点名>_API_KEY`（`-`/`.`→`_`）；节点里写 `env_slug` 可让同家服务的多个节点共用一个变量（`aigc2d` 与 `aigc-2d-gpt` 都用 `IMAGE_MAKER_AIGC2D_API_KEY`）。解析在 `api_backend.resolve_api_key` / `get_api_config`，`_api_key_source` 标明 `env:变量名` / `config` / `none`；界面只显示变量名，永不显示密钥值。
  - **顶层密钥也走环境变量（2026-09-21 新增，别再踩坑）**：`conf/config.json` **顶层**的 `api_key` / `nsfw_api_key`（「文本分析 API」和「NSFW 文本 API」）以前只能写在配置里，图片分析的 **Step 1**、画风压缩、booru tag 翻译、SD 工作流的 LLM 步骤都用它 —— 于是「把 key 挪进 `.env`」对这条链完全无效，照样 401 `Invalid token`。现在统一为 `IMAGE_MAKER_TEXT_API_KEY` / `IMAGE_MAKER_NSFW_API_KEY`（通用名 `TEXT_API_KEY` / `NSFW_API_KEY` 也认），解析在 `api_backend.resolve_text_api_key` / `resolve_nsfw_api_key` / `apply_secret_env_overrides`，接入点：`app.get_text_config`（GUI 全部走它）+ 获取模型列表按钮、`sd_workflow_core.load_text_api_config_from_file`、`single_gen_debug_tab` 的 CompressPromptThread、`analysis_pipeline.analyze_single_image`、`tools/compress_styles.py`、`tools/translate_booru_tags.py`、`tools/doujin_translator.py`、`utils/filter_series_tags.py`、`utils/verify_pixiv_tags.py`。顶层两个 key 已置空（值只在 `.env`）；`apply_secret_env_overrides` 只改内存副本，**绝不回写**配置文件。回归用例：`tests/test_secret_env.py`。
  - **密钥文件 `.env`（重要）**：`setx` 设的用户环境变量**对在它之前启动的进程无效**（常驻 DSH / 编辑器 / 老终端内存里的环境块是旧的，spawn 出的子进程也拿不到）。所以密钥统一写仓库根目录 `.env`（模板 `.env.example`，`.env` 已 gitignore），由 `utils/env_loader.ensure_env_loaded()` 在 `api_backend.py` 导入时读取。优先级：**真实环境变量 > `.env` > `conf/config.json`**。改了 `.env` 后，已经开着的 GUI 要重启才会生效（导入时装载一次）。
  - **GUI 的图片节点 key 也必须解析环境变量**：各分析/生图 Tab 拿到的是 app.py 注入的 `img_config_getter_func`，现在统一是 `AppWindow.get_img_config`（`resolve_api_key` 先查环境变量，再回落到输入框；**节点配置要连 `env_slug` 一起传**，否则 `aigc-2d-gpt` 会去找 `IMAGE_MAKER_AIGC_2D_GPT_API_KEY` 而不是共用的 `IMAGE_MAKER_AIGC2D_API_KEY`）。只读界面输入框的话，`apis.aigc2d.api_key` 留空时自动生工会误报「生图 API Key 不能为空，请检查【全局配置】」并跳过 —— 2026-09-21 修的就是这个。同理「文本分析 API」的 key 框由 `_apply_text_key_hints` 提示变量名，图片节点 key 框由 `_apply_img_key_hint` 提示。
  - **设置页（全局配置）也要读环境变量合并后的配置（2026-09-23 修，用户实际报错）**：`app.py` 的 `load_config` / `on_api_type_changed` 以前用 `json.load` 直读 `conf/config.json`，于是「节点定义 + 密钥都只在 `.env`」的部署里，界面看不到 `IMAGE_MAKER_NODES` 定义的节点、`IMAGE_MAKER_CURRENT_API` 也不生效 —— 界面停在配置文件里剩下的那个节点（如 `whatup`）、key 显示为空，一点生图就报上面那句话。现在这两处都调 `api_backend.apply_env_nodes(config)`（`load_config` 内部也用它，是同一段逻辑抽出来的），并把环境变量里的节点名补进 `api_type_combo`（否则用户根本选不到）；`save_image_config` **不把环境变量拥有的字段抄进配置文件**（`_env_owned_fields`，否则配置会压过 .env、改了 .env 不生效），界面上这些字段置灰只读（`_apply_img_field_locks`：base_url/model/key）。回归用例：`tests/test_secret_env.py` 的 `test_settings_page_reads_env_only_nodes` / `test_settings_page_resolves_shared_env_slug` / `test_save_image_config_does_not_freeze_env_node` / `test_settings_page_locks_env_owned_fields`。
- `conf/config-sd.json`: SD 相关配置
- `conf/config-styles.json`: 画风预设（运行时读取；版本化文件在子模块 `submodules/image-maker-artstyle/config-styles.json`，本地 `conf/` 副本为 gitignore）
- `conf/config-z-image.json`: z-image 本地模型目录记忆
- `conf/config-autocomplete.json`: 自动补全配置
- `conf/config-cohere.json`: Cohere 相关配置
- `prompts/gpt-image-optimize/config.json`: **gpt-image 产物优化（重绘提线）的 prompt 与参数固件**（`system_prompt` / `detail_suffix` / `model` / `resolution` / `aspect_ratio` / `repeat` / `save_sub_dir`）；GUI 重绘模式会读写它，理论见 `docs/gpt-image-optimize/`
- `modules/image_generation/conf/config.json`: 模块内配置模板/副本（非主读取路径）

## 5. 艺术风格参考图模式

生图时可在「画风指令」基础上附加一张样例参考图（仅参考其艺术画风）。共有 4 种模式：

| 模式 key | UI 文案 | 行为 |
|---|---|---|
| `off` | 关闭 | 只使用画风指令文本，不带参考图 |
| `head` | 头部插入 | 样式指令头部追加「艺术风格参考」指令，参考图作为附件随图传入 |
| `priority` | 参考优先 | 参考图主导画风，画风指令改用压缩版 `prompt_compressed`（缺失时本地启发式压缩） |
| `interleave` | 图文交错 | 参考指令放在图后（`post_instructions`），图+文交错引导 |

### 5.1 统一接入点（改动生图逻辑必看）

- `utils/styles.py`:
  - `normalize_style_entry` / `style_prompt` / `style_prompt_compressed` / `style_prompt_gpt` / `style_ref_image`: 从 config-styles.json 条目取值（兼容旧字符串格式与 `{"prompt", "ref_image", "prompt_compressed", "prompt_gpt"}` 格式）
  - `ref_image_valid(path)`: 参考图路径存在性校验（`os.path.exists`）
  - `assemble_style_instructions(mode, ..., prompt_compressed, prompt_gpt)`: 按模式组装头部/图后指令；传了 `prompt_gpt` 就优先用它（gpt-image 通道）
  - `build_ref_gen_params(styles, style_name, mode, api_type="")`: **统一组装函数**，返回 `(head_instructions, post_instructions, ref_image_paths)`；参考图无效时自动回退 `off`；`api_type` 含 `gpt` 时改用 `prompt_gpt` 短版
  - `save_styles_file(path, styles)`: 写回 config-styles.json
- `utils/style_gpt.py`: **gpt-image 通道的「短版字段式画风说明」规则与取用**：`FIELD_KEYS`（8 字段）/ `validate_prompt_gpt`（字段齐全、150~680 字符、禁主体词）/ `parse_fields` / `format_prompt_gpt` / `repair_prompt_gpt(_lenient)` / `is_gpt_image_api_type` / `resolve_style_prompt` / `compose_prompt_gpt` / `build_conversion_prompts`。规则与维护说明：`docs/gpt-image-tid-style/prompt-gpt-rules.md`
  - **渲染语言条款（`repaint_clauses`，2026-09-24 补齐跨画风覆盖）**：字段是画风条目上一组英文条款，链路用在**首图**（`build_gpt_image_request(extra_clauses=…)` → `RENDERING LANGUAGE (follow exactly):` 段）与**重绘**（`run_pipeline(style_clauses=…)` → `STYLE LANGUAGE (from the reference image):` 段）。原来只有 `tinkle` 一条手写过（`BEST-PIPELINE.md` §十三，19 条 2707 字符），其余 29 条画风一条都没有、且 `sync_styles_to_submodule.py` 不同步该字段（版本化文件里连 tinkle 的都没有）。现在 `resolve_style_clauses(entry)`：**手写优先，缺失就按该画风自己的 `prompt_gpt` 8 字段确定性派生**（`derive_style_clauses`，8~12 条：调色板/光照/笔触/线条分级/材质/构图密度/细节/避免 + 两条通用「保彩度、别发灰」），`→ (clauses, source)`，`source` ∈ `entry` / `derived` / `none`。回归用例：`tests/test_analysis_gen.py`、`tests/test_analysis_channel.py`；审计用 `python tools/convert_styles_gpt.py --check`
- `utils/style_ref_widget.py`: `StyleRefModeCombo` 共享下拉控件
  - `load_saved_style_ref_mode()` / `save_style_ref_mode(mode)`: 读写 `conf/config.json` 顶层 `style_ref_mode`
  - `set_modes_available(has_ref)`: 无参考图时禁用非 `off` 项并回退 `off`
  - `effective_mode(has_ref)`: 无参考图/关闭 → `off`
- `modules/others/api_backend.py`:
  - `to_base64_compressed(path)` / `_maybe_compress_image_path(path)`: 参考图超 2048px 时自动压缩后再发送（与分析流程一致）
  - `generate_image_aigc2d` / `generate_image_whatai` / `_post_images_edits_request`: 已支持 `post_instructions` 与参考图附件压缩
  - `generate_image_aigc2d_gpt` / `build_gpt_image2_payload` / `normalize_gpt_image2_size` / `gpt_image2_output_extension`: **gpt-image-2（aigc2d `apis.aigc-2d-gpt`）专用通道**，与 Gemini 的 `/v1beta/models` 通道分离；无参考图走 `POST /v1/images/generations`(JSON)，带参考图走 `POST /v1/images/edits`(multipart `image[]`，最多 16 张)；尺寸只允许 `1024x1024` / `1536x1024` / `1024x1536` 三档（`auto`、2K/4K、自定义 WxH 一律收敛），不支持 `background=transparent` 与 `input_fidelity`；high 画质单张常需 3~5 分钟（`timeout` 默认 600s）。专用 Tab 见 `modules/image_generation/gpt_image2_tab.py`；同 Tab 的「站点」下拉还能切到 `autodl`（`apis.autodl`），走 `generate_image_openai_image(api_type="autodl")`，尺寸含 `auto`/`1792x1024`，一次一张且无法中途取消（旧 `autodl_image_edit_tab.py` 已合并归档）
  - 同一通道还挂着 gpt-image 全家（2026-09 实测 `GET /v1/models` 可见）：`gpt-image-2`、`gpt-image-2-c`(按次计费)、`gpt-image-1` / `-1-mini` / `-1.5`、`dall-e-3`，以及 **`gpt-image-2.5-flare` / `gpt-image-2.5-sunburst`**（ChatGPT Images 2.5，2026-09-08 发布；端点/字段与 2 一致，模型名换成 2.5 即可，`-c` 后缀=按次计费通道）。实测 edits 垫图：flare 1024x1536 medium 约 30s，sunburst 同参数约 180s。autodl 站点没有 2.5
  - 模型清单与探测：`GPT_IMAGE2_SITE_MODELS`（按站点的常用模型，aigc2d 含 2.5/`-c`，autodl 仅 gpt-image-2）、`GPT_IMAGE2_MODEL_NOTES`（下拉悬浮说明）、`pick_gpt_image_models(site, models, extra)`（拼下拉选项）、`resolve_models_endpoint(base, api_type)`（推 `GET /v1/models` 地址）、`list_available_models(api_type, config_path, timeout)`（实时拉取，无 PyQt）。GUI 模型下拉与 `tools/gpt_image2_gen.py --list-models` 共用这套

### 5.2 已接入 Tab（全部共用 `StyleRefModeCombo` + `build_ref_gen_params`）

- 单图分析 `single_analyzer.py`、批量分析 `batch_analyzer.py`、批量提示词生图 `prompt_generator.py`、角色设计 `char_design.py`、批量图片编辑 `image_edit.py`、单图调试生图 `single_gen_debug_tab.py`
- 每个 Tab 在 `update_styles()` 与切换画风时都会校验参考图文件是否存在，不存在则参考类模式不可用
- SD 类 Tab（sd_theme_batch / sd_storyline / sd_workflow）不适用：只把画风拼进 SD 提示词，不走内联参考图 API

### 5.3 压缩版指令（prompt_compressed）

- `tools/compress_styles.py`: 批量 LLM 压缩（默认目标 ~700 字符，约原文 1/4）
- `single_gen_debug_tab.py` 的样式编辑器与 `app.py` 设置页画风管理：提供「请求 LLM 重新生成」按钮，调用 `CompressPromptThread`（读 `conf/config.json` 文本 API）实时生成并回填

## 6. 修改建议流程（给 AI 助手）

0. 先读 `PROJECT_REQUIREMENTS.md`，遵守目录边界与「不产生一次性脚本」纪律。
1. 先看 `app.py`，确认功能所在 Tab 和调用链。
2. 再看对应模块文件（如 `char_design.py`、`image_edit.py`）。
3. 涉及配置持久化时，同时检查 `conf/` 与模块内 `conf/` 是否存在重复配置。
4. 修改后优先检查被改文件诊断，再做最小验证（`python -m py_compile` + 导入检查 + 能跑就跑 pytest）。
   - 跑 pytest **不要给命令套管道或重定向**（`python -m pytest -q -p no:cacheprovider` 直接跑即可）：`| Select-Object`、`> out.txt 2>&1` 会让 PowerShell 为原生进程开匿名管道，本机沙箱下 spawn 直接被拒并报 `[sandbox: file access denied]`，而提权是「按命令一次性授权」，于是每跑一次弹一次窗。原理与正确写法见 `PROJECT_REQUIREMENTS.md` 第 2 节。
5. **不要往 `data/<YYYYMMDD>/` 写测试数据**：分析类落盘入口已接入 `utils/output_isolation.resolve_output_target`，pytest 会话里（`tests/conftest.py` 自动设 `IMAGE_MAKER_TEST_OUTPUT=1`）产出统一落到 `data/test-result/<日期>/` 且带 `test-` 前缀。新增会落盘的用例优先用 `tmp_path`；发现日期目录混进测试 json/txt（典型的 `…-hash_0-title_0.json` 这种没有对应图片的），用 `python -m utils.output_isolation --apply` 搬走。
   - **`utils/post_process.run_pipeline` 也已接入（2026-09-23 修，用户实际反馈）**：以前用例里 `run_pipeline([img], steps)` 不传 `final_dir` 时，默认落到 `data/<今天>/`，一天下来日期目录里堆了 90 多个 `img-233500-a35670-final-sline50.png` 废文件。现在 `IMAGE_MAKER_TEST_OUTPUT=1` 时 `run_pipeline` 把默认落盘目录改写成 `data/test-result/<日期>/` 且产物名加 `test-` 前缀（`final_output_path()`，执行 `IMAGE_MAKER_TEST_OUTPUT=1` 时生效）。
   - 兜底搬迁的识别范围也扩了：除 `…-hash_0-title_0.json` 一族外，还认「**测试临时图名 + run-id + 工序串**」的流水线产物（`img-` / `in` / `shot` / `out` / `base` / `patch` / `multi` / `tight` … + `-final-…` / `-sline50` / `-local-…` / `-tone` / `-ink` / `-crop`），见 `output_isolation.PIPELINE_TEST_ARTIFACT_RE`；会话结束时 `tests/conftest.py` 自动搬走，手动搬用 `python -m utils.output_isolation --apply`。新增用例若会跑 `run_pipeline`，请显式传 `final_dir=str(tmp_path/...)`。

## 7. 当前已知状态

- `z-image` 模块代码保留，但 `app.py` 中未执行 `generation_tabs.addTab(self.z_image_edit_tab, ...)`，因此默认隐藏。
- `z-image` 加载逻辑为本地目录模式，不走 HuggingFace 自动下载。
- `conf/config-styles.json` 为主仓库 gitignore 文件（运行时读取），版本化文件位于子模块 `submodules/image-maker-artstyle/config-styles.json`；修改画风后需同步到子模块并提交。
- `.venv` 已删除（只有 pip/setuptools，从未安装依赖）；统一用系统 Python。
- `cache/` 全部在用：`history/`（make-pic 槽位导入历史）、`temp/`（make-pic 临时图，可清空内容）、`sd-req/`（SD 工作流缓存）、`last_state.json`（make-pic 状态）。
- `log/` 保留全部历史日志（约 17GB，用户选择不清理）。
- **本机出网只对 python 放行**：`curl.exe` 直连 HTTPS 返回 HTTP 000，PowerShell/.NET（HttpWebRequest）报 "connection closed on receive"，而 python（requests/urllib，TLS1.3/1.2 均可）直连正常。系统 WinINET 里配了 `http://127.0.0.1:9567` 代理，但 python 不用代理也能通。**需要联网验证/抓取时一律用 python**，不要用 curl 或 `Invoke-WebRequest`；DSH 自带的 `download_file` / `web_fetch` 走宿主代理也能用。

## 8. 快速批量生图运行手册（本机 SD-WebUI txt2img，给 AI 助手）

用户说「用本机 WebUI 生图」时，**不要再做一连串环境/依赖探查**，直接按下面步骤走。

- 服务地址：`http://127.0.0.1:7860`（`conf/config-sd.json` 的 `sd_url`，Forge/SD-WebUI）。
  - `GET /sdapi/v1/options`：健康检查（返回 `sd_model_checkpoint` / `sd_vae` / `forge_additional_modules`）
  - `POST /sdapi/v1/txt2img`：文生图（批量生图用这个）
  - `POST /sdapi/v1/img2img`：图生图（封装在 `utils/webui_img2img_client.py`）
- 正确 Python：**系统 Python `C:\Program Files\Python310\python.exe`**（自带 requests / PyQt6 / PIL）。`.venv` 已删除，不要再提它。
- **批量生图一律用 `tools/sd_batch_gen.py`**（无 PyQt，签名/参数与旧模板一致）：
  - 主题=提示词文件：`python tools/sd_batch_gen.py --theme-file prompts/sd-batch/<主题>.txt`
  - 主题文件语法：每行一条提示词；`WxH|提示词` 表示该张单独尺寸；`#` 注释；默认尺寸用 `--width/--height`。
  - 参数：`--output-subdir`（data/<日期>/ 下子目录）、`--prefix`（文件名前缀）、`--sd-group`（配置组）、`--start-idx`（补生成）、`--dry-run`（预检）。
  - 现有主题：`yuri_cg`、`yuri_batch`、`yuri_guofeng`、`yuri_lolita`、`lolita_batch`、`loli_*`、`dance`、`elegant_life*`、`miku_birthday*`、`street_elegant*`、`vintage_fashion`、`white_heels`、`black_heels`、`garter_white`、`white_pantyhose`、`sensual2`、`backless` 等 30 组（每个主题文件头部注释标注原脚本、默认尺寸、目标子目录与前缀）。
- 参数来源：`tools/sd_batch_gen.py` 直接读 `conf/config-sd.json`（`sd_url`、`current_sd_group`、`sd_config_groups`（每组 `sd_model`/`sd_vae`(list)/`sampler`/`scheduler`/`steps`/`cfg_scale`）、`fixed_prompt`、`fixed_negative_prompt`、`last_used_style`、`webui_extra_payload`），无需 import `sd_workflow_core`（那会连带拉起 PyQt6）。
- payload 拼装规则（`tools/sd_batch_gen.py` 的 `build_payload`，与原 `sd_gen_yuri_cg.py` 逐字一致）：
  - prompt = `fixed_prompt` + 每张提示词 + `last_used_style`
  - negative = 负向模板(`data/negative_prompts/common-negative.txt`) + `fixed_negative_prompt`
  - `override_settings`：`sd_model_checkpoint` = 当前组 `sd_model`；`forge_additional_modules` = 当前组 `sd_vae`（非 Automatic 项）；再合并 `webui_extra_payload`
- 常用尺寸（项目在 `sd_workflow_core.py` 的 `STORY_RESOLUTION_PRESETS` 已约定）：16:9 = **1824x1024**；9:16 = 1024x1824；3:2 = 1536x1024；2:3 = 1024x1536；1:1 = 1024x1024。
- 输出目录：`data/<YYYYMMDD>/<子目录>/<prefix>_<时间戳毫秒>_<idx>.png`（如 `data/20260816/sdyuri/yuri_*.png`）。
- 运行方式：后台跑要加 `-u`（`python -u tools/sd_batch_gen.py ...`）否则 stdout 被缓冲看不到进度；也可不加，靠 `glob data/**/sdyuri/*.png` 看文件增长。
- **禁止**：复制 `tools/sd_batch_gen.py` 或从 `useless/scripts/sd_gen/` 拿旧脚本改成新主题——新主题直接加一个 `prompts/sd-batch/<主题>.txt` 文件即可。
- 最简启动流程：① `python tools/sd_batch_gen.py --theme-file prompts/sd-batch/<主题>.txt --output-subdir <子目录> --prefix <前缀>`；② `run_in_background` + `python -u`；③ glob 输出目录看进度、`job_output` 等结束。若不确定服务在不在，只做一次 `GET /sdapi/v1/options` 即可。
