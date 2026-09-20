# Image Maker Agent Guide

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

- `single_analyzer.py`: 单图分析（Step 1~5 模块级函数 + Qt WorkerThread）
- `analysis_pipeline.py`: **无头全链路分析**（Step 1~5 编排 + 投稿格式落地 `save_result_to_source`，CLI 见 `tools/analyze_fashion.py`；`analyze_image_step1` 供批量出图后单步分析）
- `batch_analyzer.py`: 批量分析
- `style_analyzer.py`: 多图画风提取
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
- `gpt_image_optimize.py`: **gpt-image 产物优化（Gemini 重绘提线）运行时**：`load_config` / `save_config`（读写 `prompts/gpt-image-optimize/config.json`）、`build_repaint_prompt`（主 prompt + 细节后缀拼接）、`plan_output`（按源文件名派生输出前缀与子目录）；被 GUI Tab、CLI、后端共用
- `output_isolation.py`: **测试产出隔离**：`resolve_output_target(save_dir, base_filename)` 在 `IMAGE_MAKER_TEST_OUTPUT=1` 时把 `data/<日期>/…` 改道到 `data/test-result/<日期>/…` 并加 `test-` 前缀（未设该环境变量时行为与改动前一致）；`find_legacy_test_artifacts` / `relocate_legacy_test_artifacts` 负责搬迁历史污染，命令行入口 `python -m utils.output_isolation [--apply]`。接入点：`single_analyzer.on_process_finished`、`batch_analyzer.save_result`、`analysis_pipeline.save_result_to_source`；`tests/conftest.py` 自动开启并兜底搬迁

### `tools/`（无头 CLI 工具）

- `compress_styles.py`: 批量调用 LLM 把 config-styles.json 中每个画风的完整指令压缩为 `prompt_compressed` 并固化（支持 `--only <样式名>` 单独处理）
- `sd_batch_gen.py`: **本机 SD-WebUI 批量生图统一引擎**（无 PyQt；读 `conf/config-sd.json` + `prompts/sd-batch/*.txt` 主题文件；`--dry-run` 预检）
- `fashion_batch.py`: **Fashion 批量采集-生图无头 CLI**（`--profile <名称>`，画像在 `prompts/fashion-batch-themes.json`）
- `analyze_fashion.py`: 头图全链路分析 CLI（落投稿格式，功能在 `modules/image_analysis/analysis_pipeline.py`）
- `gpt_image2_gen.py`: **gpt-image-2 生图/编辑无头 CLI**（站点 `new.aigc2d` / `autodl` 二选一；支持参考图走 `/images/edits`、`--prompt-file` 批量、`--dry-run`、`--list-sites`（含各站点常用模型）、`--list-models`（实时 GET /v1/models 只列 gpt-image/dall-e 系列）、`--timeout` 临时覆盖；纯 python 无 PyQt，适用于只能跑 python 出网的机器）
  - 产物优化：`--repaint --image <产物.png>` 走 Gemini 重绘提线（同 GUI「重绘」模式），`--repaint-model/--repaint-resolution/--repaint-repeat/--repaint-prompt-file` 可调，`--repaint-show-prompt` 只打印最终固件不调接口
- `doujin_translator.py`: 同人本批量翻译 GUI（独立入口）
- `translate_booru_tags.py`: booru tags 翻译 CLI
- `web-probe.py`: 网页抓取 CLI（`utils/web_probe_cli.py` 的薄壳入口）

### `tests/`（测试与对比脚本）

- `calc_style_similarity.py`: CLIP 向量 + HSV 直方图 + 线条统计的「画风贴近度」量化对比
- `test_style_ref.py`: 艺术风格参考图模式下的生图对比实验脚本

### `docs/`（文档与实验记录）

- `fashion-pipeline-strategy.md` / `fashion-theme-spec-template.md` / `publish_server_api.md` / `pyqt6-migration-checklist.md`: 既有文档
- `gpt-image-optimize/`: **gpt-image 产物优化的理论与全部实验数据**（`README.md` 总纲 + `metrics.md` 指标定义与已知偏差 + 四个批次实验 `experiments-A/B/C/D-*.md` + `FINAL-PROMPT.md` 固件快照 + `data/` 指标原始表与各版 prompt 全文 + `tools/` 分析脚本快照）。**要改重绘 prompt 或调指标口径，先读这里。**
- `gpt-image-tid-style/`: **gpt-image 系列上的「画风生图」可行性实验（tid 画风，2026-09-20；无头 CLI + 16 张产物 + CLIP/HSV 指标 + 官方文档快照）**。结论：`prompt` 上限是官方文档的 **32000 字符**，实测 11k~40k 全都能出图（**「前置词太长导致无法生图」不成立**）；长提示词真正的代价是**主体崩 + 参考图失效**——11k 全量指令会把「少女站在浅水」画成水下精灵，把画风说明压到 **400~700 字符**时风格贴近度与主体符合度同时最好。gpt-image-2 的 `input_fidelity` 不可调、输入图一律高保真，所以它**把参考图当"要编辑的原图"**（会连参考图角色的发色/瞳色一起搬），必须用「保留什么 / 换掉什么」的短指令显式约束。「图文交错」在 gpt-image 通道里只是拼成一条 prompt，**交错结构不存在**。`gpt-image-2.5-flare` 同输入下 output token 约为 gpt-image-2 的 1/3、更快、画风更贴。**给 gpt-image 通道调画风注入逻辑前先读这里。**

### `models/`（模型资源）

- `models/wd14/`: WD14 模型与标签资源
- `models/upscaler/`: Upscaler 模型目录（`OmniSR`、`Real-CUGAN`、`SRFormer-Light`）
- `models/ESRGAN/`: ESRGAN 模型目录
- `models/segmentation/`: 分割相关模型目录（`GroundingDINO`、`sam2`、`hf-cache`）

## 4. 配置文件索引（按实际读取路径）

- `conf/config.json`: **统一配置**——文本分析/通用开关 + 图片生成 API（`current_api` / `apis.*` / 顶层 `style_ref_mode`）、图片相关界面记忆（`gpt_image2` / `webui_img2img` / `diff_cg` / `cached_image_models`）。
  （原先图片 API 单独放在 `conf/config-image.json`，2026-09 已合并到本文件；`api_backend.load_config` 仍保留旧文件回退读取，仅为兼容没做迁移的老机器。）
  - **密钥来源**：`apis.<节点>.api_key` 留空，改用 `IMAGE_MAKER_<节点名>_API_KEY`（`-`/`.`→`_`）；节点里写 `env_slug` 可让同家服务的多个节点共用一个变量（`aigc2d` 与 `aigc-2d-gpt` 都用 `IMAGE_MAKER_AIGC2D_API_KEY`）。解析在 `api_backend.resolve_api_key` / `get_api_config`，`_api_key_source` 标明 `env:变量名` / `config` / `none`；界面只显示变量名，永不显示密钥值。
  - **密钥文件 `.env`（重要）**：`setx` 设的用户环境变量**对在它之前启动的进程无效**（常驻 DSH / 编辑器 / 老终端内存里的环境块是旧的，spawn 出的子进程也拿不到）。所以密钥统一写仓库根目录 `.env`（模板 `.env.example`，`.env` 已 gitignore），由 `utils/env_loader.ensure_env_loaded()` 在 `api_backend.py` 导入时读取。优先级：**真实环境变量 > `.env` > `conf/config.json`**。
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
  - `normalize_style_entry` / `style_prompt` / `style_prompt_compressed` / `style_ref_image`: 从 config-styles.json 条目取值（兼容旧字符串格式与新的 `{"prompt", "ref_image", "prompt_compressed"}` 格式）
  - `ref_image_valid(path)`: 参考图路径存在性校验（`os.path.exists`）
  - `assemble_style_instructions(mode, ...)`: 按模式组装头部/图后指令
  - `build_ref_gen_params(styles, style_name, mode)`: **统一组装函数**，返回 `(head_instructions, post_instructions, ref_image_paths)`；参考图无效时自动回退 `off`
  - `save_styles_file(path, styles)`: 写回 config-styles.json
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
