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
- `gpt_image2_tab.py`: gpt-image-2 专用 Tab（「站点」下拉切 `new.aigc2d`=`apis.aigc-2d-gpt` / `autodl`=`apis.autodl`，「模式」下拉切生图/编辑；独立于全局 API 类型；aigc2d 尺寸仅 3 档、autodl 含 auto/1792x1024；两站点都支持参考图垫图与编辑）
  - 旧的独立 Tab `autodl_image_edit_tab.py` 已合并进本 Tab，文件归档在 `useless/junk/autodl_image_edit_tab.py`
- `z_image_edit_tab.py`: z-image 编辑（代码保留，默认不在主 UI Tab 显示）
- `conf/config-image.json`: 图片生成相关配置模板（模块内）

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

### `tools/`（无头 CLI 工具）

- `compress_styles.py`: 批量调用 LLM 把 config-styles.json 中每个画风的完整指令压缩为 `prompt_compressed` 并固化（支持 `--only <样式名>` 单独处理）
- `sd_batch_gen.py`: **本机 SD-WebUI 批量生图统一引擎**（无 PyQt；读 `conf/config-sd.json` + `prompts/sd-batch/*.txt` 主题文件；`--dry-run` 预检）
- `fashion_batch.py`: **Fashion 批量采集-生图无头 CLI**（`--profile <名称>`，画像在 `prompts/fashion-batch-themes.json`）
- `analyze_fashion.py`: 头图全链路分析 CLI（落投稿格式，功能在 `modules/image_analysis/analysis_pipeline.py`）
- `gpt_image2_gen.py`: **gpt-image-2 生图/编辑无头 CLI**（站点 `new.aigc2d` / `autodl` 二选一；支持参考图走 `/images/edits`、`--prompt-file` 批量、`--dry-run`、`--list-sites`、`--timeout` 临时覆盖；纯 python 无 PyQt，适用于只能跑 python 出网的机器）
- `doujin_translator.py`: 同人本批量翻译 GUI（独立入口）
- `translate_booru_tags.py`: booru tags 翻译 CLI
- `web-probe.py`: 网页抓取 CLI（`utils/web_probe_cli.py` 的薄壳入口）

### `tests/`（测试与对比脚本）

- `calc_style_similarity.py`: CLIP 向量 + HSV 直方图 + 线条统计的「画风贴近度」量化对比
- `test_style_ref.py`: 艺术风格参考图模式下的生图对比实验脚本

### `models/`（模型资源）

- `models/wd14/`: WD14 模型与标签资源
- `models/upscaler/`: Upscaler 模型目录（`OmniSR`、`Real-CUGAN`、`SRFormer-Light`）
- `models/ESRGAN/`: ESRGAN 模型目录
- `models/segmentation/`: 分割相关模型目录（`GroundingDINO`、`sam2`、`hf-cache`）

## 4. 配置文件索引（按实际读取路径）

- `conf/config.json`: 文本分析与通用开关配置（代码中读取）
- `conf/config-image.json`: 图片生成 API 主配置（代码中读取，含顶层 `style_ref_mode` 全局参考模式持久化）
- `conf/config-sd.json`: SD 相关配置
- `conf/config-styles.json`: 画风预设（运行时读取；版本化文件在子模块 `submodules/image-maker-artstyle/config-styles.json`，本地 `conf/` 副本为 gitignore）
- `conf/config-z-image.json`: z-image 本地模型目录记忆
- `conf/config-autocomplete.json`: 自动补全配置
- `conf/config-cohere.json`: Cohere 相关配置
- `modules/image_generation/conf/config-image.json`: 模块内配置模板/副本（非主读取路径）

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
  - `load_saved_style_ref_mode()` / `save_style_ref_mode(mode)`: 读写 `conf/config-image.json` 顶层 `style_ref_mode`
  - `set_modes_available(has_ref)`: 无参考图时禁用非 `off` 项并回退 `off`
  - `effective_mode(has_ref)`: 无参考图/关闭 → `off`
- `modules/others/api_backend.py`:
  - `to_base64_compressed(path)` / `_maybe_compress_image_path(path)`: 参考图超 2048px 时自动压缩后再发送（与分析流程一致）
  - `generate_image_aigc2d` / `generate_image_whatai` / `_post_images_edits_request`: 已支持 `post_instructions` 与参考图附件压缩
  - `generate_image_aigc2d_gpt` / `build_gpt_image2_payload` / `normalize_gpt_image2_size` / `gpt_image2_output_extension`: **gpt-image-2（aigc2d `apis.aigc-2d-gpt`）专用通道**，与 Gemini 的 `/v1beta/models` 通道分离；无参考图走 `POST /v1/images/generations`(JSON)，带参考图走 `POST /v1/images/edits`(multipart `image[]`，最多 16 张)；尺寸只允许 `1024x1024` / `1536x1024` / `1024x1536` 三档（`auto`、2K/4K、自定义 WxH 一律收敛），不支持 `background=transparent` 与 `input_fidelity`；high 画质单张常需 3~5 分钟（`timeout` 默认 600s）。专用 Tab 见 `modules/image_generation/gpt_image2_tab.py`；同 Tab 的「站点」下拉还能切到 `autodl`（`apis.autodl`），走 `generate_image_openai_image(api_type="autodl")`，尺寸含 `auto`/`1792x1024`，一次一张且无法中途取消（旧 `autodl_image_edit_tab.py` 已合并归档）

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
