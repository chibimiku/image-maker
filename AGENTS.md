# Image Maker Agent Guide

本文件用于给 Trae/AI 助手提供项目快速索引。  
每次开始改动前，请先阅读本文件，再进入具体代码文件。

## 1. 项目目录总览（当前结构）

- 主界面入口：`app.py`
- 业务模块目录：`modules/`
- 工具与运行时目录：`utils/`
- 配置目录：`conf/`
- 模型目录：`models/`
- 独立入口脚本：`make-pic.py`、`sd-make-pic.py`、`doujin_translator.py`、`translate_booru_tags.py`

### 根目录核心文件

- `app.py`: 主窗口、Tab 组织、跨模块联动
- `make-pic.py`: 生图脚本（独立入口）
- `sd-make-pic.py`: SD 生图脚本（独立入口）
- `doujin_translator.py`: 翻译脚本（独立入口）
- `translate_booru_tags.py`: booru tags 翻译脚本（独立入口）
- `test.py`: 调试/测试脚本

## 2. 模块目录索引

### `modules/image_analysis/`（图片分析）

- `single_analyzer.py`: 单图分析
- `batch_analyzer.py`: 批量分析
- `style_analyzer.py`: 多图画风提取
- `json_dataset_tab.py`: JSON 数据集导出
- `pic_cate_tab.py`: 图片分类切分

### `modules/image_generation/`（图片生成与编辑）

- `prompt_generator.py`: 批量提示词与生图
- `image_edit.py`: 批量图片编辑
- `char_design.py`: 角色设计生成
- `single_gen_debug_tab.py`: 单图调试生图
- `webp_compressor.py`: PNG/WebP 压缩
- `upscaler_tab.py`: 图片 Upscaler
- `flux2_client_tab.py`: WebUI Img2Img 客户端
- `diff_cg_tab.py`: 差分 CG 生成
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

### `tools/`（独立脚本）

- `compress_styles.py`: 批量调用 LLM 把 config-styles.json 中每个画风的完整指令压缩为 `prompt_compressed` 并固化（支持 `--only <样式名>` 单独处理）

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

### 5.2 已接入 Tab（全部共用 `StyleRefModeCombo` + `build_ref_gen_params`）

- 单图分析 `single_analyzer.py`、批量分析 `batch_analyzer.py`、批量提示词生图 `prompt_generator.py`、角色设计 `char_design.py`、批量图片编辑 `image_edit.py`、单图调试生图 `single_gen_debug_tab.py`
- 每个 Tab 在 `update_styles()` 与切换画风时都会校验参考图文件是否存在，不存在则参考类模式不可用
- SD 类 Tab（sd_theme_batch / sd_storyline / sd_workflow）不适用：只把画风拼进 SD 提示词，不走内联参考图 API

### 5.3 压缩版指令（prompt_compressed）

- `tools/compress_styles.py`: 批量 LLM 压缩（默认目标 ~700 字符，约原文 1/4）
- `single_gen_debug_tab.py` 的样式编辑器与 `app.py` 设置页画风管理：提供「请求 LLM 重新生成」按钮，调用 `CompressPromptThread`（读 `conf/config.json` 文本 API）实时生成并回填

## 6. 修改建议流程（给 AI 助手）

1. 先看 `app.py`，确认功能所在 Tab 和调用链。
2. 再看对应模块文件（如 `char_design.py`、`image_edit.py`）。
3. 涉及配置持久化时，同时检查 `conf/` 与模块内 `conf/` 是否存在重复配置。
4. 修改后优先检查被改文件诊断，再做最小验证。

## 7. 当前已知状态

- `z-image` 模块代码保留，但 `app.py` 中未执行 `generation_tabs.addTab(self.z_image_edit_tab, ...)`，因此默认隐藏。
- `z-image` 加载逻辑为本地目录模式，不走 HuggingFace 自动下载。
- `conf/config-styles.json` 为主仓库 gitignore 文件（运行时读取），版本化文件位于子模块 `submodules/image-maker-artstyle/config-styles.json`；修改画风后需同步到子模块并提交。
