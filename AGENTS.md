# Image Maker Agent Guide

本文件用于给 Trae/AI 助手提供项目快速索引。  
每次开始改动前，请先阅读本文件，再进入具体代码文件。

## 1. 项目入口

- 主界面入口: `app.py`
- 配置文件目录: `conf/` 目录下 `config-*.json`
- 功能模块目录: `modules/`（下设 `image_analysis` / `image_generation` / `others`）
- 工具函数目录: `utils/`
- 独立入口脚本: 根目录下 `make-pic.py`、`sd-make-pic.py`、`doujin_translator.py` 等
- 运行结果目录: `data/<YYYYMMDD>/...`（按功能分子目录）

## 2. 功能模块索引

### 核心 UI 与路由

- `app.py`: 主窗口、Tab 组织、全局配置读写、跨模块联动逻辑

### 图片分析相关 → `modules/image_analysis/`

- `modules/image_analysis/single_analyzer.py`: 单图分析
- `modules/image_analysis/batch_analyzer.py`: 批量分析
- `modules/image_analysis/style_analyzer.py`: 多图画风提取
- `modules/image_analysis/json_dataset_tab.py`: JSON 数据集导出
- `modules/image_analysis/pic_cate_tab.py`: 图片分类切分

### 图片生成/编辑相关 → `modules/image_generation/`

- `modules/image_generation/prompt_generator.py`: 批量提示词与生图
- `modules/image_generation/image_edit.py`: 批量图片编辑
- `modules/image_generation/char_design.py`: 角色设计生成
- `modules/image_generation/z_image_edit_tab.py`: z-image 编辑（当前在主 UI 中已隐藏，不显示 Tab）
- `modules/image_generation/webp_compressor.py`: PNG/WebP 压缩
- `modules/image_generation/flux2_client_tab.py`: WebUI Img2Img 客户端
- `modules/image_generation/upscaler_tab.py`: 图片 Upscaler
- `modules/image_generation/single_gen_debug_tab.py`: 单图调试生图
- `modules/image_generation/diff_cg_tab.py`: 差分 CG 生成

### 其它功能 → `modules/others/` 或根目录

- `modules/others/api_backend.py`: API 后端相关封装（多模块共享）
- `modules/others/tag_completer.py`: 标签补全（SD 生图脚本使用）
- `modules/others/booru_tag_generator.py`: booru tag 生成器
- `doujin_translator.py`: 翻译相关功能（独立入口）
- `make-pic.py`: 生成脚本（独立入口）
- `sd-make-pic.py`: SD 相关生成脚本（独立入口）
- `test.py`: 测试脚本
- `translate_booru_tags.py`: booru tags 翻译（独立入口）

### 工具与模型辅助 → `utils/`

- `utils/wd14_tagger.py`: WD14 打标逻辑
- `utils/booru_tags.py`: booru tags 处理
- `utils/pic_cate.py`: 分类切分逻辑
- `utils/task_runtime.py`: 任务运行时工具
- `utils/image_upscale_runtime.py`: Upscale 运行时
- `utils/upscaler.py`: Upscaler 管道
- `utils/upscaler_arch.py`: Upscaler 架构定义
- `utils/webui_img2img_client.py`: WebUI Img2Img API 客户端
- `models/wd14/`: WD14 模型与标签资源

## 3. 关键配置文件

- `conf/config.json`: 文本分析 API、NSFW 开关、最后使用画风、分类页状态
- `conf/config-image.json`: 图片生成 API 配置（按 API 类型分组）
- `conf/config-sd.json`: SD 相关配置
- `conf/config-styles.json`: 画风预设
- `conf/config-z-image.json`: z-image 页本地模型目录记忆
- `conf/config-autocomplete.json`: 自动补全配置
- `conf/config-cohere.json`: Cohere 相关配置

## 4. 修改建议流程（给 AI 助手）

1. 先看 `app.py`，确认该功能所在 Tab 和调用链。
2. 再看对应模块文件（如 `char_design.py`、`image_edit.py`）。
3. 涉及配置持久化时，同时检查对应 `config-*.json` 的读写逻辑。
4. 修改后优先检查被改文件诊断，再做最小验证。

## 5. 当前已知状态

- `z-image` 模块代码保留，但在 `app.py` 中已移除 `generation_tabs.addTab(...)`，因此默认隐藏。
- `z-image` 加载逻辑已限制为本地目录模式（不走 HuggingFace 自动下载）。

