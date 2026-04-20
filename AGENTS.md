# Image Maker Agent Guide

本文件用于给 Trae/AI 助手提供项目快速索引。  
每次开始改动前，请先阅读本文件，再进入具体代码文件。

## 1. 项目入口

- 主界面入口: `app.py`
- 配置文件目录: 项目根目录下 `config-*.json`
- 工具函数目录: `utils/`
- 运行结果目录: `data/<YYYYMMDD>/...`（按功能分子目录）

## 2. 功能模块索引

### 核心 UI 与路由

- `app.py`: 主窗口、Tab 组织、全局配置读写、跨模块联动逻辑

### 图片分析相关

- `single_analyzer.py`: 单图分析
- `batch_analyzer.py`: 批量分析
- `style_analyzer.py`: 多图画风提取
- `json_dataset_tab.py`: JSON 数据集导出
- `pic_cate_tab.py`: 图片分类切分

### 图片生成/编辑相关

- `prompt_generator.py`: 批量提示词与生图
- `image_edit.py`: 批量图片编辑
- `char_design.py`: 角色设计生成
- `z_image_edit_tab.py`: z-image 编辑（当前在主 UI 中已隐藏，不显示 Tab）
- `webp_compressor.py`: PNG/WebP 压缩

### 其它功能

- `api_backend.py`: API 后端相关封装
- `tag_completer.py`: 标签补全相关
- `doujin_translator.py`: 翻译相关功能
- `make-pic.py`: 生成脚本
- `sd-make-pic.py`: SD 相关生成脚本

### 工具与模型辅助

- `utils/wd14_tagger.py`: WD14 打标逻辑
- `utils/booru_tags.py`: booru tags 处理
- `utils/pic_cate.py`: 分类切分逻辑
- `utils/task_runtime.py`: 任务运行时工具
- `models/wd14/`: WD14 模型与标签资源

## 3. 关键配置文件

- `config.json`: 文本分析 API、NSFW 开关、最后使用画风、分类页状态
- `config-image.json`: 图片生成 API 配置（按 API 类型分组）
- `config-sd.json`: SD 相关配置
- `config-styles.json`: 画风预设
- `config-z-image.json`: z-image 页本地模型目录记忆
- `config-autocomplete.json`: 自动补全配置
- `config-cohere.json`: Cohere 相关配置

## 4. 修改建议流程（给 AI 助手）

1. 先看 `app.py`，确认该功能所在 Tab 和调用链。
2. 再看对应模块文件（如 `char_design.py`、`image_edit.py`）。
3. 涉及配置持久化时，同时检查对应 `config-*.json` 的读写逻辑。
4. 修改后优先检查被改文件诊断，再做最小验证。

## 5. 当前已知状态

- `z-image` 模块代码保留，但在 `app.py` 中已移除 `generation_tabs.addTab(...)`，因此默认隐藏。
- `z-image` 加载逻辑已限制为本地目录模式（不走 HuggingFace 自动下载）。

