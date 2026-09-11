# 项目要求（PROJECT REQUIREMENTS）— DSH / AI 助手必读

> **本文件是项目硬性要求清单。任何 DSH / AI 助手在本仓库开展修改前，必须先完整阅读
> `PROJECT_REQUIREMENTS.md` 与 `AGENTS.md`，并全程遵守本文条款。**
> 本文条款优先于一般编码习惯；与 AGENTS.md 冲突时以本文件为准。

## 0. 总原则：目录即职责，脚本不散养

本目录（`D:\code\image-maker`）曾经把所有一次性功能都做成根目录脚本，导致根目录一度
混入 30+ 个 `sd_gen_*.py`、fashion 实验脚本、调试脚本和数据脚本。**现在根目录只允许
出现启动入口与文档**。新功能必须落到对应职责目录，功能复用靠模块，不是靠复制文件。

## 1. 目录边界（谁该住哪里）

| 目录 | 职责 | 禁止 |
|---|---|---|
| 根目录 | 只保留：`app.py`、`make-pic.py`、`sd-make-pic.py`、`publish_server.py`（启动入口）＋ `AGENTS.md`、`PROJECT_REQUIREMENTS.md`、`readme.md`、`.gitignore` | 禁止新增任何 .py 脚本、图片、JSON、日志 |
| `modules/` | 业务功能模块（GUI Tab / 服务 / 钩子），按领域分子包 | 禁止放一次性实验逻辑 |
| `utils/` | 跨模块复用的纯工具函数与运行时 | 禁止放只被一个脚本用的逻辑 |
| `tools/` | 无头 CLI 工具（用户手动跑的命令行入口） | 禁止每个实验新建一个脚本；同族功能必须合并为一个 CLI |
| `prompts/` | LLM prompt 模板与批量主题数据（.md/.txt/.json） | 禁止把提示词硬编码进 .py |
| `conf/` | 运行时配置（`*.json`，gitignore；白名单除外） | 禁止把机器相关路径写进代码 |
| `tests/` | pytest 用例与实验对比脚本 | 禁止与测试无关的散文件 |
| `data/` | 运行时数据（生图输出 `data/<YYYYMMDD>/...`、采集素材、缓存 json） | 禁止放功能脚本（已全部清走） |
| `cache/` | 运行缓存（`history/`、`temp/`、`sd-req/`、`last_state.json` 仍被 make-pic / SD 工作流使用） | 勿删在用内容；`temp/` 可定期清空 |
| `log/` | 按天日志（api_backend 持续写入） | 勿清理旧日志（用户选择保留） |
| `useless/` | 历史归档（gitignore）：`scripts/`（如 `sd_gen/`、`fashion/`、`fashion-batch-legacy/`、`misc/`）、`junk/`、旧版本备份 | 归档文件不参与运行、不作为代码引用 |
| `outputs/` | SD 工作流默认输出目录（被 `sd_workflow_core` 使用） | 保留 |

## 2. 运行与启动（硬性规定）

- **唯一 Python**：系统 Python `C:\Program Files\Python310\python.exe`（自带 requests / PyQt6 / PIL / numpy / opencv）。
  **`.venv` 已于 2026-09 删除**——它只有 pip/setuptools，从未安装任何项目依赖，不要再创建或提及虚拟环境。
- 启动 GUI：`python app.py`（主界面）；`python make-pic.py`（赛博暖暖）；`python sd-make-pic.py`（SD 工作流单窗）；`python publish_server.py`（发布 Server，默认端口 18765）。
- 后台跑批量/长任务必须加 `-u`（`python -u ...`），否则 stdout 被缓冲看不到进度。
- 禁止在项目里 `pip install` 到系统 Python 以外的环境；依赖清单在 `requirements.txt` / `requirements-dev.txt`（pytest）。

## 3. 批量生图纪律（最重要）

- **SD 批量**：一律使用 `tools/sd_batch_gen.py`，主题文案放 `prompts/sd-batch/*.txt`
  （每行一条提示词；`WxH|提示词` 表示该张单独尺寸；`#` 为注释）。
  禁止再复制模板创建新的 `sd_gen_*.py`。原 30 个脚本已归档 `useless/scripts/sd_gen/`。
- **Fashion 批量（采集→生图→分析）**：一律使用 `tools/fashion_batch.py --profile <名称>`，
  主题画像（character_spec / extra_prompt / save_subdir / collect_base / count / 分析开关）在
  `prompts/fashion-batch-themes.json` 扩展，不要改脚本、不要复制脚本。
- 配置一律读取 `conf/config-sd.json`（SD）与 `conf/config-image.json`（AIGC2D），
  **禁止硬编码模型名 / VAE / 采样器 / 尺寸 / API 地址**。
- 尺寸约定（与 `sd_workflow_core.STORY_RESOLUTION_PRESETS` 一致）：
  16:9=1824x1024，9:16=1024x1824，3:2=1536x1024，2:3=1024x1536，1:1=1024x1024。
- 输出目录：生图到 `data/<YYYYMMDD>/<子目录>/`；采集素材到 `data/fashion-collector/<base>/...`；不要输出到根目录。

## 4. 开发纪律

1. 新增功能：写进 `modules/<领域>/` 下对应模块；通用逻辑进 `utils/`；CLI 入口进 `tools/`（薄壳，只做参数解析与流程编排）。
2. **不要为单个实验创建脚本**——同族功能合并成一个 CLI（参数化）+ 数据文件（prompts/conf）。
3. 提示词模板一律放 `prompts/`，运行时用 `utils/prompt_loader.py` 读取；禁止把长 prompt 字符串写死在 .py。
4. 修改前先读：`PROJECT_REQUIREMENTS.md` → `AGENTS.md` → 目标模块；涉及配置持久化时检查 `conf/` 与模块内 `conf/` 是否重复。
5. 改动后最小验证：改过的 `.py` 至少 `python -m py_compile`；涉及启动链路跑一次导入检查；能跑 pytest 就跑 `python -m pytest -q`（offscreen 冒烟测试会扫 tools/ 与 modules/）。
6. 归档：确定不再使用的脚本/残留 → 移动（不是删除）到 `useless/scripts/<分类>/` 或 `useless/junk/`，并在最终回复中列明。

## 5. 禁止事项清单

- 根目录新增 .py / .json / 图片 / 日志。
- 新造 `sd_gen_*.py`、`fashion_pipeline*.py`、`test*.py` 之类的一次性脚本。
- 在代码里写死机器路径、代理地址（`http://127.0.0.1:7897` 这类可进 `prompts/fashion-batch-themes.json` 画像但不过代码常量）、端口以外的秘密。
- 删除 `cache/history`、`cache/sd-req`、`cache/last_state.json`（make-pic 与 SD 工作流在用）。
- 动 `docs/fashion-pipeline-strategy.md` 的基线内容（策略留档）。
- 在无网络/无 SD-WebUI 时直接跑生图脚本（先确认 `http://127.0.0.1:7860` 存活，用 `--dry-run` 预检）。

## 6. 常用入口速查

| 想做什么 | 命令 |
|---|---|
| SD 批量生图（主题=提示词文件） | `python tools/sd_batch_gen.py --theme-file prompts/sd-batch/<主题>.txt` |
| SD 批量（补生成/续跑） | 加 `--start-idx N --output-subdir <同目录> --prefix <同前缀>` |
| 只预检不生成 | `python tools/sd_batch_gen.py --theme-file ... --dry-run` |
| Fashion 批量采集生图 | `python tools/fashion_batch.py --profile backless_blonde` |
| 头图全链路分析（投稿格式） | `python tools/analyze_fashion.py --dir data/<日期>/<子目录>` |
| gpt-image-2 生图/编辑（无头） | `python tools/gpt_image2_gen.py --prompt "..." [--site autodl] [--image ref.png] [--dry-run]` |
| booru 标签翻译 | `python tools/translate_booru_tags.py` |
| 同人本翻译 | `python tools/doujin_translator.py` |
| 网页抓取 | `python tools/web-probe.py <fetch\|next-data\|regex\|links\|download> ...` |
| 画风指令压缩 | `python tools/compress_styles.py` |
| pytest | `python -m pytest -q` |
