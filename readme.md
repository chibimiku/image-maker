# image-maker 本地 booru tagger 配置说明

> 开发/改动代码前，建议先阅读项目根目录的 `AGENTS.md`（模块索引与改动入口说明）。

## 画风参考图模式

生图时可以在「画风指令」之外附加一张样例参考图，仅参考它的艺术画风（不改变构图与内容）。

### 四种模式

| 模式 | 说明 |
|---|---|
| 关闭 | 只使用画风指令文本 |
| 头部插入 | 样式指令头部追加「艺术风格参考」指令，样例图作为附件 |
| 参考优先 | 样例图主导画风，画风指令改用压缩版指令（`prompt_compressed`，缺失时本地自动压缩） |
| 图文交错 | 参考指令放在样例图之后，图+文交错引导 |

适用 Tab：单图分析、批量分析、批量提示词生图、角色设计、批量图片编辑、单图调试生图。
参考图缺失或文件不存在时，参考类模式会自动禁用并回退到「关闭」。

### 配置格式

`config-styles.json`（画风预设）条目从旧字符串升级为对象：

```json
{
  "tinkle-style": {
    "prompt": "完整画风指令……",
    "ref_image": "data/style-ref/tinkle-test.png",
    "prompt_compressed": "LLM 压缩后的精简版指令……"
  }
}
```

- `prompt_compressed` 用于「参考优先」模式；缺失时生图会使用本地启发式压缩（头 900 + 尾 800 字符）兜底
- 全局参考模式选择保存在 `conf/config-image.json` 顶层 `style_ref_mode`（所有 Tab 共享）
- 批量生成压缩版指令：`python tools/compress_styles.py [--only <样式名>]`
- 样式编辑器（单图调试 Tab / 设置页画风管理）支持「请求 LLM 重新生成」压缩版指令按钮

### 参考图过大自动压缩

参考图最长边超过 2048px 时，会在发送给生图 API 前自动等比压缩（与分析流程一致的策略），避免超大 payload。

## 安装说明

1. 安装 Python 依赖：

```bash
pip install -r requirements.txt
```

如需运行本地测试，请额外安装：

```bash
pip install -r requirements-dev.txt
```

2. 下载 WD14 模型文件（任选一个系列，如 ConvNextV2）：
   - `model.onnx`
   - `selected_tags.csv`

3. 将模型放到以下任一位置：
   - `data/models/wd14/model.onnx`
   - `models/wd14/model.onnx`
   - `wd14_tagger_model/model.onnx`

   对应的标签文件放到同目录下的 `selected_tags.csv`。

4. `booru tags` 的词表过滤会读取 `config-autocomplete.json` 中的 `csv_path`，默认是：
   - `data/tags/danbooru.csv`

## 图片 Upscaler 模型放置说明

`图片Upscaler` 目前支持以下架构类型：

- `ESRGAN(DAT)`
- `SRFormer-Light`
- `OmniSR`
- `Real-CUGAN`

### 1. 模型目录规则

- `ESRGAN(DAT)`：
  - `data/models/ESRGAN/`
  - `models/ESRGAN/`
- `SRFormer-Light`：
  - `models/upscaler/SRFormer-Light/`
- `OmniSR`：
  - `models/upscaler/OmniSR/`
- `Real-CUGAN`：
  - `models/upscaler/Real-CUGAN/`

> 说明：`Real-CUGAN` 官方项目参考：<https://github.com/bilibili/ailab/tree/main/Real-CUGAN>

### 2. 文件扩展名规则（按设备）

- 当推理设备选择 `CPU优先` 或 `自动CUDA` 时：
  - 只扫描 `*.pt`、`*.pth`、`*.safetensors`
- 当推理设备选择 `NPU优先(ONNX推理)` 时：
  - 只扫描 `*.onnx`

### 3. 依赖说明

- `CPU/CUDA` 的本地架构推理依赖：
  - `torch`
  - `spandrel`
- `SRFormer-Light` / `OmniSR` / `Real-CUGAN` 额外建议安装：
  - `spandrel-extra-arches`
- `NPU(ONNX)` 推理依赖：
  - `onnxruntime`

示例安装命令：

```bash
pip install torch spandrel spandrel-extra-arches onnxruntime
```

## Prompt 目录说明

- 运行时使用的系统 Prompt、模板 Prompt 统一放在项目根目录的 `prompts/`
- 代码不再读取 `data/prompts/`
- 如果代码所需的 Prompt 文件缺失，界面或脚本会直接报错并中止，不再使用代码内置默认 Prompt 兜底
- 目前 `prompts/` 下包含单图分析、画风提取、同人翻译、booru tag 翻译、差分 CG、SD 提示词生成、角色设计和图片编辑等相关模板

## SD 批量工作流

当前 `图片生成 -> SD 批量工作流` 的流程已经调整为“统一走主设置页配置接口，再在工作流页执行任务”。

### 1. 先完成设置

在 `设置` 页中，先准备两类配置：

- `文本分析 API`
  - 配置常规大模型的 `Base URL`、`API Key`、`分析模型`
- `文本分析（NSFW）`
  - 如果题材需要更宽松的分析接口，可单独配置一套 NSFW 专用大模型
- `SD-WebUI接口配置`
  - 配置 `SD API URL`
  - 管理 `配置组`
  - 为每个配置组设置 `Checkpoint`、`VAE`、`Sampler`、`Scheduler`、`Steps`、`CFG`
  - 如有需要，可填写 `WebUI 附加 Payload`

说明：

- `SD-WebUI接口配置` 已从 `SD 批量工作流` 页面移出，统一放到 `设置` 中管理
- 旧的 Cohere 分支已移除，`SD 批量工作流` 现在只复用 `文本分析 API` / `文本分析（NSFW）`

### 2. 再进入 SD 批量工作流

进入 `图片生成 -> SD 批量工作流` 后，按以下顺序操作：

1. 填写 `绘画主题`
2. 选择 `Prompt 风格预设`
3. 选择或编辑 `正向模板`
4. 选择或编辑 `反向模板`
5. 视需要勾选 `使用文本分析（NSFW）配置`
6. 视需要勾选 `启用 System Prompt 兼容模式`
7. 设置：
   - `大模型请求轮数(Y)`
   - `单次返回组数(X)`
   - `附加固定正向提示词`
   - `附加固定反向提示词`
8. 点击 `保存配置并开始生成`

### 3. 当前执行逻辑

工作流运行时会按下面的顺序处理：

1. 读取 `prompts/sd-make-system_prompt.md`
2. 将 `绘画主题 + 正向模板` 发给文本分析大模型
3. 让大模型一次返回多组差异化的 SD 提示词及尺寸
4. 把返回结果缓存到 `cache/sd-req/`
5. 将 `固定正向提示词 + LLM 返回提示词 + 画风预设` 拼成最终正向提示词
6. 将 `反向模板 + 固定反向提示词` 拼成最终反向提示词
7. 读取 `设置 -> SD-WebUI接口配置` 中当前选中的配置组
8. 调用本地 `Stable Diffusion WebUI /sdapi/v1/txt2img`
9. 将生成图片保存到 `data/<日期>/sdmake/`

### 4. 使用建议

- 先在 `设置 -> SD-WebUI接口配置` 中把常用模型整理成多个配置组，再在工作流里频繁切换主题
- 如果是普通题材，默认走 `文本分析 API`
- 只有在确实需要时，再勾选 `使用文本分析（NSFW）配置`
- 如果 `WebUI 附加 Payload` 填写了 JSON，开始运行前会先校验格式

## 本地测试

- 已添加 `pytest` 基础测试配置：`pytest.ini`
- 已添加测试目录：`tests/`
- 已添加 mock 数据目录：`tests/mock_data/`

当前测试重点：

- `utils/prompt_loader.py` 的路径解析、读取、模板替换、缺失文件检测
- 关键 Prompt 文件是否存在
- Python 代码中是否还残留 `data/prompts` 引用
- `prompts/tmp.txt` 是否已清理
- GUI 入口与 PyQt6 迁移相关的库存检查与 smoke 测试

运行命令：

```bash
pytest
```

## 服饰采集策略文档

服饰采集与少女生图相关的当前基线、站点策略、配置来源、产物路径和后续 `theme/style/hybrid` 扩展规划，统一记录在：

- `docs/fashion-pipeline-strategy.md`
- `docs/fashion-theme-spec-template.md`

建议在调整服饰采集策略前，先更新这两份文档，再落代码。

## 网页抓取 CLI

项目根目录新增了一个轻量命令行工具：`web-probe.py`。

适合这些场景：

- 快速抓网页 HTML
- 提取 `__NEXT_DATA__`
- 跑正则拿链接或字段
- 批量提取 `href/src`
- 把结果直接落盘成文本或 JSON

### 1. 抓取网页原文

```bash
python web-probe.py fetch "https://wear.jp/women-category/onepiece/dress/" --print-chars 1000
```

保存到文件：

```bash
python web-probe.py fetch "https://wear.jp/women-category/onepiece/dress/" --out cache/wear_dress.html
```

### 2. 提取 Next.js `__NEXT_DATA__`

抓整段 JSON：

```bash
python web-probe.py next-data "https://wear.jp/women-category/onepiece/dress/" --out cache/wear_dress_next_data.json
```

只取某个路径：

```bash
python web-probe.py next-data "https://wear.jp/yyuk1101a/26674416/" --query "props.pageProps.coordinateItems[0]"
```

### 3. 正则提取

提取页面中的图片链接：

```bash
python web-probe.py regex "https://lolibrary.org/items/ap-delicious-lemonade-jsk" "https://[^\"']+\\.(jpg|jpeg|png|webp)[^\"']*" --limit 20
```

提取第一个捕获组并去重：

```bash
python web-probe.py regex "https://wear.jp/women-category/shoes/sandal/" "https://images\\.wear2\\.jp/[^\"']+" --group 0 --unique
```

### 4. 提取 href/src 链接

提取绝对链接：

```bash
python web-probe.py links "https://lolibrary.org/search?brands[]=angelic-pretty" --attr href --contains "/items/" --absolute --limit 20
```

提取图片源：

```bash
python web-probe.py links "https://wear.jp/yyuk1101a/26674416/" --attr src --contains "imgz.jp"
```

### 5. 读取本地文件再处理

如果你已经先把 HTML 存到本地，也可以继续分析：

```bash
python web-probe.py next-data cache/wear_dress.html --from-file
python web-probe.py regex cache/wear_dress.html "coordinate/[^\"']+" --from-file
```

### 6. 附加 Header

```bash
python web-probe.py fetch "https://example.com" --header "Accept: text/html" --header "X-Test: 1"
```

### 7. Cookie 与登录态

如果目标站点需要登录态，可以直接传 Cookie：

```bash
python web-probe.py fetch "https://example.com/private" --cookie "sessionid=abc; csrftoken=xyz"
```

也可以从文件读取 Cookie：

```bash
python web-probe.py fetch "https://example.com/private" --cookie-file cache/cookies.txt
```

`--cookie-file` 支持三种格式：

- 浏览器插件导出的 JSON
- Netscape cookie jar 格式
- 纯 `Cookie` 字符串文本

例如：

```bash
python web-probe.py next-data "https://wear.jp/some/private/page" --cookie-file cache/wear_cookies.json
```

也可以指定一个目录，按域名自动寻找 cookies 文件：

```bash
python web-probe.py fetch "https://wear.jp/some/private/page" --cookie-dir-auto cache/browser-cookies
```

例如目录里存在这些文件之一即可自动命中：

- `wear.jp.json`
- `wear.jp.txt`
- `www.wear.jp.json`
- `www.wear.jp.cookies.txt`

### 8. 直接下载链接

从页面里提取图片链接并直接下载：

```bash
python web-probe.py download "https://wear.jp/yyuk1101a/26674416/" --attr src --contains "imgz.jp" --download-dir cache/downloads
```

### 9. 为什么不能直接复用当前浏览器身份

当前这个本地 Agent 运行环境默认没有直接控制你正在使用的浏览器，也不会自动读取你的浏览器 Profile、Cookie 数据库或登录会话。

主要原因有两类：

- 权限与安全：浏览器 Cookie、登录态、Profile 数据属于敏感凭据，默认不应该被自动读取
- 工具边界：当前项目里可直接复用的是 Python/文件系统/HTTP 请求能力，没有现成的“附着到你当前浏览器会话并代发请求”的安全工具链

所以这里不是“只能 Python 才能爬”，而是：

- 当前 Agent 最稳定、最可审计、最容易复现的方式，是 Python 发 HTTP 请求
- 如果你希望复用浏览器身份，最现实的做法是先从浏览器导出 cookies，再交给 `web-probe.py`

后续如果你想继续扩展，也可以做两种方向：

- 增加“读取浏览器导出的 cookies 文件并自动请求”
- 再进一步接入独立浏览器自动化方案（如 Playwright / Selenium），但那就不再是现在这种轻量 CLI 了

## PyQt6 人工冒烟

建议在真实桌面环境下额外做一轮主界面人工冒烟，重点验证 `app.py`。

启动命令：

```powershell
python app.py
```

如需跳过启动阶段的 `onnxruntime` 预热，可使用：

```powershell
$env:IMAGE_MAKER_SKIP_ONNXRUNTIME_PRELOAD=1
python app.py
```

建议检查项：

- 主窗口是否正常打开，是否存在启动即崩溃或空白界面
- 主界面多组 Tab 来回切换是否流畅，是否出现卡死、焦点异常、内容空白
- `单图分析` 中拖拽本地图片后，预览、按钮状态、日志是否正常更新
- `单图分析` 中复制图片到剪贴板后按 `Ctrl+V`，预览和日志是否正常更新
- 托盘通知相关路径在系统托盘可用或不可用时都不应导致程序崩溃

通过标准：

- 不崩溃
- 拖拽可用
- 剪贴板粘贴可用
- 多 Tab 切换可用
- 托盘通知路径不崩溃

建议记录模板：

```md
### app.py 人工冒烟记录

- 日期：
- 环境：Windows 桌面 / 是否跳过 onnxruntime 预热
- 启动：通过 / 失败
- 多 Tab 切换：通过 / 异常
- 单图分析拖拽：通过 / 异常
- 单图分析剪贴板 Ctrl+V：通过 / 异常
- 托盘通知：通过 / 不可见但不崩 / 异常
- 备注：
```

## config-autocomplete.json 配置项（中文）

- `local_booru_tagger_model_path`
  - 本地 WD14 模型 `model.onnx` 路径，支持相对路径和绝对路径。
  - 为空时按内置候选路径自动查找。

- `local_booru_tagger_tags_path`
  - 本地 WD14 标签定义文件 `selected_tags.csv` 路径。
  - 为空时按内置候选路径自动查找。

- `local_booru_tagger_max_tags`
  - WD14 推理后最多保留的候选 tag 数量。

- `local_booru_tagger_general_threshold`
  - General 类标签阈值（0~1）。

- `local_booru_tagger_character_threshold`
  - Character 类标签阈值（0~1）。

- `local_booru_tagger_meta_threshold`
  - Meta 类标签阈值（0~1）。

- `local_booru_tagger_rating_threshold`
  - Rating 类标签阈值（0~1）。

- `local_booru_tagger_keep_rating_tags`
  - 是否在最终候选中保留 rating 类标签（`true/false`）。
  - `false` 时即使分数达到阈值也会过滤掉 rating 标签。

- `local_booru_tagger_use_autocomplete_filter`
  - 是否使用 `csv_path` 指向的 danbooru 词表做二次过滤（`true/false`）。

## 示例配置

```json
{
  "enable_autocomplete": true,
  "csv_path": "data/tags/danbooru.csv",
  "max_results": 50,
  "min_chars": 2,
  "local_booru_tagger_model_path": "data/models/wd14/model.onnx",
  "local_booru_tagger_tags_path": "data/models/wd14/selected_tags.csv",
  "local_booru_tagger_max_tags": 60,
  "local_booru_tagger_general_threshold": 0.35,
  "local_booru_tagger_character_threshold": 0.35,
  "local_booru_tagger_meta_threshold": 0.75,
  "local_booru_tagger_rating_threshold": 0.75,
  "local_booru_tagger_keep_rating_tags": false,
  "local_booru_tagger_use_autocomplete_filter": true
}
```
