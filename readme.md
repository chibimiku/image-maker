# image-maker 本地 booru tagger 配置说明

> 开发/改动代码前，建议先阅读项目根目录的 `AGENTS.md`（模块索引与改动入口说明）。

## 多图画风提取、报告与续训

「多图画风提取」首次默认 5 轮、每轮检查 4 张图，支持导入外部 Agent 的筛图清单、三路测试生图、十二维视觉比较、观察报告后手动选版本及从指定结果续训。打开已有结果后默认新增 3 轮，例如已有 3 轮时继续第 4–6 轮。每次运行使用独立目录，并保留旧终审候选供比较。

默认勾选「自动比较」与「自动深度指标」：视觉比较每批最多 4 个候选，保留成功批次缓存；Gram、AdaIN、LPIPS、CSD 独立运行，默认 **NPU 优先**。选「仅 Intel NPU」可避免回退到 CUDA。报告包含逐张对照、覆盖率、模型参数和实际设备证据，指标不换算百分比、不自动覆盖画风。更新 Python 代码后通常须重启 app；2026-10-06 本轮按用户要求暂不重启，已有 GUI 不会热更新，新 CLI 立即读取新版。

详见 [操作与字段说明](docs/style-analyzer-workflow.md)、[深度指标验证记录](docs/style-extraction/deep-metrics-integration-20261004.md) 和 [DeepSeek 的 taya_oco 再识别 3 轮提示词](docs/style-extraction/deepseek-taya-oco-resume-3rounds.md)。

## 双图画风相似度与面部 / 发丝指标

`图片生成 → 画风相似度` 支持输入两张图片，显示 **8 组整图指标 + 13 项面部 / 发丝细项**。双图页、CLI 和「多图画风提取」统一调用 `utils/style_similarity.py` 的 `compare_images`；提取流程将各轮及终审的每张候选与每张源图分别配对，不另写一套评分逻辑。

- 整图：Gram、AdaIN、LPIPS、CSD，以及明度层次、多尺度边缘、线条连续性、负空间。
- 面部 / 发丝：发丝细腻程度、**仅发丝**连贯性、眼睛亮度、上下眼睑间距、睫毛画法统计、眼宽、双眼间距、眼睑弧度；另补虹膜占比、虹膜内高光、上下眼线粗细、眼角倾斜、眼鼻口比例。几何分量按脸宽或眼宽归一化。
- 距离与局部统计贴近度分别报告，保留原始分量、逐对结果、覆盖率及版本；不合成未经校准的相似百分比，也不冒充 LoRA 训练 loss。任意参考配对缺失，该项正式均值保持空值。
- 双图页默认开启「自动面部定位（文本 API）」，会调用当前看图文本模型生成并缓存定位候选；关闭后仅用已保存定位。画风提取的同名开关默认关闭，需手动开启。

「面部 / 发丝定位…」可查看叠加曲线、滚轮放大、中键平移、逐点校正及确认定位。自动候选标记 `provisional`，仅显示暂定测量；两图定位都经过人工确认后才计正式均值。闭眼、遮挡、低分辨率或视角不可比不补分；发丝采样带必须全部位于标出的头发内部。定位按图片 SHA-256 保存到 `cache/style-regions/`，修改保留历史，结果缓存区分公式、图片 / 定位 hash、设备与精度。

在项目根目录运行：

```powershell
# 本地指标；面部细项使用已保存定位，首次无定位时标为 unavailable：
python tools/style_metrics_verify.py --similarity A.png B.png --device cpu --out cache/temp/similarity/result.json
# 自动生成面部 / 发丝定位候选（会调用文本 API）：
python tools/style_metrics_verify.py --similarity A.png B.png --locate-regions --device auto-npu
# 一张候选逐张比较多张参考图：
python tools/style_metrics_verify.py --similarity A.png B.png --reference C.png --reference D.png --device cpu
# 已有画风提取结果：各轮候选 × 全部源图：
python tools/style_metrics_verify.py --comparison-state path/to/state.json --device auto-npu
# 导入已检查且匹配图片 hash 的定位：
python tools/style_metrics_verify.py --similarity A.png B.png --regions A-regions.json --regions B-regions.json --device cpu
# 共享入口及面部细项 unittest：
python -m unittest discover -s tests -p test_style_similarity.py
python -m unittest discover -s tests -p test_style_face_metrics.py
```

`--similarity` 是日常比较入口；原有 `--pair` 保留为四项深度指标的部署验证入口。默认双图 JSON / HTML 输出到 `cache/temp/style-similarity/<时间戳>/`。整图 `status` 与面部 `face_status` 分开，自动候选不能算面部完整成功。

验证：相关完整回归 **138 passed / 5 skipped**，最后调整另复跑 **42 项全部通过**；真实 CPU 双图的整图八组均成功。局部测量通过可控图样验证，**自动定位尚未完成真实动漫图在线精度验收**。本轮不重启正在运行的 app，新页面待正常重启后加载。

完整公式、字段与边界见 [共享画风相似度说明](docs/style-extraction/STYLE-SIMILARITY-20261006.md)。

真实 Gemini 带参考图测试见 [Puracotte / Sakurapion 实测](docs/style-extraction/GEMINI-SIMILARITY-TEST-20261006.md)：两张产物×两张参考的全图八组指标均成功；Sakurapion 产物的深度指标更偏向 Puracotte。自动面部/发丝定位叠图精度未通过，细项仍需人工校正确认，不能把候选分数当正式相似度。

下一步可靠性验收按 [DeepSeek 受控实验协议](docs/style-extraction/DEEPSEEK-SIMILARITY-CONTROLLED-EXPERIMENT-20261006.md) 执行，允许有账本的技术失败重试，禁止按分数挑选成功尝试；目前协议待执行。

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
- 全局参考模式选择保存在 `conf/config.json` 顶层 `style_ref_mode`（所有 Tab 共享）
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

项目根目录新增了一个轻量命令行工具：`tools/web-probe.py`。

适合这些场景：

- 快速抓网页 HTML
- 提取 `__NEXT_DATA__`
- 跑正则拿链接或字段
- 批量提取 `href/src`
- 把结果直接落盘成文本或 JSON

### 1. 抓取网页原文

```bash
python tools/web-probe.py fetch "https://wear.jp/women-category/onepiece/dress/" --print-chars 1000
```

保存到文件：

```bash
python tools/web-probe.py fetch "https://wear.jp/women-category/onepiece/dress/" --out cache/wear_dress.html
```

### 2. 提取 Next.js `__NEXT_DATA__`

抓整段 JSON：

```bash
python tools/web-probe.py next-data "https://wear.jp/women-category/onepiece/dress/" --out cache/wear_dress_next_data.json
```

只取某个路径：

```bash
python tools/web-probe.py next-data "https://wear.jp/yyuk1101a/26674416/" --query "props.pageProps.coordinateItems[0]"
```

### 3. 正则提取

提取页面中的图片链接：

```bash
python tools/web-probe.py regex "https://lolibrary.org/items/ap-delicious-lemonade-jsk" "https://[^\"']+\\.(jpg|jpeg|png|webp)[^\"']*" --limit 20
```

提取第一个捕获组并去重：

```bash
python tools/web-probe.py regex "https://wear.jp/women-category/shoes/sandal/" "https://images\\.wear2\\.jp/[^\"']+" --group 0 --unique
```

### 4. 提取 href/src 链接

提取绝对链接：

```bash
python tools/web-probe.py links "https://lolibrary.org/search?brands[]=angelic-pretty" --attr href --contains "/items/" --absolute --limit 20
```

提取图片源：

```bash
python tools/web-probe.py links "https://wear.jp/yyuk1101a/26674416/" --attr src --contains "imgz.jp"
```

### 5. 读取本地文件再处理

如果你已经先把 HTML 存到本地，也可以继续分析：

```bash
python tools/web-probe.py next-data cache/wear_dress.html --from-file
python tools/web-probe.py regex cache/wear_dress.html "coordinate/[^\"']+" --from-file
```

### 6. 附加 Header

```bash
python tools/web-probe.py fetch "https://example.com" --header "Accept: text/html" --header "X-Test: 1"
```

### 7. Cookie 与登录态

如果目标站点需要登录态，可以直接传 Cookie：

```bash
python tools/web-probe.py fetch "https://example.com/private" --cookie "sessionid=abc; csrftoken=xyz"
```

也可以从文件读取 Cookie：

```bash
python tools/web-probe.py fetch "https://example.com/private" --cookie-file cache/cookies.txt
```

`--cookie-file` 支持三种格式：

- 浏览器插件导出的 JSON
- Netscape cookie jar 格式
- 纯 `Cookie` 字符串文本

例如：

```bash
python tools/web-probe.py next-data "https://wear.jp/some/private/page" --cookie-file cache/wear_cookies.json
```

也可以指定一个目录，按域名自动寻找 cookies 文件：

```bash
python tools/web-probe.py fetch "https://wear.jp/some/private/page" --cookie-dir-auto cache/browser-cookies
```

例如目录里存在这些文件之一即可自动命中：

- `wear.jp.json`
- `wear.jp.txt`
- `www.wear.jp.json`
- `www.wear.jp.cookies.txt`

### 8. 直接下载链接

从页面里提取图片链接并直接下载：

```bash
python tools/web-probe.py download "https://wear.jp/yyuk1101a/26674416/" --attr src --contains "imgz.jp" --download-dir cache/downloads
```

### 9. 为什么不能直接复用当前浏览器身份

当前这个本地 Agent 运行环境默认没有直接控制你正在使用的浏览器，也不会自动读取你的浏览器 Profile、Cookie 数据库或登录会话。

主要原因有两类：

- 权限与安全：浏览器 Cookie、登录态、Profile 数据属于敏感凭据，默认不应该被自动读取
- 工具边界：当前项目里可直接复用的是 Python/文件系统/HTTP 请求能力，没有现成的“附着到你当前浏览器会话并代发请求”的安全工具链

所以这里不是“只能 Python 才能爬”，而是：

- 当前 Agent 最稳定、最可审计、最容易复现的方式，是 Python 发 HTTP 请求
- 如果你希望复用浏览器身份，最现实的做法是先从浏览器导出 cookies，再交给 `tools/web-probe.py`

后续如果你想继续扩展，也可以做两种方向：

- 增加“读取浏览器导出的 cookies 文件并自动请求”
- 再进一步接入独立浏览器自动化方案（如 Playwright / Selenium），但那就不再是现在这种轻量 CLI 了

## 单图分析 + 生图 CLI（与主界面同一条执行链）

单图分析界面的「生成时使用的画风预设」可选 **随机**。每次提交分析任务时从已启用的实际画风预设中抽取一个（不含「默认(无附加)」），同一任务的自动原始/优化生图及随后手动生图沿用该结果。队列、分析 JSON 的 `generation_style_name` 和生图请求记录的都是抽中的画风名；「随机」只保存为界面选择，不同步到其他 Tab。单独点生图按钮且当前任务没有绑定随机画风时，会在该次生图入口抽取。

在仓库根目录用系统 Python 运行。`tools/analyze_fashion.py` 的旧 `--dir` 用法仍是只分析；加入 `--image`（可重复）或 `--generate` 后，命令会在无头 Qt 中创建主界面使用的同一个 `SingleAnalyzerWidget`，使用同一套分析线程、生图线程、GPT 断点、身份/人体/最终审计和发布判断。默认生成优化提示词图；失败返回非零，原始分析 JSON 与过程断点仍保留。

流程边界：`single_analyzer.WorkerThread` 做 Step 1–5 分析并保存 JSON/TXT；Gemini 通道把所选原始/优化描述交给 `ImageGenWorkerThread`，GPT 通道用分析内容锚组首图请求、由 `GptImageGenWorkerThread` 执行重绘与门禁。两条链的图片产物最后经可选 JPG 处理并由队列状态机收尾。CLI 只映射控件和排队，不复制模型调用逻辑，也不保存 GUI 偏好。分析 JSON 已存在时的 GPT 实验入口是 `tools/analysis_gpt_run.py`。

```powershell
python -u tools/analyze_fashion.py --image "data/source/example.jpg" --style sheya-style --channel gemini --reference-mode priority --generate refined --dry-run
python -u tools/analyze_fashion.py --image "data/source/example.jpg" --style sheya-style --channel gemini --reference-mode priority --generate refined
python -u tools/analyze_fashion.py --image "data/source/example.jpg" --style sheya-style --channel gpt --generate refined --repaint --quality high --scope full
python -u tools/analyze_fashion.py --dir "data/source" --generate both --channel gpt --test-output
python -u tools/analyze_fashion.py --image "data/source/example.jpg" --generate none --save-to-source
```

`--dry-run` 只创建无头分析 Tab、校验图片与画风并打印实际控件映射；不调用分析或生图 API。`--test-output` 把产物隔离到 `data/test-result/<日期>/`。默认正式产物仍在 `data/<日期>/`；`--save-to-source` 只适用于 `--generate none`，也不能与 `--test-output` 同用。API key 由现有 `.env`/`conf/config.json` 解析，不在命令行传递。

| 分析 Tab 选项 | CLI 参数 |
|---|---|
| 本地图/目录批量、原始/优化提示词生图 | `--image`（可重复）/ `--dir`、`--generate none\|original\|refined\|both` |
| NSFW 文本接口、服装检查/风格覆盖、去照片风格、booru 数量 | `--nsfw`、`--outfit-check` / `--outfit-style`、`--remove-photo-style`、`--booru-tag-limit` |
| 画风与参考图模式 | `--style`、`--reference-mode off\|head\|priority\|interleave` |
| 生图通道 | `--channel gemini\|gpt`；Gemini 可用 `--image-api` / `--image-model` 覆盖当前节点 |
| GPT 画质、首图接口、尺寸跟随 | `--quality`、`--first-pass-mode`、`--size-follow-input` / `--no-size-follow-input` |
| GPT 重绘、范围、结构线、局部区域 | `--repaint` / `--no-repaint`、`--scope`、`--structure`、`--local --region <区域键>` |
| GPT 色调、目标与加墨 | `--tone --tone-target style\|photo`、`--ink` |
| JPG 后处理 | `--jpg-upscale --upscale-model <名称> --upscale-by <倍率> --webp-target-mb <MB>` |
| 请求超时、文本模型 | `--timeout`、`--text-model` |
| 分析/生图长宽比覆盖 | `--aspect-ratio-first`、`--aspect-ratio-second`（默认保持分析比例） |

`--region stable-four` 选择界面的实验性四区串联；其余区域键可用 `python -u tools/analyze_fashion.py --help` 和 `utils/post_process.py` 的 `REGION_LABELS` 查看。未传 `--style` 时沿用 GUI 保存的画风。GPT 通道使用 GUI 固定的 `aigc-2d-gpt` 节点；`--image-api` 与 `--image-model` 只针对 Gemini。界面上的手动停止可在 CLI 中用 `Ctrl+C`。已有 GPT 断点可用下列旧实验入口从默认失败工序续跑；该入口默认隔离测试产出，请勿把它当作正式发布命令：

```powershell
python -u tools/analysis_gpt_run.py --app-resume-checkpoint "data/<日期>/analysis-gpt-image/<任务>/generation-checkpoint.json"
```

若已有分析 JSON，GPT 生图实验仍可用 `python -u tools/analysis_gpt_run.py --json <分析.json> --style <画风> --dry-run`。该命令的实验参数可能与 GUI 默认值不同；要复现 UI 的完整分析到生图流程，请用上面的 `tools/analyze_fashion.py` 入口。

### 命令行拾取历史：从结果日志列出过程、选图续跑或发布

GUI 的分析队列右键 **「拾取历史」** 会自动扫描 `data/<日期>/analysis-gpt-image/<任务>/generation-checkpoint*.json`，在窗口顶部选择该任务已保存的尝试，再按时间查看过程文件和工序。选中某条队列任务时只展示它自己的断点；要浏览全部已保存的 GPT 任务，可在队列空白处右键选择「拾取历史」。无须手工挑选 JSON。Gemini 分析若在 Step 1 就失败，尚未进入生图工序，因此不会有 GPT 生图断点。Step 1 如显示「服务端内容过滤导致响应被截断」，表示接口返回 `finish_reason=content_filter`，该次响应不能解析为完整分析 JSON；需检查输入图、分析提示词或文本分析服务。

**Gemini 通道的重跑**：Gemini 生图不写工序断点，所以选中这类队列任务右键「拾取历史」时不再只弹「没有断点」，而是询问是否**按相同参数新建一条队列任务**重跑——弹窗里列出通道、画风、画幅、提示词长度与沿用原任务号，并可勾选重跑「优化提示词 / 原始提示词」（默认勾选该任务上次实际跑过的那种）。确认后队列里多出一条 `[进行中·Gemini 生图]` 的任务，画风、画幅、提示词、分析产物路径都取自原记录，生图通道固定为 Gemini（不跟当前界面单选走），原任务与原图保留不动。沿用原任务号是为了让新产物仍能被 `publish_server.py` 关联到同目录的投稿 JSON。记录里没有可用提示词（分析未完成）或标明是 gpt-image 通道但断点已丢失时，会给出对应提示而不是静默改通道。

`tools/analyze_fashion.py --image ... --channel gpt --generate refined` 完成每张图后，会在终端打印任务结果，同时逐项保存到 `data/<日期>/analysis-cli-results-<时间>-<短码>.json`；带 `--test-output` 时日志和产物都进入 `data/test-result/<日期>/`。日志中的 `checkpoints` 指向各任务的 `generation-checkpoint.json`。失败时仍可读取断点中的成功产物和审计文件。

下面的三条命令在完整分析 CLI `tools/analyze_fashion.py` 中直接可用；`tools/analysis_gpt_run.py` 也接受同样的历史参数。输入可以是**结果日志、单个 `generation-checkpoint.json` 或任务目录**。先 `list`，再从打印的编号里选择 `--task`（日志中的任务，默认 1）和 `--pick`（该任务的过程文件编号）：

```powershell
python tools/analyze_fashion.py --history-list "data/<日期>/analysis-cli-results-<时间>-<短码>.json"
python tools/analyze_fashion.py --history-resume "data/<日期>/analysis-cli-results-<时间>-<短码>.json" --task 1 --pick 15
python tools/analyze_fashion.py --history-publish "data/<日期>/analysis-cli-results-<时间>-<短码>.json" --task 1 --pick 15
```

`list` 先打印任务状态、hash、分析 JSON、最终产物与错误，再按文件时间列出工序名称、图片、审计 JSON、提示词与绝对路径；`--history-json` 改为结构化 JSON 输出，便于脚本提取 `task`、`pick`、`stage` 和 `path`。也可直接传断点，例如：

```powershell
python tools/analyze_fashion.py --history-list "data/20260928/analysis-gpt-image/65ebea5b-renian-221533-e26768/generation-checkpoint.json"
```

这条任务的实际候选图 `65ebea5b-final-rescue-1_222055-f5eb51.jpg` 在列表中是 **15 号**。经人工确认后，将它直接选为最终结果的命令是：

```powershell
python tools/analyze_fashion.py --history-publish "data/20260928/analysis-gpt-image/65ebea5b-renian-221533-e26768/generation-checkpoint.json" --pick 15
```

本机执行结果是 `data/20260928/65ebea5b_renian-final-rescue-1_222055-f5eb51.jpg` 软链；已确认它指向所选过程 JPG，且 `publish_server.py` 能匹配同目录的 `65ebea5b` 分析 JSON。命令只把图片放到发布目录，**不会自动加入发布 Server 队列**；启动 `python publish_server.py` 后，将这张最终图拖入队列即可。原失败断点仍标记为 `error`。同一候选图重复执行发布命令会产生带序号的另一份最终图，请勿重复操作。

`--history-resume` **从选中图之后的工序重新推断**，创建独立 `generation-checkpoint-pickup-*.json`，复用前面的成功结果，再执行后续审计和修订；它会调用模型，可能产生费用，失败仍返回非零并保留候选图。选中 `final_review` 的失败候选图时，会再次执行最终复核。原断点不改动，新尝试在同目录写独立的 `resume-result-<断点名>.json`，同时刷新 `resume-result.json` 供查看最近一次结果。`--history-publish` 不调用模型：人工将选中图作为最终图放进原日期目录，优先创建软链，系统不允许时复制；保留原审计失败状态，并写 `published-final.json` 记录选图来源。发布图的文件名把任务 hash 放在首个下划线字段，便于 `publish_server.py` 关联同目录的分析 JSON。

实验入口 `tools/analysis_gpt_run.py --json ... --output-dir ...` 的 `request.json` 也可以用 `--history-list` 查看、用 `--history-publish` 人工选图；它没有逐阶段断点，因此不能用 `--history-resume` 续跑。需要可选节点的重推断，请用上面的完整分析 CLI 生成断点。

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

MAYLA 采集依赖可选的 Playwright/Chromium；未安装时主界面仍可启动，只有执行 MAYLA 采集时会提示缺少依赖。视频生成 Tab 的输出目录在“生成参数”中设置；SD 工作流读取配置时不会自动写回 `conf/config-sd.json`。

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
