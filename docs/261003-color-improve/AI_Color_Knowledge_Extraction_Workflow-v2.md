# AI 配色知识增强 · 工作指引 v2

> **目标读者**：Codex（或任何在本仓库施工的编码助手）。
> **上游文档**：`docs/261003-color-improve/AI_Color_Knowledge_Extraction_Workflow.md`（下称「原计划书」）。
> **本版定位**：原计划书是按「一本有文字层的电子书」写的，与本项目现实对不上；
> v1 改成可执行的工程件后，由 `gpt-6-sol` 做了一轮技术审阅
> （审阅记录：`docs/261003-color-improve/review-gpt6-sol-20261002.md`），本版是**逐条回应审阅意见之后**的定稿。
> 施工前必须先读 `PROJECT_REQUIREMENTS.md` → `AGENTS.md` → 本文件。

---

## 0. 一页速览

**一句话目标**：把《超人气配色手册》这本书变成一个**能被本项目生图链路直接吃下去**的配色知识库，
让「配色」从「提示词里随手写两个 mood 词」变成三件有据可查的东西：
**实测色值**、**可执行英文条款**、**确定性的色调目标图**。

**输入**：`docs/261003-color-improve/15659159.pdf`（150 页扫描件，无文字层，详见第 1 节）。

**产物**：

| 产物 | 路径 | 生命周期 |
|---|---|---|
| 页图 | `cache/temp/color-book/pages/page-NNN.jpg` + `manifest.json` | 中间产物，可删 |
| 页级知识 | `cache/temp/color-book/page-json/page-NNN.json` | 中间产物，可续跑 |
| 实测色板 | `cache/temp/color-book/swatches/page-NNN.json` | 中间产物 |
| 规则表 | `data/color-knowledge/color_theory.json` | 稳态 |
| 配色库 | `data/color-knowledge/palette_library.json` | 稳态 |
| 光照规则 | `data/color-knowledge/lighting_rules.json` | 稳态 |
| 页码索引 | `data/color-knowledge/knowledge_index.json` | 稳态 |
| **tone 目标图** | `data/color-knowledge/targets/<slug>.png` + `targets.json` | 稳态，**生图链路直接消费** |
| 色板核对图 | `data/color-knowledge/palette-check.png` | 人眼验收 |

**四条铁律**（违反任何一条，整条链的产出都不可信）：

1. **不问视觉模型要 hex。** 这是扫描印刷页，颜色是 CMYK 网点，模型给的三位数只能是印象值。
   色值一律由 OpenCV 在页面上实测（第 7 节）；模型只负责读**色名**与**色彩关系**。
2. **不凭印象补全。** 版块没有就 `null`/`[]`，读不清就进 `unreadable`。空值不是失败，"编一个"才是。
3. **中间产物只落 `cache/temp/`。** 只有第 9 节列出的注入点才允许碰 `conf/` 与生产代码。
4. **没有实测证据的结论不许写成已完成。** 本项目的历史文档有硬规矩：
   没在线复现过就写「未验证」。本文件每一条数字后面都跟着可复核的产物或命令。

### 0.1 本版已经跑通的部分（每一条都可复核）

| 工序 | 实测结果 | 复核方式 |
|---|---|---|
| `extract` | 150/150 页写出、0 跳过、73.2 MB、页图 1824×2784 | `cache/temp/color-book/pages/manifest.json`；进程退出码 0 |
| `analyze` | 7 页抽样（PDF 12/13/14/22/29/37/99）全部成功、0 失败、共 8 分 11 秒 | `cache/temp/color-book/page-json/*.json`（每份带 `_model` 字段） |
| 页型判定 | 7/7 正确：`front_matter`×1 / `theory`×2 / `case_analysis`×2 / `note`×2 | 同上 |
| 页级 JSON 质量 | 理论页读出孟塞尔系统的色相/明度/彩度三分；案例页 5 个色名全对；笔记页（用配色方案表现季节）产出春夏秋冬 4 条可执行规则 | `page-014.json` / `page-022.json` / `page-037.json` |
| `measure` | 印刷 p17、p24 两页色板 **5/5 命中**，实测 hex 与印刷色板肉眼一致 | `data/color-knowledge/palette-check.png` |
| `merge` | 规则 16 条 / 光照 7 条 / 配色库 4 条 / 页索引 7 页 | `data/color-knowledge/*.json` 的 `count` 字段 |
| `targets` | 2 张无纸白 tone 目标图生成成功（p21 全张 1824×2784；落色案例裁到 1824×1789，剔掉了白底图表） | `data/color-knowledge/targets/targets.json` |
| `verify` | 核对图当场抓出两处色名误读（见 7.3） | `data/color-knowledge/palette-check.png` |

**尚未验证，不许写成已完成**：

- **第一遍 55~56 页的失败率与耗时分布**——实测的 7 页全是"好读页"（白底、版式规整），
  没有覆盖整页深色插图、配色亮点密排版这类难读页型；
- `FALLBACK_MODEL` 回退路径**一次都没被触发过**（7 页全在一次成功），等于没验证；
- 4 路并行**没真跑过**（本版全程串行），"一页一个文件所以不会互相覆盖"只是推理；
- **色板实测在 32 个案例页上的命中率**——只在 p17、p24 两页验证了 5/5，
  且已见反例（p90 误命中、p63/115/127/135 无主色调排、p29 有一个色靠栅格补齐）；
  这直接决定第 8 节「≥26 条」的判据能不能达到；
- `targets` 只在 2 个案例上跑过，"前后邻居取画面占比更高者"这条规则没在 32 个案例上验证；
- 注入点在生图质量上的量化收益、`color_clauses` 对最终图的目视影响；
- 知识库合成后的**规则去重率与互斥风险**（10 个画师风格差异很大，可能产出互相矛盾的规则）；
- 真实中断下的续跑（7 页里一次中断都没有）；
- **实际 token 成本**——`call_text_model` 把响应的 `usage` 丢掉了，现在连 token 都记不下来，
  成本只能靠时长外推。

**把上面这些测掉的最小代价**（对着第一遍的 55~56 页做，不用碰 150 页）：

| 做法 | 成本 | 消掉哪几条 |
|---|---|---|
| 分层抽样 20 页：每种页型各 3~4 页，**必须含整页深色插图和配色亮点密排版** | ~25 分钟 | 失败率分布、回退路径 |
| 抽完统计 JSON 完整率 / 色板命中率 / 失败页型分布 | 0 | 同上 |
| 故意在第 5 页 kill 进程再重启 | 2 分钟 | 续跑 |
| 4 路并行各跑 5 页（共 20 页） | ~10 分钟 | 并行安全 |
| 给页级 JSON 加 `usage` 落盘，再用 20 页外推 | 改 3 行 | token 成本 |
| 20 页先合成一次，人工看有没有互斥规则 | 0 | 去重/互斥 |

### 0.2 施工第一步：先看真实接口，别看文档猜

文档里的命令在写的时候都对过代码，但**代码会变**。所以任何一步开工前先跑：

```powershell
cd D:\code\image-maker
& "C:\Program Files\Python310\python.exe" tools\color_knowledge.py --help
& "C:\Program Files\Python310\python.exe" tools\color_knowledge.py extract --help
& "C:\Program Files\Python310\python.exe" tools\color_knowledge.py analyze --help
& "C:\Program Files\Python310\python.exe" tools\color_knowledge.py measure --help
& "C:\Program Files\Python310\python.exe" tools\color_knowledge.py merge   --help
& "C:\Program Files\Python310\python.exe" tools\color_knowledge.py map     --help
& "C:\Program Files\Python310\python.exe" tools\color_knowledge.py targets --help
& "C:\Program Files\Python310\python.exe" tools\color_knowledge.py verify  --help
& "C:\Program Files\Python310\python.exe" tools\color_knowledge.py styles  --help
```

**帮助输出与本文件冲突时，以帮助输出为准，并把本文件改过来。**

---

## 1. 现实核查：这份 PDF 到底是什么

### 1.1 纯扫描件，没有文字层

对 `15659159.pdf`（73.3 MB，PDF-1.3）做对象级解析的实测结果：

| 探测项 | 结果 | 含义 |
|---|---|---|
| `/Font` 出现次数 | **0** | 没有任何字体对象 → **没有文字层**，`pdftotext` 类工具取不到一个字 |
| `/Type /Page` | 150 | 150 页 |
| `/Image` + `/DCTDecode` | 150 + 150 | 每页正好一张 JPEG |
| `/FlateDecode` / `/JPXDecode` | 0 / 0 | 没有别的压缩流 |
| 页图尺寸 | 1824 × 2784 | 约 220 dpi 的 A4 扫描 |

**结论**：整条链必须走「页图 → 视觉模型」。原计划书按"读文本"估的成本模型不成立。
**复现**：`python -c "..."` 见第 5 节注释里的对象解析逻辑，或直接看 `utils/color_knowledge.extract_pdf_pages`。

### 1.2 页码换算：印刷页码 = PDF 页序号 − 5

书前 5 页是封面/CIP/目录，正文与 PDF 序号差 5。

**验证方式（可复现）**：把 PDF 第 1~40 页的页脚条带裁下来拼成一张长图交给视觉模型逐条读页码，
再用目录页里 3 条独立条目核对：如何看色彩 p.8 → PDF 13、配色的基础 p.11 → PDF 16、给米 p.15 → PDF 20。
三处全部对上，偏移量固定为 5。

**这条写死进工具**（`analyze --offset 5`），不许让模型自己猜页码：
实测无页码的目录页（PDF 12）模型会"读"出一个 7。工具的处理是
`printed_page = pdf_page − offset` 为唯一真值，模型读数另存 `printed_page_read` 只用于自检，
两者不一致写进 `warnings`。

### 1.3 全书结构

| PDF 页 | 印刷页 | 内容 | 页型 |
|---|---|---|---|
| 1–2 | — | 封面（含腰封） | `front_matter` |
| 3, 7 | — | CIP 版权页 | `front_matter` |
| 4–5, 11–12 | — | 目录（两段） | `front_matter` |
| 6 | — | 扉页 | `front_matter` |
| 8 | — | **本书的阅读方法**（案例版式的说明书） | `front_matter` |
| 9 | — | **图表的解说**（色相平衡/色调平衡怎么读） | `front_matter` |
| 10 | — | 本书的特征 | `front_matter` |
| 13–19 | 8–14 | **理论**：如何看色彩 / 配色的基础 / 蒙塞尔系统 / NCD 色相环 | `theory` |
| 20–148 | 15–143 | **十个画师章节** + 每篇作品的配色拆解 + 「笔记」页 | 六种页型，见附录 B |
| 149 | 144 | 插画师的赠言 | `other` |
| 150 | — | 封底 | `front_matter` |

> **关于作品数量**：附录 A 的清单是**逐条数目录页（PDF 4/5/11/12）数出来的**，得到
> 33 篇作品 + 16 个笔记页 + 7 页理论 = 56 页。这个数字是**推算值**，
> 以 `analyze` 跑完后 `map` 输出的实际 `case_analysis` 页数为准；
> 两者不一致时以 `map` 为准，并回头修附录 A。

完整目录见附录 A。

### 1.4 每篇作品是固定版式，这才是真正好用的地方

书里第 8 页「本书的阅读方法」和第 9 页「图表的解说」自己把版式讲清楚了，
每篇作品分析页固定包含七个板块：

1. **主色调** —— 一排 **5 个圆面色板**，每个下面印中文传统色名（蓝紫色 / 桃红色 / 蔷薇色…）
2. **所用色彩的色相和色调范围** —— 色相 × 色调散点范围图
3. **色相平衡** —— 有彩色/无彩色比例条
4. **色调平衡** —— 4 类色调占比饼图
5. **配色亮点 A/B/C/D…** —— 逐块分析，每块一张局部放大 + 一段说明
6. **笔记** —— 从色彩心理学/造型心理学角度解读
7. **绘画过程中的配色方案要点** —— 草稿到成稿的步骤图，每步带一小排色板

**这个版式是本任务的杠杆**：色板是规整的等距圆点 → 可几何检测（第 7 节）；
文字按板块切分 → 可整页喂给视觉模型（第 6 节）。

### 1.5 原计划书里哪些假设不成立

| 原计划书 | 现实 |
|---|---|
| 「读取 PDF 中的文字和图片内容」 | 没有文字层，只有页图 |
| 「不要进行普通总结，提取美术规则」 | 方向对，但必须先有一遍**逐页事实提取**，否则模型只会给你一段读后感 |
| 新建 `AI-Art-Color-Assistant/`（`knowledge/` `prompt_engine/` `postprocess/` `main.py`） | **与 `PROJECT_REQUIREMENTS.md` 第 1 节的目录职责直接冲突**，工具有了新的归宿（第 2 节） |
| 「用 Pillow / OpenCV / numpy 做后处理」 | 对，但本项目已有 `utils/post_process.py` 的 tone 工序，不要另起一套 |
| 「生成 LUT」 | `tone_calibrate` 是**均值级**幂律亮度匹配 + 饱和度按比缩放，**不是 LUT、不是直方图匹配**。原计划书这句是错的 |
| 第六阶段「完整系统结构」 | 删掉，改成第 9 节的**一个**注入点 |

---

## 2. 与原计划书的逐条差异

| 原计划书 | 本版做法 | 为什么 |
|---|---|---|
| 第一阶段：直接读 PDF 出 `color_theory.json` | 拆 5 道工序：抽页 → 视觉提取 → 色板实测 → 合成 → 目标图 | 150 页一次喂模型会超上下文；逐页落盘才能续跑、才能审计某条规则来自哪一页 |
| 第二阶段：「分析 PDF 中所有配色案例图片」 | 改成「读案例页的**色板**，并实测色值」 | 案例图本身是插画，模型只能给印象；印在旁边的色板和色名才是作者的结论 |
| 第三阶段：`prompt_engine/analyzer.py` | 不新建引擎。产出一个 CLI 开关 `--color <slug>`，把条款挂进既有通道 | `utils/analysis_gen.build_first_pass_request` 已经是「首图请求的唯一组装点」（GUI 与 CLI 共用），再造一个引擎就是双份真值 |
| 第四阶段：`prompt_builder.py` | 不新建。落 `data/color-knowledge/` + `prompts/color-knowledge/*.md` | 同上 |
| 第五阶段：`postprocess/` 目录 | 不新建。接 `utils/post_process.py` 的 tone 工序 | 同一套参数已被分析 Tab 与 gpt-image-2 Tab 共用，另起一套会让两条链路分叉 |
| 第六阶段：`AI-Art-Color-Assistant/` | 全删 | 违反目录职责（`PROJECT_REQUIREMENTS.md` 第 1 节） |
| 「明天开始时建议顺序」 | 改成第 12 节的里程碑 | 加了人工验收与审阅闭环 |

---

## 3. 目标与非目标

### 3.1 目标

1. 第一遍 56 页（理论 + 笔记 + 案例首页）全部产出**完整**页级 JSON；案例页里**至少 80%** 测到 5 个实测色值。
2. 产出四份知识库 JSON，每条规则可追到「哪一页、哪个板块、哪个页型」。
3. 每个案例产出**一张无纸白无文字的 tone 目标图**。
4. 完成**一个**注入点的端到端闭环（第 9 节），并有固定条件下的在线对照记录。
5. 结论写进 `AGENTS.md` 的一条 `>` 摘要，附证据文档路径。

### 3.2 非目标

- **不做 GUI**。这是离线批处理，跑一次出一份知识库，不进 `app.py` 的 Tab。
- **不做「输入一句话自动出配色方案」的实时模块**。项目缺的是可复用的**条款与色调目标**，
  不是再叠一层 LLM 调用；而且它会绕过 `build_first_pass_request` 这个唯一组装点。
- **不训练/微调模型**。
- **不做逐像素 LUT / 直方图匹配**（`tone` 是均值级；硬做等于重写 `post_process.py`）。
- **不把知识库塞进 `gpt_image_prompt`**（第 9 节 D，明确列为不推荐先做）。
- **不为每个案例注册一条画风条目**（实测不成立，见第 9 节 A 的说明）。

---

## 4. 阶段 0：环境与接口

### 4.1 环境（照抄，别改）

- 唯一 Python：`C:\Program Files\Python310\python.exe`（3.10.11）。**不要新建虚拟环境。**
- 已装：`requests` / `numpy` / `opencv-python` / `pillow` / `PyQt5` / `PyQt6`。
- **没有装 PDF 库，也不许装**（`PROJECT_REQUIREMENTS.md` 第 2 节）。所以抽页走纯 Python：
  PDF 1.3 的每页 `/XObject` 指着一个 `/DCTDecode`（JPEG）流，把那段字节原样切出来即可——
  不重新编码、不丢画质。见 `utils/color_knowledge.extract_pdf_pages`。
- 后台长任务**必须 `python -u`**，否则看不到进度。
- **不要给 python 命令套管道或重定向**（`| Select-Object` / `> out.txt 2>&1`），
  沙箱会拒绝这种 spawn（表现为 `Access is denied`），而且每次都会弹一次授权窗，纯属浪费。

### 4.2 文本 / 视觉接口

- 端点：`https://new.aigc2d.com/v1`（`conf/config.json` 顶层 `base_url`）。
- 密钥：`IMAGE_MAKER_TEXT_API_KEY`，写在仓库根 `.env`，由 `utils/env_loader` 装载，
  `api_backend.resolve_text_api_key` 解析。**日志与界面只允许出现变量名，不许出现密钥值。**
- 调用封装直接用现成的（**不要自己拼 POST**，密钥解析/base_url 归一化/超时口径都在它里面）：

  ```python
  utils.analysis_gpt_prompt.call_text_model(base_url, api_key, model, system_prompt, user_prompt,
                                            timeout=180, max_tokens=4000, image_path="", image_paths=None)
  ```

- 模型名实测（`GET /v1/models`，共 419 个）：

| 用途 | 模型 | 实测 |
|---|---|---|
| 逐页视觉提取（默认） | `gpt-6-sol` | ✅ 7/7 页成功，JSON 结构稳定 |
| 视觉提取自动回退（第 2 次尝试） | `gpt-5.6-luna`（= 项目分析链路默认模型） | 已在 `FALLBACK_MODEL` 里配好 |
| 纯文本审阅 | `gpt-6-sol` / `gpt-6.1-sol` | ✅ 两个都通（`GET /v1/models` 里都在） |
| 更省成本的可选项 | `gemini-2.5-flash` | 项目里已在用，位置见下 |

> **模型名注意**：接口里**没有** `gpt6-sol`，正确写法是 `gpt-6-sol`。

> **`gemini-2.5-flash` 的既有使用位置**（供想降成本时参考，不是本流程的默认）：
> `modules/image_analysis/analysis_pipeline.py` 的 `analyze_image_step1`；
> `modules/fashion_collection/collection_service.py` 的 `_analyze_dress_colors_via_llm`。
> 换模型前先在 3~5 页上做质量抽查（色名准确率、`rules` 可执行率）。

### 4.3 成本与时长（实测）

| 操作 | 实测 | 要处理多少页 |
|---|---|---|
| `extract` 抽页 | 本地，秒级 | **全部 150 页**（已完成，0 跳过） |
| `measure` 色板实测（OpenCV，本地零成本） | 约 0.5~2 秒/页 | 全部 150 页也就 1~5 分钟，**顺手全跑**；真正有用的是其中 32 个案例页 |
| `targets` tone 目标图 | 每案例 1 次裁剪，本地 | 32 个案例页 + 它们的邻居插图页 |
| **`analyze` 视觉提取（唯一花钱的工序）** | **7 页共 8 分 11 秒 → 平均 70 秒/页**（区间 18~120 秒，`gpt-6-sol`，1824×2784，`max_completion_tokens=16000`） | **第一遍只有 55~56 页**，不是 150 页 |

> **「全量 150 页」不是一个要做的任务。** 这里只有一件事是全量的：抽页（本地，已完成）。
> 花钱的视觉提取按第 6 节的白名单只跑 55~56 页。
> 第二遍（只补配色亮点页，约 35 页）是**可选项且默认不做**——理由见 6.0。
> 若真要做：串行约 40 分钟 / 4 路并行约 12 分钟。

**并行做法**（安全）：`analyze` 是「一页一个文件」，按页段拆成 N 个后台进程即可。

```powershell
& "C:\Program Files\Python310\python.exe" -u tools\color_knowledge.py analyze --only 13-19
& "C:\Program Files\Python310\python.exe" -u tools\color_knowledge.py analyze --only 20-60
& "C:\Program Files\Python310\python.exe" -u tools\color_knowledge.py analyze --only 61-105
& "C:\Program Files\Python310\python.exe" -u tools\color_knowledge.py analyze --only 106-150
```

每个进程独立读 manifest、独立写自己的页 JSON；完整 JSON 自动跳过，中断了重跑就接着来。
**不要同时跑两个覆盖同一页段的进程**（会互相覆盖同名的页 JSON）。

### 4.4 目录归属与合规依据

`PROJECT_REQUIREMENTS.md` 第 1 节的目录职责写的是：
`cache/` = 运行缓存（`temp/` 可定期清空）；`data/` = 运行时数据（生图输出、采集素材、缓存 json）；
禁止的是**在 `data/` 放功能脚本**。因此本方案的归属是：

| 类别 | 位置 | 依据 |
|---|---|---|
| 中间产物 | `cache/temp/color-book/` | `cache/temp/` 允许清空，正合中间产物 |
| 知识库 JSON / 目标图 / 核对图 | `data/color-knowledge/` | 属"运行时数据/缓存 json"；**不含任何可执行脚本** |
| 提示词 | `prompts/color-knowledge/*.md` | 提示词一律放 `prompts/`，用 `utils/prompt_loader` 读 |
| 代码 | `utils/color_knowledge.py`（逻辑）+ `tools/color_knowledge.py`（CLI 薄壳） | `utils/` 放跨模块工具，`tools/` 放 CLI 编排 |
| 证据文档 | `docs/261003-color-improve/` | `docs/` 放文档与实验记录，**不放脚本** |

> 若后续有人主张知识库应改成 `data/<YYYYMMDD>/`：**不要照做**。
> 那是"按天生图产物"的约定，知识库是跨日期的稳态资源；
> 真要改，先把 `PROJECT_REQUIREMENTS.md` 改掉并同步所有命令与 `AGENTS.md`。

---

## 5. 阶段 1：抽页

```powershell
cd D:\code\image-maker
& "C:\Program Files\Python310\python.exe" -u tools\color_knowledge.py extract `
    --pdf docs\261003-color-improve\15659159.pdf `
    --out cache\temp\color-book\pages
```

产出：`page-001.jpg` … `page-150.jpg` + `manifest.json`。

**成功判据（三条都要满足，缺一条就算失败）**：

1. **进程退出码 = 0**。工具有意设计成：只要有任意一页被跳过（`bad_jpeg_header` /
   `not_dctdecode` / `out_of_range`），就打印告警并 **return 1**——"跳过坏图但看起来跑了 150 页"
   是最容易骗过自己的失败形态，所以不能用日志末行的字面量当判据。
2. `cache/temp/color-book/pages/` 下 `page-*.jpg` 数量 = 150。
3. `manifest.json` 每条记录都带 `file` 与 `bytes`；`skipped` 字段一条都没有。

**踩过的坑**：不要用「在 PDF 字节流里搜 `FFD8` 再找 `FFD9`」的朴素做法。
实测那样能搜出 **158** 段（含嵌入缩略图），会错页。必须走对象图：
页对象 → `/XObject /Im0` → 图像对象 → `stream…endstream` 之间原样切片。

---

## 6. 阶段 2：逐页视觉提取

```powershell
# 先小批验证（就是本版实测的那 7 页）
& "C:\Program Files\Python310\python.exe" -u tools\color_knowledge.py analyze `
    --pages-dir cache\temp\color-book\pages `
    --out cache\temp\color-book\page-json `
    --only 12,13,14,22,29,37,99 --offset 5 --model gpt-6-sol
```

**第一遍的页清单（55~56 页）**：理论 `13-19`（7 页）；案例首页 32 页；笔记页 16 页——
后两类见**附录 A 的加粗行**，逐条可核。
**第一遍不跑的页**：章节封面 / 整页插图 / 配色亮点页 / 绘画过程页 / 前置页 / 149–150，理由见 6.0。
**第二遍（可选，默认不做）**：只补配色亮点页；绘画过程页不建议跑。

### 6.0 哪些页要跑、哪些页不跑、为什么

**结论：150 页里只有 55~56 页值得花钱做视觉提取。** 逐页的取值理由：

| PDF 页 | 页数 | `analyze`？ | 理由 |
|---|---|---|---|
| 1–12 前置 | 12 | ❌ | 封面/CIP/目录/阅读方法/图表的解说。目录已人工读完（附录 A 就是这么来的），阅读方法页只是版式说明书 |
| 13–19 理论 | 7 | ✅ **第一遍** | 孟塞尔系统 / 色调 / NCD 色相环 / 配色的基础 —— 规则的源头 |
| 章节封面 | 10 | ❌ | 整页就一个画师名 |
| 整页插图 | ~33 | ❌ | 模型读了也没用，它的用途是当 `targets` 的**素材**（本地裁剪即可，不用调模型） |
| **案例首页** | **32** | ✅ **第一遍** | 主色调 5 色板 + 色相/色调平衡 —— **全书唯一有色板可实测的页型** |
| **笔记页** | **16** | ✅ **第一遍** | 信息密度最高，一页直接出 2~4 条可执行规则 |
| 配色亮点页 | ~35 | ⏸ 第二遍 | A/B/C/D 的分析文本与首页的 `hue_balance` / `tone_balance` 高度重复，去重后新增有限 |
| 绘画过程页 | ~35 | ❌ **不建议** | "草稿→成稿怎么画"的步骤图。本项目要的是**配色与色调**，画法步骤对生成提示词几乎没有落点 |
| 149–150 | 2 | ❌ | 赠言 / 封底 |

**第一遍 = 7 + 32 + 16 = 55~56 页。**

**为什么不顺手把 150 页全跑了**：

1. 绘画过程页那 35 页的边际价值接近零——`process_points` 到目前为止**没有任何下游消费者**，
   它只是给人看的辅助字段。为它多花 40 分钟和对应 token 不划算。
2. 配色亮点页那 35 页会产出大量与首页重复的 `rules`，被 `concept` 去重压掉之后，
   净新增可能只有十几条。
3. 更重要的：**第一遍跑完就能判断知识库够不够用**。判断标准是
   「案例页是否都拿到了实测色板」+「`prompt_fragments` 能不能拼出一条像样的配色条款」，
   这两条与总页数无关。先跑 56 页、先合成、先看质量，再决定第二遍——这是成本最低的顺序。

**第一遍跑完之后，什么情况下才值得做第二遍**：第一遍的在线对照（第 9.2 节）**确认有收益**，
但`color_theory.json` 的规则覆盖不到某些题材（比如书里专门讲"味道/音乐/季节"的笔记页规则被去重吃掉了），
这时才回头补第二遍。

### 6.1 提示词契约

- system：`prompts/color-knowledge/page-extract-system.md`
- user：`prompts/color-knowledge/page-extract-user.md`
  （占位符 `{pdf_page}` `{printed_page}` `{chapter_hint}` `{page_types}`，
  由 `utils/prompt_loader.render_prompt_file` 做纯 `str.replace` 替换）

提示词里七条硬性规定，改提示词时**不要删**（每条都对应一个坑）：

1. 只写这一页真实看得到的内容，版块不存在就 `null`/`[]`；
2. **不要编 hex**（色值由第 7 节实测）；
3. 版块标题、作品名、画师名、故事主题逐字照抄；
4. `rules` 是唯一允许把理论翻译成生图语言的地方：必须有触发条件、必须可执行；
5. `prompt_keywords` 用英文逗号短语，禁 `masterpiece`/`8k`/`best quality`，不写具体角色和画风名；
6. `post_process_strategy` 只允许确定性操作（饱和度、色相、暗部染色、高光滚降、分离色调、LUT 家族、曲线）；
7. 读不清就跳过并记进 `unreadable`，不许糊猜。

### 6.2 输出 schema（页级 JSON）

```jsonc
{
  "pdf_page": 22, "printed_page": 17, "printed_page_read": 17,
  "page_type": "case_analysis",        // 枚举见 utils/color_knowledge.PAGE_TYPES
  "chapter": "给米（Gemi）",            // 以 DEFAULT_CHAPTER_MAP 查表为准，模型读数不符会写 warnings
  "artist": "给米",
  "work_title": "带上你的文字", "story_theme": "讲述绚丽的夕阳和旅行的故事",
  "title": "带上你的文字", "headings": ["主色调", "所用色彩的色相和色调范围", "色相平衡", "色调平衡"],
  "summary_zh": "…",

  // ↓ v1 核心字段：下游真的会读的（第一遍必产）
  "main_palette": [                    // 只有色名，hex 由第 7 节贴回来
    {"order": 1, "name_zh": "蓝紫色", "name_en": "blue violet",
     "role": null, "share_hint": "大面积夜空与海面", "hex": "#2e2f5d",
     "hsv": [233, 51, 36], "hex_source": "measured", "swatch_xy": [152, 974],
     "needs_review": false}
  ],
  "color_relations": ["analogous", "warm-cool contrast"],
  "rules": [{
    "concept": "continuous_hue_gradient", "explanation": "…",
    "applies_to": "palette",   // palette | lighting | postprocess | mood | composition | atmosphere
    "trigger": "需要表现夕阳、自然景象或具有旅行感的连续空间氛围时使用",
    "prompt_keywords": ["continuous blue-violet to reddish-brown hue gradient"],
    "negative_keywords": ["random hue jumps"],
    "color_strategy": "…", "post_process_strategy": "…"
  }],
  "prompt_fragments": ["blue-violet to reddish-brown continuous hue gradient, purple-red dominant palette"],
  "unreadable": [],
  "warnings": [],

  // ↓ 辅助字段：给人看、给第二遍用；下游不读也不影响
  "highlight_points": [{"id": "A", "title": "…", "body": "…"}],
  "note_points": [{"title": "…", "body": "…"}],
  "process_points": ["…"],
  "hue_tone_range": "…", "hue_balance": "…", "tone_balance": "…",

  // ↓ 由工具写入
  "swatches_measured": [], "palette_row_measured": [], "_model": "gpt-6-sol"
}
```

**`prompt_fragments` 是全场最有用的字段**：这一页知识压成的一行英文短语串，
可直接粘提示词，也可派生条款。`main_palette` 在 `merge` 之前**没有 hex**，这是故意的。

**关于字段数量**：`highlight_points` / `note_points` / `process_points` / `hue_*` 属于辅助字段，
第一遍允许为空且**不设"每页至少几条"的门槛**。
一个理论页读完确实没有可执行规则时，允许 `rules: []`，但必须同时写上 `unreadable` 或
在 `summary_zh` 里说明这一页是什么——**空值要给理由，不许硬凑**。

### 6.3 缓存与续跑规则

- 一页一个文件 `page-NNN.json`。
- **跳过条件**：文件存在 **且** 通过完整性校验
  （`utils.color_knowledge.is_complete_page_json`：能解析成 dict、含
  `pdf_page`/`page_type`/`printed_page` 三个键、且 `page_type` 在 `PAGE_TYPES` 枚举内）。
  校验不过的会被**删除并重跑**，日志会打一行 `已有文件但字段不全，重跑`。
  这一条是为了防止"模型超时留下半截文件 → 下一轮被当成已完成跳过 → 知识库默默缺页"。
- 失败重试：第 1 次用 `--model` 指定模型，第 2 次自动换 `FALLBACK_MODEL`（`gpt-5.6-luna`）；
  两次都失败就跳过该页、打印原因，**不写半成品 JSON**。
- 收尾核对：`map` 输出的行数必须等于目标页数；缺页单独补跑。

---

## 7. 阶段 3：色板实测（本版相对原计划书最实质的改动）

```powershell
& "C:\Program Files\Python310\python.exe" -u tools\color_knowledge.py measure `
    --pages-dir cache\temp\color-book\pages `
    --out cache\temp\color-book\swatches
```

`measure` 只做**几何筛色板 + 量平均色**，不需要页级 JSON，可以和 `analyze` 并行跑。

### 7.1 为什么必须实测

书里的色板是印刷出来的圆点，颜色经过 CMYK 网点、纸张、扫描仪三重失真。
让视觉模型"看"出色值就是让它编；而色板形状极其规整——**白底上一排等距实心圆**——
几何检测比模型靠谱一个数量级。实测结果：

`pdf p22 / 印刷 p17 / 带上你的文字`（复现：`measure --only 22` 后看 `swatches/page-022.json`）

| 序 | 印刷色名 | 实测 hex | 来源 |
|---|---|---|---|
| 1 | 蓝紫色 | `#2e2f5d` | 轮廓检测 |
| 2 | 桃红色 | `#d74b7e` | 轮廓检测 |
| 3 | 粉红色 | `#fbb987` | 栅格补齐 |
| 4 | 蔷薇色 | `#e64b3f` | 轮廓检测 |
| 5 | 红桦色（模型读成"红棕色"） | `#4b2022` | 轮廓检测 |

`pdf p29 / 印刷 p24 / "落色"的日子`：`#020202` / `#3b6056` / `#c9a544` / `#923f23` / `#2b1006`
（第一个确实是纯黑，核对图确认无误）。

**已验证范围仅限这两页 5/5 命中**；其余页面的命中率、误检率、栅格补齐的正确率都还没有统计，
第 7.3 节的人工核对就是为这个缺口存在的。

### 7.2 算法（`utils/color_knowledge.py`）

| 步骤 | 函数 | 做法 |
|---|---|---|
| 1. 轮廓检测 | `measure_swatches` | HSV 取 `S ≥ 40 且 30 ≤ V ≤ 250` → 形态学开闭 → `findContours` → `4πA/P²` 圆度 + 半径 + 面积比三重过滤 |
| 2. 成排 | `group_swatch_rows` | 按 y 聚成排（容差 40px），排内按 x 排序 |
| 3. 判等距栅格 | `_row_unit` | 以最小间距为基准，其余间距须接近它的整数倍（容差 22%）。**一排 5 个缺 1 个时间距是 2 倍，只有整数倍判据认得出** |
| 4. 选主色调排 | `pick_palette_row` | 恰好 5 个圆、半径彼此差 ≤15%、间距是等距栅格 → 取最靠上的一排 |
| 5. 栅格补齐 | `_complete_row` | 只认到 3~4 个时，把栅格往两边各延一格，滑窗取「命中已知圆最多、其次最靠左」的 5 格；空格位按半径 55% 采样圆盘，要求「与这一行纸白差 ≥22」且「圆盘内几乎不含纸白（`paper_frac ≤ 0.25`）」 |
| 6. 贴回 | `attach_swatches` | 按位置顺序把实测色值写进 `main_palette[i].hex`，并记 `swatch_xy` / `hex_source` / `needs_review` |

两个反直觉但关键的设计选择，改代码时别"优化"掉：

- **用 `paper_frac` 而不是标准差判"是不是色板"**：扫描网点噪声让浅色色板的
  std 高达 60+，std 完全区分不了"色板"和"白纸"；"圆盘里有没有纸白"才是决定性判据。
- **至少认到 3 个真圆才允许补齐**：只认到 2 个多半是插画上的巧合（实测第 90 页会误命中）。

**顺序对齐的诚实处理**：色名（模型读）与色值（OpenCV 量）按**位置顺序**对齐。
`attach_swatches` 会记 `swatch_xy` 与 `hex_source`；当色名条数与实测圆点数不一致，
或某个色值是栅格补齐出来的，对应条目会被标 `needs_review: true`。
**标记只提示，不自动修正**——位置到底对不对，只能靠 7.3 的人眼。

### 7.3 人工核对（**这一步不许省**）

```powershell
& "C:\Program Files\Python310\python.exe" -u tools\color_knowledge.py verify `
    --page-json cache\temp\color-book\page-json `
    --pages-dir cache\temp\color-book\pages `
    --out data\color-knowledge\palette-check.png
```

核对图每行 = 一页：上面是页面上色板那一排的原图裁剪，下面是实测色值复原的色块 + hex + 色名。
**扫一眼就能发现"补齐补歪了"或"色名与色值对不上"。**

实测这一关当场抓出两处：

- p22：模型把印刷的「**红桦色**」读成「红棕色」；
- p29：模型把「**丁子色**」读成「丁香色」。

色值全对，只有色名要人工改。**这就是这一步不能省的理由**——
如果跳过它直接落条微调，这两处错会一路带进生图条款里，而且很难在成图上归因。

处理办法：在 `palette_library.json` 里直接把错的名字改掉；
如果某个色值本身可疑，把该条目 `hex_source` 改成 `rejected`，第 9 节的消费端会跳过它。

---

## 8. 阶段 4：合成知识库

```powershell
& "C:\Program Files\Python310\python.exe" -u tools\color_knowledge.py merge `
    --page-json cache\temp\color-book\page-json `
    --swatches cache\temp\color-book\swatches `
    --pages-dir cache\temp\color-book\pages `
    --out data\color-knowledge
```

`merge` 做两件事：把实测色值贴回页级 JSON；再合成四份文件。

| 文件 | 顶层 | 关键内容 |
|---|---|---|
| `color_theory.json` | `{"rules": [...], "count": N}` | 每条 = 页级 `rules[]` 一条 + `source{pdf_page, printed_page, page_type, work_title, chapter}`；按 `concept` 去重（首见优先） |
| `palette_library.json` | `{"palettes": [...], "count": N}` | 一篇作品一条：`main_palette`（含实测 hex/hsv）、`hue_tone_range`、`hue_balance`、`tone_balance`、`color_relations`、`highlight_points`、`process_points` |
| `lighting_rules.json` | `{"rules": [...], "count": N}` | `color_theory` 里 `applies_to ∈ {lighting, mood, atmosphere}` 的子集 |
| `knowledge_index.json` | `{"pages": [...], "count": N}` | 逐页 页型/标题/章节，审计用 |

**验收判据（不设规则条数门槛）**：

1. 第一遍 56 页**全部有完整页级 JSON**（用 `is_complete_page_json` 逐份校验，缺一份就补跑）。
2. `palette_library.json` 里**带实测 hex 的案例 ≥ 26 条**（按附录 A 的 33 篇 × 80%）。
3. `color_theory.json` 的 `count` **只表示实际写入的条数**，不设「每页至少一条」的下限——
   硬门槛会逼着后面的人往空页里塞规则，那正是第 4 条铁律禁止的事。
4. 每条进入知识库的规则/色板都保留 `pdf_page` / `printed_page` / `page_type` / `article` 来源。
5. `lighting_rules.json` 的条数 ≤ `color_theory.json` 的条数（它是子集过滤器，不该更多）。

### 8.1 tone 目标图（不要跳过这一步）

```powershell
& "C:\Program Files\Python310\python.exe" -u tools\color_knowledge.py targets `
    --page-json cache\temp\color-book\page-json `
    --pages-dir cache\temp\color-book\pages `
    --out data\color-knowledge\targets
```

产出 `data/color-knowledge/targets/<slug>.png` + `targets.json`。

**为什么不能用"色卡图"当 tone 目标**（这是 v1 的错误做法，被审阅抓出来）：
`tone_calibrate` 只读参考图的 **LAB 明度均值 + HSV 饱和度均值**。
带白底和文字的色卡，纸白会把目标明度往上拉、饱和度往下拉，等于给了一道"偏白偏灰"的假目标。
所以要的是**一幅真的画**，而不是一幅"关于配色的图"。

**做法**：

- 拿案例首页的**前后邻居页**里「画面占比更高」的那张——书里「整页插图 + 分析页」的前后顺序不固定
  （实测带上你的文字是插在前、落色是插在后），不能写死 `pdf_page − 1`；
- 用最大连通域裁剪（`crop_to_art`）剔掉纸白边、页码、书眉，以及挂在作品页下方的白底小图表；
- `targets.json` 记录 `source_page` 与 `art_coverage` 供核对。

实测：`color-…-p17` ← p21（画面占比 96%，未裁）；`color-…-p24` ← p30
（裁到 1824×1789，把下方两块白底图表剔掉了）。

**验收**：随手打开 `targets/` 里 3 张图，确认画面里没有大片纸白、没有文字。

---

## 9. 阶段 5：注入生图链路（**只做一个，做完再考虑下一个**）

### 9.0 先排除掉一条看起来很美、实际不成立的路

v1 曾打算「把每个案例注册成一条 `color-*` 画风条目」，**实测否定**：

- `utils/style_gpt.style_prompt_gpt` 对缺 `prompt_gpt` 的条目返回空串；
  `utils/styles.build_ref_gen_params` 在四种参考图模式下都返回 `('', '', [])`；
  `resolve_style_clauses` 返回 `([], 'none')`。
  也就是说，一条"只有 ref_image + color_clauses、没有 prompt_gpt"的条目**贡献为零**——
  它既不给画风文字，也不给条款。
- 给这类条目补编 `prompt_gpt` 更不行：8 个字段里只有 `Palette` 有实测依据，
  其余 7 个（Brushwork / Edges / Texture / …）书里没有信息，编出来就是伪造。
- 再退一步：就算能用，GUI 的画风下拉是**单选**，用户选了配色条目就丢了画风，等于用不了。

**结论：配色知识不能以"画风条目"的形态存在，必须以"附加档案"的形态叠加在当前画风之上。**

### 9.1 推荐做法：一个 `--color <slug>` 开关，两个效果

**效果 1：`color_clauses` 进首图请求**

知识库为每个案例派生一组英文条款（由该页 `prompt_fragments` + `rules[].prompt_keywords` 归并、
去重、限长得到），走既有的 `extra_clauses` 通道。

**效果 2：`tone.reference_path` 指向该案例的 tone 目标图**

让「色调校准」这一步把生成图压向书里那幅画的平均明度与饱和度。

**改动点（先按符号名检索，行号是 2026-10-02 的参考位置）**：

| 文件 | 位置 | 改动 |
|---|---|---|
| `tools/analysis_gpt_run.py` | `main()` 的 argparse 段（参考 :901 附近） | 加 `ap.add_argument("--color", default="", help="配色档案 slug，如 color-gemi-p17")` |
| `tools/analysis_gpt_run.py` | `tone_ref = ref if (args.tone_target == "style" and ref) else source`（参考 :1074） | 改成 `tone_ref = args.tone_reference or (ref if (args.tone_target == "style" and ref) else source)`，并新增 `--tone-reference <path>` |
| `tools/analysis_gpt_run.py` | 同处 | 当 `--color` 生效时把 `steps["tone"]["tone_target"]` 设成 `"custom"` 以外的任何非 `"style"` 值 |
| `utils/analysis_gen.py` | `build_first_pass_request(...)` 的条款拼接处（参考 :288-300） | 加一个 `color_clauses` 参数，拼进 `extra_clauses` |
| `utils/color_knowledge.py` | 新增 | `load_color_profile(slug)`：读 `palette_library.json` + `color_theory.json`，产出 `color_clauses` 与 `tone_target_path` |

**为什么 `tone_target` 要设成非 `"style"` 的值**：`utils/analysis_gen.py` 里只有
「`tone_target == "style"` 且 `style_ref_path` 是可读文件」这一条分支会**覆盖** `reference_path`。
经全仓检索确认：`utils/post_process.py` **完全不读** `tone_target`，只读 `reference_path`。
所以把 `tone_target` 设成别的值就能安全保住自定义目标图。

**`color_clauses` 的作用域必须写清楚**：

- ✅ 影响：**首图请求**（`build_first_pass_request` 是首图请求的唯一组装点，GUI 与 CLI 共用）。
- ❌ 不影响：`repaint` / `structure` / `local` / `tone` / `ink` 五道后处理工序。
  重绘走的是另一条通道（`run_pipeline(style_clauses=...)`，源头是画风条目的 `repaint_clauses`）。
  如果配色条款也要影响重绘，**那是第二个注入点**，需要单独指定注入函数、字段和回归用例，
  不许拿"首图组装点"去推断全流程生效。

**条款落盘前的自检**（写进 `load_color_profile`）：
非空、ASCII、单条 ≤160 字符、单案例 ≤6 条。
理由：`utils/style_gpt.py` 的 `CLAUSE_MAX_CHARS = 1600` 与
`utils/analysis_gpt_prompt.py` 的 `COMPOSED_WARN_CHARS = 2500` 是提示词长度护栏，
条款写太长会把画风参考图压掉（这是本项目实测过的失效模式）。

**`conf/config-styles.json` 的修改纪律（如果后续真要做条目化）**：
必须走「读旧 → 只改自己的键 → 写回」，先写 `.tmp` 再原子替换，
并且**用项目现成的 `utils/atomic_io`**（它专门处理 D 盘 `os.replace` 偶发 `WinError 5` 的重试），
不要自己写裸 `os.replace`。`conf/config-styles.json` 没有备份，写坏一次要重配 34 个画风。

### 9.2 在线对照协议（固定这些，否则结论无效）

| 项 | 固定为 |
|---|---|
| 源图 | 一张固定的分析产物 JSON（例如 `data/20261002/20261002-195354-46fd0fbc-紫薔薇黒髪姫.json`） |
| 画风 | 固定一条（例如 `kishida-mel-style`），**两组都用同一条** |
| 模型 / 尺寸 / 画质 | 同一次实验内完全一致 |
| 提示词 | 除"是否带 `--color`"外完全一致 |
| 输出目录 | 每组一个独立 `--output-dir`，文件保留 |
| 组数 | 每组 ≥2 张（模型有随机性） |

**要记录的东西**：完整命令行、源图路径、生成文件名、耗时、
`tests/calc_style_similarity.py` 的 **HSV 直方图相似度**（配色主指标）、
`tests/style_render_metrics.py` 的 `render_score`（**辅助指标**）、人工目视结论。

> ⚠️ `render_score` 由明度层次 35% + 线条 25% + 边缘 25% + 负空间 15% 组成，
> **完全不含色彩项**。拿它当配色收益的主证据是错的——配色走
> `tests/calc_style_similarity.py` 的 HSV 直方图那一路。

**判定标准**：只有出现**可重复**的差异（≥2 组同向），才允许说"这个注入点有收益"。
单次差异一律记「未确认」。

---

## 10. 阶段 6：验证与回归

| 检查 | 命令（在 `D:\code\image-maker` 下执行） | 通过标准 |
|---|---|---|
| CLI 接口没变 | `& "C:\Program Files\Python310\python.exe" tools\color_knowledge.py --help` | 8 个子命令都在：`extract` `analyze` `measure` `merge` `map` `targets` `verify` `styles` |
| 抽页完整 | `... extract ...` | **退出码 0** 且 150 个 `page-NNN.jpg` |
| 页级 JSON 完整 | `... map --pages-dir cache\temp\color-book\pages --page-json cache\temp\color-book\page-json` | 目标页数 = 56，且每行都有 `page_type` |
| 色值可核对 | 打开 `data\color-knowledge\palette-check.png` | 色名与色块一一对得上 |
| 目标图干净 | 打开 `data\color-knowledge\targets\` 里 3 张 | 无大片纸白、无文字 |
| 知识库 schema | 看四份 JSON 的 `count` 与 `source` 字段 | 满足第 8 节 5 条判据 |
| 生产链路没坏 | `& "C:\Program Files\Python310\python.exe" -m pytest -q -p no:cacheprovider` | **退出码 0**、终端输出无失败用例；**不要套管道或重定向** |
| 条款没超预算 | `... tools\analysis_gpt_run.py --json <产物> --dry-run` | 打印的提示词总长 < 2000 字符 |

**纪律**：任何改动 `utils/*.py` 之后**必须重启 app**（GUI 进程里是旧模块）。
没在线复现过的结论，只能写「未验证」。

---

## 11. 阶段 7：审阅闭环（有牙齿才留）

本项目的既有做法是让 sol 系列模型做技术审阅。2026-10-02 给 `tools/sol_review.py` 加了
**纯文本审阅模式**（原来只有图片模式），本文件就是这条链的第一份产物：

```powershell
& "C:\Program Files\Python310\python.exe" tools\sol_review.py `
    --text docs\261003-color-improve\AI_Color_Knowledge_Extraction_Workflow-v2.md `
    --brief prompts\color-knowledge\workflow-review.md `
    --model gpt-6-sol `
    --out docs\261003-color-improve\review-gpt6-sol-20261002.md
```

- 模型名 `gpt-6-sol`（不是 `gpt6-sol`）；模型空回复时加大 `--max-tokens`。
- 审阅简报（`prompts/color-knowledge/workflow-review.md`）自带一张《项目事实卡》，
  并明确要求：事实卡没写的必须写「无法验证」，不许替作者圆场。

**审阅要有牙齿**，否则只是又一次模型调用。规定：

1. 审阅输出必须被整理成一张**结构化问题清单**（问题 / 涉及章节 / 责任文件 / 状态 / 复验命令），
   贴进证据文档；
2. 每条问题必须有终态：`已修`（附改后原文）/ `不采纳`（附理由与反证）/ `待验证`（附验证方案）；
3. **问题清单没有清空之前，不许写进 `AGENTS.md`**；
4. 若某轮审阅产不出可执行的问题清单（只有泛泛而谈），
   则该轮按无效处理，退回人工检查清单，**不把审阅模型输出当验收依据**。

---

## 12. 里程碑与排期

| # | 里程碑 | 产出 | 判据 |
|---|---|---|---|
| M0 | CLI 接口确认 + 抽页 | 150 页图 + `manifest.json` | 退出码 0、页数 150、无 skipped |
| M1 | 理论 + 笔记页（7 + 16 = 23 页）提取 | 23 份页级 JSON | 每份通过完整性校验 |
| M2 | 案例首页（33 篇作品，伯劳鸟与鸟同页 → 实占 32 页）提取 | 32 份页级 JSON | 同上；可与 M1 合并成 55~56 页一批 |
| M3 | 色板实测 + 人工核对 | `palette-check.png` 通过 | ≥26 条案例有实测 hex |
| M4 | 知识库合成 + tone 目标图 | 四份 JSON + `targets/` | 第 8 节 5 条判据 |
| M5 | **一个**注入点落地 + 在线对照 | `--color` 生效 + 对照记录 | 第 9.2 节协议齐备、差异可重复 |
| M6 | 审阅闭环 + 写进 `AGENTS.md` | 一条 `>` 摘要 + 证据文档 | 问题清单清空 |
| M7（可选） | 第二遍：只补配色亮点页（约 35 页） | 页级 JSON | 只在 M5 确认有收益、且规则覆盖不够时再做 |

---

## 13. 纪律清单（对齐 `PROJECT_REQUIREMENTS.md`）

1. 根目录不新增任何 `.py` / `.json` / 图片 / 日志。
2. 不新建一次性脚本；同族功能合并进 `tools/color_knowledge.py` 的子命令。
3. 提示词一律放 `prompts/`，不许把长 prompt 写死在 `.py`。
4. 配置写回走「读旧 → 只改自己的键 → 写回」，用 `utils/atomic_io` 做原子替换。
5. 密钥只走 `.env` / 环境变量，日志与界面只允许出现变量名。
6. 产物落 `data/color-knowledge/`，中间产物落 `cache/temp/`；
   **不许往根目录或 `data/<日期>/` 扔中间文件**。
7. 改过 `.py` 必须重启 app；跑 pytest **不许套管道或重定向**。
8. 不宣称未验证的结论；每条数字后面跟一个可复核的产物或命令。

---

## 附录 A：全书目录（第一遍页清单）

> 本清单是**逐条数目录页**（PDF 4/5/11/12）得到的，属推算值；
> `analyze` 跑完后以 `map` 输出的实际 `case_analysis` 页数为准。
> **加粗** = 第一遍要跑的页；`PDF 页` 就是 `analyze --only` 要填的号。

### 理论（PDF 13–19 / 印刷 8–14）

| PDF 页 | 印刷页 | 内容 |
|---|---|---|
| **13** | 8 | 如何看色彩（孟塞尔色彩系统：色相 / 明度 / 彩度） |
| **14** | 9 | 色调（Tone）· 明度·彩度·色相的定义与分级 |
| **15** | 10 | NCD 色相环系统 / PCCS 色相环与色调坐标 |
| **16–19** | 11–14 | 配色的基础（①涂色范围 ②制作底色 ③色形的排列构成 ④配色和光的原理） |

### 给米（Gemi）· PDF 20 起

| PDF 页 | 印刷页 | 内容 |
|---|---|---|
| 20–21 | 15–16 | 章节封面 / 整页插图 |
| **22** | 17 | **带上你的文字** 讲述绚丽的夕阳和旅行的故事 |
| 23–28 | 18–23 | 配色亮点 A–H / 绘画过程中的配色要点 |
| **29** | 24 | **"落色"的日子** 讲述刺激感官的色彩缤纷的故事 |
| 30–32 | 25–27 | 插图 / 亮点 / 过程 |
| **33** | 28 | **隐藏在透明的雨中** 讲述含蓄的情感故事 |
| **34** | 29 | 笔记：色彩名称、色相和色调 |
| **35** | 30 | **夏天的感觉** 讲述炎热季节里鲜活时刻的故事 |
| **36** | 31 | **与你共享的时光** 讲述色彩浓厚的秋季的故事 |
| **37** | 32 | 笔记：用配色方案表现季节 |

### 波特歌（potg）· PDF 38 起

| PDF 页 | 印刷页 | 内容 |
|---|---|---|
| **40** | 35 | **紫藤** 讲述自然与舒适的故事 |
| **41** | 36 | 笔记：自然和谐配色法 |
| **47** | 42 | **水底花圃** 讲述水底光线之美的故事 |
| **48** | 43 | 笔记：配色方案中的色彩数量 |
| **50** | 45 | **即使这样也要画下去** 讲述意志坚强的情感故事 |
| **51** | 46 | 笔记：膨胀色和收缩色 |
| **53** | 48 | 笔记：情绪和配色方案 |

### 樱田千寻（Sakurada Chihiro）· PDF 54 起

| PDF 页 | 印刷页 | 内容 |
|---|---|---|
| **55** | 50 | **射手座的糖苹果** 讲述怀旧美食的故事 |
| **59** | 54 | **极光下的甜瓜苏打水** 讲述幻想与现实相遇的故事 |
| **60** | 55 | 笔记：味道和配色方案 |
| **61** | 56 | **天空之色果酱** 讲述天空色彩变幻的故事 |
| **65** | 60 | 笔记：落日为什么是红色的？（光与色彩） |

### 亚纪（Aki）· PDF 66 起

| PDF 页 | 印刷页 | 内容 |
|---|---|---|
| **68** | 63 | **星泉的守护神** 讲述幻想世界的故事（淡调） |
| **72** | 67 | **晚宴** 讲述豪华、充满幻想的故事（暗调） |
| **73** | 68 | 笔记：插画意象和配色方案 |
| **76** | 71 | **女巫与午后** 讲述神秘午后的故事 |
| **79** | 74 | 笔记：根据面积选择配色方案 |

### 约尔·福杰（Fajyobore）· PDF 80 起

| PDF 页 | 印刷页 | 内容 |
|---|---|---|
| **82** | 77 | **爱丽丝梦游仙境** 登场！讲述充满戏剧化的故事 |
| **89** | 84 | **洗碗** 讲述精致生活一角的故事 |
| **93** | 88 | **不和** 讲述充满异国风情的故事 |

### 巴尼石门（Banishment）· PDF 94 起

| PDF 页 | 印刷页 | 内容 |
|---|---|---|
| **95** | 90 | **放晴** 讲述雨过天晴的故事 |
| **99** | 94 | 笔记：眼睛看到色彩的过程 |
| **100** | 95 | **勿忘我** 讲述另一个世界的故事 |
| **101** | 96 | **一天** 讲述通过风感受时间流逝的故事 |
| **105** | 100 | **天旋地转** 讲述一段天旋地转的奇妙经历 |

### 寺田寺（Terada Tera）· PDF 106 起

| PDF 页 | 印刷页 | 内容 |
|---|---|---|
| **107** | 102 | **专属空间** 模拟器？数字化？讲述专属世界的故事 |
| **110** | 105 | 笔记：清色和浊色的区别一 |
| **114** | 109 | **放松** 讲述休闲一刻的故事 |
| **116** | 111 | **伯劳鸟 / 鸟** 讲述异想天开的故事 / 讲述闪闪发光的故事 |

### 木野花日兰子（Konohana Hiranko）· PDF 118 起

| PDF 页 | 印刷页 | 内容 |
|---|---|---|
| **120** | 115 | **年轻的贞德** 讲述女英雄的苦难故事 |
| **126** | 121 | **想遗忘的事情** 讲述不施粉黛的情感自然流露的故事 |
| **129** | 124 | 笔记：清色和浊色的区别二 |

### 三土（REDUM）· PDF 130 起

| PDF 页 | 印刷页 | 内容 |
|---|---|---|
| **132** | 127 | **海岛之歌** 讲述琴声悠扬的海边故事 |
| **136** | 131 | **夜长姬和耳男** 讲述小说的高潮情节 |
| **137** | 132 | 笔记：音乐和色彩搭配 |

### 拉斯库（RASUKU）· PDF 138 起

| PDF 页 | 印刷页 | 内容 |
|---|---|---|
| **140** | 135 | **爱的渴望** 讲述优雅女性的故事 |
| **143** | 138 | **小憩** 讲述在宁静秋日里小憩的故事 |
| **144** | 139 | **魅惑的甜瓜味** 讲述专注于个人爱好的故事 |
| **145** | 140 | **熊猫** 讲述爱与治愈的故事 |
| **146** | 141 | 笔记：插画中的色彩搭配 |
| **147** | 142 | 笔记：根据意象选择配色方案 |
| 149 | 144 | 插画师的赠言 |
| 150 | — | 封底 |

**推算合计**：33 篇作品（伯劳鸟与鸟同页，实占 32 个分析页）+ 16 个笔记页 + 7 页理论 ≈ **55~56 页**。

---

## 附录 B：页型判定表

| `page_type` | 判据 | 后续动作 |
|---|---|---|
| `front_matter` | 封面/CIP/目录/扉页/阅读方法/图表的解说 | 只进 `knowledge_index`，不进知识库 |
| `theory` | 有「孟塞尔」「色相」「明度」「彩度」「色相环」等标题 | 规则主要来源 |
| `note` | 标题带「笔记」，讲色彩心理/五感/季节/意象 | 规则主要来源（信息密度最高） |
| `chapter_cover` | 整页画师名 | 跳过 |
| `illustration` | 整页插画、无分析版块 | 跳过（是 `targets` 的素材来源） |
| `case_analysis` | 有「主色调」+「色相平衡」+「色调平衡」 | **色板实测 + 配色库主条目 + tone 目标** |
| `case_highlight` | 有「配色亮点」+ A/B/C/D 标记块 | 规则来源（`highlight_points`），第二遍，默认不跑 |
| `case_process` | 有「绘画过程中的配色要点」+ 步骤图 | `process_points`，**无下游消费者，不建议跑** |
| `other` | 归不进去的 | 人看一眼再说 |

---

## 附录 C：术语与字段对照（原计划书 → 本版）

| 原计划书字段 | 本版字段 | 说明 |
|---|---|---|
| `concept` | `concept` | 保留，改成英文小写下划线标识 |
| `explanation` | `explanation` | 保留 |
| `scene` | `trigger` | 改名：必须是**触发条件**（什么画面/情绪下用它），不是"场景"这种模糊词 |
| `prompt_keywords` | `prompt_keywords` | 保留，约束改为：英文逗号短语、禁废话词、禁角色与画风名 |
| `negative_keywords` | `negative_keywords` | 保留 |
| `color_strategy` | `color_strategy` | 保留 |
| `post_process_strategy` | `post_process_strategy` | 保留，限定为**确定性操作** |
| — | `applies_to` | 新增：`palette`/`lighting`/`postprocess`/`mood`/`composition`/`atmosphere`，决定进哪个下游 |
| `palette_library.primary_color` | `main_palette[i].hex` | **必来自实测** |
| `secondary_color` / `accent_color` | `main_palette[i].role` | 由书里文字说明判断；判不出就是 `null`，不许硬凑 |
| `palette_library.theme` | `story_theme` | 照抄书里那句「讲述…的故事」 |
| `palette_library.postprocess` | 无该字段 | 后处理由第 8.1 节的 tone 目标图 + 第 9.1 节的 tone 注入承担 |
| `lighting_rules.json` | `lighting_rules.json` | 保留文件名，内容是 `applies_to ∈ {lighting, mood, atmosphere}` 的子集 |
| — | `prompt_fragments` | 新增：一页知识压成一行英文短语串，直接可粘进提示词 |
| — | `needs_review` / `hex_source` / `swatch_xy` | 新增：色值贴回时的可审计标记 |

---

## 附录 D：v1 → v2 的修改清单（逐条回应审阅）

| # | 审阅意见 | 本版处理 |
|---|---|---|
| 1 | 「33 篇作品」无法由事实卡推出 | 附录 A 标注为「逐条数目录页得出的推算值」，并写明以 `map` 输出为准 |
| 2 | §0.1「已跑通」缺证据 | 每条结果后面加「复核方式」列，指向具体产物文件或退出码 |
| 3 | §7 的色值/阈值/算法写成已实测结论 | 明确「已验证范围仅限 p17/p24 两页 5/5 命中」，其余命中率/误检率标注为未统计；补第 7.3 节人工核对兜底 |
| 4 | `gemini-2.5-flash`、`sol_review --text` 未确认 | 前者补上两个**具体调用文件**；后者是本版新增并已实测通过（第 11 节写明是新增） |
| 5 | `data/color-knowledge/` 的目录依据 | 第 4.4 节补上 `PROJECT_REQUIREMENTS.md` 第 1 节的条款引用与归属表，并明确"若有人主张改成日期目录，不要照做" |
| 6 | 「五步命令」缺参数 | §0.2 增加「先跑 `--help`」的前置步骤；所有示例命令改成完全显式（含 `--pages-dir` / `--out` / `--offset` / `--model`） |
| 7 | `verify` 子命令无法确认 | 写明它是本版新增（与 `targets` 一起），并给出完整参数与退出码含义 |
| 8 | 抽页成功判据缺退出码 | 判据升级为三条；工具改为「有任何页被跳过就 return 1」 |
| 9 | 续跑分不清完整/半成品 JSON | 新增 `is_complete_page_json` 校验；校验不过的**删除并重跑**（第 6.3 节） |
| 10 | pytest 命令缺解释器/工作目录 | 第 10 节统一写明工作目录与完整解释器路径，并强调禁管道/重定向 |
| 11 | 依赖行号有风险 | 第 9.1 节改为「先按符号名检索」+「行号为 2026-10-02 的参考位置」 |
| 12 | 在线对照缺固定条件 | 新增第 9.2 节完整对照协议（源图/画风/模型/尺寸/目录/组数/记录项/判定标准） |
| 13 | `count ≥ 56` 与「可空」自相矛盾 | 第 8 节判据改为「页级 JSON 完整 + 空值要给理由」，明确不设规则条数下限 |
| 14 | `atmosphere` 不在 `applies_to` 枚举 | 枚举统一加入 `atmosphere`，提示词模板同步 |
| 15 | 色板位置与色名顺序不一致时静默写入 | `attach_swatches` 新增 `needs_review` / `hex_source` / `swatch_xy`，第 7.2 节写明"标记只提示，不自动修正" |
| 16 | 色卡图当 tone 目标会被纸白稀释 | **采纳并改设计**：新增 `targets` 工序，产出无纸白无文字的整幅画面（最大连通域裁剪）；色卡只用于人工核对 |
| 17 | `tone_target="custom"` 下游未必接受 | 改为「设成任何非 `"style"` 的值」，并附上检索证据：`post_process.py` 完全不读 `tone_target`，只读 `reference_path` |
| 18 | `color_clauses` 未说明后续工序是否继承 | 第 9.1 节明确作用域：只影响首图；重绘要走 `repaint_clauses`，且那是第二个注入点 |
| 19 | 每案例注册画风条目会污染 GUI 下拉 | **实测否定该路线**（缺 `prompt_gpt` 的条目贡献为零），改为「附加档案 + `--color` 开关」（第 9.0 / 9.1 节） |
| 20 | 用 `render_score` 评价配色收益 | 主指标改为 `tests/calc_style_similarity.py` 的 HSV 直方图，`render_score` 降为辅助并注明它不含色彩项 |
| 21 | 同时推进 A/B/C 三个注入点 | 改为「只做一个，做完再考虑下一个」；非目标里明确写「不为每个案例注册画风条目」 |
| 22 | 全量 150 页分析属过度 | 第 6 节新增 **6.0「哪些页要跑、哪些页不跑」**：只有抽页是全量的（本地、已完成），花钱的视觉提取按白名单只跑 55~56 页；绘画过程页归为"不建议跑"（`process_points` 无下游消费者）；第二遍收窄到只补配色亮点页 |
| 23 | schema 字段过多、无消费者 | 第 6.2 节把字段拆成「第一遍必产（下游会读）」与「辅助（第二遍/给人看）」两档 |
| 24 | 审阅闭环没接入阻断 | 第 11 节增加「结构化问题清单 + 终态 + 清空后才写 `AGENTS.md` + 无效审阅退回人工」四条规定 |

**同时新增（审阅没提，但施工需要）**：

- `DEFAULT_CHAPTER_MAP`（画师章节查表）——不给表模型会瞎猜章节（实测 p22 被判成「理论」）；
- `utils/atomic_io` 作为配置原子写回的统一入口（审阅提的是"先 `.tmp` 再 `os.replace`"，
  本项目已有更稳的封装）；
- `crop_to_art` 的最大连通域裁剪，以及「形态学核不能超过 41」这条实测经验
  （61 会把插画和下方的白底图表粘成一块，参考图里又混进三成纸白）。
