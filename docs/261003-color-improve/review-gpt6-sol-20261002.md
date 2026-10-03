# gpt-6-sol 审阅（2026-10-02 21:43）

- 对象：`cache\temp\color-book\workflow-v1.md`
- 模型：gpt-6-sol
- tokens：{"prompt_tokens": 16543, "completion_tokens": 6051, "total_tokens": 22594, "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 1289}}

## 简报

# 角色

你是这个仓库（`D:\code\image-maker`，Python 3.10 的 AI 生图/图片分析工具集）的**技术评审**。
下面这份《AI 配色知识增强 工作指引》是给「另一个 AI 编码助手（Codex）」照着施工的说明书，
不是给你看的代码。请按第五节给的五段结构出评审意见。

# 评审纪律

1. **不要客套、不要复述、不要总结文档。** 只写问题与改法。
2. 每条意见都要**指名到具体章节/文件名/命令**，并给出**可直接抄进文档的替换文本**（中文，一两句）。
3. 你只能依据文末的《项目事实卡》判断事实性。事实卡没写的，一律写「无法验证，需作者拿证据」，
   **不要替作者圆场、不要假装确认**。
4. 判断"可执行性"的标准是：一个只有本仓库上下文、没有外部信息的助手，
   照抄文档里的命令能不能跑出东西来。缺前置条件、缺判断成功与否的标准，都算缺口。
5. 第五节「最小可行路径」是**必答项**：必须真的砍到 5 步，不许写"以上都重要"。

# 项目事实卡（ground truth，由只读代码调研得出）

## 运行环境
- 唯一 Python：系统 Python 3.10.11（`C:\Program Files\Python310\python.exe`）。**已删除 .venv**，
  禁止新建虚拟环境。已装：requests / PyQt5 / PyQt6 / pillow / numpy / opencv-python。
- **未装任何 PDF 库**（PyMuPDF / pypdf / pdfplumber 全无），按项目规定也不该 pip 装依赖。
- 跑 pytest / python 时**禁止套管道或重定向**（`| Select-Object`、`> out.txt 2>&1`），
  沙箱会拒绝这种 spawn，必须让输出直接回传。

## 目录职责（硬性）
- 根目录只放启动入口（`app.py` / `make-pic.py` / `sd-make-pic.py` / `publish_server.py`）与三份 md。
- `modules/` 业务模块；`utils/` 跨模块工具；`tools/` 无头 CLI；`prompts/` LLM 提示词模板（.md/.txt/.json）；
  `conf/` 运行时配置（gitignore）；`data/<YYYYMMDD>/` 运行时产物；`cache/temp/` 可清空的临时区；
  `docs/` 文档与实验记录（**禁止放可执行脚本**，脚本快照只能进 `docs/*/tools/`）。
- 提示词一律放 `prompts/`，用 `utils/prompt_loader.read_prompt_file(相对路径)` 读；
  `render_prompt_file(path, {key: value})` 是纯 `str.replace("{key}", value)`，支持任意键、多出来的占位符不报错。

## 文本/视觉 API
- 端点 `https://new.aigc2d.com/v1`（`conf/config.json` 顶层 `base_url`），
  密钥走环境变量 `IMAGE_MAKER_TEXT_API_KEY`（写在仓库根 `.env`，由 `utils/env_loader` 装载），
  `api_backend.resolve_text_api_key` 解析。归一化用 `utils.analysis_gpt_prompt.normalize_chat_base`。
- 调用封装 `utils.analysis_gpt_prompt.call_text_model(base_url, api_key, model, system_prompt, user_prompt,
  timeout=180, max_tokens=4000, image_path="", image_paths=None)`，内部 `POST /chat/completions`，
  附图时把图按 `utils/style_gpt.to_data_url` 转 data URL 塞进 `image_url`。
- 分析链路默认视觉模型是 `conf/config.json` 顶层 `model` = `gpt-5.6-luna`。
  实测 `GET /v1/models` 有 419 个模型，其中 sol 系：`gpt-5.6-sol` / `gpt-5.6-sol-max` /
  `gpt-5.6-sol-ultra` / `gpt-6-sol` / `gpt-6.1-sol`。实测 `gpt-6-sol`、`gpt-6.1-sol` 都能正常返回。
- 审阅工具已有：`tools/sol_review.py`（图片 + 简报 → sol 审阅 → 写 md）。

## 分析链路（现状）
- `modules/image_analysis/single_analyzer.py`：Step1 `step_1_analyze_image`（vision，prompt =
  `prompts/style-analy.md` + `prompts/single-analyzer-system.md`）→ Step2 `step_2_refine_description`
  （`refine-desc.md` + `refine-desc-system.md`）→ Step3 outfit 检查 → Step4 去照片感 → Step5 重算 pixiv 标签。
- 落盘产物顶层字段固定为：`english_description` / `short_description` / `booru-tags` / `japanese_title` /
  `chinese_title` / `pixiv_tags` / `aspect_ratio` / 一堆 `*_before_*` 快照 / `gpt_image_prompt`(≤1400) /
  `gpt_image_prompt_short`(≤500) / `source_image_path` / `task_hash` / `generation_style_name`。
  **没有任何结构化色彩字段**；颜色/光照只以自然语言存在于那几个文本字段里。
- `utils/analysis_gen.py`：`build_first_pass_request(...)` 是「gpt-image 首图请求的唯一组装点」，
  内部 `build_gpt_image_request(...)`，可注入的文本段有：画风说明 `style_text`、内容锚 `content_text`、
  参考图排除句、`extra_clauses`（渲染成 `RENDERING LANGUAGE (follow exactly):` 段）、`user_hint`。
  `pipeline_steps_from_flags(...)` 产出 `repaint` / `structure` / `local` / `tone` / `ink` 五个工序；
  `tone` 段有 `reference_path` 键，`run_gpt_image_pipeline` 里 `tone_target=="style"` 时用画风参考图回填。

## 画风系统
- `conf/config-styles.json` 每个条目的字段是并集（没有固定 schema）：`prompt` / `prompt_compressed` /
  `prompt_gpt` / `ref_image` / `enabled` / `repaint_clauses` / `generation_clauses` /
  `identity_correction_clauses` / `proportion_clauses` / `motif_clauses` / `post_adjustment` 等。
- `utils/style_gpt.py` 的 `FIELD_KEYS` = Palette / Lighting / Brushwork / Edges / Texture /
  Composition density / Detail level / Avoid，即 `prompt_gpt` 的 8 个字段；
  `validate_prompt_gpt` 要求 150 ≤ 全长 ≤ 680 字符、单字段 ≤ 160、字段齐全、**不得出现主体词**。
- `tools/convert_styles_gpt.py` 是转 `prompt_gpt` 的 CLI（`--force/--only/--check/--dry-run/--no-image`），
  `--check` 会额外报告 `repaint_clauses` 的覆盖情况（手写/派生/无）。

## 已有色彩相关能力
- `tools/style_ref_stats.py` 的 `stats_for(path)` 可对**任意图片**算：k-means 5 主色 + 占比、
  色相分布、亮度/饱和均值、白底占比、边缘密度。`to_text(name, st)` 转成给 LLM 的一行客观描述。
  这是仓库里唯一「把量化色彩数据当权威依据拼进 prompt」的先例（`convert_styles_gpt.py` 用它）。
- `utils/post_process.py` 的 tone 工序 `tone_calibrate(...)`：**均值级**的幂律亮度匹配 +
  HSV 饱和度按比缩放 + LAB 色度/肤色微调，不是直方图匹配、不生成 LUT。
  目标色从 `reference_path` 用 `reference_brightness()`（LAB L 均值）/ `reference_saturation()`（HSV S 均值）取。
- `tests/style_render_metrics.py` 的 `render_score` **不含任何色彩项**（只有明度层次 35% / 线条 25% /
  边缘 25% / 负空间 15%）。配色贴近度要用 `tests/calc_style_similarity.py` 的 HSV 直方图那一路。
- 仓库里**没有任何色彩知识库 JSON**，也没有任何碰过 PDF 的脚本。
- 2026-09-29 的一批 colour/palette 实验 prompt（`prompts/gpt-image-optimize/*colour*`、
  `*palette*`）**全部未进生产**，其中「限制主色族种类」的实验结论是**不是合适的通用解**
  （见 `docs/gpt-image-tid-style/FUZICHOCO-CONTINUITY-20260929.md`）。

## 本次任务已经实测到的事实（可以当已证实）
- 目标 PDF `docs/261003-color-improve/15659159.pdf`：73.3 MB，PDF-1.3，
  **150 个页面对象、150 个图像对象、全部是 `/DCTDecode`（JPEG），`/Font` 出现 0 次 → 没有文字层**，
  纯扫描件。页图 1824×2784。
- 印刷页码 = PDF 页序号 − 5（用 3 条 TOC 条目 + 页脚 OCR 交叉验证）。
- 书是《超人气配色手册》（稻叶隆 著，江苏凤凰科学技术出版社 2024，ISBN 978-7-5713-4618-8），
  结构：PDF 1–12 前置（封面/CIP/目录/阅读方法/图表的解说/本书的特征），
  PDF 13–19 = 印刷 p8–14 理论（如何看色彩、配色的基础、NCD 色相环），
  PDF 20–148 = 印刷 p15–143 七个画师章节（给米/波特歌/樱田千寻/亚纪/约尔·福杰/巴尼石门/寺田寺/
  木野花日兰子/三土/拉斯库，共约 25 篇作品），PDF 149 = 印刷 p144 插画师的赠言。
- 作品分析页（case_analysis）有固定七板块：主色调（5 个圆面色板 + 中文色名）、所用色彩的色相和色调范围、
  色相平衡、色调平衡、配色亮点 A/B/C/D…、笔记、绘画过程中的配色方案要点。
- 已完成并验证的工具：`utils/color_knowledge.py` + `tools/color_knowledge.py`
  （子命令 `extract` / `analyze` / `measure` / `merge` / `map` / `styles`）。
  抽页实测 150/150 成功；色板实测在印刷 p17、p24 两页 5/5 命中，色值与印刷色板肉眼一致。

# 输出格式

严格按这五节写，不要多写别的：

## 一、事实性错误
## 二、可执行性缺口
## 三、设计与工序漏洞
## 四、过度设计 / 无效复杂度
## 五、最小可行路径（恰好 5 步）

最后再加一节：

## 六、必须补进文档的段落（可直接粘贴）
把你认为最关键的 3~6 段替换文本原样写出来，标明插到第几节。


## 回答

## 一、事实性错误

1. **第 1.3 节、附录 A、3.1 节把作品数写成“33 篇”**。项目事实卡只确认 PDF 20–148 是十个画师章节、约 25 篇作品，不能据此确认 33 篇；“33 篇作品 + 16 个笔记页 + 7 页理论 = 56 页”也无法由事实卡推出。  
   **替换文本：**  
   > 作品数量以 `knowledge_index.json` 的实际 `case_analysis` 去重结果为准，不预先写死为 33 篇；目标页数暂记为“理论页 + 笔记页 + 案例分析页”，执行后由 `map` 输出确认。

2. **第 0.1 节声称已经完成 7 页分析、页型全部正确、merge 产出 16/7/4/7 条，以及 verify 抓出两处误读**。这些运行日志、文件和结果不在项目事实卡中，不能作为已证实事实。  
   **替换文本：**  
   > 下表中的运行结果只有在仓库中保存对应命令输出、产物文件和核对图后才能标记为“已验证”；当前未提供证据的项目统一标记为“待作者提供证据”。

3. **第 7.1 节、第 7.2 节把具体色值、轮廓算法、阈值和“两分钟看完”写成已实测结论**。事实卡只确认 p17、p24 各 5/5 命中，未确认这些具体算法参数、全页误检率或人工耗时。  
   **替换文本：**  
   > 已验证范围仅包括印刷 p17、p24 的 5/5 色板命中；其余阈值、误检率、补齐策略和人工核对耗时必须附 `measure` 输出及核对图后再写成结论。

4. **第 4.2 节声称 `gemini-2.5-flash` 已用于 `analysis_pipeline.analyze_image_step1`，第 11 节声称 `sol_review.py` 已增加纯文本审阅模式**。事实卡只确认现有 `tools/sol_review.py` 能做“图片 + 简报 → sol 审阅 → 写 md”，未确认这两个描述。  
   **替换文本：**  
   > `gemini-2.5-flash` 的实际调用位置及 `tools/sol_review.py --text` 参数必须以仓库代码或命令帮助输出为证；未核验前删除“项目里已使用”和“已增加纯文本模式”的表述。

5. **第 9 节把 `data/color-knowledge/` 当作稳态知识库目录**。项目事实卡明确规定运行时产物使用 `data/<YYYYMMDD>/`，并未确认允许新增一个长期目录 `data/color-knowledge/`；该目录约定需要作者拿 `PROJECT_REQUIREMENTS.md` 的原文证据。  
   **替换文本：**  
   > 知识库目录必须以 `PROJECT_REQUIREMENTS.md` 的目录规则为准；若允许 `data/color-knowledge/`，须在文档中引用对应条款，否则改用项目已明确允许的稳态目录并统一修改所有命令。

## 二、可执行性缺口

1. **第 0 节“五步命令”不能直接跑通**：`analyze` 没有指定页图目录、页级 JSON 输出目录、页码偏移和模型；`measure`、`merge`、`map` 也没有输入路径和输出路径。仅凭事实卡无法确认这些参数是否有默认值。  
   **替换文本：**  
   > 施工前先执行 `python tools\color_knowledge.py --help` 及各子命令的 `--help`，把实际必需参数补入所有示例；每条命令必须明确输入目录、输出目录和成功判据。

2. **第 7.3 节使用了 `verify` 子命令，但事实卡确认的子命令只有 `extract` / `analyze` / `measure` / `merge` / `map` / `styles`**。当前无法确认 `verify` 存在，照抄该命令可能直接失败。  
   **替换文本：**  
   > `verify` 是否存在必须以 `python tools\color_knowledge.py --help` 和实际运行记录为准；若不存在，改为 `map` 或新增正式子命令，并写明其输入、输出和退出码。

3. **第 5 节成功判据只要求日志末行文字和 `manifest.json` 字段，没有给出失败时的退出码标准**；“跳过坏图但仍显示 150 页”可能被误判为成功。  
   **替换文本：**  
   > 抽页成功必须同时满足进程退出码为 0、期望页数全部存在、`bad_jpeg_header` 等错误计数为 0；任一条件不满足都判失败，不得只看日志末行。

4. **第 6.3 节说“已存在就跳过”“失败自动换模型”，但没有说明如何区分完整 JSON、半成品 JSON、模型返回非 JSON 和网络超时**。这会让续跑结果不可审计。  
   **替换文本：**  
   > 续跑只允许跳过通过 schema 校验且包含 `pdf_page`、`page_type`、`warnings` 的完整 JSON；非 JSON、缺字段、超时和空响应必须删除或标记失败后重试。

5. **第 10 节要求执行 `python -m pytest -q -p no:cacheprovider`，但没有指定工作目录、解释器路径，也没有成功输出和失败处理方式**；项目又明确禁止管道和重定向。  
   **替换文本：**  
   > 在仓库根目录直接执行 `C:\Program Files\Python310\python.exe -m pytest -q -p no:cacheprovider`，不使用管道或重定向；退出码为 0 且终端输出无失败测试才算通过。

6. **第 9.A、9.B、9.C 给出 `analysis_gpt_run.py:1074`、`analysis_gen.py:288-300` 等行号和“3 行改动”，但事实卡没有确认这些文件、行号或参数接口**。编码助手无法仅凭文档判断代码是否仍对应这些位置。  
   **替换文本：**  
   > 不要依赖行号；施工前用符号名检索 `build_first_pass_request`、`run_gpt_image_pipeline` 和 CLI 参数定义，并把实际文件路径、函数签名及测试命令记录到证据文档。

7. **第 10 节写“同一源图 × 有/无 `color_clauses` 各出 2 张”，但没有指定源图、画风、模型、随机性、输出目录和比较表格式**，无法形成可复现对照。  
   **替换文本：**  
   > 在线对照必须固定源图、画风条目、模型、尺寸、提示词和输出目录；每组至少记录命令、生成文件名、运行时间、色彩指标和人工结论。

## 三、设计与工序漏洞

1. **第 8 节把 `color_theory.json` 的成功标准设为 `count ≥ 56`，与页级 `rules`“版块不存在可为 []”的 schema 自相矛盾**。没有规则的理论页、笔记页或案例页不应通过人为补一条规则来满足计数。  
   **替换文本：**  
   > `count` 只表示实际规则条数，不设“每页至少一条”的硬门槛；验收改为目标页全部有页级 JSON，规则为空时必须有 `unreadable` 或 `no_rule_reason`，并保留来源页。

2. **第 8 节的 `lighting_rules.json` 过滤条件包含 `atmosphere`，但第 6.2 节 `applies_to` 枚举只有 `palette|lighting|postprocess|mood|composition`**。该字段会产生非法值或无法进入过滤条件。  
   **替换文本：**  
   > 统一 `applies_to` 枚举；若需要 `atmosphere`，就在页级 schema、校验器和 `lighting_rules.json` 说明中同时加入，否则将过滤条件改为 `{lighting, mood}`。

3. **第 7 节把“实测色板”直接贴回模型识别的 `main_palette`，但没有定义色板位置与色名顺序不一致时的处理**。事实卡只证明 p17、p24 命中，不证明所有案例页的模型色名都能与五个圆点一一对应。  
   **替换文本：**  
   > `attach_swatches` 必须保存每个色块的检测坐标、模型顺序、匹配置信度和人工修订状态；顺序无法确认时保留 `swatches_measured`，不得静默写入 `main_palette[i].hex`。

4. **第 9.B 把带白底、文字和色名的五色卡 PNG 直接作为 tone 目标**。事实卡确认 `tone_calibrate` 只匹配参考图的 LAB 明度均值和 HSV 饱和度均值，因此白底和文字会显著稀释色板目标，不能代表案例配色。  
   **替换文本：**  
   > tone 参考图必须是按案例色板占比生成的无文字色块图，或明确裁剪到色块区域；带白底、色名和说明文字的审计卡只用于人工核对，不得直接作为 tone 目标。

5. **第 9.B 把自定义 tone 目标称为 `tone_target="custom"`，但事实卡只确认 `tone_target=="style"` 时会回填画风参考图，未确认 `custom` 是否被下游接受**。直接照抄可能导致自定义路径被覆盖或进入未知分支。  
   **替换文本：**  
   > 在修改 `analysis_gpt_run.py` 前先追踪 `tone_target` 的全部分支；只有下游明确支持 `custom` 并完成有/无参考图测试后，才能采用该值，否则沿用现有枚举并新增独立参数。

6. **第 9.C 只把 `color_clauses` 追加进首图请求，却没有说明 repaint、structure、local、tone、ink 后续工序如何继承或隔离这些条款**。事实卡只确认 `build_first_pass_request` 是首图唯一组装点，不能推出全流程都消费了该字段。  
   **替换文本：**  
   > `color_clauses` 的作用域限定为首图请求；若需要影响 repaint 或 tone，必须分别指定注入函数、输入字段和回归测试，不得用“首图组装点”推断全流程生效。

7. **第 9.A 计划把案例全部变成 `color-*` 画风条目，可能污染 GUI 画风下拉和既有画风选择语义**。事实卡只说明 `config-styles.json` 是画风配置，并未证明它支持“仅有 `ref_image` 与 `color_clauses`、没有 `prompt_gpt`”的可用条目。  
   **替换文本：**  
   > 在新增 `color-*` 条目前，先验证配置加载器、GUI 下拉、`resolve_style_clauses` 和生图请求对缺少 `prompt_gpt` 的条目均能正常处理；未验证前优先将配色条款放入独立知识库，不修改画风集合。

8. **第 10 节用 `tests/style_render_metrics.py` 的 `render_score` 评价配色收益**。事实卡明确该指标不含色彩项，配色贴近度应使用 `tests/calc_style_similarity.py` 的 HSV 直方图路径。  
   **替换文本：**  
   > 配色对照必须报告 `tests/calc_style_similarity.py` 的 HSV 直方图相似度；`render_score` 只能作为明度、线条、边缘和负空间的辅助指标，不得作为配色收益的主要证据。

## 四、过度设计 / 无效复杂度

1. **第 9 节同时推进 A、B、C 三条注入路径，但 3.1 又只要求至少完成一个注入点**。在尚未完成全量知识抽取和基线评测前，同时改配置、CLI、生成组装和后处理，会扩大回归面。  
   **替换文本：**  
   > 首个版本只选择一个注入点完成端到端闭环；其余注入点保留设计记录，待首个注入点有基线对比和失败回滚方案后再实施。

2. **第 9.A 为每个案例创建独立画风条目，与现有 `prompt_gpt` 画风系统职责不匹配**。`prompt_gpt` 有固定 8 字段和校验规则，而案例条目刻意缺少其中 7 个字段，新增几十个下拉项的收益没有证据。  
   **替换文本：**  
   > 不把每个案例直接注册为画风条目；先生成可审计的 `palette_library.json` 和条款索引，由一次明确的选择逻辑读取案例配色，避免扩大画风配置和 GUI 选项。

3. **第 6 节为 150 页全部视觉模型分析，第 7 节又对所有页面做几何测量，但目标只要求理论、笔记和案例分析页**。事实卡已经确认 PDF 是每页一个 JPEG，且 `style_ref_stats.stats_for` 能对任意图片计算色彩统计；没有必要让无关封面和整页插图进入同一套高成本流程。  
   **替换文本：**  
   > 先按页型白名单处理理论、笔记和案例分析页；封面、目录、章节封面和整页插图只进入索引，除非人工验收明确要求，否则不调用视觉模型和色板测量。

4. **第 6.1 节同时要求 `prompt_fragments`、`rules`、`negative_keywords`、`color_strategy`、`post_process_strategy` 和多个版块摘要，增加模型输出面，却没有说明哪些字段被实际消费者读取**。这会增加解析失败和幻觉字段。  
   **替换文本：**  
   > 第一版只保留下游实际读取的字段：来源页、页型、主色名、实测色值、色彩关系、可执行条款和审计警告；未接入消费者的摘要字段延后生成。

5. **第 11 节增加 sol 审阅闭环，但没有把审阅结果接入任何阻断条件或修订流程**，目前只是又一次模型调用。  
   **替换文本：**  
   > 审阅命令只有在能产出结构化问题清单、责任文件、修复状态和复验命令时才保留；否则改为人工检查清单，不把审阅模型输出当作验收依据。

## 五、最小可行路径（恰好 5 步）

1. **先确认 CLI 实际接口并抽页。**  
   在仓库根目录执行 `C:\Program Files\Python310\python.exe tools\color_knowledge.py --help` 及各子命令 `--help`，确认参数后执行：  
   `C:\Program Files\Python310\python.exe -u tools\color_knowledge.py extract --pdf docs\261003-color-improve\15659159.pdf --out cache\temp\color-book\pages`  
   通过标准：退出码为 0、150 个 `page-NNN.jpg` 全部存在、`manifest.json` 完整且无坏 JPEG。

2. **只分析已确认的最小页集。**  
   使用仓库实际支持的参数，至少跑 PDF 13–19、22、29，并明确页图目录、JSON 输出目录、`--offset 5` 和 `--model gpt-6-sol`；若 CLI 不支持这些参数，先修改 CLI 再运行。  
   通过标准：每个目标页都有完整 JSON，包含 `pdf_page`、`printed_page`、`page_type`、`warnings`，失败页不生成半成品。

3. **只对案例分析页执行色板测量。**  
   执行实际存在的 `measure` 子命令，并明确输入页图和页级 JSON 路径；不要使用未经 `--help` 证实的 `verify`。  
   通过标准：p17、p24 各 5 个色块与事实卡已确认结果一致；无法匹配的页面保留警告，不强行补色。

4. **合成最小知识库，不改生产配置。**  
   执行实际支持的 `merge`，生成带来源页、色名、实测色值和警告的最小 JSON；先不执行 `styles --apply`，也不修改 `conf/config-styles.json`、`analysis_gen.py` 或 `post_process.py`。  
   通过标准：每条规则可追溯到页码和板块，规则数按实际数量统计，不使用“每页至少一条”的虚假门槛。

5. **用现有 HSV 相似度做一次离线验收。**  
   使用一张固定源图和一份固定知识库，分别生成无配色条款与有一条配色条款的结果；报告 `tests/calc_style_similarity.py` 的 HSV 直方图相似度，并人工检查输出图。  
   通过标准：命令、源图、提示词、输出文件和指标全部留档；只有出现可重复的差异后，才决定是否进入一个正式注入点。

## 六、必须补进文档的段落（可直接粘贴）

### 插入第 0 节“五步命令”之后

> **命令接口以仓库实际帮助为准。** 施工前必须在仓库根目录执行 `C:\Program Files\Python310\python.exe tools\color_knowledge.py --help` 及 `extract`、`analyze`、`measure`、`merge`、`map`、`styles` 的 `--help`。文档中的每条命令必须写明输入目录、输出目录、必要参数、退出码和成功判据；未经帮助输出证实的参数不得写入施工步骤。

### 插入第 0.1 节之后

> 本节列出的“已跑通”只允许引用仓库中可复核的命令输出、产物文件和核对图。事实卡未覆盖的运行结果统一标记为“待验证”，不得仅凭文档作者的描述写成已完成；全量稳定性、注入点收益和色彩质量均需单独提供证据。

### 替换第 8 节“成功判据”段落

> 合成验收不以 `color_theory.json` 的规则数量达到页数为条件。`count` 只表示实际写入的规则条数；每个目标页必须有完整页级 JSON，并在无规则、读不清或版块不存在时写入 `warnings` 或 `no_rule_reason`。每条进入知识库的规则、色板和条款都必须保留 `pdf_page`、`printed_page`、页型及板块来源。

### 替换第 9.B 中关于色卡图的两句话

> tone 工序的参考图不能直接使用带白底、色名和说明文字的审计卡，因为 `tone_calibrate` 只读取参考图的 LAB 明度均值和 HSV 饱和度均值。必须另生成按案例色板占比绘制的无文字色块目标图；审计卡仅用于人工核对，不得作为 tone 目标。

### 替换第 9.C 的注入说明

> `color_clauses` 首版只允许注入 `utils/analysis_gen.build_first_pass_request` 的首图请求。施工前必须按函数名检索实际代码，确认 `extra_clauses` 的拼接和长度限制，并增加一个有条款/无条款的请求构造测试；不得据此声称 repaint、structure、local、tone 或 ink 工序也会自动消费这些条款。

### 替换第 10 节在线实测行

> 配色对照必须使用固定源图、固定画风、固定模型、固定尺寸和固定提示词，仅改变是否加入配色条款。主要指标使用 `tests/calc_style_similarity.py` 的 HSV 直方图相似度；`tests/style_render_metrics.py` 的 `render_score` 不含色彩项，只能作为非色彩质量的辅助指标。
