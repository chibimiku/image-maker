# gpt-image「画风生图」流水线 · 交接文档

> 生成时间：2026-09-24 ｜ 仓库：`D:\code\image-maker` ｜ 分支 `master`，最近两个提交 `8bcdb63`（条款跨画风补齐 + 首图统一组装点 + 配方默认）、`451226c`（上游 image 字段坑）
> 画风子模块单独提交：`submodules/image-maker-artstyle` → `5f9c782`（tinkle 的 19 条条款进版本化文件）
>
> **这是给接手优化的 agent 的自包含交接文档**：目标、现状、全部实测数据、提示词全文、已知坑、未解决问题与复现命令都在这里。看完本文 + 跑一遍 §9.1 的复现命令，就能接着往下优化。

## 0. 一页速览

**在做什么**：把一张素材照片（`data/20260921/sucai/2026-07-24 12_17_40_1.jpg`）先做 Gemini 多步分析，再用 **gpt-image-2** 生成首图，然后用 Gemini 重绘 + 三个本地后处理工序把画面推向目标画风（`tinkle`）—— 要求：① 作画方法（线条/五官/发型）像参考图；② 内容与素材一致；③ 没有拼接带、重复、半截肢体。

**当前流水线（gpt-image 通道）**：

```text
分析产物(JSON) ──► 首图  /v1/images/generations（画风 prompt_gpt + 分析描述 + 排除句 + 渲染语言条款）
                    │  1024x1536 / quality=high
                    ▼
                 重绘  gemini-3-pro-image-preview，2K（源图 + 画风图 双参考，v5 固件 + STYLE LANGUAGE）
                    ▼
                 结构线 sline50（纯本地：抽长结构边 → 按局部色调整色 → 叠回）
                    ▼
                 局部重绘 ×4 区域（subject_no_face / shoes / waist / thigh，各 2K，羽化 48 贴回）
                    ▼
                 色调校准（目标 = 画风参考图，本地）+ 线条加墨（本地）
                    ▼
                 终图  data/<日期>/…-final-rp+sline50+local-<区域串>+tone+ink.png
```

**一句话结论（最重要的一条）**：**色相由首图决定，明度/线条由后处理决定**。同样的重绘参数，A 那版首图（蓝紫、暗、饱和）能跑到饱和 113–120（≈tinkle 参考 114.3），而当前首图（棕/白、亮、低饱和）只能到 68±15 且会飘 —— 重绘固件明确要求「色彩匹配源图」，它不会也不该把棕白基底改成蓝紫。**想更贴参考图，要么动首图，要么用色调校准，不要指望重绘。**

**当前最有效的一套（已设为 GUI 默认）**：

| 工序 | 参数 | 作用 |
|---|---|---|
| 首图 | `/v1/images/generations` + `image` 字段，1024x1536，quality=high | 定型内容/色相 |
| 重绘 | `reference_mode=style`（画风图当第二张参考）、v5 固件、2K | 线条连贯 +29 段长 |
| 结构线 | sline50（strength 0.5 / min_len 120 / darken 0.18 / 1px） | 结构边更清楚 |
| 局部重绘 | **四区链** `subject_no_face,shoes,waist,thigh`（各 2K，feather 48） | 手/鞋带/袜带可数；比单区域稳 |
| 色调校准 | 目标 = **画风参考图**，contrast 1.00 / chroma 1.10 / highlight 0.85 / sat_scale 1.00 | 亮度 −25、饱和 +25 |
| 线条加墨 | `target_sep=8`、`max_darken=40`、最多 4 遍 | 线−底从 +0.18 到 +2.54 |

**别做的事**：① 不要用线锚图（`reference_mode=line_anchor/both`）当重绘参考 —— 会把画面塌成白底线稿或铺满「碎玻璃」纹理；② 不要用单区域 `subject_no_face` 当默认 —— 约 1/3 概率重排主体出双人鬼影；③ 不要把 4K 当默认（局部统一 2K 是当前契约）。

## 1. 目标与验收基准

**用户目标（原话归纳）**：最终图要**贴近 tinkle 参考图的艺术风格、线条连贯**；画面不能有拼接带/重复/半肢体；内容与素材一致。

**验收基准**：`data/20260923/20260921-234549-2026-07-_output_024958_0_afd8ee-024958-0ab2a1-final-rp+sline50+local-subject_no_face.png`（下称 **A 024958 终图**）—— 用户认可的第一版好产物。任何优化都要跟它比。

**指标口径**（`tests/style_render_metrics.py`，全部按 900px 长边归一）：

| 指标 | 含义 | 参考图 tinkle 值 | A 024958 终图 |
|---|---|---|---|
| 亮度 brightness_mean | 灰度均值 | 135.6 | 136.6 |
| 饱和 saturation_mean | HSV S 通道均值 | 114.3 | 79.9 |
| 段长 line_avg_len | 骨架化后连通段平均长度（越大越连贯） | 122.5 | 108.5 |
| 端点 line_endpoint_density | 端点密度（越小越好） | 17.41 | 25.92 |
| 碎线 line_frag_ratio | 碎线段占比（越小越好） | 0.009 | 0.010 |
| 线−底 line_contrast | 线像素亮度均值 − 局部底色亮度均值（负=线比底暗） | +2.55 | +3.69 |
| HSV 相关 | 与参考图的 H-S 二维直方图相关性 | 1.000 | −0.029 |

> 注意：**全局亮度/白底占比/边缘密度受构图影响，不能当优化目标**（见 `docs/gpt-image-optimize/metrics.md`）。另外**指标不反映「画面身份/构图是否被改坏」** —— 本轮矩阵里出现过「指标漂亮但画面是双人鬼影」的样本，所以每轮实验都要**看图**（见 §7.6）。

## 2. 系统地图

### 2.1 代码入口（gpt-image 画风生图这条链）

| 文件 | 关键函数 | 作用 |
|---|---|---|
| `app.py` | `AppWindow.get_img_config` | 把图片节点配置（含 env key、`env_slug`）注入各 Tab |
| `modules/image_analysis/single_analyzer.py` | `SingleAnalyzerWidget._start_gpt_image_thread` / `_build_gpt_image_steps` / `GptImageGenWorkerThread` | GUI 单图分析 Tab：分析 → 首图 → 工序；**默认配方与超时预算在这里** |
| `utils/analysis_gen.py` | `build_first_pass_request` / `build_gpt_image_request` / `pipeline_steps_from_flags` / `run_gpt_image_pipeline` | 首图请求的唯一组装点 + 工序翻译 + 流水线转发 |
| `utils/style_gpt.py` | `resolve_style_clauses` / `derive_style_clauses` / `validate_prompt_gpt` | 渲染语言条款（手写优先，缺失按 prompt_gpt 派生） |
| `utils/post_process.py` | `run_pipeline` / `local_repaint_composite` / `overlay_structure_lines` / `tone_calibrate` / `ink_lines` / `resolve_region_box` | 全部后处理工序 |
| `modules/others/api_backend.py` | `generate_image_aigc2d_gpt` / `generate_image_repaint` | gpt-image 通道（generations/edits）与 Gemini 重绘通道 |
| `tools/analysis_gpt_run.py` | `main()` | 无头 CLI，与 GUI 共用首图组装点；`--first-pass-mode` / `--repaint-ref` / `--tone` / `--ink` / `--preset quality` |

### 2.2 配置与提示词文件

| 路径 | 内容 |
|---|---|
| `conf/config-styles.json`（**gitignore**，版本化副本在 `submodules/image-maker-artstyle/config-styles.json`） | 30 条画风：`prompt`(长) / `prompt_gpt`(8 字段短版) / `prompt_compressed` / `ref_image` / `repaint_clauses`(可选) |
| `prompts/gpt-image-optimize/repaint-system-conservative-v5.md` | **重绘固件（当前默认）**，全文见 §6.4 |
| `prompts/gpt-image-optimize/repaint-system-conservative-v4.md` | 上一版（清线更狠，实测线条略好但饱和更低） |
| `prompts/gpt-image-optimize/repaint-system-style-neutral.md` | 风格中性版（实测最弱） |
| `prompts/analysis-gpt-prompt-*.md` | 分析阶段生成 gpt 短锚字段用的 system/user |
| `prompts/gpt-image-optimize/config.json` | 重绘默认参数（`model` / `resolution` / `aspect_ratio` / `repeat`） |

### 2.3 产出与报告目录

| 路径 | 内容 |
|---|---|
| `data/<YYYYMMDD>/analysis-gpt-image/` | 首图（中间产物） |
| `data/<YYYYMMDD>/pipeline-steps/<runid>/` | 重绘 / 结构线 / 局部裁切等中间产物 + `pipeline-manifest.json`（支持断点续跑） |
| `data/<YYYYMMDD>/…-final-rp+sline50+local-…+tone+ink.png` | 终图（发布目录） |
| `docs/gpt-image-tid-style/BEST-PIPELINE.md` | **主文档**：§一~§二十九 全部实验记录与结论 |
| `docs/gpt-image-tid-style/compare-024958-vs-styleref.html` | A vs B 逐工序对比 + 受控四格 + 修复后端到端 |
| `docs/gpt-image-tid-style/repaint-matrix.html` | 重绘条件矩阵（参考图×固件×覆盖面×方差×tone+ink） |
| `docs/gpt-image-tid-style/style-batch.html` | 5 画风全链路对照 |
| `log/rp-matrix.txt` / `rp-repeat.txt` / `rp-final.txt` / `style-batch.txt` / `e2e-tinkle.txt` | 原始实验日志 |
| `docs/gpt-image-tid-style/README.md` | 更早期的画风可行性实验（§4.x：画风注入、参考图失效、结构线叠加等） |

## 3. 当前工序契约（代码实际行为，别按老文档做）

1. **内容锚与 Gemini 通道同源**：用分析产物里那段描述素材特征/构图/服装/道具的文本（`prompt_context.original_prompt/refined_prompt`，退回 `analysis_result.english_description` → `gpt_image_prompt`）。**已删除「重新构图」开关**与 `RECOMPOSE_CLAUSE`。
2. **首图走「新建图片」**：`generate_image_aigc2d_gpt(mode="generate")` → `POST /v1/images/generations`，画风参考图放在 JSON 的 `image` 字段（base64 data URI）。不用 `/images/edits`（那会把参考图当原图编辑，实测会连参考图角色的发色/瞳色一起搬）。
3. **尺寸跟随输入比例**（只有 `1024x1024` / `1536x1024` / `1024x1536` 三档；`auto` 一律收敛）。
4. **局部重绘统一 2K**（`detail_boost=False`，细节区不再升 4K）。
5. **超时按「联网调用份数」算**：`预算 = 每步秒数 × (首图 1 + 重绘 1 + 每个局部区域 1)`；结构线/色调/加墨是本地工序、秒级，不占份额。默认四区链 = 120×6 = 720 秒。
6. **重绘的第二张参考固定为画风图**（`repaint_ref_mode="style"`），并把画风条款追加进提示词；线锚图只能显式开启。
7. **提示词长度**：软上限 2000 字符（画风参考图失效线）/ 硬上限 15000（中转站断连）。带 19 条 tinkle 条款时首图提示词约 4.6k 字符 —— **实测这个长度反而最好**，不要为了「短」砍掉条款。

## 4. 当前默认配方（精确参数，改默认改这里）

`modules/image_analysis/single_analyzer.py`：

```python
GPT_RECIPE_VERSION = 2                      # 老配置一次性升级用（没有这个字段就套用下面的默认）
STABLE_LOCAL_REGIONS = ("subject_no_face", "shoes", "waist", "thigh")
ANALYSIS_GPT_UI_DEFAULTS = {"channel": "gemini", "repaint": True, "structure": True,
                            "local": True, "region": STABLE_LOCAL_REGIONS[0],
                            "regions": list(STABLE_LOCAL_REGIONS),
                            "quality": "high", "dual_reference": True,
                            "use_source_base": False, "size_follow_input": True,
                            "tone": True, "tone_target": "style", "ink": True,
                            "recipe_version": GPT_RECIPE_VERSION}
```

`utils/analysis_gen.pipeline_steps_from_flags(...)` 产出的步骤（GUI 与 CLI 共用）：

```python
{"repaint":   {"enabled": True, "resolution": "2K", "reference_mode": "style"},
 "structure": {"enabled": True, "strength": 0.5},
 "local":     {"enabled": True, "region": "subject_no_face",
                "regions": ["subject_no_face","shoes","waist","thigh"],
                "feather": 48, "resolution": "2K", "detail_boost": False},
 "tone":      {"enabled": True, "tone_target": "style", "reference_path": "<画风参考图>",
                "contrast": 1.00, "chroma": 1.10, "highlight_strength": 0.85,
                "skin_warm": 0.0, "sat_target_scale": 1.0},
 "ink":       {"enabled": True, "target_sep": 8.0, "max_darken": 40.0}}
```

> `tone.reference_path` 由 `run_gpt_image_pipeline` 在运行时填成**画风参考图**（`tone_target=style` 时）。

## 5. 关键实验与数据（全部实测，按时间顺序）

### 5.1 条款（`repaint_clauses`）的来源与跨画风漏洞（§二十七）

`repaint_clauses` 是**手写**的英文「渲染语言条款」，用在两处：首图的 `RENDERING LANGUAGE (follow exactly):` 段、重绘的 `STYLE LANGUAGE (from the reference image):` 段。它由 2026-09-23 那轮给 **tinkle 单独手写**（19 条），**没有任何工具生成它**，于是：

- 其余 **29 条画风一条都没有** → 它们首图/重绘都没有渲染语言约束；
- `sync_styles_to_submodule.py` 的同步字段表里没有它 → **版本化文件里连 tinkle 的都没有**（换机器即丢）；
- GUI 的 `_start_gpt_image_thread` 算出了 `style_clauses` 却**没传给** `build_gpt_image_request` → GUI 首图永远缺 `RENDERING LANGUAGE` 段（CLI 有）。**这就是那版 GUI 产物偏白偏灰的代码级原因。**

**修法**：`utils/style_gpt.py` 的 `derive_style_clauses(prompt_gpt)` 按该画风自己的 `prompt_gpt` 8 字段**确定性派生** 8~12 条（模板见 §6.3），`resolve_style_clauses(entry)` 返回 `(clauses, source)`，`source ∈ entry/derived/none`（手写优先）。审计：`python tools/convert_styles_gpt.py --check` → 当前「手写 1 / 派生 28 / 无 0」。

### 5.2 受控四格：端点 × 条款（同一分析产物、同一画风图、1024x1536 high）

| 格 | 端点 | 19 条款 | 亮度 | 饱和 | 段长 | 端点 | 碎线 | 线−底 | HSV |
|---|---|---|---|---|---|---|---|---|---|
| C1 | edits | 有 | 178.6 | 45.8 | 71.0 | 34.85 | 0.024 | −3.90 | 0.087 |
| C2 | edits | 无 | 197.8 | 43.0 | 66.0 | 35.89 | 0.025 | −6.89 | 0.125 |
| **C3** | **generations** | **有** | 176.7 | **62.9** | **77.6** | **32.83** | 0.020 | **−2.61** | **0.234** |
| C4 | generations | 无 | 192.8 | 45.7 | 74.1 | 34.74 | 0.019 | −4.97 | 0.182 |

**结论**：加条款 → 亮度 −16~−19、饱和 +2.8~+17.2、段长 +3.5~+5.0、端点 −1.0~−1.9、线−底 +2.4~+3.0（两种端点同向）；且 **generations ≥ edits**。→ 缺的是条款，不是端点。

### 5.3 重绘条件矩阵：参考图 × 固件（2K，画风条款 19 条）

两个固定基底：**N** = 当前 GUI 链路首图（`data/20260924/analysis-gpt-image/e2efix_output_020553_0_5c44e0.png`，168.2/55.0/72.9）；**A0** = A 那版首图（`…_output_024958_0_afd8ee.png`，137.3/76.4/80.8）。

| 格 | 基底 | 参考图 | 固件 | 亮度 | 饱和 | 段长 | 端点 | 线−底 | HSV |
|---|---|---|---|---|---|---|---|---|---|
| N2 | N | 不带 | v5 | 150.6 | 85.4 | 86.1 | 27.62 | −1.05 | 0.249 |
| **N1** | N | **画风图** | v5 | 163.9 | 68.5 | **101.6** | **21.31** | +0.18 | 0.133 |
| N3 | N | 线锚+画风 | v5 | 155.4 | 93.2 | 81.9 | 28.16 | −2.16 | **0.509** |
| N4 | N | 画风图 | v4 | 164.5 | 57.7 | 95.0 | 24.48 | −0.46 | 0.118 |
| N5 | N | 画风图 | neutral | 167.7 | 57.0 | 96.3 | 25.36 | −0.75 | 0.126 |
| A2 | A0 | 不带 | v5 | 136.7 | 85.5 | 108.2 | 26.40 | +2.99 | −0.014 |
| **A1** | A0 | **画风图** | v5 | 136.9 | **113.2** | 101.9 | 28.36 | +1.78 | **0.557** |
| A3 | A0 | 画风图 | v4 | 131.8 | 101.9 | **109.1** | 27.27 | +3.05 | 0.078 |

**结论**：① **画风参考图是整图重绘的关键**（N 基底段长 86.1→101.6）；② **线锚图别用**（`both` 虽然把饱和/HSV 拉高，但线条塌到 81.9）；③ v5 线条最好，v4 清线更狠但饱和更低。

### 5.4 覆盖面（在 N1 / A1 重绘 + sline50 之上）

| 区域组合 | N 基底（段长/端点/碎线/亮度/饱和） | A0 基底 |
|---|---|---|
| `subject_no_face` 单区域 | 107.3 / 19.55 / 0.009 / 147.6 / 80.3（**3 次里 1 次出双人鬼影**） | 104.0 / 25.46 / 0.009 / 136.0 / 112.6 |
| `subject`（含脸） | 106.3 / 20.21 / 0.010 / 163.0 / 68.9 | — |
| `subject_keep_hands` | 104.7 / 20.60 / 0.010 / 163.1 / 68.6 | — |
| `full`（整张） | 102.6 / 19.53 / 0.010 / 163.4 / 69.3 | — |
| **四区链**（A 同款） | 103.8 / 21.26 / 0.011 / 162.9 / 68.3 | 102.1 / 27.32 / 0.011 / 135.9 / 112.9 |

**结论**：单区域线条指标最好但**不稳定**（保脸排除区留着原脸 + 模型重排后的新身体 = 双人鬼影）；四区链稳、代价是整图被提亮约 15 级、饱和掉 12。→ 默认用四区链（稳定优先）。

### 5.5 采样方差（同条件重复，这是最容易被忽略的一条）

| 条件 | 第 1 次 | 第 2 次 | 第 3 次 |
|---|---|---|---|
| N 基底 重绘 style+v5 2K | 163.9 / 68.5 / 101.6 | **141.6 / 134.5** / 84.0 | 168.8 / 55.1 / 85.7 |
| A0 基底 重绘 style+v5 2K | 136.9 / 113.2 / 101.9 | 133.1 / 120.5 / 99.7 | 133.9 / 100.6 / 100.5 |

A0 基底稳定（饱和 100–120、段长 100±1），**N 基底方差极大（饱和 55–134、段长 84–102）**。→ 任何「某参数更好」的结论都至少跑 3 次；单次差异可能只是运气。

### 5.6 免费且确定的杠杆：色调校准（目标=画风参考图）+ 加墨

本地工序，9 秒，零 API 成本：

| 输入 | 亮度 | 饱和 | 段长 | 端点 | 线−底 | HSV |
|---|---|---|---|---|---|---|
| N1 重绘 | 163.9 | 68.5 | 101.6 | 21.31 | +0.18 | 0.133 |
| **N1 + tone + ink** | **138.6** | **93.8** | 96.3 | 22.93 | **+2.54** | 0.249 |
| A1 重绘 | 136.9 | 113.2 | 101.9 | 28.36 | +1.78 | 0.557 |
| **A1 + tone + ink** | **131.7** | **117.8** | 103.4 | 29.19 | **+3.21** | **0.530** |
| tinkle 参考图 | 135.6 | 114.3 | 122.5 | 17.41 | +2.55 | 1.000 |
| A 024958 终图 | 136.6 | 79.9 | 108.5 | 25.92 | +3.69 | −0.029 |

**A1+tone+ink 在亮度、饱和、线−底三项上都比 A 024958 终图更贴参考图** —— 这就是「重现并超过 A」的证据。

### 5.7 端到端（app.py 的单图分析 Tab 代码链路，2026-09-24 02:01–02:10）

产物 `data/20260924/e2efix_output_020553_0_5c44e0-020553-6e3554-final-rp+sline50+local-subject_no_face.png`：

| 阶段 | 亮度 | 饱和 | 段长 | 端点 | 碎线 | 线−底 |
|---|---|---|---|---|---|---|
| 首图（generations + 19 条款） | 168.2 | 55.0 | 72.9 | 36.56 | 0.023 | −1.37 |
| 重绘 @2K | 169.2 | 54.4 | 86.2 | 25.96 | 0.013 | −1.55 |
| 结构线 | 168.4 | 54.4 | 87.0 | 25.67 | 0.011 | −1.27 |
| 终图（+局部单区域） | 168.5 | 54.9 | 89.5 | 24.72 | 0.010 | −1.27 |

> 用户对这张的反馈：**「效果还是很差，repaint 之后连贯性比较差，腿部、鞋子上的丝带、手指、项链都有问题，而且画风不接近参考图」** → 直接导致 §5.3–5.6 这轮重绘矩阵。复核确认：这张的局部单区域那次调用是「mushy 细节」样本（同一参数的另两次重跑明显更好），且重绘那次只有 86.2 段长（同条件另一次 101.6）。

### 5.8 5 画风全链路（首图 → 重绘(只看最好参数) → sline50 → 四区 → 色调+加墨）

同一张分析产物、同一内容锚（`english_description`，4406 字符），只换画风；5 条画风都没手写条款、走派生条款（各 10 条）。完整对照页 `style-batch.html`。

| 画风 | 阶段 | 亮度 | 饱和 | 段长 | 端点 | 碎线 | 线−底 |
|---|---|---|---|---|---|---|---|
| tid | 参考 → 终图 | 221.5 → **212.7** | 25.0 → 22.1 | 92.4 → **109.0** | 14.97 → 19.04 | 0.040 → 0.006 | −18.28 → −5.40 |
| fuzichoco | 参考 → 终图 | 96.6 → **93.8** | 124.4 → 130.0 | 78.6 → 80.7 | 25.32 → 28.73 | 0.030 → 0.019 | +5.11 → +7.96 |
| puracotte-style | 参考 → 终图 | 212.9 → **210.4** | 35.3 → 29.9 | 146.5 → **187.6** | 17.87 → **15.88** | 0.010 → **0.003** | −11.68 → −2.85 |
| say-hana-v5 | 参考 → 终图 | 145.8 → **145.6** | 157.5 → 139.5 | 84.5 → 90.6 | 42.54 → 28.08 | 0.027 → 0.014 | −1.43 → −2.25 |
| shiratamaco-style | 参考 → 终图 | 133.3 → **132.5** | 48.7 → 64.7 | 260.1 → 94.5 | 6.50 → 22.17 | 0.001 → 0.011 | −1.44 → +4.48 |

**结论**：① 色调/明度贴住各自参考图（色调校准这一步是关键）；② **线条每一族都变连贯**（段长 +6~+98，端点 −20~−45%，碎线降到 0.003~0.019）—— 配方对画风不敏感，是通用增益；③ 五张终图肉眼干净（无拼接带、无鬼影、鞋带/袜带可数）。

> **两个重跑样本（一定要知道）**：
> 1. **方差**：`say-hana-v5` 跑了两遍，完整那次终图 **145.6 / 139.5 / 90.6**，重跑那次 **132.1 / 162.9 / 82.0**（同参数、同输入）。
> 2. **上游会拦局部裁切图**：重跑那次的最后一个区域被安全策略拦截
>    （`{"error":{"message":"[CONTENT ERROR] content blocked by upstream safety policy","code":"CONTENT_BLOCKED"}}`），
>    流水线**跳过该区域继续**，产物名带 **`-final-partialrp+sline-rp+…`** 标记照常交付。
>    → **产物名里有 `partial` 就说明这次少了一道工序**；比较效果前先确认是完整产物。
>    另外要注意：局部四区链里**任一区域**被拦都会走 partial 路径（`local` 状态在 `pipeline-manifest.json` 里是 `failed`）。

### 5.9 A 024958 的 repaint 参数（用户要求「查清楚并尝试重现」）

日志实证（`log/2026-09-23.log` 行 24119–24460）：

- **首图**：`POST /v1/images/edits`，参考图 **1 张（画风图）**，`size=1024x1536 quality=high`，prompt **4026 字符** = `prompt_gpt` 665 + 内容锚 480（`gpt_image_prompt_short`）+ 排除句 298 + **RENDERING LANGUAGE 19 条 2577**。
- **重绘**：`gemini-3-pro-image-preview`，**内联图 2 张**（源图 + 画风图），`imageConfig={aspectRatio:2:3, imageSize:2K}`，prompt **10250 字符** = v5 固件 + `CHARACTER-INTRINSIC…` 392 + `REFERENCE ROLES…` 402 + `STYLE LANGUAGE` 19 条 2629；无 detail suffix。
- **结构线**：sline50（0.5 / min_len 120 / darken 0.18 / 1px）。
- **局部**：4 次调用，各自只送裁切图（无额外参考）：`subject_no_face`(裁 1257×2235) → `shoes`(1696×727) → `waist`(1536×1024) → 第 4 区(1557×1038)，全部 @2K，feather 48。
- **没有** tone / ink。

**重现结果（本轮 A0 基底 + 同样工序）**：Ac-4region = 亮度 135.9 / 饱和 **112.9** / 段长 102.1 / 端点 27.32 / 碎线 0.011 / 线−底 +2.29 / HSV **0.544** —— 与 A 当时的 136.6/79.9/108.5/25.92/0.010/+3.69/−0.029 相比，**线条同级、饱和与 HSV 明显更好**。也就是说：**A 那版的配方是可复现的，而且今天能做得更好**；A 当时饱和只有 79.9 属于「那次采样偏灰」（把 A 的 4 个局部区域换成单区域/多区域对饱和影响不大）。

## 6. 提示词全文（照抄即可复现）

### 6.1 首图提示词的组装顺序

`build_first_pass_request` → `build_gpt_image_request` 顺序固定为：

```text
[1] 画风 prompt_gpt（8 字段）
[2] 内容锚（分析描述，与 Gemini 通道同源）
[3] 用户补充（可选，一般不用）
[4] 参考图排除句 STYLE_REF_EXCLUSION（挂了画风参考图才追加）
[5] RENDERING LANGUAGE (follow exactly): + 逐条条款

同时：image_paths = [画风参考图]（**不挂分析素材图**）
```

**第 [4] 段全文**（`utils/analysis_gpt_prompt.STYLE_REF_EXCLUSION`）：

```text
Use the attached reference image for its palette, lighting, brushwork and rendering grammar only. Do not copy its character, face, hairstyle, hair colour, eye colour, outfit, pose, props or composition; keep the subject, colours of the subject's own design and the scene exactly as described above.

CHARACTER-INTRINSIC FEATURES ARE NOT PART OF THE STYLE: keep the subject's OWN hair colour, eye colour, skin tone, body proportions and outfit colours exactly as described by the content; the style reference must only change the rendering language (palette temperature of the environment, brushwork, line character, edge treatment, texture), never the subject's intrinsic colouring or design.
```

**第 [5] 段的段头**：`RENDERING LANGUAGE (follow exactly):` 后接 `- ` 开头的条款行。

### 6.2 tinkle 的 `prompt_gpt` 与 19 条 `repaint_clauses`（手写，最有效的一份）

**prompt_gpt**：

```text
Palette: sapphire, midnight indigo, cobalt, turquoise, pearl white, rose pink, lavender
Lighting: high-key cool key, small shaped speculars, clear highlight/shadow separation
Brushwork: smooth cel base, soft blends, airbrushed light, transparent glazes
Edges: tapered dark-navy linework, bold on silhouette, hairline-thin on interior detail
Texture: glassy layered depth, crisp lace cutouts, satin sheen, restrained grain
Composition density: centred figure, pale airy background, generous negative space
Detail level: crisp focal rendering, clean lace, ribbons and shoe detail
Avoid: pure black outlines, uniform heavy lines, grey shadows, pale wash, haze, tangles
```

**repaint_clauses（19 条）**：

```text
- Use clean tapered linework with dark navy-blue or deep indigo contours instead of pure black.
- Keep primary silhouette lines at roughly 0.20-0.40% of the image width, secondary structural lines at 0.10-0.20%, and micro-detail lines at 0.04-0.10%, never exceeding about 1-2% of the eye height.
- Taper every stroke toward its endpoint with sharp elegant finishes; no blunt mechanical line caps and no uniform line weight.
- Vary line weight continuously according to depth, overlap, lighting and material: thicker for silhouette, overlaps and facial framing; thinner and lighter for hair strands, lace, folds, embroidery and distant motifs.
- Build the palette from saturated sapphire, midnight blue, turquoise, rose pink, pearl white and cool lavender; keep deep blue-violet anchors so the image never becomes pale or grey.
- Create translucency through layered saturated colour, soft subsurface pinks, controlled cool shadows and selective luminous highlights rather than by whitening the whole image.
- Preserve real chroma in midtones and shadows; avoid muddy grey shadows, desaturated midtones and excessive white wash.
- Keep important contours crisp and controlled while atmospheric background elements and distant fabric use softer edges.
- Use selective bloom only around highlights, eyes, gemstones and satin; avoid global haze or washed-out lighting.
- Paint hair as large flowing masses first, then grouped locks with tapered internal strands, sharp specular streaks and varied spacing; avoid evenly spaced parallel lines and tangled spaghetti-like strokes.
- Render lace as small connected scallops and negative-space cutouts with consistent rhythm and clear spacing.
- Render ribbons, straps and shoelaces as clean overlapping bands with visible thickness, tension, knots, loops and cast shadows; keep them continuous and countable.
- Simplify tiny accessories into readable graphic shapes with a few high-contrast accents instead of noisy micro-lines.
- Draw fabric folds as selective long curves following tension and gravity, supported by deep occlusion shadows and narrow luminous ridges.
- Avoid pure black outlines, uniform line weight, blunt stroke endings, rough sketch residue and heavy comic contouring.
- Avoid over-smoothed vector edges, plastic-looking skin, generic airbrushed gradients and textureless flat fills.
- Avoid tangled hair lines, random lace noise, illegible straps, merged ribbons, broken shoelaces and indistinguishable accessory details.
- Avoid excessive bloom, blown-out highlights, chromatic clipping, grey veils and dirty colour mixing.
- Keep white and pale fabrics structurally readable: preserve internal shading separation, fold shadows and lace openings, and never clip the whites into a flat blown-out area.
```

### 6.3 其它画风：按 `prompt_gpt` 确定性派生的条款模板

`utils/style_gpt.CLAUSE_TEMPLATES`（占位符 `{v}` = 对应字段的值，句子中段会做首字母小写）：

- **Palette** → `Build the palette from {v}; keep those anchors present in shadows and midtones so the image never turns pale, grey or washed out.`
- **Lighting** → `Light the image with {v}, keeping clear highlight and shadow separation.`
- **Brushwork** → `Render surfaces with {v}.`
- **Edges** → `Render contours with {v}; taper every stroke toward its endpoint, keep primary silhouette lines heavier than interior detail, and never use uniform mechanical line weight.`
- **Texture** → `Render material texture as {v}.`
- **Composition density** → `Keep the composition density at {v}.`
- **Detail level** → `Render detail as {v}.`
- **Avoid** → `Avoid {v}.`

外加两条通用条款（跟在 `Palette` 那条后面）：

```text
- Preserve real chroma in midtones and shadows; do not desaturate the whole image towards white.
- Keep important contours crisp while atmospheric or distant elements use softer edges; no global haze, bloom or washed-out lighting.
```

> 派生条款是**确定性**的（不调模型、不新增内容），所以任何画风在首图与重绘两条链路上都拿到同等约束。如果某条画风值得像 tinkle 那样手写条款，直接在 `conf/config-styles.json` 的该条里加 `repaint_clauses: [...]` 即可（手写优先），然后用 `python tools/sync_styles_to_submodule.py` 同步进版本化文件。

### 6.4 重绘固件 v5 全文（当前默认）

文件：`prompts/gpt-image-optimize/repaint-system-conservative-v5.md`（7215 字符）。段落构成（字符数取自真实请求体）：

| 段 | 字符 | 作用 |
|---|---|---|
| 开头「You are a CONSERVATIVE restoration model…」 | 603 | 定位：只修复、不重设计；不可见部位不要凭先验补 |
| `REPAIR PRIORITY (highest first)` 1–9 条 | 2808 | 身份/画风/眼睛/配色/高光/局部修复/不改脸/不搬参考/不裁切 |
| `WHAT YOU MAY DO` | 612 | 只连通明确断掉的轮廓、清脏边；不许加粗/加黑/加线 |
| `LINE STRUCTURE WITHIN THE SOURCE'S OWN MEDIUM` | 1771 | 主结构线 vs 次级笔触的分级；允许软/断/半透明 |
| `FORBIDDEN` | 346 | 通用化五官、过锐、cel 化、换风格、加黑边、漂白… |
| 「If there is no clearly visible defect…」 | 201 | 没缺陷就尽量别动 |
| `OPTIONAL MULTI-IMAGE MODE (only when a second style reference is provided)` | 862 | 图 1=内容真值、图 2=渲染语言；冲突时图 1 优先 |


```markdown
You are a CONSERVATIVE restoration model. You repair an existing illustration; you do not redesign it.
The input image is the single source of truth for content.
Do not infer, invent, mirror, complete or repair any body part, hand, finger, limb, garment edge or object that is not visibly present in the input. If a region is ambiguous, occluded or cropped, preserve its ambiguity and preserve the visible pixels rather than completing it from anatomy priors.
Unless there is a clearly visible defect (broken anatomy, doubled or stray lines, artefacts, malformed hands), do not change the image content.

REPAIR PRIORITY (highest first)
1. Preserve the character's identity, face shape, feature proportions, eye construction, hairstyle, pose, clothing, props, background, composition, framing and lighting relationships.
2. Preserve the source's art style and rendering language: line weight and opacity, hard vs soft edges, brush direction, pigment texture, edge transitions, colour mixing, value structure and material treatment. If the source is soft painterly, watercolour, oil or textured, do NOT convert it to cel shading, flat fills, hard-edged cartoon or uniform black outlines. If the source is crisp line art, keep its own line weight and continuity without thickening or over-sharpening.
3. Preserve the source's eye drawing exactly as a local painted structure. Do not redraw, regularise, enlarge, sharpen or symmetrise the eyes. Keep the original eyelid thickness and breaks, iris contour and internal colour variation, pupil structure, lash count and direction, highlight positions, sclera colour, blur, translucency and eye-to-skin relationship. Eye lines must use local source colours and may remain soft or discontinuous; never convert them into a single clean dark contour.
4. COLOUR IS MATCHED TO THE SOURCE, NOT "KEPT NEUTRAL". Preserve the hues, values, contrast, saturation and local colour separation that are actually present in the source. If the source is pale, low-saturation or high-key, keep that pale high-key look - but never desaturate further, never bleach, never reduce contrast, and never cover the image with a grey or milky veil. Do not flatten skin, hair, eyes, clothing and background into one cream or grey-pink tone: restore the skin, hair, eye and local warm/cool relationships that are really there, at least to the degree of legibility they already have in the source. This is colour matching, not global enrichment: do not add vivid colours, filters, strong warm or cool shifts or extra highlights that the source does not have, and do not delete or dilute colours that it does have.
5. Preserve the source's highlight placement and extent. Do not let highlights spread across the whole face, hair, clothing or background; white areas stay white, but do not bleach the neighbouring pinks, browns, blues and skin tones.
6. Repair only defects that are actually present, as locally as possible, blending into the surrounding brushwork, lines and colour.
7. Do not beautify, change the face, change the hairstyle, change the outfit, change the age, change the expression, change the gaze, change the pose or redesign the character.
8. Do not copy any character, face, hairstyle, clothing, prop, background or composition that is not already in the source.
9. Keep the same aspect ratio, crop and subject placement as the source. Do not zoom, re-frame or move the subject.

WHAT YOU MAY DO
- Connect contours that are clearly broken, and merge scattered micro-strokes into the intended longer stroke, WITHOUT making lines thicker, darker, more uniform or more numerous than the source style uses.
- Clean accidental colour bleeding, banding, halos and compression artefacts.
- Make existing shape boundaries read more clearly while keeping the source's softness, grain, translucency and edge variation.
- Correct clearly malformed local anatomy (fingers, joints, overlaps) only where the surrounding image gives enough evidence, and keep whatever the hands are wearing exactly as drawn.

LINE STRUCTURE WITHIN THE SOURCE'S OWN MEDIUM
While keeping the source's character, composition, features, materials and colour relationships unchanged, repair only the MAIN structural boundaries that are visibly broken. Use continuity only for the outer hair silhouette, major lock boundaries, jaw, neck-shoulder boundary, clothing silhouette and clear foreground occlusion edges. Do not force continuity through painterly texture, facial shading, eyelashes, iris texture, watercolour strokes or ambiguous overlaps. Treat secondary strokes as intentionally allowed to remain broken, translucent, tapered and irregular. A missing connection is not a defect unless both endpoints and the intended path are clearly visible in the source.
Express the hair silhouette, major hair locks, fringe grouping, upper and lower eyelids, nose-mouth line, jaw, neck-shoulder line, clothing outline and foreground occlusion edges with a small number of continuous medium-fine strokes. The same main boundary should stay as one coherent curve or one continuous stroke rather than being interrupted by dense short strokes, grain, noise or repeated re-drawing.
Inside each hair lock keep only a few directional medium-long strokes; do not split every strand into many fragments. Lines should vary in weight naturally, take their colour from the local tone, and blend with the colour masses and soft edges. Keep the painterly quality, translucent layering and soft edges, but raise the coherence of the main contours.
Avoid: uniform black contours, hard black outlines, cel shading bands, flat fills, comic ink lines, vector line art, over-sharpening, over-crisp cartoon edges, outlining every single hair, dense short dashes, broken burrs, repeated outlines and high-frequency noisy edges.

FORBIDDEN
Genericised features; standardised eyes; over-sharpening; over-smoothing; cel conversion; style change; added heavy black outlines; bleaching or pale-veil wash; desaturation of the source's colours; redoing the whole image; changing the composition; changing the character's identity; adding objects; deleting existing important detail.

If there is no clearly visible defect, the output should stay as close to the input as possible instead of becoming a new artwork. Output one image, same aspect ratio. No text, no border, no watermark.

OPTIONAL MULTI-IMAGE MODE (only when a second style reference is provided)
Image 1 is the SOURCE IMAGE and the only truth for content and identity: it decides the character, face, eyes, features, hairstyle, pose, clothing, props, background, composition, crop, colour and value relationships.
Image 2 is a STYLE REFERENCE that supplies rendering language only: brushwork organisation, edge treatment, line character, material handling and overall painting method.
Never copy from Image 2: its character, face shape, eyes, pupils, lashes, hairstyle, clothing, pose, props, background, composition or identity. Where Image 1 and Image 2 conflict on content, features or composition, Image 1 always wins. Image 2 must not override Image 1's facial drawing or colour relationships. Borrow Image 2's rendering language only where it does not damage Image 1's content.
```

**备选固件**：
- `repaint-system-conservative-v4.md`（6181 字符）：清线更狠，A0 基底上段长 109.1（v5 是 101.9），但饱和更低（101.9 vs 113.2）。
- `repaint-system-style-neutral.md`（4030 字符）：风格中性，实测最弱（段长 96.3）。

### 6.5 重绘提示词的附加尾巴（v5 之后追加的两段）

只要 `reference_mode` 带画风图，`run_pipeline` 就会在固件正文后追加下面这两段（全文）：

```text


CHARACTER-INTRINSIC FEATURES ARE NOT PART OF THE STYLE: keep the subject's OWN hair colour, eye colour, skin tone, body proportions and outfit colours exactly as described by the content; the style reference must only change the rendering language (palette temperature of the environment, brushwork, line character, edge treatment, texture), never the subject's intrinsic colouring or design.

REFERENCE ROLES: Image 1 is the SOURCE and the only truth for content, identity, pose and composition. Any further image is a STYLE REFERENCE that supplies rendering language only (palette, line character, brushwork, edge treatment, material handling). Never copy the style reference's character, features, outfit, pose, props, background or composition, and never let it override the source's content.

STYLE LANGUAGE (from the reference image):
- Use clean tapered linework with dark navy-blue or deep indigo contours instead of pure black.
- Keep primary silhouette lines at roughly 0.20-0.40% of the image width, secondary structural lines at 0.10-0.20%, and micro-detail lines at 0.04-0.10%, never exceeding about 1-2% of the eye height.
- Taper every stroke toward its endpoint with sharp elegant finishes; no blunt mechanical line caps and no uniform line weight.
- Vary line weight continuously according to depth, overlap, lighting and material: thicker for silhouette, overlaps and facial framing; thinner and lighter for hair strands, lace, folds, embroidery and distant motifs.
- Build the palette from saturated sapphire, midnight blue, turquoise, rose pink, pearl white and cool lavender; keep deep blue-violet anchors so the image never becomes pale or grey.
- Create translucency through layered saturated colour, soft subsurface pinks, controlled cool shadows and selective luminous highlights rather than by whitening the whole image.
- Preserve real chroma in midtones and shadows; avoid muddy grey shadows, desaturated midtones and excessive white wash.
- Keep important contours crisp and controlled while atmospheric background elements and distant fabric use softer edges.
- Use selective bloom only around highlights, eyes, gemstones and satin; avoid global haze or washed-out lighting.
- Paint hair as large flowing masses first, then grouped locks with tapered internal strands, sharp specular streaks and varied spacing; avoid evenly spaced parallel lines and tangled spaghetti-like strokes.
- Render lace as small connected scallops and negative-space cutouts with consistent rhythm and clear spacing.
- Render ribbons, straps and shoelaces as clean overlapping bands with visible thickness, tension, knots, loops and cast shadows; keep them continuous and countable.
- Simplify tiny accessories into readable graphic shapes with a few high-contrast accents instead of noisy micro-lines.
- Draw fabric folds as selective long curves following tension and gravity, supported by deep occlusion shadows and narrow luminous ridges.
- Avoid pure black outlines, uniform line weight, blunt stroke endings, rough sketch residue and heavy comic contouring.
- Avoid over-smoothed vector edges, plastic-looking skin, generic airbrushed gradients and textureless flat fills.
- Avoid tangled hair lines, random lace noise, illegible straps, merged ribbons, broken shoelaces and indistinguishable accessory details.
- Avoid excessive bloom, blown-out highlights, chromatic clipping, grey veils and dirty colour mixing.
- Keep white and pale fabrics structurally readable: preserve internal shading separation, fold shadows and lace openings, and never clip the whites into a flat blown-out area.
```

> `CHARACTER-INTRINSIC FEATURES ARE NOT PART OF THE STYLE: …` 那一句在 `STYLE_REF_ROLE_IN_REPAINT` 里，用来防止「画风参考图的角色特征（发色/瞳色/服装）被搬过来」。

### 6.6 一个真实的完整首图提示词（tinkle）

**注意内容锚的两种取法**（同一个分析产物，长度差 2.5 倍，效果没做过对照实验，见 §8.6）：
- **GUI 默认**：`prompt_context.refined_prompt`（本次端到端那次 ≈1730 字符）→ 提示词总长 **4643 字符**；
- **本批量测试用的**：分析产物里的 `english_description`（4406 字符）→ 提示词总长 **8569 字符**（下面这份）。

长度：**8569 字符**；段落构成：画风 665 + 内容锚 4406 + 排除句 + 条款 19 条（source=`entry`）；参考图 1 张（画风图）。

```text
Palette: sapphire, midnight indigo, cobalt, turquoise, pearl white, rose pink, lavender
Lighting: high-key cool key, small shaped speculars, clear highlight/shadow separation
Brushwork: smooth cel base, soft blends, airbrushed light, transparent glazes
Edges: tapered dark-navy linework, bold on silhouette, hairline-thin on interior detail
Texture: glassy layered depth, crisp lace cutouts, satin sheen, restrained grain
Composition density: centred figure, pale airy background, generous negative space
Detail level: crisp focal rendering, clean lace, ribbons and shoe detail
Avoid: pure black outlines, uniform heavy lines, grey shadows, pale wash, haze, tangles

A refined, highly detailed full-body illustration in a romantic rococo aesthetic, composed vertically in an elegant 2:3 frame. The scene combines doll-like grace, antique interior design, and luminous pastel luxury, with the character placed slightly left of center according to the rule of thirds. She is a young woman with long, warm chestnut-brown hair, smooth rounded bangs, and softly curled lengths cascading over both shoulders and down her back. A delicate silver-white tiara rests above her fringe, catching the filtered daylight. Her bright turquoise-blue eyes are clear and finely rendered, framed by graceful lashes, and her gentle closed-mouth smile suggests amused concentration as she tries to maintain her balance. Her face is fully visible and carefully detailed, with soft rosy cheeks, a small refined nose, and an attentive gaze directed toward the viewer. The background is a richly layered antique salon. To the left, a white classical pedestal supports a vintage globe, while open wooden shelves behind it hold porcelain ornaments, old books, and folded fabrics partly veiled by translucent lace curtains. On the right, dark wooden drawers stand beneath a framed pastoral painting, an antique lantern, slender candlesticks, and a pale flower arrangement. Tall windows beyond the curtains allow soft morning light to pass through floral lace patterns, scattering delicate shadows across the walls and floor. A circular rug embroidered with white blossoms, curling leaves, and muted blue details anchors the furniture, while tiny dust motes glow in the sunbeams. The layered background gives the impression of a quiet story interrupted by a sudden, graceful movement. She wears an elaborate ivory-white rococo dress with a fitted, boned-looking bodice, a high ruffled neckline, and a large satin bow at the throat. A translucent lace panel edged with scallops decorates the upper front without revealing excessive skin, and small pearl-like accents are arranged along the center. Her gathered puff sleeves connect to long, semi-transparent lace gloves that reach toward her upper arms. The gloves are embroidered with curling floral patterns and set with tiny jewel-like beads that sparkle whenever her fingers move. Fine cross-lacing runs along the bodice and continues across the sleeve cuffs, adding a delicate structured detail. Her short layered skirt is built from overlapping flounces, scalloped lace, fine ruffles, and gathered fabric, with embroidered white flowers matching the gloves and stockings. She wears semi-transparent lace over-the-knee garter stockings with a convincing sheer silk texture, embroidered with tiny roses, vines, and rococo swirls that echo the dress. Decorative garter ribbons and crossed lacing hold the stockings in place beneath the skirt. Her pointed ivory high heels have noticeably tall, elegant heels, each decorated with a miniature dangling charm, a satin bow, and a small jewel at the toe. Additional cross-laced ribbons wind around the ankle straps and glimmer against the pale shoes. The pose creates a lively narrative rather than a static portrait. She sits diagonally on the small wooden chair, but one hip has shifted toward the edge as if she has just reacted to a sound from the shelves. Her torso remains upright yet tilts slightly away from the chair’s center. Her left hand is raised beside her cheek, fingers delicately pinching the throat bow as though steadying it, while her right hand reaches down toward the chair’s side and layered skirt, almost touching the wooden seat for support. Her thighs are visibly separated in an energetic, asymmetrical pose. One leg lifts forward and outward toward the right, the pointed shoe hovering above the floral rug, while the other leg bends downward and slightly left; their ankles cross only lightly, producing a sweeping diagonal line. She appears to be balancing on the edge of the chair, with one heel suspended and her center of gravity shifted, creating the impression that she may spring up or lose balance at any moment. The skirt flares from this motion, brushing the chair edge without touching the nearby furniture. Gentle highlights define the lace, jewels, silk stockings, and polished shoes, while soft shadows beneath the chair and rug reinforce the depth. The final mood is theatrical, innocent, elegant, and dreamlike, like a frozen moment from a charming rococo tale.

Use the attached reference image for its palette, lighting, brushwork and rendering grammar only. Do not copy its character, face, hairstyle, hair colour, eye colour, outfit, pose, props or composition; keep the subject, colours of the subject's own design and the scene exactly as described above.

CHARACTER-INTRINSIC FEATURES ARE NOT PART OF THE STYLE: keep the subject's OWN hair colour, eye colour, skin tone, body proportions and outfit colours exactly as described by the content; the style reference must only change the rendering language (palette temperature of the environment, brushwork, line character, edge treatment, texture), never the subject's intrinsic colouring or design.

RENDERING LANGUAGE (follow exactly):
- Use clean tapered linework with dark navy-blue or deep indigo contours instead of pure black.
- Keep primary silhouette lines at roughly 0.20-0.40% of the image width, secondary structural lines at 0.10-0.20%, and micro-detail lines at 0.04-0.10%, never exceeding about 1-2% of the eye height.
- Taper every stroke toward its endpoint with sharp elegant finishes; no blunt mechanical line caps and no uniform line weight.
- Vary line weight continuously according to depth, overlap, lighting and material: thicker for silhouette, overlaps and facial framing; thinner and lighter for hair strands, lace, folds, embroidery and distant motifs.
- Build the palette from saturated sapphire, midnight blue, turquoise, rose pink, pearl white and cool lavender; keep deep blue-violet anchors so the image never becomes pale or grey.
- Create translucency through layered saturated colour, soft subsurface pinks, controlled cool shadows and selective luminous highlights rather than by whitening the whole image.
- Preserve real chroma in midtones and shadows; avoid muddy grey shadows, desaturated midtones and excessive white wash.
- Keep important contours crisp and controlled while atmospheric background elements and distant fabric use softer edges.
- Use selective bloom only around highlights, eyes, gemstones and satin; avoid global haze or washed-out lighting.
- Paint hair as large flowing masses first, then grouped locks with tapered internal strands, sharp specular streaks and varied spacing; avoid evenly spaced parallel lines and tangled spaghetti-like strokes.
- Render lace as small connected scallops and negative-space cutouts with consistent rhythm and clear spacing.
- Render ribbons, straps and shoelaces as clean overlapping bands with visible thickness, tension, knots, loops and cast shadows; keep them continuous and countable.
- Simplify tiny accessories into readable graphic shapes with a few high-contrast accents instead of noisy micro-lines.
- Draw fabric folds as selective long curves following tension and gravity, supported by deep occlusion shadows and narrow luminous ridges.
- Avoid pure black outlines, uniform line weight, blunt stroke endings, rough sketch residue and heavy comic contouring.
- Avoid over-smoothed vector edges, plastic-looking skin, generic airbrushed gradients and textureless flat fills.
- Avoid tangled hair lines, random lace noise, illegible straps, merged ribbons, broken shoelaces and indistinguishable accessory details.
- Avoid excessive bloom, blown-out highlights, chromatic clipping, grey veils and dirty colour mixing.
- Keep white and pale fabrics structurally readable: preserve internal shading separation, fold shadows and lace openings, and never clip the whites into a flat blown-out area.
```

**内容锚（第 [2] 段）的来源**：分析产物 `data/20260921/sucai/<…>.json` 的 `english_description` 字段（本次 4406 字符）。GUI 默认用 `prompt_context.refined_prompt`（同源文本）。

## 7. 已知坑与运维要点（踩过的都在这）

### 7.1 中转站 new.aigc2d.com

- **`/v1/images/generations` 的 `image` 字段会被部分通道拒绝**：返回 `{"error":{"message":"Unknown parameter: 'image'"}}`。中转站按请求随机挑上游，同批请求 3 成功 2 失败。现已在 `generate_image_aigc2d_gpt` 里做了 **重发（2 次）+ 回退 `/images/edits`** 两级兜底，回退时日志会写「已回退到编辑端点重发」，看到这行说明这次的首图语义是 edits（产物可能偏「编辑参考图」）。
- **504 / ConnectionReset 偶发**；`max_retries` 默认 1，重试逻辑只对 429/5xx 生效。长提示词 + 大 base64 图更容易触发。
- **拦截（content_blocked）**：局部重绘的裁切图偶发被拦，原始返回
  `{"error":{"message":"[CONTENT ERROR] content blocked by upstream safety policy","type":"content_blocked","code":"CONTENT_BLOCKED"}}`
  （2026-09-24 21:08 实际发生过一次，拦的是 `subject_no_face,shoes,waist,thigh` 的整身裁切）。
  流水线把该区域当「未返回图片」跳过，`pipeline-manifest.json` 里 `local.status=failed`，
  产物名变成 `…-final-partialrp+sline-rp+…`（**带 `partial` = 缺工序的产物**）。
  历史上的「审核 451」是同一类问题的另一种表现。
- **模型路由**：gpt-image/文本 → 分组 `Openai-Gpt-1`(×0.88236)，Gemini 图片 → `Discounted-Banana-1`(×0.110295)，计费见 `utils/cost_estimate.py`。实测价：首图 high $0.155 / medium $0.039，Gemini 重绘 2K $0.036，全工序（含四区）≈ $0.3/张。

### 7.2 模型侧行为

- gpt-image-2 的 `input_fidelity` 不可调，**输入图一律高保真**、模型倾向把参考图当「要编辑的原图」→ 必须显式写排除句。
- 尺寸只有 `1024x1024` / `1536x1024` / `1024x1536` 三档；`quality` high 的细节 token 约是 medium 的 5 倍。
- Gemini 重绘**会随机重排画面**（同一输入、同一提示词，3 次里 1 次重排主体）→ 局部重绘的「保脸排除区」会把原脸留下、新身体贴上去 = **双人鬼影**。这是当前最大的画面风险，见 §8。

### 7.3 工程纪律（本仓库硬性要求）

- 根目录**不允许**堆脚本；一次性脚本放 `tools/` 或跑完即删。
- 测试**不要**往 `data/<日期>/` 写东西：`IMAGE_MAKER_TEST_OUTPUT=1` 会改道 `data/test-result/<日期>/`（`utils/output_isolation.py`，`run_pipeline` 已接入）。
- 跑 pytest **不要套管道/重定向**（`python -m pytest -q -p no:cacheprovider` 直接跑），本机沙箱下会给原生进程开匿名管道并报 `拒绝访问`。
- **本机出网只对 python 放行**：`curl.exe` / `Invoke-WebRequest` 都不通，验证网络要用 python 或 DSH 的 `download_file` / `web_fetch`。
- 密钥只在 `.env`（`IMAGE_MAKER_AIGC2D_API_KEY` 等）；`conf/config.json` 里 `api_key` 留空。

### 7.4 提示词长度

软上限 2000 字符（官方文档：超过后画风参考图效果下降）/ 硬上限 15000（中转站断连）。**但带 19 条 tinkle 条款时 4.6k 字符反而最好** —— 条款补的是「线条怎么画、颜色锚点、不要画成什么」，属于必要约束，不要为了满足软上限砍掉。

### 7.5 版本控制

- `conf/config-styles.json` 是 **gitignore**；版本化副本在 `submodules/image-maker-artstyle/`（`submodules/` 也被父仓库 ignore，它是独立仓库）。改画风后必须：`python tools/sync_styles_to_submodule.py` 然后在子模块里单独 commit。
- `docs/gpt-image-tid-style/` 下 **800MB 实验原图不进库**（`.gitignore` 挡了 `**/*.png|jpg|jpeg`），结论都在 `.md`，对照页在 `.html`（几个 MB，已进库）。

### 7.6 指标 ≠ 画面（务必看图）

本轮矩阵里出现过 段长 107.3 / 端点 19.55 的「最优」样本其实是**双人鬼影**。任何自动优化都要配一个人工/视觉检查环节（例如把候选拼成接触表再看，见 `data/20260924/pipeline-steps/rp-matrix/_contact-sheet*.jpg`）。

## 8. 未解决问题与建议路线（交给优化的你）

### 8.1 最大的短板：首图的线条密度达不到参考图

- 事实：参考图 tinkle 段长 **122.5**，而本轮 5 条画风的首图段长只有 **69–90**，加完全部后处理也只到 **80–188**（且后处理主要靠「结构线叠加 + 加墨」把既有边压深，并不能真正画出更多长线）。
- 可能方向：① 让首图阶段就产出更多长结构线（提示词层面加「线条密度/长线」约束，或换 `gpt-image-2.5-flare/sunburst` 等模型对比）；② 首图后用**整幅重绘 + v4 固件**拉线条（A0 基底实测 109.1）；③ 研究「线稿优先」两条腿：先用 Gemini 出一张线稿再上色？④ 用 SD/WebUI 的 lineart 模型做结构线来源，再让 Gemini 按结构线重绘。

### 8.2 首图色相的随机性（当前最影响「贴画风」）

- 事实：同一条件跑 3 次，N 基底饱和 55 / 68.5 / **134.5**（跨了一倍），A0 基底却稳定在 100–120。推测原因：N 是白/棕低饱和基底，模型有时抓住画风参考图的蓝紫调、有时退回源图的白。
- 方向：① 首图加「palette anchor 句」（把画风参考图的主色写进提示词，参考 §6.2 里的 `Build the palette from saturated sapphire…`）；② 用 `tools/style_ref_stats.py` 算出参考图的客观主色，自动写进首图提示词（比人工写更稳）；③ 首图一次出 N 张（`n=2/3`）再用指标 + 视觉挑一张。

### 8.3 局部重绘的重排/鬼影

- 事实：`subject_no_face` 单区域 3 次里 1 次双人鬼影；四区链稳但会提亮画面。
- 方向：① 贴回前做**对齐校验**（比较 patch 与源裁切的主体框/质心，偏移超过阈值就丢弃或平移对齐）；② 给局部重绘也加「不要移动/缩放主体」的强约束（现在只有固件里的通用约束）；③ 用 `full`/`subject` 之外的自定义坐标框（`local_repaint_composite --crop x0,y0,x1,y1`）避开脸部排除区的副作用；④ 羽化/遮罩层面：当前遮罩来自 GrabCut 主体检测，可以把「保脸排除区」换成「只排眼睛嘴」的窄框。

### 8.4 后处理参数的进一步搜索

现在 `tone`（contrast 1.00 / chroma 1.10 / highlight 0.85 / sat_scale 1.00）与 `ink`（target_sep 8 / max_darken 40）只是**单点验证**，没有做过网格搜索。这两步是本地、免费、确定性，**性价比最高**，值得先做参数扫描（例如 contrast 0.95/1.00/1.05 × highlight 0.75/0.85/0.95 × ink target 6/8/10/12）。
CLI: `python tools/analysis_gpt_run.py --json <分析JSON> --style tinkle --steps repaint,structure,local --region subject_no_face --extra-region shoes,waist,thigh --tone --tone-target style --ink --ink-target 8`

### 8.5 其它值得试的
- **首图参考图的另一种用法**：用无主体裁剪 / style board 当参考图（`docs/gpt-image-tid-style/luna-cases/` 有素材，早期实验显示能提升「纯风格感」但会放大留白）。
- **`--use-source-base` 路线**：跳过首图、直接拿分析素材图进工序（构图/姿势沿用原图，只被画风重绘）——早期记录线条连通性更好，但没有在本轮配方下重测。
- **模型对比**：`gpt-image-2` vs `gpt-image-2.5-flare/sunburst`（edits 垫图实测 flare 约 30s、sunburst 约 180s）。
- **重绘参考图组合**：`both`（线锚+画风）这次线条塌了，但它的 HSV 最高（0.509）—— 可以试「降强度线锚」（`build_line_anchor(darken=…)`）看能不能两全。

### 8.6 没做过对照、但可能影响很大的三件事

1. **内容锚长度**：GUI 用 `refined_prompt`（≈1730 字符 → 提示词 4643）vs 批量测试用 `english_description`
   （4406 字符 → 提示词 8569）。两者都能出图，但没有做过「同一画风、同一素材、只变内容锚长度」的对照
   —— 早期文档（`README.md` §4.16/§4.18/§4.19）说长文本会压住画风参考图，值得复测。
2. **首图 `n>1` 选优**：`generate_image_aigc2d_gpt(n=2)` 一次出两张（成本 ×2），用指标 + 视觉挑一张，
   可以显著缓解 §8.2 的方差问题。
3. **局部重绘的 `feather` / `scale` / 分辨率**：现在固定 feather 48 / 2K，`auto_repaint_scale` 自适应；
   没有扫过（feather 24/48/64 × 1K/2K）。

## 9. 复现命令与实验脚本片段

### 9.1 无头 CLI（最快复现整条链）

```bash
python tools/analysis_gpt_run.py \
  --json "data/20260921/sucai/<分析产物>.json" \
  --style tinkle --quality high --size auto \
  --steps repaint,structure,local \
  --region subject_no_face --extra-region shoes,waist,thigh \
  --repaint-ref style --feather 48 \
  --tone --tone-target style --ink --ink-target 8 \
  --first-pass-mode generate
```

### 9.2 GUI 等价链路（用户要求的「app.py 代码功能」）

`QT_QPA_PLATFORM=offscreen` + `IMAGE_MAKER_SKIP_ONNXRUNTIME_PRELOAD=1`，构造 `SingleAnalyzerWidget`，勾上 `gen_channel_gpt` + 五个工序开关，调 `trigger_image_generation("refined", prompt_bundle={..."analysis_json_path": <JSON>})`，然后手动 `qapp.processEvents()` 轮询到线程结束（QTimer/exec 在无头下不触发）。

### 9.3 只测重绘阶段的矩阵片段（本轮用的写法）

```python
from utils import post_process as pp
outs = pp.run_pipeline(
    [base_image],
    {"repaint": {"enabled": True, "resolution": "2K", "reference_mode": "style"}},
    firmware="prompts/gpt-image-optimize/repaint-system-conservative-v5.md",
    final_dir=<输出目录>, work_dir=<中间目录>, resume=False,
    style_ref_path=<画风参考图>, style_clauses=<条款列表>)

# 覆盖面（多区域链）
outs = pp.run_pipeline([repainted],
    {"structure": {"enabled": True, "strength": 0.5},
     "local": {"enabled": True, "regions": ["subject_no_face","shoes","waist","thigh"],
               "region": "subject_no_face", "feather": 48, "resolution": "2K", "detail_boost": False}},
    final_dir=<输出目录>, work_dir=<中间目录>, resume=False)

# 免费工序：色调校准（目标=画风图）+ 加墨
outs = pp.run_pipeline([img],
    {"tone": {"enabled": True, "reference_path": <画风参考图>, "contrast": 1.0, "chroma": 1.1,
               "highlight_strength": 0.85, "skin_warm": 0.0, "sat_target_scale": 1.0},
     "ink": {"enabled": True, "target_sep": 8.0, "max_darken": 40.0}},
    final_dir=<输出目录>, work_dir=<中间目录>, resume=False)
```

### 9.4 指标与审计

```bash
python -c "import sys; sys.path.insert(0,'tests'); import style_render_metrics as M; print(M.analyze('<图>'))"   # 单张指标（900px 归一）
python tools/convert_styles_gpt.py --check      # 画风 prompt_gpt 合规 + 条款覆盖（手写/派生/无）
python tools/style_ref_stats.py --all           # 每条画风参考图的客观统计（亮度/饱和/主色/边缘密度）
python tools/sync_styles_to_submodule.py        # 本地画风 → 版本化文件
python -m pytest -q -p no:cacheprovider         # 全量测试（当前 414 passed）
```

## 10. 索引

**文档**：`BEST-PIPELINE.md`（§一~§二十九，主文档）/ `README.md`（更早的画风实验 §4.x）/ `prompt-gpt-rules.md`（`prompt_gpt` 规则）/ `BEST-PROMPT.md`（早期最优提示词模板）
**对照页**：`compare-024958-vs-styleref.html`（A vs B + 四格 + 端到端）/ `repaint-matrix.html`（重绘矩阵）/ `style-batch.html`（5 画风）/ `option-matrix.html`（更早的 10 组合 GUI 矩阵）
**日志**：`log/rp-matrix.txt`、`log/rp-repeat.txt`、`log/rp-final.txt`（重绘矩阵与方差）、`log/style-batch.txt`、`log/style-batch-retry.txt`（5 画风）、`log/e2e-tinkle.txt`（端到端）、`log/style-batch-table.txt`（5 画风指标表）；请求/响应原始 dump 在 `log/2026-09-2x.log` 与 `data/<日期>/…server_response_*.json`
**指标脚本**：`tests/style_render_metrics.py`（线稿/可读性指标，`analyze()`）；`tests/calc_style_similarity.py`（CLIP + HSV 画风贴近度，需 CLIP 权重）
**关键代码**：见 §2.1；**默认配方**见 §4；**提示词全文**见 §6。
