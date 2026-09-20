# 批次 C · Gemini 3 Pro Image 重绘：修手 + 连通发丝

- 日期：2026-09-20
- 目的：确认 `gemini-3-pro-image-preview` 的编辑能力，并找到"对 gpt-image 产物做整体重绘提线"的有效 prompt
- 源图（**全程同一张**）：`data/20260919/anime_lolita_cmp_v3/gpt-image-2_output_193138_0_d09c70.png`（1024×1536）
- 输出：2K、2:3（→ 1696×2528），单张 30–100s

## 官方依据（2026-09 抓取）

来源：[Gemini API · Image generation](https://ai.google.dev/gemini-api/docs/image-generation)

| 能力 | 官方说明 |
|---|---|
| 图生图 / 编辑 | 图片作为 `input` 一部分传入（REST: `type:"image"` + base64）；与文本混排 |
| 多轮编辑 | "Multi-turn conversation is the recommended way to iterate on images"，用 `previous_interaction_id` 接续 |
| 编辑场景 | 官方示例含「把铅笔草图变成成品车照片，保留线条但升级渲染」——与本方案同类 |
| 参考图上限 | `gemini-3-pro-image`：高保真 5 张 / 合计 14 张；另有 3 张风格参考、5 张角色一致性参考 |
| 分辨率 | 1K / 2K / 4K（默认 1K），字段 `response_format.image_size`（必须大写 K） |
| 宽高比 | 默认**匹配输入图**；可用 `aspect_ratio` 覆盖 |
| 思考模式 | 先出中间 "thought images" 精修构图（不计费） |
| 水印 | 所有输出含 SynthID |

本项目通道：`modules/others/api_backend.py::generate_image_aigc2d`
→ `POST https://new.aigc2d.com/v1beta/models/gemini-3-pro-image-preview:generateContent`，
参考图走 `inline_data`，参数走 `generationConfig.imageConfig.{aspectRatio,imageSize}`，
参考图超 2048px 自动压缩（`to_base64_compressed`）。

## 变体与结果

| 变体 | 指令结构（全文见 `data/p_C*.txt`） | lap_var↑ | soft↓ | edge/soft↑ | frag | colors | 手 | 发丝(右侧) |
|---|---|---|---|---|---|---|---|---|
| A1（v1，无修复指令） | "重绘成干净赛璐璐动画" + 画风 5 条 | 2016 / 2628(复跑) | 32.7 / 20.1% | 0.354 / 0.699 | 0.40 / 0.44 | 810 / 726 | **复制了源图的坏手** | 一般 |
| A2 | "只要求线条连贯、细节清晰" | 1413 | 37.4% | 0.260 | 0.30 | 941 | 无改善 | 温和改善 |
| C1 | 长版：修复 + 画风（一次性全塞） | 1684 | 35.5% | 0.301 | 0.37 | 852 | 一般 | 一般 |
| C2 | 短版：修复优先 + 画风在后 | 2506 | 24.8% | 0.569 | 0.42 | 811 | 可读 | 好 |
| C3 | 分步（Step1 手 / Step2 头发 / Step3 其余） | 2002 | 27.3% | 0.430 | 0.41 | 788 | 清晰偏硬 | 好 |
| **C4** | **负面清单优先**（禁止融合/多指/缺指…） | **2336** / 2059(复跑) | **18.2** / 27.6% | **0.734** / 0.431 | 0.41 / 0.41 | 857 / 679 | 好 | 好（复跑稳定） |
| **C5** | 画风段完整前置 + 修复做编号项 | **2993** | 22.2% | 0.668 | 0.47 | 809 | 很好 | 很好 |
| C6 | 画风段单独成段置于最前 | 2575 | 31.8% | 0.408 | 0.42 | 825 | 好 | 好 |
| **C7** | **C4 结构 + 分优先级（手>发>其他）** | 2268 | 26.3% | 0.503 | 0.44 | 819 | **最好** | **最好** |

> `A1` 与 `C4` 各有一列复跑值，用来区分"真实提升"与"抽卡噪声"（见 `metrics.md` 第 3.4 节）。

## 关键发现

1. **"重绘"与"修错"必须显式拆开说**——这是本批次最大的增量。
   A1 只说"重绘成赛璐璐"，模型会**忠实复制源图的错误**（手糊、发断全部继承）。
   必须写：`the source hands are malformed - reconstruct both instead of copying them`。
   对应负面清单：`no fused, melted, duplicated, missing or extra digits, no finger merging into the palm/wrist/sleeve`。
2. **负面清单比正面形容词管用**：C4 的手部改善主要来自那串 `no fused/melted/...`。
3. **要求模型自检能提高命中率**：
   `Before finishing, count five fingers on each hand and trace the long strands on the right side for continuity`。
4. **画风段与修复段的顺序有影响**：C5（画风完整前置 + 修复编号项）明显优于 C6（画风独立成段放最前）：2993 vs 2575、soft 22% vs 32%。
5. **别一次要求太多**：C1 同时强调"修手 + 修发 + 保画风 + 全画面锐化"，结果最软（1684）。
   集中在手与发丝上反而全项最优。
6. **残留问题**：C4/C5 的树皮、湖面反射、远景比源图**简化**了（"全画面平涂化"的副作用）；
   `frag` 全部升高到 0.41–0.48，但这是平涂线稿在像素级的正常表现，肉眼看到的是更干净的线。

## 复现

```powershell
$py = "C:\Program Files\Python310\python.exe"
$src = "data\20260919\anime_lolita_cmp_v3\gpt-image-2_output_193138_0_d09c70.png"
& $py -u docs\gpt-image-optimize\tools\run_gemini_repaint.py `
    --source $src --prompt-file docs\gpt-image-optimize\data\p_C7_c4refined.txt `
    --prefix C7 --subdir gemini_repaint_C --resolution 2K --aspect 2:3
```

产物：`data/20260919/gemini_repaint_C/`、`data/20260920/gemini_repaint_D/`（C4 复跑、C5、C6、C7）
指标：`data/metrics_C.json`、`data/metrics_D.json`
可视化：`data/20260920/repaint_iter_report/index.html`；
局部放大：`data/20260920/gemini_repaint_D/inspect_D/`（手 3× / 发丝 2.2×）
