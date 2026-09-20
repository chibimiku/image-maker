# 批次 A · gpt-image-2 vs gpt-image-2.5（同 prompt 同构图）

- 日期：2026-09-19
- 站点：`new.aigc2d`（`apis.aigc-2d-gpt`），通道 `POST /v1/images/generations`
- 固定参数：`size=1024x1536`、`quality=high`、`output_format=png`、`n=1`、无参考图
- 目的：摸清 gpt-image 家族在 anime 风格上的线条/细节基线，决定后续是否值得用 Gemini 重绘补救

## A-1 三模型同 prompt

Prompt（短版，一行，488 字符，见 `data/prompt_line_v3.txt`）：

```
Anime illustration style, a girl with long pastel pink hair and large brown eyes sitting on a thick mossy tree branch above a calm lake on a sunny afternoon, wearing a pink and cream lace frilled sweet Lolita one-piece dress with a big ribbon headdress, dappled golden sunlight, blue sky, pastel colors, clean cel shading, crisp continuous line art, sharp fine detail, medium full body shot.
```

| 模型 | lap_var↑ | tenengrad↑ | 边缘密度 | frag↓ | iso↓ | hf_ratio↑ | colors |
|---|---|---|---|---|---|---|---|
| gpt-image-2 | 1181 | 7420 | 0.1148 | 0.0202 | 0.0593 | 0.00464 | 996 |
| **2.5-flare** | **1506** | **9020** | **0.1354** | 0.0236 | 0.0607 | **0.00648** | 1170 |
| 2.5-sunburst | 953 | 7522 | 0.1244 | **0.0153** | **0.0514** | 0.00460 | 607 |
| 2.5-flare-c | 718 | 5534 | 0.0844 | 0.0148 | 0.0520 | 0.00282 | 1170 |

梯度剖面（细线锐利度）：

| 模型 | p50 | p90 | p99 | 梯度>20% 占比 | 梯度>30% 占比 |
|---|---|---|---|---|---|
| gpt-image-2 | 0.0383 | 0.1932 | 0.4760 | 9.40% | 4.07% |
| **2.5-flare** | **0.0477** | **0.2454** | **0.5289** | **14.09%** | **6.63%** |
| 2.5-sunburst | 0.0513 | 0.2210 | 0.4845 | 12.04% | 5.05% |
| 2.5-flare-c | 0.0300 | 0.1706 | 0.4360 | 7.64% | 3.17% |

### 观察

- **2.5-flare**：线条最锐最实，蕾丝镂空、发丝可数；背景更"画"，边缘略锯齿。**综合细节第一。**
- **2.5-sunburst**：细节量不输 flare，**线更连贯**（碎片率最低），会给远景加村庄/雪山/天鹅等景深；
  整体"厚涂 + 空气感"，缺 flare 的刀刻感。
- **2.5-flare-c**（按次计费通道）：同参数明显更软，**不用于线条对比**。
- 用量参照：gpt-image-2 `output_tokens=1105`；两个 2.5 都是 `1372`。耗时 2 约 30–35s，2.5 约 30–55s。

### Safety 行为（重要）

同一份**详细** Lolita prompt（含年龄与袜类描述）：

- `gpt-image-2` 稳定通过；
- `gpt-image-2.5-flare` / `-flare-c` 连续被拒 6 次：`upstream_error: rejected by the safety system`；
- 压成一行、缩短、去掉年龄与袜类描述后，**一次通过**。
- `2.5-flare-c` 的响应会带 `"moderation": "auto"` 与 `revised_prompt` 字段。

→ 写入代码/文档的口径：**2.5 的 moderation 明显比 2 严**，prompt 要更克制。

## A-2 细节预算（解释"为什么糊"）

多次实验里混入过一张「极简 prompt」产物（无角色、颜色、场景描述），它的指标与满构图完全不同：

| 样本 | 强边(>20%) | 强边像素数 | 平坦(<2%) | 渐变(2–10%) | colors |
|---|---|---|---|---|---|
| 极简 prompt（近景特写 + 纯色背景） | 5.82% | 91,575 | **71.3%** | **16.4%** | 498 |
| 满构图 gpt-image-2 | 7.28% | 114,484 | 42.3% | 37.8% | 1095 |
| 满构图 gpt-image-2（更细 prompt） | 9.40% | 147,850 | 36.2% | 38.8% | 996 |
| 满构图 2.5-flare | 14.09% | 221,590 | 32.2% | 36.5% | 1170 |

→ **细节预算模型**（总强边像素量受限，元素越多每个越细）；渐变区是"糊"的物理来源。
→ 由此得出两条有效路线：**降复杂度** 或 **换更强模型 + 事后重绘**。

## 复现

```powershell
$py = "C:\Program Files\Python310\python.exe"
$f  = "docs\gpt-image-optimize\data\prompt_line_v3.txt"
foreach ($m in @("gpt-image-2","gpt-image-2.5-flare","gpt-image-2.5-sunburst")) {
  & $py -u tools\gpt_image2_gen.py --site new.aigc2d --model $m --prompt-file $f `
      --size 1024x1536 --quality high --output-format png --n 1 `
      --output-subdir anime_lolita_cmp_v3 --prefix $m
}
```

产物：`data/20260919/anime_lolita_cmp_v3/`；指标：`data/metrics_v3_prompt_short.json`、`data/line_profile_all.txt`；
可视化：`data/20260919/anime_lolita_cmp_report/index.html`。
