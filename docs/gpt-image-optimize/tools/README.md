# tools/ · 实验分析脚本快照

> 这些脚本原本在临时工作目录里用于本次研究，**现快照存档**，便于以后复算指标或重跑实验。
> 它们属于"研究工具"，不是产品链路的一环；产品链路的入口是 GUI 的「重绘」模式与
> `tools/gpt_image2_gen.py --repaint`。
> 运行方式与其他项目脚本一致：系统 Python `C:\Program Files\Python310\python.exe`。

| 脚本 | 作用 |
|---|---|
| `ab_metrics.py` | 全局指标：`lap_var / hard_edge / soft / flat / edge_soft / frag / iso / colors`（先缩放到同宽 1024） |
| `lace_metrics.py` | 蕾丝保真指标：`lace_teng / lace_hard / lace_iso / frill_cycles / skirt_teng` |
| `crop_inspect.py` | 区域接触表：把多张图的同一区域裁出来横排（手 / 发丝 / 脸 / 蕾丝 / 树皮），支持 `--scale` 放大 |
| `grid_overlay.py` | 给图叠 10×10 百分比网格，用于标定相对区域坐标 |
| `make_contact_sheet.py` | 三联缩略图 + 4 处局部放大（早期版本，功能被 `crop_inspect.py` 覆盖） |
| `compare_metrics.py` | 早期综合指标（锐度 + 边缘连通性 + 块效应），保留作参考 |
| `line_profile.py` | 梯度剖面 + 细线骨架连通性（p50/p90/p99、`frac_gt_30`、`thin_frag`） |
| `detail_budget.py` | 「细节预算」统计：强边像素数 / 平坦 / 渐变占比，支撑 `README.md` 第 2 节 |
| `style_budget.py` | 同上但更细（含色彩量化与中心区占比），早期版本 |
| `run_gemini_repaint.py` | 单模型重绘执行器：`--source / --prompt-file / --prefix / --resolution / --aspect` |
| `make_repaint_compare.py` | 重绘前后并排 + 局部放大对比图 |
| `build_ab_page.py` | 由 JSON payload 生成 A/B 对比 HTML（同步缩放 / 平移 / 指标表） |
| `archive_experiments.py` | **归档脚本**：把对比页 + 指标 + 关键影像压进 `experiments/`，重写页内图片引用、刷新 `manifest.json` |
| `scan_hardcode.py` | **写死配置扫描**：查模型名 / 尺寸 / 比例 / 站点 / 目录 / 长 prompt 是否被硬编码进 .py |

> `grid_overlay.py` 与 `line_profile.py` 里还有 `--out=` 之类的小工具参数，直接看文件头注释即可。

## 常用命令

```powershell
$py = "C:\Program Files\Python310\python.exe"
$t  = "docs\gpt-image-optimize\tools"

# 指标
& $py "$t\ab_metrics.py"  <img1> <img2> --out out_ab.json
& $py "$t\lace_metrics.py" <img1> <img2> --out out_lace.json

# 局部放大（区域名见 crop_inspect.py 顶部 REGIONS）
& $py "$t\crop_inspect.py" --out outdir --regions handL2,handR,hairr,lace --scale 3 --items "A=a.png" "B=b.png"

# 重绘一张（与 GUI 重绘模式同通道）
& $py "$t\run_gemini_repaint.py" --source <src.png> --prompt-file prompts\gpt-image-optimize\repaint-system.md --prefix demo
```

区域坐标（相对整图，同构图可直接复用）：

| 区域名 | 相对矩形 (x, y, w, h) | 用途 |
|---|---|---|
| `handL` / `handL2` | (0.70, 0.48, 0.24, 0.20) / (0.71, 0.46, 0.13, 0.12) | 撑在树枝上的手（`L2` 是紧裁，用于数手指） |
| `handR` | (0.27, 0.14, 0.22, 0.17) | 扶脸的手 |
| `hairr` | (0.62, 0.22, 0.30, 0.30) | 画面右侧长发 |
| `face` | (0.30, 0.08, 0.30, 0.20) | 脸 |
| `lace` / `hem` / `chest` / `skirt` | (0.16,0.42,0.42,0.24) / (0.20,0.44,0.34,0.18) / (0.34,0.20,0.26,0.18) / (0.14,0.36,0.46,0.22) | 蕾丝 / 下摆 / 胸前 / 裙身 |
| `branch` | (0.10, 0.58, 0.30, 0.28) | 苔藓树枝 |
