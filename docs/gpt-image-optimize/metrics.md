# 指标定义与口径（含已知偏差）

所有指标脚本都在 `tools/`（本目录快照），统一**先缩放到同宽 1024 再计算**，否则分辨率会白送锐度。

## 1. 全局指标（`tools/ab_metrics.py`）

| 指标 | 定义 | 方向 | 解读 |
|---|---|---|---|
| `lap_var` | 灰度图拉普拉斯方差 | ↑ 越好 | 整体锐度 / 高频能量。单看它会被"噪点多的图"骗高 |
| `hard_edge` | 梯度幅值 > 20% 峰值的像素占比 | 参考 | 画面里有多少"硬边" |
| `soft` | 梯度幅值落在 2%–10% 的像素占比 | ↓ 越好 | **半软过渡区**。这是"发虚"的物理来源；越低说明画面越靠明确的线/色块定义 |
| `flat` | 梯度 < 2% 的像素占比 | 参考 | 平涂色块比例。越高越"干净"，但也可能是细节被简化了 |
| `edge_soft` | `hard_edge / soft` | ↑ 越好 | "线稿化程度"。实测最能把"干净线稿"和"柔和厚涂"分开的单一指标 |
| `frag` | Canny 边缘连通域中 < 8px 小碎块占边缘的比例… | ↓ 一般认为好 | **见第 3 节偏差，密集蕾丝图上不可用** |
| `iso` | 3×3 邻域内边缘邻居 ≤ 1 的孤立边缘像素占比 | ↓ 越好 | 断线的比例 |
| `colors` | 16 级量化后不同 RGB 三元组数量 | 参考 | 色彩预算。平涂化会明显下降（1000 → 700 上下） |

## 2. 蕾丝 / 褶皱保真指标（`tools/lace_metrics.py`）

只看两个相对区域（相对整图坐标，重绘图同构图可直接复用）：

- `lace` = `(0.18, 0.44, 0.34, 0.20)`：裙摆荷叶边
- `skirt` = `(0.14, 0.34, 0.48, 0.24)`：裙身整体

| 指标 | 定义 | 方向 | 解读 |
|---|---|---|---|
| `lace_teng` | 该区域 Sobel 能量均值 | ↑ 越好 | **蕾丝信息量**。被"合并成大圆瓣"时会明显掉 |
| `lace_hard` | 该区域梯度 > 20% 的占比 | ↑ 参考 | 蕾丝边缘的硬朗程度 |
| `lace_iso` | 该区域孤立边缘占比 | ↓ 越好 | 蕾丝是否连成结构 |
| `frill_cycles` | 沿水平扫描线做差分后数符号变化次数，取中位数 | ↑ 参考 | **单位宽度里能数到的"边/褶"数量**。相邻行合并会掉 |
| `skirt_teng` | 裙身区 Sobel 能量均值 | ↑ 越好 | 褶皱、包边缝、层次的数量 |

**用法**：判断"细节有没有被简化"就用 `lace_teng` + `frill_cycles`；
判断"线条有没有变清楚"用 `soft` + `edge_soft`；两组必须一起看。

## 3. 已知偏差（踩过的坑，务必记住）

### 3.1 `frag` 在密集蕾丝图上会反向误判

批次 D 实测：把蕾丝画清楚之后 `frag` **升高**（0.44 → 0.49）。
原因：`frag` 把"几百个小扇贝弧"数成了几百个"小碎块"，于是判定"线更碎"。
但肉眼放大看到的是**更多、更细、更清楚的结构**。

→ **规则：看蕾丝是否被简化，用 `lace_teng` / `frill_cycles`；不要用 `frag`。**
`frag` 只适合"结构简单、线应该很长"的图（例如批次 A/B 的全身像对比）。

### 3.2 `soft` 会被"画面元素多少"污染

元素少的图（近景特写 + 纯色背景）天然 `soft` 低。
跨图比较时先确认构图复杂度可比；本目录所有对比都是同源同构图，所以可用。

### 3.3 分辨率不是可比量

重绘产物 1696×2528 直接测 `lap_var` 会天然高于 1024×1536。必须先缩放到同宽。

### 3.4 抽卡噪声

同一 prompt、同参数、同一源图，两次结果实测：

| 样本 | lap_var run1 / run2 | soft run1 / run2 |
|---|---|---|
| B0 基线 | 845 / 725 | 38.5% / 38.4% |
| B4 | 1509 / 926 | 35.2% / 36.9% |
| C4 | 2336 / 2059 | 18.2% / 27.6% |

→ 结论：**任何"提升"必须复跑 2 次以上**。`soft` 这类结构型指标比 `lap_var` 稳定（B0/B2/B5 的 soft 两次都在 ±1% 内），
所以优先用 `soft` / `edge_soft` 做判据，`lap_var` 只作参考。

### 3.5 边缘/锐度指标看不见"美不美"

指标只能证明"线条更明确、结构更多"，不能证明"更好看"（例如 C4 把背景简化了，`soft` 很漂亮但画面变单调）。
**最终一律以局部放大图肉眼裁决**（脸 / 手 / 右侧发丝 / 裙摆蕾丝 / 树木水面）。

### 3.6 跨比例不能直接比指标（重绘时必须让输出跟随源图）

重绘产物与源图**比例不同**时，区域指标（`lace_*`、`frill_cycles`）的坐标会错位，
`soft/edge_soft` 也会因为构图被拉伸/裁切而失真。
因此功能的默认行为是 `aspect_ratio = "auto"`（不下发 `aspectRatio`，由模型匹配输入图尺寸），
换任意比例的源图都不会变形——细节见 `README.md` 第 6 节。

## 4. 复现方法

```powershell
# 全局指标（可传多张）
python docs\gpt-image-optimize\tools\ab_metrics.py <img1> <img2> ... --out out.json

# 蕾丝保真指标
python docs\gpt-image-optimize\tools\lace_metrics.py <img1> <img2> ... --out lace.json

# 局部放大接触表（脸/手/发丝/蕾丝/树皮）
python docs\gpt-image-optimize\tools\crop_inspect.py --out outdir --regions handL2,handR,hairr,lace --scale 3 --items "NAME=path.png" ...

# 单张重绘（同 GUI「重绘」模式）
python docs\gpt-image-optimize\tools\run_gemini_repaint.py --source <src.png> --prompt-file prompts\gpt-image-optimize\repaint-system.md --prefix demo
```

## 5. 代码里的配置引用（排查"写死"用）

改任何默认值都应当只改 `prompts/gpt-image-optimize/config.json`；代码里的字面量全是**兜底默认**（配置文件缺失时才生效）：

| 位置 | 兜底来源 |
|---|---|
| `utils/gpt_image_optimize.py::DEFAULTS` | 与 config.json 同义的默认值（**唯一**一处字面量；加字段时这里同步） |
| `modules/others/api_backend.py::generate_image_repaint` | 全部取 `DEFAULTS[...]`，不写死模型名/尺寸/比例/接口 |
| `modules/image_generation/gpt_image2_tab.py` | 同上，取 `REPAINT_DEFAULTS[...]`；界面选项从 config.json 的 `*_options` 读取 |
| `modules/others/api_backend.py` 里的站点/接口映射（`GPT_IMAGE2_SITE_*`） | 站点清单的**单一事实来源**（非重绘专属） |

用 `python docs\gpt-image-optimize\tools\scan_hardcode.py` 可以一键扫描是否又出现写死的模型名/尺寸/比例/目录。
