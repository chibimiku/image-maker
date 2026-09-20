# experiments/ · 四批实验的归档（含全分辨率原图）

> **图就放在本目录里，不依赖 `data/`。** `data/` 被 `.gitignore` 忽略（克隆不带），
> 所以实验产物一律**迁进来**：这里是**原始分辨率、原样格式**（PNG 仍是 PNG），没有压缩副本。
> `manifest.json` 里同时留了原始存放路径与 SHA256，方便回查。

## 内容

| 路径 | 内容 |
|---|---|
| `report-A/index.html` | 批次 A 三模型对比页（可缩放/平移、指标表）；配套 2 个 prompt 文件 |
| `report-A-B/index.html` | 批次 B 的 8 组对比页（含复跑），配套 payload 与两种 prompt 说明 |
| `report-C/index.html` | 批次 C 的 10 个重绘变体对比页，配套 C1–C7 prompt 全文 |
| `report-D/index.html` | 批次 D 的蕾丝保真对比页，配套 L1–L5 prompt 全文 |
| `metrics/` | 各批次指标原始表（`metrics_*.json` / `line_profile_*.txt`） |
| `img/batch-A/` | 源图 + gpt-image-2 / 2.5-flare / 2.5-sunburst / 超长版两图（1024×1536） |
| `img/batch-B/` | 前置画风 6 变体 + B0/B2/B4/B5 复跑（1024×1536） |
| `img/batch-C/` | 重绘 9 变体 + A1/A2 + 源图（重绘为 1696×2528） |
| `img/batch-D/` | 蕾丝保真 5 变体 + C7 复跑（1696×2528） |
| `img/evidence/` | 手 / 发丝 / 蕾丝放大证据、最早跑通的两张重绘产物 |
| `manifest.json` | 每张图 ↔ 原始路径 ↔ SHA256 ↔ 尺寸 |

对比页是自包含 HTML（内嵌 payload），双击即可离线打开；页面里的图片引用已指向 `../img/batch-*/`
（4 个页面共 36 处引用，0 死链）。

## 证据图（不挂页面，但看结论必看）

| 文件 | 看什么 |
|---|---|
| `evidence/lace-L5-vs-C7-3.4x.jpg` | 裙摆蕾丝 3.4×：源图 / C7（大圆瓣） / L2 / L5（小扇贝） |
| `evidence/hands-L5-vs-source-3x.jpg` | 支撑手 3×：源图（糊成一团）→ L5（四指分开压树皮 + 拇指） |
| `evidence/hands-face-hand-3x.jpg` | 扶脸手 3×：源图/A1 的"方块手" → C3/C4 指节可分的拳头 |
| `evidence/hem-frill-rows-2.2x.jpg` | 荷叶边行数是否被合并 |
| `evidence/chest-lace-2.2x.jpg` | 胸前蕾丝与蕾丝围裙层 |
| `evidence/hair-right-2.2x.jpg` | 画面右侧长发连续性 |
| `evidence/repaint-before-after-v2src.jpg` + `repaint-lace-v2src.jpg` | 第一次重绘验证：整图并排 + 蕾丝局部 |
| `evidence/repaint-A1-lace-vs-src.jpg` / `repaint-A2-lace-vs-src.jpg` | A1 / A2 重绘的裙摆前后对比 |
| `evidence/repaint-first-test-v3src.jpg` / `repaint-first-test-v2src.png` | 最先跑通的两张重绘产物 |

## 体积

| 项目 | 值 |
|---|---|
| 影像数量 | 46 张（34 张页面图 + 12 张证据图），**全分辨率** |
| 单张大小 | 1024×1536 约 2.6 MB；重绘 1696×2528 约 8 MB |
| `img/` 合计 | **≈ 201 MB** |
| `experiments/` 合计 | ≈ 201 MB（对比页与指标表各几十 KB） |

体积大的原因就是**保留原始分辨率**（用户要求"直接迁移图片进来"）。
如果哪天需要控制仓库大小，可选方案（按推荐顺序）：
1. 把 `img/batch-C`、`img/batch-D` 的重绘大图转成 PNG 无损优化（`oxipng -o4`，通常省 20–30%）；
2. 重绘产物降到 1440×2160 再转 PNG；
3. 只在仓库里留 `evidence/` 与批次 A/B（页面截图级别够看），重绘大图另行备份。

## 页面引用规则（改归档脚本时必读）

报告页 payload 里的图片键是**页内命名**（`v3_source.png`、`b0_baseline.png`、`src.png`、`l5.png`…），
**不是**原始产物文件名。所以 `archive_experiments.py` 的 `PAGE_MAP` 必须按页内键写——
最初用原始文件名做映射，结果 4 个页面的图全被兜底成占位图，就是这个坑。

## 重新生成归档

```powershell
python docs\gpt-image-optimize\tools\archive_experiments.py
```

脚本会：清空并重建 `img/` → 按 `MIGRATE` 表复制**原图** → 按 `PAGE_MAP` 改写 4 个页面的引用 →
校验无死链 → 刷新 `manifest.json`。

它读的原始目录（本机，gitignore，仅作为来源）：

```
data/20260919/anime_lolita_cmp_v3/        data/20260919/anime_lolita_cmp_v2/
data/20260919/anime_lolita_prefixB/       data/20260919/anime_lolita_prefixB_rep2/
data/20260919/gemini_repaint/             data/20260919/gemini_repaint_A/
data/20260920/gemini_repaint_C/           data/20260920/gemini_repaint_D/
data/20260920/gemini_repaint_L/
```
