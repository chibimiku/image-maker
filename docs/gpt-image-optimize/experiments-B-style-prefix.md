# 批次 B · 只改前置画风 prompt 的 6 种写法（含复现性检验）

- 日期：2026-09-19
- 目的：**能不能靠改生图 prompt 本身治好线条连续性**（在不动模型、不动后处理的前提下）
- 方法：内容部分与批次 A 的短版 prompt **逐字不变**（脚本校验过一致性），只替换**前置画风部分**

内容固定段（不允许改）：

```
a girl with long pastel pink hair and large brown eyes sitting on a thick mossy tree branch above a calm lake on a sunny afternoon, wearing a pink and cream lace frilled sweet Lolita one-piece dress with a big ribbon headdress, dappled golden sunlight, blue sky, pastel colors, clean cel shading, crisp continuous line art, sharp fine detail, medium full body shot.
```

## 6 个前置变体（全文见 `data/p_B_prefixes.md`）

| 代号 | 前置写法 | 假设 |
|---|---|---|
| B0 | `Anime illustration style,` | 基线（= 批次 A 短版） |
| B1 | `Traditional Japanese anime cel animation illustration, flat cel shading, bold even-weight continuous outlines, limited pastel palette,` | 画风名词没钉死渲染方式 → 自由发挥成半厚涂 |
| B2 | `Clean vector-like anime line art, flat colours, minimal shading, limited palette, simple background rendering,` | 细节预算没限制 → 用渐变代替线条 |
| B3 | `An anime illustration from a high-quality animation production, drawn by a professional animation key animator with a fine pen,` | 交给"画师语境"更稳 |
| B4 | `Anime illustration with flawless unbroken line work, no blurry areas anywhere in the frame, every detail sharply rendered,` | 把要求写成可检查的结果指标 |
| B5 | 前置 `Anime illustration with clean crisp cel-shaded line art,` + **后置**一段可核对要求（轮廓闭合 / 发丝蕾丝画成独立可数形状 / 每种材质一个硬边阴影 / 背景同样清晰 / 不要柔和渐变） | 只在尾部约束不够，需要前后夹击 |

## 结果（B0/B2/B4/B5 各跑两次）

| 变体 | lap_var（两次） | soft（两次） | frag（两次） | 判定 |
|---|---|---|---|---|
| B0 基线 | 845 / 725 | 38.51 / 38.35 | 21.6 / 19.5 | 基准 |
| B1 赛璐璐名词 | 1348 / — | 31.48 / — | 36.95 / — | 线条更清楚但碎片率涨 70% |
| B2 矢量线稿+限预算 | 1355 / 781 | **28.46 / 27.89** | 37.9 / 27.0 | **soft 唯一两次都稳定下降** |
| B3 动画师语境 | 823 / — | 38.45 / — | 20.17 / — | **≈ 基线，完全没收益** |
| B4 结果指标式 | 1509 / 926 | 35.17 / 36.92 | 28.0 / 28.0 | **不稳定**：锐度两次差 63% |
| **B5 前后双重约束** | 1231 / **1714** | **25.67 / 25.17** | 26.2 / 38.1 | **唯一复现成功的改进** |

## 结论

1. **单次生成噪声极大**：同参数同 prompt，`lap_var` 在 725–1509 之间跳（±60%）。
   凡是没有复跑的对比都不算数——B4 第一次看像神改进（1509），复跑掉到 926，还不如基线。
2. **只加前置风格名词 ≈ 没用**：B3 的"职业动画师用细笔绘制"与基线同值；
   B1 换来清晰度但把线条碎成小段，是拆东墙补西墙。
3. **有效写法 = 前置钉画风 + 后置钉可被检查的要求**（B5）：
   `soft` 两次都落在 25% 出头，是全部变体里最低也最稳的。
4. 因此：**改生图 prompt 能小幅帮上忙，但换不来根本改善** → 需要 Gemini 重绘（批次 C/D）。

## 复现

`data/` 下没有直接可跑的 prompt 文件（B 组内容由脚本拼装），照下表在批次 A 的固定段前加前置即可；
产物在 `data/20260919/anime_lolita_prefixB/`、`data/20260919/anime_lolita_prefixB_rep2/`；
指标 `data/metrics_B.json`、`data/metrics_B_rep2.json`；
可视化页 `data/20260919/ab_report/index.html`（含 8 个分组）。
