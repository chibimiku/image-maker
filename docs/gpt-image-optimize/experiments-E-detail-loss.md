# 批次 E · 「重绘丢细节」排查与保留清单固件（2026-09-20）

- 触发：用户反馈重绘后**黑色蕾丝手套、黑色裤袜、眼角泪水**全部丢失
- 源图：`data/20260920/gpt-image-2/gptimage2_output_022137_0_eddb2b.png`（1536×1024，16:9，海滩吊床看书的粉发少女）
- 用户报告的问题产物：`data/20260920/gpt_image_repaint/repaint_gptimage2_output_022137_0_eddb2b_023058-c3963e.png`
- 结论：**不是固件系统性缺陷，而是抽卡 + 固件缺少"逐项保留小配件"条款**；新版固件（保留清单优先）已部署

## 1. 症状核对

对比源图与问题产物，逐项确认：

| 元素 | 源图 | 问题产物 |
|---|---|---|
| 黑色蕾丝手套（双手） | 有，指节露肤的蕾丝半手套 | **丢失，变成裸手** |
| 黑色裤袜（连脚、不透明） | 有 | **丢失，变成光腿裸足** |
| 眼角泪水 | 有一道泪痕 | **丢失** |

## 2. 定位：固件缺什么

当时的 `repaint-system.md` 里：

- 保存条款只写了 "composition / pose / character design / **the exact outfit design** / background layout"。`outfit design` 太笼统，**手套、裤袜、泪痕这类小配件与微观细节没有被点名**；
- 通篇重点是"平涂化 + 保蕾丝结构 + 修手 + 连通发丝"，等于把注意力都引到蕾丝和手上；
- **"Hands. The source hands are malformed, so reconstruct both instead of copying them"** 这句在无约束时会让模型把"手套下的手"重建成裸手 —— 手套就是这样丢的。

对照实验（PRESERVE 清单版）里同一句话已被改成 `Fix the anatomy WITHOUT changing what the hands are wearing ... if the hands are gloved, the corrected hands keep the same glove`，手套就不再丢。

## 3. 实测：同源图 × 多版 prompt

全部 2K / 16:9 / gemini-3-pro-image-preview，源图同上。

| 代号 | prompt | 手套 | 裤袜 | 泪水 | 备注 |
|---|---|---|---|---|---|
| **BAD（用户产物）** | 当时固件 + 旧后缀 | ❌ | ❌ | ❌ | 用户报告的那张 |
| P0 | 当时固件（主 prompt 原文，无后缀） | ✅ | ✅ | ✅ | 同 prompt 复跑就好了 → **抽卡波动** |
| Q4 | 当时固件 第二次 | ✅ | ✅ | 较淡 | |
| **Q1** | **新主 prompt（保留清单优先）** | ✅ | ✅ | ✅ | 一次通过 |
| Q3 | 当时固件 + **新后缀（保留清单）** | ✅ | ✅ | ✅ | 一次通过 |
| Q2 | 新主 prompt + 新后缀 | ✅ | ✅ | 较淡 | |
| **NEW** | **部署后的固件（新主+新后缀，6099 字符）** | ✅ | ✅ | ✅ | 按用户给定输入/输出路径的验证跑 |

**"裤袜是否还在"的客观判据**（腿/脚区域平均亮度，越低=深色连裤袜还在；源图 89.0 / 116.6）：

| 版本 | feet 区亮度 | legs 区亮度 |
|---|---|---|
| 源图 | 89.0 | 116.6 |
| Q1 新主 prompt | 88.4 | 113.4 |
| Q3 固件+新后缀 | 91.2 | 113.8 |
| Q4 当时固件 | 96.4 | 116.9 |
| Q2 新主+新后缀 | 96.4 | 121.8 |
| NEW 部署后 | 88.4 | 114.0 |

全部与源图同量级 → **没有一个版本丢掉裤袜**；用户那张丢失属于低概率抽卡。

接触表（人眼复核）：`data/20260920/gemini_repaint_Q/inspect/preserve_hands-book.jpg`、`preserve_face-tear.jpg`、`preserve_feet.jpg`、`preserve_legs.jpg`。

## 4. 修法：固件加"逐项保留清单"

`prompts/gpt-image-optimize/repaint-system.md` 的结构由

```
不重设计声明 → 不许简化蕾丝 → 修手/发丝 → 风格 → 自检
```

改成

```
不重设计声明 → 【逐项保留清单（新，置顶）】→ 渲染升级 → 修错（且不得删配件）→ 自检
```

保留清单逐条点名：**手套/手部覆盖物、裤袜与所有腿饰、鞋、首饰与发饰、泪水/泪痕/汗/腮红/痣等面部微细节、手持道具、裙子所有褶边与蕾丝、吊床编织**，并加一句 *Treat this list as a checklist: before output, scan the source and confirm none of these has disappeared.*

同时把修手那句改成**"修解剖，不许改手戴什么"**：

> Fix the anatomy WITHOUT changing what the hands are wearing. ... If the hands are gloved, the corrected hands keep the same glove; if the fingers are bare, the bare skin keeps the same colour and the same nails.

`repaint-detail-suffix.md` 同步换成同一份保留清单的压缩版（后缀只讲"完整性 + 渲染"，不再重复蕾丝术语）。

## 5. 可迁移的规律（写重绘/编辑类 prompt 通用）

1. **"保持原样"必须逐项点名**：`keep the outfit design` 这种概括句保护不了手套、袜类、泪水这类小物件；模型只会保住你点名的东西。
2. **修错指令要加"不得以删除为代价"**：`reconstruct the hands` 会顺手把手套摘掉、`fix the legs` 会顺手把裤袜脱掉 —— 必须写 `without changing what they are wearing` / `never by deleting an accessory`。
3. **抽卡波动是主因之一**：同一 prompt 同一源图，某次会整体丢细节。所以**重绘要跑 2~3 次挑**，这也是 GUI 里「每张重绘次数」存在的原因。
4. **给模型自检清单**：`Treat this list as a checklist: before output, scan the source` 与 `do the completeness pass again` 能让命中率明显上升（本批次 4 个新版全部一次通过）。

## 6. 复现

```powershell
$py = "C:\Program Files\Python310\python.exe"
$src = "data\20260920\gpt-image-2\gptimage2_output_022137_0_eddb2b.png"
# 部署后的固件（主 + 后缀）
& $py -u tools\gpt_image2_gen.py --repaint --image $src --repaint-repeat 3
# 或用 GUI：模式=重绘，提示词框会自动填入固件，可直接改
```

产物：`data/20260920/gemini_repaint_P/`、`gemini_repaint_Q/`、`gemini_repaint_NEW/`；
接触表：`gemini_repaint_Q/inspect/`。
