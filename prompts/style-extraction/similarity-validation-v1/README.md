# 画风相似度受控实验 · 实验包（`similarity-validation-v1`）

对应协议：`docs/style-extraction/DEEPSEEK-SIMILARITY-CONTROLLED-EXPERIMENT-20261006.md`（1.0，2026-10-06）。

运行的**冻结参数**（先声明后使用，运行中不得更改）：

| 项 | 值 |
|---|---|
| 随机种子 | `20261006` |
| 设备 / 精度 | `cpu` / `fp32`（整场唯一；CUDA 只作旁证，不进主结果） |
| 主画风检索指标 | **CSD**（其余七组为次要/诊断） |
| 并列容差 | `1e-9 × max(1, |best|)`，并列不计入主准确率，另报分摊准确率 |
| E0 对称性容差 | `|a−b| ≤ 1e-6 + 1e-5 × max(|a|,|b|)` |
| 每个技术 job | 最多 3 次尝试 = 首次 + 2 次重试 |
| 运行目录 | `data/test-result/20261006/style-similarity-controlled-v1/run-20261006-01/` |

## 本目录文件

| 文件 | 用途 |
|---|---|
| `protocol-lock.json` | 冻结的协议版本、种子、设备、主指标、验收门槛（由 CLI 生成，此处仅为说明） |
| `E3-interventions.md` | 8 种受控局部干预与负对照的几何定义（与代码常量一致） |
| `sampling-rule.md` | 原作品分层选样与近重复处理的确定规则 |
| `blind-annotation-protocol.md` | 给人类标注者/盲评者的操作说明（E4、E5） |

长提示词不写进 Python：E4 的区域定位提示词沿用共享文件
`prompts/style-extraction/face-regions-v1.md`；E5 的视觉复核提示词沿用
`prompts/style-comparison-v2.md`（12 维评分 + 4 个门禁 + 证据 schema）。
本目录只保存实验**定义**与**给人类看的说明**。
