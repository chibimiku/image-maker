# 固定色板执行验证：离线协议、预检与待确认项

日期：2026-10-04。状态：**离线预检已完成（passed），尚未生成任何图片**。本文件是本轮的交付说明；协议原文在 `prompts/color-knowledge/fixed-palette-v1.json`，预检结果在 `data/test-result/20261004/color-fixed-palette-v1/`。

目标：**稳定执行指定的颜色组合与区域分配**，不是改善画质、不与基线比好看、不做质量排行榜。旧图重评已收尾，本轮不再为旧图追加评审请求。

## 1. 协议构成（已冻结）

| 文件 | 作用 |
|---|---|
| `prompts/color-knowledge/fixed-palette-v1.json` | 执行协议：主体、画法、保护块 M、7 组 × 2 版本条款、规模定义、评审维度与晋级规则、预算策略、限制 |
| `prompts/color-knowledge/fixed-palette-v1-simple.md` | 简洁版条款的撰写来源与撰写规则（颜色 + 主次 + 区域 + 允许中性色） |
| `prompts/color-knowledge/fixed-palette-v1-knowledge.md` | 知识编译版条款的来源（同一色板同一区域，只增加明度/彩度层级与色彩关系说明） |
| `prompts/color-knowledge/fixed-palette-review-system.md` | 评审提示词：逐张分开记录颜色组合 / 区域分配 / 额外色块 / 保护色 / 内容 / 画法 / 不可验证原因 |
| `tools/color_knowledge.py`（`palette` 子命令） | `preflight` / `preview` / `generate` / `review-plan` / `review` / `status`；未新增一次性脚本 |
| `utils/color_experiment.py` | 固定色板消费层：spec 校验、色板自检、冲突检查、请求组装、预检与预算、按组评审与两个汇总指标 |
| `tests/test_color_fixed_palette.py` | 离线回归用例 14 条（见 §6） |

### 1.1 主体与颜色授权（新的中性色诊断用例）

保留春日河畔单人钓鱼主题，但改用中性色主体：银灰发、象牙白洋装、深中性色厚底鞋；**新增色只作用于既有的两个区域**——腰间的宽腰带、两只鞋的蝴蝶结。保护块 M 明确写「只允许给既有腰带与鞋蝴蝶结改色」，并禁止新增物件、新增第二个人、改变时间/天气/姿势/服装设计/画法。点缀区域被要求在构图里可见，看不见就记 `unverifiable`，不计成功。

### 1.2 两种条款版本的公平性（自检项）

预检逐组核对：同一色板的两版**条款条数相同（各 7 条）**、都点名主色与点缀色、都指定落点区域、都保留同一批固有色与同一批禁止项。知识版只把「同一要求」写得更具体（明度阶梯、彩度对比、阴影归属），不新增物件、不改构图或光照时间，也不删掉简洁版的关键约束。P1 还显式写了「红色不要用粉色替代」，自检按通过处理。

## 2. 计划矩阵与两种规模

| 组 | 色板 | 版本 | 条款数 | 说明 |
|---|---|---|---|---|
| P1-simple | 冷蓝主色 + 红色点缀 | 简洁 | 7 | 用户点名的组合，最简说法 |
| P1-knowledge | 冷蓝 + 红 | 知识编译 | 7 | 同一色板与区域，编译版对照 |
| P2-simple | 深绿 + 玫红 | 简洁 | 7 | 诊断性候选（换色相，检验泛化） |
| P2-knowledge | 深绿 + 玫红 | 知识编译 | 7 | 对照 |
| P3-simple | 蓝紫 + 金 | 简洁 | 7 | 诊断性候选 |
| P3-knowledge | 蓝紫 + 金 | 知识编译 | 7 | 对照 |
| B0 | 无指定配色 | — | 0 | 同一主体基线，用来辨认「指定色板」与模型默认配色的差别 |

| 规模 | 组 | 计划产物 | 生成调用上界 | 评审（计划） | 评审（额外） | 评审上界 |
|---|---|---|---|---|---|---|
| **full（默认）** | 7 | **21 张** | 21 | 7 | 2 | 9 |
| reduced（备选） | 3 | **9 张** | 9 | 3 | 1 | 4 |

缩减版只覆盖 **P1 两版 + B0**（3 组 × 3 轮 = 9 张）。规模属于协议的一部分：选了 reduced 就必须用新的输出目录，运行中禁止自动扩组。

## 3. 请求预览（离线生成，与生成路径同一份组装函数）

| 组 | 版本 | 请求字符数 |
|---|---|---|
| P1-simple | 简洁 | 2466 |
| P1-knowledge | 知识编译 | 2773 |
| P2-simple | 简洁 | 2456 |
| P2-knowledge | 知识编译 | 2738 |
| P3-simple | 简洁 | 2452 |
| P3-knowledge | 知识编译 | 2750 |
| B0 | 无条款 | 1729 |

段落结构：`subject`（同一主体）→ `rendering`（同一画法）→ `COLOUR PLAN:`（有条款的组）→ `PRESERVE:`（**全部组逐字相同的保护块 M**）。
完整正文用 `palette preview` 打印，或读 `palette-preflight.json` 的 `request_preview.prompts[].prompt`。请求不附源图、不附画风图，不重绘、不做色调校准或本地后处理。

## 4. 调用预算（生成前确认用）

- **生成**：按目标数生成（每张一次调用）+ 命令行显式设 `IMAGE_MAKER_IMAGE_MAX_RETRIES=0` 关闭底层 HTTP 重试 ⇒ **收费调用数 = 产物数**：full 21 次、reduced 9 次。已实测该环境变量的实际效果（配置里 `max_retries=1`，设 0 后生效）。
- **评审**：按组计，每组 3 张图在同一次调用里评审 ⇒ full 7 次、reduced 3 次；另外 **full 2 次 / reduced 1 次**额外额度只用于事先约定的失败恢复或证据不足，不反复询问直到给出满意评价。
- **费用估算**（本地定价接口，仅参考；实际账单以服务商为准）：生成约 0.0183 USD/张、评审约 0.0034 USD/次 ⇒ full 规模上界约 **0.41 USD**（计费分组 `Discounted-Banana-1`）。
- 失败策略：连续两次请求失败即停止该区块并报告；成功产物续跑复用；失败、拒绝与不可验证原样保留，不为凑成功率替换样本。

## 5. 执行命令（尚未执行）

```powershell
# 默认 21 张矩阵（7 组 × 3 轮）
$env:IMAGE_MAKER_IMAGE_MAX_RETRIES = '0'   # 关闭底层 HTTP 重试：收费调用数 = 产物数
& 'C:/Program Files/Python310/python.exe' -u tools/color_knowledge.py palette generate `
    --out data/test-result/20261004/color-fixed-palette-v1 --scale full

# 生成完成后按组评审（21 张 = 7 次调用）
& 'C:/Program Files/Python310/python.exe' -u tools/color_knowledge.py palette review `
    --out data/test-result/20261004/color-fixed-palette-v1 --scale full
```

缩减版：把两条命令的 `--scale full` 换成 `--scale reduced`，并换一个新的输出目录。
只做预检/只看请求（不收费）：`palette preflight` / `palette preview` / `palette review-plan` / `palette status`。

## 6. 验证环境与测试

| 项 | 实际值 |
|---|---|
| 解释器 | `C:\Program Files\Python310\python.exe` · Python 3.10.11 |
| pytest | 9.0.3（用户 site-packages，无需安装） |
| 编译检查 | `python -m py_compile utils\color_experiment.py tools\color_knowledge.py` → 0 |
| 固定色板用例 | `tests/test_color_fixed_palette.py` 14 条：两版对照公平性、基线无条款、点缀色与区域必须点名、显式禁用替代色识别、预检离线通过、调用数按图计、请求预览与生成路径一致、保护块授权范围、评审响应校验（漏图/漏条款/非法状态/0 起编号）、状态枚举与晋级规则、输出目录隔离、无硬编码密钥 |
| 重评用例（隔离修正） | `tests/test_color_reassessment.py`：`test_recovery_uses_the_request_of_that_call_not_the_current_plan` 改为**只读夹具 + 临时输出目录**（等价于 `--request-dir`），并断言交付记录的时间戳没有被改写，避免测试重写已交付报告的依据 |
| 全量回归 | `python -m pytest -q -p no:cacheprovider` → **765 passed, 12 subtests passed**（另有解释器退出阶段 onnxruntime 导入的 access violation 打印，出现在测试全部通过之后，与本次改动无关） |

## 7. 预检发现

- 问题：**0 个**。
- 提示 1：请求把模型固定为 `gemini-3.1-flash-image-preview`，而配置里当前是 `gemini-3-pro-image-preview` ⇒ 属于**待确认项 2**。
- 提示 2：底层 HTTP 重试当前配置为 1 ⇒ 生成时**必须**显式设 `IMAGE_MAKER_IMAGE_MAX_RETRIES=0`，否则收费调用数无法按产物数推算。

## 8. 待确认项

1. **规模**：默认 `full`（21 张 / 7 组），备选 `reduced`（9 张 / 3 组：P1 两版 + B0）。
2. **模型**：跑哪两个之一——(a) 沿用 spec 里的 `gemini-3.1-flash-image-preview`（与 2026-10-03 批次同款，便于跨批次对照）；(b) 改用配置当前模型 `gemini-3-pro-image-preview`（同批内公平，但与旧批次不同模型，需标注「新模型实验」）。
3. **预算上界**：full = 生成 21 次 + 评审 9 次；reduced = 生成 9 次 + 评审 4 次。确认后才发起任何收费请求。
4. **额外评审额度**的使用条件（失败恢复 / 证据不足）是否按本文件的约定执行。
5. **不自动串行**：本轮结束后不会自动进入重绘/工序保留试验，那一步另开预检与预算。

**未经确认不会执行生成。** 需要的话我可以先按 `reduced` 规模只跑 P1 两版 + B0 的 9 张，或先只打印某一组的完整请求正文供你核对措辞。

## 9. 文件入口

- 本轮交付说明（本文件）：[FIXED-PALETTE-PREFLIGHT-20261004.md](FIXED-PALETTE-PREFLIGHT-20261004.md)
- 执行清单（命令 + 预算 + 待确认项）：`data/test-result/20261004/color-fixed-palette-v1/palette-plan.md`
- 预检结果与请求预览全文：`data/test-result/20261004/color-fixed-palette-v1/palette-preflight.json`
- 协议与提示词：`prompts/color-knowledge/fixed-palette-v1.json`、`fixed-palette-v1-simple.md`、`fixed-palette-v1-knowledge.md`、`fixed-palette-review-system.md`
- 上游计划：[NEXT-FIXED-PALETTE-PLAN-20261004.md](NEXT-FIXED-PALETTE-PLAN-20261004.md)；旧图重评结论：[COLOR-CONFORMANCE-REASSESSMENT-20261004.md](COLOR-CONFORMANCE-REASSESSMENT-20261004.md)
- 说明：`data/test-result/20261004/` 下另有若干 `20261004-*.json/.txt` 与 `img-*-final-sline*.png`，是**全量回归测试**由 `tests/conftest.py` 的输出隔离开关改写到测试目录的产物，不属于本次实验；需要清理时用 `python -m utils.output_isolation --apply`（未执行）。
