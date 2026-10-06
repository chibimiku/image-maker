# 画风相似度严格受控实验（结果）

> 协议 1.0（2026-10-06）· 运行 id `run-20261006-01` · 生成 2026-10-06T13:29:06 · 设备 cpu / fp32

**本文件是执行结果，不是通过声明。** 计算成功、`status=ok`、单元测试通过都不等于画风效度成立；所有未完成、缺人类输入、缺额度的项按实际状态列出。

## 1. 阶段状态

| 阶段 | 状态 | 计划 | 实际 | 独立作品/主体 | 缺项 ID | 原因 |
|---|---|---|---|---|---|---|
| E0 | 完成 | 6 | 6 | 6 | — | 零收费本地计算 |
| E1 | 完成 | 24 | 24 | 24 | — | 四种画风 × 12 个独立作品（6 参考 / 6 query） |
| E2 | 完成 | 12 | 12 | 12 | — | 12 个 anchor + P/N + N0–N4 变体；P/N 由自动匹配代理填写 |
| E3 | 部分 | 162 | 162 | 6 | — | 可控绘制图样机制测试，不代表真实图效度 |
| E4 | awaiting_human | 8 | 8 | 8 | H1, H2, H1-repeat(≥24h), 人工变体结构核查 | 自动定位已跑并保留叠图；人工标注与人工间重复性必须由实际人类提供 |
| E5 | 部分 | 20 | 2 | 2 | 新增 18 槽位（未授权额度）, 两名人类 ≥24 个有效独立题 | 历史 2 张 Gemini 产物的正式视觉复核已按两种呈现顺序运行 （4 次）；新增生图无授权额度，人类盲评待作答 |

协议偏离（单列）：

| # | 偏离 | 影响 | 处理 |
|---|---|---|---|
| 1 | E2 的 P/N（困难对照）选择依据由确定性自动匹配规则生成，不是「不知道指标值的标注者」填写 | 困难对照的选择依据缺少独立人工背书；规则只读取冻结属性，未读取任何指标值 | 在 E2 结果与报告第 5 节单列为协议偏离；人工标注列为待项 |
| 2 | 第 4 种画风用 Renian 替代协议优先清单里的 TID | 与协议原始画风清单不完全一致，未测风格不得外推 | 在看分数之前决定并记录在 protocol-lock.json 的 style_substitution |
| 3 | E3 使用参数化可控绘制图样而非真实插画 | 只证明测量机制会响应指定变化，不代表真实插画效度 | 协议 §7 明确允许，并在 E3 结果与报告中声明 |
| 4 | E4 实际运行 8 张（6 个可控案例 + 2 张历史 Gemini 图），协议写的是 10 张（6 个案例 + 4 张历史图） | 历史图分母比协议少 2 张（那次历史测试只产出 2 张图），覆盖率分母相应变小 | 在 E4 结果与报告中写明实际分母 8 与差额 2 的来源，不把 8 说成 10 |
| 5 | E4 人工标注与 E5 人类盲评尚未提供 | 自动定位可用率、人工间重复性、候选排序效度均无法验收 | 登记 awaiting_human，并交付盲评包与标注入口，未用模型代填 |
| 6 | E5 的视觉复核只完成「参考图正序 / 倒序」两种呈现顺序，候选级顺序敏感性无法检验 | 每个画风只有 1 个候选，候选顺序对评分的影响未测 | 在 E5 结果中写明检验的是参考图顺序，不冒充候选级顺序敏感性 |
| 7 | E5 新增 18 个生图槽位没有费用授权 | E5 只有历史 2 张产物，最多 1 组组内比较，远少于 24 个有效独立题 | 登记 awaiting_budget，未发出任何新增生图请求 |

## 2. 数据来源、split、近重复核查、混杂分布

随机种子 `20261006`；主指标 **csd**（预注册）；设备/精度 **cpu/fp32**（整场唯一，未中途切换）。

| 画风 | 数据源 | 候选池 | 可用池 | 参考/query | 预留 E2 池 | 自动剔除重复 | 待人工核查的模糊对 |
|---|---|---|---|---|---|---|---|
| kishida_mel | C:\data\train\kishida_mel-2d\tagged | 240 | 238 | 6/6 | 226 | 2 | — |
| puracotte | C:\data\train\puracotte-2d | 240 | 240 | 6/6 | 228 | 0 | — |
| renian | C:\data\train\renian-2d\fanbox | 240 | 79 | 6/6 | 67 | 161 | — |
| sakurapion | C:\data\train\sakurapion-2d\train | 147 | 146 | 6/6 | 134 | 1 | — |

确定性重复剔除规则与阈值、分层选样规则见 [sampling-rule.md](../../../prompts/style-extraction/similarity-validation-v1/sampling-rule.md)；被剔除项逐条记录在 `PLAN/images.json`。

**画风替换（在看分数之前决定）**：本地 TID 仅有 data/style-ref/tid-fullbody-background-candidate.jpg 与 tid.png 两张素材，没有可验证来源的原作品集；改用来源可验证的 C:\data\train\renian-2d（FANBOX 原作品）作为第 4 种画风。

混杂分布（如实列出不平衡，不声称已消除）：

| 画风 | 分类计数 |
|---|---|
| kishida_mel | query:bright:6, query:clean:6, query:face_closeup:6, reference:bright:6, reference:clean:6, reference:face_closeup:6, reserved_e2_pool:bright:182, reserved_e2_pool:clean:226, reserved_e2_pool:dark:44, reserved_e2_pool:face_closeup:4, reserved_e2_pool:full_body_or_scene:187, reserved_e2_pool:upper_body:35 |
| puracotte | query:bright:5, query:clean:6, query:dark:1, query:face_closeup:6, reference:bright:5, reference:clean:6, reference:dark:1, reference:face_closeup:6, reserved_e2_pool:bright:187, reserved_e2_pool:clean:228, reserved_e2_pool:dark:41, reserved_e2_pool:full_body_or_scene:202, reserved_e2_pool:upper_body:26 |
| renian | query:bright:6, query:clean:6, query:face_closeup:1, query:full_body_or_scene:5, reference:bright:5, reference:clean:6, reference:dark:1, reference:face_closeup:2, reference:full_body_or_scene:4, reserved_e2_pool:bright:53, reserved_e2_pool:clean:67, reserved_e2_pool:dark:14, reserved_e2_pool:full_body_or_scene:59, reserved_e2_pool:upper_body:8 |
| sakurapion | query:bright:6, query:clean:6, query:face_closeup:4, query:full_body_or_scene:2, reference:bright:6, reference:clean:6, reference:face_closeup:5, reference:full_body_or_scene:1, reserved_e2_pool:bright:132, reserved_e2_pool:clean:134, reserved_e2_pool:dark:2, reserved_e2_pool:full_body_or_scene:107, reserved_e2_pool:upper_body:27 |

冻结包：[blind-map.json](<D:\code\image-maker\data\test-result\20261006\style-similarity-controlled-v1\run-20261006-01\PLAN\blind-map.json>)、[code-freeze.json](<D:\code\image-maker\data\test-result\20261006\style-similarity-controlled-v1\run-20261006-01\PLAN\code-freeze.json>)、[e2-plan.json](<D:\code\image-maker\data\test-result\20261006\style-similarity-controlled-v1\run-20261006-01\PLAN\e2-plan.json>)、[e5-plan.json](<D:\code\image-maker\data\test-result\20261006\style-similarity-controlled-v1\run-20261006-01\PLAN\e5-plan.json>)、[images.json](<D:\code\image-maker\data\test-result\20261006\style-similarity-controlled-v1\run-20261006-01\PLAN\images.json>)、[pairs.json](<D:\code\image-maker\data\test-result\20261006\style-similarity-controlled-v1\run-20261006-01\PLAN\pairs.json>)、[plan-full.json](<D:\code\image-maker\data\test-result\20261006\style-similarity-controlled-v1\run-20261006-01\PLAN\plan-full.json>)、[protocol-lock.json](<D:\code\image-maker\data\test-result\20261006\style-similarity-controlled-v1\run-20261006-01\PLAN\protocol-lock.json>)、[runtime.json](<D:\code\image-maker\data\test-result\20261006\style-similarity-controlled-v1\run-20261006-01\PLAN\runtime.json>)、[transforms.json](<D:\code\image-maker\data\test-result\20261006\style-similarity-controlled-v1\run-20261006-01\PLAN\transforms.json>)、[weights.json](<D:\code\image-maker\data\test-result\20261006\style-similarity-controlled-v1\run-20261006-01\PLAN\weights.json>)

代码/提示词 SHA-256、权重清单、依赖版本：`PLAN/code-freeze.json`、`PLAN/weights.json`、`PLAN/runtime.json`。

E5 冻结的生成参数（历史产物来源，未新增请求）：`PLAN/e5-plan.json`；历史请求回放见 `data/test-result/20261006/style-similarity-gemini/*/*_aigc2d_replay_*.json`（已脱敏）。

## 3. E0 实现一致性与错误传播

| 检查项 | 结果 | 证据 |
|---|---|---|
| E0.1-同图深度指标 | 通过 | {"max_abs_distance": 0.0, "tolerance": 1e-06, "images": 6} |
| E0.1-同图CSD≈1 | 通过 | {"csd_values": [1.0, 1.0000001192092896, 1.0000001192092896, 0.9999999403953552, 1.000000238418579, 1.0000001192092896], "tolerance": 0.0001} |
| E0.1-同图本地四组≈1 | 通过 | {"count": 24, "min": 1.0, "max": 1.0, "tolerance": 1e-06} |
| E0.2-对称性 | 通过 | {"cases": 120, "failures": [], "rule": "|a-b| <= 1e-6 + 1e-5*max(|a|,|b|)"} |
| E0.3-重复与重启复算 | 通过 | {"values": {"gram": [2.386009707553069, 2.386009707553069], "adain": [66.86684465535349, 66.86684465535349], "lpips": [0.6656175851821899, 0.6656175851821899], "csd": [0.5178946852684021, 0.5178946852684021], "tone": [0.7798792613999693, 0.7798792613999693], "edges": [0.9534484571933333, 0.9534484571933333], "lines": [0.8591088236673484, 0.8591088236673484], "space": [0.3990996323199713, 0.3990996323199713]}, "max_ab |
| E0.3-完整缓存复用 | 通过 | {"cache_hit": true, "values": {"gram": 2.386009707553069, "adain": 66.86684465535349, "lpips": 0.6656175851821899, "csd": 0.5178946852684021, "tone": 0.7798792613999693, "edges": 0.9534484571933333, "lines": 0.8591088236673484, "space": 0.3990996323199713}} |
| E0.3-hash变化不复用（指纹） | 通过 | {"sha256": "208414eeb8ff82436ca311535b033f7b0a73ea4ce6abb0210019784fd5c16e2b", "cache_hit": false, "values": {"gram": 2.386009707553069, "adain": 66.86684465535349, "lpips": 0.6656175851821899, "csd": 0.5178946852684021, "tone": 0.7798792613999693, "edges": 0.9534484571933333, "lines": 0.8591088236673484, "space": 0.3990996323199713}, "values_identical_to_original": true, "note": "一像素扰动经 512px bicubic 预处理后可能完全被吸收；这一条 |
| E0.3-hash变化不复用（数值） | 通过 | {"sha256": "fd544bbc3292149e5723d023bd701ca338311b70d33bc95500631ac5114ae609", "cache_hit": false, "values": {"gram": 2.3546983601766605, "adain": 66.2656523754952, "lpips": 0.6632215976715088, "csd": 0.5178946852684021, "tone": 0.7799988764672389, "edges": 0.9546094591380115, "lines": 0.8595768376847662, "space": 0.3990996323199713}, "differ_from_original": {"gram": true, "adain": true, "lpips": true, "csd": false,  |
| E0.3-hash不一致拒绝 | 通过 | {"claimed": "4bda9c8042064100c79670c84e1cf4fa5df87843e5cbb3d4e54e2335788835a6", "actual": "fd544bbc3292149e5723d023bd701ca338311b70d33bc95500631ac5114ae609", "content_was_changed": true, "validate_manifest_error": "ValueError: 图片已变化：D:\\code\\image-maker\\data\\test-result\\20261006\\style-similarity-controlled-v1\\run-20261006-01\\e0\\same-path-overwritten.png", "compare_error": "ValueError: 图片已变化：D:\\code\\image-ma |
| E0.4-三入口一致 | 通过 | {"query": "D:\\code\\image-maker\\data\\test-result\\20261006\\style-similarity-gemini\\puracotte-style-v2\\test-puracotte-style-v2_111916-5611c0.jpg", "references": ["C:\\data\\train\\puracotte-2d\\puracotte\\puracotte_2025-02-12_01-27-35.png", "C:\\data\\train\\sakurapion-2d\\train\\014.jpg", "C:\\data\\train\\kishida_mel-2d\\tagged\\kishida_mel-p1_2025-02-17_00-03-43.png"], "device": "cpu", "entries": {"cli": {"re |
| E0.5-不可读图片 | 通过 | {"status": "pass", "detail": {"result_status": "partial", "metric_status": {"gram": "error", "adain": "error", "lpips": "error", "csd": "error", "tone": "error", "edges": "error", "lines": "error", "space": "error"}, "face_status": "partial"}} |
| E0.5-区域hash错误 | 通过 | {"status": "pass", "detail": {"message": "ValueError: 区域文件版本或图片 hash 不匹配"}} |
| E0.5-配对缺失 | 通过 | {"status": "pass", "detail": {"count": 1, "expected": 3, "status": "partial", "mean": null, "median": null, "min": 0.5, "max": 0.5}} |
| E0.5-闭眼 | 通过 | {"status": "pass", "expected_status": "unavailable", "metric_status": {"eye_brightness": "unavailable", "eye_height": "unavailable", "eye_width": "unavailable"}, "specific_status_values": ["unavailable"]} |
| E0.5-遮挡 | 通过 | {"status": "pass", "expected_status": "unavailable", "metric_status": {"eye_brightness": "unavailable", "eye_height": "unavailable", "eye_width": "unavailable"}, "specific_status_values": ["unavailable"]} |
| E0.5-视角不同 | 通过 | {"status": "pass", "expected_status": "not_comparable", "metric_status": {"eye_brightness": "not_comparable", "eye_height": "not_comparable", "eye_width": "not_comparable"}} |

## 4. E1 原作品画风辨别（主要效度实验）

query 24 个 × 4 个画风参考集 → 聚合单元 96，逐图配对 576。主指标 **csd**。

| 指标 | Top-1 | 分摊并列 | 命中/总数 | 并列/错误/缺失 | Wilson 95% | cluster bootstrap 95%（query 簇） |
|---|---|---|---|---|---|---|
| adain | 0.5417 | 0.5417 | 13/24 | 0/11/0 | [0.351, 0.721] | [0.333, 0.75] |
| csd | 0.6667 | 0.6667 | 16/24 | 0/8/0 | [0.467, 0.82] | [0.458, 0.833] |
| edges | 0.5417 | 0.5417 | 13/24 | 0/11/0 | [0.351, 0.721] | [0.333, 0.75] |
| gram | 0.5 | 0.5 | 12/24 | 0/12/0 | [0.314, 0.686] | [0.292, 0.667] |
| lines | 0.2917 | 0.2917 | 7/24 | 0/17/0 | [0.149, 0.492] | [0.125, 0.458] |
| lpips | 0.4167 | 0.4167 | 10/24 | 0/14/0 | [0.245, 0.612] | [0.208, 0.625] |
| space | 0.25 | 0.25 | 6/24 | 0/18/0 | [0.12, 0.449] | [0.083, 0.417] |
| tone | 0.25 | 0.25 | 6/24 | 0/18/0 | [0.12, 0.449] | [0.083, 0.417] |

各画风命中率（主指标）：kishida_mel 2/6、puracotte 3/6、renian 6/6、sakurapion 5/6

4×4 混淆矩阵（行=真实画风，列=预测）：

| 真实\预测 | kishida_mel | puracotte | renian | sakurapion |
|---|---|---|---|---|
| kishida_mel | 2 | 1 | 2 | 1 |
| puracotte | 0 | 3 | 1 | 2 |
| renian | 0 | 0 | 6 | 0 |
| sakurapion | 0 | 1 | 0 | 5 |

错误 / 并列 / 缺失的 query ID：`kishida_mel-017`（真实 kishida_mel，预测 puracotte）；`kishida_mel-063`（真实 kishida_mel，预测 sakurapion）；`kishida_mel-125`（真实 kishida_mel，预测 renian）；`kishida_mel-187`（真实 kishida_mel，预测 renian）；`puracotte-133`（真实 puracotte，预测 renian）；`puracotte-189`（真实 puracotte，预测 sakurapion）；`puracotte-182`（真实 puracotte，预测 sakurapion）；`sakurapion-089`（真实 sakurapion，预测 puracotte）

| 主指标分布 | n | 均值 | 中位数 | p10 | p90 | 标准化间隔 |
|---|---|---|---|---|---|---|
| 同风格 | 24 | 0.73633 | 0.739588 | 0.626499 | 0.82777 | 0.960982 |
| 异风格 | 72 | 0.670862 | 0.672496 | 0.601137 | 0.747472 | — |

**结论**：主指标未达到预注册门槛（各风格 ≥2/3 与 Top-1 ≥80% 需同时满足）

试验集只作诊断：区间按 query/work 取样（不能把 576 个相关配对当 576 个独立样本），同风格参考集等权，不把不同画风的参考图混进一个均值。

## 5. E2 内容与配色干扰对照

| 指标 | 正确偏好 | n | 正确率 | 并列 | 错误 | 配色相近分块（n/正确率） | 景别+亮暗+背景匹配分块（n/正确率） |
|---|---|---|---|---|---|---|---|
| adain | 3 | 12 | 0.25 | 0 | 9 | 10/0.3 | 12/0.25 |
| csd | 2 | 12 | 0.1667 | 0 | 10 | 10/0.2 | 12/0.167 |
| edges | 5 | 12 | 0.4167 | 0 | 7 | 10/0.3 | 12/0.417 |
| gram | 3 | 12 | 0.25 | 0 | 9 | 10/0.3 | 12/0.25 |
| lines | 3 | 12 | 0.25 | 0 | 9 | 10/0.1 | 12/0.25 |
| lpips | 6 | 12 | 0.5 | 0 | 6 | 10/0.6 | 12/0.5 |
| space | 4 | 12 | 0.3333 | 0 | 8 | 10/0.2 | 12/0.333 |
| tone | 9 | 12 | 0.75 | 0 | 3 | 10/0.7 | 12/0.75 |

确定性干扰（N0–N4）后仍把 anchor 判为自身画风的比例：

| 变体 | 仍为自身画风 | 翻转 | 缺失 | 预测正确率 | 翻转到的画风 |
|---|---|---|---|---|---|
| N0 | 7 | 5 | 0 | 0.5833 | renian, sakurapion |
| N1 | 8 | 4 | 0 | 0.6667 | puracotte, sakurapion |
| N2 | 7 | 5 | 0 | 0.5833 | kishida_mel, sakurapion |
| N3 | 7 | 5 | 0 | 0.5833 | renian, sakurapion |
| N4 | 8 | 4 | 0 | 0.6667 | renian, sakurapion |
| original | 7 | 5 | 0 | 0.5833 | renian, sakurapion |

N1/N2 是配色/明度敏感性诊断，不要求指标不变；N3/N4 涉及边界与插值，不预设画风必然相同；不同尺度的分数不得直接相减比较。变体参数、掩膜与 hash 见 `E2/variants-frozen.json`，人工核查接触表见 `E2/contact-sheets/`。

## 6. E3 局部测量的受控响应

- 目标响应方向命中：**0.125**（门槛 0.9）；达标 6/48 个（案例 × 干预）
- 通过单调性检验的干预：hair_continuity
- 未通过单调性检验的干预：eye_brightness, eye_curvature, eye_gap, eye_height, eye_width, eyelashes, hair_fineness
- 负对照：12/12 个案例级负对照通过（要求贴近度精确为 1.0）
- 门槛判定：未通过（需同时满足方向命中 ≥ 0.9 与负对照全通过）
- 受控干预维度：hair_fineness, hair_continuity, eye_brightness, eye_height, eye_width, eye_gap, eyelashes, eye_curvature
- 诊断维度（无对应干预，**不得**宣称已验证）：eye_highlights, eye_tilt, face_ratios, iris_ratio, lid_weight

剂量响应（贴近度越低表示与该干预的原图差异越大）：

| 案例 | 干预 | 目标项 | 剂量贴近度（原→弱→中→强） | 单调 | 判定 |
|---|---|---|---|---|---|
| E3-PU-01 | hair_fineness | hair_fineness | 1 → 0.881 → 0.8665 → 0.7882 | 否 | 未通过 |
| E3-PU-01 | hair_continuity | hair_continuity | 1 → 0.41 → 0.3981 → 0.3715 | 是 | 通过 |
| E3-PU-01 | eye_brightness | eye_brightness | 1 → 0.9885 → 0.9943 → 0.9972 | 否 | 未通过 |
| E3-PU-01 | eye_height | eye_height | 1 → 0.9756 → 0.9524 → 0.9091 | 否 | 未通过 |
| E3-PU-01 | eye_width | eye_width | 1 → 0.9756 → 0.9524 → 0.9091 | 否 | 未通过 |
| E3-PU-01 | eye_gap | eye_gap | 1 → 0.9684 → 0.9388 → 0.8846 | 否 | 未通过 |
| E3-PU-01 | eyelashes | eyelashes | 1 → 0.9511 → 0.9535 → 0.9182 | 否 | 未通过 |
| E3-PU-01 | eye_curvature | eye_curvature | 1 → 0.9762 → 0.9545 → 0.9167 | 否 | 未通过 |
| E3-PU-02 | hair_fineness | hair_fineness | 1 → 0.881 → 0.7885 → 0.8299 | 否 | 未通过 |
| E3-PU-02 | hair_continuity | hair_continuity | 1 → 0.4105 → 0.3984 → 0.3725 | 是 | 通过 |
| E3-PU-02 | eye_brightness | eye_brightness | 1 → 0.9886 → 0.9943 → 0.9972 | 否 | 未通过 |
| E3-PU-02 | eye_height | eye_height | 1 → 0.9756 → 0.9524 → 0.9091 | 否 | 未通过 |
| E3-PU-02 | eye_width | eye_width | 1 → 0.9756 → 0.9524 → 0.9091 | 否 | 未通过 |
| E3-PU-02 | eye_gap | eye_gap | 1 → 0.9655 → 0.9333 → 0.8749 | 否 | 未通过 |
| E3-PU-02 | eyelashes | eyelashes | 1 → 0.9645 → 0.9621 → 0.9237 | 否 | 未通过 |
| E3-PU-02 | eye_curvature | eye_curvature | 1 → 0.9762 → 0.9545 → 0.9167 | 否 | 未通过 |
| E3-PU-03 | hair_fineness | hair_fineness | 1 → 0.881 → 0.8537 → 0.8718 | 否 | 未通过 |
| E3-PU-03 | hair_continuity | hair_continuity | 1 → 0.4108 → 0.3979 → 0.3715 | 是 | 通过 |
| E3-PU-03 | eye_brightness | eye_brightness | 1 → 0.9884 → 0.9943 → 0.9972 | 否 | 未通过 |
| E3-PU-03 | eye_height | eye_height | 1 → 0.9756 → 0.9524 → 0.9091 | 否 | 未通过 |
| E3-PU-03 | eye_width | eye_width | 1 → 0.9756 → 0.9524 → 0.9091 | 否 | 未通过 |
| E3-PU-03 | eye_gap | eye_gap | 1 → 0.9709 → 0.9434 → 0.8929 | 否 | 未通过 |
| E3-PU-03 | eyelashes | eyelashes | 1 → 0.9585 → 0.954 → 0.9418 | 否 | 未通过 |
| E3-PU-03 | eye_curvature | eye_curvature | 1 → 0.9762 → 0.9545 → 0.9167 | 否 | 未通过 |
| E3-SA-01 | hair_fineness | hair_fineness | 1 → 0.881 → 0.7885 → 0.8333 | 否 | 未通过 |
| E3-SA-01 | hair_continuity | hair_continuity | 1 → 0.4103 → 0.3984 → 0.3718 | 是 | 通过 |
| E3-SA-01 | eye_brightness | eye_brightness | 1 → 0.9927 → 0.9964 → 0.9982 | 否 | 未通过 |
| E3-SA-01 | eye_height | eye_height | 1 → 0.9756 → 0.9524 → 0.9091 | 否 | 未通过 |
| E3-SA-01 | eye_width | eye_width | 1 → 0.9756 → 0.9524 → 0.9091 | 否 | 未通过 |
| E3-SA-01 | eye_gap | eye_gap | 1 → 0.9701 → 0.9419 → 0.8902 | 否 | 未通过 |
| E3-SA-01 | eyelashes | eyelashes | 1 → 0.972 → 0.9677 → — | 否 | 未通过 |
| E3-SA-01 | eye_curvature | eye_curvature | 1 → 0.9762 → 0.9545 → 0.9167 | 否 | 未通过 |
| E3-SA-02 | hair_fineness | hair_fineness | 1 → 0.8571 → 0.7676 → 0.7404 | 否 | 未通过 |
| E3-SA-02 | hair_continuity | hair_continuity | 1 → 0.4105 → 0.3981 → 0.3714 | 是 | 通过 |
| E3-SA-02 | eye_brightness | eye_brightness | 1 → 0.9892 → 0.9946 → 0.9973 | 否 | 未通过 |
| E3-SA-02 | eye_height | eye_height | 1 → 0.9756 → 0.9524 → 0.9091 | 否 | 未通过 |
| E3-SA-02 | eye_width | eye_width | 1 → 0.9756 → 0.9524 → 0.9091 | 否 | 未通过 |
| E3-SA-02 | eye_gap | eye_gap | 1 → 0.9664 → 0.935 → 0.8779 | 否 | 未通过 |
| E3-SA-02 | eyelashes | eyelashes | 1 → 0.9556 → 0.9595 → 0.9235 | 否 | 未通过 |
| E3-SA-02 | eye_curvature | eye_curvature | 1 → 0.9762 → 0.9545 → 0.9167 | 否 | 未通过 |
| E3-SA-03 | hair_fineness | hair_fineness | 1 → 0.8393 → 0.6522 → 0.8115 | 否 | 未通过 |
| E3-SA-03 | hair_continuity | hair_continuity | 1 → 0.4106 → 0.3979 → 0.3715 | 是 | 通过 |
| E3-SA-03 | eye_brightness | eye_brightness | 1 → 0.9914 → 0.9957 → 0.9979 | 否 | 未通过 |
| E3-SA-03 | eye_height | eye_height | 1 → 0.9756 → 0.9524 → 0.9091 | 否 | 未通过 |
| E3-SA-03 | eye_width | eye_width | 1 → 0.9756 → 0.9524 → 0.9091 | 否 | 未通过 |
| E3-SA-03 | eye_gap | eye_gap | 1 → 0.9716 → 0.9448 → 0.8954 | 否 | 未通过 |
| E3-SA-03 | eyelashes | eyelashes | 1 → 0.9615 → 0.9576 → 0.9348 | 否 | 未通过 |
| E3-SA-03 | eye_curvature | eye_curvature | 1 → 0.9762 → 0.9545 → 0.9167 | 否 | 未通过 |

负对照明细（要求贴近度精确为 1.0）：

| 案例 | 负对照 | 期望不变项 | 实测贴近度 | 判定 |
|---|---|---|---|---|
| E3-PU-01 | NC1-hair | hair_fineness, hair_continuity | hair_fineness=1, hair_continuity=1 | 通过 |
| E3-PU-01 | NC2-eye | eye_brightness | eye_brightness=1 | 通过 |
| E3-PU-02 | NC1-hair | hair_fineness, hair_continuity | hair_fineness=1, hair_continuity=1 | 通过 |
| E3-PU-02 | NC2-eye | eye_brightness | eye_brightness=1 | 通过 |
| E3-PU-03 | NC1-hair | hair_fineness, hair_continuity | hair_fineness=1, hair_continuity=1 | 通过 |
| E3-PU-03 | NC2-eye | eye_brightness | eye_brightness=1 | 通过 |
| E3-SA-01 | NC1-hair | hair_fineness, hair_continuity | hair_fineness=1, hair_continuity=1 | 通过 |
| E3-SA-01 | NC2-eye | eye_brightness | eye_brightness=1 | 通过 |
| E3-SA-02 | NC1-hair | hair_fineness, hair_continuity | hair_fineness=1, hair_continuity=1 | 通过 |
| E3-SA-02 | NC2-eye | eye_brightness | eye_brightness=1 | 通过 |
| E3-SA-03 | NC1-hair | hair_fineness, hair_continuity | hair_fineness=1, hair_continuity=1 | 通过 |
| E3-SA-03 | NC2-eye | eye_brightness | eye_brightness=1 | 通过 |

非目标响应矩阵（每个干预对 13 项的平均贴近度；越大 = 越不敏感）：

| 干预 | eye_brightness | eye_curvature | eye_gap | eye_height | eye_highlights | eye_tilt | eye_width | eyelashes | face_ratios | hair_continuity | hair_fineness | iris_ratio | lid_weight |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eye_brightness | 0.996 | 1 | 1 | 1 | 0.986 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| eye_curvature | 0.996 | 0.962 | 1 | 0.949 | 1 | 1 | 1 | 0.984 | 0.987 | 1 | 1 | 0.95 | 1 |
| eye_gap | 1 | 1 | 0.949 | 1 | 0.997 | 1 | 1 | 0.997 | 1 | 1 | 1 | 0.999 | 0.999 |
| eye_height | 0.997 | 0.959 | 1 | 0.959 | 1 | 1 | 1 | 0.987 | 0.993 | 1 | 1 | 0.961 | 0.997 |
| eye_width | 0.998 | 1 | 0.857 | 1 | 0.985 | 1 | 0.959 | 0.966 | 0.993 | 1 | 1 | 0.997 | 0.972 |
| eyelashes | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.963 | 1 | 1 | 1 | 1 | 1 |
| hair_continuity | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.545 | 0.813 | 1 | 1 |
| hair_fineness | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.867 | 1 | 1 |

限制：机制测试基于参数化可控图样（标注坐标与像素同源），**不代表真实插画的效度**；真实图效度由 E1/E2/E4/E5 承担。

## 7. E4 自动定位与人工标注误差

分母含全部 **8** 张：6 个可控案例 + **2** 张历史 Gemini 图（协议写的是 6 + 4 = 10 张；该次历史测试只产出 2 张，已单列为协议偏离，历史失败图全部保留）。自动返回 8、可用 4、人工确认 0；返回率 1、可用率 0.5。

| 图 | 类型 | 自动提案 | 视角 | 未完成维度 |
|---|---|---|---|---|
| E3-PU-01 | synthetic_case | ok | frontal | — |
| E3-PU-02 | synthetic_case | ok | frontal | — |
| E3-PU-03 | synthetic_case | ok | frontal | hair_continuity, hair_fineness |
| E3-SA-01 | synthetic_case | ok | frontal | hair_continuity, hair_fineness |
| E3-SA-02 | synthetic_case | ok | frontal | — |
| E3-SA-03 | synthetic_case | ok | frontal | — |
| GEMINI-puracotte | historical_gemini | ok | frontal | hair_continuity, hair_fineness |
| GEMINI-sakurapion | historical_gemini | ok | frontal | hair_continuity, hair_fineness, lid_weight |

| 维度 | 自动可用 | 总数 | 覆盖率 |
|---|---|---|---|
| eye_brightness | 8 | 8 | 1 |
| eye_curvature | 8 | 8 | 1 |
| eye_gap | 8 | 8 | 1 |
| eye_height | 8 | 8 | 1 |
| eye_highlights | 8 | 8 | 1 |
| eye_tilt | 8 | 8 | 1 |
| eye_width | 8 | 8 | 1 |
| eyelashes | 8 | 8 | 1 |
| face_ratios | 8 | 8 | 1 |
| hair_continuity | 4 | 8 | 0.5 |
| hair_fineness | 4 | 8 | 0.5 |
| iris_ratio | 8 | 8 | 1 |
| lid_weight | 7 | 8 | 0.875 |

自动眼睑曲线 vs 可控图样解析真值（相对眼宽；**只对合成案例有意义**）：

| 图 | 曲线 | 平均偏差 / 眼宽 | 95 分位 / 眼宽 |
|---|---|---|---|
| E3-PU-01 | viewer_left_lower_lid | 0.0538 | 0.0922 |
| E3-PU-01 | viewer_left_upper_lid | 0.054 | 0.1007 |
| E3-PU-01 | viewer_right_lower_lid | 0.0509 | 0.0906 |
| E3-PU-01 | viewer_right_upper_lid | 0.0521 | 0.0979 |
| E3-PU-02 | viewer_left_lower_lid | 0.0708 | 0.1046 |
| E3-PU-02 | viewer_left_upper_lid | 0.0712 | 0.1342 |
| E3-PU-02 | viewer_right_lower_lid | 0.0688 | 0.1072 |
| E3-PU-02 | viewer_right_upper_lid | 0.0572 | 0.1082 |
| E3-PU-03 | viewer_left_lower_lid | 0.0481 | 0.087 |
| E3-PU-03 | viewer_left_upper_lid | 0.0907 | 0.1434 |
| E3-PU-03 | viewer_right_lower_lid | 0.05 | 0.087 |
| E3-PU-03 | viewer_right_upper_lid | 0.0951 | 0.1446 |
| E3-SA-01 | viewer_left_lower_lid | 0.0539 | 0.0952 |
| E3-SA-01 | viewer_left_upper_lid | 0.0618 | 0.1116 |
| E3-SA-01 | viewer_right_lower_lid | 0.0506 | 0.091 |
| E3-SA-01 | viewer_right_upper_lid | 0.0594 | 0.1154 |
| E3-SA-02 | viewer_left_lower_lid | 0.0628 | 0.0969 |
| E3-SA-02 | viewer_left_upper_lid | 0.0548 | 0.1094 |
| E3-SA-02 | viewer_right_lower_lid | 0.0642 | 0.0995 |
| E3-SA-02 | viewer_right_upper_lid | 0.0586 | 0.1004 |
| E3-SA-03 | viewer_left_lower_lid | 0.0531 | 0.0897 |
| E3-SA-03 | viewer_left_upper_lid | 0.0585 | 0.1047 |
| E3-SA-03 | viewer_right_lower_lid | 0.0511 | 0.0904 |
| E3-SA-03 | viewer_right_upper_lid | 0.0623 | 0.1094 |

人工部分：**人工标注尚未提供；自动 proposal 始终 confirmed=false。**

叠图人工观察（自动候选，未确认，仅作证据；已核对 `GEMINI-puracotte-auto-overlay.png`）：发丝路径整条落在面部与马甲上、头发区域多边形越出头发并压住脸与衣服，与叠图里可见的实际发束不一致；这也解释了为什么该图的 `hair_fineness`/`hair_continuity` 保持 unavailable，而 E3-PU-03 / E3-SA-01 两个合成案例出现同类越界。

A↔H1 / H1↔H2 / H1↔H1-repeat 的偏差与排名翻转率：**未计算（awaiting_human）**。标注入口见 `E4/human-annotation-entry.html`，schema 见 `E4/annotation-brief.json`。

自动候选叠图（未确认，仅供人工核查）：`E3-PU-01-auto-overlay.png`、`E3-PU-02-auto-overlay.png`、`E3-PU-03-auto-overlay.png`、`E3-SA-01-auto-overlay.png`、`E3-SA-02-auto-overlay.png`、`E3-SA-03-auto-overlay.png`、`GEMINI-puracotte-auto-overlay.png`、`GEMINI-sakurapion-auto-overlay.png`

## 8. E5 真实 Gemini 候选排序与独立复核

| 画风 | 呈现顺序 | 最佳候选 | 候选数 | 门禁触发 | 选择状态 |
|---|---|---|---|---|---|
| puracotte | forward | history/gemini/0 | 1 | 无 | ranked |
| puracotte | reverse | history/gemini/0 | 1 | 无 | ranked |
| sakurapion | forward | — | 1 | history/gemini/0:explicit_subject_mismatch | no_eligible |
| sakurapion | reverse | — | 1 | history/gemini/0:explicit_subject_mismatch | no_eligible |

顺序敏感性：{"rows": [{"style_id": "puracotte", "candidate": "history/gemini/0", "forward": 7.541, "reverse": 7.413, "abs_delta": 0.1280000000000001, "best_forward": "history/gemini/0", "best_reverse": "history/gemini/0"}, {"style_id": "sakurapion", "candidate": "history/gemini/0", "forward": 7.703, "reverse": 7.779, "abs_delta": 0.07599999999999962, "best_forward": null, "best_reverse": null}], "ranked_best_changed": false, "note": "两个顺序独立运行、都保留；不得择优保留。"}

新增生成槽位：计划 **18**，已授权 **0**，状态 `awaiting_budget`。冻结参数：{"generation_entry": "app.py SingleGenDebugWidget（真实 GUI 生成入口）→ ImageGenWorkerThread → modules/others/api_backend.generate_image_aigc2d", "channel": "gemini", "model": "gemini-3-pro-image-preview", "resolution": "2K", "aspect_ratio": "2:3", "style_ref_mode": "priority", "reference_attachment_order": ["画风参考图（config ref_image，位于文后 STYLE 段之前）"], "text_length_policy": "参考优先模式使用 prompt_compressed", "gpt_image_used": false, "repaint_used": false, "post_process_used": false, "subject_text": "prompts/style-extraction/similarity-validation-20261006.txt", "note": "冻结自 2026-10-06 的实测快照（见 PLAN/e5-plan.json 与历史 request replay）。"}

盲评包：`D:\code\image-maker\data\test-result\20261006\style-similarity-controlled-v1\run-20261006-01\e5\blind\blind-questionnaire.html`（题数 6，门槛 24），状态 **awaiting_human**；作答模板与隐藏映射见同目录 `blind-answers-template.json` / `blind-map.json`。

视觉模型与人类的一致率、两人分歧率、重复题一致性：**未计算（缺人类作答）**。视觉复核与自动定位同属同端点模型族，**不是独立第三方评审**。

## 9. 尝试账本摘要

```json
{
  "rows": 642,
  "jobs": 113,
  "by_stage_status": {
    "E0|ok": 451,
    "E0|reused": 1,
    "E1|ok": 48,
    "E2|ok": 48,
    "E2|reused": 1,
    "E3|ok": 48,
    "E3|reused": 1,
    "E4|failed": 3,
    "E4|ok": 16,
    "E5|failed": 6,
    "E5|ok": 12,
    "E5|reused": 2,
    "PLAN|ok": 5
  },
  "retried_jobs": [
    "e4:propose:E3-PU-01",
    "e4:propose:E3-PU-02",
    "e4:propose:E3-SA-02"
  ],
  "retried_job_count": 3,
  "jobs_repeated_across_runs": [
    "E5:cached",
    "e0:cache-reuse",
    "e0:entry-cli",
    "e0:entry-extraction",
    "e0:repeat-a",
    "e0:repeat-b",
    "e0:self-1",
    "e0:self-2",
    "e0:self-3",
    "e0:self-4",
    "e0:self-5",
    "e0:self-6",
    "e0:sym-1-2",
    "e0:sym-1-3",
    "e0:sym-1-4",
    "e0:sym-1-5",
    "e0:sym-1-6",
    "e0:sym-2-1",
    "e0:sym-2-3",
    "e0:sym-2-4",
    "e0:sym-2-5",
    "e0:sym-2-6",
    "e0:sym-3-1",
    "e0:sym-3-2",
    "e0:sym-3-4",
    "e0:sym-3-5",
    "e0:sym-3-6",
    "e0:sym-4-1",
    "e0:sym-4-2",
    "e0:sym-4-3",
    "e0:sym-4-5",
    "e0:sym-4-6",
    "e0:sym-5-1",
    "e0:sym-5-2",
    "e0:sym-5-3",
    "e0:sym-5-4",
    "e0:sym-5-6",
    "e0:sym-6-1",
    "e0:sym-6-2",
    "e0:sym-6-3",
    "e0:sym-6-4",
    "e0:sym-6-5",
    "e1:kishida_mel-017",
    "e1:kishida_mel-063",
    "e1:kishida_mel-113",
    "e1:kishida_mel-125",
    "e1:kishida_mel-187",
    "e1:kishida_mel-227",
    "e1:puracotte-025",
    "e1:puracotte-098",
    "e1:puracotte-133",
    "e1:puracotte-182",
    "e1:puracotte-189",
    "e1:puracotte-239",
    "e1:renian-002",
    "e1:renian-005",
    "e1:renian-008",
    "e1:renian-016",
    "e1:renian-018",
    "e1:renian-062",
    "e1:sakurapion-001",
    "e1:sakurapion-003",
    "e1:sakurapion-041",
    "e1:sakurapion-074",
    "e1:sakurapion-089",
    "e1:sakurapion-135",
    "e2:compare:kishida_mel-017",
    "e2:compare:kishida_mel-063",
    "e2:compare:kishida_mel-113",
    "e2:compare:puracotte-025",
    "e2:compare:puracotte-098",
    "e2:compare:puracotte-133",
    "e2:compare:renian-002",
    "e2:compare:renian-005",
    "e2:compare:renian-062",
    "e2:compare:sakurapion-041",
    "e2:compare:sakurapion-074",
    "e2:compare:sakurapion-089",
    "e2:variants:kishida_mel-017",
    "e2:variants:kishida_mel-063",
    "e2:variants:kishida_mel-113",
    "e2:variants:puracotte-025",
    "e2:variants:puracotte-098",
    "e2:variants:puracotte-133",
    "e2:variants:renian-002",
    "e2:variants:renian-005",
    "e2:variants:renian-062",
    "e2:variants:sakurapion-041",
    "e2:variants:sakurapion-074",
    "e2:variants:sakurapion-089",
    "e3:E3-PU-01",
    "e3:E3-PU-02",
    "e3:E3-PU-03",
    "e3:E3-SA-01",
    "e3:E3-SA-02",
    "e3:E3-SA-03",
    "e4:propose:E3-PU-03",
    "e4:propose:E3-SA-01",
    "e4:propose:E3-SA-03",
    "e4:propose:GEMINI-puracotte",
    "e4:propose:GEMINI-sakurapion",
    "e5:review:puracotte:dataset",
    "e5:review:puracotte:forward",
    "e5:review:puracotte:reverse",
    "e5:review:sakurapion:forward",
    "e5:review:sakurapion:reverse",
    "plan:freeze"
  ],
  "jobs_repeated_across_runs_count": 107,
  "note": "账本按 job_id 累积；同一 job 在多轮 CLI 调用里重复执行（重复测量/复用）不计为重试。",
  "failed_rows": 9,
  "cost_unknown_jobs": [
    "e4:propose:E3-PU-01",
    "e4:propose:E3-PU-02",
    "e4:propose:E3-PU-03",
    "e4:propose:E3-SA-01",
    "e4:propose:E3-SA-02",
    "e4:propose:E3-SA-03",
    "e4:propose:GEMINI-puracotte",
    "e4:propose:GEMINI-sakurapion",
    "e5:review:puracotte:dataset",
    "e5:review:puracotte:forward",
    "e5:review:puracotte:reverse",
    "e5:review:sakurapion:forward",
    "e5:review:sakurapion:reverse"
  ],
  "http_attempts_unknown": [
    "e4:propose:E3-PU-01",
    "e4:propose:E3-PU-02",
    "e4:propose:E3-SA-02",
    "e5:review:puracotte:dataset"
  ],
  "charged_jobs": [
    "e4:propose:E3-PU-01",
    "e4:propose:E3-PU-02",
    "e4:propose:E3-PU-03",
    "e4:propose:E3-SA-01",
    "e4:propose:E3-SA-02",
    "e4:propose:E3-SA-03",
    "e4:propose:GEMINI-puracotte",
    "e4:propose:GEMINI-sakurapion",
    "e5:review:puracotte:dataset",
    "e5:review:puracotte:forward",
    "e5:review:puracotte:reverse",
    "e5:review:sakurapion:forward",
    "e5:review:sakurapion:reverse"
  ],
  "reused_jobs": [
    "E0:cached",
    "E2:cached",
    "E3:cached",
    "E5:cached"
  ]
}
```

计费：本轮唯一产生外部请求的是「E4 自动区域定位」（文本/视觉端点）与「E5 视觉复核」（同族端点）；两者按次计费但账户分项单价未知，统一记 `cost=unknown`、`may_have_charged=true`，且不声称符合费用上限。**没有发出任何新增生图请求**（18 槽位待额度）。

计费 job 共 **13** 个：E4 自动定位 8 张 + E5 视觉复核 4 次运行（两画风 × 参考图正序/倒序）。其中 3 个 E4 job 首次响应不符合 schema、重试一次后成功；1 个 E5 job 的连续失败来自本地状态接线的实现缺陷（已修），不是服务端拒绝。`attempt_id` 在同一 job_id 上**跨多次 CLI 调用累计**，所以会出现大于 3 的编号；协议允许的「首次 + 2 次重试」是单次运行的规则。

## 10. 用途结论

| 用途 | 结论 | 依据 |
|---|---|---|
| 实现可用（同图 / 对称 / 缓存 / 入口一致 / 失败状态） | 通过 | E0 全部关键项通过（16 项检查） |
| 全图八组作为画风检索辅助 | 仅可辅助（未过预注册门槛） | E1 主指标 csd Top-1=0.6667（门槛 0.8）；E2 困难对照 CSD 正确偏好率=0.1667（门槛 0.75，未达标） |
| 自动定位默认参与局部可靠测量 | 未验证 | E4 自动可用 4/8（可用率 0.5）；人工确认 0；人工间重复性未提供 → 人工间重复性与人工对照尚未提供，且自动可用率/发丝采样带有效率需人工确认后复核 |
| 人工确认定位后的局部细项测量 | 未验证 | E3 受控图样目标方向命中 0.125（门槛 0.9）；负对照 未全部通过；未验证维度 eye_highlights、eye_tilt、face_ratios、iris_ratio、lid_weight |
| 自动选最佳候选（候选排序） | 未验证 | 人类作答 0；盲评包 6 个问题中真正的候选 A/B 比较只有 1 题，门槛 24 个有效独立题；视觉模型已复核 4 次运行，但两人一致性、方向一致率均无法计算 |

## 11. 尚需人工 / 额度 / 数据的具体项目

| # | 类型 | 具体缺项 | 阻塞的实验 |
|---|---|---|---|
| 1 | human | E4 的两名人类标注者（H1/H2）与 H1 隔 ≥24 小时复标；眼睑曲线、虹膜、睫毛、脸轮廓、头发 mask/path 叠图核查 | 自动定位偏差、人工间重复性、自动 vs 人工排名翻转率、自动定位能否默认参与正式测量 |
| 2 | human | E5 两名人类各 ≥24 个有效独立 A/B 题（含 6 个重复题检验个人一致性）；当前盲评包里真正的组内候选 A/B 题只有 1 个 | 候选排序效度、视觉模型与人类的一致率 |
| 3 | budget | E5 新增 18 个生图槽位（2 画风 × 3 主体 × 3 重复）的费用授权；当前已授权 0 | E5 新增候选、组内排序与盲评题数 |
| 4 | human | E2 的 N0–N4 变体人工核查（人物结构是否被意外改变）与「配色/构图匹配」人工标注 | 困难对照的选择依据是否成立 |
| 5 | data | TID 画风可验证原作品集（本地仅有 2 张素材），当前用 Renian 替代 | 与协议原始画风清单完全对齐 |
| 6 | data | 确认集（每画风 20 个新独立 query 作品） | 从「试验集诊断」升级为通用可靠性结论 |

## 12. 下一轮唯一优先改进与验证方案

先解决自动定位在真实动漫图上的精度问题：以 6 个可控案例的解析真值 + 4 张历史图建立人工叠图基线，把眼睑曲线偏差（相对眼宽）与发丝采样带有效率作为唯一验收指标，先做人工定位条件下的稳定性检验，再考虑改进定位提示词；改提示词属于新版本，不回改本次结果。

## 附：结果摘要模板（按协议 §12 填写）

```text
协议/代码版本：1.0 / 代码 hash 见 PLAN/code-freeze.json
冻结包与运行目录：D:\code\image-maker\data\test-result\20261006\style-similarity-controlled-v1\run-20261006-01
实际独立原作品/query/主体组数：48/24/6
E0：complete；关键失败：[]
E1：主指标 csd n=24 Top-1=0.6666666666666666 区间={'n': 24, 'k': 16, 'p': 0.6666666666666666, 'low': 0.4670631683813174, 'high': 0.8202780967270225} 每画风={kishida_mel:2/6, puracotte:3/6, renian:6/6, sakurapion:5/6}
E2：困难对照 CSD 正确偏好 2/12；变体翻转见 E2 阶段结果
E3：目标响应 0.125；负对照 未全通过；门槛未通过
E4：自动有效 4/8；人工确认 0/8（awaiting_human）
E5：人类有效独立 A/B 题 = 0（盲评包共 6 个问题，其中真正的候选 A/B 比较只有 1 题，其余为参考画风识别题；门槛 24，awaiting_human）；视觉已复核 4/4 次运行（两画风 × 参考图正序/倒序）；排序一致性未计算
重试/未知收费/缓存复用：3 / 13 / 4（重试 job 数 / 费用未知 job 数 / 复用 job 数）
哪些用途可用、哪些只能探索、哪些不能用：见报告第 10 节
尚需人工/额度/数据的具体项目：见报告第 11 节
下一轮唯一优先改进及验证方案：先解决自动定位在真实动漫图上的精度问题：以 6 个可控案例的解析真值 + 4 张历史图建立人工叠图基线，把眼睑曲线偏差（相对眼宽）与发丝采样带有效率作为唯一验收指标，先做人工定位条件下的稳定性检验，再考虑改进定位提示词；改提示词属于新版本，不回改本次结果。
```