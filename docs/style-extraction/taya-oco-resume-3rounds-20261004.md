# taya_oco：在既有 3 轮结果上新增第 4–6 轮识别、对账精修与测试（2026-10-04）

在已核查的 taya_oco 基线上续跑 3 轮（第 4、5、6 轮），沿用最终分析母版做对账精修、局部裁剪细化、全量终审与多用途 Prompt 包，并对每个新增轮次与新增终审版本跑完 Gemini 直出、GPT 首图、GPT→Gemini 重绘三路测试；随后对同一比较集合做 v2 十二维视觉重评与 Intel NPU 本地深度指标。原结果、原轮次、原终审候选、原画风配置均未覆盖，最佳版本留给用户人工选择。

本次没有修改 App 业务代码（直接同步调用同一个 `StyleIterativeWorkerThread.run()`），没有创建虚拟环境，没有重新下载/安装模型权重。

## 1. 基线与续跑来源

| 项 | 值 |
|---|---|
| 源 JSON | `data/20261004/taya_oco_style_iter_result.json` |
| SHA-256（执行前后一致） | `D37C3658C0CB16B9A89F6164F8C66703ACB490A913B902BB2EC6D1B9B13564D7` |
| 基线 `updated_at` | 2026-10-04 19:09:18（32 个迭代步骤，已完成第 1–3 轮） |
| 数据集 | `data/style-datasets/taya_oco-9273d2a46328/images`，12 张，全部存在且可读 |
| 固定测试主体 | `a girl sitting beside lake, she has pink hair and brown eyes, spring,` |
| 测试画风参考图 | `data/style-datasets/taya_oco-9273d2a46328/images/145818801_p0.png`（2250×3539 → 比例吸附 2:3） |
| 基线终审母版 | 18385 字符，SHA-256 `a63e5df78c624e369956e88a116e16262553b0340d743c45f94c74e291aff667` |
| 基线终审三路 | `success`（Gemini 直出 / GPT 首图 / GPT→Gemini 重绘各 1 张） |
| 基线视觉比较 | `style-comparison-v1`，12 行，`best_id = final/gemini_direct/0`（历史判断，不作为本次结论） |

执行前搜索了全部 `*_style_iter_result.json`：除上述文件外，`data/` 下只有 millon-knots、sakurapion、kishida-mel、renian、komori-hikki 等其他画风，以及 `data/style-datasets/taya_oco-820660e60023`（另一个同名数据集，未使用）。**没有比该文件更新的 taya_oco 续跑记录**，故以它为基线。

`_get_current_prompts(iterations)` 恢复的正文与 `final_art_style_prompts` 逐字节一致（同为 18385 字符、同一 SHA-256），记录无缺损，不需要修复，也不是从零提取。

## 2. 执行方式与参数

- 无头 runner（运行脚本快照，不参与运行链路）：[taya_oco_resume_runner.py](tools/taya_oco_resume_runner.py)，执行副本 `cache/temp/taya_oco_resume_runner.py`。
- 业务模块：`modules.image_analysis.style_analyzer.StyleIterativeWorkerThread`（同步调用其 `run()`）、`StyleComparisonWorker` / `calculate_ranking` / `write_comparison_report`、`tools/style_metrics_verify.py --comparison-state --device npu`。
- 文本端点/模型取自 `conf/config.json`（`https://new.aigc2d.com/v1`、`gpt-5.6-luna`），密钥经 `api_backend.resolve_text_api_key` 从环境变量解析（日志只打印变量名，不含密钥值）。
- 图片节点沿用既有配置：`aigc2d` = `gemini-3-pro-image-preview`（2K），`aigc-2d-gpt` = `gpt-image-2`（1024×1536）。
- 参数：`total_rounds=3`（= 本次新增 3 轮，不是 6）、`images_per_round=4`、`file_prefix=taya_oco`、测试生图开启、比例 `auto`（跟随参考图 → 2:3）、文本请求超时 600s（无头覆盖，界面默认值更短）。
- 输出目录（绝对路径，独立于源 JSON 结果文件）：`data/20261004/style-extraction/taya_oco/resume-20261004-213600/`。
- 线程结束状态 `success`，字节 21:36 → 21:47（约 11 分钟），64 个迭代步骤（基线 32 + 新增 32）。

## 3. 新增轮次实际做了什么

每轮 = 1 次「全 12 张参考图对账精修」+ 4 次「单图差异检查」+ 1 次该轮 Prompt 包派生 + 三路测试生图。

| 轮 | 对账精修改后母版长度 | 本抽查的 4 张参考图 | 差异检查置信度 | 该轮测试图状态 |
|---|---:|---|---:|---|
| 4 | 13270（step 37） | 106964347 / 139008664 / 105354606 / 104148485 | 0.98 ×4 | success |
| 5 | 10574（step 42） | 143837423 / 105354606 / 131716466 / 139008664 | 0.98 ×4 | success |
| 6 | 12149（step 47） | 145818801 / 131716466 / 106964347 / 113548787 | 0.99 ×4 | success |

抽查图由 worker 随机抽样（无固定种子），因此与第 1–3 轮不是同一批；第 1–3 轮的收费请求没有被重跑。

3 轮之后的收尾工序（全部为本次新增的收费请求）：

- **局部裁剪细化**：12 张参考图共裁出 60 个区域（head_hair / face_closeup / upper_body / lower_body / detail_center 各 12），按「同区域跨参考图」分成 15 批分析（置信度 0.82–0.94），合并（`local_merge`，12149 → 21036 字符，**合并步骤自报置信度 0.5**）。
- **全量终审**：12 张参考图 + 完整迭代历史送审，21036 → **16014 字符**（置信度 0.99），SHA-256 `bc1d435b259d5c24…`。
- **多用途 Prompt 包**：GPT 短版 520 字符，`validate_prompt_gpt` 校验通过（`gpt_image_prompt_errors` 为空）。

新增终审母版与基线终审母版不同（长度 16014 vs 18385），是本次对账+局部细化+终审的实际产物，不是复用旧文本。

## 4. 三路测试生图：成功/失败

新增 12 张测试图，**12/12 成功，失败 0、未运行 0**（每轮 3 张 + 新增终审 3 张）。

| 版本 | Gemini 直出 | GPT 首图 | GPT→Gemini 重绘 |
|---|---|---|---|
| 第 4 轮 | 1 | 1 | 1 |
| 第 5 轮 | 1 | 1 | 1 |
| 第 6 轮 | 1 | 1 | 1 |
| 新增终审 | 1 | 1 | 1 |

各通道产物目录：

- 第 4 轮 `test-generations/round-04/`：`round-04-gemini-direct_213208-f3d544.jpg`、`round-04-gpt-first_output_213333_0_6906c7.png`、`…-213333-d833ba-final-rp_213403-d2a3d0.jpg`
- 第 5 轮 `test-generations/round-05/`：`round-05-gemini-direct_213653-1a1265.jpg`、`round-05-gpt-first_output_213727_0_f6465a.png`、`…-213727-42bb6f-final-rp_213753-5bfd78.jpg`
- 第 6 轮 `test-generations/round-06/`：`round-06-gemini-direct_214017-9ce0f3.jpg`、`round-06-gpt-first_output_214045_0_62e1d9.png`、`…-214045-32a8f0-final-rp_214121-cbbbde.jpg`
- 新增终审 `test-generations/final/`：`final-gemini-direct_214611-01f56f.jpg`、`final-gpt-first_output_214643_0_cf5762.png`、`…-214643-a0f3de-final-rp_214713-7c0e87.jpg`

每张测试图都绑定「当时实际使用的母版 + 派生 Prompt 包」，写在该 stage 的 `prompts_used` / `prompt_variants` 里（见 `test_images.round_4/5/6`、`final_test_images`）。旧轮次与旧终审的产物全部保留，没有任何用旧图冒充新测试的情况。

## 5. 视觉比较（v2 十二维量表）

### 5.1 主比较集：24 个候选（旧 3 轮 + 旧终审 + 新 3 轮 + 新终审）

- 结果：`automatic_comparison`（`style-comparison-v2`）、`automatic-comparison.json`、自包含报告 `automatic-comparison.html`。
- 候选顺序：`final`（新终审）/ `final_v1_baseline`（旧终审，独立 stage）/ `round_1`…`round_6`，每个 stage 三通道各 1 张。
- 旧终审没有被子集或新 final 覆盖：它同时存在于 `test_images.final_v1_baseline` 与 `final_test_images_baseline`，并带自己的 Prompt 包。

| 通道（8 个 stage） | 总分 | 画风分 | 主体符合度 | 门禁 |
|---|---|---|---|---|
| gemini_direct ×8 | 8.343（round_5 为 8.3） | 8.05（round_5 为 8.0） | 10.0 | 全部通过 |
| gpt_first_pass ×8 | 7.865（round_5 为 8.3） | 8.9（round_5 为 8.0） | 2.0（round_5 为 10.0） | 7 个 `reference_content_copied` + `explicit_subject_mismatch`；**round_5 通过** |
| gpt_repainted ×8 | 7.808 | 8.833 | 2.0 | 全部 `reference_content_copied` + `explicit_subject_mismatch` |

**这个集合有一个重要局限（如实记录）**：36 张图（12 源图 + 24 候选）一次请求时，模型给出了 24 条互不相同的文字理由，但数值只有 4 种组合（基本按通道取常量），导致 gemini_direct 组出现 7 个候选并列 8.343。因此主集的 `best_id = final/gemini_direct/0` 是**并列时的候选 ID 排序结果，不构成对新终审的真实偏好**，不能当作「最后一轮更好」的证据。round_5 的 GPT 首图是唯一未被门禁排除的 GPT 通道候选（它没有抄参考角色的发饰与服装）。

### 5.2 补充比较集：8 个 gemini_direct 候选（同一量表/同一模型/同一源图与主体）

为在唯一全部通过门禁的通道内取得可区分排名，另做一次独立评分（20 张图：12 源图 + 8 候选），结果单独存放，**不与主集分数混排**（`input_hash` 不同）：

| 候选 | 总分 | 画风分 | 主体 | 门禁 |
|---|---:|---:|---:|---|
| final/gemini_direct/0（新终审） | 8.551 | 8.383 | 9.5 | 通过 |
| **round_4/gemini_direct/0** | 8.508 | 8.333 | 9.5 | 通过 |
| round_3/gemini_direct/0 | 8.494 | 8.317 | 9.5 | 通过 |
| round_5/gemini_direct/0 | 8.459 | 8.275 | 9.5 | 通过 |
| round_6/gemini_direct/0 | 8.381 | 8.183 | 9.5 | 通过 |
| final_v1_baseline/gemini_direct/0（旧终审） | 8.246 | 8.025 | 9.5 | 通过 |
| round_1/gemini_direct/0 | 8.246 | 8.025 | 9.5 | 通过 |
| round_2/gemini_direct/0 | 8.161 | 7.925 | 9.5 | 通过 |

（本表分数与 5.1 表不可直接比较：来自不同请求、不同 `input_hash`。）

## 6. 本地深度指标（Intel NPU，未转 CUDA）

- 命令：`python tools/style_metrics_verify.py --comparison-state <新状态 JSON> --device npu`
- 结果：`deep-feature-comparison-npu-fp16.json`、自包含报告 `deep-feature-comparison.html`、运行日志 `deep-feature-comparison.log`
- 请求设备 `npu`，实际 `npu`，`fallback=false`，`backend=openvino-npu`，设备名 `Intel(R) AI Boost`，精度 `fp16`；**本次没有探测也没有使用 CUDA**（运行日志 0 处 `cuda`；结果 JSON 里唯一的 `cuda` 字样来自固定的 LIMITS 说明文本「NPU FP16 与 CUDA/CPU FP32 分开保存」，执行证据 `execution_devices` 全部是 NPU）。
- 执行证据：VGG 特征编码器、LPIPS、CSD 三个模型的 `execution_devices` 均为 `["NPU"]`（compiled `EXECUTION_DEVICES=NPU`、`INFERENCE_PRECISION_HINT=float16`、NPU 驱动 1003717、编译器 393219）；按部署口径，Gram/AdaIN 的通道统计与 CSD 点积在 CPU。
- 覆盖：24 个候选 × 12 张源图 = **288 组真实配对**，四项全部 `status=ok`，每行 `count=12 / expected=12`，没有把缺失当 0。
- 同图自检：Gram 0、AdaIN 0、LPIPS 0（容差 1e-6）、CSD 1.0（容差 1e-4），全部通过。
- 权重 SHA-256 全部匹配：vgg19 `dcbb9e9dad56…`、lpips_alex `df73285e35b2…`、lpips trunk `7be5be791159…`、csd `40e92fad63a3…`、clip_vit_l14 `b8cca3fd41ae…`。
- 预处理（写入结果 JSON 与报告）：VGG-19 直接拉伸 512×512、Gram 5 层 / AdaIN 4 层；LPIPS-Alex v0.1 直接拉伸 256×256；CSD 224 短边缩放 + 中心裁剪；ImageNet 均值方差归一化。
- `training_parameters` 记录为 `{"total_rounds": 3, "images_per_round": 4}`（本次新增轮数语义）。

四项指标（全部 12 张源图等权，均值 / 中位数 / 范围见 JSON；**不换算百分比、不并入视觉总分**）：

| 候选 | CSD ↑ | Gram ↓ | AdaIN ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| round_4/gemini_direct | 0.633446 | **2.5960e-05** | **70.551** | 0.629150 |
| final/gemini_direct（新终审） | 0.615441 | 2.9852e-05 | 81.134 | 0.659790 |
| final_v1_baseline/gemini_direct（旧终审） | 0.635686 | 3.0594e-05 | 80.752 | 0.664510 |
| round_3/gemini_direct | **0.668403** | 3.5260e-05 | 82.128 | 0.626302 |
| round_5/gemini_direct | 0.624054 | 4.3879e-05 | 82.750 | **0.625000** |
| round_6/gemini_direct | 0.606350 | 3.0231e-05 | 76.745 | 0.664958 |
| round_1/gemini_direct | 0.579300 | 3.4595e-05 | 84.843 | 0.668254 |
| round_2/gemini_direct | 0.617028 | 2.9528e-05 | 78.566 | 0.681112 |
| round_5/gpt_repainted | 0.746770 | 3.9528e-05 | 88.926 | 0.583168 |
| round_1/gpt_repainted | 0.738094 | 4.1749e-05 | 91.791 | 0.584885 |
| round_6/gpt_repainted | 0.727530 | 4.1814e-05 | 92.463 | 0.590398 |
| final/gpt_repainted | 0.700243 | 3.8196e-05 | 85.773 | 0.576823 |
| final/gpt_first_pass | 0.677037 | 4.9981e-05 | 101.939 | 0.597310 |
| round_6/gpt_first_pass | 0.669912 | 4.7611e-05 | 98.466 | 0.589884 |
| round_4/gpt_first_pass | 0.616664 | 5.7657e-05 | 114.621 | 0.659871 |
| round_5/gpt_first_pass | 0.603737 | 6.5998e-05 | 125.971 | 0.643311 |

（表中只列部分行，完整 24 行在 JSON/HTML 里；未列出的 `final_v1_baseline/round_1/round_2/round_3` 的 GPT 通道与 gemini_direct 数值同样完整。）

**四个通道内排名（只看通过门禁的 gemini_direct）：**

- CSD ↑：round_3 (0.6684) > 旧终审 (0.6357) > round_4 (0.6334) > round_5 > round_2 > 新终审 (0.6154) > round_6 > round_1
- Gram ↓：round_4 (2.596e-05) > round_2 > 新终审 > round_6 > 旧终审 > round_1 > round_3 > round_5
- AdaIN ↓：round_4 (70.55) > round_6 > round_2 > 旧终审 > 新终审 > round_3 > round_5 > round_1
- LPIPS ↓：round_5 (0.6250) > round_3 > round_4 > 新终审 > 旧终审 > round_6 > round_1 > round_2

## 7. 最佳与备选（供人工选择，未写入画风配置）

**最佳建议：第 4 轮的 Prompt 包，生图通道 = Gemini 直出（`round_4/gemini_direct/0`）。**

依据：

1. **门禁**：v2 四项门禁全否，主体符合度 10/10（补充集 9.5/10），是允许自动选用的候选之一。
2. **视觉**：补充集第 2 名 8.508，与第 1 名（新终审 8.551）只差 0.043，属于同一档；五项分组里「线条与边缘」8.5 为 8 个候选中最高。
3. **深度指标**：Gram **第 1**（2.596e-05，明显低于其余）、AdaIN **第 1**（70.55，比第 2 名低 6.2），CSD 第 3（0.6334，距第 1 名 0.035）、LPIPS 第 3（0.6292，距第 1 名 0.004）——四项全部进前三，是唯一在两个「越低越好」的多尺度特征距离上同时领先的候选。
4. **主体**：目视粉发、棕眼、湖边坐姿、春季（樱花/新绿/水面）全部保留，参考角色（白发带大蝴蝶结、白蕾丝+白丝袜+靴子的角色）没有被搬入。

**备选 1：新增终审包（`final/gemini_direct/0`）** —— 视觉第 1（8.551），但 Gram 第 3、AdaIN 第 5、CSD 第 6、LPIPS 第 4，四项深度指标都不在前二；理由与最佳只差 0.043。若更看重「终审是全量参考图统一审查的产物」这一流程理由，可以选它，但**不能仅因为它是最新版本**而选它。

**备选 2：第 3 轮包（`round_3/gemini_direct/0`）** —— CSD 第 1（0.6684，画风描述子最接近 12 张源图），LPIPS 第 2，但 Gram 第 7、AdaIN 第 6；目视它也是最华丽的一版（粉蕾丝、层叠荷叶边），与 TAYA 参考图的高密度装饰语言最像，代价是更碎、更花。

**不一致的地方（取舍）**：CSD 与视觉/其他三项并不同向——CSD 最高的是 `round_5/gpt_repainted`（0.7468）、`round_1/gpt_repainted`（0.7381）等**重绘通道**候选，而它们全部因「抄入参考角色身份/服装 + 主体不符」被门禁排除。也就是说「深度特征更像画风」在这一集合里与「把参考角色搬进来」高度相关，**不能单独用 CSD 推翻主体门禁**，这与 millon-knots 的观察一致。

**没有强制选最佳**：以上三项都通过了门禁，最终选用哪一个由用户看过 `automatic-comparison.html` / `deep-feature-comparison.html` 后决定；本次不覆盖 App 已安装的 taya_oco 画风配置。

## 8. 主体保持检查（粉发 / 棕眼 / 湖边坐姿 / 春季）

- 模型门禁（v2）：8 个 gemini_direct 候选 `explicit_subject_mismatch=false`、`reference_content_copied=false`、`uncertain=false`，主体分 10/10（主集）与 9.5/10（补充集）；GPT 首图与重绘 15/16 个候选被标 `reference_content_copied=true` + `explicit_subject_mismatch=true`（主体分 2/10），round_5 的 GPT 首图例外。
- 人工目视（本次产出的对照图 `eligible-gemini-direct.jpg`、`face-upper-gemini_direct.jpg`）：8 个 gemini_direct 候选全部保留粉色长发、棕色虹膜、水边坐姿与春季樱花/新绿背景；差异主要在服装设计（白裙 / 米色蕾丝 / 鼠尾草绿裙）与笔触密度，不属于主体被替换。
- GPT 首图/重绘目视（`candidates-gpt_first_pass.jpg`、`candidates-gpt_repainted.jpg`）：多数候选出现参考角色的白色蕾丝短裙、绑带过膝袜、蝴蝶结头饰与抬手的构图，与门禁判定一致，**不建议用于晋升画风**。

## 9. 失败、异常与未完成项

1. **主比较集的数值压缩**（见 §5.1）：36 张图一次评分时模型只按通道给分，gemini_direct 出现 7 个并列；`best_id` 是 ID 排序产物。已用补充比较集补足可区分排名，并明确标记两集合分数不可混排。
2. **本地 runner 的一次自身缺陷**：第一次 `compare` 在付费调用成功、状态 JSON 已写盘之后，于写汇总步骤抛 `UnboundLocalError`（`write_comparison_report` 未在公共路径导入）。已修复并重跑；重跑命中 worker 的评分缓存（`cached=true`），**没有重复付费评分**，也没有丢失任何结果。
3. **局部合并步骤自报置信度 0.5**（`round-6-local-merge`）：仍产出了合并结果并被终审采用，但该步骤的把握偏低，记录在案。
4. **迭代记录标签重写**：worker 把局部细化/终审记录的 `round` 写成 `round-{本次新增轮数}-*`，在续跑中会与基线的同名记录混淆，因此 runner 把本次新增的 17 条记录改标为 `round-6-local-*` / `round-6-after`，映射全部记在 `resume_provenance.relabeled_iterations`（无内容改动）。
5. **界面进度文案口径**：worker 打印「Round 4/3」这类文字（分母是本次新增轮数），属于既有实现的口径，不影响数据。
6. **成本未核验**：本次没有核对账单金额。
7. **未做**：未写 `conf/config-styles.json`、未复制画风参考图、未发布图片、未做 CUDA/CPU 版深度指标对照（按要求不自动转 CUDA）、未重跑第 1–3 轮、未更换数据集或重新筛图。

## 10. 产物路径

输出根目录：`data/20261004/style-extraction/taya_oco/resume-20261004-213600/`

| 用途 | 路径 |
|---|---|
| 续跑训练状态 JSON（保留旧 32 步 + 新增 32 步；含 `automatic_comparison` 与 `deep_feature_comparison`） | `taya_oco_style_iter_result.json` |
| 续跑前的完整种子状态（深拷贝，含 `resume_origin`） | `resume-seed-state.json` |
| 续跑清单（基线 SHA-256、源图、轮次、提示词哈希、模板哈希） | `resume-manifest.json` |
| 续跑摘要（步骤数、各 stage、通道状态、重标记录） | `resume-summary.json` |
| 逐步骤运行日志 | `resume-run.log` |
| 旧终审基线快照 | `baseline-snapshots/final-test-images-baseline.json` |
| 旧 v1 视觉比较快照 | `baseline-snapshots/automatic-comparison-v1-baseline.json` |
| 主 v2 视觉比较（24 候选） | `automatic-comparison.json`、`automatic-comparison.html`、`comparison-summary.json` |
| 补充 v2 视觉比较（8 个 gemini_direct 候选） | `supplementary-comparisons/automatic-comparison.json`、`…/automatic-comparison.html`、`…/gemini-direct-only-summary.json`、`…/gemini-direct-only.json` |
| 深度指标（NPU fp16） | `deep-feature-comparison-npu-fp16.json`、`deep-feature-comparison.html`、`deep-feature-comparison.log` |
| 待审查 style_entry 候选（9 个：新终审 / 新增 3 轮 / 旧终审 / 旧 3 轮 / 基线原包） | `style-entry-candidates.json` |
| 新增测试图 | `test-generations/round-04|round-05|round-06|final/` |
| 本次局部裁剪区域（60 张） | `local_crops_20261004_214121/` |
| 人工对照图（候选接触表、面部对照、源图接触表） | `candidates-gemini_direct.jpg`、`candidates-gpt_first_pass.jpg`、`candidates-gpt_repainted.jpg`、`eligible-gemini-direct.jpg`、`face-upper-gemini_direct.jpg`、`dataset-references.jpg` |
| 运行脚本快照（仅复算用） | [taya_oco_resume_runner.py](tools/taya_oco_resume_runner.py) |
| 原始基线（未改动，SHA-256 与执行前一致） | `data/20261004/taya_oco_style_iter_result.json` |

`style-entry-candidates.json` 里每个条目都带 `style_entry`（`prompt` / `prompt_gpt` / `repaint_clauses` / 可选母题）、对应候选 ID、v2 分项与深度指标，并显式标记 `applied_to_style_config: false`。

## 11. 验证

- `python -m pytest -q -p no:cacheprovider tests/test_style_analyzer_prompt_pack.py tests/test_style_analyzer_layout.py tests/test_style_comparison.py tests/test_style_deep_comparison.py tests/test_style_dataset_manifest.py tests/test_style_metrics.py` → **70 passed, 5 skipped**（慢测按部署设置跳过）。
- 本次没有修改 `modules/`、`utils/`、`tools/` 或 `prompts/` 下的任何业务文件，因此不需要重启 App 才能读这些结果；若要在界面里打开新结果 JSON，直接「打开训练结果」即可。
- 源 JSON 执行前后 SHA-256 相同，未覆盖原记录、原候选、原画风配置。
