# 给 DeepSeek：taya_oco 在最新已有结果上新增 3 轮识别

以下整段可直接作为任务提示词。此文件记录续跑要求，不表示任务已经执行。

```text
请在 D:\code\image-maker 中执行 taya_oco 画风分析续跑，在已经完成的最新结果基础上再新增 3 轮识别与对账精修。不要只给方案，完成分析、测试、比较及报告。

先完整阅读 PROJECT_REQUIREMENTS.md、AGENTS.md、docs/style-analyzer-workflow.md 和 docs/style-extraction/deep-metrics-integration-20261004.md。仅使用系统 Python C:\Program Files\Python310\python.exe；复用现有业务模块，不创建虚拟环境，不重新安装已部署模型，不把一次性脚本放根目录，不打印密钥。

已核查的基线：
- 结果：D:\code\image-maker\data\20261004\taya_oco_style_iter_result.json
- 该记录 updated_at 为 2026-10-04 19:09:18，已完成第 1–3 轮；每轮检查 4 张，源图 12 张。
- 源图来自该 JSON 的 dataset.images（当前数据集为 data\style-datasets\taya_oco-9273d2a46328\images），不要换成另一个同名数据集或重新筛图。
- 固定主体：a girl sitting beside lake, she has pink hair and brown eyes, spring,
- 画风测试参考图：D:\code\image-maker\data\style-datasets\taya_oco-9273d2a46328\images\145818801_p0.png
- 既有视觉比较为 v1，best_id=final/gemini_direct/0；这是历史判断，不是要求你强制保留最佳或认为最新一轮必然最佳。

执行前：搜索是否存在比上述结果更新的 taya_oco 续跑记录，检查实际 iterations、final_art_style_prompts、resume_origin、dataset 与测试图存在性。若有更新且确实是同一画风的有效续跑，使用它作为基线并在报告说明。记录所选源 JSON 的绝对路径、SHA-256、源图名单、已完成轮次和基线提示词哈希。不能按目录更新时间误选报告或其他画风。

续跑方式：
1. 使用完整 existing_state 续跑，恢复最新的最终分析母版。当前基线应新增第 4、5、6 轮；参数 total_rounds=3 表示新增 3 轮，不要设置成 6。沿用 images_per_round=4。核对 _get_current_prompts(iterations) 恢复的正文与 final_art_style_prompts 一致；若记录有缺损，明确修复或报告，不能悄悄从零提取。
2. 输出到独立且不重名的 data/<执行日期>/style-extraction/taya_oco/resume-<时间戳>/，使用绝对 output_dir，file_prefix 保持 taya_oco。深拷贝旧状态，记录续跑来源，保留旧 iterations 和旧轮次 test_images；原结果、原候选、原画风配置都不覆盖。旧终审测试及其 Prompt 包单独保留为基线候选，新终审写 final_test_images。
3. 复用 StyleIterativeWorkerThread 及既有提示词模板：每轮全图共性对账、4 张单图差异检查；3 轮完成后执行局部裁剪细化、全量终审及多用途 Prompt 包。已有分析基线、原图、裁剪及成功产物按适用条件复用，不重跑第 1–3 轮的收费请求。
4. 每个新增轮次及新增终审版本均测试 Gemini 直出、GPT 首图、GPT→Gemini 重绘，沿用同一主体、参考图与比例。使用已有配置及环境密钥解析入口，不改模型/端点配置。每张测试图绑定实际当时使用的母版和派生 Prompt 包；任何失败如实标 partial/failed/not_run，不能用旧图冒充新测试，也不能把没测试的版本选为最佳。已有成功测试不重复收费。
5. 比较旧轮次、旧终审和新增轮次/新终审的全部可用候选。把旧终审候选以明确的独立 stage 纳入比较，不让新的 final 覆盖它。使用同一套 v2 十二维视觉量表重评，不能直接混排旧 v1 与新 v2 分数；同时保留主体保持、参考内容抄入、严重结构缺陷和不确定门禁。重点检查粉发、棕眼、湖边坐姿、春季主体是否被参考人物颜色/服装/构图替换。
6. 对同一个比较集合计算 Gram、AdaIN、LPIPS、CSD 本地深度指标，优先 Intel NPU，记录实际 EXECUTION_DEVICES 和 FP16。希望保证不占用笔记本 CUDA 显存时直接使用 --device npu；NPU 不可用或失败则记录原因，本次不要自动转 CUDA。CPU 后备可以另建独立结果，必须标明设备/精度，不能混算。CLI 复用 tools/style_metrics_verify.py --comparison-state <比较集合JSON> --device npu。四项对全部 12 张源图逐张等权比较，保存完整配对、均值/中位数/范围、覆盖率、权重哈希、预处理和同图自检；缺失不置零，不转换成百分比，不伪造 LoRA training loss，不把深度指标机械混入视觉总分。
7. 全部联网请求完成之后再做推荐。给出最佳提示词版本及具体生图通道，并解释视觉依据、主体门禁和四项深度指标是否一致；若更像画风的候选却抄入参考角色，指出取舍。不要默认最后一轮最佳；若没有候选通过门禁，明确说明，不强选。仅生成供用户审查的候选 style_entry，不覆盖 app 已安装的 taya_oco，也不发布图片。

产出：独立目录中的训练状态 JSON（保留旧轮次+新增3轮）、旧终审基线快照、新增每轮及终审的三路实际图像/状态、v2 视觉比较 JSON/自包含 HTML、本地深度指标 JSON/自包含 HTML、待审查 style_entry，以及 docs/style-extraction/taya-oco-resume-3rounds-<执行日期>.md。总结列出原基线→新运行、实际新增轮次、各通道成功/失败数、参数、最佳/备选依据、未完成项和全部路径。报告只能依据真实结果，不编造成功或文件。日志不要包含客户端凭据。
```
