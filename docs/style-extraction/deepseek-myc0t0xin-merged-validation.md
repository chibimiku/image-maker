# DeepSeek：myc0t0xin 按字段合并与验证

将以下内容作为 DeepSeek 的任务提示词。

在 D:\code\image-maker 工作，先读 PROJECT_REQUIREMENTS.md、AGENTS.md、docs/style-analyzer-workflow.md 和 docs/style-extraction/myc0t0xin-comparison-recovery-20261005.md。使用系统 Python C:\Program Files\Python310\python.exe，不创建新环境、不安装或重新部署模型，不修改其他正在进行的工作。

任务是验证 myc0t0xin 的跨字段组合，不再运行 5 轮分析，不把两段风格描述拼接或总结成一段。原始结果为：

`D:\code\image-maker\data\20261004\style-extraction\myc0t0xin\run-20261004-225929-156628\myc0t0xin_style_iter_result.json`

先核验源文件、12 张数据集参考图、实际字段与 SHA-256。第 2 轮 GPT 首图和第 5 轮 Gemini 直出总分并列，但来自不同字段和不同模型。第 5 轮 GPT 短版不合格，不许使用它，也不许回退到 master_prompt 充当 GPT 短版；不要把旧分数赋给新组合。

## 组合与控制变量

构造三个候选包，均调用 normalize_style_prompt_package 重新校验，并保留逐字段来源/hash：

| 包 | gemini_full_prompt → prompt | gpt_image_prompt → prompt_gpt | gemini_repaint_clauses → repaint_clauses |
|---|---|---|---|
| A：第 2 轮对照 | 第 2 轮 | 第 2 轮 | 第 2 轮 |
| B：按通道合并 | 第 5 轮 | 第 2 轮 | 第 2 轮 |
| C：重绘条款对照 | 第 5 轮 | 第 2 轮 | 第 5 轮 |

按字段精确复制，不先改写或压缩。三包的 optional_motifs 均为空，关闭可选母题，排除装饰内容注入的干扰。必要的元数据可以新增，但不能冒充来源轮次的完整原包。第 2 轮 GPT 短版必须保持原文和 hash，校验失败就停止 GPT 相关请求并报告；禁止修改 validity 布尔值绕过校验。

注意：repaint_clauses 不只作用于重绘，测试逻辑也把它们作为 extra_clauses 放进 GPT 首图。因此 B 与 C 的 GPT 首图也需要重新测试；不能认为复制 prompt_gpt 就保证请求相同。

## 实际验证

复用现有业务函数 StyleIterativeWorkerThread._generate_test_image、_test_image_record 及实际后端；若需要可重复的 CLI 能力，扩展现有同族工具，不新建一次性运行脚本。每组、每次重复使用独立绝对输出目录，原运行、原图、原报告保留。输出根目录为 data/<实际日期>/style-extraction/myc0t0xin/merged-validation-<时间戳>/。

1. 先做无联网请求预检。保存每包完整字段、逐字段来源/hash、最终生效提示词/hash、角色明确的输入图顺序与哈希、模型/尺寸/比例、附加条款、参考模式、母题设置；不保存凭据。需证明预检和实际发送使用同一个组装函数，不能只把候选包当作最终请求。
2. 使用原运行同一参考图及固定主体：`a girl sitting beside lake, she has pink hair and brown eyes, spring,`。生图模型/站点从当前配置读取，记录与旧运行是否一致；所有新对照使用相同模型和参数，不跨通道直接归因。
3. A/B/C 各测试 Gemini 直出、GPT 首图、GPT→Gemini 重绘，每路重复 2 次，第一阶段最多 18 张输出。GPT 首图仍采用原有画风图；重绘统一使用新的文字模式 reference_mode=none、clauses_without_image=true、scope=full，只发送该次 GPT 首图，附加对应候选的画风条款。不附加完整画风图、不叠加其他旧画风。新的对照不是旧完整参考图重绘的同条件复现。
4. 请求成功即保存，失败单独记录；恢复时复用成功产物，不能因评分或报告失败重复收费生图。不得把空产物标成功，未执行项明确 not_run。第一阶段失败先报告，避免不断重试加量。
5. 用全部 12 张源图、当前固定 v2 量表及四项主体门禁比较。每批最多 4 个候选，保存响应、维度证据和本地计算；复制角色、服装、书本、椅子、花框等与主体不符的候选必须排除。缺失/重复 ID 按现有有限重试流程处理，不能补造分数。报告同一包各通道及重复样本的分数、门禁、波动；不由某一次好图断言包整体最佳。
6. 对新候选独立计算 Gram、AdaIN、LPIPS、CSD，NPU 优先；不修改设备默认、不为 NPU 可用任务探测 CUDA。记录实际设备/精度、权重 hash、覆盖率与同图自检。计算失败或设备回退明确记录，指标不混入视觉总分、不换算百分比。原输入 hash 相同的成功缓存可以复用。
7. 人工看图复核，给出 B 是否比 A 更合适、C 的条款是否带来收益或身份漂移。如果第一阶段存在清晰且通过门禁的推荐，再用该推荐测试一个不同的主体：短蓝发、绿眼女孩站在室内窗边读书、秋季傍晚；仍三路各 2 次，最多另 6 张，单独比较状态文件，不能与原主体混入同一个自动比较集合。第二阶段用于发现参考内容泄露与跨内容退化，不宣称验证全部泛化。

## 交付

交付 app 可读取的训练结果 JSON（dataset、iterations/来源种子、parameters、test_images、prompt_variants、automatic_comparison、deep_feature_comparison）、各候选包、字段来源清单、脱敏请求审计、全部生成图、两份自包含报告和一份中文结论。报告必须同时展示原参考图、12 项视觉分数、主体门禁与独立深度指标，包含失败/未执行项和按通道的建议。

推荐内容必须说明 prompt、prompt_gpt、repaint_clauses 各取自哪里及为什么，并区分「字段合法」「已生成」「复核通过」「尚未覆盖的场景」。保留 optional_motifs 关闭的验证配置；不要在验证后偷偷恢复未测试的母题。

最终提供合法的 proposed-style-entry.json，ref_image 指向原始参考图，enabled 默认 false，带字段来源与验证审计路径。此次仅验证和提出方案，不自动写入正式 conf/config-styles.json、不提交或发布画风，等待用户观察报告后选用。中文总结真实执行数量、费用估计依据、失败、限制与可复现入口。不要借本任务重新安装 lpips、open_clip、clip 或改动其他实验文件。
