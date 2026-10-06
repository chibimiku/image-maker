# Gram v2 修订与二阶段验证交接

## 已修复

默认 Gram 版本为 `gatys-layer-sum/v2`。`G=FFᵀ/(C·H·W)` 已归一化，单层平方差只除以 4；等价于原始 Gram 平方差除以 `4 C² (HW)²`。五层等权是项目约定。[原论文](https://arxiv.org/pdf/1508.06576) 单层归一化不要求再除一次 C²。

旧版 `normalized-gram-extra-channel/v1` 可通过 `gram_distance(formula_version=GRAM_LEGACY_VERSION)` 显式复算。旧每层 contribution 乘该层 channels² 可恢复新版贡献，但不同层不能以同一个常数换算旧总分。旧报告与原始 JSON 保留并标明历史口径。

GUI 深度比较中的重复实现同步修正，版本升为 `style-deep-comparison-v2`；缓存和报告新增 `gram-v2` 文件名，避免覆盖旧版。输入尺寸、特征层、权重及 ONNX 编码器未变，App 主评分公式未改。用户要求暂不重启，现有 GUI 仍使用已加载模块；新 CLI 使用新版代码。

真实图片对改用独立只读模式：每个 `--pair` 都逐指标计算；同图自检使用第一张图与自身，不把不同图误当零距离用例，也不强制真实图片符合合成 soft/hard 排序。结果保存绝对路径、图片 SHA-256、实际设备、精度和 Gram 版本。`--no-compare-cpu` 现在真正生效；请求跨设备比较但基准不可用时不冒充通过。缺依赖/权重记 unavailable，值为空，不使用替代算法。

## 验证

- 新增 `tests/test_style_metrics_revision.py` 13 项通过；既有 `test_style_deep_comparison.py` 8 项通过。覆盖原始公式等价性、历史复算、空间样本重复不衰减、多层换算、GUI 同口径、真实配对不误判以及缺项不填零。
- 修改的 Python 文件通过 py_compile。
- 真实 VGG 权重 CPU/FP32 校准：Gram same=0、soft=0.14264702288486536、hard=61.77492081422077；AdaIN same=0、soft=37.186454224794474、hard=212.27273658961408。记录 `data/test-result/style-metrics-cpu-gram-v2-20261006.json`。这些是合成校准样本，不是色彩实验效果。
- 已有 S2-A-1/S2-B-1 原图只读配对：Gram/AdaIN 可计算，同图检查通过；当前 Python 的 lpips/clip 缺失，两个指标记 unavailable。记录 `data/test-result/style-metrics-real-pair-v2-dependency-check-20261006.json`，退出码 1 表示未全部可用，不代表有效指标失败。
- 未安装依赖、未下载权重、未调用付费 API、未修改实验原图、未重启 app。新版 CUDA/NPU 未复验，不能沿用旧版报告宣称验证通过；当前没有 pytest，未运行其他 pytest 套件。

## 下一步

先核对上一份清色 9 图＋3 评审是否已经执行，复用成功缓存，不重复付费。随后按冻结的 A 基准／B 当前清色／C 候选清色做三组平衡盲评，主判据是指定蓝红表面的去灰、鲜艳程度、明度与保护色；深度指标只辅助描述整体变化，不决定配色成功。

清色有稳定信号后再扩大到第二场景／色板；无稳定信号则保留试验选项，转做二阶段宽松主次面积关系，不无限堆叠形容词。第三阶段研究既有材质与光线、色彩知觉、画风迁移及重绘保色。情绪／季节仍由用户在 refine 阶段选用，不覆盖固定色板。禁用本地蒙版、羽化与叠加。

完整交接见 `prompts/color-knowledge/DEEPSEEK-STAGE2-CLEAR-WITH-METRICS-20261006.md`。
