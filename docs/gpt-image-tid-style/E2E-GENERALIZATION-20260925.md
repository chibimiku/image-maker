# 画风泛化、身份门禁与端到端复现实验（2026-09-25）

> 后续配方已升级：完整画风图现为默认，并加入基于实际首图 prompt 的最多两轮身份修订。见 [STYLE-REF-TWO-CORRECTIONS-20260925.md](STYLE-REF-TWO-CORRECTIONS-20260925.md)。下文保留 source-only 阶段的历史实验结论。

## 结论

当前较稳的默认链路是：

1. GPT 首图使用画风 `prompt_gpt`、画风参考图和分析产物的 `gpt_image_prompt`（约 1400 字符身份完整锚）。
2. Gemini v5 保守重绘只接收 GPT 首图，`reference_mode=none`、`scope=full`、2K；不再接收画风图。
3. 身份审计比较初始分析描述与首图/重绘图。有高置信差异时只做一次定点修订；修订后即使仍有差异，也保留修订图并标记复核，不再回退 GPT 首图。

“画风图 + 去具体配色文字”不能可靠解决 Gemini 的角色泄露。它在一轮 tinkle 测试中把棕发/紫瞳改成灰蓝发/绿瞳，并把短裙改成长裙；因此 `style_neutral` 仅保留为实验模式。

## tinkle 配方定位

历史满意图的 GPT 源图确实已经有很强的 tinkle 画风。日志表明它来自旧 `/images/edits` 路径、单张画风参考图、`1024x1536 high`。历史 GPT 源图指标为亮度 137.3、饱和度 76.4、平均段长 80.8、端点密度 33.51；后续 Gemini 和本地工序是在这一较好基底上继续加工。

当前 `/images/generations` 三种内容锚的对照：

| 内容锚 | 提示词总长 | GPT 首图（亮度/饱和/段长/端点） | 观察 |
|---|---:|---|---|
| 分析全文 | 8569 | 172.5 / 60.9 / 72.8 / 37.57 | 身份完整，但比历史图更亮、更碎，画风图作用较弱 |
| 500 字符短锚 | 4643 | 148.2 / 57.2 / 80.1 / 36.83 | 画风更接近历史，但短锚漏掉发色/瞳色，首图出现紫瞳 |
| 1400 字符身份完整锚 | 5434 | 167.4 / 63.8 / 67.8 / 38.52 | 棕发、青蓝瞳、服装和构图均保留；source-only 重绘后段长 74.4、端点 37.06 |

因此默认使用身份完整锚。它仍未复现历史图的深蓝密度和全部线稿质量；差异同时来自旧 `edits`/当前 `generations` 端点与随机性，不能把历史结果归因于重绘参数 alone。

## Gemini 参考图与二次修订

在 tinkle 短锚首图上，把画风图作为 Gemini 第二张输入并附加“只读笔触/边缘、不得读配色”的文字，输出仍发生：

- 发色：暖棕 → 银灰蓝；
- 瞳色：紫色 → 绿色；
- 短层叠裙 → 长拖尾裙；
- 饱和度：57.2 → 113.6。

说明视觉参考的具体颜色和角色结构具有高于文字排除句的影响。source-only 对照保持了原发色、瞳色、构图和服装。

身份定点修订对局部差异有效：另一轮 tinkle 重绘删除了长蕾丝手套和长袜，审计两次均以 0.99 置信度识别；修订后手套和长袜恢复，脸、姿势与背景基本保持。复审仍指出领口偏低，说明修订需要再次验证。

对大范围漂移，修订不可靠。`say-hana-v4-r2` 的 source-only 重绘把动漫短裙坐姿改成古典长裙肖像，审计识别出头发、瞳色、裙型、领结、手套、头饰等 6 项重大差异；一次修订后仍有裙长和领口差异。按 2026-09-25 的新规则，流水线保留修订图并标记人工复核，不再回退首图。

## 五画风 × 两轮结果

固定参数：`gpt-image-2`、`/images/generations`、`high`、跟随素材比例；首图挂画风参考图；Gemini 使用 v5、2K、source-only、full。画风为 `tinkle`、`goto-p`、`say-hana-v4`、`noir-aiart`、`tid`，两轮轮换使用 `sucai` 中的室内白裙、屋顶蓝裙、店铺粉裙和紫发舞姬分析产物。

| 组 | 身份门禁 | 段长变化 | 端点变化 | 结果 |
|---|---|---:|---:|---|
| goto-p-r1 | accept | +10.0 | +1.91 | 线段显著变长，但端点指标变差 |
| goto-p-r2 | correct | -1.7 | +1.30 | 领口局部修订后复审通过 |
| noir-aiart-r1 | accept | 0.0 | +0.05 | 基本无变化 |
| noir-aiart-r2 | accept | -5.6 | +0.22 | 重绘线指标退步 |
| say-hana-v4-r1 | review | +1.0 | -4.20 | 线连通改善；身份审计接口拒绝该屋顶样本 |
| say-hana-v4-r2 | correct/review | +7.9 | -11.70 | 指标看似大幅改善，但角色/构图完全漂移；一次修订后仍需人工复核 |
| tid-r1 | accept | +0.5 | +2.58 | 变化很小，端点指标退步 |
| tid-r2 | review | -1.6 | -0.87 | 变化很小；身份审计接口拒绝该屋顶样本 |
| tinkle-r1 | accept | -0.8 | -0.39 | 基本无变化 |
| tinkle-r2 | review | +5.5 | +0.55 | 段长改善；身份审计接口拒绝该屋顶样本 |

Gemini 重绘并不稳定提高线条：10 组里只有部分样本段长或端点指标改善，且 `say-hana-v4-r2` 证明“线指标更好”可能来自整图重画。自动指标只能做故障筛查，必须与身份门禁一起使用。

3 个屋顶样本的文本视觉审计接口返回 HTTP 400，即使把审计图缩到最长边 1536 仍会发生。流水线现在把它记录为 `review` 而不是让成功的生图任务失败；这些样本需要人工看横向页。

## 复现与产物

- 横向离线对比：`docs/gpt-image-tid-style/e2e-generalization-20260925.html`
- 画廊清单：`docs/gpt-image-tid-style/e2e-generalization-20260925.json`
- 批次规格：`data/test-result/20260925/e2e-five-styles/spec.json`
- 每组完整请求、提示词、首图、重绘、审计和修订：`data/test-result/20260925/e2e-five-styles/runs/<style>-r<round>/`
- 指标表：`data/test-result/20260925/e2e-five-styles/metrics.json`
- 审计摘要：`data/test-result/20260925/e2e-five-styles/summary.json`

复跑命令：

```powershell
python -u tools/style_e2e_matrix.py --spec data/test-result/20260925/e2e-five-styles/spec.json --output-dir data/test-result/20260925/e2e-five-styles/runs --workers 2
```
