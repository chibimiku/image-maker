# 单图分析全画风回归（2026-09-25～26）

## 结论

单图分析 GUI 已固化为配方版本 7：默认走 gpt-image 通道，首图使用 `/images/generations`，第一次 Gemini
重绘使用“GPT 首图 + 完整画风图”，随后只用当前图与 GPT 首图做质量修订，再做最多两轮 source-only
身份定点修订。质量和身份修订显式锁定当前图宽高比，不再把 `auto` 交给 Gemini。

日期根目录只发布每次任务最终选中的一张图，并保留分析投稿所需 JSON/TXT；GPT 首图、首次重绘、质量审计、
质量修订、身份审计、身份修订、实际 prompt 和请求清单放入
`data/<日期>/analysis-gpt-image/<单次任务>/`。无头 CLI 加 `--publish-final` 时遵循同一规则。

## 受控回归

- 固定素材：`data/20260921/sucai/2026-07-24 12_17_40_1.jpg`
- 固定分析产物：`data/20260921/sucai/20260921-234549-2026-07-24 12-白花絹幕の少女.json`
- 画风：配置中的 29 个非默认条目全部执行；同名不同版本均独立执行。
- 首图：`gpt-image-2`、`high`、输入比例、`/images/generations`。
- 重绘：Gemini 2K、完整画风图、`scope=full`、v5 保守固件。
- 后续：质量门禁一次，身份门禁最多两次，最终不回退。

最终 29/29 均有可用发布图；18 个触发质量修订，13 个触发身份修订，共 15 轮。最终身份复审仍标出
2 个有差异的样本：`dall-e-v3`（minor）和 `goto-p`（major），应在画廊中人工复核，不能据此宣称身份
偏离已经普遍解决。`noir-art-style` 没有画风参考图，因此只验证文本画风路径。

## 比例故障与复验

首轮发现 8 个样本在质量或身份修订后变成横向双联图：`dall-e`、`dall-e-v3`、`iris-mix-style`、
`say-hana-style`、`tid`、`noir-aiart`、`goto-p`、`tinkle`。根因是第一次完整画风图重绘已按源图锁定
`2:3`，但后续修订仍传 `aspect_ratio=auto`；Gemini 在收到“当前图 + GPT 首图”时偶发把两张输入解释为
横向比较构图。

修复后复用原来的 GPT 首图重跑这 8 个样本，只改变重绘阶段的随机采样与比例约束。所有修订请求均明确
下发 `2:3`，最终 8/8 都保持单幅竖图；整个 29 图集合也全部为竖图。`iris-mix-style` 首次补跑的完整
画风重绘遇到接口超时并生成 partial，随后单独重试成功；partial 已移回过程目录，不作为发布图。

这次修复解决的是画布与构图退化。它不会保证每次身份修订都成功：例如 `goto-p` 两轮后仍有服装颜色偏差，
`dall-e-v3` 仍有轻微身份差异。`noir-aiart` 的比例修复复验中，质量审计接口超时，因此最终采用已经成功的
完整画风图重绘；请求清单保留了该失败状态，便于后续重试。

## 产物

- 横向画廊：`docs/gpt-image-tid-style/style-sweep-20260925.html`
- 机器可读摘要：`docs/gpt-image-tid-style/style-sweep-20260925.json`
- 原始 29 画风过程：`data/20260925/analysis-gpt-image/style-sweep-20260925/`
- 8 个比例修复复验过程：`data/20260926/analysis-gpt-image/style-sweep-ratio-fix/`
- 最终落盘审计：`data/20260925/analysis-gpt-image/style-sweep-20260925/sweep-final-audit.json`

画廊按画风家族横向排列，卡片默认显示最终发布图；展开后可查看 GPT 首图、首次完整画风图重绘与最终
过程图。JSON 中记录最终图、过程目录、质量与身份门禁结果、线条指标及相对参考图指标。
