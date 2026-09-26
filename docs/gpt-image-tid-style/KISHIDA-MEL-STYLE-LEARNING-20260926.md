# Kishida Mel 画风学习（2026-09-26）

- 数据集：`D:\code\image-maker\data\style-learning\kishida-mel-20260926`，12 张选图，参考图 `01-crystal-action.png`。
- 训练：5 轮；每轮固定同一测试主体，同时生成 Gemini 直接图、GPT-image-2 首图、Gemini 完整画风图重绘。
- 结果：Gemini 直接图的内容保持最好；GPT 首图已受参考人物影响；完整参考重绘会进一步放大参考角色、服装、姿势与场景泄露。
- 限制：GPT 服务器响应记录为 `quality: low`。自动线条指标不等同于身份与内容一致性。
- 应用：最终画风已安装为 `kishida-mel-style`；GPT 短版经修正后的渲染词校验通过。
- 对比页：`D:\code\image-maker\docs\gpt-image-tid-style\kishida-mel-5round-20260926.html`。

五轮均值显示完整参考重绘确实修复了 GPT 首图的一部分碎线：平均连通段从 92.1 提高到 131.8 px，碎线占比从 0.012 降到 0.005，端点密度从 26.6 降到 16.3；但 Gemini 直接图仍更好（197.7 px / 0.003 / 9.8）。重绘饱和度由 63.4 提高到 83.5，同时伴随严重内容泄露，因此线条改善不能单独作为采用依据。

## 参考图修订

首次保存的 `01-crystal-action.png` 是高对比战斗场景异常样本，不适合作为 Kishida Mel 日常人物生成的唯一参考。配置又缺少 `prompt_compressed`，使“参考优先”模式用启发式方式压缩长说明，删掉了位于正文中段的五官、眼睛和发束规则。实测结果退回成普通干净动画风。

现已改用 `10-blue-rose-dress.png`，并固化专门的参考优先短说明，集中描述彩色细线、圆润眼型、带状发束、粉彩水彩、服装结构和故事书式背景。使用同一楼梯人物内容复跑后，结果从通用赛璐璐转为明显的水彩纸纹、暖色细线、柔和五官和装饰性背景。对比页见 `kishida-mel-reference-fix-20260926.html`。
