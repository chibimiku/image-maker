# Sakurapion 画风学习（2026-09-26）

- 数据集：`D:\code\image-maker\data\style-learning\sakurapion-20260926`，12 张选图，参考图 `02-tropical-table.jpg`。
- 训练：5 轮；每轮固定同一测试主体，同时生成 Gemini 直接图、GPT-image-2 首图、Gemini 完整画风图重绘。
- 结果：Gemini 直接图的内容保持最好；GPT 首图已受参考人物影响；完整参考重绘会进一步放大参考角色、服装、姿势与场景泄露。
- 限制：GPT 服务器响应记录为 `quality: low`。自动线条指标不等同于身份与内容一致性。
- 应用：最终画风已安装为 `sakurapion-style`；GPT 短版经修正后的渲染词校验通过。
- 对比页：`D:\code\image-maker\docs\gpt-image-tid-style\sakurapion-5round-20260926.html`。

五轮均值同样证明完整参考重绘能修复一部分 GPT 碎线：平均连通段从 100.5 提高到 142.7 px，碎线占比从 0.014 降到 0.006，端点密度从 25.9 降到 14.2；Gemini 直接图仍达到 216.2 px / 0.004 / 10.2。重绘平均亮度升到 172.8，并且四轮近似复刻参考人物与热带道具，所以自动指标给出的线条改善不代表结果可用。
