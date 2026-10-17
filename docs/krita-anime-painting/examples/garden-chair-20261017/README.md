# 花园小憩：新画布完整练习

2026-10-17；1600×2200。通过同一 Krita 文档从空白逐工序绘制，期间实际读图后修过草稿腿势/椅腿、虹膜裁切、硬明暗及腮红可见性。没有导入旧成图或参考像素。

- [最终 PNG](garden-pause.png) / [可编辑分层 KRA](garden-pause.kra)，49 个节点，保留隐藏草稿/线稿。
- [面部](head.png) / [手与躯干](hands.png) / [镜像](mirror.png) / [灰度](gray.png)。
- [工序对照](stage-board.png)；`stages/` 中保留实际阶段和失败版 PNG，`recipes/` 中是实际阶段 JSON。
- [完整独立 scene](final-scene.json) / [绘制前要素及返修说明](elements.json) / [技术验证](validation.json)。
- [逐工序读图记录](../../learning-records/whole/20261017-garden-chair.md) / [用户指定的参考](../../references/puracotte-149567189/README.md)。

![完整练习](garden-pause.png)

目前可见改善是整体侧倾、软腮红、虹膜反射和不等宽发束。全身仍偏图解，椅子虽补出扶手和坐面，坐姿缩短/承重还不够自然；手指分组和衣褶仍简化。此图是本轮练习成稿，不是达到参考完成度的证明。

复算先读 [环境探索](../../ENVIRONMENT.md)。场景依赖本机桥接协议和 `c4) Thin Brush Regular - deevad 25.01`；准确来源、下载和许可见 [资源安装记录](../../resources/deevad-install-20261017.md)。只有腮红短线用了该原生笔刷，其余主要为程序规划的曲线/遮罩在 Krita 分层栅格化；局部使用原生高斯柔化，半径为实际栅格像素，不能把 scene 的 scale 当滤镜半径缩放规则。

示例包含一份最终 KRA 和所有本轮阶段 PNG/JSON，其他阶段 KRA 仍留本地。`final-scene.json` 已在本机独立新文档从零重放，导出 PNG 与逐工序最终稿字节 SHA-256 一致；这是本机同环境验证，不保证另一台机器像素一致，仍需先验证工具兼容。
