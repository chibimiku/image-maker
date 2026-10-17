# 参考图五官拆解与坐标草稿流程

本轮目标是用户指定的 `data/20261015/b1b382ab_232123-d0dec0.jpg`。先观察原图，再查各部位画法，准备 JSON 元素和位置，最后在新画布按草稿、线稿、底色、五官、柔光的顺序实画。

## 分项经验

- [目标与上一版的差异](reference-comparison.md)
- [脸型与朝向](head-and-direction.md)
- [眉毛](brows.md)
- [眼睛](eyes.md)
- [鼻子](nose.md)
- [嘴](mouth.md)
- [耳朵](ears.md)
- [草稿、线稿与氛围工序](workflow.md)
- [透视验证要求](perspective-verification.md)：Proko、Love Life Drawing、Drawabox 的规则及实际检查边界。
- [本轮执行与验收](experiment.md)

## 先准备元素，再绘制

[可编辑 JSON 模板](../templates/facial-study.template.json) 是本轮的输入，不是绘制完成后补写的说明。包含：

1. 参考路径、实际尺寸、参考观察的置信边界。
2. 画布、视图区、正面及左右转头的 yaw/pitch/roll。
3. 头部曲面、脸宽、颅顶、眼线、眉线、鼻底、口线、下巴位置。
4. 眼睛开合、眼距、虹膜和瞳孔比例；眉、鼻、口、耳的尺寸与线宽。
5. 视线补偿、软光方向、冷暖配色、遮挡约定。
6. 绘制顺序、阶段导出与最终隐藏草稿的规则。

`author_facial_study.py` 消费这些数值并在绘制之前导出 `resolved-landmarks.json`。同一组位置同时生成草稿定位框、线稿与后续五官，减少互相脱节。不是用事后标记伪装成前置草稿。

## 产物位置

`data/test-result/20261016/krita-facial-study/perspective-final/` 保存最新透视版：

- 新建空白 KRA、最终可编辑 KRA。
- `01-coordinate-guides.png` 到 `12-complete-study.png` 的连续阶段。
- 正面、转头和五官小样细节导出。

准确场景输入、模板 hash、坐标表与包围盒检查位于同级 `perspective-input/`。早期浅曲面平行投影试画位于 `painting/` 和 `final/`，供比较，不当作最新透视版。
