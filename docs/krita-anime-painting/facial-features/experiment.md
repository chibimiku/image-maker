# 本轮执行：先元素，后草稿，再逐项绘制

日期：2026-10-16。用户先要求拆分参考五官、准备 JSON 和方向，再追加坐标草稿、工序顺序及来自系统美术教学网站的透视验证要求。本轮均保留在同一任务范围内执行。

## 具体准备与执行

先检查原图尺寸、导出面部及五官检查裁剪；再查眉、眼、鼻、嘴、耳、脸型与方向资料。参考对比见 [reference-comparison.md](reference-comparison.md)。

先写 [facial-study.template.json](../templates/facial-study.template.json)，再生成关键点数值和草稿；绘制后才添加的报告与模板本身有明确区别。最终准备记录的模板 SHA-256：

`66d53b4113ff20b88a8f085a0a362b6a80275bd15a9e016607e404aeeda73178`

用户强调透视后，采用 [透视验证要求](perspective-verification.md) 的最新版本。准确输入位于 `data/test-result/20261016/krita-facial-study/perspective-input/`，最新产物位于 `perspective-final/`。

## 实际保留的顺序

1. `00-new-blank.kra`：先保存初始新空白画布。
2. `01-coordinate-guides.png`：中线与眉、眼、鼻、口、下巴坐标。
3. `01b-perspective-structure.png`：包围盒、原边方向延长线、侧面圆投影。
4. `02-construction-sketch.png`：Pencil-2 参与的结构草稿、眼框与虹膜占位。
5. `03-clean-contours.png`：独立 G-pen 清理轮廓；草稿隐藏。
6. `04-skin-flats.png`：线稿下方的底色层显示，检查线隐藏。
7. `05-brows.png`、`06-eyes.png`、`07-nose.png`、`08-mouth.png`、`09-ears.png`：分别绘制，不跳过中间阶段。
8. `10-hair-context.png`：仅提供开放前额的发框上下文。
9. `11-soft-light.png`：暖皮肤与冷蓝发色的轻柔协调。
10. `12-complete-study.png`：三个朝向加五项独立部位练习卡。

## 本轮透视检查的结果

包围盒 12 条边按轴分组，有限 VP 的最大直线距离残差约 `9.3e-12` 场景单位，无穷远方向的平行检查残差为零。它仅说明本程序生成的盒子投影自洽，**不证明人物透视或造型已完成美术验收**。

同参数下的左右眼投影宽度：

| 朝向 | 画面左眼 | 画面右眼 |
|---|---:|---:|
| 正面 | 97.90 | 97.90 |
| 向观看者右转 | 106.69 | 69.80 |
| 向观看者左转 | 69.99 | 106.67 |

单位是场景坐标；最终像素乘 2。数字是构造结果，不是参考图的测量结果。鼻与嘴位置分别采用额外前突深度，相机、yaw/pitch/roll 与五官坐标统一。

## 目视复核与修正

最初清理脖子时误用了闭合色形的轮廓，造成下巴处矩形横线；后改为开放侧线，去掉横线。眼睛开合度略增、虹膜上部稍暗，让省略画法保持可读性。

最新透视版检查了包围盒草稿和完整练习板。优点是五官位置、朝向、线稿和颜色来自同一模板，鼻口比前一张全身图更接近参考的省略程度。仍有明显限制：练习刻意露出额头，不能当作完整参考发型；头部体块、侧面与下颌的连接仍偏程序化；耳朵远侧仅用弱化近似，未完成完整遮挡模型；双眼神态与参考并未完全一致。

## 验证与文件

KRA 与 PNG 的解码、尺寸、图层和 hash 记录在 `verification.json`。最终文件保持活动文档，草稿与检查层在 KRA 中保留但隐藏。模板原样复制为 `elements-used.json`，准确场景复制为 `scene-used.json`。

本轮新增内容不接入 image-maker 的生产生图链；没有收费图片／文本 API 调用，不改默认模型配置，不重启或提交 Git。

## 复算

```powershell
python docs/krita-anime-painting/tools/author_facial_study.py --template docs/krita-anime-painting/templates/facial-study.template.json --output-dir data/test-result/20261016/krita-facial-study/reproduction-input
python tools/krita_studio.py render --scene data/test-result/20261016/krita-facial-study/reproduction-input/facial-study.scene.json --output-dir data/test-result/20261016/krita-facial-study/reproduction
```

改 JSON 后重新生成坐标与草稿，不应只移动最终五官图层来掩盖坐标错误。
