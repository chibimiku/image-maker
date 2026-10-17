# Krita 二次元绘画学习与本机实作

更新：2026-10-17。目标：通过本机 Krita 绘制蓝发少女坐椅插画，逐步积累可复用的绘画与自动化经验。

- [整体练习计划与执行队列](PRACTICE-PLAN.md)：七阶段目标、检查标准和自主执行规则。
- [按计划首批执行](head-parts/planned-batch-20261017.md)：结构修正、虹膜遮挡修复、五官冷暖整合及实际复算验证。

- [插画完成度差距与发束层级](hairstyles/detail-gap-and-hierarchy.md)：对照用户参考，记录平面感、长条感、细丝平均分布的原因，以及未通过的渲染实验。

- [独立 Topic：发型设计与分片](hairstyles/README.md)：先定发型与前后分区，逐片轮廓检查后统一上色打光；包含参考拆解、JSON 模板与新练习。

- [头发第三轮画质研究](hair-structure/quality-pass-20261017.md)：缩浅刘海缺口，整理分束、发饰遮挡和颈部投影，保存原生罩染笔刷实验与局限。

- [独立 Topic：动漫人物表情](expressions/README.md)：单独资料、要素 JSON 与六格练习，不改头发主稿。
- [头发材质第二轮](hair-structure/material-practice.md)：连贯大形、软硬边缘、高光与实际自检。

## 阅读入口

- [绘画流程与取舍](painting-notes.md)：从结构到线稿、平涂、硬阴影、柔和过渡与收尾。
- [资源清单](resources.md)：作者原站、许可、用途、下载与安装状态。
- [本机实验记录](experiments-20261016.md)：笔刷试样、第二版、遇到的问题和验证边界。
- [眼睛与密集发丝](eyes-and-hair.md)：第三轮新增学习资料与新空白画布的细化方案。
- [第三轮实际结果](experiments-20261016-eyes-hair.md)：从新空白文件开始、细节特写、遮罩问题及修正记录。
- [五官分项研究与坐标草稿](facial-features/README.md)：目标对比、逐项画法、模板、分阶段练习与透视验证。
- [头部分项识图与连续练习](head-parts/README.md)：眉、鼻、嘴、耳、脸部体积、发型及连接问题的逐项复盘。
- [动漫头发结构专项](hair-structure/README.md)：按用户建议重新学习头骨、发根、大形与光影，再进行独立练习。
- [场景配方快照](tools/author_scene.py)：仅供研究复算，不接入 image-maker 的生产链。

## 产物

研究输出统一位于 `data/test-result/20261016/krita-anime-study/`：

- `inventory.json`：实际安装的笔刷、滤镜参数及原生 API 探针。
- `brush-sampler/brush-sampler.png` 与 `.kra`：八种原生笔刷的比较。
- `azure-study-v2.scene.json`：第二版全部图层、曲线、笔刷与滤镜参数。
- `painting/azure-study-v2.png` 与 `.kra`：第二版输出。
- `painting/03-with-bloom.png` / `04-clean-render.png`：同一绘制结果的柔光与清晰版。
- `painting/render-record.json`：实际执行的图层、笔刷和滤镜清单。
- **最终查看版**：`reviewed/azure-study-v2-refined.png` / `.kra`，修正目视发现的问题，原实验版另存保留。
- `review-refinement.scene.json`：针对实际文档的局部修正；`verification.json`：PNG 解码、尺寸和 SHA-256。

## 可复用入口

```powershell
python tools/krita_studio.py status
python tools/krita_studio.py inventory --record data/test-result/20261016/krita-anime-study/inventory.json
python docs/krita-anime-painting/tools/author_scene.py
python tools/krita_studio.py render --scene data/test-result/20261016/krita-anime-study/azure-study-v2.scene.json --output-dir data/test-result/20261016/krita-anime-study/painting
python tools/krita_studio.py revise --scene data/test-result/20261016/krita-anime-study/review-refinement.scene.json --output-dir data/test-result/20261016/krita-anime-study/reviewed
```

`tools/krita_studio.py` 是参数化本机桥接入口，不含人物造型或特定机器的绝对路径。绘画内容独立保存为 JSON；渲染发生在运行中的 Krita 内。每次 render 创建新文档，重复实验请使用新的输出目录保存各轮证据。

`revise` 要求活动文档与场景内 `expected_document` 完全匹配，且尺寸相同；同名目标层必须唯一。它修改指定图层并另存为新文件，既有图层的未指定属性保留。重新复算修正版应先 render 原实验版，再 revise；已另存的修正版不满足原始 expected_document，这是预期的误操作防护。

## 学习边界

2026-10-17 追加：[教程驱动的头颈体积练习](head-parts/tutorial-volume-20261017.md)，记录三篇作者教程、实际绘制工序、原尺寸自检与返修。

查阅教程能改善步骤与工具选择，不能直接替代人物结构和造型能力。当前是程序规划轮廓加原生笔刷的研究；不是训练出一个绘画模型，也不是人用数位笔的自由手绘。第二版沿用第一版自绘的基础构图，重点验证材质分组、细线、原生喷枪和局部滤镜，不能把细节变多等同于达到用户参考图的水平。

第三轮按用户要求重新构图并新建空白文档，不再沿用前两版造型。最新交付为 `data/test-result/20261016/krita-anime-study-v3/final/azure-silken-light.kra` 与 PNG；原版本及失败对照保留。
