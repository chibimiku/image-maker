# 头发专项第二步：大形、软硬边缘与高光

沿用头发专项的流程：草稿→大形→明暗→高光细丝→识图修改。在已检查过的面部上重画头发，没有另画全身，也没有改人物表情。

## 实际观察与修改

- 上一张结构练习的亮带像头箍，正式头像的均匀细丝像塑料纤维。新稿先连接头顶、刘海和侧发的包络。
- 第一遍大形阴影仍有过硬切面；增加细线后还有纸片感，不能把这一步称作上色完成。
- 修正时去掉部分硬切面，宽阔色调放到独立层，使用 Krita Gaussian blur：horizRadius=16、vertRadius=16，再保留少数清晰的窄转折和短亮线。滤镜只作用于头发色调层。
- 使用 Basic-5 Size Opacity 画三条外围碎发；内部细丝主要仍由曲线规划，不称自由手绘。
- 复用头像配方时发现旧睫毛版本，已恢复上一轮审阅过的尖细睫毛。这是恢复已有结果，不是表情试验。

## 复看与下一步

顶部不再有独立帽盖和整圈亮带，但刘海仍偏厚，细丝分布仍人工化，后发切面偏直。柔化不能代替发束造型；头饰附近细丝的遮挡还需整理。当前是阶段练习，未达到参考完成度。

下一步仍按头发策略推进：根部转折与主发束厚薄→头饰遮挡→侧发与耳朵前后关系→发尾，不靠平均增加线数掩盖大形。表情学习另见 [独立 Topic](../expressions/README.md)，未回写本稿。

## 产物

- [要素 JSON](../../../data/test-result/20261016/krita-hair-material-study/elements.json)
- [草稿](../../../data/test-result/20261016/krita-hair-material-study/painting/01-hair-construction.png) / [大色块](../../../data/test-result/20261016/krita-hair-material-study/painting/02-hair-masses.png)
- [第一遍材质](../../../data/test-result/20261016/krita-hair-material-study/painting/azure-hair-material.png)
- [当前修正版 PNG](../../../data/test-result/20261016/krita-hair-material-study/painting/azure-hair-final.png) / [分层 KRA](../../../data/test-result/20261016/krita-hair-material-study/painting/azure-hair-final.kra)
- [造型脚本](../tools/author_hair_material.py) / [修正脚本](../tools/review_hair_material.py)

文件名 final 仅指本轮保存阶段，不代表人物或头发已完成。
