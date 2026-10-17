# 面部与材质精修小样：能力边界和实际对照

用户指出轮廓尚可但精细程度不足。本次针对同一作品先做面部小样，未重画整幅或增加发丝数量。

实际分三步执行，每步在 Krita 导出并查看局部后再继续：

1. 肤色：脸部冷暖渐变、刘海投影、4 个 Airbrush Soft 原生低不透明度笔触。
2. 眼睛：替换原有两个虹膜层，增加上眼睑投影、虹膜上下层次、不规则短曲纹、小高光与下缘反光；保留原有杏眼位置与朝向。
3. 头发：增加少量发束大块阴影与分段高光，限制在前发束实际形状内。

结果：眼部层次与部分发束明暗更清楚；肤色改变在缩略图上较弱。整幅仍存在矢量路径感、头发内部规律重复、布料和手部体积不足。这个小样没有证明当前程序绘制方式可以达到参考图的精细完成度。

后续重点应是面部体积、发束分组、材质差异、边缘软硬与有归属的局部细节；不能用路径数量作为画质指标。每个局部还需同时检查放大图和缩略图。

- [前后面部对照](../../data/test-result/20261016/krita-seated-practice-v4/face-refinement/before-after-face.png)
- [精修 KRA](../../data/test-result/20261016/krita-seated-practice-v4/face-refinement/azure-face-refined.kra)
- [精修 PNG](../../data/test-result/20261016/krita-seated-practice-v4/face-refinement/azure-face-refined.png)
- [构造脚本](tools/refine_seated_face.py)

实际 append → revise → append 成功，原稿另存保留；脚本编译通过。对照拼图仅用于检查，由已导出的两个局部图拼接，最终作品仍由 Krita 输出。没有付费生图、安装资源或提交 Git。
