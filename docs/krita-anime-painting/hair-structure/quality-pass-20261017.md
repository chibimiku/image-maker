# 头发第三轮：分束、遮挡与整体画质

## 新读资料与取舍

- [Laovaan：Anime Hair Tutorial for Beginners](https://www.clipstudio.net/how-to-draw/archives/171638)。读了公开正文，未观看视频。采用先定发根与外轮廓、前后头发分层、变化束宽与弯曲、再清理少数尖细边缘的顺序。长发尤其应先检查整体形状。教程的 CSP 操作不直接当作 Krita 的菜单说明。
- [Esseyli：How to Draw Hair in Manga and Semi-Realistic Styles](https://www.clipstudio.net/how-to-draw/archives/162964)。读到公开正文中的漫画构造、分层、明暗与刘海投影部分；没有声称完整观看视频或逐图观察。采用先确认轮廓再加明暗、根据共同光源安排高光、把刘海遮挡纳入皮肤投影的思路。

本项目的应用判断：现稿不是缺少线数，而是过于均匀的纤维掩盖了发束，局部模糊又保留了错误的矩形根部阴影。因此先改形状，再加有限细节。以上是本项目自己的诊断。

## 实际五工序与自检

| 工序 | 看图发现 | 实际处理 |
|---|---|---|
| 流向修正草稿 | 单片刘海过大；新增缺口又太深，像三角锯齿 | 缩浅缺口，保留少量不同长度的分束 |
| 大色块 | 需要先验证没有细线也能读出同一发型 | 连贯头皮包络内部做不同宽度的渐变发束；不另画一圈帽盖 |
| 明暗和细丝 | 旧稿根部像矩形硬块，头饰有细丝穿过 | 去掉矩形根部色块；亮区渐隐；前发使用独立蒙版 |
| 遮挡与后发复核 | 半透明头饰不能完全遮住上层细丝；后发太平，颈部投影是一条硬带 | 制作不透明形状遮挡；后发补沿流向的色调与低对比纹理；颈部投影保留位置而渐淡下边缘 |
| 原生笔刷收尾 | 大形确定后才适合增加笔触变化 | Bristles-4 Glaze 四条罩染路径与 Airbrush Soft 一条微弱反光路径，均限制在头发内部 |

每阶段实际导出后查看，再进入下一阶段。原生路径带压力渐变；主体轮廓和大多数细线仍由程序规划，不能称数位笔自由手绘。五条路径也不是画质指标。

## 最终识图评价

相较上一轮：矩形发根阴影消失、纹理网感降低、发饰遮挡更清楚、颈部硬带减轻，宝石眼依然是视觉重点。没有添加表情改动，表情 Topic 的文件未写入本轮。

不足：大刘海依然厚且偏平，宽阔渐变色块有矢量感；右侧后发像直板，根部与内部细丝仍有算法规律性；当前只检查到头像范围，不能评价画面外的发尾。原生罩染的可见改进很轻微，不能用“调用了新笔刷”代替视觉改善的证据。未达到原参考的插画完成度。

下一轮先练一束头发的圆转体积和投影，再迁移到刘海与后发；每束只保留少数清晰边缘，转折附近渐变，暗部减少纹理。发尾另开完整可见的练习范围，不能在裁切头像里宣称发尾已画好。表情仍单独学习。

## 保存与复核

- [当前 PNG](../../../data/test-result/20261017/krita-hair-quality-pass/painting/azure-hair-quality-reviewed.png) / [分层 KRA](../../../data/test-result/20261017/krita-hair-quality-pass/painting/azure-hair-quality-reviewed.kra)
- [首遍流向草稿](../../../data/test-result/20261017/krita-hair-quality-pass/painting/01-flow-review.png) / [大色块](../../../data/test-result/20261017/krita-hair-quality-pass/painting/02-lock-masses.png)
- [要素 JSON](../../../data/test-result/20261017/krita-hair-quality-pass/elements.json) / [本机文件检查](../../../data/test-result/20261017/krita-hair-quality-pass/verification.json)
- [造型配方脚本](../tools/author_hair_quality_pass.py) / [识图后修正配方](../tools/review_hair_quality_pass.py)

首遍草稿 PNG/KRA 保留了较深缺口；作者脚本和后续配方已经改为浅缺口。不要把修正后的 JSON 说成首遍草稿的逐字快照。

2400×2600 输出，六张 PNG 均完整解码，阶段 KRA 的 ZIP 结构可读；这只确认文件可用，不证明艺术质量。两个新增研究脚本编译通过。旧稿未覆盖，无生图 API 请求、无安装资源、无生产配置改动。
