# 头部分项练习与识图复盘

新增：[149567189 用户参考：朝向、腮红和浅色头发体积](puracotte-149567189-head-analysis.md)，含独立局部小样与失败记录。

最新：[面部四轮局部迭代](face-iterations-20261017.md)，每轮实际读图决定下一轮，保留鼻钩、厚唇和浓眼线失败；[下一批方案](../learning-records/face/NEXT-PLAN.md)转向下半脸比例和颧颊体块。

用户要求连续练习各部位，每个部分先用识图能力评价，再保存经验。本轮按下列顺序实际绘制、导出、查看、修改、复看；不是仅运行一遍脚本后自动记作通过。

1. [眉毛](01-brows.md)：收尖与毛流；后续高分辨率复看又打散重复短线。
2. [鼻子](02-nose.md)：第一稿仍像 C，继续修改为弱轮廓和小色面。
3. [嘴部](03-mouth.md)：减淡初稿偏重的唇色，收窄开口。
4. [耳朵](04-ears.md)：原图被挡而无法验证，先画无遮挡结构练习。
5. [脸部体积](05-face-planes.md)：补冷暖过渡，并修正亮面盖淡眼睛的问题。
6. [头发三轮修改](06-hair.md)：拒绝头箍状第一稿，修复发根及遮挡。
7. [眼周与耳饰连接](07-eye-context-and-jewelry.md)：高分辨率复看后继续修改睫毛、眉毛与耳垂挂点。
8. 用户随后要求头发先专项学习，已转入 [动漫头发结构专项](../hair-structure/README.md)。这一步没有直接把新结构小样套回人物。

## 当前保存的画布

- [头部分项修改后的高分辨率图](../../../data/test-result/20261016/krita-seated-practice-v4/head-practice/final-head-reviewed/azure-head-study.png)
- [同一头部的分层 KRA](../../../data/test-result/20261016/krita-seated-practice-v4/head-practice/final-head-reviewed/azure-head-study.kra)
- [耳部构造与简化练习](../../../data/test-result/20261016/krita-seated-practice-v4/head-practice/ear-study/ear-structure.png)
- [修改前分项识图图板](../../../data/test-result/20261016/krita-seated-practice-v4/head-practice/before-review-board.png)
- [绘制前评价记录](../../../data/test-result/20261016/krita-seated-practice-v4/head-practice/pre-drawing-assessment.json)

高分辨率头部图为同一曲线与图层配方在 Krita 新画布重新绘制，非旧 PNG 插值放大。它用于暴露重复规律与连接问题，不表示全部形体已准确。人物整体仍有程序路径感；脖子投影、头发自然度等问题仍保留在待改进列表。

## 本轮技术发现

实际像素探测发现这台 Krita 的内置 Qt SVG 渲染器未正常显示相对坐标径向渐变。桥接渲染工具将每处渐变按形状边界转换为显式画布坐标后，耳部和肤色软色块正常出现。共享渐变、路径、父级变换与原有显式坐标均完成实际回归探测，详见 [探测结果](../../../data/test-result/20261016/krita-seated-practice-v4/head-practice/gradient-probe.json)。这仅证明本机这些探针，不外推所有 SVG 实现。
