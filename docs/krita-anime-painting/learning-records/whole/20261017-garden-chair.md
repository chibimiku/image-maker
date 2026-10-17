# 整体插画练习：花园坐姿 / 20261017 garden-chair

状态：完整练习及逐工序识图完成。此记录在绘制前建立，以下按真实过程追加；旧稿未覆盖。

## 环境、现状与目标

本机 Krita 5.3.3 / bridge 1.0.0，已读取状态，旧活动稿已保存；通过 `tools/krita_studio.py` 的本机 MCP 桥接执行。原生 `c4) Thin Brush Regular - deevad 25.01` 已安装；使用时另记录实际层。没有付费生成、没有新增下载。

实际重新看过用户参考头部裁剪和旧《A quiet page》全图。旧稿主要问题：姿态僵直，头发等距排线、衣褶像色条，脸和整体光色关系弱。本轮不复用旧图像或旧头部 scene，重新规划完整坐椅构图：浅蓝长发、白蓝裙、头部轻侧倾、手靠脸、另一手搭裙面。

## 来源与采用范围

- 用户参考实际读图及 [头部分析](../../head-parts/puracotte-149567189-head-analysis.md)：只采用软腮红、发束大面/接缝及边缘主次的方法，作者实际图层未知。
- Proko / Stan Prokopenko，[How to Draw Gesture – Step by Step](https://www.proko.com/course-lesson/how-to-draw-gesture-step-by-step)：本轮读公开 Lesson Notes，采用动作线、胸腔/骨盆倾斜、肢体连续节奏；未看视频或付费课程。
- Proko / CELSYS，[Draw Expressive Hand Poses from Imagination](https://www.clipstudio.net/how-to-draw/archives/156141)：本轮读公开正文的手掌/腕/指分组，先手掌体块、明确各指动作，之后才加指甲；未看嵌入视频。
- Drawabox 作者站内 [透视讲评](https://drawabox.com/r/artfundamentals/post/7uvodp)，结合 [已有验证要求](../../facial-features/perspective-verification.md)：检查已经画出的边，区分同向直线与装饰曲线；不把程序构造残差当美术通过。
- 沿用已读的 Tsunoju 肤色和 Esseyli 头发文字教程，见上述参考分析，不重复声称新学视频。

## 要素和检查

本轮文件在 `data/test-result/20261017/krita-garden-chair/`，绘制前生成 `elements.json`。画布 1600×2200，逻辑坐标 800×1100。头部所有五官/刘海一起用同一局部侧倾坐标；长发尾部另随重力下垂。椅子座面、扶手和脚用同一一点透视架构，曲线装饰不强行套直线消失点。

检查顺序：整体动作和坐面接触 → 头部局部轴与手腕 → 遮挡后的线稿 → 冷暖底色 → 大明暗 → 眼/腮红/不等宽发束细化 → 全图、面部、手、缩略、镜像与灰度。明显结构错误退回前一步；记录仍未解决的图解感。

## 工序实录

| 工序 | 阶段文件 | 实际识图、问题和修改 |
|---|---|---|
| 草稿 | sketch/01-sketch.png → sketch-reviewed/01b-sketch-reviewed.png | 全图实看；初稿腿太直、椅前腿从错误的后端起笔。双膝向侧前方移动、椅前腿改从前角垂直下落；复看后继续，仍未证明正确人体缩短 |
| 轮廓 | contour/02-contour.png | 全图实看，按身体/后发/前发遮挡整理；手靠脸侧仍像三指符号，后续补分组提示，未称准确解剖 |
| 底色 | flats/03-flats.png / head | 全图与面部实看；发现虹膜越出眼裂下缘，停止细化 |
| 眼裂返修 | flats-reviewed/03b-eye-crop-reviewed-head.png | 以眼裂凸多边形与虹膜轮廓求交，避免本机 SVG clipPath 限制；复看越界已消除，再进入明暗 |
| 大明暗 | shaded-bases、garden、planes 的阶段 PNG | 实看全图与头部；背景远景、大受光面、颈投影及软腮红建立，但发束有硬尖片、裙褶像贴片 |
| 明暗减弱 | planes-softened/05b-planes-softened-head.png | 独立图层降低不透明度，Krita 原生 gaussian blur 1.5 像素，实看边缘有所缓和；形状仍硬，不能把滤镜当造型修复 |
| 细化 | details/06-detail.png / head / hands | 实看三尺度；虹膜有分层反射，局部发流与衣饰补出；腮红短线过弱、裙褶仍重、椅扶手被后发完全盖住 |
| 局部返修 | details-reviewed/06b-detail-reviewed-head.png | 原生短线由实际 .76px 提到 1.2px，独立层60%；重绘原笔触层前清空，避免重复叠加；裙褶层53%、gaussian blur 5px。复看短线可见且不压眼睛 |
| 整体收尾 | final/07-garden-pause.png | 实看；将椅前扶手、前腿和部分坐垫放在后发之前、人体/前发之后，座椅关系更明确。前景植物不遮挡脸和手 |
| 最后检查 | stage-board.png、final/mirror.png、final/gray.png | 实际查看全图、面部、手部、工序缩略、镜像及灰度；最高细节/暗对比集中眼部，整体仍偏浅且图解化 |

## 输出与经验

所有本轮实际阶段 PNG 和 JSON、最终 49 节点 KRA、PNG、局部、镜像/灰度与独立 scene 已收录 [整体练习示例](../../examples/garden-chair-20261017/README.md)。其他阶段 KRA 本机保留。程序路径和遮罩在 Krita 分层栅格化，只有腮红短线实际使用 c4 原生笔刷，共30条；不描述成全程自由手绘。

相对旧《A quiet page》，头部侧倾、虹膜反射、上颊软血色、发束主次和前后遮挡更明确。仍未解决：坐姿承重和缩短、指节/拇指体块、裙褶的布料逻辑、宽刘海带片感、背景叶形过规整。镜像后这些问题仍存在，没有写成“参考级”完成。

下一轮先做胸腔/骨盆与坐垫接触小样，明确大腿向前的缩短和膝盖体面；手掌/拇指单独练，再回整体。头发优先改中束形状与接缝过渡，不继续单纯加同类细线。

用户追加要求整理上传：范围为本知识目录的研究文档、此前未提交的面部研究、此次完整图/阶段及指定 Puracotte 参考。来源保留、无令牌/桥接配置/下载笔刷二进制；仓库其他应用代码改动不提交。本轮实际检查及分层文件技术结果见示例 validation.json；它不证明画质达标。
