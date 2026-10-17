# 2026-10-16 本机研究记录

## 环境与连接

Krita 5.3.3；本机 MCP Bridge 1.0.0。通过用户已开启的桥接访问 localhost，令牌只在内存中读取，不写入报告。

运行桥接的 Python 为 Krita 内置 3.13.5；外部 CLI 为系统 Python 3.10。两者不是同一套 Qt：Krita 内部使用 PyQt5，本项目日常 Python 使用 PyQt6。不要把外部 Qt 对象传入 Krita 进程。

## 发现：可调用原生笔刷

旧桥接 `draw` 的形状绘制使用 QPainter 写像素，本身不等于原生笔刷。通过 `run_python` 可以调用当前 Node 的 `paintLine`、`paintPath`、`paintEllipse`，并由 View 切换真实 brush preset。

来源：[KDE Node API](https://api.kde.org/legacy/krita/html/classNode.html)、[KDE View API](https://api.kde.org/legacy/krita/html/classView.html)。

### 本机实际版本差异

网上 API 文档的 `paintLine` 使用 QPointF；本机绑定实际要求 **QPoint**。第一次试样收到 TypeError，没有成功完成输出。修正为整数点后，八种笔刷实际完成绘制。`paintEllipse` 同样使用 QRect；`paintPath` 使用 QPainterPath。

不要只按网页签名推断本机运行结果。本轮错误与修正没有导致既有作品被覆盖。

### 压感采样

QPainterPath 的逐段采样配合 `paintLine` 两端压力可以收尖。18 段试样在长曲线上可见转折；第二版的收尖笔触改为 72 段。普通细发丝用原生 `paintPath`，避免所有笔画都切成短段。这是本轮的近似压力控制，不是真实数位笔输入记录。

## 八种笔刷试样

![原生笔刷比较](../../data/test-result/20261016/krita-anime-study/brush-sampler/brush-sampler.png)

真实调用：Gpen、Basic-5、Airbrush、水彩颗粒、Glaze、Blender Blur、Texture Hair、Sparkles。试样中 Gpen 与 Basic-5 的结果接近，喷枪软边和颗粒／纹理差异明显。Blender 在一条已画好的 Basic-5 色带上测试，避免空白层上无法混色的问题。

这些是当前曲线、颜色和尺寸下的结果，不代表各笔刷全部性能。

## 第二版设计

沿用第一版自绘 SVG 的基础构图与色块，重新组织为材质组，并增加原生笔刷。不是重新生成参考图，也没有将参考图片贴入画布。

- 椅子在后发上方，保持座沿、扶手可辨认。
- 调整腿部上段曲线，尝试更清楚的膝部转折。
- 重画五官细节，减少粗轮廓，加入刘海投影和腮红。
- 头发分别绘制软阴影、内部细束和断续高光。
- 衣装硬阴影与喷枪过渡分层；裙摆追加细褶与小面积刺绣。
- 原生笔刷计划 186 条路径／直线；每条路径可能含多段压力绘制，因此不是“186 次 API 调用”。最终执行清单见 render-record。
- Gaussian Blur：腮红层水平／垂直半径 8 像素；可选柔光层半径 34 像素。以上为 1800×2700 图像像素，不是场景坐标。
- 保存柔光与清晰两个版本，默认最终输出采用清晰版。

## 证据与验收

真实 Krita 绘制成功，记录耗时约 69.2 秒，186 条笔触定义全部执行，成图实际使用五类预设。`.kra` 检查共 41 个层节点：7 个组、34 个绘画层（包含 Krita 初始空白层）。不同数量口径不可混用。

输出 PNG 均完成解码，尺寸为 1800×2700；笔刷试样为 1000×1100。图像 hash 保存在 `verification.json`。柔光与清晰图的 hash 不同，证明开关确实影响输出；这不证明柔光版更好。

### 目视复核与修正

1. 初版刘海投影形成过硬的箭头形，改成较淡的窄投影。
2. 试图用更复杂的腿部曲线强调膝盖，实际产生不自然的内凹折角，撤回该轮廓改动，保留膝部的明暗处理。这是本轮失败的造型试验。
3. 原生喷枪沿路径绘制在部分起笔位置显得方，针对独立的头发软阴影层使用半径 22 的 Gaussian Blur，并降至 70% 不透明度；细发丝层降至 65%，保留底图清晰轮廓。
4. 丝袜软阴影层再做半径 12 的 Gaussian Blur，均为实际图像像素。
5. 复核后另存 `reviewed/azure-study-v2-refined.kra` 和 PNG，没有用最终修正版覆盖第一版或本轮原始实验版。

![复核后第二版](../../data/test-result/20261016/krita-anime-study/reviewed/azure-study-v2-refined.png)

相比第一版，皮肤与布料明暗、细线和裙摆装饰更丰富，头发有了局部分束。仍存在明显限制：整体偏程序化平涂，头发大块形状偏硬，长腿与正面构图仍使坐姿体积较弱，交叠手指较简略；参考图的细腻人物造型、柔和材质和自然发束尚未达到。没有用户成品验收，也没有“换了笔刷就达到参考精度”的结论。

### 下一轮应优先改什么

优先重画低细节的三分之四坐姿结构稿，把骨盆压在座面上、将大腿方向和膝盖位置画清楚；再改头发的主要束形，最后才做高光与装饰。不要继续只堆滤镜或头发丝来回避结构问题。

### 代码验证

复用 CLI 与场景快照通过 Python 编译；真实 Krita 已验证原生笔刷、7 个剪贴材质组、Multiply／Screen、Gaussian Blur、指定文档修正、KRA 保存及 PNG 导出。不修改应用运行配置，不调用收费图片服务，不重启 app 或 Krita，不提交 Git。
