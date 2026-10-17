# 第三轮：空白重画，眼睛与密集发丝

## 交付位置

最终文件位于 `data/test-result/20261016/krita-anime-study-v3/final/`：

- `azure-silken-light.kra`：2400×3600，35 个层节点，其中 31 个绘画层、4 个组。
- `azure-silken-light.png`：全图。
- `eyes-detail.png`：394×182，原像素眼睛局部。
- `face-hair-detail.png`：638×684，面部和刘海局部。
- `hair-detail.png`：418×797，长发局部。
- `00-new-blank-document.kra`：开始绘制前保存的新空白文档。
- `scene-used.json`：最终轮使用的完整绘制数据。

启动前桥接返回打开文档数 0。三轮渲染均走 `render` 新建文档，没有对旧作品执行 `revise`。最终 `.kra` 内分层保存可继续编辑。

## 研究与实际采用

来源与方法见 [眼睛与密集发丝研究](eyes-and-hair.md)。本轮重新设计了头脸、手、服装、坐姿和椅子，不读取前两版图像、场景或 SVG。保留蓝白配色、长发、坐椅、樱花主题。

实际绘制定义：15 个主要发束，1,082 条细发线／游丝／高光线，176 条虹膜内部纹理。原生 G-pen、Airbrush Soft、Stamp Sparkles 用于额外细节与明暗。基础精细曲线由 Qt 在 Krita 内渲染，不能将所有线条写作原生笔刷手绘。

眼部拆成眼白与眼睑投影、虹膜内部、睫毛与水线、反光等多层。新增瞳孔周围纹理、深浅变化、上下睑差异、眉毛细线与小组睫毛。头发分后发、头顶、刘海、肩前发束与游丝，束内细线沿同一中心走向收窄。

## 三次新画布渲染及目视纠正

1. `painting/`：首轮。目视发现发束根部截平、发顶细线穿到额头、瞳孔和白色主反光偏大。
2. `reviewed/`：改为细根逐渐展开；缩小瞳孔和主反光，修正睫毛轮廓的回程路径。仍有细线越过额头。
3. `final/`：使用显式透明度遮罩，把发顶细线限制在头发形状内，把虹膜限制在眼睑及虹膜椭圆的交集内。原图和局部重新导出、检查。

本机这一组 `QSvgRenderer` 输出没有可靠实现所用 SVG `clipPath`。不是所有 SVG 或所有 Qt 版本的普遍结论。具体修复是将遮罩单独渲染为透明 QImage，再用 QPainter `CompositionMode_DestinationIn` 相乘。复核面部局部，额头越界细线消失，虹膜边界被眼睑截断。

这一经验已合并到可复用 `tools/krita_studio.py` 的 `svg_masks`：支持多张独立遮罩取交集。不依赖重新安装 Krita 或换插件。

## 图像检查

![眼睛原尺寸特写](../../data/test-result/20261016/krita-anime-study-v3/final/eyes-detail.png)

![脸与发丝局部](../../data/test-result/20261016/krita-anime-study-v3/final/face-hair-detail.png)

全图与三个细节 PNG 均完成解码，实际尺寸和 SHA-256 见 `verification.json`；KRA ZIP、层树以及初始空白图的像素范围已检查。脚本编译与 scoped diff-check 通过。

最终图的眼部层次、发丝数量与清晰度比前两版明显增加。仍有可见限制：发束内的线条偏程序化、部分根部和发顶轮廓依然较几何；眼睛和衣装偏干净平涂，人体及折叠手指仍简化。不能将线条统计当成已达到参考图自然程度的证据。

未调用收费生成服务，未修改模型配置、重启应用或提交 Git。

## 复算

从仓库根目录运行，Krita 与本机桥接须已启动：

```powershell
python docs/krita-anime-painting/tools/author_eyes_hair_study.py
python tools/krita_studio.py render --scene data/test-result/20261016/krita-anime-study-v3/new-canvas.scene.json --output-dir data/test-result/20261016/krita-anime-study-v3/reproduction
```

每轮使用不同输出目录，避免覆盖已有证据。配方快照只用于这张原创研究作品复算，不接入生产链。
