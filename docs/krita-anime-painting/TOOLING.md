# 工具、复算与迁移限制

先读 [环境探索](ENVIRONMENT.md)。这份工具说明描述特定研究实现，绘画知识可迁移，执行协议不能假定通用。

## 能直接阅读和使用的内容

Markdown、PNG 与 JSON 无需启动 Krita。`templates/practice.template.json` 是语义规划模板，不直接交给渲染器；`templates/facial-study.template.json` 是五官专项脚本输入，含本地参考相对路径，应自行替换。`examples/latest-head.scene.json` 才是附带完整图层内容的渲染场景，不读取历史画面。

`examples/` 是接收方的优先复算入口。其余 `author_*.py`、`review_*.py`、`refine_*.py`、`inspect_*.py` 保存历史研究配方，很多需要原机 `data/test-result`，且有顶层执行/写文件行为。不要批量运行或 import 它们以探查环境。源码可阅读，不意味着完整历史输入随包提供。

## 本机桥接 CLI

仓库根 `tools/krita_studio.py` 仅使用宿主 Python 标准库，通过特定 HTTP 桥接将代码送入 Krita。代码在 Krita 进程内使用其 PyQt5、SVG 与绘画 API；不要求也不应该因此给宿主 Python 盲装 PyQt5。其他 Krita/Qt 版本需要重新探测。

确认兼容接口后，在仓库/解压包根目录操作：

```text
python tools/krita_studio.py status
python tools/krita_studio.py inventory --record <新的输出目录>/inventory.json
python tools/krita_studio.py render --scene docs/krita-anime-painting/examples/latest-head.scene.json --output-dir <新的输出目录>/painting --record <新的输出目录>/render.json
```

输出目录应不存在或为空，避免覆盖同名文件。先做极小新画布与导出测试，再考虑完整场景；完整示例为 2000×3000，有较多图层，不是最小能力探针。没有接口时直接阅读示例或人工实现，不把替代工具生成的图说成 Krita 实画。

| 命令 | 行为及前提 |
|---|---|
| `status` | 读取桥接状态；并非读取 Windows 应用列表 |
| `inventory` | 通过 `run_python` 读取实际笔刷、滤镜、API；需要兼容桥接 |
| `render` | 创建新文档，按完整场景画图层并保存 KRA/PNG |
| `append` | 在指定活动文档追加不重名图层 |
| `revise` | 在指定活动文档替换唯一同名目标层 |

`append/revise` 要求 `expected_document` 与当前活动文档完整路径一致，尺寸也一致。另存后路径已改变，旧配方再次运行会拒绝，这是防护；应先核对活动文档与最新文件，再生成新的修正配方，不能删掉防护盲改。渲染失败时先检查已保存文件和现有图层，避免重复追加。

## 图层与坐标

场景记录 width/height、scale、图层及内容。SVG 自身含 viewBox；原生 strokes 的坐标通过场景 scale 处理，两者原点要显式换算。负 viewBox 原点、压力采样和局部放大都要小样验证。

`svg_masks` 用 DestinationIn 保留像素，`svg_erase_masks` 用 DestinationOut 扣除；这是显式栅格遮罩。`visible` 控制目标层可见，`visibility` 字典可在新增层时切换既有辅助层，避免 append 同名图层冲突。

Krita 组内继承 Alpha 要基于正确的不透明底形，不能把背景混入范围。用于扣眼睛遮挡的应是实际前发，不是把全部后发包络也扣掉；后者曾导致虹膜整片消失。

## 已遇到的问题及诊断

| 现象 | 本机发现与处理 | 可外推到哪 |
|---|---|---|
| SVG clipPath 不生效 | 本机 Qt SVG 子集不可靠，改显式 alpha 遮罩 | 不代表所有 SVG 引擎不支持 |
| 径向软色块透明 | 相对坐标径向渐变异常；按目标形状转换为 userSpaceOnUse，通过探针复查 | 仅本机测过的形状/变换，不保证任意 SVG |
| 眼睛消失 | 后发轮廓误用于前景扣除；只用真正前发并受眼裂限制 | 先查遮罩与层级，不先改颜色 |
| 眼周浅边 | 擦除区域大于眼裂，明暗层留下浅圈；收紧范围 | 需在原尺寸和整脸复看 |
| 保存 KRA 后 PNG 导出失败 | 等待绘制/保存完成后单独补导出曾成功 | 不是唯一根因的证明 |
| append 同名辅助层 | 用 visibility 切换已有层，新层名称保持唯一 | 不应为了隐藏草稿重复创建同名层 |
| 笔刷名称/滤镜字段错误 | 从 inventory 读取准确值，先画试样 | 原机器参数不能保证新机器有效 |
| 细线更多仍平 | 属于造型/明暗失败，回修主中束与转折 | 不用增加分辨率或滤镜回避 |

## 验证与打包

脚本语法校验不应 import 历史配方。知识包构建器会只编译源码文本、解析 JSON、检查本地文档链接范围并对归档文件逐一记 SHA-256；不启动应用。历史数据链接缺失会单列为原机外部证据，核心入口链接缺失则构建失败。

KRA 可作为 ZIP 检查完整性，PNG 应实际打开查看；这些技术检查与艺术质量分开。复算的像素一致性只适用于相同环境和确定性输入，不能据此承诺不同 Qt/Krita 版本完全一致。
