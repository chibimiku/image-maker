# 笔刷与教程资源

核查日期：2026-10-16。只把作者项目、作者原站和 Krita 官方文档作为资源入口。

2026-10-17 打包说明：下述“已下载”和“本机”均是原作者历史状态。知识包和 Git 不包含下载的 ZIP/bundle 二进制，只保留来源、许可和下载清单；链接到这些本地资源的文件可能不存在。接收方先探索环境，再决定是否从作者站获取与导入，不能据此假定已安装。

## 实际使用与下载状态必须分开

截至本次整理，下载的 David Revoy 2025-01 包**未导入、未用于这些练习**；Rakurri 包也未下载或使用。历史实际笔触使用本机已有的 Krita 默认预设，试样与采用情况见 [本机实验](experiments-20261016.md) 和 [任务取舍](painting-notes.md)。最新头颈完整场景没有原生笔刷笔触，使用的是程序路径/遮罩经 Krita 分层渲染，不应说成使用了某个下载笔刷。

朋友无需为查看或复算这个最新场景下载上述笔刷。以后若采用新包，必须在对应练习报告记录：作者页、直接下载地址、包/预设版本与准确名称、许可、下载校验、安装/启用状态、使用它的具体图层与前后效果；不能只写“用了头发笔刷”。

## 已下载：David Revoy 2025-01

- 作者页：[Krita brushes 2025-01 bundle](https://www.davidrevoy.com/article1060/krita-brushes-2025-01-bundle)
- 作者页实际链接：[deevad-bundle_25.01.zip](https://www.peppercarrot.com/extras/resources/deevad-bundle_25.01.zip)
- 大小：8,509,341 bytes。
- ZIP SHA-256：`4c628a9418fcde63abacafdcb143881f2cbbf907275cb4f72335545841cf8173`，与作者页公布值一致。
- 文件：[ZIP](resources/deevad-bundle_25.01.zip)、[可导入的 Deevad_25.01.bundle](resources/Deevad_25.01.bundle)、[下载清单](resources/manifest.json)。
- 作者明确将**笔刷**许可为 CC0。网页文章、演示作品是另外的内容和许可，不能把笔刷的许可套到示例图片。
- 作者要求 Krita 5.2.6 或更新，本机 5.3.3 满足版本条件；这不代表每支笔刷已实测。
- 适合研究：线稿、上色、混色、细节的成套工作方式；作者还提供了针对旧版包的讲解，称多数方法适用于此版。
- 状态：已下载、校验、解包；尚未导入运行中的资源库，尚未实测此包。第二版使用本机自带笔刷。
- 安装：Krita「设置 → 管理资源 → 导入资源」，选择解出的 `.bundle`，再查找 `deevad 25.01` 标签。以压缩包里的真实文件名为准，作者文章安装段保留了旧版文件名。

## 已核查候选：Rakurri Brush Set V2.1

- [作者 GitHub](https://github.com/Rakurri/rakurri-brush-set-for-krita)
- [作者指定 V2.1 发布页](https://github.com/Rakurri/rakurri-brush-set-for-krita/releases/tag/V2.1)
- 作者允许用于个人作品且不要求署名。未据此推断可以任意重新分发或重许可全包。
- 用途：混色、绘制、特殊笔刷的扩展研究。
- 注意：作者说明很多笔刷依赖倾斜输入。本机的脚本绘画接口没有在本轮传入倾斜，因此不能期待与数位笔演示完全相同。官方 API 不同版本可能增加输入能力，应先检查本机方法。
- 状态：查阅说明，未下载、未安装、未实测。

## 官方资源索引

[Krita Resources](https://docs.krita.org/en/resources_page.html) 列出笔刷包与纹理等入口；进一步优先跟随到作者原站，记录来源和许可。

本轮原则：有明确目标再增加笔刷；不批量导入大量相似预设。下载不等于安装，安装不等于完成画法验证。
