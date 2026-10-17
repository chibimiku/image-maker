# 二次元绘画研究与 Krita 实作知识包

整理日期：2026-10-17。供人和协作 agent 阅读。目标是细腻的蓝发少女坐椅插画；目前仍处于练习阶段，尚未达到参考插画完成度。

最新：[新画布整体练习《花园小憩》](examples/garden-chair-20261017/README.md)，含最终 PNG、分层 KRA、全部本轮阶段图/配方和读图记录。用户追加要求上传后，[Puracotte 原参考与头部观察图](references/puracotte-149567189/README.md) 已单独收录，保留第三方来源与权利说明。

参考拆解：[头部分析](head-parts/puracotte-149567189-head-analysis.md)，覆盖倾斜方向、五官、柔和腮红、发束转折和遮挡；附 [独立腮红/发束小样](examples/puracotte-head-study-20261017/README.md)。

最新本地研究：[面部四轮迭代及读图决策](head-parts/face-iterations-20261017.md)，含 [可直接查看的各轮图与完整场景](examples/face-iterations-20261017/README.md)。旧交接基线保留，不能把旧包 KRA 当本批新稿。

接续：[面部比例与颧颊A/B实验](head-parts/face-proportions-20261017.md)，全部绘制由Krita MCP桥接执行；[David Revoy包已导入启用并试画](resources/deevad-install-20261017.md)，试用不等于采用，保留稿关闭了无明确收益的罩染层。

**第一步先读 [环境探索](ENVIRONMENT.md)。接收方不一定安装 Krita、Python、MCP 或相同笔刷，不能直接照抄本机路径、端口和命令。阅读资料无需安装任何软件。**

## 推荐阅读顺序

1. [环境探索](ENVIRONMENT.md)：只读确认能力，再选人工绘制、现有接口或受支持的自动化方式。
   随后读 [分部位学习路径](LEARNING-PATH.md)：每个部位先学习、独立练习、识图返修，并更新 [学习记录索引](learning-records/README.md)。
2. [统一绘画手册](HANDBOOK.md)：归纳草稿、透视、五官、头发、光影、坐姿与自检。
3. [协作交接](COLLABORATION.md)：实际进度、待解决项、任务划分、交付约定和 agent 入门指令。
4. [工具与迁移限制](TOOLING.md)：JSON、桥接、图层遮罩、失败经验和历史脚本依赖。
5. [完整目录与资料索引](CATALOG.md)：所有专题、历史轮次和作者来源的入口。
6. [随包示例](examples/README.md)：无需本机历史目录即可查看的图像、独立场景与模板。
7. [整体练习计划](PRACTICE-PLAN.md)：七阶段目标与未完成项。

## 内容布局

| 路径 | 内容 |
|---|---|
| 上述总览文件 | 归纳后的通用知识、环境与交接要求 |
| `facial-features/`、`head-parts/` | 五官、朝向、透视、逐部位自检与头颈体积 |
| `hairstyles/`、`hair-structure/` | 发型与分区、中小束、材质、失败实验 |
| `expressions/` | 独立表情专题，避免干扰发型主稿 |
| `templates/` | 方向、位置、光照及检查项目的输入模板 |
| `examples/` | 精选自绘成果、可独立渲染场景及来源映射 |
| `tools/` | 历史研究脚本、只读环境探查和知识包构建器 |
| 仓库根 `tools/krita_studio.py` | 特定本机桥接 CLI；ZIP 保持这一相对路径 |

## 完整查看与转交

克隆仓库或解压知识包后，从本页开始。所有积累的 Markdown、JSON 模板与研究脚本均保留；统一手册是归纳入口，专题文件保留来源、推理和失败记录。ZIP 根另有 `START-HERE.md`。

历史文件里的 `data/test-result/...`、`D:\...` 指向原作者机器，通常不在仓库或知识包内。它们是现场证据位置，不是接收方必须具有的目录。优先看 `examples/`；缺失历史文件应记为缺失，不猜测结果。

知识包包括全部学习文档、脚本、精选图、最新独立场景、`examples/garden-chair-20261017/garden-pause.kra`，以及用户明确指定收录的 Puracotte 原参考。其他历史 KRA/原参考、第三方教程原图/视频、插件安装程序、令牌、私有配置及下载的笔刷二进制仍不包含。笔刷来源与原记录见 [资源清单](resources.md)，按接收方环境另行选择。

```text
python docs/krita-anime-painting/tools/build_knowledge_package.py --output <新的输出目录>
```

打包只读取材料并写 ZIP/清单，不连接 Krita、不运行历史绘制脚本。包含范围及 SHA-256 见生成的 `package-report.json` 与 ZIP 内 `MANIFEST.json`。原作者仓库有最新 KRA 时会附上；仅从 Git 重建时可不含该二进制，报告会明确列出缺失。

## 当前结论

宝石眼已有较清楚的上暗下亮、反射与眼睑遮挡；头颈练习增加投影和转折。头发仍偏规整带片，鼻部体积、自然神态、衣料、完整坐姿与椅子透视仍需练习。脚本成功、分辨率提高、笔刷或路径数量增加都不是画质通过依据。

绘制主要是程序规划路径和遮罩，在本机 Krita 中分层渲染，部分历史实验使用原生笔刷和滤镜。没有把它描述成自由手绘、训练模型或已完成成品。

旧版总览与本机早期产物入口保留于 [历史总览](LOCAL-HISTORY.md)，其中“最新”等措辞仅代表当时轮次，应以本页及协作交接为准。
