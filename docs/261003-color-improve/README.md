# 配色知识与生图试验文档入口

## 当前阶段（2026-10-03）

- [配色工作流设计](COLOR-WORKFLOW-DESIGN.md)：分析生图/直接生图共用下拉、LLM 自动选择、作用范围、方案字段、附加工序、评价及缓存契约。
- [春日钓鱼第一阶段结果](PILOT-SPRING-FISHING-RESULTS.md)：Gemini Flash 三组各两张，真实请求、观察、指标与局限。
- [离线方案卡与六图浏览](../../data/test-result/20261003/color-spring-fishing-v1/gallery.html)：含原图、方案条款、自动选择与匿名评价。

目前只有独立 CLI 试验，生产 GUI 尚未接入。结论是“颜色组织发生变化”，还没有证明方案稳定优于基线。柔蓝组第一张出现两个人，保留为内容失败样本。

## 前置研究与知识来源

- [DeepSeek 工作总结](WORK-SUMMARY.md)：转录和知识抽取的历史记录；后续消费层事实以新的设计和试验文档为准。
- [v2 知识提取工作流](AI_Color_Knowledge_Extraction_Workflow-v2.md)：前置工作流方案与审阅回应，部分提议尚未实施。
- 书籍按章离线站点：150 页图文转录（含原书 PDF）只在本机保留，已 gitignore，clone 出来的仓库里没有 `book-html/`；原文与转录者适配应区分。
- 知识库：`data/color-knowledge/`；本批方案、来源及近似色板：`prompts/color-knowledge/pilot-spring-fishing.json`。

后续流程优先验证跨题材收益、原有画风兼容及重绘保留，再实施完整 UI 和生产审计。此次没有发布候选图，也没有修改既有画风/质量/身份固件。
