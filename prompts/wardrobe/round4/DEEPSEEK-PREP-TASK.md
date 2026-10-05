# DeepSeek 下一任务：Round4 离线准备

请先读 PROJECT_REQUIREMENTS.md、AGENTS.md，再读 `docs/style-extraction/LOLITA-ROUND4-PLAN-20261005.md` 和上一份素材研究。当前任务只准备，不执行在线提取、审查或生图；新一轮额度尚未确定。Round3 用户主动追加额度是有效授权，不需重新追究它。

源目录为 `C:\data\train\train\lo_v2_train\train\source\img`，只读。使用 `data/test-result/20261005/wardrobe-reference-study/inventory.json` 和图集，优先 classical/sweet/gothic。不要重扫成另一个训练任务，不修改源标签或源图。

1. 按 selection.md 逐图筛样，形成每类 8 提取 + 3 留出 + 3 边界的候选清单、结构依据及设计组。配额不足则如实说明。不要用目录名、标签或颜色自动批准样本。
2. 组织 6 个最多 4 图的实际提取请求草案和 3 个留出核对请求草案，使用 reference-study/extract-wardrobe.md，保留 ID/hash/附件顺序。不发送这些请求。留出图不进入提取正文或附件。
3. 依照 planning.json 组织 12 个完整转化请求草案和 2 个保留保护请求草案。使用人写草稿只能称 draft，不能冒充尚未联网产生的提取衣装包。记录组装来源与模板 hash；待提取完成替换候选并重新冻结。
4. 检查现有 tools/wardrobe_experiment.py、utils/wardrobe_experiment.py、tools/wardrobe_report_v2.py 的 Round3 硬编码。复用同族工具，必要时做最小参数化，不能复制一次性脚本。规划材料不是现有可执行 spec，不能直接传 CLI 后谎称 prepare 已通过。评价总结果必须包含人物数量、场景、配饰等全部必要条件。
5. 验证新路径与预算不误用 Round3 ledger；明确新 run-dir，但准备任务中不发收费请求。成功缓存继续复用，故障和未知发送按尝试记录；以后用户追加额度正常落授权记录。
6. 输出准备报告：已筛/不足数量、请求清单、待决定字段、可用离线测试、首波 14 张生成的真实调用清单与建议预算、留出隔离检查、必要适配 diff、明确尚未执行的阶段。材料存 prompts/wardrobe/round4，产物隔离存 data/test-result，不在根目录堆文件。

不改生产预设、GUI、全局配置，不发布，不重启 App；本轮准备任务没有涉及生产运行模块。若代码改动涉及生产则按项目规定验证与重启。不要为了写“准备成功”放宽真实缺陷。准备完成后等待用户对在线阶段的执行指示。
