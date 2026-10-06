# 接续实验：先修正授权合同冲突

当前状态：**离线预检完成，未发送**。本文件是下一次执行的任务书，不代表已经执行或必须现在消费 API。用户将本文件交给执行者并要求运行时，按下面的冻结协议执行。

先读 `PROJECT_REQUIREMENTS.md`、`AGENTS.md`、`CODEX-CONTENT-ANCHOR-REVIEW-20261005.md`。沿用 `tools/color_knowledge.py generation-knowledge`；不新建一次性脚本、不改生产 GUI/配置/门禁。

## 唯一改动与样本

- 新协议：`prompts/color-knowledge/book-region-contract-v1.json`。
- 新内容配置：`prompts/color-knowledge/book-region-contract-content-v1.json`。与旧 `fixed-palette-v1.json` **只有 protection_block 不同**：删除“只重染腰带与鞋结”的冲突，将最后的授权范围与前文 7 个区域统一，并保留原内容约束。
- C16 内容参考分支，两轮，共 **2 次生图、1 次分组评审**；不生成新文字分支、不用源色接近目标的 C13。不新增局部色调、材质、面积或色相数量规则。
- 同一源图 hash、Flash/1K/3:2、无画风附件、零自动重试。参考图为旧 C13-simple-r1，绝不能按美感换图。
- 输出：`data/test-result/20261005/book-region-contract-v1/`。
- 新 plan hash：`8051d5fd7129c9465f35ab544a6bdf286a23a32fb2da3f64969664068eb70abb`。

2 份 preview 中附件已实际编码，`sent=false`；当前没有调用槽或生成结果。冻结评审端点/模型应与旧批一致，生成端点核对实际配置为 `new.aigc2d.com`；不一致就停止，不改冻好的计划。

## 执行范围

执行时只上传冻结内容源和本轮候选用于生成/评审，不能上传 PDF、项目目录、配置或凭据。关闭底层重试；成功复用，失败保留，不补抽、不换图、不额外修色。历史 C13-text 的评审超时不在本轮范围，不占用这 1 次评审或擅自补评。

```powershell
$env:IMAGE_MAKER_IMAGE_MAX_RETRIES="0"
& "C:\Program Files\Python310\python.exe" -m unittest discover -s tests -p "test_book_*.py"
& "C:\Program Files\Python310\python.exe" -u tools/color_knowledge.py generation-knowledge preview --spec prompts/color-knowledge/book-region-contract-v1.json --out data/test-result/20261005/book-region-contract-v1
& "C:\Program Files\Python310\python.exe" -u tools/color_knowledge.py generation-knowledge run --spec prompts/color-knowledge/book-region-contract-v1.json --out data/test-result/20261005/book-region-contract-v1
& "C:\Program Files\Python310\python.exe" -u tools/color_knowledge.py generation-knowledge review --spec prompts/color-knowledge/book-region-contract-v1.json --out data/test-result/20261005/book-region-contract-v1
& "C:\Program Files\Python310\python.exe" -u tools/color_knowledge.py generation-knowledge report --spec prompts/color-knowledge/book-region-contract-v1.json --out data/test-result/20261005/book-region-contract-v1
```

## 必须逐图回答

1. 是否实际单人、源主要姿势/构图保持、无重复画幅/标签/色卡？人体连接另检查，灰度通过不能代替它。
2. 蓝色是否落在水面、天空、远景植被；黄色是否仅落在人物脚下近岸干地；腰带与左右鞋结是否各有橙红点缀？
3. 对岸干地、近景植物及已有小花是否保持**源图实际颜色**？源近景叶片为粉紫，不能因为“植物应绿色”而擅自改绿并判通过。
4. 肤色、银灰发、象牙白裙、深中性鞋体是否保持？授权饰带/鞋结与保护鞋体分开检查。
5. `protected_ok` 是否与逐区域问题自洽？原模型响应保留，离线更正另写绑定产图 hash 的 adjudications，不覆盖原判。
6. 边界模糊处记 partial/unverifiable；色族与区域判断不冒充精确面积或校准色度测量。

交付原请求/响应、调用记录、逐图及汇总 JSON、内嵌全部结果的评价页，并单写 `EXECUTION-REPORT.md`。报告分开列出：生成失败、自动评审缺失、明确配色/保护失败、内容失败、通过。失败不从分母消失。

本轮是修正自相矛盾后的诊断性复验。历史旧请求不自洽，不能作为干净的随机对照；两张新图即使都通过，也只支持继续多内容源验证，不能声称已证明改词的因果收益或稳定性。完成后停下，不自动进入局部色调、材质、蒙版或生产接线。
