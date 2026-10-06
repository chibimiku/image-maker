# myc0t0xin 按字段合并验证（执行记录 2026-10-05）

> 后续审计发现第二次生成复用了第一次图片，本记录中的“24 张”“12 次调用”及第 5 轮字段优劣结论不能作为独立样本证据。已补测并去重，修正说明见 [独立样本补测与缓存修正](myc0t0xin-independent-samples-20261005.md)。运行目录报告已更新，补测前 JSON 与断点保存在 `before-independent-supplement/`。

对照任务书：`docs/style-extraction/deepseek-myc0t0xin-merged-validation.md`。
**只做验证与方案，未写入 `conf/config-styles.json`、未提交、未发布画风。**

## 做了什么

- 从 `data/20261004/style-extraction/myc0t0xin/run-20261004-225929-156628/myc0t0xin_style_iter_result.json`
  （sha256 `57ab73dc…0507`）按字段复制第 2/5 轮，构造三个候选包并重新校验：
  A 全取第 2 轮、B 的 `prompt` 取第 5 轮其余取第 2 轮、C 再把 `repaint_clauses` 换成第 5 轮。
  三包 `optional_motifs` 均为空。**没有重跑分析轮次，也没有把两段画风描述拼成一段。**
- 预检不联网、也不自拼提示词：直接调用真实组装函数
  `StyleIterativeWorkerThread._assemble_test_generation_requests`，重绘那一路再用
  `utils.post_process.assemble_repaint_request`（`run_pipeline` 发送前的那一份）。
- 三路测试各 2 次：第一阶段 3 包 × 3 路 × 2 = 18 张；第二阶段换主体 1 包 × 3 路 × 2 = 6 张。
  实际计费生图调用 12 次，其余按「实际传输内容逐字一致」复用既有产物。
- 视觉复核用 app 同一段代码（`compare_state_in_process`）在无头下跑，5 + 2 批；深度指标在
  Intel NPU（OpenVINO / FP16）上独立计算，12 源图等权。

## 关键结论

1. **「重绘条款也进入 GPT 首图」被实测证实**：A 与 B 的 `gpt_first_pass` / `gpt_repainted`
   实际请求逐字一致（可复用），而 C 换了条款后这两条通道的请求都变了 —— 只比配置字段会误判。
2. **第 5 轮的两个字段都不如第 2 轮**：
   - `gemini_full_prompt` 取第 5 轮（B/C 的 `prompt`）：Gemini 直出漂向参考集内容，视觉分反而更低；
   - `gemini_repaint_clauses` 取第 5 轮（C）：4 张 GPT 产物全部复制参考图 0 的角色设计，
     触发 `reference_content_copied` 门禁被排除。
3. 第 5 轮 GPT 短版（158 字符、只有 Palette 有值）校验不通过，**没有任何一包采用它**；
   B/C 的 `prompt_gpt` 按设计取第 2 轮，故 GPT 通道照常发送，没有回退母版顶替。
4. 同图不同分（GPT 首图 8.353 vs 7.711）说明视觉分只适合同批次相对排序，差值 < 0.6 不宜定论。
5. 逐字段建议 = **包 A**：`prompt` / `prompt_gpt` / `repaint_clauses` 全部取第 2 轮，
   `optional_motifs` 保持空、`enabled=false`，等用户观察报告后自行选用。

## 交付物位置

`data/20261005/style-extraction/myc0t0xin/merged-validation-20261005-103500/`

- `FINAL-CONCLUSION-zh.md`：中文最终结论（执行数量、费用依据、失败、限制、复现入口）
- `proposed-style-entry.json`：待选用的画风条目草稿（`ref_image` 指向原始参考图，`enabled=false`）
- `merged-validation-summary.json`：两阶段汇总（含两阶段的比较与深度结果）
- `primary/`：第一阶段（湖边粉发少女）
  - `myc0t0xin_style_iter_result.json`（app 可读）、`precheck.json`、`candidate-packages.json`、
    `request-audit.json`、`cost-estimate.json`
  - `automatic-comparison.html` / `.json`（视觉复核）、`deep-feature-comparison.html`（深度指标）
  - `merged-validation-report.html`（自包含主报告）、`conclusion-zh.md`、`manual-review-notes.md`、
    `contact-sheet.jpg`、`reference-sheet.jpg`、`generated/<包>/attempt-<n>/`
- `stage2-window-blue/`：第二阶段（窗边短蓝发绿眼少女），结构同上

## 代码与回归

- 新增 `tools/style_merge_validate.py`（预检 / 生图 / 复核 / 深度 / 交付物五个子命令，全部参数化）
  与 `modules/image_analysis/style_merge_validation.py`（候选包构造、预检、按通道复用、报告与建议）。
- `modules/image_analysis/style_analyzer.py`：把三路请求组装抽成
  `_assemble_test_generation_requests`（唯一组装点），`_generate_test_image` 改为调用它；
  Gemini / GPT 两段各自收口，某一路组装失败不再遮住另一路。
- `modules/image_analysis/style_comparison.py`：把视觉复核主体抽成
  `compare_state_in_process(...)`，GUI 线程与无头 CLI 共用同一段评分代码。
- `utils/post_process.py`：`assemble_repaint_request` 的返回值补记
  `clauses_without_image` 与 `scope`，让「文字模式」可以被预检核验而不是靠反推。
- 回归用例 `tests/test_style_merge_validation.py`（10 项，不联网）：
  字段逐字复制、第 5 轮短版不放行、预检 = 实际发送、按通道复用的边界、失败不复用。
