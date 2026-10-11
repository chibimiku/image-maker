# 投稿链「JSON → 交付图」定位修复（2026-10-09）

## 问题

用户问：*「35616ebb 传图片路径走的是 `find_metadata_json`，所以挂到了它自己的精灵 JSON；
如果拖那张 JSON，`find_image_for_json` 会解析到 `Downloads\db8f99d1-….png`（输入源图），
投出去就是错的图。这是啥 bug 导致的？」*

结论：不是崩溃型 bug，是 **`source_image_path` 一个字段被两个保存模式写成两种含义，
而投稿链无条件相信它**。

## 根因

写方 `modules/image_analysis/single_analyzer.py` 的 `on_process_finished` 有两条保存路径：

| 保存模式 | JSON 名里的 hash 段 | `source_image_path` 写的是 | 一致？ |
|---|---|---|---|
| `save_to_source=True`（保存到原图同目录、不生成图片）`:4628-4633` | 被分析那张图的**第一段**（`source_key`） | 被分析那张图（= 要投稿的那张） | ✅ |
| `save_to_source=False`（生成流程）`:4634-4636` | 生成任务的 `task_hash` | **生成时的输入源图**（`:4590-4594` 的 `meta_source_snapshot`） | ❌ 两个不同文件 |

读方 `publish_server.find_image_for_json` 是照第一种场景写的（`analysis_pipeline.py:6`
写着「JSON 含 source_image_path/task_hash + prompts TXT，可直接拖入投稿 Server」），
「优先使用 JSON 内 `source_image_path`（精确、无歧义）」—— `os.path.isfile` 一过就返回，
**不校验这个路径跟 JSON 自己的 hash 段有没有关系**。

下游也没有任何兜底：`add_json_to_queue` 走 `json_path_override`，**故意绕过**了
`find_metadata_json` 的一致性判断。护栏只做了反方向一半：`find_metadata_json` 有
「原图共用 Gemini 等前缀时，不拿明确指向另一张原图的 JSON」，注释甚至写着
「生成产物的 hash 关联仍可指向分析时的源图」——作者知道这件事，但 JSON→图片 这一向没有对等检查。

顺带暴露的第二个缺陷：`find_image_for_json` 用「同 hash 前缀恰好一张」选图，
而多轮任务会在同目录留下被打回的 `-aN` 候选（`<发布名>-a1_011346-xxxx.jpg`），
两张命中 → 返回 `None` → 拖 JSON 直接报「无法确定对应原图」。
历史上有 2026-10-07 的 `rejected-` 改名工具临时缓解，但那是数据侧补丁，不是修规则。

## 修复（`publish_server.py`）

新增共用身份判据与分层解析：

- `image_name_keys(name)`：文件名的第一个下划线分段 + 文件名里独立出现的 8 位 hex
  （兼容 `ajicoma-960cddea-quality-refine_…-final.jpg` 这类画风前缀旧命名）。
- `is_rejected_candidate(name)`：`…-aN_…` 或改过名的 `rejected-…`。
- `json_task_key/json_task_time`：JSON 名里的 key 与 `HHMMSS`。
- `find_image_for_json(json_path)` 分层：
  1. **队列既有配对**（`_queued_image_for_json`）—— 库里已经配好的照用，重拖不换图；
  2. **同轮首图审计台账** `selected_file`（`_ledger_selected_file`，要求 `selected_file` 与 key 一致）；
  3. 同目录里 key 对得上、且**只有一张不是打回候选**；
  4. key 对得上的全是打回候选 → 就用它（首图审计耗尽时就是这么交付的，历史 19 例）；
  5. 旧「保存到原图同目录」流程：`source_image_path` 存在**且文件名与 key 一致**；
  6. 都确定不了 → `None`（不猜、不静默配错）。
- `add_json_to_queue` 末尾加闸门：定位到的图与 JSON 的 key 对不上就拒（`error="mismatched_pairing"`），
  定位不到就用更准确的 `error="ambiguous_image"`。
- GUI 拖拽失败提示从「投稿 JSON 缺失或无效」改为「有文件未加入发布队列」，
  并把新的两类错误一起弹窗（原来只弹 metadata 类）。
- `find_metadata_json`（图片 → JSON）只做等价重构（复用 `image_name_keys` / `json_task_key`），**行为零变化**。

## 历史兼容性验证（只读）

`data/test-result/20261008/publish-status-check/history_regression.py` 扫全部 **3100** 个历史任务 JSON：

| 指标 | 旧规则 | 新规则（生产路径） | 新规则（去掉队列层） |
|---|---|---|---|
| 与发布队列真值一致（1707 个有发布记录的 JSON） | 982 | **1707 / 1707** | 1694 |
| 配错 | **703** | **0** | 1 |
| 返回 None | 22 | **0** | 12 |

- 反向查找（图片 → JSON）对队列里 1736 条可解析记录逐条比对：**1736/1736 与改动前完全一致**（零变化）。
- 不变量全 0 违例：返回的图必须在盘；必须与 JSON 的 key 对得上（队列既有配对除外）；
  同一轮存在非打回图时不得返回打回候选。
- 唯一的 `files_only` 分歧 `20261006-234440-0040e4ff`：该 JSON 的台账只记得本轮交付的
  `0040e4ff_234440-first-984aca-a1_234520-374b09.jpg`，而用户实际发的是后来重跑的
  `0040e4ff_234833-…`；生产路径由第 1 层（队列既有配对）覆盖，产出仍与当年发布一致。

## 测试

- 新 `tests/test_publish_server_image_lookup.py` **18 项**：生成流程源图不再被当成成品、
  外部源图永不返回、`-aN` 有交付名时跳过 / 只剩它时采用 / `rejected-` 降权、
  台账与队列两种权威来源、旧画风前缀命名、旧「保存到原图同目录」流程、
  Gemini 旧 key、歧义返回 None、入队闸门两类拒绝、反向查找不受影响。
- `tests/test_publish_server_metadata.py` 6 项原样通过。
- 全量 `python -m pytest tests/ -q --ignore=tests/test_style_ref.py`：
  改动前 **13 failed / 1579 passed / 5 skipped** → 改动后 **13 failed / 1598 passed / 5 skipped**，
  失败清单**逐条相同**（style-metrics / refine-atmosphere / analysis-gpt-prompt-skipping /
  save-to-source 命名 / 历史重跑 / 失败日志，全是本周既有），零新增回归。

## 现场验证

- 本日 `data/20261008` 的 10 个任务 JSON：**10/10 认到的图与各自审计台账 `selected_file` 完全一致**；
  `35616ebb` 由修复前的 `Downloads\db8f99d1-….png` 改认成 `35616ebb_233237-first-c1d0e4.jpg`；
  重拖一律被判重挡下，队列行数不变（`verify-live-after-fix.json`）。
- 之前已按图片路径入队的 9 条正在被油猴脚本正常消费（截至 2026-10-09 01:5x：
  5 条双平台 `uploaded`、5 条单平台 `uploaded`），修复没有影响上传链。

## 生效条件与遗留

- **必须重启 `publish_server`**：GUI 进程里是旧模块，拖拽解析走的是内存里的旧函数。
- 未改生成链（`source_image_path` 的写入语义保持原样 —— 上游还要用它读源图像素/做审计）。
- 未做 `-aN` 候选的落子目录/改名（`tools/rename_rejected_candidates.py` 仍在，需要时可用）；
  规则层已经不再被它们绊住。
- 未 commit、未改配置与画风，未动 `publish_queue.db` 的任何既有行。
