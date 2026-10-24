# 图片分析「转化二次元」（2026-10-24）

用户要求：在**图片分析页**新增「转化为二次元」——把当前输入的真人照片交给**图片 edit 接口**，
按给定的固定长提示词（全文逐字）重绘成日系二次元美少女商业插画；**通道跟随界面上当前选择的
Gemini / gpt-image**。

## 功能位置与行为

- **按钮**：底部固定区「② 原始提示词生图 / ② 优化提示词生图」同一行，新增「转化二次元」。
  独立动作：**不需要先做文本分析**，加载了输入图（拖拽 / Ctrl+V 粘贴）即可点。
- **输入**：当前预览的照片。剪贴板输入（内存 PIL 对象）会先落快照（`_save_clipboard_snapshot`）
  拿到文件路径再发送。
- **提示词**：`prompts/anime-convert/edit-prompt.md`，用户提供的全文（约 6000 字符，十五个章节），
  **逐字发送、不改写、不拼接**；不挂画风参考图、不追加渲染条款。
- **比例**：`utils.post_process.snapped_aspect_ratio(输入照片)`——输出比例锁定为输入照片最接近的
  官方比例，避免宽图出竖图。
- **产物**：`data/<日期>/anime-convert/`（`save_sub_dir="anime-convert"`，文件前缀 `anime-convert`），
  首个产物旁写 `<产物>.anime-convert.json` 请求快照（通道 / 模型 / 比例 / prompt 字符数 / 源图 SHA-256）。
- **并发**：挂进 `_active_img_threads`，与既有生图共用「终止当前生图」按钮、超时倒计时与
  `_pipeline_busy` 互斥；有任务在跑时点击只提示跳过。不进分析队列、不建历史任务（`meta_task_hash=""`）。

## 通道路由（跟随「生图通道」单选）

| 界面选择 | 实际请求 | 模型 / 尺寸来源 |
| --- | --- | --- |
| gpt-image | `generate_image_aigc2d_gpt`，带图 → `/v1/images/edits`（multipart） | `aigc-2d-gpt` 配置；尺寸按输入照片比例收敛到 3 档（`pick_gpt_image2_size_for_images`） |
| Gemini（api_type=aigc2d） | `generate_image_aigc2d`（generateContent，照片为内联附件） | 全局生图配置的模型；`face_quality_boost=False`（正文已含完整面部要求，不追加英文后缀） |
| Gemini（其它 api_type） | `generate_image_whatai`（内部再分发 openai-image / autodl / openrouter / whatai） | 全局生图配置的模型与 api_type |

超时预算：gpt 通道读 `aigc-2d-gpt` 配置 `timeout`（默认 600s）；Gemini 通道读界面文本超时设置。

## 代码位置

- `prompts/anime-convert/edit-prompt.md`：固定提示词全文。
- `utils/anime_convert.py`：纯逻辑（prompt 读取 / 请求组装 / 通道路由 / 请求快照），可脱离 Qt 测试。
- `modules/image_analysis/single_analyzer.py`：
  - `AnimeConvertWorkerThread`（QThread，取消 = `requestInterruption`，失败只报日志不影响队列）；
  - `anime_convert_btn` 按钮 + `trigger_anime_convert` / `_on_anime_convert_finished` / `_sync_anime_convert_btn`；
  - `show_preview` / `show_clipboard_preview` 同步按钮可用；生图启动 / 取消 / 收尾同步禁用与恢复。

## 验证

- 新增 `tests/test_anime_convert.py` **16 passed**：固件全文十五章节齐全、无模板占位符；
  请求组装（比例锁定 2:3、子目录、gpt 三档尺寸、缺文件报错）；三个通道的路由断言
  （gpt=edits 带图、aigc2d 不加英文面部后缀、autodl 走 whatai）；无产物不写快照；
  界面接线（初始禁用 / 预览后可用 / 无图与忙时跳过 / 线程登记 / 收尾恢复按钮）。
- 编译：`app.py` / `single_analyzer.py` / `utils/anime_convert.py` / 测试文件 `py_compile` 通过；
  `utils.anime_convert` 导入正常（prompt 6036 字符）。
- 相关回归：`test_analysis_channel` + `test_analysis_gen` + `test_analysis_gpt_prompt_skipping`
  100 passed / 2 failed；`test_analysis_fallback` + `test_analysis_cli_unittest` +
  `test_quality_failure_reporting` + `test_anime_convert` 55 passed / 2 failed。
  **4 个失败均为既有失败、与本功能无关**（后续同日已另行修复，见下）。

### 同日追加：五个既有失败测试修复（0 生产代码变更）

与本功能无关、但同日按用户要求把当时可见的 5 个既有失败用例修绿（均为测试未跟上既有生产改动）：

1. `test_analysis_gpt_prompt_skipping::test_worker_thread_only_calls_gpt_prompt_builder_when_asked`
   （×2）：WorkerThread 显式传 `secondary_config`（Step 2 DeepSeek 次级通道在测试隔离下读不到
   NSFW key，会在进入打桩的 step_2 前直接 error）。
2. `test_analysis_cli_unittest::test_cli_applies_analysis_ui_options`：用例内
   patch `postprocess_enabled→True`（默认 False 时 `_build_gpt_image_steps` 返回 {}，断言不到映射）。
3. `test_analysis_cli_unittest::test_random_style_resolves_at_task_entry_and_stays_fixed_for_generation`：
   两处显式 trigger 补 patch `get_api_config`（gpt 通道入口 key 预检在隔离环境下为空）。
4. `test_analysis_refine_request::test_worker_keeps_step2_refused_state`：同 1 的根因与修法。

修复后 16 套件 **431 passed ＋ 2 subtests** 全绿（analysis / gpt_image2 / refine_state /
style_gpt_prompt 家族整批复跑）。
- **未做线上收费请求**；模型实际转化效果未在线验证。需重启 app 加载新代码。
