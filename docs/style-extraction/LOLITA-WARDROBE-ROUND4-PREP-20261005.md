# Round4 离线准备报告（2026-10-05）

> 后续修正：部分图像注释和设计组判断已被人工纠正，留出图已查看，不再作为独立验证。旧请求草案不用于执行；当前生图包与边界见 [WARDROBE-ROUND4-READY-20261005.md](WARDROBE-ROUND4-READY-20261005.md)，交接使用 `prompts/wardrobe/round4/DEEPSEEK-GENERATION-TASK.md`。

任务：按 `prompts/wardrobe/round4/DEEPSEEK-PREP-TASK.md` 做**只准备、不执行**的离线环。
本轮没有发送任何收费请求（提取、留出核对、生图全部未发），没有截图、没有改源素材、
没有改生产预设/GUI/全局配置，没有发布，也没有重启 App。

- 计划书：`docs/style-extraction/LOLITA-ROUND4-PLAN-20261005.md`
- 素材研究：`docs/style-extraction/LOLITA-REFERENCE-DATASET-STUDY-20261005.md`
- 源目录（只读）：`C:\data\train\train\lo_v2_train\train\source\img`
- 产物隔离目录：`data/test-result/20261005/wardrobe-round4-prep/`
- 材料目录：`prompts/wardrobe/round4/`
- 计划中的新 run-dir：`data/test-result/20261005/wardrobe-round4-r1/`（**尚未创建**）

## 一、已筛/不足数量

筛样结论写在 `selection.json`（42 条记录，每条带源路径、源 SHA256、画幅、可见结构、
选入理由、不确定项与设计组 ID）。候选池与带 ID 接触图在
`candidates.json` + `sheets/`（每类 150 张候选、10 张接触图，格子标 ID/画幅/尺寸/分组）。

| 系列 | 提取正例 | 留出正例 | 边界/反例 | 提取正例画幅覆盖 | 设计组数（提取） |
|---|---:|---:|---:|---|---:|
| classical | 8 / 8 | 3 / 3 | 3 / 3 | portrait 7、long_portrait 1（2 个目录） | 7 |
| sweet | 8 / 8 | 3 / 3 | 3 / 3 | portrait 6、long_portrait 2（2 个目录） | 7 |
| gothic | 8 / 8 | 3 / 3 | 3 / 3 | portrait 5、long_portrait 3（2 个目录） | 8 |

**如实说明的不足（没有用目录名、标签或颜色补数）**

1. `selection.md` 要求提取正例尽量跨至少 3 个画幅目录、任一画幅不超过 4 张。
   本轮**达不到**：每类只覆盖 2 个画幅目录，且 portrait 数量超过 4（7 / 6 / 5）。
   原因是素材本身极度偏竖幅（classical 401 张里 portrait 280、landscape 只有 35），
   而横图与方图里绝大多数是坐姿、躺姿、半身或裙体被裁掉的构图。本轮逐图看过每类
   150 张候选（含所有 landscape / long_landscape / square 候选），没有任何横图/方图
   候选能同时读清上身、腰部与裙体。为此另外做了两件事，避免「没看就说不行」：
   - 对三张横图候选按源分辨率重新裁了腰—裙区域（`crops/*-skirtzone.jpg`）再看一遍，
     结论仍是姿势/角度不可读，因此不入选；
   - 把已复核但未入选的横图候选（如 SWX121 双人并排图）登记在
     `selection.json#reviewed_but_not_selected`，供在线阶段判断能否当组合视图用。
2. 提取正例每类**正好 8 张**（配额上限），因此把若干结构清楚、但同类结构已有代表的
   样本（CLX132、SWX071、SWX081、SWX090、SWX109、GTX091）登记为已复核未入选，
   而不是塞进 8 张里。它们是第一顺位替补，不是「被拒绝」。
3. 候选池之外的图片（classical 251 / sweet 283 / gothic 262 张）**没有进入本轮视觉复核**，
   所以只登记为「未复核」，不写成「被拒绝」，也没有据此宣布任何类别纯度。
4. 全部入选仍是 **provisional**：本轮的复核证据是接触图（源图的等比缩略），
   不是源分辨率裁剪。提取阶段必须按原分辨率复核，`review_status` 已标
   `reviewed_in_contact_sheet`。
5. 边界例覆盖了要求的三类：只有颜色像/泛用洋装或偶像衣装（CLX113、SWX054、SWX086）、
   同色系但结构不成立（CLX098、GTX048、GTX079）、姿势或裁切导致结构不可读
   （CLX120、SWX145、GTX107，其中 GTX107 的腰线与领口直接写 unknown）。

## 二、请求清单（全部为草案，未发送）

清单与模板 hash 在 `prompts/wardrobe/round4/requests/manifest.json`；
每个请求一个 JSON（`system_prompt` / `user_text` / `content_blocks` / `attachments` /
`attachment_order` / 模板与正文 SHA256），全部标 `status: draft_not_sent`、`sent: false`、
`real_http_requests: 0`。

**提取请求 6 个（每个 4 图，使用 `templates/extract-wardrobe-round4.md`）**

| 槽位 | 系列 | 附件顺序（ID，源 SHA256 前缀） |
|---|---|---|
| `X1-classical-a` | classical | CLX015 `ccbb1584`、CLX076 `101fa461`、CLX080 `b430e4b9`、CLX082 `6d833f27` |
| `X2-classical-b` | classical | CLX094、CLX102、CLX127、CLX131 |
| `X3-sweet-a` | sweet | SWX019、SWX021、SWX025、SWX035 |
| `X4-sweet-b` | sweet | SWX062、SWX068、SWX077、SWX079 |
| `X5-gothic-a` | gothic | GTX004、GTX011、GTX014、GTX021 |
| `X6-gothic-b` | gothic | GTX024、GTX068、GTX077、GTX098 |

（完整 hash 见各请求 JSON 的 `attachments[].source_sha256`；表内只列第一张的短前缀示意。）

**留出核对请求 3 个（每个 3 图，使用 `templates/heldout-check-round4.md`）**

| 槽位 | 系列 | 附件顺序 |
|---|---|---|
| `H1-classical` | classical | CLX014、CLX044、CLX129 |
| `H2-sweet` | sweet | SWX028、SWX040、SWX124 |
| `H3-gothic` | gothic | GTX015、GTX047、GTX094 |

**转化草案 12 个 + 保留保护草案 2 个**（`F4-<case>-<family|preserve>-gpt-r1`）
对应 `planning.json#slots` 第一波；共同条款取 `reference-study/full-transformation.md`，
系列块取 `reference-study/{classical,sweet,gothic}.md`，保留路径取
`reference-study/preserve.md`（round3 的 `variants/preserve.md` 同文）。
每个草案记录 `assembled_from`（模板路径 + SHA256）与正文 SHA256/字符数（4095–4467 字符）。

**留出隔离检查（`offline-checks.json`，23 条全过）**：留出图不在任何提取请求的附件里、
留出 ID 不出现在任何提取请求正文里、提取图不在留出请求附件里、提取与留出的**设计组
完全不交叉**（跨 split 组交集为空）、9 张留出图恰好覆盖 3 个核对请求。

## 三、待决定字段

以下字段在准备阶段**有意留空**，需要在线阶段或用户决定：

| 字段 | 现状 | 为什么不能由准备任务填 |
|---|---|---|
| 视觉模型名 | 请求里写 `UNSET_AT_PREP_TIME` | 运行时登记，写死会把端点/模型混进冻结材料 |
| 生图模型与参数 | 直接引用 `round3/spec/plan.json#generation_params` 的冻结值 | 沿用记录即可，本轮不新增参数 |
| `cli_compatible` / `online_execution_authorized` | `false` / `false` | 新额度未授权；在线执行需要用户指示 |
| 候选衣装包（3 个系列） | 人写草稿 | 提取尚未联网执行，草稿不能冒充提取产物 |
| 评价字段的最终取舍 | 已按 `planning.json#required_report_fields` 冻结 | 若用户要改判据，需要先改计划再重冻结 |
| 第一波是否加做在线评分 | 建议 0 次 | 计划书建议第一波先人工视觉复核 |
| 第二波触发条件 | 未定 | 计划书写明「第一波系统性失败则暂缓」 |

## 四、可用的离线测试

- 回归用例：`tests/test_wardrobe_round4_prep.py`（20 条，全过）——覆盖「非 round3 的 schema
  判未通过」「在线阶段被守卫拦住」「不支持的计划连 run-dir/账本都不建」「round3 指纹不变」
  「报告字段按计划声明读取」「留出隔离」「草稿标注」「评价字段齐全」「首波 14 张清单」。
- 同族旧回归：`tests/test_wardrobe_experiment.py` + `tests/test_wardrobe_report_v2.py`（60 条，全过）。
- 本轮自检脚本：`cache/temp/round4_offline_checks.py`（23 条确定性检查，全过，输出
  `offline-checks.json`）。**注意**：`cache/temp` 下的脚本是临时材料，不是交付链路；
  在线阶段应把「选择 → 请求草案 → 预检」这一族并进 `prompts/wardrobe/round4/spec` 与
  同族 CLI，不要把一次性脚本留在运行路径里。
- 全量 `python -m pytest -q -p no:cacheprovider`：1083 passed / 5 skipped / **3 failed**。
  3 条失败都在**别的在建工作流**里（`tests/test_save_to_source_dir.py` 的
  `test_save_to_source_filename_format`，`tests/test_single_analyzer_history_rerun.py` 的
  `test_analysis_submission_freezes_save_and_generation_options[False]` 与
  `test_switch_generation_save_generation_applies_only_to_new_submissions`），
  报错点是 `modules/image_analysis/single_analyzer.py` 的 `generation_wardrobe`/gen_targets
  相关行为（`git status` 里该文件带未提交改动）。**与本次改动无关**：本次只改了
  `utils/wardrobe_experiment.py`、`tools/wardrobe_experiment.py`、`tools/wardrobe_report_v2.py`，
  以及新增 `tests/test_wardrobe_round4_prep.py` 与 round4 材料。没有为了让报告好看而放宽它们。

## 五、首波 14 张的真实调用清单与建议预算

首波 = 计划里 `wave: 1` 的 14 个槽位，通道固定 `gpt`（`generation_params.gpt`：
`aigc-2d-gpt` / `gpt-image-2` / 1024x1536 / n=1 / quality=high / png / 无参考图）。
清单同时写入 `offline-checks.json#wave1_call_list`。

| # | slot | case | 系列 | 正文字符 | 正文 SHA256(前10) |
|---:|---|---|---|---:|---|
| 1 | F4-U-classical-gpt-r1 | U | classical | 4098 | `2a048cd1df` |
| 2 | F4-U-sweet-gpt-r1 | U | sweet | 4154 | `658605177d` |
| 3 | F4-U-gothic-gpt-r1 | U | gothic | 4095 | `15e101d0b1` |
| 4 | F4-H-classical-gpt-r1 | H | classical | 4336 | `1a6f071d12` |
| 5 | F4-H-sweet-gpt-r1 | H | sweet | 4392 | `2e5b66382d` |
| 6 | F4-H-gothic-gpt-r1 | H | gothic | 4333 | `037158c6f8` |
| 7 | F4-W-classical-gpt-r1 | W | classical | 4379 | `35ec9c0498` |
| 8 | F4-W-sweet-gpt-r1 | W | sweet | 4435 | `315635a3c6` |
| 9 | F4-W-gothic-gpt-r1 | W | gothic | 4376 | `cb4f926782` |
| 10 | F4-R-classical-gpt-r1 | R | classical | 4411 | `d3cdeff984` |
| 11 | F4-R-sweet-gpt-r1 | R | sweet | 4467 | `44d68336ab` |
| 12 | F4-R-gothic-gpt-r1 | R | gothic | 4408 | `dbc7127f36` |
| 13 | F4-GP-preserve-gpt-r1 | GP | 保留 | 1625 | `1c03dd3eb1` |
| 14 | F4-GF-preserve-gpt-r1 | GF | 保留 | 1597 | `60ee32789a` |

模板冻结 hash（`requests/manifest.json#templates`）：
`templates/extract-wardrobe-round4.md`、`templates/heldout-check-round4.md`、
`templates/score-round4.md`、`templates/review-round4.md`、
`reference-study/full-transformation.md`、`reference-study/{classical,sweet,gothic}.md`
各记 SHA256 与字节数；每个草案的 `assembled_from` 再记各自的组合来源与 hash。

建议预算（**建议值，不是授权**，与 `planning.json#proposed_budgets_not_granted` 一致）：

- 视觉提取/留出核对：9 次（6 批提取 + 3 批留出）
- 文本汇总/请求审查：2 次
- 第一波生图：14 次（失败也算尝试，不保证 14 张成功图）
- 第二波生图：14 次（Gemini，与第一波同一份冻结候选；第一波系统性失败则暂缓）
- 在线评分：0 次；若要做，建议逐图 14 次并单列，不与上面混算
- 重试预留：0 次（重试要单独声明用途）

## 六、账本与预算隔离

- 新 run-dir `data/test-result/20261005/wardrobe-round4-r1/`：**尚未创建**
  （`offline-checks.json` 里 `run_dir_created: false`）。它的账本是独立的
  `ledger.json`，与 Round3 的 `data/test-result/20261005/wardrobe-round3-prep/ledger.json`
  完全不同名、不同目录。
- Round3 账本本轮**未被读改写**：自检记录了它的路径、上限、attempts 数与 SHA256
  （`edbf407b…`），仅供对账；本轮没有 `plan_attempt`、没有 `grant_budget`。
- 预算键不共用：Round3 账本是 `text/generation/vision/model_list` 四个 HTTP 上限键，
  Round4 的建议表按功能分行（提取 9 / 文本 2 / 第一波 14 / 第二波 14 / 评分 0 / 重试 0），
  自检断言两者的键集合不相交，避免「读错账本把 Round3 剩余额度当本轮额度」。
- 成功缓存复用：本轮不动 `budget-slots/` 与 `cache/`；在线阶段沿用同族规则
  （成功调用可复用，失败与未知结果按一次尝试记录，被用户授权放行的门写进
  `gate_overrides` 并在报告里标注「放行 ≠ 通过」）。
- 以后用户追加额度：按 `ExperimentLedger.grant_budget()` 正常落授权记录
  （授权人/理由/旧值/新值/当时已消耗），不重置已消耗、不换 run-id。

## 七、必要适配 diff

三处同族工具的 Round3 硬编码已做**最小参数化**（全部对 round3 行为保持兼容：
round3 的 `spec_hash`、`validate_character_spec` 结论与报告字段口径逐字不变，旧回归 60 条全过）。

1. `utils/wardrobe_experiment.py`

```diff
-def spec_hash(spec_dir, *, characters_file="", base_pack=""):
-    ...
-    for folder in ("variants", "characters"):
-        parent = os.path.join(os.path.dirname(spec_dir), folder)
++    digest_spec = plan.get("spec_digest") or {}
++    packs = [base_pack or digest_spec.get("base_pack") or plan.get("base_pack") or ""] + digest_spec["base_packs"]
++    directories = digest_spec.get("declared_dirs") or [root/"variants", root/"characters"]
++    # declared_files/base_packs 逐文件摘要；没声明就走原默认，round3 指纹不变
+
+def assert_online_supported(plan) -> None:
+    # 新实验 schema 的在线阶段守卫：cli_compatible / online_execution_authorized 必须为 true
+
+def plan_variant_ids(plan) -> list: ...
+
+ def validate_character_spec(spec_dir, repo_root="."):
+     schema = plan.get("schema_version")
+     if schema and schema != "image-maker.wardrobe-round3.v1":
+         return {"ok": False, "unsupported_schema": schema,
+                 "issues": ["本工具只实现 round3 的 spec 校验…"], ...}
-    template_files = [... 硬编码 round3 模板清单 ...]
+    template_files = plan.get("template_files") or [... round3 默认清单 ...]
```

2. `tools/wardrobe_experiment.py`

```diff
-    round_label = "round3" if "round3" in self.spec_dir else "round2"
-    self.run_id = args.run_id or f"wardrobe-{round_label}-…"
+    slug = plan.get("round_label") or <schema 派生>
+    self.pack_root_label = plan.get("pack_root_label") or os.path.dirname(self.spec_dir)
+    self.run_id = args.run_id or f"wardrobe-{slug}-…"

-        rows[os.path.join("prompts/wardrobe/round3", relative)]      # _spec_source_files
+        rows[os.path.join(self.pack_root_label, relative)]
-        prior_dir = "data/test-result/20261005/wardrobe-pilot-v1"    # _local_model_evidence
+        prior_dirs = plan.get("prior_runs") or [<pilot 目录默认值>]

+# cmd_compose / cmd_prepare：遇到 unsupported_schema 明确判未通过（exit 1），
+#   不再用 round3 的 base-pack 组装路径硬套
+# Experiment.__init__：schema 不受支持时在 `os.makedirs(out_dir)` **之前**就拒绝，
+#   连 run 目录与账本都不建（冒烟测试里曾因此在 data/test-result 下留过两个空目录，
+#   已在本轮清理并加回归用例锁住）
+# main()：在线子命令（--run/--resolver/--review/--calibrate/--generate/--score/--pairs/
+#   --pick-vision-model/--calibrate-recompute）先过 assert_online_supported 守卫
```

3. `tools/wardrobe_report_v2.py`

```diff
+def field_map_for_plan(plan):
+    # 字段集名与字段表来自 plan["report_fields"]（schema 决定 field_set），
+    # 缺失时回落到内建 round3/round2 口径
-    candidates = [spec_dir, os.path.join("prompts", "wardrobe", "round3", "spec")]
+    candidates = [spec_dir] + list(PLAN_CONDITION_CANDIDATES)   # round4/spec 在前
+    # 条件来源必须有人物卡，否则跳过并记录，不再无条件套用 round3
+    sampling_mode = "paired_arms" | "direct_sampling_no_pairing"; waves = {...}
```

## 八、明确尚未执行的阶段

以下**一件都没做**，不要当成已完成：

1. 提取请求、留出核对请求、转化/保留草案**全部未发送**（0 次 HTTP、0 次收费）。
2. 三个系列的**候选衣装包未产生**：现有 `reference-study/{classical,sweet,gothic}.md`
   是人写初稿，草案里也按 `hand_written_draft` 标注；**不能**称 extracted / trained /
   已验证，也不能用它们与后续候选做「孰优」比较。
3. **没有生成任何图片**，第一波/第二波都未跑；因此没有效果评价、没有人工复核结论、
   没有离线评测结果。
4. **看图校准未执行**（`spec/calibration-cases.json` 只登记计划用例与判据，
   `calibration_status: not_run_no_images`）。
5. **在线文本就绪审查未执行**（`templates/review-round4.md` 只有模板）。
6. **在线评分未执行**，也没有据此写任何「通过」。
7. `prompts/wardrobe/round4/spec` 是离线冻结的 spec 骨架（`cli_compatible: false`），
   **不是可执行 spec**：`--compose` / `--prepare` 会明确报 unsupported 并 exit 1，
   `--generate` 会被在线守卫在发请求前拦住（实测退出码 1）。
8. 中式/日式两类的筛样与衣装包未开始（`planning.json#deferred_families`）。
9. 生产接入（预设、GUI、提交快照、历史重跑、衣装审计）**没有动**；
   本轮没有改任何生产模块，所以**没有重启 App**，也没有触发 App 重启要求。

准备任务到此为止，等待用户对在线阶段的执行指示（新额度、通道顺序、是否先只跑第一波）。
