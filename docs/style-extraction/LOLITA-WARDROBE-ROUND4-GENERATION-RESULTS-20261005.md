# Round4 衣装生图执行结果（20261005）

本文件只记录**生图调用的执行账本**：哪些槽位真的发了请求、哪些出了图、哪些失败。
**「生成成功」只表示接口返回了图片并落盘，不等于服装效果通过**——衣装是否符合预期
（古典/甜美/哥特/中式/日式的结构、保留类别的改款/只补全、画风并发的一致性）
尚未评审，需由 Codex 看图后另行判定。本报告不做视觉评分、不筛样、不改模板、不发布。

## 执行环境

| 项 | 值 |
|---|---|
| 冻结包 | `prompts/wardrobe/round4/generation-pack.json`（schema `image-maker.wardrobe-generation-pack.v1`） |
| 包 SHA-256 | `fb8c24faf1f57bdf15b21ff78d7d1de1b606bda7c20b2bfb585cdd92facd8f70` |
| 输出目录（run-dir） | `data/test-result/20261005/wardrobe-round4-r1` |
| Python | `C:\Program Files\Python310\python.exe`（系统 Python，工作目录 `D:\code\image-maker`） |
| 入口 | `tools/wardrobe_experiment.py --frozen-pack … --prepare / --generate --channel … / --status` |
| 端点与凭据 | 沿用 `conf/config.json` + `.env` 现有配置（`aigc-2d-gpt` / `aigc2d`）；密钥未打印、未复制、未写出 |
| 模型 | GPT 通道 `gpt-image-2`（1024x1536，quality=high，png）；Gemini 通道 `gemini-3-pro-image-preview`（2K，aspect 2:3） |
| 隐藏重试 | `IMAGE_MAKER_IMAGE_MAX_RETRIES=0`（一次尝试 = 一次 HTTP） |

## 实际调用数

| 通道 | 冻结槽位 | 首轮生图 HTTP | 授权补跑 HTTP | 生图 HTTP 合计 | 文本 | 视觉评价 | 模型列表 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT | 20 | 20 | 0 | 20 | 0 | 0 | 0 |
| Gemini | 20 | 20 | 2 | 22 | 0 | 0 | 0 |
| 合计 | 40 | 40 | **2** | **42** | **0** | **0** | **0** |

- 零 HTTP 预检 `--prepare` 退出码 0（`40 frozen requests; 0 HTTP`），未产生任何请求。
- 发送前原子占用预算：`send-budget/slots/` 共 42 个槽位文件，`send-budget/images.jsonl`
  记录了 42 次 `reserved` + 42 次 `settled`（含失败，失败**不退还**额度）。
- 两条首轮 `--generate` 命令**顺序执行**（未并行）；GPT 通道退出码 0，Gemini 通道退出码 1。
- 追加的 2 次调用来自**用户显式授权**（见下方「补跑」一节），不是自动重试。

## 结果统计（含补跑后）

| 状态 | GPT | Gemini | 合计 |
|---|---:|---:|---:|
| 成功（图片已落盘，hash 与样本状态一致） | 20 | 19 | **39** |
| 失败（`failed_no_image`，无图片） | 0 | 1 | **1** |
| 未发送（本次未尝试） | 0 | 0 | **0** |
| 未知 / pending（原因不明的挂起槽位） | 0 | 0 | **0** |
| 合计 | 20 | 20 | **40** |

- **成功 39 / 失败 1 / 未发送 0 / 未知 0**；40 个槽位全部尝试完毕，目录内实有 **39 张图片**。
- 唯一未出图的槽位在两次尝试中都遭遇**上游内容安全拦截**（HTTP 400），
  不是超时、不是本地错误、不是额度冲突；按规则未再自动重发。
- 失败槽位在 `samples.json` 中保留 `status=failed_no_image`、空 `saved_files`、
  空 `output_sha256`，上一次尝试归档在 `attempts`，原始响应体留在同目录
  `*_aigc2d_server_raw_*.txt`。

### 补跑（用户显式追加授权）

首轮结束额度为 40/40 用尽。用户明确指示「你补充失败的2图即可 添加到报告里」，
按任务书「用户主动追加授权是有效授权」处理：

| 项 | 值 |
|---|---|
| 追加额度 | `generation_http_attempts` 40 → **42**（只抬上限，不重置已消耗，不换 run-dir） |
| 授权记录 | `ledger.json` 的 `budget_grants[0]`（授权人 `user`，含理由与对话来源原话） |
| 补跑范围 | `--retry-slots F4-H-sweet-gemini-waterink-style,F4-R-gothic-gemini-kishida-mel-style`（恰好 2 个失败槽位） |
| 重发标签 | `<slot>#g1`（避开 `send_budget` 对同一 run_id 的重复派发拦截，两次尝试在流水里可分辨） |
| 成功产物 | 18 张首轮成功图**全部按文件 hash 复用**，未重发 |
| 补跑结果 | 1 张成功、1 张再次被上游拦截 |

补跑产出（已追加进报告与本目录）：

| 槽位 | 补跑结果 | 文件 | 字节 | SHA-256 |
|---|---|---|---:|---|
| `F4-H-sweet-gemini-waterink-style` | 成功 | `F4-H-sweet-gemini-waterink-style_223555-d8c967.jpg` | 2,348,936 | `9ae8eaff04c79830e7302419c0882e9204b4f71fd8d3cff469eb78065c287227` |
| `F4-R-gothic-gemini-kishida-mel-style` | 再次失败（HTTP 400 `CONTENT_BLOCKED`） | — | — | — |

补跑后仍失败的槽位**不再继续重发**：同一份正文连续两次被同一策略拦截，
继续重发既无新信息又继续消耗额度，需要新授权或改正文（改正文属于 Codex 的范围）。

### 失败槽位明细（按通道）

| 槽位 sample_id | 通道 | 案例 / 衣装 / 画风 | 第 1 次尝试 | 第 2 次尝试（授权补跑） |
|---|---|---|---|---|
| `F4-R-gothic-gemini-kishida-mel-style` | gemini | R（制服改款）+ 哥特 + `kishida-mel-style` | 22:27:22 HTTP 400 `CONTENT_BLOCKED`：`[CONTENT ERROR] content blocked by upstream safety policy`（request id `20261005222722426526826DUmFX3LN`） | 22:35:55 HTTP 400 `CONTENT_BLOCKED`（request id `2026100522355555009366434V9vNcf`） |
| `F4-H-sweet-gemini-waterink-style` | gemini | H（hoodie 改款）+ 甜美 + `waterink-style` | 22:27:18 HTTP 400 `PROHIBITED_CONTENT`：`request blocked by (PROHIBITED_CONTENT): content is prohibited under official usage policies.`（code `request_body_blocked`，request id `20261005222719244004057pNR5Uby9`） | 22:35:16 **成功出图**（38.5 秒） |

- 两个失败槽位是**仅有的带参考图的 Gemini 槽位**（各 1 张艺术画风参考图，`image_paths` 非空）。
  其中 `waterink-style` 槽位补跑即成功，说明该参考图组合并不必然触发拦截；
  仍失败的 `kishida-mel-style` 槽位在两次相同正文下都被拦截。
- 同内容的 GPT 通道对应槽位（`F4-H-sweet-gpt-waterink-style`、`F4-R-gothic-gpt-kishida-mel-style`）
  两次都正常出图。每条件 n=1，**不足以据此断言拦截与参考图存在因果关系**。

- 其余 38 个槽位首轮无失败、无超时、无截断。
- GPT 通道 20 张：`/images/generations` 18 张 + `/images/edits`（带 1 张画风参考图）2 张，单张耗时约 31~142 秒。
- Gemini 通道 19 张：单张耗时约 20~39 秒。
- 上游对两张 GPT 产物返回了 `output_tokens=1105` 的短用量记录（对应带参考图的槽位），
  其余为 `5488`；两者均正常出图并落盘，本报告不据此评价质量。


## 交付保留内容

`data/test-result/20261005/wardrobe-round4-r1/` 下保留：

- 39 张成功图片（20 张 `.png` GPT + 19 张 `.jpg` Gemini，文件名以 `sample_id` 开头）。
- `frozen-pack.json`：本次执行的冻结包快照（含包 SHA-256 与逐条请求正文、`prompt_sha256`）；
  若目录已属于另一个冻结包，入口会拒绝执行，因此该快照可证明产物与包一一对应。
- `samples.json`：40 个槽位状态账本（`success` / `failed_no_image`），含逐槽 `prompt_sha256`、
  `output_sha256`、开始/结束时间；补跑槽位另有 `rerun` / `rerun_label` / `rerun_granted_at`
  与 `attempts`（上一次失败归档）。`.pre-rebuild` 备份是重建那一条归档前的原样副本。
- `send-budget/`：`slots/`（42 个原子占位）+ `images.jsonl`（reserved/settled 事件流水）。
- `ledger.json`：追加额度的授权记录（上限、旧值、授权人、理由、来源、时间）。
- 附带的 `*_server_response_*.json` / `*_aigc2d_server_raw_*.txt` 为逐次调用的原始响应留档。
- 后端生成的 `*_replay_*.json` / `*.py` 请求回放文件（内含未掩码凭据）已被入口逐次删除，
  目录中不含任何密钥副本。

## 本次为补跑所做的入口改动（供复核）

冻结包入口原本**没有任何补跑路径**：额度硬等于包里的槽位数（40 已在首轮用尽），
且 `send_budget` 拒绝同一 `run_id` 二次派发——即已失败槽位无法在不绕过账本的前提下重发。
经用户授权后，在 `tools/wardrobe_experiment.py` 的冻结包分支增加：

1. `--grant-frozen-budget generation --grant-limit N`：必须同时给出 `--authorized-by` /
   `--grant-reason` / `--grant-source`，否则拒绝；上限只抬不降、已消耗不动、run-dir 不换，
   授权写进 `ledger.json`。重复执行同一条授权命令是幂等的（不重复登记、不重复发送）。
2. `--retry-slots`：只补指定 sample_id；不填则补所选通道里所有已失败槽位。
   成功产物仍按文件 hash 复用，绝不重发。
3. 重发使用 `<slot>#gN` 标签，两次尝试在 `images.jsonl` 里可分辨。
4. 新建 `ledger.json` 的上限基线取**冻结包槽位数**（不是任务书默认 32），
   并按 `send-budget/slots/` 的实际发送次数对账已消耗。
5. 修复一处记账缺陷：补跑时上一次的失败记录会被新记录覆盖，导致归档丢失；
   现改为显式带入 `attempts`。

回归验证（离线，未联网）：临时 run 目录上确认「补跑前失败槽位被重新派发／成功槽位未重发／
上一次失败被归档／重发标签带授权代次／上限=包槽位数+追加／重复执行幂等」
全部通过；`tests/test_wardrobe.py` 与 `tests/test_wardrobe_experiment.py` 共 **80 项通过**。
（全量 `python -m pytest -q` 另有 3 项失败，均在 `single_analyzer` 相关用例，
来自仓库内**其他未提交改动**：产品新增 `meta_wardrobe`，而用例仍传 `Mock`；
与本轮冻结包入口改动无关。）


## 复用与续跑口径

- 成功产物按文件 hash 复用：重跑同一条命令只会打印 `[reuse/skip]`，不重发。
- 已失败或未知槽位在**默认口径**下再次运行会被跳过，不自动重发。
- 补跑必须走用户显式授权（`--grant-frozen-budget generation --grant-limit N` +
  `--authorized-by/--grant-reason/--grant-source`），保持本轮账本与输出目录不变，
  不更换输出目录、不重置账本。
- 追加授权属于用户主动决定，不是程序故障。
- 当前额度已用尽（42/42，含 2 次授权补跑）；`F4-R-gothic-gemini-kishida-mel-style`
  仍无图，若还要再试必须再次取得用户明确授权，或先修改正文（后者属于 Codex 的范围）。
- 本次未执行文本审计、自动重绘、身份修订或发布——这是检查首图效果的实验模式。

## 结论边界

- 本报告能证明的：40 个槽位各调用一次，另经用户授权追加 2 次补跑，
  39 张图片成功落盘且 hash 与账本一致，1 张因上游内容安全策略两次失败并如实记账。
- 本报告**不能**证明的：服装效果是否达标、五家族是否各自可辨识、
  「完整转化／改款保留类别／只补全」是否符合预期、画风并发是否与衣装冲突。
  这些必须由 Codex 看图后判定，未评审前不得写成「服装效果通过」。
- 本轮每条件 n=1（补跑槽位 n=2），即使评审通过也只能用于筛查效果，不能据此宣称统计优势。
