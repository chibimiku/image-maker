# 独立衣装 Round5 生成执行记录（首图效果实验）

本轮只做「跑图 + 落盘 + 记录执行」，**不做效果判定**。所有图都是首图（无重绘、无后处理、无评分、无文本/视觉模型调用、无发布）。
执行者：DSH。判定与看图由 Codex 负责。**「接口成功」不等于「服装效果通过」。**

## 1. 冻结输入与隔离目录

| 项 | 值 |
|---|---|
| 冻结包 | `prompts/wardrobe/round5/generation-pack.json`（schema `image-maker.wardrobe-generation-pack.v1`） |
| 包内槽位 | 28（GPT 14 / Gemini 14），内衣族合计 10 |
| 包自校验摘要 | `a4d77b9719e1ab8bc58d91225c701c37a97156a81ecd2fc9b43ec7ee9003608f` |
| 包文件字节 SHA-256 | `0e255a73a63af258…`（**不是**上面的字段，见 §6 说明） |
| run 目录 | `data/test-result/20261005/wardrobe-round5-r1` |
| 预检 | `--prepare` → `[ready] 28 frozen requests; 0 HTTP; pack a4d77b97…` |
| 生图通道 | 顺序执行，GPT 先、Gemini 后，**未并行**；未重启 app，未中断现有任务 |

冻结包正文、衣装快照、画风参数、参考附件及顺序均按原样派发。**没有改词、换模型、加模板、重新提取、调用评分/文本模型、重绘或发布。**

## 2. 调用与结果总账

| 指标 | 数量 |
|---|---|
| 计划槽位 | 28 |
| 实际落盘派发（`send-budget/slots/` 槽位文件） | **28**（GPT 14 + Gemini 14） |
| 成功（有产物、hash 校验通过） | **25**（GPT 13 / Gemini 12） |
| 失败（占槽位，无产物） | **3**（GPT 1 / Gemini 2） |
| 未知 / 挂起（pending_unknown） | **0** |
| 未尝试 | **0** |
| 自动重试 | 0 次（`IMAGE_MAKER_IMAGE_MAX_RETRIES=0`） |
| 追加授权 | **无**（本轮没有任何授权补跑，见 §5） |
| 文本 / 视觉收费调用 | 0 |
| 实际 HTTP POST 次数 | 30 次（28 次计费槽位 + 2 次内置端点回退，见 §4.2） |

产物：25 个文件，合计 53,331,200 字节。GPT 通道 1024x1536 PNG；Gemini 通道 1696x2528 JPEG（2:3、2K）。
`samples.json` 里 25 条 success 的 `output_sha256` 已全量复核，**0 处不匹配**（文件未被改动）。

时间线（本地时间）：首个请求 2026-10-05 23:59:10 → 最后一个请求结束 2026-10-06 00:25:40。

## 3. 逐槽位结果

| 槽位 | 案例 | 通道 | 衣装族 | 状态 | 发送 | 完成 | 产物文件 |
|---|---|---|---|---|---|---|---|
| F5-GP-gpt | GP | gpt | lolita-classical | success | 23:59:10 | 00:00:30 | F5-GP-gpt_output_000030_0_45ca0b.png |
| F5-GF-gpt | GF | gpt | lolita-gothic | success | 00:00:30 | 00:01:45 | F5-GF-gpt_output_000145_0_207ee9.png |
| F5-LP-gpt | LP | gpt | cute-lingerie | success | 00:01:45 | 00:03:03 | F5-LP-gpt_output_000303_0_973f79.png |
| F5-LF-gpt | LF | gpt | cute-lingerie | success | 00:03:03 | 00:04:17 | F5-LF-gpt_output_000417_0_c9542b.png |
| F5-HA-gpt | HA | gpt | lolita-sweet | success | 00:04:17 | 00:05:27 | F5-HA-gpt_output_000527_0_3bbe44.png |
| F5-LA-gpt | LA | gpt | cute-lingerie | success | 00:05:27 | 00:07:16 | F5-LA-gpt_output_000716_0_5304cb.png |
| F5-UC-gpt | UC | gpt | lolita-sweet | success | 00:07:17 | 00:08:35 | F5-UC-gpt_output_000835_0_990adf.png |
| F5-HC-gpt | HC | gpt | lolita-sweet | success | 00:08:35 | 00:09:55 | F5-HC-gpt_output_000955_0_2d6182.png |
| **F5-LC-gpt** | LC | gpt | cute-lingerie | **failed_no_image** | 00:09:55 | 00:11:09 | —（上游安全拒绝） |
| F5-LU-gpt | LU | gpt | cute-lingerie | success | 00:11:09 | 00:12:15 | F5-LU-gpt_output_001215_0_d20898.png |
| F5-C-waterink-style-image-gpt | C-waterink-style-image | gpt | lolita-sweet | success | 00:12:16 | 00:13:54 | F5-C-waterink-style-image-gpt_output_001354_0_fb9c0c.png |
| F5-C-waterink-style-text-gpt | C-waterink-style-text | gpt | lolita-sweet | success | 00:13:54 | 00:15:39 | F5-C-waterink-style-text-gpt_output_001539_0_1b3ecd.png |
| F5-C-kishida-mel-style-image-gpt | C-kishida-mel-style-image | gpt | lolita-gothic | success | 00:15:39 | 00:17:12 | F5-C-kishida-mel-style-image-gpt_output_001712_0_ab72db.png |
| F5-C-kishida-mel-style-text-gpt | C-kishida-mel-style-text | gpt | lolita-gothic | success | 00:17:12 | 00:18:32 | F5-C-kishida-mel-style-text-gpt_output_001832_0_4e96e1.png |
| F5-GP-gemini | GP | gemini | lolita-classical | success | 00:18:35 | 00:19:38 | F5-GP-gemini_001938-33fa70.jpg |
| F5-GF-gemini | GF | gemini | lolita-gothic | success | 00:19:38 | 00:20:01 | F5-GF-gemini_002001-bb171c.jpg |
| F5-LP-gemini | LP | gemini | cute-lingerie | success | 00:20:01 | 00:20:26 | F5-LP-gemini_002025-1b9d80.jpg |
| F5-LF-gemini | LF | gemini | cute-lingerie | success | 00:20:26 | 00:20:52 | F5-LF-gemini_002052-a80c6a.jpg |
| F5-HA-gemini | HA | gemini | lolita-sweet | success | 00:20:52 | 00:21:14 | F5-HA-gemini_002114-e53fb0.jpg |
| F5-LA-gemini | LA | gemini | cute-lingerie | success | 00:21:14 | 00:21:47 | F5-LA-gemini_002147-e3d8a8.jpg |
| F5-UC-gemini | UC | gemini | lolita-sweet | success | 00:21:47 | 00:22:12 | F5-UC-gemini_002212-ab9602.jpg |
| F5-HC-gemini | HC | gemini | lolita-sweet | success | 00:22:12 | 00:22:36 | F5-HC-gemini_002236-a7f73f.jpg |
| F5-LC-gemini | LC | gemini | cute-lingerie | success | 00:22:36 | 00:23:21 | F5-LC-gemini_002321-da0973.jpg |
| F5-LU-gemini | LU | gemini | cute-lingerie | success | 00:23:21 | 00:24:00 | F5-LU-gemini_002400-c8097e.jpg |
| **F5-C-waterink-style-image-gemini** | C-waterink-style-image | gemini | lolita-sweet | **failed_no_image** | 00:24:00 | 00:24:15 | —（上游内容拦截） |
| **F5-C-waterink-style-text-gemini** | C-waterink-style-text | gemini | lolita-sweet | **failed_no_image** | 00:24:15 | 00:24:30 | —（上游内容拦截） |
| F5-C-kishida-mel-style-image-gemini | C-kishida-mel-style-image | gemini | lolita-gothic | success | 00:24:30 | 00:25:15 | F5-C-kishida-mel-style-image-gemini_002515-f51135.jpg |
| F5-C-kishida-mel-style-text-gemini | C-kishida-mel-style-text | gemini | lolita-gothic | success | 00:25:15 | 00:25:40 | F5-C-kishida-mel-style-text-gemini_002540-fe2432.jpg |

## 4. 失败记录（占槽位，无产物，未自动重试）

### 4.1 失败 ID 与原因

| 失败 ID | 通道 | 时间 | 状态 | 上游原因 | 证据文件 |
|---|---|---|---|---|---|
| F5-LC-gpt | gpt | 00:09:55→00:11:09 | failed_no_image | `moderation_blocked`：`safety_violations=[sexual]`（上游安全系统拒绝，request id 20261006000955385590200lO7OJ8Ya） | `F5-LC-gpt_aigc-2d-gpt_server_raw_001109_2cb44f.txt`、`…_server_response_001109_2a2d37.json` |
| F5-C-waterink-style-image-gemini | gemini | 00:24:00→00:24:15 | failed_no_image | HTTP 400 `CONTENT_BLOCKED`：`content blocked by upstream safety policy`（request id 20261006002400994341090YBGsi2HN） | `F5-C-waterink-style-image-gemini_aigc2d_server_raw_002415_5cbdeb.txt` |
| F5-C-waterink-style-text-gemini | gemini | 00:24:15→00:24:30 | failed_no_image | HTTP 400 `CONTENT_BLOCKED`：`content blocked by upstream safety policy`（request id 20261006002419333288520A4qfMoZn） | `F5-C-waterink-style-text-gemini_aigc2d_server_raw_002430_719c59.txt` |

三点事实，供 Codex 判定：

1. `F5-LC-gpt` 与 `F5-LC-gemini` 是**同一槽位同一正文**的两个通道：Gemini 通道成功、GPT 通道被上游安全系统拒绝。差异只在通道与端点服务端策略，正文未改。
2. `C-waterink` 组的**两个变体在 Gemini 通道同时被拦**（有图附件与纯文字各一次），因此该画风组本轮在 Gemini 通道**没有可用产物**，图/文对照在 Gemini 侧不成立；GPT 通道该组两个变体都成功（见 §4.2）。
3. 这三条失败**没有重发**：按任务书「失败和超时占槽位、不自动重试、不换输出目录重置」执行，也没有超出 28 槽的额度消耗。

### 4.2 GPT 通道两个参考图槽位的端点回退（非模式改动）

`F5-C-waterink-style-image-gpt` 与 `F5-C-kishida-mel-style-image-gpt` 的首次 POST 被上游拒绝：

```
{"error":{"message":"Unknown parameter: 'image'.","type":"invalid_request_error","param":"image","code":"unknown_parameter"}}
```

工具（既有代码，非本轮改动）随即把**同一份冻结请求体**重发到 `/images/edits` 并成功，日志记为「上游不支持「新建图片」模式的参考图字段，已回退到编辑端点重发」。要点：

- 请求正文、画风参数、参考附件与顺序**没有改动**，改的只是上游接受参考图的端点；
- 这是包内 `params.mode=generate` 在当前上游能力下的一次传输级回退，**不是**把它改成别的参考模式；
- 代价：这两槽各多花一次 HTTP POST（故总计 30 次 POST / 28 个计费槽位）；第一次被拒的请求未见计费证据；
- 证据：`F5-C-waterink-style-image-gpt_aigc-2d-gpt_server_response_001218_f2075a.json`（拒绝）与 `…_001354_5ab5b5.json`（回退成功）、`F5-C-kishida-mel-style-image-gpt_aigc-2d-gpt_server_response_001541_e49286.json` 与 `…_001712_acaedd.json`。
- 两次成功产物在 `samples.json` 里各只记 1 个槽位，账本口径未被这次回退放大。

## 5. 追加授权与每次尝试

- 本轮**没有**任何用户追加额度授权，因此没有执行 `--grant-frozen-budget generation`、`--grant-limit`、`--retry-slots` 等补跑路径；`--help` 已核对，当前参数名与任务书一致。
- 每个槽位**恰好一次**尝试：`send-budget/slots/001.json … 028.json` 共 28 个槽位文件，`send-budget/images.jsonl` 中 28 条 `reserved` + 25 条 `success` + 3 条 `failed_no_image`，**没有任何 `#gN` 补跑标签**（该后缀只在拿到追加授权时才会出现）。
- 3 条失败槽位保持失败状态，未被覆盖、未被复用成功产物。

## 6. 证据保留位置（全部原样保留）

| 文件 / 目录 | 内容 |
|---|---|
| `data/test-result/20261005/wardrobe-round5-r1/samples.json` | 28 槽位逐条状态、prompt_sha256、产物路径与 `output_sha256`、起止时间 |
| `data/test-result/20261005/wardrobe-round5-r1/frozen-pack.json` | 本次运行使用的冻结包副本（`sha256` 字段 = `a4d77b97…`） |
| `data/test-result/20261005/wardrobe-round5-r1/send-budget/` | `slots/001–028.json`（每个已派发的槽位各一份）、`images.jsonl`（预留/结算流水） |
| 25 个产物 | GPT 13 张 PNG + Gemini（含对照组）12 张 JPEG，均在 run 目录根下 |
| 服务器原始返回 | `*_aigc-2d-gpt_server_response_*.json`、失败原始体 `*_server_raw_*.txt`、Gemini 拦截原始体 `*_aigc2d_server_raw_*.txt` |

关于摘要口径：包内 `sha256` 字段是**去掉该字段后的规范 JSON 摘要**（`utils/wardrobe_experiment.py` 的自校验口径），不是 pack 文件的字节哈希；两者都记在上表 §1。运行期若包内容被改动，`--prepare/--generate` 会直接报错退出——本轮预检与实际派发均通过，`samples.json.pack_sha256` 与之一致，无「账本冲突」。

未生成 `ledger.json`：该文件只在发生追加授权时创建，本轮没有授权，故不存在（这也与「无追加授权」互为佐证）。

## 7. 本轮边界（不要越读）

- 这是**首图效果实验**，不是完整审计链路，也不是发布：无重绘、无质量/身份审计、无评分、无视觉模型判定、无投稿 JSON。
- 25 个 success 只说明**接口返回并成功落盘**；服装保留/改款/完整转化是否达标，**一律未判**，等 Codex 看图。
- 同一对槽位的 GPT 与 Gemini 图由两个不同模型通道生成，**不可**用「同槽位两图差异」推断通道优劣（还有端点回退的影响，见 §4.2）。
- 人物仍是 32 岁成年女性的单主体、灰底、全身双脚可见设定；**不能**据此宣称多人物/多场景泛化。
- 本轮未重启 app，也未改动任何 Python/提示词/配置；`conf/config.json` 与 `.env` 仅为读取凭据，**密钥未输出**（日志中的 Authorization 已由后端 mask）。
