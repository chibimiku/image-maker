# 图片分析「备用方案」（第一选择拒绝分析时改走第二个端点）

> 2026-09-30 实施并在线复验；2026-10-01 修「拒绝不是异常」这个漏判。实现在
> `utils/analysis_fallback.py`，回归用例 `tests/test_analysis_fallback.py`（22 条）+
> `tests/test_analysis_fallback_status.py`（9 条，专门管工序状态）。

## 1. 解决什么问题

图片分析的第一步（Step 1，Vision 请求）用的第一选择模型（本机默认 `gpt-5.6-luna`）
偶尔会**拒绝分析图片**。三种典型表现，都发生在 Step 1 拿到响应之后、解析之前：

| 表现 | 在响应里的样子 | 现有代码的判定 |
|---|---|---|
| 服务端内容过滤 | `finish_reason = "content_filter"` | `_safe_json_from_response` 抛「服务端内容过滤」 |
| 模型拒绝 | `choice.message.refusal` 有值 | 抛「响应 content 为 None (refusal: …)」 |
| 空返回 | `content = None`，`finish_reason = "stop"` | 同上（refusal 可能为空，只有 finish_reason 线索） |

这类失败**重试没有意义**：同一张图必然被同样拒绝，`text_retry_*` 那套（429/5xx/超时）
只会浪费 5 分钟 × 5 次。唯一出路是换一个模型 —— 这就是「备用方案」。

## 2. 工作方式

```
Step 1 Vision 请求
  └─ call_with_retry(...)                       ← 429/5xx/超时 仍按原配置重试
       └─ call_with_refusal_fallback(...)       ← 新增的一层
            ① 第一选择返回「可用结果」→ 直接返回（不产生任何额外请求、不额外计费）
            ② 失败但不是「被拒」→ 原样抛出（交给上面的重试 / 报错）
            ③ 被拒 → 用**同一份请求体**改发给备用端点
                 · 同一张图（同一个 data URI，不重新压缩、不改 detail）
                 · 同一段提示词（同一个 system + user 消息）
                 · 只换 base_url / model / key
            ④ 备用端点也失败 → 抛「两边原因合并」的异常，两个失败都进日志
```

### 2.1 「被拒」不一定是异常（2026-10-01 修正，很关键）

`content_filter` / `refusal` / `content=None` 这类拒绝**通常是 HTTP 200 + 正常 body**，
异常路径根本抓不到它。最初的实现只在 `call()` 抛异常时才切备用端点，于是：

1. 第一选择返回一个「正常但内容是被拒」的响应 → 不切备用端点；
2. 响应继续往上传，在解析那一步（`_safe_json_from_response`）抛错；
3. 队列里这条任务被置位成**失败**——明明备用方案配好了、图片也没有任何问题。

现在把**解析纳入这一层的判定范围**（`step_1_analyze_image` 把「发请求 + 解析 JSON」做成
一个闭包传进来，见 `call_handles_parsing`），于是：

- 拒绝响应在备用判定之前就被识别为「被拒」，**真的会切到备用端点重跑**；
- 备用端点用同一个闭包、但换它自己的客户端（`call(client, **kwargs)`），
  所以**两条端点走同一套解析与拒绝判定**，返回口径一致。

配套的第二个判定是**软拒绝**（`is_refusal_text`）：模型完全没报错、`finish_reason=stop`，
只是把 `english_description` 回成 `抱歉，我无法分析这张图片。` / `I can't help with that` /
`N/A` 之类。这类响应 JSON 合法、能正常解析，只能看结果文本：

- 命中拒绝/审核话术，或命中 `not provided` / `unavailable` 之类占位词 → 算拒绝；
- 「本该是英文的描述」里出现中日文字符且篇幅极短（≤100 字符）→ 算拒绝（最后一道保险，
  阈值取很小以免误伤正常结果）。

只要切成备用端点并且备用端点接住了，**工序状态就是 `success`**（队列标绿）；
只有「第一选择被拒 + 备用端点也失败」才标红为「被拒(未分析)」或「失败」。

要点：

- **触发条件可配**（`fallback_trigger`）：
  - `refusal`（默认）：只在「被拒 / 审核拦截」时切。判据是异常文本里的
    中英文关键词（`content_filter` / `refusal` / `审核` / `cannot help with` / `safety` …）。
  - `any`：第一选择任何失败都切（含超时、断连、429 —— 注意重试会先跑完）。
- **同一端点不重发**：如果备用端点与第一选择是同一个模型+同一个 base（例如分析 Tab 勾了
  「使用 nsfw 接口」而备用方案正好复用 NSFW 通道），直接抛出原异常，不做无意义的重发。
- **备用端点不认 `response_format` 时自动降级**：去掉该参数再试一次（有些兼容端点不支持
  `json_object`），而不是直接失败。
- **第一选择被拒时队列标红为「被拒(未分析)」**，与「超时」「失败」区分开，日志里一眼可辨。
  **但只有在备用端点也没接住时才标红**：备用端点成功解析时，这条任务的状态是 `success`，
  日志里是「✅ 备用端点返回成功」。
- **开关是运行期状态**：`WorkerThread.use_fallback` 在线程创建时定型，跑到一半改配置不会翻掉
  本次分析。分析 Tab 的勾选框只影响之后提交的任务。

## 3. 配置

### 3.1 界面

「设置 → 文本分析 API」页，失败重试那几行的下方是**备用方案**区块：

| 控件 | 对应配置键 | 说明 |
|---|---|---|
| 备用方案（勾选框） | `fallback_enabled` | 关掉则第一选择被拒时直接失败 |
| 备用 Base URL | `fallback_base_url` | 留空则自动复用「文本分析（NSFW）」的 base |
| 备用分析模型 | `fallback_model` | 必须是**能读图**的模型（见 §4） |
| 备用 API Key | `fallback_api_key` | 按项目约定留空，值放 `.env`；界面只显示变量名 |
| 备用触发条件 | `fallback_trigger` | `refusal` / `any` |
| 提示行 | — | 实时显示「生效端点 / key 来源 / 触发条件」 |

分析 Tab 的 nsfw 那一行有一个紧凑勾选框「被拒时用备用方案」，与设置页共用同一份配置（勾
或取消会写回设置页并落盘），**不新增常驻行**（布局契约见 `AGENTS.md` §2）。

### 3.2 环境变量

| 变量 | 作用 | 优先级 |
|---|---|---|
| `IMAGE_MAKER_FALLBACK_TEXT_API_KEY`（通用名 `FALLBACK_TEXT_API_KEY`） | 备用端点密钥 | 最高 |
| `IMAGE_MAKER_FALLBACK_TEXT_BASE_URL` | 备用端点 base（配置文件里不必存机器地址） | 高于 `fallback_base_url` |
| `IMAGE_MAKER_FALLBACK_TEXT_MODEL` | 备用模型 | 高于 `fallback_model` |
| `IMAGE_MAKER_FALLBACK_TEXT_TRIGGER` | 触发条件（`refusal` / `any`） | 高于 `fallback_trigger` |

密钥解析顺序（`resolve_fallback_api_key`）：
**专用环境变量 → `fallback_api_key` → 借用 `IMAGE_MAKER_NSFW_API_KEY`（NSFW 通道）**。

所以本机**最少配置**是「填好端点与模型，key 一个都不用新加」：NSFW 通道本来就指向
`https://api.deepseek.com`、密钥在 `IMAGE_MAKER_NSFW_API_KEY` 里。想给备用方案一把
独立的钥匙，再加 `IMAGE_MAKER_FALLBACK_TEXT_API_KEY` 即可。

### 3.3 本机当前落盘值

`conf/config.json`（备份 `conf/config.json.bak-fallback`）：

```json
{
  "fallback_enabled": true,
  "fallback_base_url": "https://api.deepseek.com",
  "fallback_model": "deepseek-flash",
  "fallback_api_key": "",
  "fallback_trigger": "refusal"
}
```

## 4. 模型选择（实测记录，2026-09-30）

`GET https://api.deepseek.com/v1/models` 该端点只有两个模型，同一张 512×512 测试图、
同一种 OpenAI `image_url` data URI 传法：

| 模型 | 结果 |
|---|---|
| `deepseek-flash` | ✅ 能读图：正确描述画面内容与画法，返回结构化 JSON |
| `deepseek-v4-pro` | ❌ 回 `"The provided image is unsupported or unavailable…"`（不看图） |

由此有一条兜底规则：**复用 NSFW 通道时，若该通道的模型名命中已知「不读图」家族
（`v4-pro` / `reasoner` / `chat` / `coder` 等），自动改用 `deepseek-flash` 并在日志里说明**
（`_vision_capable_reuse_model`）。名字里含 `flash` / `vision` / `vl` / `omni` 的一律原样使用，
其他自定义模型名不猜、不动。

## 5. 在线复验（2026-09-30 01:43）

强制第一选择抛 refusal，跑 `step_1_analyze_image`：

```
[备用分析端点] ⚠️ Step 1 Vision 请求 第一选择被拒/失败：Step 1 响应 content 为 None
              (refusal: 'I cannot help with that.', finish_reason: content_filter)
[备用分析端点] ↪️ Step 1 Vision 请求 改用备用方案（文本分析页配置）：deepseek-flash @
              https://api.deepseek.com，key 借用 NSFW 文本 API 那把，触发条件=refusal
              重发同一份请求（同一张图、同一段提示词）
[备用分析端点] ✅ Step 1 Vision 请求 备用端点返回成功：模型 deepseek-flash
```

产出结构化结果（`chinese_title` / `japanese_title` / `english_description` /
`pixiv_tags` / `booru-tags` 全部有值），日志与结果快照在
`cache/temp/fallback-live-result.json`（临时验证产物，可删）。

## 6. 已知边界

- 备用端点替换的是 **Step 1**。Step 2~5 仍用界面配置的第一选择；第一选择连续拒绝时
  这几步会照旧失败（Step 2 失败不阻塞出图，Step 3/4 是可选工序）。
  需要整条链都走备用端点时，应把「文本分析 API」本身切到该端点，而不是依赖本机制。
- 触发判定基于异常文本 + 响应文本里的关键词。若某个端点用完全不带这些词的措辞表达拒绝，
  需要把 `should_use_fallback` / `REFUSAL_MARKERS` / `_SOFT_REFUSAL_MARKERS` 补一条，
  或把触发条件设成 `any`。
- **拒绝响应如果连备用端点都不接**（两边都被拒），任务仍然是红的——
  这是正常的：换模型也救不回来，需要人工改图或换输入。
- 备用端点也会计费：只在第一选择被拒时才发生，成功路径零额外请求。

## 7. 回归用例

| 文件 | 覆盖 |
|---|---|
| `tests/test_analysis_fallback.py` | 触发判定、配置解析（含环境变量与 NSFW 复用）、三个分支、同一端点不重发、Step 1 端到端 |
| `tests/test_analysis_fallback_status.py` | **拒绝走"正常返回"也必须切备用**、软拒绝判定、备用端点回拒绝不算成功、`WorkerThread.last_status` 必须是 `success` / 只有两边都失败才 `refused` |
