"""图片分析（Step 1 Vision）的「备用分析端点」。

背景：图片分析的第一选择（默认 `gpt-5.6-luna` 之类）偶尔会**拒绝分析图片** ——
返回 `content_filter`、`refusal`，或者 content 为 None 只丢一句审核话术。这类失败
重试多少次都没用（同一张图必然被同样拒绝），属于「换一个模型就有救」的失败。

这里把「备用端点」做成一层薄封装，挂在 `step_1_analyze_image` 的 Vision 请求外面：

- 配置来源与顶层文本 API 一致：`conf/config.json` 顶层 `fallback_*` 键 + 环境变量覆盖；
- 密钥环境变量 `IMAGE_MAKER_FALLBACK_TEXT_API_KEY`（通用名 `FALLBACK_TEXT_API_KEY` 也认）；
  没配这把时**自动复用 NSFW 文本 API**（`IMAGE_MAKER_NSFW_API_KEY` / `nsfw_*` 键），
  因为本机 NSFW 通道本来就指向 deepseek，省得同一个 key 存两份；
- 只有第一选择**以「被拒 / 审核」方式失败**才切换（`fallback_trigger="refusal"`，默认），
  或配置成 `any` 让任何失败都切过去；
- 备用请求是「同一份请求体换个 base_url/model/key」，不改图片、不改提示词
  （关键：请求体里一切与图片有关的东西都不重新处理，所以两次请求送的是同一张图）。

为什么把「是否允许备用」做成运行期参数 `use_fallback`：GUI 上有个勾选框，
一次分析的整个链路（Step 1~5）都要按同一次勾选状态走，不能在跑的过程中被改配置翻掉。

回归用例：`tests/test_analysis_fallback.py`。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "conf", "config.json")

# 备用端点的密钥环境变量（解析优先级同顶层文本 API：真实环境变量 > .env > 配置文件）
FALLBACK_API_KEY_ENV_NAMES = ("IMAGE_MAKER_FALLBACK_TEXT_API_KEY", "FALLBACK_TEXT_API_KEY")
# 触发条件也可用环境变量钉死（省得界面上被误改）：refusal / any
FALLBACK_TRIGGER_ENV_NAMES = ("IMAGE_MAKER_FALLBACK_TEXT_TRIGGER",)
# 端点与模型也可只写环境变量（配置文件里不必存机器相关地址）
FALLBACK_BASE_URL_ENV_NAMES = ("IMAGE_MAKER_FALLBACK_TEXT_BASE_URL",)
FALLBACK_MODEL_ENV_NAMES = ("IMAGE_MAKER_FALLBACK_TEXT_MODEL",)

# 复用来源（NSFW 文本通道）只保证「是个能聊天的模型」，不保证能读图。
# 实测（2026-09-30，https://api.deepseek.com）：
#   deepseek-flash    → 能正确读图（认得出画面内容）；
#   deepseek-v4-pro   → 明确回 "image is unsupported / no image was provided"。
# 所以复用时会把这几个「已知不看图」的模型换成第一个能看图的备选，并写进日志。
NSFW_REUSE_FALLBACK_MODEL = "deepseek-flash"
VISION_INCAPABLE_MODEL_HINTS = ("v4-pro", "deepseek-v4-pro", "deepseek-reasoner", "deepseek-chat", "deepseek-coder")

TRIGGER_REFUSAL = "refusal"
TRIGGER_ANY = "any"
DEFAULT_TRIGGER = TRIGGER_REFUSAL
# 界面下拉的 (值, 文案)：GUI 与测试共用同一份，避免两处文案漂移
FALLBACK_TRIGGER_CHOICES = (
    (TRIGGER_REFUSAL, "第一选择被拒/审核拦截时（推荐）"),
    (TRIGGER_ANY, "第一选择任何失败时"),
)
# 兼容别名（旧引用）
TRIGGER_CHOICES = FALLBACK_TRIGGER_CHOICES

# 判「被拒/审核」的证据。小写后做子串匹配，中英文都要覆盖：
# 服务端可能回 `Sorry, I can't help with that`，也可能回一句中文审核话术。
REFUSAL_MARKERS = (
    "content_filter",
    "content filter",
    "内容过滤",
    "refusal",
    "拒绝响应",
    "拒绝分析",
    "无法分析",
    "不能分析",
    "无法协助",
    "cannot assist",
    "can't assist",
    "cannot help with",
    "can't help with",
    "i'm unable to",
    "i am unable to",
    "unable to analyze",
    "safety",
    "审核",
    "安全策略",
    "内容政策",
    "违规",
    "prohibited",
    "blocked by",
)

# 「软拒绝」证据：模型**没有报错**（HTTP 200 / finish_reason=stop），只是正文里回了一句
# 拒绝分析的话（或直接不按模板输出）。这类响应能正常解析成 JSON，所以异常路径根本抓不到它 ——
# 只能看结果文本本身。为了不误伤正常结果，要求「命中拒绝话术」且「描述字段里出现中日文字符」
# 两个条件同时成立（`english_description` 约定是英文；正常结果里出现中文才可疑）。
_SOFT_REFUSAL_MARKERS = (
    "无法分析",
    "不能分析",
    "无法協助",
    "无法协助",
    "無法分析",
    "無法協助",
    "未能分析",
    "拒绝分析",
    "拒绝响应",
    "抱歉，我",
    "很抱歉",
    "i can't help with",
    "i cannot help with",
    "i can't assist",
    "i cannot assist",
    "i'm unable to",
    "i am unable to",
    "unable to analyze",
    "unable to analyse",
    "cannot analyze",
    "cannot analyse",
    "can't analyze",
    "can't analyse",
    "content policy",
    "content_filter",
    "as an ai",
    # 占位/空壳结果：模型没分析，却按模板把字段回成 "N/A" 之类
    "not provided",
    "no description",
    "unavailable",
)
# 占位符（"整段描述就是一个 N/A"）：两三个字母的串在正常长描述里也会出现，
# 所以只在文本极短时才算证据 —— 见 `is_refusal_text`
_PLACEHOLDER_MARKERS = ("n/a", "na", "none", "null", "unknown", "未提供", "无")
# CJK 统一表意文字（含扩展 A）与日文假名：用来判断「本该是英文的描述」其实不是英文
_CJK_RANGES = ((0x3040, 0x30FF), (0x31F0, 0x31FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xFF66, 0xFF9D))

_LOG_PREFIX = "[备用分析端点]"


def _log(log_callback, message: str) -> None:
    text = f"{_LOG_PREFIX} {message}"
    if log_callback:
        try:
            log_callback(text)
            return
        except Exception:  # noqa: BLE001 - 日志回调坏掉不该影响分析
            pass
    print(text, flush=True)


def _first_env_value(names):
    """按顺序取第一个非空环境变量，返回 (值, 变量名)。"""
    for name in names:
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value, name
    return "", ""


def mask_secret(value) -> str:
    """日志里只暴露密钥的头尾几字符，其余打码（与项目其它地方口径一致）。"""
    text = str(value or "")
    if not text:
        return "(空)"
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}...{text[-4:]}"


@dataclass
class FallbackConfig:
    """一个备用分析端点。空配置（enabled=False）表示没有可用的备用方案。"""

    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    trigger: str = DEFAULT_TRIGGER
    response_format: bool = True
    # 说明文本字段的读取顺序（界面提示 / 日志用：告诉用户 key 是从哪个变量来的）
    key_source: str = "none"
    base_url_source: str = "config"
    model_source: str = "config"
    source_label: str = "备用方案"
    extra: dict = field(default_factory=dict)

    def normalized(self) -> "FallbackConfig":
        trigger = str(self.trigger or "").strip().lower()
        if trigger not in (TRIGGER_REFUSAL, TRIGGER_ANY):
            trigger = DEFAULT_TRIGGER
        return FallbackConfig(
            enabled=bool(self.enabled),
            base_url=str(self.base_url or "").strip(),
            api_key=str(self.api_key or "").strip(),
            model=str(self.model or "").strip(),
            trigger=trigger,
            response_format=bool(self.response_format),
            key_source=str(self.key_source or "none"),
            base_url_source=str(self.base_url_source or "config"),
            model_source=str(self.model_source or "config"),
            source_label=str(self.source_label or "备用方案"),
            extra=dict(self.extra or {}),
        )

    def missing(self):
        """缺哪些必需字段（用于日志与界面提示）。"""
        missing = []
        if not self.base_url:
            missing.append("base_url")
        if not self.api_key:
            missing.append("api_key")
        if not self.model:
            missing.append("model")
        return missing

    def describe(self) -> str:
        """一句话描述（**不含密钥值**，只含变量名/来源）。"""
        key_part = {
            "env": f"key 来自环境变量（{self.key_source}）",
            "config": "key 来自配置文件",
            "nsfw": "key 借用 NSFW 文本 API 那把",
        }.get(str(self.key_source).split(":")[0], "key 未配置")
        if str(self.key_source).startswith("env:"):
            key_part = f"key 来自环境变量 {str(self.key_source)[4:]}"
        return (
            f"{self.source_label}：{self.model} @ {self.base_url}，{key_part}，"
            f"触发条件={self.trigger}"
        )


def resolve_fallback_api_key(cfg: dict = None) -> str:
    """备用端点密钥：环境变量优先，其次配置里的 `fallback_api_key`。

    与顶层文本/NSFW 的约定一致 —— 配置文件里该键应留空、真实值放 `.env`。
    本机 NSFW 通道本来就指向 deepseek，所以**专用变量没配时会借用 NSFW 那把钥匙**：
    这样「只填端点与模型、不重复存 key」也能直接用。
    """
    value, _name = _first_env_value(FALLBACK_API_KEY_ENV_NAMES)
    if value:
        return value
    configured = str((cfg or {}).get("fallback_api_key") or "").strip()
    if configured:
        return configured
    return str((cfg or {}).get("nsfw_api_key") or "").strip()


def fallback_api_key_source(cfg: dict = None) -> str:
    """备用 key 的来源（env:VAR / config / nsfw / none），只暴露变量名不暴露值。"""
    value, name = _first_env_value(FALLBACK_API_KEY_ENV_NAMES)
    if value:
        return f"env:{name}"
    if str((cfg or {}).get("fallback_api_key") or "").strip():
        return "config"
    return "nsfw" if str((cfg or {}).get("nsfw_api_key") or "").strip() else "none"


def fallback_api_key_env_name() -> str:
    """界面上提示用户「该配哪个变量」用的名字。"""
    return FALLBACK_API_KEY_ENV_NAMES[0]


def resolve_fallback_trigger(cfg: dict = None, default: str = DEFAULT_TRIGGER) -> str:
    """触发条件：环境变量 > 配置 > 默认。非法值一律回落 `refusal`（保守：不乱切）。"""
    value, _name = _first_env_value(FALLBACK_TRIGGER_ENV_NAMES)
    trigger = str(value or (cfg or {}).get("fallback_trigger") or default).strip().lower()
    return trigger if trigger in (TRIGGER_REFUSAL, TRIGGER_ANY) else DEFAULT_TRIGGER


def _vision_capable_reuse_model(model: str) -> str:
    """复用 NSFW 通道时挑一个「能读图」的模型名。

    只看模型名：命中已知不看图的家族（如 `deepseek-v4-pro`）就换成
    `NSFW_REUSE_FALLBACK_MODEL`（实测能读图），其余原样返回。名字不像已知家族时
    不动手（别猜用户自己接的端点）。
    """
    name = str(model or "").strip()
    lowered = name.lower()
    if not name:
        return name
    if any(hint in lowered for hint in ("flash", "vision", "vl", "omni")):
        return name
    if any(hint == lowered or hint in lowered for hint in VISION_INCAPABLE_MODEL_HINTS):
        return NSFW_REUSE_FALLBACK_MODEL
    return name


def load_fallback_config(config_path: Optional[str] = None, allow_nsfw_reuse: bool = True) -> FallbackConfig:
    """读备用端点配置；没有就返回 `enabled=False`。**不抛异常**（读不到按没配处理）。

    两个来源，按顺序取第一个「三要素齐全」的：

    1. `fallback_base_url` / `fallback_model`（+ `fallback_api_key`）
       —— GUI「文本分析 API」页的『备用方案』区块写这三个键；
       端点/模型/密钥三者都可以改用环境变量：
       `IMAGE_MAKER_FALLBACK_TEXT_BASE_URL` / `IMAGE_MAKER_FALLBACK_TEXT_MODEL` /
       `IMAGE_MAKER_FALLBACK_TEXT_API_KEY`；
    2. 顶层 NSFW 文本 API：`nsfw_base_url` / `nsfw_api_key` / `nsfw_model`
       —— 来源 1 没配端点时复用（本机 NSFW 通道指向 DeepSeek，密钥在
       `IMAGE_MAKER_NSFW_API_KEY` 里，等于「备用方案不用额外配 key」）。

    来源 2 是**兜底复用**：来源 1 没填端点时才生效，所以不会跟用户显式配置打架。
    「端点齐全」= base_url + model 有值；key 缺了不算没配（本机约定密钥放 `.env`），
    但真正调用时三要素缺一不可（`FallbackConfig.missing()`）。
    """
    from modules.others.api_backend import apply_secret_env_overrides

    path = config_path or CONFIG_PATH
    data: dict = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            data = loaded
    except Exception:  # noqa: BLE001 - 配置缺失/损坏都按"没有备用方案"处理
        data = {}

    # 只在内存副本上套用环境变量，绝不回写磁盘
    try:
        data = apply_secret_env_overrides(dict(data))
    except Exception:  # noqa: BLE001
        pass

    candidates = [
        {
            # 没写 `fallback_enabled` 时按"填了就算启用"处理（旧配置/手工写的配置不该被判成关闭）
            "enabled": bool(data.get("fallback_enabled", True)),
            "base_url": _first_env_value(FALLBACK_BASE_URL_ENV_NAMES)[0]
            or str(data.get("fallback_base_url") or "").strip(),
            "api_key": resolve_fallback_api_key(data),
            "model": _first_env_value(FALLBACK_MODEL_ENV_NAMES)[0]
            or str(data.get("fallback_model") or "").strip(),
            "trigger": resolve_fallback_trigger(data),
            "response_format": bool(data.get("fallback_response_format", True)),
            "key_source": fallback_api_key_source(data),
            "source_label": "备用方案（文本分析页配置）",
            "extra": {
                "response_effort": str(data.get("fallback_response_effort") or "").strip().lower() or None,
            },
        },
    ]
    if allow_nsfw_reuse:
        nsfw_key = str(data.get("nsfw_api_key") or "").strip()
        nsfw_model = str(data.get("nsfw_model") or "").strip()
        reuse_model = _vision_capable_reuse_model(nsfw_model)
        candidates.append(
            {
                "enabled": True,   # 复用来源不需要额外开关：它本来就是一条可用的文本通道
                "base_url": str(data.get("nsfw_base_url") or "").strip(),
                "api_key": nsfw_key,
                "model": reuse_model,
                "trigger": resolve_fallback_trigger(data),
                "response_format": True,
                "key_source": str(data.get("_nsfw_api_key_source") or ("config" if nsfw_key else "none")),
                "source_label": "备用方案（复用 NSFW 文本 API 通道）",
                "extra": {
                    "response_effort": None,
                    "model_swapped_from": nsfw_model if reuse_model != nsfw_model else "",
                },
            }
        )

    # 「端点齐全」= base_url + model 有值。key 缺了不算"没配"：本机约定是密钥放 `.env`
    # （`IMAGE_MAKER_FALLBACK_TEXT_API_KEY`），配置文件里该键本来就一直空着；
    # 若把"key 空"当成没配置，用户会出现"界面填了端点却怎么都不生效、也不知道该配哪个变量"。
    for candidate in candidates:
        cfg = FallbackConfig(**candidate).normalized()
        if cfg.base_url and cfg.model:
            return cfg
    return FallbackConfig().normalized()


def should_use_fallback(exc: Exception, trigger: str = DEFAULT_TRIGGER) -> bool:
    """该不该把这个异常送到备用端点。`trigger="any"` 时恒为 True。"""
    trigger = str(trigger or DEFAULT_TRIGGER).strip().lower()
    if trigger == TRIGGER_ANY:
        return True
    text = str(exc or "").lower()
    return any(marker in text for marker in REFUSAL_MARKERS)


def is_refusal_error(exc: Exception) -> bool:
    """这个异常是不是「模型/审核拒绝」而不是「对面服务坏了」。"""
    return should_use_fallback(exc, TRIGGER_REFUSAL)


def _preview(text, limit: int = 120) -> str:
    """日志/异常里引用一段文本：压平换行并截断，避免把整段正文糊进队列日志。"""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _has_cjk(text: str) -> bool:
    return any(any(low <= ord(char) <= high for low, high in _CJK_RANGES)
               for char in str(text or ""))


def is_refusal_text(text) -> bool:
    """这段文本像不像「模型拒绝分析这张图」？

    命中的是**模型用自然语言解释自己为什么不做**的情形（`抱歉，我无法分析这张图片`、
    `I can't help with that`、或把字段回成 `N/A` 之类空壳）。调用方只喂"本该是英文"的字段
    （`english_description`）；那里出现中日文字符且篇幅极短，说明模型没按模板分析，
    而是拿母语解释 —— 这是最后一条保险，阈值取很小以免误伤正常结果。
    """
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    if any(marker in lowered for marker in REFUSAL_MARKERS):
        return True
    if any(marker in lowered for marker in _SOFT_REFUSAL_MARKERS):
        return True
    # 占位符单列：`n/a` 这类两三个字母的串在正常长描述里没意义，只按"整段就是一个占位符"判
    if len(lowered) <= 40 and any(placeholder in lowered for placeholder in _PLACEHOLDER_MARKERS):
        return True
    return len(lowered) <= 100 and _has_cjk(lowered)


def _response_choices(response):
    try:
        return list(getattr(response, "choices", None) or [])
    except TypeError:  # noqa: BLE001 - choices 不是可迭代对象时按"取不到"处理
        return []


def _response_refusal_text(response) -> str:
    """把响应里所有「可能是拒绝话术」的文本拼起来（取不到就返回空串）。"""
    parts = []
    for choice in _response_choices(response):
        message = getattr(choice, "message", None)
        parts.append(str(getattr(message, "refusal", "") or ""))
        content = getattr(message, "content", None)
        if content is None:
            continue
        if isinstance(content, str):
            parts.append(content)
            continue
        try:
            for part in content:
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", "")
                parts.append(str(text or ""))
        except TypeError:
            parts.append(str(content))
    return "\n".join(part for part in parts if part)


def _make_client(api_key: str, base_url: str, timeout=None):
    """建备用端点客户端；测试里 monkeypatch 本函数即可完全离线。"""
    from openai import OpenAI

    kwargs: dict = {"api_key": api_key, "base_url": base_url}
    if timeout:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)


def _unsupported_param_error(exc: Exception) -> bool:
    """对面不认某个参数（response_format / reasoning_effort）—— 属于"降级再试一次"的情形。"""
    text = str(exc or "").lower()
    markers = (
        "unsupported",
        "not supported",
        "unknown parameter",
        "unrecognized",
        "invalid parameter",
        "不支持的参数",
        "未支持",
    )
    return any(marker in text for marker in markers)


def _is_timeout_error(exc: Exception) -> bool:
    name = type(exc).__name__
    return name in ("APITimeoutError", "APIConnectionError", "TimeoutError") or isinstance(
        exc, (TimeoutError, ConnectionError)
    )


def _response_metadata(response) -> dict:
    """取响应上的路由信息（new-api 中转站会带 X-Routing-Group），只作日志线索。"""
    meta: dict = {}
    for attr in ("model", "_request_id", "id"):
        value = getattr(response, attr, None)
        if value:
            meta[attr] = value
    return meta


def build_fallback_kwargs(fallback: FallbackConfig, request_kwargs: dict) -> dict:
    """把第一选择的请求体改造成备用端点的请求体（同一张图、同一段提示词）。"""
    fb = fallback.normalized()
    payload = dict(request_kwargs or {})
    payload["model"] = fb.model
    if not fb.response_format:
        payload.pop("response_format", None)
    effort = fb.extra.get("response_effort")
    if effort:
        payload["reasoning_effort"] = effort
    return payload


def call_with_refusal_fallback(
    *,
    call,
    request_kwargs: dict,
    fallback: Optional[FallbackConfig],
    primary_client=None,
    primary_base_url: str = "",
    primary_model: str = "",
    timeout_seconds=None,
    log_callback=None,
    step_label: str = "Vision 请求",
    use_fallback: Optional[bool] = None,
    refusal_text=None,
    call_handles_parsing: bool = False,
):
    """执行第一选择的调用；被拒时改用备用端点**重发同一份请求体**。

    - 第一选择成功 → 直接返回，不碰备用端点（不产生额外费用）。
    - 第一选择失败且不该切 / 没有备用配置 / 本任务禁用备用 → 原样抛出原异常
      （上层 `call_with_retry` 与日志处理完全不用改）。
    - 第一选择被拒且备用端点也失败 → 抛出「原异常 + 备用端点失败原因」的合并异常，
      两个原因都进日志，避免只看到第二条报错而误判成"主端点没试过"。

    **被拒不一定是异常**：`finish_reason=content_filter` / `refusal` 这类拒绝经常是
    HTTP 200 + 正常 body，异常路径根本抓不到。所以调用方可以传 `refusal_text`（一个
    「从响应里取可疑文本」的回调），命中的响应会被当成拒绝处理 —— 也就是说**会真的切到
    备用端点重跑，而不是一路成功返回、最后在解析/落盘那一步炸掉**（那会让队列里这条任务
    明明是"第一选择被拒、备用端点已接住"却标成失败）。

    `use_fallback=None` 表示由配置决定（配置齐全就启用）；显式 `False` 可用于
    「本次分析不切备用」的界面开关。
    `primary_base_url` / `primary_model` 用来识别「备用端点其实是同一个」的情况
    （例如分析 Tab 勾了『使用 nsfw 接口』，而备用方案正是复用 NSFW 通道）——
    那种情况下切换毫无意义，直接抛出原异常。

    `call` 默认是「发请求的裸方法」（如 `client.chat.completions.create`），返回原始响应；
    传 `call_handles_parsing=True` 时表示 `call` 是「发请求 + 解析」的闭包
    （`step_1_analyze_image` 就是这么用的）：它的第一个位置参数是**要用的客户端**，
    备用端点会把自己的客户端传进去，因此两条端点走同一套解析与拒绝判定。
    `refusal_text(response)` 用来从原始响应里取「可疑文本」，解析后会丢掉
    `finish_reason` / `refusal` 这类信息 —— 判定逻辑见 `is_refusal_text`。
    """
    kb = request_kwargs or {}
    primary_error: Optional[Exception] = None
    try:
        response = call(**kb)
    except Exception as exc:  # noqa: BLE001 - 需要按类型判定后再决定
        primary_error = exc
    else:
        if refusal_text is not None:
            # 看结果文本本身：拒绝类响应可能连 exception 都没有
            try:
                text = refusal_text(response)
            except Exception:  # noqa: BLE001 - 判定回调坏掉不该影响正常返回
                text = ""
            if text and is_refusal_text(text):
                primary_error = ValueError(
                    f"{step_label} 第一选择返回了拒绝/空壳响应（{_preview(text)}）")

    if primary_error is None:
        return response

    try:
        fb = (fallback or FallbackConfig()).normalized()
        enabled = fb.enabled if use_fallback is None else bool(use_fallback)
        if not enabled or not fb.base_url or not fb.api_key or not fb.model:
            raise primary_error
        if not should_use_fallback(primary_error, fb.trigger):
            _log(
                log_callback,
                f"{step_label} 第一选择失败（{type(primary_error).__name__}），"
                f"触发条件为「{fb.trigger}」→ 不切备用端点。",
            )
            raise primary_error
        # client 自带的 base_url 比调用方传的字符串更权威（真实客户端一定有）
        client_base_url = str(getattr(primary_client, "base_url", "") or "").strip() or str(primary_base_url or "").strip()
        primary_model_name = str(kb.get("model") or primary_model or "").strip()
        same_endpoint = (
            client_base_url.rstrip("/") == fb.base_url.rstrip("/")
            and primary_model_name == fb.model
        )
        if same_endpoint:
            _log(
                log_callback,
                f"{step_label} 备用端点与第一选择是同一个（{fb.model} @ {fb.base_url}），"
                "切换无意义 → 不重发。",
            )
            raise primary_error

        _log(log_callback, f"⚠️ {step_label} 第一选择被拒/失败：{primary_error}")
        swapped_from = str(fb.extra.get("model_swapped_from") or "").strip()
        if swapped_from:
            _log(
                log_callback,
                f"↪️ 备用来源的模型 {swapped_from} 已知不读图，自动改用能读图的 {fb.model}。",
            )
        _log(log_callback, f"↪️ {step_label} 改用{fb.describe()} 重发同一份请求（同一张图、同一段提示词）")

        # 备用请求怎么发：
        # - `call_handles_parsing=True` → `call` 是「发请求 + 解析」的闭包（`step_1` 的用法）。
        #   第一个位置参数就是**要用的客户端**，所以这里把备用客户端交给它 ——
        #   两条端点于是走同一套解析与拒绝判定，返回口径也一致（都是 `(响应, 解析结果)`）；
        # - 其它情况（`call` 没声明自己处理解析）→ 直接打备用客户端的裸端点，
        #   兼容 `call=client.chat.completions.create` 这类老用法。
        call_is_parse_wrapper = bool(call_handles_parsing)
        client = primary_client
        if call_is_parse_wrapper or client is None or (
                client_base_url and client_base_url.rstrip("/") != fb.base_url.rstrip("/")):
            client = _make_client(fb.api_key, fb.base_url, timeout=timeout_seconds)
        payload = build_fallback_kwargs(fb, kb)
        attempts = [payload]
        if "response_format" in payload:
            attempts.append({k: v for k, v in payload.items() if k != "response_format"})

        last_error: Optional[Exception] = None
        for index, attempt_payload in enumerate(attempts):
            try:
                if call_is_parse_wrapper:
                    response = call(client, **attempt_payload)
                else:
                    response = client.chat.completions.create(**attempt_payload)
            except Exception as fb_error:  # noqa: BLE001
                last_error = fb_error
                if index == 0 and _unsupported_param_error(fb_error):
                    _log(
                        log_callback,
                        f"↪️ 备用端点不接受 response_format（{fb_error}），去掉该参数重试一次。",
                    )
                    continue
                break
            # 备用端点的原始响应也可能本身就是一句拒绝（`call` 是裸请求方法时才看得到）
            if refusal_text is not None and not call_is_parse_wrapper:
                try:
                    text = refusal_text(response)
                except Exception:  # noqa: BLE001
                    text = ""
                if text and is_refusal_text(text):
                    last_error = ValueError(f"备用端点返回拒绝/空壳响应（{_preview(text)}）")
                    _log(log_callback, f"↪️ {step_label} 备用端点返回的不是分析结果：{last_error}")
                    break
            meta = _response_metadata(response[0] if isinstance(response, tuple) and response else response)
            suffix = f"（{', '.join(f'{k}={v}' for k, v in meta.items())}）" if meta else ""
            _log(
                log_callback,
                f"✅ {step_label} 备用端点返回成功：模型 {fb.model}{suffix}",
            )
            return response

        detail = "" if last_error is None else f"{type(last_error).__name__}: {last_error}"
        if last_error is not None and _is_timeout_error(last_error):
            detail = f"备用端点超时/连接失败（{detail}）"
        _log(log_callback, f"❌ {step_label} 备用端点也失败：{detail or '未知原因'}")
        raise RuntimeError(
            f"{step_label} 第一选择失败（{type(primary_error).__name__}: {primary_error}）；"
            f"备用端点 [{fb.model}] @ {fb.base_url} 同样失败（{detail or '未知原因'}）"
        ) from primary_error
    except Exception as exc:  # noqa: BLE001 - 统一用 raise ... from 保留"第一选择"这个根因
        if exc is primary_error:
            raise
        raise exc from primary_error
