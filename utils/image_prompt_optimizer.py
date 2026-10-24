"""Text-only prompt expansion shared by the Gemini and GPT image tabs."""
import json

from openai import OpenAI

from utils.analysis_gpt_prompt import load_text_api_config
from utils.analysis_fallback import call_with_refusal_fallback, is_refusal_text, load_fallback_config
from utils.prompt_loader import read_prompt_file


def parse_optimized_prompt(response):
    choices = list(getattr(response, "choices", None) or [])
    if not choices:
        raise ValueError("提示词优化返回空响应")
    choice = choices[0]
    message = choice.message
    text = str(getattr(message, "content", None) or "").strip()
    if (getattr(choice, "finish_reason", "") == "content_filter"
            or getattr(message, "refusal", None) or not text or is_refusal_text(text)):
        raise ValueError("提示词优化 content_filter/refusal: 返回拒绝或空内容")
    if getattr(choice, "finish_reason", "") == "length":
        raise ValueError("提示词优化响应被截断，保留原文")
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text).get("optimized_prompt")
    except (ValueError, AttributeError) as exc:
        raise ValueError("提示词优化返回的 JSON 无效，保留原文") from exc
    if not isinstance(value, str) or not value.strip():
        raise ValueError("提示词优化缺少 optimized_prompt，保留原文")
    value = value.strip()
    if is_refusal_text(value):
        raise ValueError("提示词优化 refusal: 返回拒绝话术")
    if len(value) > 2000:
        raise ValueError("优化结果超过 2000 字符，保留原文")
    return value


def optimize_image_prompt(brief, context=None, config_path=None, log_callback=None):
    brief = str(brief or "").strip()
    if not brief:
        raise ValueError("请先输入要优化的提示词")
    cfg = load_text_api_config(config_path)
    if not all(cfg.get(k) for k in ("base_url", "model", "api_key")):
        raise ValueError("请先配置文字分析 API 的地址、模型和密钥")
    fallback = load_fallback_config(config_path, require_vision=False)
    if log_callback:
        log_callback(f"提示词优化：文字分析模型 {cfg['model']}；只发送文字与界面选项")
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=120, max_retries=0)
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": read_prompt_file("image-prompt-optimize-system.md")},
            {"role": "user", "content": json.dumps({"brief": brief, "context": context or {}}, ensure_ascii=False)},
        ],
        "max_completion_tokens": 3000,
    }

    def call(selected_client=None, **kwargs):
        return parse_optimized_prompt((selected_client or client).chat.completions.create(**kwargs))

    try:
        return call_with_refusal_fallback(
            call=call, request_kwargs=payload, fallback=fallback, primary_client=client,
            primary_base_url=cfg["base_url"], primary_model=cfg["model"],
            timeout_seconds=120, log_callback=log_callback or (lambda _message: None), step_label="提示词优化",
            call_handles_parsing=True,
        )
    finally:
        client.close()
