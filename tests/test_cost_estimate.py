# -*- coding: utf-8 -*-
"""提示词长度护栏与成本估算（utils/cost_estimate.py）。"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from utils import cost_estimate as ce  # noqa: E402


def test_prompt_budget_soft_and_hard():
    b = ce.prompt_budget(1325)
    assert b["total_chars"] == 1325 and not b["over_soft"] and not b["over_hard"]
    assert b["remaining_soft"] == ce.PROMPT_SOFT_LIMIT - 1325
    warn = ce.prompt_budget(2500)
    assert warn["over_soft"] and not warn["over_hard"]
    over = ce.prompt_budget(16000)
    assert over["over_hard"] and over["remaining_hard"] == 0
    text = ce.format_budget(over)
    assert "已超硬上限" in text


def test_prompt_budget_counts_style_text():
    b = ce.prompt_budget(1200, style_chars=543)
    assert b["total_chars"] == 1743 and b["style_chars"] == 543


def test_gpt_image_token_cost_matches_measured_usage():
    """按实测 token 用量核对公式：medium 1024x1536 + 1 张参考图 ≈ $0.039。"""
    pricing = {"group_ratio": {"Openai-Gpt-1": 0.88236}, "models": dict(ce.FALLBACK_PRICING)}
    low = ce.estimate_gpt_image_cost(1325, "1024x1536", "low", ref_images=1, pricing=pricing)
    med = ce.estimate_gpt_image_cost(1325, "1024x1536", "medium", ref_images=1, pricing=pricing)
    high = ce.estimate_gpt_image_cost(1325, "1024x1536", "high", ref_images=1, pricing=pricing)
    assert low["usd"] < med["usd"] < high["usd"]
    assert 0.03 <= med["usd"] <= 0.05
    assert med["text_tokens"] == 415 and med["image_input_tokens"] == 1105
    assert med["output_tokens"] == 1105
    # 不挂参考图更便宜
    no_ref = ce.estimate_gpt_image_cost(1325, "1024x1536", "medium", ref_images=0, pricing=pricing)
    assert no_ref["usd"] < med["usd"]


def test_long_prompt_cost_growth_is_small():
    """文本长度对成本影响很小（真正的风险是功能性的，不是钱）。"""
    pricing = {"group_ratio": {"Openai-Gpt-1": 0.88236}, "models": dict(ce.FALLBACK_PRICING)}
    a = ce.estimate_gpt_image_cost(1300, "1024x1536", "medium", ref_images=1, pricing=pricing)
    b = ce.estimate_gpt_image_cost(11300, "1024x1536", "medium", ref_images=1, pricing=pricing)
    assert b["usd"] - a["usd"] < a["usd"]  # 差值不足一倍


def test_per_call_models_use_model_price():
    pricing = {"group_ratio": {"Discounted-Banana-1": 0.110295}, "models": dict(ce.FALLBACK_PRICING)}
    rp = ce.estimate_gemini_repaint_cost(pricing=pricing)
    assert 0.03 <= rp["usd"] <= 0.04          # 0.33 × 0.110295
    assert rp["billed"] == "per_call"
    twice = ce.estimate_gemini_repaint_cost(repeat=2, pricing=pricing)
    assert abs(twice["usd"] - rp["usd"] * 2) < 1e-9


def test_pipeline_breakdown_totals():
    pricing = {"group_ratio": {"Openai-Gpt-1": 0.88236, "Discounted-Banana-1": 0.110295},
               "models": dict(ce.FALLBACK_PRICING)}
    est = ce.estimate_pipeline(1325, quality="medium", include_analysis=True, pricing=pricing)
    keys = [s["key"] for s in est["steps"]]
    assert keys == ["generate", "repaint", "structure", "local", "analysis"]
    structure = next(s for s in est["steps"] if s["key"] == "structure")
    assert structure["usd"] == 0.0            # 结构线叠加是纯本地
    assert abs(est["total_usd"] - sum(s["usd"] for s in est["steps"])) < 1e-9
    assert est["total_cny"] == est["total_usd"] * est["cny_rate"]
    assert 0.10 <= est["total_usd"] <= 0.16
    # 关掉后处理只剩下首图
    only_first = ce.estimate_pipeline(1325, include_repaint=False, include_structure=False,
                                      include_local=False, pricing=pricing)
    assert len(only_first["steps"]) == 1


def test_format_pipeline_cost_contains_steps():
    pricing = {"group_ratio": {"Openai-Gpt-1": 0.88236, "Discounted-Banana-1": 0.110295},
               "models": dict(ce.FALLBACK_PRICING)}
    est = ce.estimate_pipeline(1200, quality="high", pricing=pricing)
    text = ce.format_pipeline_cost(est)
    assert "单张全工序估算" in text and "$" in text
    assert "首图" in text


def test_fallback_pricing_covers_our_models():
    for name in ("gpt-image-2", "gemini-3-pro-image-preview", "gpt-image-2-c"):
        entry = ce.model_price_entry(name, pricing={"models": {}})
        assert entry, name
