# -*- coding: utf-8 -*-
"""把 config-styles.json 里每个画风的完整说明书转换成 gpt-image 通道用的「短版字段式说明」
（`prompt_gpt`），并固化回同一个文件。

转换规则（与 utils/style_gpt.py、prompts/style-gpt-convert-*.md 一致）：
- 只保留 8 个字段：Palette / Lighting / Brushwork / Edges / Texture / Composition density /
  Detail level / Avoid，每行一个，值 3~14 词，整份 150~480 字符；
- **只描述画法**，不许出现主体/角色/发色/瞳色/姿势/服装/场景词（校验会直接拒收并重试）；
- Avoid 只写"不该像什么"的渲染层反模式。

用法:
  python tools/convert_styles_gpt.py                     # 为缺少 prompt_gpt 的画风补全
  python tools/convert_styles_gpt.py --force             # 全部重转
  python tools/convert_styles_gpt.py --only tid,ajicoma  # 只转指定画风
  python tools/convert_styles_gpt.py --dry-run           # 只打印将要发出的请求，不调用模型
  python tools/convert_styles_gpt.py --check             # 只校验已有 prompt_gpt 是否合规
"""
import argparse
import json
import os
import sys
import time

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.others.api_backend import resolve_text_api_key  # noqa: E402
from utils.styles import ref_image_valid, style_ref_image  # noqa: E402
from utils.style_gpt import (  # noqa: E402
    MAX_CHARS,
    build_conversion_prompts,
    format_prompt_gpt,
    parse_fields,
    repair_prompt_gpt,
    repair_prompt_gpt_lenient,
    to_data_url,
    validate_prompt_gpt,
)

CONFIG_TEXT_FILE = os.path.join(PROJECT_ROOT, "conf", "config.json")
CONFIG_STYLES_FILE = os.path.join(PROJECT_ROOT, "conf", "config-styles.json")
DEFAULT_STATS_FILE = os.path.join(PROJECT_ROOT, "cache", "temp", "style-ref-stats.json")
FALLBACK_BASE_URL = "https://new.aigc2d.com/v1"
DEFAULT_TIMEOUT = 300
NETWORK_BACKOFF_S = (3, 8, 15)


def load_text_config():
    if not os.path.isfile(CONFIG_TEXT_FILE):
        return {}
    with open(CONFIG_TEXT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_chat_base(base_url: str) -> str:
    """把 conf 里各种形态的 base_url 归一成可拼 /chat/completions 的地址。"""
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        return ""
    # 文本节点历史上填过 Gemini 风格的 .../v1beta/models
    for tail in ("/v1beta/models", "/chat/completions"):
        if url.endswith(tail):
            url = url[: -len(tail)]
            break
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def call_text_model(base_url, api_key, model, system_prompt, user_prompt, timeout=DEFAULT_TIMEOUT,
                    max_tokens=4000, image_path=""):
    """直接 POST /chat/completions：支持推理模型（max_completion_tokens），可选附图（多模态）。"""
    url = f"{normalize_chat_base(base_url)}/chat/completions"
    if image_path:
        content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": to_data_url(image_path)}},
        ]
    else:
        content = user_prompt
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "max_completion_tokens": max_tokens,
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(str(data["error"])[:300])
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("响应没有 choices")
    return str((choices[0].get("message") or {}).get("content") or "").strip()


def convert_one(base_url, api_key, model, name, style_text, attempts=3, timeout=DEFAULT_TIMEOUT,
                image_path="", image_stats_text=""):
    """返回 (prompt_gpt, errors)；全部尝试失败时 prompt_gpt 为空。"""
    system_prompt, user_prompt = build_conversion_prompts(
        name, style_text, has_image=bool(image_path), image_stats_text=image_stats_text
    )
    last_errors = []
    for attempt in range(1, attempts + 1):
        try:
            raw = call_text_model(base_url, api_key, model, system_prompt, user_prompt,
                                  timeout=timeout, image_path=image_path)
        except Exception as exc:  # noqa: BLE001
            last_errors = [f"{type(exc).__name__}: {exc}"]
            print(f"  [warn] 第 {attempt} 次请求失败: {last_errors[0]}")
            time.sleep(NETWORK_BACKOFF_S[min(attempt - 1, len(NETWORK_BACKOFF_S) - 1)])
            continue
        candidate = repair_prompt_gpt(raw)
        ok, errors = validate_prompt_gpt(candidate)
        if not ok:
            # 长度类问题可以本地压缩兜底，其它问题（缺字段/主体词）交给重试
            lenient = repair_prompt_gpt_lenient(raw)
            if lenient and lenient != candidate:
                ok2, errors2 = validate_prompt_gpt(lenient)
                if ok2:
                    return lenient, []
        if ok:
            return candidate, []
        last_errors = errors
        print(f"  [warn] 第 {attempt} 次不合规: {'; '.join(errors[:4])}")
        # 把上一次的错误回灌给模型，提升下一次命中率
        user_prompt = (
            user_prompt
            + "\n\nYour previous answer was rejected: "
            + "; ".join(errors[:4])
            + "\nAnswer again with ONLY the 8 field lines, strictly following the rules."
        )
        time.sleep(1)
    return "", last_errors


def load_ref_stats(path: str) -> dict:
    """读 tools/style_ref_stats.py 产出的统计（缺失时返回空，走纯文本转换）。"""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def ensure_ref_stats(stats_path: str, styles: dict, names) -> dict:
    """统计文件不存在时现场补算（本地 OpenCV，秒级）。"""
    stats = load_ref_stats(stats_path)
    missing = [n for n in names if n not in stats]
    if not missing:
        return stats
    try:
        import importlib.util

        module_path = os.path.join(PROJECT_ROOT, "tools", "style_ref_stats.py")
        spec = importlib.util.spec_from_file_location("style_ref_stats", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 无法载入 style_ref_stats（改为纯文本转换）: {exc}")
        return stats
    for name in missing:
        ref = style_ref_image(styles, name)
        if not ref_image_valid(ref):
            continue
        stats[name] = module.stats_for(ref)
    if stats:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"[info] 已补算 {len(missing)} 个画风的参考图统计 -> {stats_path}")
    return stats


def stats_text_for(stats: dict, name: str) -> str:
    """把统计 dict 转成给模型看的一行描述。"""
    st = (stats or {}).get(name)
    if not st:
        return ""
    try:
        import importlib.util

        module_path = os.path.join(PROJECT_ROOT, "tools", "style_ref_stats.py")
        spec = importlib.util.spec_from_file_location("style_ref_stats", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.to_text(name, st)
    except Exception:  # noqa: BLE001
        return ""


def save_styles(path, styles):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(styles, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)


def check_styles(styles, only=None):
    from utils.style_gpt import FIELD_KEYS, parse_fields, resolve_style_clauses, validate_prompt_gpt

    bad = 0
    clause_stats = {"entry": [], "derived": [], "none": []}
    for name, entry in styles.items():
        if only is not None and name not in only:
            continue
        gpt_text = ""
        full_text = ""
        if isinstance(entry, dict):
            gpt_text = str(entry.get("prompt_gpt") or "")
            full_text = str(entry.get("prompt") or "")
        elif isinstance(entry, str):
            full_text = entry
        if not gpt_text.strip() and not full_text.strip():
            print(f"[n/a ] {name:24s}    -      画风本身为空（无附加画风的占位条目）")
            continue
        ok, errors = validate_prompt_gpt(gpt_text)
        mark = "OK  " if ok else "BAD "
        print(f"[{mark}] {name:24s} {len(gpt_text):>4} 字符" + ("" if ok else "  " + "; ".join(errors[:3])))
        if not ok:
            bad += 1
        clauses, source = resolve_style_clauses(entry if isinstance(entry, dict) else {})
        clause_stats[source].append(name)
    print(f"\n不合规 {bad} 个")
    print(f"渲染语言条款：手写 {len(clause_stats['entry'])} / 派生 {len(clause_stats['derived'])} / "
          f"无 {len(clause_stats['none'])}")
    for label, key in (("手写（画风条目 repaint_clauses）", "entry"), ("按 prompt_gpt 派生", "derived")):
        if clause_stats[key]:
            print(f"  {label}: " + ", ".join(sorted(clause_stats[key])[:12])
                  + (" …" if len(clause_stats[key]) > 12 else ""))
    if clause_stats["none"]:
        print("  无条款（连 prompt_gpt 都没有，首图/重绘都没有渲染语言约束）: "
              + ", ".join(sorted(clause_stats["none"])))
    # 手写条款与字段版说明来自同一批模型输出，字段变了条款可能已经过时
    for name in sorted(clause_stats["entry"]):
        entry = styles.get(name) or {}
        fields = parse_fields(str(entry.get("prompt_gpt") or ""))
        missing = [k for k in FIELD_KEYS if k not in fields]
        if missing:
            print(f"  ⚠ {name}: 有手写条款但 prompt_gpt 缺字段 {missing}，条款可能与说明不一致")
    return bad


def main():
    parser = argparse.ArgumentParser(description="生成/刷新画风的 gpt-image 短版说明（prompt_gpt）")
    parser.add_argument("--force", action="store_true", help="已有 prompt_gpt 也重新转换")
    parser.add_argument("--only", default="", help="只处理这些画风，逗号分隔")
    parser.add_argument("--dry-run", action="store_true", help="只打印请求，不调用模型")
    parser.add_argument("--check", action="store_true", help="只校验现有 prompt_gpt，不改文件")
    parser.add_argument("--no-image", action="store_true", help="不把画风参考图发给文本模型（默认会发，以图为准）")
    parser.add_argument("--stats-only", action="store_true", help="只把参考图的客观统计作为依据（不发图，更快更稳；默认行为）")
    parser.add_argument("--stats-file", default=DEFAULT_STATS_FILE, help="参考图统计缓存文件")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="单次请求超时秒数")
    args = parser.parse_args()

    with open(CONFIG_STYLES_FILE, "r", encoding="utf-8") as f:
        styles = json.load(f)

    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None

    if args.check:
        return 1 if check_styles(styles, only) else 0

    text_cfg = load_text_config()
    base_url = normalize_chat_base(text_cfg.get("base_url") or FALLBACK_BASE_URL)
    api_key = resolve_text_api_key(text_cfg)
    model = str(text_cfg.get("model") or "").strip()
    if not args.dry_run and not (base_url and api_key and model):
        print("[error] 缺少文本 API 配置（conf/config.json 的 base_url/model + IMAGE_MAKER_TEXT_API_KEY）")
        return 1

    changed, skipped, failed, stale = 0, 0, [], []
    todo = [n for n in styles if (only is None or n in only)]
    stats = ensure_ref_stats(args.stats_file, styles, todo)
    for name in list(styles.keys()):
        entry = styles[name]
        if isinstance(entry, str):
            entry = {"prompt": entry}
            styles[name] = entry
        prompt_text = str(entry.get("prompt") or "").strip()
        if only is not None and name not in only:
            continue
        if not prompt_text:
            print(f"[skip] {name}: 说明书为空")
            skipped += 1
            continue
        if entry.get("prompt_gpt") and not args.force:
            ok, _ = validate_prompt_gpt(entry["prompt_gpt"])
            if ok:
                print(f"[keep] {name}: 已有合规 prompt_gpt（--force 可重转）")
                skipped += 1
                continue
            print(f"[redo] {name}: 已有 prompt_gpt 不合规，重转")
        stats_text = stats_text_for(stats, name)
        ref_path = str(entry.get("ref_image") or "") if isinstance(entry, dict) else ""
        use_image = (not args.stats_only) and (not args.no_image) and ref_image_valid(ref_path)
        if args.dry_run:
            system_prompt, user_prompt = build_conversion_prompts(
                name, prompt_text, has_image=use_image, image_stats_text=stats_text
            )
            print(f"\n===== DRY-RUN {name}（说明书 {len(prompt_text)} 字符；依据="
                  f"{'统计' if stats_text else ('图' if use_image else '纯文')}）")
            print("SYSTEM:", system_prompt.splitlines()[0], "...")
            print("USER 头:", user_prompt.splitlines()[0])
            print("USER 依据:", (stats_text[:160] + "...") if stats_text else "(无)")
            continue

        basis = "统计" if stats_text else ("图+文" if use_image else "纯文")
        print(f"[run ] {name}: 说明书 {len(prompt_text)} 字符 / 依据 {basis} -> ", end="", flush=True)
        result, errors = convert_one(
            base_url, api_key, model, name, prompt_text, timeout=args.timeout,
            image_path=ref_path if use_image else "", image_stats_text=stats_text,
        )
        if result:
            entry["prompt_gpt"] = result
            changed += 1
            print(f"{len(result)} 字符 OK")
            if entry.get("repaint_clauses"):
                stale.append(name)
                print(f"       ⚠ {name} 有手写 repaint_clauses（{len(entry['repaint_clauses'])} 条），"
                      f"prompt_gpt 已改 → 请复核条款是否还成立（首图与重绘都会用它）")
        else:
            print("失败: " + "; ".join(errors[:3]))
            failed.append(name)
        time.sleep(0.3)

    if args.dry_run:
        print("\nDRY-RUN 结束，未修改文件")
        return 0

    if changed:
        save_styles(CONFIG_STYLES_FILE, styles)
        print(f"\n已更新 {changed} 个画风 -> {CONFIG_STYLES_FILE}")
    else:
        print("\n无变更")
    print(f"跳过/保留: {skipped}；目标长度 {MAX_CHARS} 字符以内")
    if failed:
        print("失败画风:", ", ".join(failed))
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
