import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List

from openai import OpenAI


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
INPUT_CSV_PATH = os.path.join(BASE_DIR, "data", "tags", "danbooru_e621_merged.csv")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "data", "tags", "danbooru_e621_merged_zh.csv")
PROGRESS_PATH = os.path.join(BASE_DIR, "data", "tags", "danbooru_e621_merged_zh.progress.json")


SYSTEM_PROMPT = (
    "你是 booru tag 翻译助手。"
    "把英文 tag 翻译成简洁准确的中文，适合 UI 展示。"
    "不要输出解释。"
    "必须严格返回 JSON 对象，键为原始英文 tag，值为中文翻译。"
    "若无法翻译则保留原词。"
)


def load_config(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "base_url": str(cfg.get("base_url", "")).strip(),
        "api_key": str(cfg.get("api_key", "")).strip(),
        "model": str(cfg.get("model", "")).strip() or "gpt-4o-mini",
    }


def load_input_tags(path: str) -> List[str]:
    tags = []
    seen = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            tag = str(row[0]).strip().lower()
            if not tag or tag in seen:
                continue
            seen.add(tag)
            tags.append(tag)
    return tags


def load_existing_translations(path: str) -> Dict[str, str]:
    result = {}
    if not os.path.isfile(path):
        return result
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            en = str(row[0]).strip().lower()
            zh = str(row[1]).strip()
            if en and zh:
                result[en] = zh
    return result


def load_progress(path: str) -> Dict[str, int]:
    if not os.path.isfile(path):
        return {"last_index": -1}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"last_index": int(data.get("last_index", -1))}
    except Exception:
        return {"last_index": -1}


def save_progress(path: str, last_index: int):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"last_index": int(last_index), "updated_at": int(time.time())}, f, ensure_ascii=False, indent=2)


def save_translations(path: str, data: Dict[str, str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    items = sorted(data.items(), key=lambda x: x[0])
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for en, zh in items:
            writer.writerow([en, zh])


def build_user_prompt(batch_tags: List[str]) -> str:
    return (
        "请把下面 booru tags 翻译为中文，返回 JSON 对象。"
        "仅返回 JSON，不要 markdown，不要代码块。\n\n"
        + json.dumps(batch_tags, ensure_ascii=False)
    )


def translate_batch(
    client: OpenAI,
    model: str,
    batch_tags: List[str],
    request_timeout: int = 90,
    max_retries: int = 3,
) -> Dict[str, str]:
    user_prompt = build_user_prompt(batch_tags)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                timeout=request_timeout,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            result = {}
            if isinstance(parsed, dict):
                for tag in batch_tags:
                    zh = parsed.get(tag)
                    if isinstance(zh, str) and zh.strip():
                        result[tag] = zh.strip()
                    else:
                        result[tag] = tag
                return result
            last_err = f"响应不是 JSON 对象: {type(parsed)}"
        except Exception as e:
            last_err = str(e)
        time.sleep(1.5 * attempt)
    raise RuntimeError(f"翻译批次失败: {last_err}")


def chunk_list(data: List[str], size: int):
    for i in range(0, len(data), size):
        yield i, data[i : i + size]


def main():
    parser = argparse.ArgumentParser(description="调用模型接口翻译 booru tag CSV")
    parser.add_argument("--batch-size", type=int, default=20, help="每次请求翻译多少个 tag")
    parser.add_argument("--save-every", type=int, default=10, help="每多少个批次落盘一次")
    parser.add_argument("--max-batches", type=int, default=0, help="最多处理多少批次，0 表示不限")
    parser.add_argument("--model", type=str, default="", help="覆盖 config.json 的模型名")
    parser.add_argument("--request-timeout", type=int, default=90, help="单次请求超时秒数")
    parser.add_argument("--reset", action="store_true", help="重置进度并从头开始")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    if args.model.strip():
        cfg["model"] = args.model.strip()
    if not cfg["api_key"] or not cfg["base_url"] or not cfg["model"]:
        raise RuntimeError("config.json 缺少 base_url / api_key / model")

    print("加载输入标签中...")
    all_tags = load_input_tags(INPUT_CSV_PATH)
    print(f"输入标签总数: {len(all_tags)}")

    existing = {} if args.reset else load_existing_translations(OUTPUT_CSV_PATH)
    progress = {"last_index": -1} if args.reset else load_progress(PROGRESS_PATH)
    start_index = int(progress.get("last_index", -1)) + 1
    if start_index < 0:
        start_index = 0

    print(f"已存在翻译数: {len(existing)}")
    print(f"从索引 {start_index} 开始处理")

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    done_batches = 0
    processed = 0

    for absolute_start, batch_tags in chunk_list(all_tags, args.batch_size):
        absolute_end = absolute_start + len(batch_tags) - 1
        if absolute_start < start_index:
            continue

        missing = [t for t in batch_tags if t not in existing]
        if missing:
            translated = translate_batch(
                client,
                cfg["model"],
                missing,
                request_timeout=max(10, int(args.request_timeout)),
            )
            existing.update(translated)
        processed += len(batch_tags)
        done_batches += 1

        save_progress(PROGRESS_PATH, absolute_end)
        if done_batches % max(1, args.save_every) == 0:
            save_translations(OUTPUT_CSV_PATH, existing)

        print(
            f"批次 {done_batches} 完成 | 索引 {absolute_start}-{absolute_end} | "
            f"累计处理 {processed} | 当前翻译总数 {len(existing)}"
        )

        if args.max_batches > 0 and done_batches >= args.max_batches:
            print("达到 max-batches，提前结束。")
            break

    save_translations(OUTPUT_CSV_PATH, existing)
    print("全部完成，结果已写入:")
    print(OUTPUT_CSV_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"运行失败: {e}", file=sys.stderr)
        sys.exit(1)
