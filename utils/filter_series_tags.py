"""
Pixiv 标签第二轮 LLM 筛选

从已清理的标签中进一步移除：
1. 具体作品/系列名称（例：東方Project、Fate/GrandOrder、原神）
2. 具体角色名称（例：初音ミク）

保留：
- 泛指的性别/年龄/类型描述（女の子、男の子、美少女、少女、幼女、お姉さん 等）
- 生物种类泛指（猫、犬、天使、悪魔、エルフ 等）
- 职业/身份泛指（魔法少女、メイド 等）

用法：python utils/filter_series_tags.py --batch-size 50
"""

import os
import sys
import json
import time
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

CACHE_FILE = os.path.join(BASE_DIR, "data", "pixiv_tags_cache.json")
BACKUP_FILE = os.path.join(BASE_DIR, "data", "pixiv_tags_cache.before_series_filter.json")

from modules.others.api_backend import fetch_llm_json


def load_cache() -> list[dict]:
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_backup(tags: list[dict]):
    with open(BACKUP_FILE, "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)
    print(f"已备份到 {BACKUP_FILE}")


def save_result(tags: list[dict]):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)
    print(f"已保存到 {CACHE_FILE}")


def build_batch_prompt(tags: list[dict], start_idx: int, batch_size: int) -> tuple[str, list[str]]:
    batch = tags[start_idx:start_idx + batch_size]
    lines = []
    tag_names = []
    for i, item in enumerate(batch):
        tag = item["tag"]
        en = ", ".join(item.get("en_keywords", [])[:5])
        lines.append(f'  [{i+1}] tag="{tag}" | en_keywords=[{en}]')
        tag_names.append(tag)
    return "\n".join(lines), tag_names


def parse_llm_response(response_text: str) -> dict[str, bool]:
    """解析 LLM 响应，返回 {tag_name: is_generic}"""
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", response_text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], dict):
            data = data["results"]
        result = {}
        for key, val in data.items():
            if isinstance(val, bool):
                result[key] = val
            elif isinstance(val, (int, float)):
                result[key] = bool(val)
        return result
    return {}


def filter_batch(tags: list[dict], start_idx: int, batch_size: int,
                 base_url: str, api_key: str, model: str) -> dict[str, bool]:
    """发送一批标签给 LLM，返回 {tag_name: is_generic(应保留)}"""
    prompt_text, tag_names = build_batch_prompt(tags, start_idx, batch_size)

    system_prompt = """你是一个标签分类助手。请判断以下每个 Pixiv 日文标签是否是「具体作品/系列名称」或「具体角色名称」。

【需要移除的（回答 false）】：
- 具体作品/系列名称：東方Project、Fate/GrandOrder、原神、艦隊これくしょん、鬼滅の刃、ポケモン、VOCALOID、SPY×FAMILY 等
- 具体角色名称：初音ミク、Hatsune Miku 等，只要是具体有名有姓的角色名

【需要保留的（回答 true）】：
- 泛指性别年龄：女の子、男の子、少女、幼女、お姉さん、美少女、ロリ、ショタ 等
- 泛指生物种类：猫、犬、狐、天使、悪魔、エルフ、吸血鬼、人魚、ロボット、ケモノ、獣人、ドラゴン、妖怪 等
- 泛指职业身份：魔法少女、メイド、巫女、シスター、ナース、忍者、バニーガール 等
- 泛指人物特征：ちびキャラ 等
- 以及所有服装、发型、表情、动作、配饰、背景、天气、画风等非角色标签（这些不在判断范围，直接回答 true）

请以 JSON 格式返回，只返回一个 JSON 对象。key 为标签名，value 为 true(保留/泛指) 或 false(移除/具体作品或角色名)。"""

    user_content = f"""请判断以下 {len(tag_names)} 个 Pixiv 标签中，哪些是「具体作品系列名」或「具体角色名」（应移除），哪些是「泛指/通用描述」（应保留）：

{prompt_text}"""

    print(f"  批次 [{start_idx+1}-{start_idx+len(tag_names)}] 发送中...", end=" ")
    response = fetch_llm_json(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_content=user_content,
        temperature=0.1,
    )

    if not response:
        print("无响应")
        return {}

    results = parse_llm_response(response)
    keep_count = sum(1 for v in results.values() if v)
    remove_count = sum(1 for v in results.values() if not v)
    print(f"返回 {len(results)} 个结果（保留 {keep_count}, 移除 {remove_count}）")
    return results


def load_api_config() -> tuple[str, str, str]:
    config_path = os.path.join(BASE_DIR, "conf", "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["nsfw_base_url"], config["nsfw_api_key"], config["nsfw_model"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    tags = load_cache()
    print(f"已加载 {len(tags)} 个标签")

    save_backup(tags)

    base_url, api_key, model = load_api_config()
    print(f"API: model={model}")

    total = len(tags)
    all_results: dict[str, bool] = {}

    for start_idx in range(args.start, total, args.batch_size):
        batch_results = filter_batch(tags, start_idx, args.batch_size,
                                     base_url, api_key, model)
        all_results.update(batch_results)
        if start_idx + args.batch_size < total:
            time.sleep(1.0)

    kept = []
    removed = []
    for item in tags:
        tag = item["tag"]
        if tag not in all_results:
            kept.append(item)  # 未返回的保守保留
        elif all_results[tag]:
            kept.append(item)
        else:
            removed.append(tag)

    print(f"\n=== 筛选结果 ===")
    print(f"保留: {len(kept)} 个")
    print(f"移除（作品系列/具体角色）: {len(removed)} 个")

    if removed:
        print(f"\n被移除的标签:")
        for tag in removed:
            print(f"  - {tag}")

    save_result(kept)

    from collections import Counter
    cat_counts = Counter(item["category"] for item in kept)
    print(f"\n保留标签分类统计:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
