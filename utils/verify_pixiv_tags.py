"""
Pixiv 标签批量 LLM 验证脚本

作用：将本地缓存的 Pixiv 标签分批次发给大模型，验证每个标签是否描述画面视觉内容。
剔除以下类型的标签：
1. 元信息标签：オリジナル（原创）、創作、イラスト、ファンアート 等
2. 收藏/互动数据标签：xxx人收藏、xxxusers入り 等
3. 不直接描述画面可见内容的标签

保留：服装、发型发色、表情动作、背景环境、配饰道具、身体特征、画风技法等描述画面内容的标签。

用法：python utils/verify_pixiv_tags.py [--batch-size 50] [--dry-run]
"""

import os
import sys
import json
import time
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

CACHE_FILE = os.path.join(BASE_DIR, "data", "pixiv_tags_cache.json")
BACKUP_FILE = os.path.join(BASE_DIR, "data", "pixiv_tags_cache.backup.json")
RESULT_FILE = os.path.join(BASE_DIR, "data", "pixiv_tags_cache.verified.json")

# 延迟导入，避免循环依赖
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


def build_tag_batch_for_prompt(tags: list[dict], start_idx: int, batch_size: int) -> tuple[str, list[str]]:
    """构建一批标签的 prompt 文本，返回 (prompt_text, tag_names_list)"""
    batch = tags[start_idx:start_idx + batch_size]
    lines = []
    tag_names = []
    for i, item in enumerate(batch):
        tag = item["tag"]
        en = ", ".join(item.get("en_keywords", [])[:5])
        cat = item.get("category", "other")
        lines.append(f'  [{i+1}] tag="{tag}" | en_keywords=[{en}] | category="{cat}"')
        tag_names.append(tag)
    return "\n".join(lines), tag_names


def parse_llm_response(response_text: str, expected_count: int) -> dict[str, bool]:
    """
    解析 LLM 返回的 JSON 响应。
    期望格式: {"tag_name": true/false, ...} 或 {"results": {"tag_name": true, ...}}
    true = 描述画面内容，false = 不描述/元信息
    """
    # 尝试直接解析
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # 尝试提取 JSON 片段
        match = re.search(r"\{[\s\S]*\}", response_text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                print(f"  [WARN] 无法解析 LLM 响应为 JSON，前200字符: {response_text[:200]}")
                return {}
        else:
            print(f"  [WARN] 响应中未找到 JSON 对象")
            return {}

    # 查找 results
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], dict):
            data = data["results"]
        # 过滤出布尔值映射
        result = {}
        for key, val in data.items():
            if isinstance(val, bool):
                result[key] = val
            elif isinstance(val, (int, float)):
                result[key] = bool(val)
        return result
    return {}


def verify_batch_llm(tags: list[dict], start_idx: int, batch_size: int, 
                     base_url: str, api_key: str, model: str) -> dict[str, bool]:
    """发送一批标签给 LLM 验证，返回 {tag_name: is_visual} 的映射"""
    prompt_text, tag_names = build_tag_batch_for_prompt(tags, start_idx, batch_size)
    
    system_prompt = """你是一个标签分类助手。以下是一批 Pixiv 的日文标签列表。
请判断每个标签是否直接描述「画面的视觉内容」。

【画面视觉内容】包括但不限于：
- 人物特征：性别年龄、发型发色、瞳色、体型、服饰、妆容
- 服装配饰：制服、和服、泳装、帽子、眼镜、首饰、武器道具
- 动作姿势：站姿、坐姿、跑、跳、拥抱、挥手
- 表情状态：微笑、哭泣、愤怒、惊讶
- 背景环境：室内室外、天空、海洋、教室、森林、城市
- 季节天气：春夏秋冬、雨雪、夕阳、夜空
- 画风技法：水彩、厚涂、素描、水墨、像素风、线稿
- 光影色彩：逆光、月光、霓虹、单色调
- 具体作品名/角色名：Fate、初音ミク、鬼滅の刃等（虽然描述的是作品归属，但直接关联画面人物）

【非画面视觉内容】包括但不限于：
- 创作来源：オリジナル(原创)、創作、ファンアート(同人)、リクエスト、模写
- 平台数据：xxxusers入り、xxx人收藏、R-18、閲覧注意
- 元描述：イラスト(插画)、漫画、落書き、4コマ
- 纯互动信息：記念絵、リクエスト、支援絵、お祝い

请以 JSON 格式返回，只返回一个 JSON 对象，key 为标签名，value 为 true(描述画面内容) 或 false(不描述/元信息)。
例如：{"制服": true, "オリジナル": false}"""

    user_content = f"""请判断以下 {len(tag_names)} 个 Pixiv 标签是否描述画面视觉内容：

{prompt_text}"""

    print(f"  批次 [{start_idx+1}-{start_idx+len(tag_names)}] 发送中...", end=" ")
    response = fetch_llm_json(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_content=user_content,
        temperature=0.1,  # 低温度以获得更确定性的结果
    )
    
    if not response:
        print("无响应")
        return {}
    
    results = parse_llm_response(response, len(tag_names))
    visual_count = sum(1 for v in results.values() if v)
    print(f"返回 {len(results)} 个结果（{visual_count} visual, {len(results)-visual_count} non-visual）")
    return results


def load_api_config() -> tuple[str, str, str]:
    """从 config.json 加载 DeepSeek v4 文本 API 配置"""
    config_path = os.path.join(BASE_DIR, "conf", "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    base_url = config.get("nsfw_base_url", "https://api.deepseek.com")
    api_key = config.get("nsfw_api_key", "")
    model = config.get("nsfw_model", "deepseek-v4-pro")

    if not api_key:
        raise RuntimeError("未找到 deepseek API key，请检查 conf/config.json 中的 nsfw_api_key")

    return base_url, api_key, model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pixiv 标签批量 LLM 验证")
    parser.add_argument("--batch-size", type=int, default=50, help="每批发送的标签数")
    parser.add_argument("--dry-run", action="store_true", help="仅打印计划，不实际发送请求")
    parser.add_argument("--start", type=int, default=0, help="从第几个标签开始（用于断点续传）")
    args = parser.parse_args()
    
    tags = load_cache()
    print(f"已加载 {len(tags)} 个标签")
    
    # 备份原文件
    save_backup(tags)
    
    if args.dry_run:
        # Dry run：只打印分批计划
        total_batches = (len(tags) + args.batch_size - 1) // args.batch_size
        print(f"\n[dry-run] 计划分 {total_batches} 批发送：")
        for b in range(total_batches):
            start = b * args.batch_size
            end = min(start + args.batch_size, len(tags))
            print(f"  批次 {b+1}: 标签 #{start+1} ~ #{end} ({end-start} 个)")
        # 打印一些样例标签
        print("\n例（前10个标签）：")
        for i, item in enumerate(tags[:10]):
            print(f"  {item['tag']} [{item['category']}]")
        return
    
    # 加载 API 配置
    try:
        base_url, api_key, model = load_api_config()
        print(f"API: base_url={base_url[:40]}..., model={model}")
    except Exception as e:
        print(f"加载 API 配置失败: {e}")
        print("请检查 conf/config-image.json")
        sys.exit(1)
    
    # 分批发送
    total = len(tags)
    all_results: dict[str, bool] = {}
    
    for start_idx in range(args.start, total, args.batch_size):
        batch_results = verify_batch_llm(
            tags, start_idx, args.batch_size,
            base_url, api_key, model
        )
        all_results.update(batch_results)
        
        # 每批次之间短暂延迟，避免触发限流
        if start_idx + args.batch_size < total:
            time.sleep(1.0)
    
    # 汇总
    kept = []
    removed = []
    missed = []
    
    for item in tags:
        tag = item["tag"]
        if tag not in all_results:
            missed.append(tag)
            kept.append(item)  # 未分类的保守保留
        elif all_results[tag]:
            kept.append(item)
        else:
            removed.append(tag)
    
    print(f"\n=== 验证结果 ===")
    print(f"保留（视觉内容标签）: {len(kept)} 个")
    print(f"移除（非视觉/元信息）: {len(removed)} 个")
    print(f"未被 LLM 返回（保守保留）: {len(missed)} 个")
    
    if removed:
        print(f"\n被移除的标签 ({len(removed)}):")
        for tag in removed:
            print(f"  - {tag}")
    
    # 保存
    save_result(kept)
    
    # 按分类统计
    from collections import Counter
    cat_counts = Counter(item["category"] for item in kept)
    print(f"\n保留标签分类统计:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
