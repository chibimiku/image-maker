"""
Pixiv 标签快速本地匹配器

工作原理：
1. 接收 WD14 tagger 对图片的预测结果（英文 booru tags）
2. 与本地缓存的 Pixiv 热门标签的 en_keywords 做匹配
3. 返回候选的 Pixiv 日文标签列表

匹配策略（刻意宽松，精度要求低，速度快）：
- 完全匹配：WD14 tag 与 en_keyword 完全一致 → 最高分
- 子串匹配：WD14 tag 包含 en_keyword 或反之 → 中等分
- 未匹配：不返回

性能目标：纯 Python 字典/集合操作，无模型推理，< 0.1 秒

日志开关：
    设置环境变量 PIXIV_TAG_MATCH_DEBUG=1 可开启匹细节日志。
    或在代码中设置 logging.getLogger("pixiv_tag_matcher").setLevel(logging.DEBUG)
"""

import os
import json
import math
import logging
from typing import Optional

# ---- 日志配置 ----
_logger = logging.getLogger("pixiv_tag_matcher")
# 支持环境变量和手动设置两种方式开启 debug
if os.environ.get("PIXIV_TAG_MATCH_DEBUG", "").strip() in ("1", "true", "yes", "on"):
    _logger.setLevel(logging.DEBUG)
    if not _logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        _logger.addHandler(_h)

# ---- 常量 ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = os.path.join(BASE_DIR, "data", "pixiv_tags_cache.json")
MAX_RESULTS = 15  # 最多返回的候选标签数


def _fmt_count(count: int) -> str:
    """将标签作品数格式化为可读字符串，如 8.5M / 800K / 500"""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.0f}K"
    return str(count)


class PixivTagMatcher:
    """Pixiv 标签匹配器，单例模式，首次初始化后常驻内存"""

    _instance: Optional["PixivTagMatcher"] = None

    def __init__(self):
        self._tags: list[dict] = []                   # 完整标签列表
        self._tag_by_name: dict[str, dict] = {}        # tag名 → 标签对象
        self._exact_index: dict[str, list[str]] = {}   # en_keyword(小写) → [tag名列表]
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "PixivTagMatcher":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, force_reload: bool = False):
        """加载缓存的 Pixiv 标签并构建索引"""
        if self._loaded and not force_reload:
            return

        if not os.path.exists(CACHE_FILE):
            # 缓存不存在，先触发抓取
            from utils.pixiv_tag_scraper import scrape_pixiv_tags
            scrape_pixiv_tags(force_refresh=True)

        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                self._tags = json.load(f)
        except Exception:
            self._tags = []

        self._build_index()
        self._loaded = True

    def _build_index(self):
        """构建精确匹配索引"""
        self._tag_by_name = {}
        self._exact_index = {}

        for item in self._tags:
            tag = item["tag"]
            self._tag_by_name[tag] = item

            for keyword in item.get("en_keywords", []):
                kw = keyword.strip().lower().replace(" ", "_")
                if not kw:
                    continue
                if kw not in self._exact_index:
                    self._exact_index[kw] = []
                self._exact_index[kw].append(tag)

    def match(self, booru_tags: list[str], max_results: int = MAX_RESULTS) -> list[str]:
        """
        根据 WD14 预测的 booru tags 匹配 Pixiv 日文标签。

        Args:
            booru_tags: WD14 预测的英文标签列表（已做 normalize，空格分隔或下划线均可）
            max_results: 最多返回的候选标签数

        Returns:
            匹配到的 Pixiv 日文标签列表，按匹配度降序
        """
        if not self._loaded:
            self.load()

        if not booru_tags:
            _logger.debug("[match] 输入为空，返回空列表")
            return []

        # 归一化输入标签
        normalized_inputs = []
        for tag in booru_tags:
            t = str(tag).strip().lower().replace(" ", "_")
            t = t.replace("-", "_")
            if t:
                normalized_inputs.append(t)

        _logger.debug(
            "[match] 输入 booru tags (%d 个): %s",
            len(normalized_inputs),
            ", ".join(normalized_inputs),
        )

        # 打分：{pixiv_tag: score}
        # 基础分：精确匹配 > 子串匹配
        # 权重：log(count) 缩放到 0~3 范围，count 越大权重越高
        scores: dict[str, float] = {}

        # 记录每条得分的来源（用于 debug 日志）
        # 格式: {pixiv_tag: [(input_tag, match_type, matched_kw, score_contribution), ...]}
        _score_sources: dict[str, list[tuple[str, str, str, float]]] = {}

        for input_tag in normalized_inputs:
            # 1. 精确匹配
            if input_tag in self._exact_index:
                for pixiv_tag in self._exact_index[input_tag]:
                    item = self._tag_by_name.get(pixiv_tag)
                    count = item.get("count", 0) if item else 0
                    pop_weight = self._count_weight(count)
                    contribution = 2.0 + pop_weight
                    scores[pixiv_tag] = scores.get(pixiv_tag, 0) + contribution
                    if pixiv_tag not in _score_sources:
                        _score_sources[pixiv_tag] = []
                    _score_sources[pixiv_tag].append(
                        (input_tag, "exact", input_tag, contribution)
                    )
                    _logger.debug(
                        "  [exact  ] %-22s -> %-20s  (count=%-7s pop_w=%.2f)  +%.2f => %.2f",
                        f"'{input_tag}'",
                        f"'{pixiv_tag}'",
                        _fmt_count(count),
                        pop_weight,
                        contribution,
                        scores[pixiv_tag],
                    )

            # 2. 子串匹配（遍历所有 en_keywords）
            for kw, pixiv_tags in self._exact_index.items():
                if kw == input_tag:
                    continue  # 已在上面处理
                # 双向子串匹配
                if kw in input_tag or input_tag in kw:
                    for pixiv_tag in pixiv_tags:
                        item = self._tag_by_name.get(pixiv_tag)
                        count = item.get("count", 0) if item else 0
                        pop_weight = self._count_weight(count)
                        contribution = 1.0 + pop_weight * 0.5
                        scores[pixiv_tag] = scores.get(pixiv_tag, 0) + contribution
                        if pixiv_tag not in _score_sources:
                            _score_sources[pixiv_tag] = []
                        _score_sources[pixiv_tag].append(
                            (input_tag, "substr", kw, contribution)
                        )
                        _logger.debug(
                            "  [substr ] %-22s -> %-20s  (kw='%s' count=%-7s pop_w=%.2f)  +%.2f => %.2f",
                            f"'{input_tag}'",
                            f"'{pixiv_tag}'",
                            kw,
                            _fmt_count(count),
                            pop_weight,
                            contribution,
                            scores[pixiv_tag],
                        )

        if not scores:
            _logger.debug("[match] 无任何匹配结果")
            return []

        # 排序并返回前 N 个
        sorted_tags = sorted(scores.items(), key=lambda x: -x[1])

        # debug: 打印最终排名
        if _logger.isEnabledFor(logging.DEBUG):
            lines = []
            for i, (tag, score) in enumerate(sorted_tags[:max_results]):
                item = self._tag_by_name.get(tag, {})
                count = item.get("count", 0)
                sources = _score_sources.get(tag, [])
                src_detail = ", ".join(
                    f"{inp}->{typ}({kw})" for inp, typ, kw, _ in sources
                )
                lines.append(
                    f"\n  #{i+1} {tag}: count={_fmt_count(count)}  score={score:.2f}  [{src_detail}]"
                )
            _logger.debug("[match] 最终结果 (前 %d 个):%s", max_results, "".join(lines))

            if len(sorted_tags) > max_results:
                _logger.debug(
                    "[match] 另有 %d 个低分匹配未返回",
                    len(sorted_tags) - max_results,
                )

        result = [tag for tag, _ in sorted_tags[:max_results]]
        return result

    @staticmethod
    def _count_weight(count: int) -> float:
        """将 Pixiv 标签作品数转换为加权分数（0~3 范围）"""
        if count <= 0:
            return 0.0
        # log10 缩放: count=1000->1.0, count=10000->1.33, count=1M->2.0, count=10M->2.33
        return min(3.0, max(0.0, (math.log10(count) - 2.0) / 2.0))

    def get_tag_count(self) -> int:
        """返回缓存的标签总数"""
        if not self._loaded:
            self.load()
        return len(self._tags)

    def reload(self):
        """强制重新加载缓存"""
        self._loaded = False
        self.load(force_reload=True)


# ---- 便捷函数 ----

def match_pixiv_tags_from_booru(booru_tags: list[str], max_results: int = MAX_RESULTS) -> list[str]:
    """
    便捷函数：从 booru tags 匹配 Pixiv 标签。
    调用此函数前确保已运行过 pixiv_tag_scraper.scrape_pixiv_tags() 生成缓存。
    """
    matcher = PixivTagMatcher.get_instance()
    return matcher.match(booru_tags, max_results=max_results)


def get_local_pixiv_tag_candidates(image_source_or_booru_tags, booru_tag_limit: int = 30, log_callback=None) -> list[str]:
    """
    一站式函数：对图片运行 WD14 → 匹配 Pixiv 标签 → 返回候选日文标签。

    这是暴露给外部调用的主入口，整合了 WD14 推理和标签匹配。
    总耗时目标是 10 秒以内（WD14 推理约占 3-8 秒，匹配 < 0.1 秒）。

    Args:
        image_source_or_booru_tags: 图片路径(str)、PIL Image 对象，或已计算好的 booru tags 列表(list)
        booru_tag_limit: WD14 输出的最大标签数（仅当传入图片时有效）
        log_callback: 可选日志回调

    Returns:
        Pixiv 日文标签候选列表
    """
    from utils.wd14_tagger import predict_local_booru_tags

    # 支持传入已计算好的 booru tags，避免重复推理
    if isinstance(image_source_or_booru_tags, list):
        booru_tags = [str(t).strip() for t in image_source_or_booru_tags if str(t).strip()]
        if log_callback:
            log_callback(f"使用已缓存的 booru 标签（{len(booru_tags)} 个）进行 Pixiv 标签匹配")
    else:
        booru_tags = predict_local_booru_tags(
            image_source_or_booru_tags,
            booru_tag_limit=max(booru_tag_limit, 40),
            log_callback=log_callback
        )

    if not booru_tags:
        if log_callback:
            log_callback("本地 WD14 标签预测为空，跳过 Pixiv 标签匹配")
        return []

    if log_callback:
        log_callback(f"开始根据 {len(booru_tags)} 个本地 booru 标签匹配 Pixiv 标签...")

    pixiv_candidates = match_pixiv_tags_from_booru(booru_tags)

    if log_callback:
        if pixiv_candidates:
            log_callback(
                f"Pixiv 标签匹配完成，共 {len(pixiv_candidates)} 个候选: {', '.join(pixiv_candidates)}"
            )
        else:
            log_callback("Pixiv 标签匹配完成，未命中任何标签")

    return pixiv_candidates


# ---- 命令行测试入口 ----

if __name__ == "__main__":
    import sys

    # 测试匹配功能
    from utils.pixiv_tag_scraper import scrape_pixiv_tags
    scrape_pixiv_tags(force_refresh=True)

    matcher = PixivTagMatcher.get_instance()
    matcher.load()
    print(f"已加载 {matcher.get_tag_count()} 个 Pixiv 标签")

    # 模拟 WD14 预测结果
    test_tags = ["1girl", "solo", "long_hair", "blue_eyes", "school_uniform",
                 "serafuku", "pleated_skirt", "thighhighs", "smile", "looking_at_viewer",
                 "ribbon", "hair_ornament", "sakura", "outdoors"]

    print(f"\n测试输入 booru tags: {test_tags}")
    result = matcher.match(test_tags)
    print(f"匹配结果 ({len(result)} 个):")
    for tag in result:
        item = matcher._tag_by_name.get(tag, {})
        count_str = f"{item.get('count', '?'):,}" if isinstance(item.get('count'), int) else item.get('count', '?')
        print(f"  {tag}  [{item.get('category', '?')}]  count: {count_str}")

    # 如果传了图片路径，做完整测试
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if os.path.exists(img_path):
            print(f"\n完整流程测试: {img_path}")
            candidates = get_local_pixiv_tag_candidates(img_path, log_callback=print)
            print(f"最终 Pixiv 标签候选: {candidates}")
