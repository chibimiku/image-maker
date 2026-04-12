import re


def normalize_booru_tags(booru_tags, limit=30, output_style="underscore"):
    if isinstance(booru_tags, str):
        raw_items = [booru_tags]
    elif isinstance(booru_tags, list):
        raw_items = booru_tags
    else:
        raw_items = []
    normalized_tags = []
    seen = set()
    for item in raw_items:
        parts = str(item).split(",")
        for part in parts:
            tag = str(part).strip().lower()
            if not tag:
                continue
            tag = re.sub(r"[_\s]+", " ", tag).strip()
            if output_style == "space":
                tag = re.sub(r"\s+", " ", tag).strip()
            else:
                tag = tag.replace(" ", "_")
                tag = re.sub(r"_+", "_", tag).strip("_")
            if not tag or tag in seen:
                continue
            seen.add(tag)
            normalized_tags.append(tag)
            if len(normalized_tags) >= limit:
                return normalized_tags
    return normalized_tags
