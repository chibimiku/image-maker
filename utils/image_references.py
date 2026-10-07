"""按图片身份重编号正文引用，避免交换图号时连续替换串号。"""
import re

REFERENCE_PATTERN = re.compile(r"(?i)(?<![A-Za-z0-9_])(image\s*)(\d+)(?!\d)|([图圖]\s*)(\d+)(?!\d)")
REMOVED_REFERENCE = "[已移除图片引用]"


def reference_changes(text, previous, current):
    """返回原正文中的 (start, end, replacement)，所有匹配基于同一旧编号表。"""
    positions = {identity: index for index, identity in enumerate(current, 1)}
    changes = []
    for match in REFERENCE_PATTERN.finditer(text):
        number_group = 2 if match.group(2) is not None else 4
        old_index = int(match.group(number_group))
        if not 1 <= old_index <= len(previous):
            continue
        new_index = positions.get(previous[old_index - 1])
        if new_index is None:
            changes.append((match.start(), match.end(), REMOVED_REFERENCE))
        elif new_index != old_index:
            changes.append((match.start(number_group), match.end(number_group), str(new_index)))
    return changes
