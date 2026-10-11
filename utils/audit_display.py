"""Display-only bilingual audit text. Decisions always use the English schema."""
def chinese_fields(report):
    value = (report or {}).get('display_zh')
    if not isinstance(value, dict):
        return {}
    return {k: value[k].strip() for k in ('summary', 'details')
            if isinstance(value.get(k), str) and value[k].strip()}


def summary(report):
    return chinese_fields(report).get('summary') or (report or {}).get('summary') or ''


def details(report):
    return chinese_fields(report).get('details') or ''
