"""Read-only product/manifest lookup and explicit manual refine checkpoint planning."""
import hashlib
import json
import os
from pathlib import Path

STAGES = ('targeted', 'whole')
HYBRID_STAGES = ('targeted', 'background', 'clothing', 'whole')
HYBRID_V8_STAGES = ('targeted', 'whole', 'background', 'clothing')
WHOLE_STAGES = ('whole',)
LEGACY_STAGES = ('person', 'hair', 'background', 'ornaments', 'lighting')
LABELS = {'clothing': '衣装修复', 'targeted': '针对性修复', 'whole': '整体优化', 'diagnosis': '诊断', 'person': '人物', 'hair': '头发', 'background': '背景',
          'ornaments': '配饰', 'lighting': '光照'}

def absolute(path):
    return os.path.normcase(os.path.abspath(path)) if path else ''

def read_record(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stages_for_record(record):
    if record.get('workflow') == 'hybrid':
        return HYBRID_STAGES if record.get('version', 8) >= 9 else HYBRID_V8_STAGES
    if record.get('workflow') == 'targeted_whole':
        return STAGES
    return WHOLE_STAGES if record.get('workflow') == 'whole' else LEGACY_STAGES

def find_record(product):
    """Follow exact post-process manifest paths, then prefer latest explicit child run."""
    if not product:
        return None
    roots = []
    parent = Path(product).absolute().parent
    for folder in (parent, *parent.parents):
        if folder.name.isdigit() and len(folder.name) == 8:
            roots = [folder]
            break
    if not roots:
        roots = [parent]
    aliases = {absolute(product)}
    for root in roots:
        for manifest in root.glob('pipeline-steps/**/pipeline-manifest.json'):
            try:
                for item in read_record(manifest).get('items', []):
                    outputs = [item.get('final')] + [s.get('out') for s in item.get('steps', [])]
                    if any(absolute(p) in aliases for p in outputs if p):
                        aliases.add(absolute(item.get('source')))
            except (OSError, ValueError, TypeError):
                continue
    records = []
    for root in roots:
        for p in root.glob('**/*-refine.json'):
            try:
                r = read_record(p)
                outputs = [r.get('selected')] + [v for s in r.get('stages', []) for v in s.get('outputs', [])]
                score = 2 if any(absolute(v) in aliases for v in outputs if v) else (1 if absolute(r.get('source')) in aliases else 0)
                records.append((score, p.stat().st_mtime_ns, p, r))
            except (OSError, ValueError, TypeError):
                continue
    matches = [row for row in records if row[0]]
    if not matches:
        return None
    best = max(matches, key=lambda row: row[:2])
    # A new child is authoritative for this prior result even if the displayed file was retained.
    seen = set()
    while absolute(best[2]) not in seen:
        seen.add(absolute(best[2]))
        children = [row for row in records if absolute(row[3].get('parent_record')) == absolute(best[2])]
        if not children:
            break
        best = max(children, key=lambda row: row[1])
    return str(best[2]), best[3]

def image_stage(record, stage):
    return next((s for s in reversed(record.get('stages', [])) if s.get('kind') == 'image' and s.get('key') == stage), None)

def default_stage(record):
    if record.get('stopped_at') in stages_for_record(record):
        return record['stopped_at']
    if record.get('workflow') in ('whole', 'targeted_whole', 'hybrid') and record.get('status') == 'finished':
        return next(stage for stage in reversed(stages_for_record(record)) if not skipped(record, stage))
    for s in record.get('stages', []):
        if s.get('kind') == 'image' and s.get('status') != 'accepted':
            return s['key']
    done = set(record.get('completed_before', [])) | {s['key'] for s in record.get('stages', []) if s.get('status') == 'accepted'}
    for stage in stages_for_record(record):
        if skipped(record, stage):
            continue
        if stage not in done:
            return stage
    return stages_for_record(record)[-1]

def skipped(record, stage):
    if stage == 'targeted':
        return record.get('targeted_check', {}).get('decision') != 'repair'
    diag = record.get('diagnosis', {})
    if record.get('workflow') == 'hybrid':
        return stage == 'clothing' and diag.get('has_people') is False
    if stage in ('person', 'hair', 'ornaments') and diag.get('has_people') is False:
        return True
    if stage == 'hair':
        return record.get('hair_check', {}).get('decision') != 'repair'
    return stage in ('person', 'background') and stage + '_tasks' in diag and not diag[stage + '_tasks'].strip()

def status_text(record):
    failed = any(s.get('status') == 'rejected' for s in record.get('stages', []))
    result = ('异常停止' if record.get('status') == 'fallback' else '已完成' if record.get('status') == 'finished' else
              '未通过，保留最近输入' if failed else '执行中')
    rows = ['Refine：' + result]
    materials = record.get('material_guidance', {}).get('materials', [])
    if materials:
        rows.append('服装材质：' + '；'.join(f"{item['region']} / {item['material']}" for item in materials))
    if record.get('parent_record'):
        rows.append('诊断：复用原诊断')
    rows += [f'{LABELS.get(stage, stage)}：复用已通过断点' for stage in record.get('completed_before', [])]
    for s in record.get('stages', []):
        key = s.get('key', '')
        label = LABELS.get(key.removesuffix('-audit'), key)
        if key.endswith('-audit'):
            report = s.get('report', {})
            label += '审计'
            value = {'pass': '通过', 'fail': '未通过'}.get(report.get('verdict'), s.get('status', '等待'))
            reasons = report.get('unacceptable_changes') or report.get('limitations') or []
            rows.append(f'{label}：{value}' + ('；' + '\n'.join(reasons) if report.get('verdict') == 'fail' else ''))
        else:
            value = {'accepted': '通过', 'rejected': '未通过', 'requested': '执行中',
                     'returned': '已返回', 'no_output': '无产物', 'skipped': '跳过'}.get(s.get('status'), s.get('status', '等待'))
            model = s.get('options', {}).get('model')
            rows.append(f'{label}：{value}' + (f'（{model}）' if model else ''))
    present = {s.get('key') for s in record.get('stages', [])} | set(record.get('completed_before', []))
    for stage in stages_for_record(record):
        if stage not in present:
            rows.append(f'{LABELS[stage]}：' + ('无需补修' if skipped(record, stage) else '尚未执行'))
    if record.get('manual_override'):
        rows.insert(0, '人工采用候选并继续（不等同于自动审计通过）')
    if record.get('error'):
        rows.append(record['error'])
    return '\n'.join(rows) or '暂无工序'

def resume_plan(path, stage=None, force=False):
    r = read_record(path)
    source = r['source']
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != r.get('source_sha256'):
        raise ValueError('源图已改变，不能复用此断点')
    stage = stage or default_stage(r)
    stages = stages_for_record(r)
    if stage not in stages:
        raise ValueError('未知 refine 工序')
    entry = image_stage(r, stage)
    if entry:
        selected = entry.get('input')
    else:
        selected = r['selected']
    remaining = stages[stages.index(stage):]
    override = None
    if force:
        if not entry or not entry.get('outputs') or entry.get('status') == 'requested':
            raise ValueError('该步骤没有已返回候选可人工采用')
        selected = entry['outputs'][0]
        remaining = stages[stages.index(stage) + 1:]
        override = {'stage': stage, 'candidate': selected, 'old_status': entry.get('status')}
    if not selected or not Path(selected).is_file():
        raise ValueError('断点输入不存在')
    image = next((s for s in reversed(r.get('stages', [])) if s.get('kind') == 'image' and s.get('options')), None)
    if not image:
        raise ValueError('历史记录缺少出图参数')
    completed = set(r.get('completed_before', [])) | {s['key'] for s in r.get('stages', []) if s.get('status') == 'accepted' and s['key'] in stages and stages.index(s['key']) < stages.index(stage)}
    if force:
        completed.add(stage)
    return {'record': r, 'source': source, 'selected': selected, 'remaining': remaining,
            'options': dict(image['options']), 'override': override, 'completed': sorted(completed), 'stage': stage}
