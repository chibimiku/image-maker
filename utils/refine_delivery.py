"""Resume delivery after refine, applying downstream steps to the NEW candidate only."""
import os
from pathlib import Path
from utils.refine_state import absolute, read_record

LOCAL_STEPS = ('structure', 'contrast', 'tone', 'ink')


def delivery_recipe(product, record):
    parent = Path(record['source']).absolute().parent
    root = next((p for p in (parent, *parent.parents) if p.name.isdigit() and len(p.name) == 8), parent)
    aliases = {absolute(product), absolute(record.get('source'))}
    matches = []
    for path in root.glob('pipeline-steps/**/pipeline-manifest.json'):
        try:
            manifest = read_record(path)
            if any(absolute(item.get('source')) in aliases or absolute(item.get('final')) in aliases
                   for item in manifest.get('items', [])):
                matches.append((path.stat().st_mtime_ns, manifest))
        except (OSError, ValueError, TypeError):
            continue
    if not matches:
        return {'steps': {}, 'style_ref_path': '', 'origin': '无历史后处理参数，仅导出最终图'}
    manifest = max(matches, key=lambda row: row[0])[1]
    if 'recipe' in manifest:
        recipe = manifest['recipe']
        return {**recipe, 'steps': {k: v for k, v in recipe.get('steps', {}).items() if k in LOCAL_STEPS},
                'origin': '复用已保存的后处理参数'}
    # Legacy manifests contain no parameters. Reconstruct only ink, whose historical
    # defaults are known; never silently invent a tone reference or send a paid call.
    keys = {s.get('key') for item in manifest.get('items', [])
            if absolute(item.get('source')) in aliases or absolute(item.get('final')) in aliases
            for s in item.get('steps', [])}
    steps = {}
    if 'ink' in keys:
        steps['ink'] = {'enabled': True, 'target_sep': 8.0, 'amount': 1.0, 'max_darken': 40.0,
                        'sigma': 6.0, 'protect_light': 0.0}
    return {'steps': steps, 'style_ref_path': '', 'origin': '历史清单：提线采用历史默认参数，其它未记录工序不补跑'}


def deliver_refine(path, record_path, recipe, log_callback=None):
    log = log_callback or (lambda message: None)
    from utils.refine_state import find_record
    found = find_record(path)
    record = found[1] if found else read_record(record_path)
    if record.get('status') != 'finished':
        log('[工序] Refine 未完成，保留断点，不生成最终交付图。')
        return [path]
    from utils.post_process import run_pipeline, date_output_dir, new_run_id, save_manifest
    log('[工序] 最终输出：' + recipe.get('origin', 'saved downstream recipe'))
    steps = recipe.get('steps', {})
    if any(v.get('enabled') for v in steps.values()):
        return run_pipeline([path], steps, log_callback=log, style_ref_path=recipe.get('style_ref_path'))
    from PIL import Image
    directory = date_output_dir()
    from utils.output_isolation import resolve_output_target
    directory, _ = resolve_output_target(directory, 'x')
    os.makedirs(directory, exist_ok=True)
    run_id = new_run_id()
    name = Path(record['source']).stem + '-' + run_id + '-final-refine.png'
    output = os.path.join(directory, name)
    with Image.open(path) as im:
        im.save(output, format='PNG')
    save_manifest(os.path.join(directory, 'pipeline-steps', run_id),
                  {'recipe': recipe, 'items': [{'source': path, 'steps': [], 'final': output}]})
    log('[工序] 最终输出已保存：' + output)
    return [output]
