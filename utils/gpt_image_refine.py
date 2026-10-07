"""Diagnosed person -> audited person -> background -> audited result, no retries."""
import hashlib
import json
import math
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from utils.prompt_loader import read_prompt_file, render_prompt_file

AUDIT_FIELDS = ('anatomy_ok', 'identity_preserved', 'iris_hue_preserved',
                'iris_value_preserved', 'iris_chroma_preserved', 'garment_macro_preserved',
                'major_damage_preserved', 'material_types_preserved', 'style_medium_preserved',
                'pose_composition_preserved', 'objects_preserved', 'stage_scope_preserved')


def confident(report):
    value = report.get('confidence') if isinstance(report, dict) else None
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and 0.8 <= value <= 1)


def audit_passes(report):
    return (confident(report) and report.get('verdict') == 'pass'
            and all(report.get(k) is True for k in AUDIT_FIELDS)
            and report.get('unacceptable_changes') == [])


def text_request(stage, system, user, images, cancel_check=None):
    from utils.analysis_gpt_prompt import load_text_api_config, call_text_model
    from utils.first_image_audit import extract_json_object
    from utils.send_budget import guarded_text_call
    if callable(cancel_check) and cancel_check():
        raise RuntimeError('Refine cancelled before analysis')
    cfg = load_text_api_config()
    raw = guarded_text_call(stage, call_text_model, cfg['base_url'], cfg['api_key'],
                            cfg['model'], system, user, image_paths=images,
                            timeout=180, max_tokens=4500)
    return {'report': extract_json_object(raw), 'raw': raw, 'model': cfg['model']}


def refine_image(source, *, image_options, log_callback=None, cancel_check=None):
    """At most two image and three text requests; keep every rejected candidate."""
    from modules.others.api_backend import generate_image_repaint, resolve_save_dir
    from utils.gpt_image_optimize import validate_repaint_aspect
    log = log_callback or (lambda message: None)
    options = dict(image_options)
    run_id = 'refine-' + uuid.uuid4().hex[:12]
    directory = resolve_save_dir(options.get('save_sub_dir'), datetime.now().strftime('%Y%m%d'))
    os.makedirs(directory, exist_ok=True)
    record_path = os.path.join(directory, run_id + '-refine.json')
    record = {'version': 1, 'source': str(source), 'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              'selected': str(source), 'stages': [], 'image_limit': 2, 'text_limit': 3}
    selected = str(source)

    def save():
        record['selected'] = selected
        with open(record_path + '.tmp', 'w', encoding='utf-8') as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        # Antivirus/indexing can briefly hold a just-written file on Windows.
        # Retry only the local atomic rename; never repeat a paid request.
        for attempt in range(3):
            try:
                os.replace(record_path + '.tmp', record_path)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def ask(stage, template, user, images):
        system = read_prompt_file('gpt-image-optimize/' + template)
        entry = {'key': stage, 'kind': 'text', 'system': system, 'user': user,
                 'images': images, 'timeout': 180, 'status': 'requested'}
        record['stages'].append(entry)
        save()
        entry.update(text_request(run_id + '-' + stage, system, user, images, cancel_check))
        entry['status'] = 'returned'
        save()
        return entry['report']

    try:
        log('[refine] 诊断实际材质、身份颜色与人体连接；仅源图，无画风附件。')
        diag = ask('diagnosis', 'refine-diagnosis.md', 'Inspect the actual source artwork.', [str(source)])
        record['diagnosis'] = diag
        if (not confident(diag) or diag.get('anatomy_ok') is not True
                or not isinstance(diag.get('has_people'), bool)
                or not isinstance(diag.get('identity_anchors'), str) or not diag['identity_anchors'].strip()
                or any(not isinstance(diag.get(k), str) for k in ('person_tasks', 'background_tasks'))
                or (diag['has_people'] and not diag['person_tasks'].strip())):
            raise RuntimeError('Source anatomy/material diagnosis uncertain; preserve source')
        for stage in ('person', 'background'):
            tasks = diag[stage + '_tasks'].strip()
            if not tasks:
                continue
            if callable(cancel_check) and cancel_check():
                raise RuntimeError('Refine cancelled before image request')
            before = selected
            prompt = render_prompt_file('gpt-image-optimize/refine-' + stage + '.md', diag)
            request = dict(options, source_paths=[before], prompt=prompt, prompt_suffix='',
                           use_detail_suffix=False, extra_reference_paths=[], repeat=1,
                           content_guard=False, refine=False, request_timeout=360, retry_limit=0,
                           file_prefix=run_id + '-' + stage, budget_run_id=run_id + '-' + stage,
                           log_callback=log_callback, cancel_check=cancel_check)
            entry = {'key': stage, 'kind': 'image', 'input': before, 'prompt': prompt,
                     'options': {k: v for k, v in request.items() if k not in ('log_callback', 'cancel_check')},
                     'status': 'requested'}
            record['stages'].append(entry)
            save()
            log('[refine] 修人物衣装' if stage == 'person' else '[refine] 人物检查通过，清理背景表面')
            paths = generate_image_repaint(**request) or []
            entry.update(outputs=paths, status='returned' if paths else 'no_output')
            save()
            if not paths or not os.path.isfile(paths[0]):
                raise RuntimeError(stage + ' returned no image; no retry')
            validate_repaint_aspect(paths[0], options['aspect_ratio'])
            audit = ask(stage + '-audit', 'refine-audit.md',
                        json.dumps({'stage': stage, 'permitted_tasks': tasks}, ensure_ascii=False),
                        [str(source), before, paths[0]])
            if not audit_passes(audit):
                entry['status'] = 'rejected'
                log('[refine] 保真/人体未通过，回退最近通过的输入；候选与审计保留。')
                break
            selected = paths[0]
            entry['status'] = 'accepted'
            save()
        record['status'] = 'finished'
    except Exception as exc:
        record.update(status='fallback', error=f'{type(exc).__name__}: {exc}')
        log('[refine] 未完成，保留最近通过的输入：' + record['error'])
    try:
        save()
    except OSError as exc:
        log('[refine] 记录保存失败，回退源图：' + str(exc))
        return str(source)
    log('[refine] 记录：' + record_path)
    return selected
