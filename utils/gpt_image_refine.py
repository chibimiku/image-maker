"""Audited Pro background, Flash clothing and final Flash whole/hair/lighting; historical resumes supported."""
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
MAX_IMAGE_REQUESTS = 4
MAX_TEXT_REQUESTS = 5
AUDIT_RETRY_LIMIT = 3  # Additional attempts after the initial transport failure.
MAX_TEXT_ATTEMPTS = 1 + (MAX_TEXT_REQUESTS - 1) * (1 + AUDIT_RETRY_LIMIT)
NETWORK_TIMEOUT_BUDGET = MAX_IMAGE_REQUESTS * 360 + MAX_TEXT_ATTEMPTS * 180


def confident(report):
    value = report.get('confidence') if isinstance(report, dict) else None
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and 0.8 <= value <= 1)


def audit_passes(report):
    return (confident(report) and report.get('verdict') == 'pass'
            and all(report.get(k) is True for k in AUDIT_FIELDS)
            and report.get('unacceptable_changes') == [])


def hair_decision(report):
    """Only an explicit, evidenced assessment may skip or trigger a paid hair edit."""
    state = report.get('hair_state') if isinstance(report, dict) else None
    needed = report.get('hair_repair_needed') if isinstance(report, dict) else None
    evidence = report.get('hair_evidence') if isinstance(report, dict) else None
    if (not confident(report) or not isinstance(evidence, list) or not evidence
            or any(not isinstance(item, str) or not item.strip() for item in evidence)):
        raise RuntimeError('Hair assessment missing or uncertain; preserve trusted input')
    if state == 'fragmented' and needed is True:
        return 'repair'
    if state in ('clear', 'not_visible') and needed is False:
        return 'skip'
    raise RuntimeError('Hair assessment missing, contradictory or uncertain; preserve trusted input')


def background_repair_passes(report):
    evidence = report.get('background_repair_evidence') if isinstance(report, dict) else None
    return (audit_passes(report) and report.get('background_repair_completed') is True
            and isinstance(evidence, list) and bool(evidence)
            and all(isinstance(item, str) and item.strip() for item in evidence))


def audit_geometry(paths):
    """Supply measured display geometry rather than asking the model to guess ratios."""
    from PIL import Image, ImageOps
    measurements = []
    for index, path in enumerate(paths, 1):
        with Image.open(path) as image:
            width, height = ImageOps.exif_transpose(image).size
        measurements.append({'image': index, 'width': width, 'height': height,
                             'width_over_height': round(width / height, 6)})
    return measurements


def targeted_tasks(report):
    """Never turn preservation prose or an uncertain diagnosis into a paid repair."""
    tasks = report.get('repair_targets') if isinstance(report, dict) else None
    fields = ('region', 'defect', 'evidence', 'repair', 'preserve')
    if (not confident(report) or report.get('targeted_repair_needed') is not True
            or not isinstance(tasks, list) or not tasks):
        return []
    if any(not isinstance(task, dict) or any(not isinstance(task.get(k), str)
           or not task[k].strip() for k in fields) for task in tasks):
        return []
    return tasks


def targeted_repair_passes(report):
    evidence = report.get('targeted_repair_evidence') if isinstance(report, dict) else None
    return (audit_passes(report) and report.get('targeted_repair_completed') is True
            and isinstance(evidence, list) and bool(evidence)
            and all(isinstance(item, str) and item.strip() for item in evidence))


def clothing_repair_passes(report):
    evidence = report.get('clothing_repair_evidence') if isinstance(report, dict) else None
    return (audit_passes(report) and report.get('clothing_repair_completed') is True
            and isinstance(evidence, list) and bool(evidence)
            and all(isinstance(item, str) and item.strip() for item in evidence))


def garment_material_guidance(report):
    """Use evidenced clothing classifications and file-backed English rules only."""
    rules = json.loads(read_prompt_file('gpt-image-optimize/refine-materials.json'))
    entries = report.get('garment_materials') if isinstance(report, dict) else None
    accepted = []
    if confident(report) and report.get('has_people') is True and isinstance(entries, list):
        for entry in entries:
            if (isinstance(entry, dict) and confident(entry)
                    and isinstance(entry.get('material'), str)
                    and entry.get('material') in rules['qualities']
                    and all(isinstance(entry.get(k), str) and entry[k].strip()
                            for k in ('region', 'evidence'))):
                accepted.append({k: entry[k] for k in ('region', 'material', 'evidence', 'confidence')})
    materials = list(dict.fromkeys(entry['material'] for entry in accepted))
    prompt = ''
    if materials:
        qualities = '; '.join(rules['qualities'][material] for material in materials)
        prompt = rules['texture_sentence'].replace('{qualities}', qualities) + ' ' + rules['lighting_sentence']
    return {'materials': accepted, 'prompt': prompt}


def text_request(stage, system, user, images, cancel_check=None):
    from utils.analysis_gpt_prompt import load_text_api_config, call_text_model
    from utils.first_image_audit import extract_json_object
    from utils.send_budget import guarded_text_call
    if callable(cancel_check) and cancel_check():
        raise RuntimeError('Refine cancelled before analysis')
    cfg = load_text_api_config()
    raw = guarded_text_call(stage, call_text_model, cfg['base_url'], cfg['api_key'],
                            cfg['model'], system, user, image_paths=images,
                            timeout=180, max_tokens=7000)
    return {'report': extract_json_object(raw), 'raw': raw, 'model': cfg['model']}


def _retryable_audit_error(exc):
    import requests
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        return response is not None and (response.status_code == 429 or 500 <= response.status_code < 600)
    return isinstance(exc, (TimeoutError, ConnectionError, requests.Timeout, requests.ConnectionError))


def _request_recorded_text(entry, stage, save, log, cancel_check):
    """Retry transport failures only; a returned report is never retried."""
    if entry['key'].endswith('-audit') and 'BILINGUAL DISPLAY OUTPUT:' not in entry['system']:
        bilingual = read_prompt_file('gpt-image-optimize/refine-audit.md').split('BILINGUAL DISPLAY OUTPUT:', 1)[1]
        entry['system'] += '\n\nBILINGUAL DISPLAY OUTPUT:' + bilingual
    limit = AUDIT_RETRY_LIMIT if entry['key'].endswith('-audit') else 0
    entry.update(retry_limit=limit, attempts=[])
    for index in range(limit + 1):
        if callable(cancel_check) and cancel_check():
            entry.update(status='failed', error='Refine cancelled before audit attempt')
            save()
            raise RuntimeError(entry['error'])
        attempt = {'number': index + 1, 'status': 'requested', 'started_at': datetime.now().isoformat()}
        entry['status'] = 'requested'
        entry['attempts'].append(attempt)
        save()
        try:
            result = text_request(stage, entry['system'], entry['user'], entry['images'], cancel_check)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            attempt.update(status='failed', error=error, finished_at=datetime.now().isoformat())
            retry = index < limit and _retryable_audit_error(exc)
            entry.update(status='retrying' if retry else 'failed', error=error)
            save()
            if not retry:
                raise
            log(f'[refine] {entry["key"]} 审计接口异常，自动重试 {index + 1}/{limit}；复用已有图片：{error}')
            # Brief backoff, cancellation remains responsive.
            for _ in range(10 * (index + 1)):
                if callable(cancel_check) and cancel_check():
                    entry.update(status='failed', error='Refine cancelled during audit retry')
                    save()
                    raise RuntimeError(entry['error'])
                time.sleep(0.1)
        else:
            attempt.update(status='returned', finished_at=datetime.now().isoformat())
            entry.update(result, status='returned')
            entry.pop('error', None)
            save()
            from utils.audit_display import summary, details
            if entry['key'].endswith('-audit'):
                log('[refine审计] ' + (summary(entry['report']) or entry['report'].get('verdict', '')) + '\n' + details(entry['report']))
            return entry['report']


def refine_image(source, *, image_options, log_callback=None, cancel_check=None,
                 resume_record=None, start_stage=None, force_candidate=False, resume_audit_only=False):
    if resume_record:
        from utils.refine_state import read_record
        if read_record(resume_record).get('workflow') not in ('whole', 'targeted_whole', 'hybrid'):
            if resume_audit_only:
                raise ValueError('旧版分步记录请使用重画步骤或人工采用候选')
            return _refine_legacy_image(source, image_options=image_options,
                                       log_callback=log_callback, cancel_check=cancel_check,
                                       resume_record=resume_record, start_stage=start_stage,
                                       force_candidate=force_candidate)
    return _refine_whole_image(source, image_options=image_options,
                               log_callback=log_callback, cancel_check=cancel_check,
                               resume_record=resume_record, start_stage=start_stage,
                               force_candidate=force_candidate, resume_audit_only=resume_audit_only)


def _refine_whole_image(source, *, image_options, log_callback=None, cancel_check=None,
                        resume_record=None, start_stage=None, force_candidate=False, resume_audit_only=False):
    from modules.others.api_backend import generate_image_repaint, resolve_save_dir
    from utils.gpt_image_optimize import validate_repaint_aspect, load_config
    log = log_callback or (lambda message: None)
    options = dict(image_options)
    config = load_config()
    plan = None
    if resume_record:
        from utils.refine_state import resume_plan
        plan = resume_plan(resume_record, start_stage, force_candidate, audit_only=resume_audit_only)
        source, options = plan['source'], plan['options']
        for key in ('source_paths', 'prompt', 'file_prefix', 'budget_run_id', 'return_metadata'):
            options.pop(key, None)
    selected = plan['selected'] if plan else str(source)
    run_id = 'refine-' + uuid.uuid4().hex[:12]
    directory = resolve_save_dir(options.get('save_sub_dir'), datetime.now().strftime('%Y%m%d'))
    os.makedirs(directory, exist_ok=True)
    record_path = os.path.join(directory, run_id + '-refine.json')
    workflow = plan['record']['workflow'] if plan else config.get('refine_workflow', 'hybrid')
    hybrid = workflow == 'hybrid'
    whole_last = hybrid and (not plan or plan['record'].get('version', 8) >= 9)
    stage_models = (dict(plan['record'].get('stage_models', {})) if plan else {
        'background': options.get('refine_background_model') or config['refine_background_model'],
        'clothing': options.get('refine_clothing_model') or config['refine_clothing_model']})
    for key in ('refine_background_model', 'refine_clothing_model'):
        options.pop(key, None)
    if hybrid:
        if not all(isinstance(stage_models.get(k), str) and stage_models[k].strip()
                   for k in ('background', 'clothing')):
            raise ValueError('Hybrid refine requires recorded background and clothing models')
        stage_models.update(whole=stage_models['clothing'], targeted=stage_models['clothing'])
    record = {'version': (9 if whole_last else 8) if hybrid else 7, 'workflow': workflow, 'source': str(source),
              'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              'selected': selected, 'stages': [], 'image_limit': MAX_IMAGE_REQUESTS if hybrid else 2,
              'text_limit': MAX_TEXT_REQUESTS if hybrid else 3, 'stage_models': stage_models,
              'audit_retry_limit': AUDIT_RETRY_LIMIT,
              'text_attempt_limit': MAX_TEXT_ATTEMPTS if hybrid else 9}
    if plan:
        record.update(parent_record=os.path.abspath(resume_record), start_stage=plan['stage'],
                      manual_override=plan['override'], completed_before=plan['completed'],
                      targeted_check=plan['record'].get('targeted_check', {}))

    def save():
        record['selected'] = selected
        with open(record_path + '.tmp', 'w', encoding='utf-8') as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        for attempt in range(3):
            try:
                os.replace(record_path + '.tmp', record_path)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def ask(key, template, user, images):
        entry = {'key': key, 'kind': 'text', 'system': read_prompt_file('gpt-image-optimize/' + template),
                 'user': user, 'images': images, 'timeout': 180, 'status': 'requested'}
        if plan and plan['audit_only'] and key == plan['stage'] + '-audit':
            old = next(s for s in reversed(plan['record']['stages']) if s.get('key') == key)
            entry.update({k: old[k] for k in ('system', 'user', 'images')})
            entry['reused_request_from'] = os.path.abspath(resume_record)
            if plan.get('refresh_audit_scope'):
                entry['system'] = read_prompt_file('gpt-image-optimize/' + template)
                entry['scope_clarification'] = 'Hybrid background/clothing require explicit hair assessment'
        record['stages'].append(entry)
        save()
        return _request_recorded_text(entry, run_id + '-' + key, save, log, cancel_check)

    def image_result(entry, request):
        if plan and plan['audit_only'] and entry['key'] == plan['stage']:
            from utils.refine_state import image_stage
            old = image_stage(plan['record'], plan['stage'])
            entry.update(prompt=old['prompt'], options=dict(old['options']),
                         reused_from_record=os.path.abspath(resume_record))
            log('[refine] 复用已返回候选，仅补跑失败审计；不重复出图。')
            return list(old['outputs'])
        return generate_image_repaint(**request) or []

    def retry_prompt(stage, prompt):
        if plan and not plan['audit_only'] and stage == plan['stage']:
            old = next((s for s in reversed(plan['record']['stages']) if s.get('key') == stage + '-audit'), {})
            report = old.get('report') or {}
            changes = report.get('unacceptable_changes')
            if report.get('verdict') == 'fail' and isinstance(changes, list) and changes:
                prompt += '\n\n' + render_prompt_file('gpt-image-optimize/refine-retry-guidance.md',
                    {'rejected_changes': json.dumps(changes, ensure_ascii=False)}).strip()
        return prompt

    try:
        # Reuse an earlier source diagnosis only for an explicit historical checkpoint.
        diag = (plan['record'].get('diagnosis') if plan else
                ask('diagnosis', 'refine-diagnosis.md', 'Inspect the actual source artwork.', [str(source)]))
        record['diagnosis'] = diag
        if (not confident(diag) or diag.get('anatomy_ok') is not True
                or not isinstance(diag.get('has_people'), bool)
                or not isinstance(diag.get('identity_anchors'), str) or not diag['identity_anchors'].strip()):
            raise RuntimeError('Source anatomy diagnosis uncertain; preserve source')
        tasks = targeted_tasks(diag)
        if not plan:
            record['targeted_check'] = {'decision': 'repair' if tasks else 'skip', 'targets': tasks,
                                       'reason': 'explicit evidenced defects' if tasks else 'no explicit actionable defects'}
            save()
        remaining = plan['remaining'] if plan else (('targeted', 'background', 'clothing', 'whole') if hybrid else ('targeted', 'whole'))
        if 'targeted' in remaining and tasks:
            if callable(cancel_check) and cancel_check():
                raise RuntimeError('Refine cancelled before targeted image request')
            before = selected
            prompt = render_prompt_file('gpt-image-optimize/refine-targeted.md', {
                'identity_anchors': diag['identity_anchors'],
                'repair_targets': json.dumps(tasks, ensure_ascii=False)})
            request = dict(options, source_paths=[before], prompt=prompt, prompt_suffix='',
                           model=stage_models['targeted'] if hybrid else options.get('model'),
                           use_detail_suffix=False, extra_reference_paths=[], repeat=1,
                           content_guard=False, refine=False, request_timeout=360, retry_limit=0,
                           file_prefix=run_id + '-targeted', budget_run_id=run_id + '-targeted',
                           log_callback=log_callback, cancel_check=cancel_check)
            entry = {'key': 'targeted', 'kind': 'image', 'input': before, 'prompt': prompt,
                     'options': {k: v for k, v in request.items() if k not in ('log_callback', 'cancel_check')},
                     'status': 'requested'}
            record['stages'].append(entry)
            save()
            log('[refine] 诊断发现明确缺陷，先针对性修复一次。')
            try:
                paths = image_result(entry, request)
                entry.update(outputs=paths, status='returned' if paths else 'no_output')
                save()
                if not paths or not os.path.isfile(paths[0]):
                    raise RuntimeError('targeted returned no image; no retry')
                validate_repaint_aspect(paths[0], options['aspect_ratio'])
                images = [str(source), before, paths[0]]
                audit = ask('targeted-audit', 'refine-audit.md',
                            json.dumps({'stage': 'targeted', 'permitted_tasks': prompt,
                                        'repair_targets': tasks,
                                        'measured_image_geometry': audit_geometry(images)}, ensure_ascii=False), images)
                if targeted_repair_passes(audit):
                    selected = paths[0]
                    entry['status'] = 'accepted'
                else:
                    entry['status'] = 'rejected'
            except Exception as exc:
                entry.update(error=f'{type(exc).__name__}: {exc}',
                             status='failed' if entry['status'] == 'requested' else 'rejected')
                if callable(cancel_check) and cancel_check():
                    raise
            if entry['status'] != 'accepted':
                log('[refine] 针对性修复未采用，保留原输入，继续后续固定工序；无重试。')
            save()
        def run_whole():
            nonlocal selected
            if callable(cancel_check) and cancel_check():
                raise RuntimeError('Refine cancelled before image request')
            prompt = read_prompt_file('gpt-image-optimize/refine-whole.md').strip()
            record['material_guidance'] = garment_material_guidance(diag)
            if record['material_guidance']['prompt']:
                prompt += '\n\n' + record['material_guidance']['prompt']
                log('固定整体修复：追加已识别服装材质的质感与光照要求')
            prompt = retry_prompt('whole', prompt)
            before = selected
            request = dict(options, source_paths=[before], prompt=prompt, prompt_suffix='',
                           model=stage_models['whole'] if hybrid else options.get('model'),
                           use_detail_suffix=False, extra_reference_paths=[], repeat=1,
                           content_guard=False, refine=False, request_timeout=360, retry_limit=0,
                           file_prefix=run_id + '-whole', budget_run_id=run_id + '-whole',
                           log_callback=log_callback, cancel_check=cancel_check)
            entry = {'key': 'whole', 'kind': 'image', 'input': before, 'prompt': prompt,
                     'options': {k: v for k, v in request.items() if k not in ('log_callback', 'cancel_check')},
                     'status': 'requested'}
            record['stages'].append(entry)
            save()
            log('[refine] 整体修复＋头发重绘＋光照重渲染（单次）')
            paths = image_result(entry, request)
            entry.update(outputs=paths, status='returned' if paths else 'no_output')
            save()
            if not paths or not os.path.isfile(paths[0]):
                raise RuntimeError('whole returned no image; no retry')
            validate_repaint_aspect(paths[0], options['aspect_ratio'])
            images = [str(source), before, paths[0]]
            audit = ask('whole-audit', 'refine-audit.md',
                        json.dumps({'stage': 'whole', 'permitted_tasks': prompt,
                                    'measured_image_geometry': audit_geometry(images)}, ensure_ascii=False), images)
            if audit_passes(audit) and hair_decision(audit) == 'skip':
                selected = paths[0]
                entry['status'] = 'accepted'
            else:
                entry['status'] = 'rejected'
                record['stopped_at'] = 'whole'
                log('[refine] 整体优化未通过审计，保留输入与候选；无自动重试。')
        if not whole_last and 'whole' in remaining:
            run_whole()
        if hybrid and not record.get('stopped_at'):
            for stage in ('background', 'clothing'):
                if stage not in remaining:
                    continue
                if stage == 'clothing' and diag['has_people'] is False:
                    continue
                if callable(cancel_check) and cancel_check():
                    raise RuntimeError('Refine cancelled before ' + stage)
                before = selected
                prompt = read_prompt_file('gpt-image-optimize/refine-' + stage + '-redraw.md').strip()
                prompt = retry_prompt(stage, prompt)
                request = dict(options, model=stage_models[stage], source_paths=[before], prompt=prompt,
                               prompt_suffix='', use_detail_suffix=False, extra_reference_paths=[], repeat=1,
                               content_guard=False, refine=False, request_timeout=360, retry_limit=0,
                               file_prefix=run_id + '-' + stage, budget_run_id=run_id + '-' + stage,
                               log_callback=log_callback, cancel_check=cancel_check)
                entry = {'key': stage, 'kind': 'image', 'input': before, 'prompt': prompt,
                         'options': {k: v for k, v in request.items() if k not in ('log_callback', 'cancel_check')},
                         'status': 'requested'}
                record['stages'].append(entry)
                save()
                log(f'[refine] {stage} → {stage_models[stage]}（单次，无重试）')
                paths = image_result(entry, request)
                entry.update(outputs=paths, status='returned' if paths else 'no_output')
                save()
                if not paths or not os.path.isfile(paths[0]):
                    raise RuntimeError(stage + ' returned no image; no retry')
                validate_repaint_aspect(paths[0], options['aspect_ratio'])
                images = [str(source), before, paths[0]]
                audit = ask(stage + '-audit', 'refine-audit.md',
                            json.dumps({'stage': stage, 'workflow': 'hybrid', 'permitted_tasks': prompt,
                                        'measured_image_geometry': audit_geometry(images)}, ensure_ascii=False), images)
                effective = background_repair_passes(audit) if stage == 'background' else clothing_repair_passes(audit)
                if effective and hair_decision(audit) == 'skip':
                    selected = paths[0]
                    entry['status'] = 'accepted'
                else:
                    entry['status'] = 'rejected'
                    record['stopped_at'] = stage
                    log(f'[refine] {stage} 未通过，保留输入和候选；停止后续。')
                    break
        if whole_last and not record.get('stopped_at') and 'whole' in remaining:
            run_whole()
        record['status'] = 'stopped' if record.get('stopped_at') else 'finished'
    except Exception as exc:
        record.update(status='fallback', error=f'{type(exc).__name__}: {exc}')
        log('[refine] 未完成，保留输入：' + record['error'])
    try:
        save()
    except OSError as exc:
        log('[refine] 记录保存失败，回退源图：' + str(exc))
        return str(source)
    log('[refine] 记录：' + record_path)
    return selected


def _refine_legacy_image(source, *, image_options, log_callback=None, cancel_check=None,
                         resume_record=None, start_stage=None, force_candidate=False):
    """At most five image and six text requests; keep every rejected candidate."""
    from modules.others.api_backend import generate_image_repaint, resolve_save_dir
    from utils.gpt_image_optimize import validate_repaint_aspect
    log = log_callback or (lambda message: None)
    options = dict(image_options)
    plan = None
    if resume_record:
        from utils.refine_state import resume_plan
        plan = resume_plan(resume_record, start_stage, force_candidate)
        source = plan['source']
        # Historical per-stage request options, with callback/request overrides rebuilt below.
        options = plan['options']
        for key in ('source_paths', 'prompt', 'file_prefix', 'budget_run_id', 'return_metadata'):
            options.pop(key, None)
    run_id = 'refine-' + uuid.uuid4().hex[:12]
    directory = resolve_save_dir(options.get('save_sub_dir'), datetime.now().strftime('%Y%m%d'))
    os.makedirs(directory, exist_ok=True)
    record_path = os.path.join(directory, run_id + '-refine.json')
    record = {'version': 5, 'source': str(source), 'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              'selected': str(source), 'stages': [],
              'image_limit': 5, 'text_limit': 6,
              'audit_retry_limit': AUDIT_RETRY_LIMIT, 'text_attempt_limit': 21}
    selected = str(source)
    repair_hair = False
    if plan:
        selected = plan['selected']
        record.update(parent_record=os.path.abspath(resume_record), start_stage=plan['stage'],
                      manual_override=plan['override'], completed_before=plan['completed'],
                      hair_check=plan['record'].get('hair_check', {}))
        repair_hair = record['hair_check'].get('decision') == 'repair'
        if plan['stage'] == 'hair' and not force_candidate:
            repair_hair = True  # Explicit manual hair retry overrides the prior skip decision.
        if force_candidate and plan['stage'] == 'person':
            old_audit = next((s.get('report', {}) for s in reversed(plan['record']['stages']) if s.get('key') == 'person-audit'), {})
            try:
                repair_hair = hair_decision(old_audit) == 'repair'
            except RuntimeError:
                repair_hair = True

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
        return _request_recorded_text(entry, run_id + '-' + stage, save, log, cancel_check)

    try:
        log('[refine] 诊断实际材质、身份颜色与人体连接；仅源图，无画风附件。')
        diag = (plan['record']['diagnosis'] if plan else
                ask('diagnosis', 'refine-diagnosis.md', 'Inspect the actual source artwork.', [str(source)]))
        record['diagnosis'] = diag
        if (not confident(diag) or diag.get('anatomy_ok') is not True
                or not isinstance(diag.get('has_people'), bool)
                or not isinstance(diag.get('identity_anchors'), str) or not diag['identity_anchors'].strip()
                or any(not isinstance(diag.get(k), str) for k in ('person_tasks', 'background_tasks'))
                or (diag['has_people'] and not diag['person_tasks'].strip())):
            raise RuntimeError('Source anatomy/material diagnosis uncertain; preserve source')
        for stage in (plan['remaining'] if plan else ('person', 'hair', 'background', 'ornaments', 'lighting')):
            if stage == 'hair' and (not diag['has_people'] or not repair_hair):
                continue
            if stage == 'ornaments' and not diag['has_people']:
                continue
            tasks = (read_prompt_file('gpt-image-optimize/refine-' + stage + '.md').strip()
                     if stage in ('hair', 'ornaments', 'lighting') else diag[stage + '_tasks'].strip())
            if not tasks:
                continue
            if callable(cancel_check) and cancel_check():
                raise RuntimeError('Refine cancelled before image request')
            before = selected
            prompt = (tasks if stage in ('hair', 'ornaments', 'lighting') else
                      render_prompt_file('gpt-image-optimize/refine-' + stage + '.md', diag))
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
            log({'person': '[refine] 修人物衣装与头发连通',
                 'hair': '[refine] 人物审计仍发现碎发：独立修复头发一次',
                 'background': '[refine] 清理背景表面',
                 'ornaments': '[refine] 独立修复配饰花纹与无意义碎线',
                 'lighting': '[refine] 独立末步：统一整体画面光照'}[stage])
            paths = generate_image_repaint(**request) or []
            entry.update(outputs=paths, status='returned' if paths else 'no_output')
            save()
            if not paths or not os.path.isfile(paths[0]):
                raise RuntimeError(stage + ' returned no image; no retry')
            validate_repaint_aspect(paths[0], options['aspect_ratio'])
            audit_images = [str(source), before, paths[0]]
            audit = ask(stage + '-audit', 'refine-audit.md',
                        json.dumps({'stage': stage,
                                    'permitted_tasks': prompt if stage in ('person', 'background') else tasks,
                                    'measured_image_geometry': audit_geometry(audit_images)}, ensure_ascii=False),
                        audit_images)
            if not audit_passes(audit):
                entry['status'] = 'rejected'
                record['stopped_at'] = stage
                log('[refine] 保真/人体未通过，回退最近通过的输入；候选与审计保留。')
                break
            if stage == 'background' and not background_repair_passes(audit):
                entry['status'] = 'rejected'
                record['stopped_at'] = stage
                log('[refine] 背景修复目标未完成或证据不明，保留输入；停止后续工序。')
                break
            if stage == 'hair' and hair_decision(audit) != 'skip':
                entry['status'] = 'rejected'
                record['stopped_at'] = stage
                log('[refine] 头发仍破碎，保留补修输入；不重复修复或继续背景。')
                break
            selected = paths[0]
            entry['status'] = 'accepted'
            save()
            if stage == 'person':
                # Reuse the person audit: no extra text call when hair is already coherent.
                record['hair_check'] = {'input': selected, 'audit_key': 'person-audit',
                                        'state': audit.get('hair_state'),
                                        'evidence': audit.get('hair_evidence')}
                save()
                decision = hair_decision(audit)
                repair_hair = decision == 'repair'
                record['hair_check']['decision'] = decision
                save()
                log('[refine] 头发检测：' + ('仍破碎，安排独立补修。' if repair_hair else '无需补修，直接进入背景。'))
        record['status'] = 'stopped' if record.get('stopped_at') else 'finished'
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
