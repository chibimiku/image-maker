"""Bounded data-driven repaint experiments, one explicit job per invocation."""
import hashlib
import json
import os
import re
import time
from pathlib import Path


def measure_annotated_regions(manifest, annotations, output_dir):
    """Record HSV for manually marked pigment samples; never auto-pass fidelity."""
    import numpy as np
    from PIL import Image, ImageDraw
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, entry in manifest.items():
        im = Image.open(entry['path']).convert('RGB').resize(tuple(annotations['canvas']))
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Region input hash changed: ' + name)
        mask = Image.new('L', im.size)
        draw = ImageDraw.Draw(mask)
        regions = annotations['regions'][name]
        for polygon in regions['include']:
            draw.polygon([tuple(p) for p in polygon], fill=255)
        for polygon in regions.get('exclude', []):
            draw.polygon([tuple(p) for p in polygon], fill=0)
        pixels = np.asarray(im.convert('HSV'))[np.asarray(mask) > 0].astype(float)
        if len(pixels) == 0:
            raise ValueError('Empty pigment mask: ' + name)
        angles = pixels[:, 0] * (2 * np.pi / 255)
        hue = float(np.degrees(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())) % 360)
        result[name] = dict(sample_pixels=len(pixels), hue_circular_degrees=hue,
                            saturation_median=float(np.median(pixels[:, 1]) / 255),
                            value_median=float(np.median(pixels[:, 2]) / 255),
                            automatic_fidelity_verdict=None)
        mask.save(output / (name + '-pigment-mask.png'))
        # Crop-only evidence: keep artwork pixels untouched and show outline separately.
        preview = im.copy()
        d = ImageDraw.Draw(preview)
        for polygon in regions['include']:
            points = [tuple(p) for p in polygon]
            d.line(points + [points[0]], fill='red', width=1)
        preview.crop(tuple(annotations['preview_box'])).resize((900,720)).save(output / (name + '-mask-preview.png'))
    (output / 'region-metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_plan(plan_path, *, job=None, audit=None, dry_run=False, log=print):
    from modules.others.api_backend import (generate_image_repaint, generate_image_aigc2d_gpt,
                                          transmitted_image_digest, IMAGE_RETRY_ENV)
    from utils import send_budget
    from utils.analysis_gpt_prompt import load_text_api_config, call_text_model
    from utils.first_image_audit import extract_json_object
    from utils.gpt_image_optimize import validate_repaint_aspect
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    output = Path(plan['output_dir']).resolve()
    if bool(job) == bool(audit):
        raise ValueError('Select exactly one image job or text audit')
    label = job or audit
    if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
        raise ValueError('Unsafe job name')
    for filename, expected in plan['frozen_hashes'].items():
        if digest(filename) != expected:
            raise ValueError('Frozen input changed: ' + filename)
    spec = plan['jobs' if job else 'audits'][label]
    gpt = bool(job and spec.get('kind') == 'gpt')
    # All prompts are files; no hidden suffix or prompt rewriting.
    if gpt:
        request = dict(prompt=Path(spec['prompt_file']).read_text(encoding='utf-8'),
                       image_paths=spec.get('image_paths', []), model=spec['model'],
                       size=spec['size'], quality=spec['quality'], output_format=spec['output_format'],
                       api_type=spec['api_type'], mode=spec.get('mode'), n=1,
                       save_sub_dir=str(output), file_prefix=label)
    elif job:
        prompt = Path(spec['prompt_file']).read_text(encoding='utf-8')
        request = dict(source_paths=[spec['source']], model=plan['model'],
                       resolution=plan['resolution'], aspect_ratio=plan['aspect_ratio'],
                       api_type=plan['api_type'], prompt=prompt, prompt_suffix='',
                       use_detail_suffix=False, extra_reference_paths=spec.get('extra_reference_paths', []), repeat=1,
                       content_guard=False, refine=False, request_timeout=360, retry_limit=0,
                       save_sub_dir=str(output), file_prefix=label, budget_run_id=label)
    else:
        request = dict(system=Path(spec['system_file']).read_text(encoding='utf-8'),
                       user=Path(spec['user_file']).read_text(encoding='utf-8'),
                       images=spec['images'], timeout=180, max_tokens=4500)
        for path in request['images']:
            if not Path(path).is_file():
                raise ValueError('Audit input missing: ' + path)
    if job:
        for path in (request['image_paths'] if gpt else request['source_paths'] + request['extra_reference_paths']):
            if not Path(path).is_file():
                raise ValueError('Image input missing: ' + path)
    log(f"{'PREVIEW' if dry_run else 'SEND'} {label}: {'gpt' if gpt else 'image' if job else 'text'}")
    if dry_run:
        return request
    output.mkdir(parents=True, exist_ok=True)
    attempt_file = output / (label + '.request.json')
    # Exclusive creation prevents accidental resending, including unknown failures.
    with attempt_file.open('x', encoding='utf-8') as handle:
        json.dump(request, handle, ensure_ascii=False, indent=2)
    environment = {send_budget.ENV_DIR: str(output / 'budget'),
                   send_budget.ENV_RUN: plan['run_id'],
                   send_budget.ENV_LIMIT: str(plan['image_limit']),
                   send_budget.ENV_TEXT_LIMIT: str(plan['text_limit']),
                   send_budget.ENV_TEXT_PER_STAGE: '1', IMAGE_RETRY_ENV: '0'}
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    result = {'id': label, 'kind': 'image' if job else 'text'}
    started = time.monotonic()
    try:
        if job:
            if gpt:
                reservation = send_budget.reserve_image_attempt(
                    run_id=label, operation=label, model=request['model'], resolution=request['size'],
                    prompt_sha256=hashlib.sha256(request['prompt'].encode('utf-8')).hexdigest(),
                    prompt_chars=len(request['prompt']),
                    reference_images=[transmitted_image_digest(p) for p in request['image_paths']],
                    note='GPT generation experiment; automatic retries disabled')
                try:
                    paths = generate_image_aigc2d_gpt(**request, log_callback=log) or []
                except Exception as exc:
                    send_budget.settle_image_attempt(reservation, status='unknown', error=str(exc))
                    raise
                send_budget.settle_image_attempt(reservation, status='success' if paths else 'failed',
                                                output_path=paths[0] if paths else '',
                                                output_sha256=digest(paths[0]) if paths else '')
            else:
                paths = generate_image_repaint(**request, log_callback=log) or []
            result['outputs'] = paths
            result['output_hashes'] = {p: digest(p) for p in paths}
            if not paths:
                raise RuntimeError('No image returned; no retry')
            validate_repaint_aspect(paths[0], plan['aspect_ratio'])
        else:
            cfg = load_text_api_config()
            if cfg['model'] != plan['text_model']:
                raise ValueError('Text model changed since plan freeze')
            raw = send_budget.guarded_text_call(
                label, call_text_model, cfg['base_url'], cfg['api_key'], cfg['model'],
                request['system'], request['user'], image_paths=request['images'],
                timeout=request['timeout'], max_tokens=request['max_tokens'])
            (output / (label + '.raw.txt')).write_text(raw, encoding='utf-8')
            result['report'] = extract_json_object(raw)
            if not isinstance(result['report'], dict):
                raise ValueError('Malformed audit JSON; raw retained')
        result['status'] = 'returned'
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}', billed='unknown')
        log(result['error'])
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        (output / (label + '.result.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return result
