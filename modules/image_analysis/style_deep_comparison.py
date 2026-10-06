"""本地深度指标对照；计算在独立进程，GUI 不加载 torch/OpenVINO。"""
import base64
import hashlib
import html
import importlib.metadata
import json
import math
import mimetypes
import os
from pathlib import Path
import statistics
import subprocess
import sys

VERSION = "style-deep-comparison-v2"
METRICS = ("gram", "adain", "lpips", "csd")
from utils.style_similarity import LIMITS



def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_manifest(state):
    from modules.image_analysis.style_comparison import candidates_from_state
    refs = state.get("dataset", {}).get("images") or []
    candidates = candidates_from_state(state)
    if not refs or not candidates:
        raise ValueError("需要画风源图及已生成的候选图")
    if len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("候选 ID 重复")
    return {"references": [{"path": p, "sha256": file_hash(p)} for p in refs],
            "candidates": [{"id": c["id"], "path": c["path"], "sha256": file_hash(c["path"])} for c in candidates]}


from utils.style_similarity import aggregate, statistics_distance, resolve_backend, ALL_METRICS, LABELS


def write_report(result, directory, state=None):
    def esc(value):
        return html.escape(str(value))
    def picture(path):
        from utils.image_encoding import compress_and_encode_image
        mime, data = compress_and_encode_image(path, max_dim=720, quality=85)
        return f'<img style="max-height:240px;max-width:260px" src="data:{mime};base64,{data}">'
    parts = ['<!doctype html><meta charset="utf-8"><title>画风深度指标报告</title>',
             '<style>body{font-family:system-ui;margin:24px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
             '<h1>画风深度指标报告</h1><p>' + esc(LIMITS) + '</p>',
             '<pre>' + esc(json.dumps({k: result.get(k) for k in ("version", "backend", "input_hash", "models", "weights", "training_parameters", "self_check", "execution", "metric_contract", "face_contract", "region_annotations", "region_geometry")}, ensure_ascii=False, indent=2)) + '</pre>',
             '<h2>源图 · 原始参考数据</h2>' + ''.join('<figure>' + picture(r["path"]) + '<figcaption>参考 ' + str(i) + '：' + esc(r["path"]) + '</figcaption></figure>' for i, r in enumerate(result["inputs"]["references"])),
             '<h2>全部候选 · 全参考集等权均值</h2><table><tr><th>候选</th>' + ''.join('<th>' + esc(LABELS[m]) + '</th>' for m in ALL_METRICS) + '</tr>']
    for row in result["rows"]:
        parts.append('<tr><td>' + esc(row["id"]) + '</td>' + ''.join(
            '<td>' + (format(row["summary"].get(m, {}).get("mean"), '.8g') if row["summary"].get(m, {}).get("mean") is not None else '缺失') + '</td>'
            for m in ALL_METRICS) + '</tr>')
    parts.append('</table>')
    if any(row.get("face_summary") for row in result["rows"]):
        from utils.style_face_metrics import LABELS as FACE_LABELS
        parts.append('<h2>面部与发丝 · 已确认定位的完整配对均值</h2><p>自动定位为 provisional，只在逐对详情展示暂定数值，不计正式均值。不可见眼睛 / 未标发丝 / 视角不一致不补分。发丝连贯性只沿标出的发丝路径采样。</p><table><tr><th>候选</th><th>指标</th><th>均值</th><th>有效配对</th></tr>')
        for row in result["rows"]:
            for metric, summary in row.get("face_summary", {}).items():
                mean = format(summary["mean"], '.6g') if summary["mean"] is not None else '未形成正式均值'
                parts.append('<tr><td>' + esc(row["id"]) + '</td><td>' + esc(FACE_LABELS[metric]) + '</td><td>' + mean + '</td><td>' + esc(str(summary["count"]) + '/' + str(summary["expected"])) + '</td></tr>')
        parts.append('</table>')
    state = state or {}
    visual = state.get("automatic_comparison") or {}
    status = state.get("automatic_comparison_status") or {}
    parts.append('<h2>原有视觉评分与主体复核</h2><p>深度指标成功不代表主体复核通过，不自动选择或导入画风。复制参考角色可能让距离更低，但仍应排除。</p>')
    if status.get("status") == "failed" or not visual:
        parts.append('<p>视觉比较未完成；未选出最佳版本。已复核 ' + esc(status.get("evaluated", 0)) + '/' + esc(status.get("expected", len(result["rows"]))) + ' 个候选。' + esc(status.get("error", "尚未运行视觉比较")) + '</p>')
        scores = {a["id"]: a for a in status.get("partial_assessments", [])}
    else:
        parts.append('<p>' + esc(visual.get("formula", "")) + '；排序第一：' + esc(visual.get("best_id")) + '（观察报告后手动选用，尚不代表已写入配置）。</p>')
        if len(visual.get("tied_best_ids", [])) > 1:
            parts.append('<p>并列候选，需人工选择：' + esc('、'.join(visual["tied_best_ids"])) + '</p>')
        scores = {a["id"]: a for a in visual.get("assessments", [])}
    from modules.image_analysis.style_comparison import DIMENSIONS, DIMENSION_LABELS
    parts.append('<table><tr><th>候选</th>' + ''.join('<th>' + esc(DIMENSION_LABELS[k]) + '</th>' for k in DIMENSIONS) + '<th>主体符合度</th><th>复核门禁与依据</th></tr>')
    for row in result["rows"]:
        score = scores.get(row["id"], {})
        gates = [label for key, label in (("reference_content_copied", "复制参考内容"), ("major_structure_defect", "严重结构缺陷"), ("explicit_subject_mismatch", "主体不符"), ("uncertain", "证据不确定")) if score.get(key)]
        parts.append('<tr><td>' + esc(row["id"]) + '</td>' + ''.join('<td>' + esc(score.get(k, "待复核")) + '</td>' for k in (*DIMENSIONS, "subject_fidelity")) + '<td>' + esc(("排除：" + '、'.join(gates) if gates else ("门禁通过" if score else "待复核")) + '；' + str(score.get("reason", ""))) + '</td></tr>')
    parts.append('</table>')
    for row in result["rows"]:
        parts.extend(['<h2>' + esc(row["id"]) + '</h2>', picture(row["path"]),
                      '<details><summary>逐张源图数值、分层参数、错误与覆盖率</summary><pre>' + esc(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre></details>'])
    target = Path(directory) / ("style-similarity-v2.html" if result.get("version") == "style-similarity/2" else "style-similarity-v1.html" if result.get("version") == "style-similarity/1" else "deep-feature-comparison-gram-v2.html" if result.get("version") == VERSION else "deep-feature-comparison.html")
    target.write_text('\n'.join(parts), encoding="utf-8")
    return str(target.resolve())


def compute(state_path, requested="auto", progress=lambda message: None):
    from utils.style_similarity import compare_images
    state_path = Path(state_path).resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    inputs = input_manifest(state)
    result = compare_images(inputs, requested, progress, state_path.parent)
    output = state_path.parent / f"deep-feature-comparison-{result['backend']['actual']}-{result['backend']['precision']}-gram-v2.json"
    return publish(state_path, inputs, result, output)


def publish(state_path, inputs, result, output):
    latest = json.loads(state_path.read_text(encoding="utf-8"))
    if input_manifest(latest) != inputs:
        raise ValueError("计算期间源图或候选列表变化，结果未写回")
    if result.get("region_annotations") is not None:
        from utils.style_face_metrics import annotation_snapshot
        paths = [r["path"] for r in inputs["references"] + inputs["candidates"]]
        if annotation_snapshot(paths) != result["region_annotations"]:
            raise ValueError("发布报告前区域定位变化，结果未写回")
    result["training_parameters"] = latest.get("parameters", {})
    result["report_path"] = write_report(result, state_path.parent, latest)
    atomic_json(output, result)
    latest["deep_feature_comparison"] = result
    if latest.get("automatic_comparison"):
        from modules.image_analysis.style_comparison import write_comparison_report
        write_comparison_report(latest, latest["automatic_comparison"], str(state_path.parent))
    atomic_json(state_path, latest)
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_worker(state_path, device, parent=None, locate_regions=False, region_config=None):
    from PyQt6.QtCore import QThread, pyqtSignal
    class Worker(QThread):
        completed = pyqtSignal(object, str)
        progress = pyqtSignal(str)
        def run(self):
            root = Path(__file__).resolve().parents[2]
            log_path = Path(state_path).resolve().parent / "deep-feature-comparison.log"
            process = None
            try:
                python = Path(sys.executable)
                if python.name.lower() == "pythonw.exe":
                    python = python.with_name("python.exe")
                with log_path.open("w", encoding="utf-8") as log:
                    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
                    arguments = [str(python), "-u", str(root / "tools/style_metrics_verify.py"),
                                 "--comparison-state", str(Path(state_path).resolve()), "--device", device]
                    if locate_regions:
                        arguments.append("--locate-regions")
                        if region_config:
                            environment.update(dict(zip(("IMAGE_MAKER_REGIONS_BASE_URL", "IMAGE_MAKER_REGIONS_API_KEY", "IMAGE_MAKER_REGIONS_MODEL"), region_config)))
                    process = subprocess.Popen(arguments,
                                               cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    offset = 0
                    while process.poll() is None:
                        if self.isInterruptionRequested():
                            process.terminate()
                            process.wait()
                            raise RuntimeError("已取消深度指标计算；原结果保留")
                        with log_path.open(encoding="utf-8", errors="replace") as reader:
                            reader.seek(offset)
                            lines = reader.read().splitlines()
                            offset = reader.tell()
                        for line in lines:
                            if line.startswith("PROGRESS "):
                                self.progress.emit(line[9:])
                        self.msleep(200)
                if process.returncode:
                    raise RuntimeError(f"深度指标进程退出码 {process.returncode}；详情见 {log_path}\n" + log_path.read_text(encoding="utf-8", errors="replace")[-2500:])
                latest = json.loads(Path(state_path).read_text(encoding="utf-8"))
                self.completed.emit(latest["deep_feature_comparison"], "")
            except Exception as exc:
                self.completed.emit({}, str(exc))
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait()
    return Worker(parent)
