import os
import json
import csv
import importlib
from PIL import Image

from utils.booru_tags import normalize_booru_tags

AUTOCOMPLETE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config-autocomplete.json")
DEFAULT_AUTOCOMPLETE_CSV_PATH = "data/tags/danbooru.csv"
LOCAL_TAGGER_MODEL_CANDIDATES = [
    "data/models/wd14/model.onnx",
    "models/wd14/model.onnx",
    "wd14_tagger_model/model.onnx"
]
LOCAL_TAGGER_TAGS_CANDIDATES = [
    "data/models/wd14/selected_tags.csv",
    "models/wd14/selected_tags.csv",
    "wd14_tagger_model/selected_tags.csv"
]
_WD14_RUNTIME_CACHE = {}


def load_autocomplete_config():
    config = {
        "enable_autocomplete": True,
        "csv_path": DEFAULT_AUTOCOMPLETE_CSV_PATH,
        "max_results": 50,
        "min_chars": 2,
        "local_booru_tagger_model_path": "",
        "local_booru_tagger_tags_path": "",
        "local_booru_tagger_max_tags": 60,
        "local_booru_tagger_general_threshold": 0.35,
        "local_booru_tagger_character_threshold": 0.35,
        "local_booru_tagger_meta_threshold": 0.75,
        "local_booru_tagger_rating_threshold": 0.75,
        "local_booru_tagger_keep_rating_tags": False,
        "local_booru_tagger_use_autocomplete_filter": True,
        "local_booru_tagger_output_style": "space"
    }
    if os.path.exists(AUTOCOMPLETE_CONFIG_PATH):
        try:
            with open(AUTOCOMPLETE_CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                config.update(loaded)
        except Exception:
            pass
    return config


def resolve_project_path(path_str):
    value = str(path_str or "").strip()
    if not value:
        return ""
    if os.path.isabs(value):
        return value
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), value)


def resolve_existing_path(path_candidates):
    for raw_path in path_candidates:
        candidate = resolve_project_path(raw_path)
        if candidate and os.path.exists(candidate):
            return candidate
    return ""


def load_autocomplete_tags_metadata():
    config = load_autocomplete_config()
    csv_path = resolve_project_path(config.get("csv_path", DEFAULT_AUTOCOMPLETE_CSV_PATH))
    tag_set = set()
    rank_map = {}
    if not csv_path or not os.path.exists(csv_path):
        return tag_set, rank_map, ""
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if not row:
                    continue
                tag = str(row[0]).strip().lower()
                if not tag:
                    continue
                tag_set.add(tag)
                rank_map[tag] = idx
    except Exception:
        return set(), {}, csv_path
    return tag_set, rank_map, csv_path


def load_wd14_labels(tags_csv_path):
    labels = []
    with open(tags_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "name" in [str(name).strip().lower() for name in reader.fieldnames]:
            for row in reader:
                name = str(row.get("name", "")).strip()
                if not name:
                    continue
                category_text = str(row.get("category", "")).strip()
                try:
                    category = int(float(category_text)) if category_text else 0
                except Exception:
                    category = 0
                labels.append((name, category))
            return labels
    with open(tags_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            name = str(row[1]).strip()
            if not name:
                continue
            category = 0
            if len(row) >= 3:
                try:
                    category = int(float(str(row[2]).strip()))
                except Exception:
                    category = 0
            labels.append((name, category))
    return labels


def resolve_local_tagger_paths():
    cfg = load_autocomplete_config()
    model_override = cfg.get("local_booru_tagger_model_path")
    tags_override = cfg.get("local_booru_tagger_tags_path")
    env_model = os.environ.get("LOCAL_BOORU_TAGGER_MODEL", "")
    env_tags = os.environ.get("LOCAL_BOORU_TAGGER_TAGS", "")
    model_path = resolve_existing_path([env_model, model_override] + LOCAL_TAGGER_MODEL_CANDIDATES)
    tags_path = resolve_existing_path([env_tags, tags_override] + LOCAL_TAGGER_TAGS_CANDIDATES)
    return model_path, tags_path


def merge_prompt_with_local_booru_tags(base_prompt, local_booru_tags):
    tags = normalize_booru_tags(local_booru_tags or [], limit=256, output_style="space")
    if not tags:
        return base_prompt
    joined = ", ".join(tags)
    extra = (
        "\n\nLocal booru tagger candidates:\n"
        + joined
        + "\nUse these as high-priority hints, then fix, delete, merge or add tags as needed and output only final optimized booru-tags."
    )
    return f"{base_prompt}{extra}"


def parse_int_with_default(raw_value, default_value, min_value=1):
    try:
        parsed = int(raw_value)
    except Exception:
        parsed = int(default_value)
    if parsed < min_value:
        parsed = min_value
    return parsed


def parse_float_with_default(raw_value, default_value, min_value=0.0, max_value=1.0):
    try:
        parsed = float(raw_value)
    except Exception:
        parsed = float(default_value)
    if parsed < min_value:
        parsed = min_value
    if parsed > max_value:
        parsed = max_value
    return parsed


def get_local_tagger_runtime_config(booru_tag_limit=30):
    cfg = load_autocomplete_config()
    runtime = {}
    runtime["max_tags"] = parse_int_with_default(cfg.get("local_booru_tagger_max_tags", booru_tag_limit), booru_tag_limit, min_value=1)
    runtime["general_threshold"] = parse_float_with_default(cfg.get("local_booru_tagger_general_threshold", 0.35), 0.35)
    runtime["character_threshold"] = parse_float_with_default(cfg.get("local_booru_tagger_character_threshold", 0.35), 0.35)
    runtime["meta_threshold"] = parse_float_with_default(cfg.get("local_booru_tagger_meta_threshold", 0.75), 0.75)
    runtime["rating_threshold"] = parse_float_with_default(cfg.get("local_booru_tagger_rating_threshold", 0.75), 0.75)
    runtime["keep_rating_tags"] = bool(cfg.get("local_booru_tagger_keep_rating_tags", False))
    runtime["use_autocomplete_filter"] = bool(cfg.get("local_booru_tagger_use_autocomplete_filter", True))
    output_style = str(cfg.get("local_booru_tagger_output_style", "space")).strip().lower()
    runtime["output_style"] = "space" if output_style == "space" else "underscore"
    return runtime


def predict_local_booru_tags(image_source, booru_tag_limit=30, log_callback=None):
    limit = int(booru_tag_limit) if str(booru_tag_limit).strip().isdigit() else 30
    if limit <= 0:
        limit = 30
    try:
        np = importlib.import_module("numpy")
        ort = importlib.import_module("onnxruntime")
    except Exception as e:
        if log_callback:
            log_callback(f"本地 booru tagger 未启用：导入 onnxruntime 或 numpy 失败（{e}），改为仅使用大模型")
            if "onnxruntime_pybind11_state" in str(e) or "DLL load failed" in str(e):
                log_callback("检测到 onnxruntime DLL 初始化失败，通常可通过在程序启动早期先加载 onnxruntime 后再初始化 PyQt 来修复")
        return []
    model_path, tags_path = resolve_local_tagger_paths()
    if not model_path or not tags_path:
        if log_callback:
            log_callback("本地 booru tagger 未启用：未找到 model.onnx 或 selected_tags.csv")
        return []
    cache_key = f"{model_path}|{tags_path}"
    session = None
    labels = None
    cached = _WD14_RUNTIME_CACHE.get(cache_key)
    if cached:
        session, labels = cached
    if session is None or labels is None:
        try:
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            labels = load_wd14_labels(tags_path)
            _WD14_RUNTIME_CACHE[cache_key] = (session, labels)
        except Exception as e:
            if log_callback:
                log_callback(f"本地 booru tagger 初始化失败: {e}")
            return []
    if not labels:
        if log_callback:
            log_callback("本地 booru tagger 初始化失败：selected_tags.csv 为空")
        return []
    runtime_cfg = get_local_tagger_runtime_config(booru_tag_limit=limit)
    tag_whitelist, tag_rank_map, autocomplete_csv_path = load_autocomplete_tags_metadata()
    try:
        if isinstance(image_source, str):
            image = Image.open(image_source)
        else:
            image = image_source.copy()
        if image.mode == "RGBA":
            base = Image.new("RGBA", image.size, "WHITE")
            base.alpha_composite(image)
            image = base.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        input_shape = session.get_inputs()[0].shape
        positive_dims = [dim for dim in input_shape if isinstance(dim, int) and dim > 4]
        target_size = max(positive_dims) if positive_dims else 448
        w, h = image.size
        scale = float(target_size) / float(max(w, h))
        resized_w = max(1, int(round(w * scale)))
        resized_h = max(1, int(round(h * scale)))
        resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        offset = ((target_size - resized_w) // 2, (target_size - resized_h) // 2)
        canvas.paste(resized, offset)
        image_array = np.asarray(canvas, dtype=np.float32)
        image_array = image_array[:, :, ::-1]
        image_array = np.expand_dims(image_array, axis=0)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: image_array})[0]
        if len(output.shape) == 2:
            probs = output[0]
        else:
            probs = output
        candidates = []
        for idx, score in enumerate(probs):
            if idx >= len(labels):
                break
            name, category = labels[idx]
            if category == 0:
                threshold = runtime_cfg["general_threshold"]
            elif category == 4:
                threshold = runtime_cfg["character_threshold"]
            elif category == 9:
                threshold = runtime_cfg["rating_threshold"]
            else:
                threshold = runtime_cfg["meta_threshold"]
            if category == 9 and not runtime_cfg["keep_rating_tags"]:
                continue
            if float(score) < threshold:
                continue
            normalized_name = str(name).strip().lower().replace(" ", "_")
            if not normalized_name:
                continue
            candidates.append((normalized_name, float(score)))
        candidates.sort(key=lambda item: item[1], reverse=True)
        filtered = []
        if runtime_cfg["use_autocomplete_filter"]:
            for tag_name, score in candidates:
                if tag_whitelist and tag_name not in tag_whitelist:
                    continue
                filtered.append((tag_name, score))
        if filtered and runtime_cfg["use_autocomplete_filter"]:
            filtered.sort(key=lambda item: (tag_rank_map.get(item[0], 10**9), -item[1]))
            tag_names = [tag for tag, _ in filtered[:runtime_cfg["max_tags"]]]
        else:
            tag_names = [tag for tag, _ in candidates[:runtime_cfg["max_tags"]]]
        normalized = normalize_booru_tags(tag_names, limit=limit, output_style=runtime_cfg["output_style"])
        if log_callback:
            if normalized:
                log_callback(f"本地 booru tagger 配置: max_tags={runtime_cfg['max_tags']}, threshold(general/character/meta/rating)={runtime_cfg['general_threshold']}/{runtime_cfg['character_threshold']}/{runtime_cfg['meta_threshold']}/{runtime_cfg['rating_threshold']}, keep_rating_tags={runtime_cfg['keep_rating_tags']}, output_style={runtime_cfg['output_style']}")
                if autocomplete_csv_path:
                    log_callback(f"本地 booru tagger 命中 {len(normalized)} 个标签（已按 autocomplete CSV 过滤）")
                else:
                    log_callback(f"本地 booru tagger 命中 {len(normalized)} 个标签")
            else:
                log_callback("本地 booru tagger 未产生可用标签，改为仅使用大模型")
        return normalized
    except Exception as e:
        if log_callback:
            log_callback(f"本地 booru tagger 推理失败: {e}")
        return []
