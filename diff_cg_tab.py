import base64
import json
import mimetypes
import os
from datetime import datetime
from PIL import Image

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QDoubleSpinBox,
    QProgressBar,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from openai import OpenAI

from utils.webui_img2img_client import WebuiImg2ImgClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_IMAGE_FILE = os.path.join(BASE_DIR, "config-image.json")
MERGE_MODE_SMART = "smart_anchor"
MERGE_MODE_FULL = "full_base"
MASK_BACKEND_GRABCUT = "grabcut"
MASK_BACKEND_GDINO_SAM2 = "gdino_sam2"
SEG_ROOT = os.path.join(BASE_DIR, "models", "segmentation")
GDINO_HF_MODEL_ID = "IDEA-Research/grounding-dino-base"
HF_CACHE_DIR = os.path.join(SEG_ROOT, "hf-cache")
SAM2_CONFIG_PATH = os.path.join(SEG_ROOT, "sam2", "sam2.1_hiera_l.yaml")
SAM2_WEIGHT_PATH = os.path.join(SEG_ROOT, "sam2", "sam2.1_hiera_large.pt")
_GDINO_SAM2_CACHE = {}


def _load_full_image_config():
    if not os.path.exists(CONFIG_IMAGE_FILE):
        return {}
    try:
        with open(CONFIG_IMAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_full_image_config(data):
    os.makedirs(os.path.dirname(CONFIG_IMAGE_FILE), exist_ok=True)
    with open(CONFIG_IMAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _guess_mime(path):
    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith("image/"):
        return mime
    return "image/png"


def _to_data_url(path):
    with open(path, "rb") as f:
        raw = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{_guess_mime(path)};base64,{raw}"


def _file_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _split_webui_parameters(parameters_text):
    text = str(parameters_text or "").strip()
    if not text:
        return "", ""
    lower = text.lower()
    marker = "\nnegative prompt:"
    idx = lower.find(marker)
    if idx < 0:
        marker = "negative prompt:"
        idx = lower.find(marker)
    if idx < 0:
        return text.strip(), ""
    positive = text[:idx].strip()
    tail = text[idx + len(marker):].strip()
    stop_tokens = [
        "\nsteps:",
        "\nsampler:",
        "\nseed:",
        "\nsize:",
        "\nmodel:",
        "\nclip skip:",
        "\nensd:",
        "\nversion:",
    ]
    lower_tail = tail.lower()
    end_idx = len(tail)
    for token in stop_tokens:
        p = lower_tail.find(token)
        if p >= 0 and p < end_idx:
            end_idx = p
    negative = tail[:end_idx].strip()
    return positive, negative


def _extract_prompts_from_image(image_path):
    if not image_path or not os.path.isfile(image_path):
        return {"prompt": "", "negative_prompt": ""}
    prompt_text = ""
    negative_text = ""
    try:
        with Image.open(image_path) as img:
            pnginfo = img.info or {}
            parameters = pnginfo.get("parameters") or pnginfo.get("prompt") or ""
            if isinstance(parameters, bytes):
                parameters = parameters.decode("utf-8", errors="replace")
            if parameters:
                prompt_text, negative_text = _split_webui_parameters(parameters)
            if not prompt_text and pnginfo.get("Description"):
                prompt_text = str(pnginfo.get("Description", "")).strip()
            if not prompt_text:
                exif = img.getexif() if hasattr(img, "getexif") else {}
                if exif:
                    raw_comment = exif.get(0x9286) or exif.get(0x010E) or ""
                    if isinstance(raw_comment, bytes):
                        raw_comment = raw_comment.decode("utf-8", errors="replace")
                    if raw_comment:
                        prompt_text, negative_text = _split_webui_parameters(str(raw_comment))
    except Exception:
        return {"prompt": "", "negative_prompt": ""}
    return {"prompt": str(prompt_text).strip(), "negative_prompt": str(negative_text).strip()}


def _merge_prompt_text(base_prompt, llm_prompt):
    base_text = str(base_prompt or "").strip()
    llm_text = str(llm_prompt or "").strip()
    if not base_text:
        return llm_text
    if not llm_text:
        return base_text
    if llm_text.lower() in base_text.lower():
        return base_text
    return f"{base_text}, {llm_text}"


def _merge_negative_text(base_negative, llm_negative):
    base_text = str(base_negative or "").strip()
    llm_text = str(llm_negative or "").strip()
    if not base_text:
        return llm_text
    if not llm_text:
        return base_text
    if llm_text.lower() in base_text.lower():
        return base_text
    return f"{base_text}, {llm_text}"


def _calc_scaled_wh(image_path, scale):
    scale_v = float(scale or 1.0)
    if scale_v <= 0:
        scale_v = 1.0
    with Image.open(image_path) as img:
        w0, h0 = img.size
    w = max(64, int(round(w0 * scale_v)))
    h = max(64, int(round(h0 * scale_v)))
    w = max(64, (w // 8) * 8)
    h = max(64, (h // 8) * 8)
    return w, h


def _build_subject_mask_grabcut(image_path, save_dir):
    cv2 = __import__("cv2")
    np = __import__("numpy")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"读取图片失败: {image_path}")
    h, w = img.shape[:2]
    if w < 32 or h < 32:
        raise ValueError("图片尺寸过小，无法生成主体掩码")
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    margin_x = max(8, int(w * 0.05))
    margin_y = max(8, int(h * 0.05))
    rect = (margin_x, margin_y, max(1, w - margin_x * 2), max(1, h - margin_y * 2))
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir,
        f"mask-{datetime.now().strftime('%H%M%S')}-{os.path.splitext(os.path.basename(image_path))[0]}.png",
    )
    cv2.imwrite(out_path, fg_mask)
    return out_path


def _get_gdino_sam2_models():
    cache_key = "default"
    if cache_key in _GDINO_SAM2_CACHE:
        return _GDINO_SAM2_CACHE[cache_key]
    sam2_cfg_candidates = [
        SAM2_CONFIG_PATH,
        os.path.join(SEG_ROOT, "sam2", "sam2_hiera_l.yaml"),
        os.path.join(SEG_ROOT, "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml"),
        os.path.join(SEG_ROOT, "sam2", "configs", "sam2", "sam2_hiera_l.yaml"),
    ]
    sam2_cfg_real = ""
    for p in sam2_cfg_candidates:
        if os.path.isfile(p):
            sam2_cfg_real = p
            break
    required = [SAM2_WEIGHT_PATH]
    missing = [p for p in required if not os.path.isfile(p)]
    if not sam2_cfg_real:
        missing.append("SAM2_CONFIG(候选路径均不存在)")
    if missing:
        raise FileNotFoundError(
            "SAM2 模型文件缺失，请先下载并放到固定目录:\n"
            f"- SAM2 配置可放: {sam2_cfg_candidates}\n"
            f"- {SAM2_WEIGHT_PATH}\n"
            f"缺失项: {missing}"
        )
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as e:
        raise RuntimeError(
            "导入 GroundingDINO(transformers)/SAM2 失败，请安装依赖: "
            "pip install transformers accelerate timm scipy safetensors && "
            "pip install git+https://github.com/facebookresearch/sam2.git"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    gdino_processor = AutoProcessor.from_pretrained(
        GDINO_HF_MODEL_ID,
        cache_dir=HF_CACHE_DIR,
    )
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_HF_MODEL_ID,
        cache_dir=HF_CACHE_DIR,
    )
    gdino_model = gdino_model.to(device)
    gdino_model.eval()
    sam2_model = build_sam2(sam2_cfg_real, SAM2_WEIGHT_PATH, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    obj = {
        "device": device,
        "gdino_processor": gdino_processor,
        "gdino_model": gdino_model,
        "sam2_predictor": sam2_predictor,
    }
    _GDINO_SAM2_CACHE[cache_key] = obj
    return obj


def _build_subject_mask_gdino_sam2(image_path, save_dir, text_prompt, box_threshold=0.30, text_threshold=0.25):
    np = __import__("numpy")
    cv2 = __import__("cv2")
    torch = __import__("torch")
    model_pack = _get_gdino_sam2_models()
    device = model_pack["device"]
    gdino_processor = model_pack["gdino_processor"]
    gdino_model = model_pack["gdino_model"]
    sam2_predictor = model_pack["sam2_predictor"]

    image_pil = Image.open(image_path).convert("RGB")
    detect_text = str(text_prompt or "person")
    if not detect_text.strip().endswith("."):
        detect_text = f"{detect_text.strip()}."
    inputs = gdino_processor(images=image_pil, text=detect_text, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = gdino_model(**inputs)
    post_fn = gdino_processor.post_process_grounded_object_detection
    # 兼容 transformers 不同版本参数名:
    # - 新版: threshold
    # - 旧版: box_threshold
    try:
        import inspect
        param_names = set(inspect.signature(post_fn).parameters.keys())
    except Exception:
        param_names = set()
    kwargs = {
        "text_threshold": float(text_threshold),
        "target_sizes": [image_pil.size[::-1]],
    }
    if "threshold" in param_names:
        kwargs["threshold"] = float(box_threshold)
    else:
        kwargs["box_threshold"] = float(box_threshold)
    results = post_fn(
        outputs,
        inputs["input_ids"],
        **kwargs,
    )
    result0 = results[0] if results else {}
    boxes = result0.get("boxes")
    labels = result0.get("labels", [])
    if boxes is None or len(boxes) == 0:
        raise ValueError("GroundingDINO 未检测到主体，请提高提示词匹配度或降低阈值")
    image_np = np.array(image_pil)

    h, w = image_np.shape[:2]
    boxes_t = boxes
    if not isinstance(boxes_t, torch.Tensor):
        boxes_t = torch.tensor(boxes_t)
    boxes_t = boxes_t.float().cpu()
    if float(boxes_t.max().item()) <= 1.5:
        # GroundingDINO 常见输出是 cx,cy,w,h 的归一化框
        x_c, y_c, bw, bh = boxes_t[:, 0], boxes_t[:, 1], boxes_t[:, 2], boxes_t[:, 3]
        x1 = (x_c - bw / 2.0) * w
        y1 = (y_c - bh / 2.0) * h
        x2 = (x_c + bw / 2.0) * w
        y2 = (y_c + bh / 2.0) * h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
    else:
        boxes_xyxy = boxes_t

    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, w - 1)
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, h - 1)

    sam2_predictor.set_image(image_np)
    final_mask = np.zeros((h, w), dtype="uint8")
    for b in boxes_xyxy.numpy():
        masks, scores, logits = sam2_predictor.predict(
            box=b,
            multimask_output=False,
        )
        if masks is None or len(masks) == 0:
            continue
        m = (masks[0] > 0).astype("uint8") * 255
        final_mask = np.maximum(final_mask, m)

    if int(final_mask.max()) == 0:
        raise ValueError("SAM2 未生成有效掩码")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir,
        f"mask-gdino-sam2-{datetime.now().strftime('%H%M%S')}-{os.path.splitext(os.path.basename(image_path))[0]}.png",
    )
    cv2.imwrite(out_path, final_mask)
    return out_path, len(boxes_xyxy), [str(p) for p in (labels or [])]


def _normalize_script(script_obj, target_count):
    if not isinstance(script_obj, dict):
        script_obj = {}
    shots = script_obj.get("shots", [])
    if not isinstance(shots, list):
        shots = []

    normalized = []
    for idx, shot in enumerate(shots, start=1):
        if not isinstance(shot, dict):
            continue
        normalized.append(
            {
                "index": idx,
                "title": str(shot.get("title", "")).strip(),
                "scene": str(shot.get("scene", "")).strip(),
                "prompt": str(shot.get("prompt", "")).strip(),
                "negative_prompt": str(shot.get("negative_prompt", "")).strip(),
                "steps": int(shot.get("steps", 28) or 28),
                "cfg_scale": float(shot.get("cfg_scale", 5.0) or 5.0),
                "denoising_strength": float(shot.get("denoising_strength", 0.58) or 0.58),
                "extra_payload": shot.get("extra_payload", {}) if isinstance(shot.get("extra_payload", {}), dict) else {},
            }
        )

    while len(normalized) < target_count:
        next_idx = len(normalized) + 1
        normalized.append(
            {
                "index": next_idx,
                "title": f"镜头{next_idx}",
                "scene": "",
                "prompt": "",
                "negative_prompt": "",
                "steps": 28,
                "cfg_scale": 5.0,
                "denoising_strength": 0.58,
                "extra_payload": {},
            }
        )

    normalized = normalized[:target_count]
    for idx, shot in enumerate(normalized, start=1):
        shot["index"] = idx
        shot["steps"] = max(1, int(shot.get("steps", 28)))
        shot["cfg_scale"] = max(0.0, float(shot.get("cfg_scale", 5.0)))
        ds = float(shot.get("denoising_strength", 0.58))
        if ds < 0.01:
            ds = 0.01
        if ds > 1.0:
            ds = 1.0
        shot["denoising_strength"] = ds

    return {
        "title": str(script_obj.get("title", "差分CG脚本")).strip() or "差分CG脚本",
        "summary": str(script_obj.get("summary", "")).strip(),
        "shots": normalized,
    }


class DiffCgScriptThread(QThread):
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, llm_cfg, image_path, shot_count, story_desc, base_prompt="", base_negative=""):
        super().__init__()
        self.llm_cfg = llm_cfg
        self.image_path = image_path
        self.shot_count = shot_count
        self.story_desc = story_desc
        self.base_prompt = str(base_prompt or "").strip()
        self.base_negative = str(base_negative or "").strip()

    def run(self):
        try:
            base_url = self.llm_cfg.get("base_url", "").strip()
            api_key = self.llm_cfg.get("api_key", "").strip()
            model = self.llm_cfg.get("model", "").strip()
            if not api_key:
                raise ValueError("请先在全局配置里填写文本分析 API Key")
            if not model:
                raise ValueError("请先在全局配置里填写文本分析模型")
            if not os.path.isfile(self.image_path):
                raise ValueError("输入图片不存在")

            self.log.emit("开始调用 LLM 生成差分CG剧本 JSON ...")
            client = OpenAI(api_key=api_key, base_url=base_url)
            image_data_url = _to_data_url(self.image_path)

            system_prompt = (
                "你是资深分镜导演和 SD 提示词工程师。"
                "必须输出严格 JSON 对象，顶层包含 title、summary、shots。"
                "shots 必须是数组，长度等于用户要求。"
                "每个 shot 必须包含: index,title,scene,prompt,negative_prompt,steps,cfg_scale,denoising_strength,extra_payload。"
                "其中 prompt 必须为英文可直接用于 SD WebUI img2img。"
                "extra_payload 是 JSON 对象，可放 alwayson_scripts、override_settings 等插件参数；不需要时给空对象。"
                "请保证镜头之间有连续变化，适合作为差分CG序列。"
            )
            user_text = (
                f"请基于输入CG图，生成 {int(self.shot_count)} 张差分CG分镜脚本。\n"
                f"可选剧情描述：{self.story_desc or '无'}\n"
                "要求：变化平滑、每镜头主体一致但动作/表情/构图有递进变化。\n"
                f"原图已有正向prompt（用于保留lora等加载）：{self.base_prompt or '无'}\n"
                f"原图已有负向prompt：{self.base_negative or '无'}\n"
                "请在构图变化描述基础上输出每个镜头的增量 prompt（也可完整prompt），最终会和原图prompt合并。"
            )

            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    },
                ],
                temperature=0.6,
            )
            content = (resp.choices[0].message.content or "").strip()
            raw_json = json.loads(content) if content else {}
            self.finished_ok.emit(_normalize_script(raw_json, int(self.shot_count)))
        except Exception as e:
            self.failed.emit(str(e))


class DiffCgAnchorThread(QThread):
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, llm_cfg, image_path, source_prompt, source_negative):
        super().__init__()
        self.llm_cfg = llm_cfg
        self.image_path = image_path
        self.source_prompt = str(source_prompt or "").strip()
        self.source_negative = str(source_negative or "").strip()

    def run(self):
        try:
            base_url = self.llm_cfg.get("base_url", "").strip()
            api_key = self.llm_cfg.get("api_key", "").strip()
            model = self.llm_cfg.get("model", "").strip()
            if not api_key:
                raise ValueError("请先配置文本分析 API Key")
            if not model:
                raise ValueError("请先配置文本分析模型")
            if not os.path.isfile(self.image_path):
                raise ValueError("输入图片不存在")

            self.log.emit("开始智能提取人物锚点 prompt ...")
            client = OpenAI(api_key=api_key, base_url=base_url)
            image_data_url = _to_data_url(self.image_path)
            system_prompt = (
                "你是 Stable Diffusion 提示词精简专家。"
                "请从原始 prompt/negative 中提取“人物身份锚点”，保留角色稳定性和lora能力，删除构图和场景限制。"
                "必须返回 JSON 对象，包含 keep_positive, keep_negative, removed_notes。"
                "keep_positive 重点保留：<lora:...>、角色名、发色、瞳色、服饰关键词、核心风格触发词。"
                "尽量删除：camera angle、background、pose、lighting、composition、shot size 等镜头约束。"
            )
            user_text = (
                f"原始正向prompt:\n{self.source_prompt or '无'}\n\n"
                f"原始负向prompt:\n{self.source_negative or '无'}\n\n"
                "请输出适合“差分CG多镜头变化”的锚点版本。"
            )
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    },
                ],
                temperature=0.2,
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = json.loads(content) if content else {}
            keep_positive = str(parsed.get("keep_positive", "")).strip()
            keep_negative = str(parsed.get("keep_negative", "")).strip()
            if not keep_positive:
                keep_positive = self.source_prompt
            if not keep_negative:
                keep_negative = self.source_negative
            self.finished_ok.emit(
                {
                    "keep_positive": keep_positive,
                    "keep_negative": keep_negative,
                    "removed_notes": str(parsed.get("removed_notes", "")).strip(),
                }
            )
        except Exception as e:
            self.failed.emit(str(e))


class DiffCgGenerateThread(QThread):
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, request_data):
        super().__init__()
        self.request_data = request_data

    def run(self):
        try:
            cfg = self.request_data
            base_image = cfg["image_path"]
            script_obj = cfg["script"]
            chain_mode = bool(cfg["chain_mode"])
            global_negative = cfg["global_negative"]
            global_extra_payload = cfg["global_extra_payload"]
            base_prompt = cfg.get("base_prompt", "")
            base_negative = cfg.get("base_negative", "")
            merge_base_prompt = bool(cfg.get("merge_base_prompt", True))
            use_shot_params = bool(cfg.get("use_shot_params", False))
            anchor_prompt = cfg.get("anchor_prompt", "")
            anchor_negative = cfg.get("anchor_negative", "")
            merge_mode = str(cfg.get("merge_mode", "smart_anchor")).strip() or "smart_anchor"
            enable_subject_inpaint = bool(cfg.get("enable_subject_inpaint", False))
            inpaint_mask_blur = int(cfg.get("inpaint_mask_blur", 8))
            inpaint_fill = int(cfg.get("inpaint_fill", 1))
            inpaint_full_res = bool(cfg.get("inpaint_full_res", True))
            inpaint_padding = int(cfg.get("inpaint_padding", 32))
            mask_backend = str(cfg.get("mask_backend", MASK_BACKEND_GRABCUT)).strip() or MASK_BACKEND_GRABCUT
            detect_prompt = str(cfg.get("detect_prompt", "person . 1girl . 1boy")).strip() or "person"
            gdino_box_threshold = float(cfg.get("gdino_box_threshold", 0.30))
            gdino_text_threshold = float(cfg.get("gdino_text_threshold", 0.25))
            mask_save_dir = os.path.join("data", datetime.now().strftime("%Y%m%d"), "diff-cg", "masks")

            client = WebuiImg2ImgClient(
                base_url=cfg["webui_base_url"],
                timeout=int(cfg["timeout"]),
            )
            outputs = []
            prev_image = base_image
            shots = script_obj.get("shots", [])
            if not shots:
                raise ValueError("剧本 JSON 中没有 shots")

            self.log.emit(f"开始执行差分CG生成，共 {len(shots)} 张 ...")
            for i, shot in enumerate(shots, start=1):
                source_image = prev_image if chain_mode else base_image
                prompt = str(shot.get("prompt", "")).strip()
                if not prompt:
                    raise ValueError(f"第 {i} 张缺少 prompt")

                shot_neg = str(shot.get("negative_prompt", "")).strip()
                final_negative = shot_neg or global_negative
                if use_shot_params:
                    steps = int(shot.get("steps", cfg["default_steps"]))
                    cfg_scale = float(shot.get("cfg_scale", cfg["default_cfg_scale"]))
                    denoise = float(shot.get("denoising_strength", cfg["default_denoise"]))
                else:
                    steps = int(cfg["default_steps"])
                    cfg_scale = float(cfg["default_cfg_scale"])
                    denoise = float(cfg["default_denoise"])
                if denoise < 0.01:
                    denoise = 0.01
                if denoise > 1.0:
                    denoise = 1.0

                final_prompt = prompt
                if merge_base_prompt:
                    if merge_mode == "full_base":
                        final_prompt = _merge_prompt_text(base_prompt, final_prompt)
                        final_negative = _merge_negative_text(base_negative, final_negative)
                    else:
                        final_prompt = _merge_prompt_text(anchor_prompt or base_prompt, final_prompt)
                        final_negative = _merge_negative_text(anchor_negative or base_negative, final_negative)

                extra_payload = {}
                extra_payload.update(global_extra_payload)
                if isinstance(shot.get("extra_payload"), dict):
                    extra_payload.update(shot.get("extra_payload"))
                if not extra_payload:
                    extra_payload = None
                if enable_subject_inpaint:
                    if mask_backend == MASK_BACKEND_GDINO_SAM2:
                        mask_path, box_count, phrases = _build_subject_mask_gdino_sam2(
                            source_image,
                            mask_save_dir,
                            text_prompt=detect_prompt,
                            box_threshold=gdino_box_threshold,
                            text_threshold=gdino_text_threshold,
                        )
                        self.log.emit(f"  GDINO检测框数: {box_count}, phrases: {phrases[:5]}")
                    else:
                        mask_path = _build_subject_mask_grabcut(source_image, mask_save_dir)
                    mask_b64 = _file_to_base64(mask_path)
                    if extra_payload is None:
                        extra_payload = {}
                    extra_payload["mask"] = mask_b64
                    if "mask_blur" not in extra_payload:
                        extra_payload["mask_blur"] = max(0, inpaint_mask_blur)
                    if "inpainting_fill" not in extra_payload:
                        extra_payload["inpainting_fill"] = max(0, inpaint_fill)
                    if "inpaint_full_res" not in extra_payload:
                        extra_payload["inpaint_full_res"] = bool(inpaint_full_res)
                    if "inpaint_full_res_padding" not in extra_payload:
                        extra_payload["inpaint_full_res_padding"] = max(0, inpaint_padding)
                    if "inpainting_mask_invert" not in extra_payload:
                        extra_payload["inpainting_mask_invert"] = 0
                    self.log.emit(f"  Inpaint掩码: {mask_path}")

                self.log.emit(f"[{i}/{len(shots)}] 生成中: {shot.get('title', f'镜头{i}')}")
                self.log.emit(
                    f"  参数: steps={steps}, denoise={denoise:.3f}, effective≈{int(round(steps * denoise))}, cfg={cfg_scale}"
                )
                result = client.img2img_image_file(
                    image_path=source_image,
                    prompt=final_prompt,
                    negative_prompt=final_negative,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    denoising_strength=denoise,
                    num_images=1,
                    seed=int(cfg["seed"]),
                    width=cfg["width"],
                    height=cfg["height"],
                    sampler_name=cfg["sampler_name"],
                    scheduler=cfg["scheduler"],
                    sd_model=cfg["sd_model"],
                    sd_vae=cfg["sd_vae"],
                    extra_payload=extra_payload,
                    return_base64=False,
                    output_dir="data",
                )
                rows = result.get("outputs", []) if isinstance(result, dict) else []
                if not rows:
                    raise ValueError(f"第 {i} 张未返回输出图片")
                out_path = str(rows[0].get("file_path", "")).strip()
                if not out_path or not os.path.isfile(out_path):
                    raise ValueError(f"第 {i} 张输出图片路径无效")
                outputs.append(
                    {
                        "index": i,
                        "title": shot.get("title", f"镜头{i}"),
                        "source_image": source_image,
                        "output_image": out_path,
                        "prompt": final_prompt,
                        "negative_prompt": final_negative,
                    }
                )
                prev_image = out_path

            save_dir = os.path.join("data", datetime.now().strftime("%Y%m%d"), "diff-cg")
            os.makedirs(save_dir, exist_ok=True)
            manifest = {
                "created_at": datetime.now().isoformat(),
                "chain_mode": chain_mode,
                "input_image": base_image,
                "title": script_obj.get("title", "差分CG脚本"),
                "summary": script_obj.get("summary", ""),
                "shots": script_obj.get("shots", []),
                "outputs": outputs,
            }
            manifest_path = os.path.join(
                save_dir,
                f"diff-cg-{datetime.now().strftime('%H%M%S')}.json",
            )
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            self.finished_ok.emit({"manifest_path": manifest_path, "outputs": outputs})
        except Exception as e:
            self.failed.emit(str(e))


class MaskPreviewWorker(QThread):
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, request):
        super().__init__()
        self.request = request

    def run(self):
        try:
            req = self.request or {}
            image_path = req.get("image_path", "")
            backend = req.get("backend", MASK_BACKEND_GRABCUT)
            save_dir = req.get("save_dir", "")
            if backend == MASK_BACKEND_GDINO_SAM2:
                mask_path, box_count, phrases = _build_subject_mask_gdino_sam2(
                    image_path=image_path,
                    save_dir=save_dir,
                    text_prompt=req.get("text_prompt", "person"),
                    box_threshold=float(req.get("box_threshold", 0.30)),
                    text_threshold=float(req.get("text_threshold", 0.25)),
                )
                self.finished_ok.emit(
                    {
                        "mask_path": mask_path,
                        "backend": backend,
                        "box_count": box_count,
                        "phrases": phrases,
                    }
                )
                return
            mask_path = _build_subject_mask_grabcut(image_path, save_dir)
            self.finished_ok.emit(
                {
                    "mask_path": mask_path,
                    "backend": backend,
                    "box_count": 0,
                    "phrases": [],
                }
            )
        except Exception as e:
            self.failed.emit(str(e))


class DiffCgTabWidget(QWidget):
    def __init__(self, text_config_getter_func):
        super().__init__()
        self.get_text_config = text_config_getter_func
        self.current_image_path = ""
        self.base_prompt_text = ""
        self.base_negative_text = ""
        self.anchor_prompt_text = ""
        self.anchor_negative_text = ""
        self.script_thread = None
        self.anchor_thread = None
        self.gen_thread = None
        self.preview_worker = None
        self.preview_pending = False
        self.preview_busy = False
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._render_inpaint_preview)
        self.init_ui()
        self.load_config()

    def init_ui(self):
        layout = QVBoxLayout(self)

        script_group = QGroupBox("步骤1：生成差分CG剧本(JSON)")
        script_form = QFormLayout()

        image_row = QHBoxLayout()
        self.image_input = QLineEdit()
        self.image_input.setPlaceholderText("选择一张基础CG图片")
        image_row.addWidget(self.image_input, stretch=1)
        self.pick_image_btn = QPushButton("选择图片")
        self.pick_image_btn.clicked.connect(self.pick_image)
        image_row.addWidget(self.pick_image_btn)
        script_form.addRow("基础CG:", image_row)

        self.shot_count_spin = QSpinBox()
        self.shot_count_spin.setRange(1, 30)
        self.shot_count_spin.setValue(6)
        self.shot_count_spin.setMaximumWidth(90)
        script_form.addRow("差分张数:", self.shot_count_spin)

        self.story_desc_edit = QTextEdit()
        self.story_desc_edit.setMinimumHeight(70)
        self.story_desc_edit.setPlaceholderText("可选：简单描述剧情方向、情绪递进、动作变化等")
        script_form.addRow("剧情描述(可选):", self.story_desc_edit)

        self.base_prompt_edit = QTextEdit()
        self.base_prompt_edit.setMinimumHeight(70)
        self.base_prompt_edit.setPlaceholderText("从原图读取到的正向 prompt（可手动编辑，用于保留lora等）")
        script_form.addRow("原图正向Prompt:", self.base_prompt_edit)

        self.base_negative_edit = QTextEdit()
        self.base_negative_edit.setMinimumHeight(55)
        self.base_negative_edit.setPlaceholderText("从原图读取到的负向 prompt（可手动编辑）")
        script_form.addRow("原图负向Prompt:", self.base_negative_edit)

        self.anchor_prompt_edit = QTextEdit()
        self.anchor_prompt_edit.setMinimumHeight(70)
        self.anchor_prompt_edit.setPlaceholderText("智能锚点正向（建议仅保留人物身份+lora触发词）")
        script_form.addRow("智能锚点正向:", self.anchor_prompt_edit)

        self.anchor_negative_edit = QTextEdit()
        self.anchor_negative_edit.setMinimumHeight(55)
        self.anchor_negative_edit.setPlaceholderText("智能锚点负向（可选）")
        script_form.addRow("智能锚点负向:", self.anchor_negative_edit)

        btn_row = QHBoxLayout()
        self.read_prompt_btn = QPushButton("读取原图Prompt")
        self.read_prompt_btn.clicked.connect(self.read_prompt_from_image)
        btn_row.addWidget(self.read_prompt_btn)
        self.extract_anchor_btn = QPushButton("智能提取人物锚点")
        self.extract_anchor_btn.clicked.connect(self.extract_anchor_with_llm)
        btn_row.addWidget(self.extract_anchor_btn)
        self.gen_script_btn = QPushButton("生成JSON剧本")
        self.gen_script_btn.clicked.connect(self.generate_script)
        btn_row.addWidget(self.gen_script_btn)
        self.normalize_script_btn = QPushButton("按张数规范化JSON")
        self.normalize_script_btn.clicked.connect(self.normalize_script_json)
        btn_row.addWidget(self.normalize_script_btn)
        script_form.addRow("", btn_row)

        self.script_json_edit = QTextEdit()
        self.script_json_edit.setMinimumHeight(220)
        self.script_json_edit.setPlaceholderText("这里会显示/编辑差分CG剧本 JSON")
        script_form.addRow("剧本JSON:", self.script_json_edit)

        script_group.setLayout(script_form)
        layout.addWidget(script_group)

        webui_group = QGroupBox("步骤2：执行 SD WebUI Img2Img 生成")
        webui_root_layout = QHBoxLayout()
        webui_left_widget = QWidget()
        webui_form = QFormLayout(webui_left_widget)
        self.webui_base_url_input = QLineEdit("http://127.0.0.1:7860")
        webui_form.addRow("WebUI地址:", self.webui_base_url_input)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(10, 3600)
        self.timeout_spin.setValue(600)
        self.timeout_spin.setSuffix(" 秒")
        self.timeout_spin.setMaximumWidth(120)
        webui_form.addRow("请求超时:", self.timeout_spin)

        self.global_negative_edit = QTextEdit()
        self.global_negative_edit.setMinimumHeight(55)
        self.global_negative_edit.setPlaceholderText("全局负面词（若分镜里有 negative_prompt 则优先用分镜值）")
        webui_form.addRow("全局Negative:", self.global_negative_edit)

        row1 = QHBoxLayout()
        self.default_steps_spin = QSpinBox()
        self.default_steps_spin.setRange(1, 200)
        self.default_steps_spin.setValue(28)
        self.default_steps_spin.setMaximumWidth(90)
        row1.addWidget(QLabel("默认步数"))
        row1.addWidget(self.default_steps_spin)
        self.default_cfg_spin = QDoubleSpinBox()
        self.default_cfg_spin.setRange(0.0, 30.0)
        self.default_cfg_spin.setSingleStep(0.1)
        self.default_cfg_spin.setValue(5.0)
        self.default_cfg_spin.setMaximumWidth(90)
        row1.addWidget(QLabel("默认CFG"))
        row1.addWidget(self.default_cfg_spin)
        self.default_denoise_spin = QDoubleSpinBox()
        self.default_denoise_spin.setRange(0.01, 1.0)
        self.default_denoise_spin.setSingleStep(0.01)
        self.default_denoise_spin.setValue(0.58)
        self.default_denoise_spin.setMaximumWidth(90)
        row1.addWidget(QLabel("默认Denoise"))
        row1.addWidget(self.default_denoise_spin)
        row1.addStretch(1)
        webui_form.addRow("默认参数:", row1)

        row2 = QHBoxLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2147483647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setMaximumWidth(130)
        row2.addWidget(QLabel("Seed"))
        row2.addWidget(self.seed_spin)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.25, 4.0)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setValue(1.00)
        self.scale_spin.setMaximumWidth(90)
        row2.addWidget(QLabel("放大倍数"))
        row2.addWidget(self.scale_spin)
        row2.addStretch(1)
        webui_form.addRow("尺寸/随机种:", row2)

        self.sampler_input = QLineEdit("Euler a")
        webui_form.addRow("Sampler:", self.sampler_input)
        self.scheduler_input = QLineEdit("Automatic")
        webui_form.addRow("Scheduler:", self.scheduler_input)
        self.sd_model_input = QLineEdit("")
        self.sd_model_input.setPlaceholderText("可选：sd_model_checkpoint")
        webui_form.addRow("Checkpoint:", self.sd_model_input)
        self.sd_vae_input = QLineEdit("Automatic")
        webui_form.addRow("VAE:", self.sd_vae_input)

        self.global_extra_payload_edit = QTextEdit()
        self.global_extra_payload_edit.setMinimumHeight(80)
        self.global_extra_payload_edit.setPlaceholderText(
            "可选：全局附加 payload JSON，可放 alwayson_scripts / override_settings 等插件参数"
        )
        webui_form.addRow("全局插件Payload(JSON):", self.global_extra_payload_edit)

        self.chain_mode_check = QCheckBox("链式差分（上一张作为下一张输入）")
        self.chain_mode_check.setChecked(True)
        webui_form.addRow("模式:", self.chain_mode_check)
        self.merge_base_prompt_check = QCheckBox("执行时合并原图Prompt（保留lora等）")
        self.merge_base_prompt_check.setChecked(True)
        webui_form.addRow("Prompt合并:", self.merge_base_prompt_check)
        self.merge_mode_combo = QComboBox()
        self.merge_mode_combo.addItem("智能锚点（推荐）", MERGE_MODE_SMART)
        self.merge_mode_combo.addItem("完整原始Prompt", MERGE_MODE_FULL)
        webui_form.addRow("合并模式:", self.merge_mode_combo)
        self.use_shot_params_check = QCheckBox("使用分镜JSON中的steps/cfg/denoise（默认关闭）")
        self.use_shot_params_check.setChecked(False)
        webui_form.addRow("参数来源:", self.use_shot_params_check)
        self.enable_subject_inpaint_check = QCheckBox("启用主体Inpaint（本地GrabCut掩码）")
        self.enable_subject_inpaint_check.setChecked(False)
        webui_form.addRow("主体Inpaint:", self.enable_subject_inpaint_check)
        self.mask_backend_combo = QComboBox()
        self.mask_backend_combo.addItem("GrabCut（轻量兜底）", MASK_BACKEND_GRABCUT)
        self.mask_backend_combo.addItem("GroundingDINO + SAM2（推荐）", MASK_BACKEND_GDINO_SAM2)
        webui_form.addRow("掩码后端:", self.mask_backend_combo)
        self.detect_prompt_input = QLineEdit("person . 1girl . 1boy")
        self.detect_prompt_input.setPlaceholderText("GroundingDINO 检测文本提示")
        webui_form.addRow("检测提示词:", self.detect_prompt_input)
        gdino_row = QHBoxLayout()
        self.gdino_box_threshold_spin = QDoubleSpinBox()
        self.gdino_box_threshold_spin.setRange(0.05, 0.95)
        self.gdino_box_threshold_spin.setSingleStep(0.01)
        self.gdino_box_threshold_spin.setValue(0.30)
        self.gdino_box_threshold_spin.setMaximumWidth(90)
        gdino_row.addWidget(QLabel("Box阈值"))
        gdino_row.addWidget(self.gdino_box_threshold_spin)
        self.gdino_text_threshold_spin = QDoubleSpinBox()
        self.gdino_text_threshold_spin.setRange(0.05, 0.95)
        self.gdino_text_threshold_spin.setSingleStep(0.01)
        self.gdino_text_threshold_spin.setValue(0.25)
        self.gdino_text_threshold_spin.setMaximumWidth(90)
        gdino_row.addWidget(QLabel("Text阈值"))
        gdino_row.addWidget(self.gdino_text_threshold_spin)
        gdino_row.addStretch(1)
        webui_form.addRow("GDINO阈值:", gdino_row)
        inpaint_row = QHBoxLayout()
        self.inpaint_mask_blur_spin = QSpinBox()
        self.inpaint_mask_blur_spin.setRange(0, 64)
        self.inpaint_mask_blur_spin.setValue(8)
        self.inpaint_mask_blur_spin.setMaximumWidth(90)
        inpaint_row.addWidget(QLabel("MaskBlur"))
        inpaint_row.addWidget(self.inpaint_mask_blur_spin)
        self.inpaint_fill_spin = QSpinBox()
        self.inpaint_fill_spin.setRange(0, 3)
        self.inpaint_fill_spin.setValue(1)
        self.inpaint_fill_spin.setMaximumWidth(90)
        self.inpaint_fill_spin.setToolTip("0=fill,1=original,2=latent noise,3=latent nothing")
        inpaint_row.addWidget(QLabel("Fill"))
        inpaint_row.addWidget(self.inpaint_fill_spin)
        self.inpaint_full_res_check = QCheckBox("FullRes")
        self.inpaint_full_res_check.setChecked(True)
        inpaint_row.addWidget(self.inpaint_full_res_check)
        self.inpaint_padding_spin = QSpinBox()
        self.inpaint_padding_spin.setRange(0, 256)
        self.inpaint_padding_spin.setValue(32)
        self.inpaint_padding_spin.setMaximumWidth(90)
        inpaint_row.addWidget(QLabel("Padding"))
        inpaint_row.addWidget(self.inpaint_padding_spin)
        inpaint_row.addStretch(1)
        webui_form.addRow("Inpaint参数:", inpaint_row)

        self.run_btn = QPushButton("执行差分CG生成")
        self.run_btn.clicked.connect(self.run_generation)
        webui_form.addRow("", self.run_btn)
        webui_root_layout.addWidget(webui_left_widget, stretch=3)

        webui_right_widget = QWidget()
        webui_right_layout = QVBoxLayout(webui_right_widget)
        webui_right_layout.setContentsMargins(0, 0, 0, 0)
        webui_right_layout.addWidget(QLabel("原图预览:"))
        self.preview_input_label = QLabel("原图预览")
        self.preview_input_label.setAlignment(Qt.AlignCenter)
        self.preview_input_label.setMinimumHeight(220)
        self.preview_input_label.setMinimumWidth(320)
        self.preview_input_label.setStyleSheet("QLabel { border: 1px solid #ccc; background: #f7f7f7; }")
        webui_right_layout.addWidget(self.preview_input_label)
        webui_right_layout.addWidget(QLabel("掩码预览:"))
        self.preview_mask_label = QLabel("掩码预览")
        self.preview_mask_label.setAlignment(Qt.AlignCenter)
        self.preview_mask_label.setMinimumHeight(220)
        self.preview_mask_label.setMinimumWidth(320)
        self.preview_mask_label.setStyleSheet("QLabel { border: 1px solid #ccc; background: #f7f7f7; }")
        webui_right_layout.addWidget(self.preview_mask_label)
        self.preview_status_label = QLabel("预览状态: 空闲")
        webui_right_layout.addWidget(self.preview_status_label)
        self.preview_progress = QProgressBar()
        self.preview_progress.setRange(0, 0)
        self.preview_progress.setVisible(False)
        webui_right_layout.addWidget(self.preview_progress)
        webui_right_layout.addStretch(1)
        webui_root_layout.addWidget(webui_right_widget, stretch=2)

        webui_group.setLayout(webui_root_layout)
        layout.addWidget(webui_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        layout.addWidget(self.log_text)

        self.image_input.textChanged.connect(self._on_image_text_changed)
        self.webui_base_url_input.editingFinished.connect(self.save_config)
        self.timeout_spin.valueChanged.connect(lambda _: self.save_config())
        self.shot_count_spin.valueChanged.connect(lambda _: self.save_config())
        self.story_desc_edit.textChanged.connect(self.save_config)
        self.base_prompt_edit.textChanged.connect(self._on_base_prompt_changed)
        self.base_negative_edit.textChanged.connect(self._on_base_negative_changed)
        self.anchor_prompt_edit.textChanged.connect(self._on_anchor_prompt_changed)
        self.anchor_negative_edit.textChanged.connect(self._on_anchor_negative_changed)
        self.global_negative_edit.textChanged.connect(self.save_config)
        self.default_steps_spin.valueChanged.connect(lambda _: self.save_config())
        self.default_cfg_spin.valueChanged.connect(lambda _: self.save_config())
        self.default_denoise_spin.valueChanged.connect(lambda _: self.save_config())
        self.seed_spin.valueChanged.connect(lambda _: self.save_config())
        self.scale_spin.valueChanged.connect(lambda _: self.save_config())
        self.sampler_input.editingFinished.connect(self.save_config)
        self.scheduler_input.editingFinished.connect(self.save_config)
        self.sd_model_input.editingFinished.connect(self.save_config)
        self.sd_vae_input.editingFinished.connect(self.save_config)
        self.global_extra_payload_edit.textChanged.connect(self.save_config)
        self.chain_mode_check.toggled.connect(lambda _: self.save_config())
        self.merge_base_prompt_check.toggled.connect(lambda _: self.save_config())
        self.merge_mode_combo.currentIndexChanged.connect(lambda _idx: self.save_config())
        self.use_shot_params_check.toggled.connect(lambda _: self.save_config())
        self.enable_subject_inpaint_check.toggled.connect(lambda _: self.save_config())
        self.enable_subject_inpaint_check.toggled.connect(self._schedule_inpaint_preview)
        self.mask_backend_combo.currentIndexChanged.connect(lambda _idx: self.save_config())
        self.mask_backend_combo.currentIndexChanged.connect(self._schedule_inpaint_preview)
        self.detect_prompt_input.editingFinished.connect(self.save_config)
        self.detect_prompt_input.editingFinished.connect(self._schedule_inpaint_preview)
        self.gdino_box_threshold_spin.valueChanged.connect(lambda _: self.save_config())
        self.gdino_box_threshold_spin.valueChanged.connect(self._schedule_inpaint_preview)
        self.gdino_text_threshold_spin.valueChanged.connect(lambda _: self.save_config())
        self.gdino_text_threshold_spin.valueChanged.connect(self._schedule_inpaint_preview)
        self.inpaint_mask_blur_spin.valueChanged.connect(lambda _: self.save_config())
        self.inpaint_fill_spin.valueChanged.connect(lambda _: self.save_config())
        self.inpaint_full_res_check.toggled.connect(lambda _: self.save_config())
        self.inpaint_padding_spin.valueChanged.connect(lambda _: self.save_config())
        self.script_json_edit.textChanged.connect(self.save_config)
        self._schedule_inpaint_preview()

    def log(self, text):
        self.log_text.append(str(text))
        bar = self.log_text.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_image_text_changed(self):
        p = self.image_input.text().strip()
        if os.path.isfile(p):
            self.current_image_path = p
            self.save_config()
            self._schedule_inpaint_preview()
            if not self.base_prompt_text and not self.base_negative_text:
                self.read_prompt_from_image(silent=True)

    def _on_base_prompt_changed(self):
        self.base_prompt_text = self.base_prompt_edit.toPlainText().strip()
        self.save_config()

    def _on_base_negative_changed(self):
        self.base_negative_text = self.base_negative_edit.toPlainText().strip()
        self.save_config()

    def _on_anchor_prompt_changed(self):
        self.anchor_prompt_text = self.anchor_prompt_edit.toPlainText().strip()
        self.save_config()

    def _on_anchor_negative_changed(self):
        self.anchor_negative_text = self.anchor_negative_edit.toPlainText().strip()
        self.save_config()

    def pick_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择基础CG图片",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if file_path:
            self.current_image_path = file_path
            self.image_input.setText(file_path)
            self.read_prompt_from_image(silent=True)
            self.save_config()
            self._schedule_inpaint_preview()

    def read_prompt_from_image(self, silent=False):
        image_path = self.image_input.text().strip()
        if not image_path or not os.path.isfile(image_path):
            if not silent:
                QMessageBox.warning(self, "提示", "请先选择有效的基础CG图片")
            return
        data = _extract_prompts_from_image(image_path)
        self.base_prompt_text = data.get("prompt", "")
        self.base_negative_text = data.get("negative_prompt", "")
        self.base_prompt_edit.blockSignals(True)
        self.base_negative_edit.blockSignals(True)
        self.base_prompt_edit.setPlainText(self.base_prompt_text)
        self.base_negative_edit.setPlainText(self.base_negative_text)
        self.base_prompt_edit.blockSignals(False)
        self.base_negative_edit.blockSignals(False)
        if not self.anchor_prompt_text:
            self.anchor_prompt_text = self.base_prompt_text
            self.anchor_prompt_edit.blockSignals(True)
            self.anchor_prompt_edit.setPlainText(self.anchor_prompt_text)
            self.anchor_prompt_edit.blockSignals(False)
        if not self.anchor_negative_text and self.base_negative_text:
            self.anchor_negative_text = self.base_negative_text
            self.anchor_negative_edit.blockSignals(True)
            self.anchor_negative_edit.setPlainText(self.anchor_negative_text)
            self.anchor_negative_edit.blockSignals(False)
        self.save_config()
        if not silent:
            self.log("已读取原图 prompt 信息。")

    def extract_anchor_with_llm(self):
        image_path = self.image_input.text().strip()
        if not image_path or not os.path.isfile(image_path):
            QMessageBox.warning(self, "提示", "请先选择有效的基础CG图片")
            return
        base_url, api_key, model_name = self.get_text_config()
        if not api_key or not model_name:
            QMessageBox.warning(self, "提示", "请先在全局配置填写文本分析模型和 API Key")
            return
        source_prompt = self.base_prompt_edit.toPlainText().strip()
        source_negative = self.base_negative_edit.toPlainText().strip()
        if not source_prompt and not source_negative:
            QMessageBox.warning(self, "提示", "请先读取原图 Prompt，再做智能提取")
            return
        self._set_running(True)
        self.anchor_thread = DiffCgAnchorThread(
            llm_cfg={"base_url": base_url, "api_key": api_key, "model": model_name},
            image_path=image_path,
            source_prompt=source_prompt,
            source_negative=source_negative,
        )
        self.anchor_thread.log.connect(self.log)
        self.anchor_thread.finished_ok.connect(self.on_anchor_ready)
        self.anchor_thread.failed.connect(self.on_anchor_failed)
        self.anchor_thread.start()

    def on_anchor_ready(self, result_obj):
        self._set_running(False)
        keep_pos = str(result_obj.get("keep_positive", "")).strip()
        keep_neg = str(result_obj.get("keep_negative", "")).strip()
        if keep_pos:
            self.anchor_prompt_text = keep_pos
            self.anchor_prompt_edit.setPlainText(keep_pos)
        if keep_neg:
            self.anchor_negative_text = keep_neg
            self.anchor_negative_edit.setPlainText(keep_neg)
        self.save_config()
        self.log("智能锚点提取完成，已更新锚点 Prompt。")

    def on_anchor_failed(self, err_text):
        self._set_running(False)
        self.log(f"智能锚点提取失败: {err_text}")
        QMessageBox.warning(self, "提取失败", str(err_text))

    def load_config(self):
        full = _load_full_image_config()
        cfg = full.get("diff_cg", {})
        self.webui_base_url_input.setText(str(cfg.get("webui_base_url", "http://127.0.0.1:7860")))
        self.timeout_spin.setValue(int(cfg.get("timeout", 600)))
        self.shot_count_spin.setValue(int(cfg.get("shot_count", 6)))
        self.story_desc_edit.setPlainText(str(cfg.get("story_desc", "")))
        self.global_negative_edit.setPlainText(str(cfg.get("global_negative", "")))
        self.default_steps_spin.setValue(int(cfg.get("default_steps", 28)))
        self.default_cfg_spin.setValue(float(cfg.get("default_cfg_scale", 5.0)))
        self.default_denoise_spin.setValue(float(cfg.get("default_denoise", 0.58)))
        self.seed_spin.setValue(int(cfg.get("seed", -1)))
        self.scale_spin.setValue(float(cfg.get("upscale_factor", 1.00)))
        self.sampler_input.setText(str(cfg.get("sampler_name", "Euler a")))
        self.scheduler_input.setText(str(cfg.get("scheduler", "Automatic")))
        self.sd_model_input.setText(str(cfg.get("sd_model", "")))
        self.sd_vae_input.setText(str(cfg.get("sd_vae", "Automatic")))
        self.global_extra_payload_edit.setPlainText(str(cfg.get("global_extra_payload", "")))
        self.chain_mode_check.setChecked(bool(cfg.get("chain_mode", True)))
        self.merge_base_prompt_check.setChecked(bool(cfg.get("merge_base_prompt", True)))
        saved_merge_mode = str(cfg.get("merge_mode", MERGE_MODE_SMART)).strip() or MERGE_MODE_SMART
        idx = self.merge_mode_combo.findData(saved_merge_mode)
        if idx < 0:
            idx = self.merge_mode_combo.findData(MERGE_MODE_SMART)
        self.merge_mode_combo.setCurrentIndex(max(0, idx))
        self.use_shot_params_check.setChecked(bool(cfg.get("use_shot_params", False)))
        self.enable_subject_inpaint_check.setChecked(bool(cfg.get("enable_subject_inpaint", False)))
        saved_backend = str(cfg.get("mask_backend", MASK_BACKEND_GRABCUT)).strip() or MASK_BACKEND_GRABCUT
        idx_backend = self.mask_backend_combo.findData(saved_backend)
        if idx_backend < 0:
            idx_backend = self.mask_backend_combo.findData(MASK_BACKEND_GRABCUT)
        self.mask_backend_combo.setCurrentIndex(max(0, idx_backend))
        self.detect_prompt_input.setText(str(cfg.get("detect_prompt", "person . 1girl . 1boy")))
        self.gdino_box_threshold_spin.setValue(float(cfg.get("gdino_box_threshold", 0.30)))
        self.gdino_text_threshold_spin.setValue(float(cfg.get("gdino_text_threshold", 0.25)))
        self.inpaint_mask_blur_spin.setValue(int(cfg.get("inpaint_mask_blur", 8)))
        self.inpaint_fill_spin.setValue(int(cfg.get("inpaint_fill", 1)))
        self.inpaint_full_res_check.setChecked(bool(cfg.get("inpaint_full_res", True)))
        self.inpaint_padding_spin.setValue(int(cfg.get("inpaint_padding", 32)))
        self.base_prompt_text = str(cfg.get("base_prompt", "")).strip()
        self.base_negative_text = str(cfg.get("base_negative", "")).strip()
        self.anchor_prompt_text = str(cfg.get("anchor_prompt", "")).strip()
        self.anchor_negative_text = str(cfg.get("anchor_negative", "")).strip()
        self.base_prompt_edit.setPlainText(self.base_prompt_text)
        self.base_negative_edit.setPlainText(self.base_negative_text)
        self.anchor_prompt_edit.setPlainText(self.anchor_prompt_text)
        self.anchor_negative_edit.setPlainText(self.anchor_negative_text)
        saved_image = str(cfg.get("image_path", "")).strip()
        if saved_image and os.path.isfile(saved_image):
            self.current_image_path = saved_image
            self.image_input.setText(saved_image)
        saved_script = str(cfg.get("script_json", "")).strip()
        if saved_script:
            self.script_json_edit.setPlainText(saved_script)
        self._schedule_inpaint_preview()

    def _set_preview_label_image(self, label_widget, image_path, fallback_text):
        if not image_path or not os.path.isfile(image_path):
            label_widget.setText(fallback_text)
            label_widget.setPixmap(QPixmap())
            return
        pix = QPixmap(image_path)
        if pix.isNull():
            label_widget.setText(fallback_text)
            label_widget.setPixmap(QPixmap())
            return
        label_widget.setPixmap(
            pix.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        label_widget.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_inpaint_preview()

    def _schedule_inpaint_preview(self, *_args):
        if self.preview_busy:
            self.preview_pending = True
            return
        self.preview_timer.start(250)

    def _render_inpaint_preview(self):
        image_path = self.image_input.text().strip()
        self._set_preview_label_image(self.preview_input_label, image_path, "原图预览")
        if not image_path or not os.path.isfile(image_path):
            self.preview_mask_label.setText("掩码预览")
            self.preview_mask_label.setPixmap(QPixmap())
            return
        if not self.enable_subject_inpaint_check.isChecked():
            self.preview_mask_label.setText("未启用主体Inpaint")
            self.preview_mask_label.setPixmap(QPixmap())
            self.preview_status_label.setText("预览状态: 空闲（未启用Inpaint）")
            self.preview_progress.setVisible(False)
            return
        preview_dir = os.path.join("data", datetime.now().strftime("%Y%m%d"), "diff-cg", "masks-preview")
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.preview_pending = True
            return
        backend = self.mask_backend_combo.currentData() or MASK_BACKEND_GRABCUT
        req = {
            "image_path": image_path,
            "backend": backend,
            "save_dir": preview_dir,
            "text_prompt": self.detect_prompt_input.text().strip(),
            "box_threshold": float(self.gdino_box_threshold_spin.value()),
            "text_threshold": float(self.gdino_text_threshold_spin.value()),
        }
        self.preview_busy = True
        self.preview_pending = False
        self.preview_progress.setVisible(True)
        self.preview_status_label.setText("预览状态: 正在生成掩码 ...")
        self.preview_worker = MaskPreviewWorker(req)
        self.preview_worker.finished_ok.connect(self._on_preview_ready)
        self.preview_worker.failed.connect(self._on_preview_failed)
        self.preview_worker.finished.connect(self._on_preview_worker_finished)
        self.preview_worker.start()

    def _on_preview_ready(self, result):
        mask_path = str((result or {}).get("mask_path", "")).strip()
        backend = str((result or {}).get("backend", "")).strip()
        box_count = int((result or {}).get("box_count", 0) or 0)
        self._set_preview_label_image(self.preview_mask_label, mask_path, "掩码预览")
        if backend == MASK_BACKEND_GDINO_SAM2:
            self.preview_status_label.setText(f"预览状态: 完成（GDINO+SAM2, boxes={box_count}）")
        else:
            self.preview_status_label.setText("预览状态: 完成（GrabCut）")

    def _on_preview_failed(self, err_text):
        self.preview_mask_label.setText(f"掩码预览失败:\n{err_text}")
        self.preview_mask_label.setPixmap(QPixmap())
        self.preview_status_label.setText("预览状态: 失败")

    def _on_preview_worker_finished(self):
        self.preview_busy = False
        self.preview_progress.setVisible(False)
        if self.preview_pending:
            self.preview_pending = False
            self.preview_timer.start(50)

    def save_config(self):
        full = _load_full_image_config()
        full["diff_cg"] = {
            "image_path": self.current_image_path,
            "shot_count": int(self.shot_count_spin.value()),
            "story_desc": self.story_desc_edit.toPlainText().strip(),
            "script_json": self.script_json_edit.toPlainText().strip(),
            "webui_base_url": self.webui_base_url_input.text().strip(),
            "timeout": int(self.timeout_spin.value()),
            "global_negative": self.global_negative_edit.toPlainText().strip(),
            "base_prompt": self.base_prompt_edit.toPlainText().strip(),
            "base_negative": self.base_negative_edit.toPlainText().strip(),
            "anchor_prompt": self.anchor_prompt_edit.toPlainText().strip(),
            "anchor_negative": self.anchor_negative_edit.toPlainText().strip(),
            "default_steps": int(self.default_steps_spin.value()),
            "default_cfg_scale": float(self.default_cfg_spin.value()),
            "default_denoise": float(self.default_denoise_spin.value()),
            "seed": int(self.seed_spin.value()),
            "upscale_factor": float(self.scale_spin.value()),
            "sampler_name": self.sampler_input.text().strip(),
            "scheduler": self.scheduler_input.text().strip(),
            "sd_model": self.sd_model_input.text().strip(),
            "sd_vae": self.sd_vae_input.text().strip(),
            "global_extra_payload": self.global_extra_payload_edit.toPlainText().strip(),
            "chain_mode": bool(self.chain_mode_check.isChecked()),
            "merge_base_prompt": bool(self.merge_base_prompt_check.isChecked()),
            "merge_mode": self.merge_mode_combo.currentData() or MERGE_MODE_SMART,
            "use_shot_params": bool(self.use_shot_params_check.isChecked()),
            "enable_subject_inpaint": bool(self.enable_subject_inpaint_check.isChecked()),
            "mask_backend": self.mask_backend_combo.currentData() or MASK_BACKEND_GRABCUT,
            "detect_prompt": self.detect_prompt_input.text().strip(),
            "gdino_box_threshold": float(self.gdino_box_threshold_spin.value()),
            "gdino_text_threshold": float(self.gdino_text_threshold_spin.value()),
            "inpaint_mask_blur": int(self.inpaint_mask_blur_spin.value()),
            "inpaint_fill": int(self.inpaint_fill_spin.value()),
            "inpaint_full_res": bool(self.inpaint_full_res_check.isChecked()),
            "inpaint_padding": int(self.inpaint_padding_spin.value()),
        }
        try:
            _save_full_image_config(full)
        except Exception:
            pass

    def _set_running(self, running):
        self.gen_script_btn.setEnabled(not running)
        self.normalize_script_btn.setEnabled(not running)
        self.run_btn.setEnabled(not running)
        self.pick_image_btn.setEnabled(not running)
        self.read_prompt_btn.setEnabled(not running)
        self.extract_anchor_btn.setEnabled(not running)

    def generate_script(self):
        image_path = self.image_input.text().strip()
        if not image_path or not os.path.isfile(image_path):
            QMessageBox.warning(self, "提示", "请先选择有效的基础CG图片")
            return
        base_url, api_key, model_name = self.get_text_config()
        llm_cfg = {"base_url": base_url, "api_key": api_key, "model": model_name}
        self.save_config()
        self._set_running(True)
        self.script_thread = DiffCgScriptThread(
            llm_cfg=llm_cfg,
            image_path=image_path,
            shot_count=int(self.shot_count_spin.value()),
            story_desc=self.story_desc_edit.toPlainText().strip(),
            base_prompt=self.base_prompt_edit.toPlainText().strip(),
            base_negative=self.base_negative_edit.toPlainText().strip(),
        )
        self.script_thread.log.connect(self.log)
        self.script_thread.finished_ok.connect(self.on_script_ready)
        self.script_thread.failed.connect(self.on_script_failed)
        self.script_thread.start()

    def on_script_ready(self, script_obj):
        self._set_running(False)
        base_prompt = self.base_prompt_edit.toPlainText().strip()
        base_negative = self.base_negative_edit.toPlainText().strip()
        anchor_prompt = self.anchor_prompt_edit.toPlainText().strip()
        anchor_negative = self.anchor_negative_edit.toPlainText().strip()
        merge_mode = str(self.merge_mode_combo.currentData() or MERGE_MODE_SMART)
        merged_shots = []
        for shot in script_obj.get("shots", []):
            if not isinstance(shot, dict):
                continue
            row = dict(shot)
            if merge_mode == "full_base":
                row["prompt"] = _merge_prompt_text(base_prompt, row.get("prompt", ""))
                row["negative_prompt"] = _merge_negative_text(base_negative, row.get("negative_prompt", ""))
            else:
                row["prompt"] = _merge_prompt_text(anchor_prompt or base_prompt, row.get("prompt", ""))
                row["negative_prompt"] = _merge_negative_text(anchor_negative or base_negative, row.get("negative_prompt", ""))
            merged_shots.append(row)
        script_obj["shots"] = merged_shots
        self.script_json_edit.setPlainText(json.dumps(script_obj, ensure_ascii=False, indent=2))
        self.save_config()
        self.log("剧本 JSON 生成成功，可手动编辑后执行。")

    def on_script_failed(self, err_text):
        self._set_running(False)
        self.log(f"剧本生成失败: {err_text}")
        QMessageBox.warning(self, "剧本生成失败", str(err_text))

    def normalize_script_json(self):
        text = self.script_json_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "剧本 JSON 为空")
            return
        try:
            obj = json.loads(text)
            normalized = _normalize_script(obj, int(self.shot_count_spin.value()))
            self.script_json_edit.setPlainText(json.dumps(normalized, ensure_ascii=False, indent=2))
            self.save_config()
            self.log("剧本 JSON 已规范化。")
        except Exception as e:
            QMessageBox.warning(self, "JSON错误", f"无法解析剧本 JSON: {e}")

    def _parse_global_extra_payload(self):
        raw = self.global_extra_payload_edit.toPlainText().strip()
        if not raw:
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("全局插件 Payload 必须是 JSON 对象")
        return data

    def run_generation(self):
        image_path = self.image_input.text().strip()
        if not image_path or not os.path.isfile(image_path):
            QMessageBox.warning(self, "提示", "请先选择有效的基础CG图片")
            return
        script_text = self.script_json_edit.toPlainText().strip()
        if not script_text:
            QMessageBox.warning(self, "提示", "请先生成或填写剧本 JSON")
            return
        try:
            script_obj = json.loads(script_text)
            script_obj = _normalize_script(script_obj, int(self.shot_count_spin.value()))
        except Exception as e:
            QMessageBox.warning(self, "JSON错误", f"剧本 JSON 无效: {e}")
            return
        try:
            global_extra_payload = self._parse_global_extra_payload()
        except Exception as e:
            QMessageBox.warning(self, "JSON错误", str(e))
            return

        req = {
            "image_path": image_path,
            "script": script_obj,
            "chain_mode": bool(self.chain_mode_check.isChecked()),
            "webui_base_url": self.webui_base_url_input.text().strip(),
            "timeout": int(self.timeout_spin.value()),
            "global_negative": self.global_negative_edit.toPlainText().strip(),
            "default_steps": int(self.default_steps_spin.value()),
            "default_cfg_scale": float(self.default_cfg_spin.value()),
            "default_denoise": float(self.default_denoise_spin.value()),
            "seed": int(self.seed_spin.value()),
            "width": None,
            "height": None,
            "sampler_name": self.sampler_input.text().strip() or "Euler a",
            "scheduler": self.scheduler_input.text().strip() or "Automatic",
            "sd_model": self.sd_model_input.text().strip(),
            "sd_vae": self.sd_vae_input.text().strip() or "Automatic",
            "global_extra_payload": global_extra_payload,
            "base_prompt": self.base_prompt_edit.toPlainText().strip(),
            "base_negative": self.base_negative_edit.toPlainText().strip(),
            "anchor_prompt": self.anchor_prompt_edit.toPlainText().strip(),
            "anchor_negative": self.anchor_negative_edit.toPlainText().strip(),
            "merge_base_prompt": bool(self.merge_base_prompt_check.isChecked()),
            "merge_mode": self.merge_mode_combo.currentData() or MERGE_MODE_SMART,
            "use_shot_params": bool(self.use_shot_params_check.isChecked()),
            "enable_subject_inpaint": bool(self.enable_subject_inpaint_check.isChecked()),
            "mask_backend": self.mask_backend_combo.currentData() or MASK_BACKEND_GRABCUT,
            "detect_prompt": self.detect_prompt_input.text().strip(),
            "gdino_box_threshold": float(self.gdino_box_threshold_spin.value()),
            "gdino_text_threshold": float(self.gdino_text_threshold_spin.value()),
            "inpaint_mask_blur": int(self.inpaint_mask_blur_spin.value()),
            "inpaint_fill": int(self.inpaint_fill_spin.value()),
            "inpaint_full_res": bool(self.inpaint_full_res_check.isChecked()),
            "inpaint_padding": int(self.inpaint_padding_spin.value()),
        }
        try:
            w, h = _calc_scaled_wh(image_path, float(self.scale_spin.value()))
            req["width"] = w
            req["height"] = h
        except Exception:
            req["width"] = None
            req["height"] = None
        self.script_json_edit.setPlainText(json.dumps(script_obj, ensure_ascii=False, indent=2))
        self.save_config()
        self._set_running(True)
        self.gen_thread = DiffCgGenerateThread(req)
        self.gen_thread.log.connect(self.log)
        self.gen_thread.finished_ok.connect(self.on_generation_success)
        self.gen_thread.failed.connect(self.on_generation_failed)
        self.gen_thread.start()

    def on_generation_success(self, result):
        self._set_running(False)
        outputs = result.get("outputs", []) if isinstance(result, dict) else []
        self.log(f"差分CG生成完成，共输出 {len(outputs)} 张。")
        manifest_path = result.get("manifest_path", "")
        if manifest_path:
            self.log(f"清单文件: {manifest_path}")
        QMessageBox.information(self, "完成", f"差分CG生成完成，共 {len(outputs)} 张。")

    def on_generation_failed(self, err_text):
        self._set_running(False)
        self.log(f"差分CG生成失败: {err_text}")
        QMessageBox.warning(self, "生成失败", str(err_text))
