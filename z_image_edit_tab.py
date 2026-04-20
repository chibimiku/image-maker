import gc
import os
import re
import json
import importlib
import traceback
import inspect
from datetime import datetime
from PIL import Image
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QLabel, QTextEdit, QMessageBox, QLineEdit, QProgressBar,
    QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


DEFAULT_PROMPTS = [
    "refine",
    "enhance details",
    "remove artifacts",
    "improve lighting"
]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Z_IMAGE_CONFIG_FILE = os.path.join(BASE_DIR, "config-z-image.json")


def _safe_prompt_name(prompt: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", prompt.strip())
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = cleaned[:48].strip("-_")
    return cleaned or "edit"


def _normalize_model_ref(model_path_or_id: str) -> str:
    if not model_path_or_id:
        return ""
    return os.path.expanduser(model_path_or_id.strip())


def _format_gb(value_bytes: float) -> str:
    return f"{(value_bytes / (1024 ** 3)):.2f} GB"


def _cuda_memory_snapshot(torch_module):
    info = {
        "available": False,
        "allocated": 0,
        "reserved": 0,
        "total": 0
    }
    try:
        if torch_module.cuda.is_available():
            info["available"] = True
            info["allocated"] = int(torch_module.cuda.memory_allocated())
            info["reserved"] = int(torch_module.cuda.memory_reserved())
            total_mem = torch_module.cuda.get_device_properties(0).total_memory
            info["total"] = int(total_mem)
    except Exception:
        pass
    return info


def detect_torch_cuda_env(model_path_or_id=""):
    result = {
        "ok": False,
        "torch_version": "未安装",
        "cuda_available": False,
        "cuda_version": "未知",
        "gpu_name": "无",
        "diffusers_installed": False,
        "message": "",
        "model_ref": model_path_or_id.strip(),
        "model_input_ok": False,
        "model_local_exists": False,
        "model_config_ok": False,
        "can_run_command": False,
        "errors": [],
        "warnings": []
    }
    torch = None
    try:
        torch = importlib.import_module("torch")
        result["torch_version"] = getattr(torch, "__version__", "未知")
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_version"] = str(getattr(torch.version, "cuda", "未知"))
        if result["cuda_available"]:
            result["gpu_name"] = str(torch.cuda.get_device_name(0))
        result["ok"] = True
    except Exception as e:
        result["errors"].append(f"未检测到可用的 PyTorch 环境: {e}")

    try:
        importlib.import_module("diffusers")
        result["diffusers_installed"] = True
    except Exception as e:
        result["diffusers_installed"] = False
        result["errors"].append(f"未检测到 diffusers: {e}")

    model_ref = _normalize_model_ref(model_path_or_id)
    if not model_ref:
        result["warnings"].append("尚未填写本地模型目录")
    else:
        result["model_input_ok"] = True
        result["model_local_exists"] = os.path.isdir(model_ref)
        if result["model_local_exists"]:
            model_index_path = os.path.join(model_ref, "model_index.json")
            if os.path.isfile(model_index_path):
                result["model_config_ok"] = True
            else:
                result["errors"].append("本地模型目录缺少 model_index.json")
        else:
            result["errors"].append("模型路径不存在")

    if torch is not None and result["cuda_available"]:
        mem = _cuda_memory_snapshot(torch)
        if mem["total"] > 0:
            result["warnings"].append(
                f"当前显存占用(本进程): 已分配 {_format_gb(mem['allocated'])} / 预留 {_format_gb(mem['reserved'])} / 总计 {_format_gb(mem['total'])}"
            )
    elif result["ok"] and not result["cuda_available"]:
        result["warnings"].append("未检测到 CUDA，可使用 CPU 运行但速度会较慢")

    result["can_run_command"] = (
        result["ok"]
        and result["diffusers_installed"]
        and result["model_input_ok"]
        and result["model_config_ok"]
        and not result["errors"]
    )

    if not result["message"]:
        if result["can_run_command"]:
            result["message"] = "检测通过：可执行生成命令"
        elif result["ok"]:
            result["message"] = "环境检测完成，但当前模型配置不可用"
        else:
            result["message"] = "环境检测失败"
    return result


class ModelLoadWorker(QThread):
    log = pyqtSignal(str)
    loaded = pyqtSignal(object, str, str, float)
    failed = pyqtSignal(str)

    def __init__(self, model_path_or_id):
        super().__init__()
        self.model_path_or_id = model_path_or_id

    def run(self):
        model_ref = _normalize_model_ref(self.model_path_or_id)
        if not model_ref:
            self.failed.emit("请先填写本地模型目录")
            return
        if not os.path.isdir(model_ref):
            self.failed.emit(f"模型目录不存在: {model_ref}")
            return
        model_index_path = os.path.join(model_ref, "model_index.json")
        if not os.path.isfile(model_index_path):
            self.failed.emit(f"模型目录缺少 model_index.json: {model_ref}")
            return

        try:
            torch = importlib.import_module("torch")
            diffusers = importlib.import_module("diffusers")
            DiffusionPipeline = getattr(diffusers, "DiffusionPipeline")
        except Exception as e:
            self.log.emit("依赖导入失败：正在收集详细错误栈...")
            self.log.emit(traceback.format_exc())
            self.failed.emit(f"缺少 z-image 运行依赖，请先安装可用环境: {e}")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.log.emit(f"正在加载模型: {self.model_path_or_id}")
        self.log.emit(f"运行设备: {device}")
        try:
            transformers = importlib.import_module("transformers")
            hf_hub = importlib.import_module("huggingface_hub")
            self.log.emit(
                "运行环境版本: "
                f"torch={getattr(torch, '__version__', 'unknown')} | "
                f"diffusers={getattr(diffusers, '__version__', 'unknown')} | "
                f"transformers={getattr(transformers, '__version__', 'unknown')} | "
                f"huggingface_hub={getattr(hf_hub, '__version__', 'unknown')}"
            )
        except Exception:
            self.log.emit("运行环境版本收集失败")
            self.log.emit(traceback.format_exc())

        before_alloc = 0
        after_alloc = 0
        if device == "cuda":
            try:
                torch.cuda.empty_cache()
                before_alloc = int(torch.cuda.memory_allocated())
            except Exception:
                before_alloc = 0

        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_ref,
                torch_dtype=dtype,
                local_files_only=True
            )
            pipe = pipe.to(device)
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
        except Exception as e:
            self.log.emit("模型加载阶段异常：")
            self.log.emit(traceback.format_exc())
            self.failed.emit(f"模型加载失败，请确认模型是否可用于图像编辑: {e}")
            return

        if device == "cuda":
            try:
                after_alloc = int(torch.cuda.memory_allocated())
            except Exception:
                after_alloc = 0
        model_vram_mb = max(0.0, (after_alloc - before_alloc) / (1024 ** 2))
        self.loaded.emit(pipe, model_ref, device, model_vram_mb)


class ZImageEditWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, image_paths, prompts, model_path_or_id, pipe, device, params):
        super().__init__()
        self.image_paths = image_paths
        self.prompts = prompts
        self.model_path_or_id = model_path_or_id
        self.pipe = pipe
        self.device = device
        self.params = params or {}

    def run(self):
        try:
            torch = importlib.import_module("torch")
        except Exception as e:
            self.failed.emit(f"缺少 PyTorch 运行环境: {e}")
            return

        pipe = self.pipe
        if pipe is None:
            self.failed.emit("模型尚未加载，请先点击“加载模型”")
            return

        try:
            call_sig = inspect.signature(pipe.__call__)
            call_params = set(call_sig.parameters.keys())
        except Exception:
            call_params = set()
        image_arg_name = None
        for candidate in ("image", "init_image", "input_image"):
            if candidate in call_params:
                image_arg_name = candidate
                break
        if image_arg_name is None:
            self.failed.emit(
                "当前模型不支持输入图片编辑。该模型是文生图管线（不接受 image 参数），请更换支持图生图的模型。"
            )
            return

        steps = int(self.params.get("steps", 30))
        guidance_scale = float(self.params.get("guidance_scale", 7.5))
        strength = float(self.params.get("strength", 0.75))
        num_images_per_prompt = int(self.params.get("num_images_per_prompt", 1))
        negative_prompt = str(self.params.get("negative_prompt", "")).strip()
        seed = int(self.params.get("seed", -1))

        today = datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join("data", today, "z-image-edit")
        os.makedirs(save_dir, exist_ok=True)

        total = len(self.image_paths) * len(self.prompts)
        done = 0
        saved_records = []

        for image_path in self.image_paths:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            for prompt in self.prompts:
                done += 1
                self.progress.emit(done, total)
                prompt_clean = prompt.strip()
                if not prompt_clean:
                    continue
                self.log.emit(f"处理中: {os.path.basename(image_path)} | prompt={prompt_clean}")
                try:
                    with Image.open(image_path).convert("RGB") as img:
                        kwargs = {
                            "prompt": prompt_clean,
                            "num_inference_steps": steps,
                            "guidance_scale": guidance_scale,
                            "num_images_per_prompt": num_images_per_prompt
                        }
                        kwargs[image_arg_name] = img
                        if "strength" in call_params:
                            kwargs["strength"] = strength
                        if negative_prompt:
                            kwargs["negative_prompt"] = negative_prompt
                        if seed >= 0:
                            effective_seed = seed + done
                            generator = torch.Generator(device=self.device)
                            generator = generator.manual_seed(effective_seed)
                            kwargs["generator"] = generator
                        result = pipe(**kwargs)
                    images = getattr(result, "images", [])
                    if not images:
                        raise RuntimeError("模型未返回图片")
                    for idx, out_img in enumerate(images):
                        ts = datetime.now().strftime("%H%M%S")
                        prompt_name = _safe_prompt_name(prompt_clean)
                        suffix = f"-{idx + 1}" if len(images) > 1 else ""
                        out_name = f"{ts}-{image_name}-{prompt_name}{suffix}.png"
                        out_path = os.path.join(save_dir, out_name)
                        out_img.save(out_path)
                        metadata = {
                            "source_image": image_path,
                            "prompt": prompt_clean,
                            "negative_prompt": negative_prompt,
                            "model": self.model_path_or_id,
                            "device": self.device,
                            "steps": steps,
                            "guidance_scale": guidance_scale,
                            "strength": strength,
                            "seed": seed if seed >= 0 else "random",
                            "num_images_per_prompt": num_images_per_prompt,
                            "saved_image": out_path,
                            "time": datetime.now().isoformat(timespec="seconds")
                        }
                        meta_name = f"{ts}-{image_name}-{prompt_name}{suffix}.json"
                        meta_path = os.path.join(save_dir, meta_name)
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=4)
                        saved_records.append(metadata)
                        self.log.emit(f"已保存: {out_path}")
                except Exception as e:
                    self.log.emit(f"处理失败: {os.path.basename(image_path)} | {prompt_clean} | {e}")

        self.finished.emit(saved_records)


class ZImageEditWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image_paths = []
        self.worker = None
        self.load_worker = None
        self.loaded_pipe = None
        self.loaded_model_ref = ""
        self.loaded_device = "cpu"
        self.model_vram_mb = 0.0
        self.init_ui()
        self.load_local_config()
        self.run_env_check()

    def init_ui(self):
        layout = QVBoxLayout(self)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("z-image 模型:"))
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("例如: /path/to/local/model（仅支持本地目录）")
        self.model_input.textChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_input, stretch=1)
        self.browse_model_btn = QPushButton("选择本地模型目录")
        self.browse_model_btn.clicked.connect(self.pick_local_model_dir)
        model_layout.addWidget(self.browse_model_btn)
        self.env_btn = QPushButton("检测 CUDA/PyTorch")
        self.env_btn.clicked.connect(self.run_env_check)
        model_layout.addWidget(self.env_btn)
        self.full_check_btn = QPushButton("完整检测可生成")
        self.full_check_btn.clicked.connect(self.run_full_check)
        model_layout.addWidget(self.full_check_btn)
        layout.addLayout(model_layout)

        model_ops_layout = QHBoxLayout()
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        self.unload_model_btn = QPushButton("卸载模型")
        self.unload_model_btn.clicked.connect(self.unload_model)
        self.unload_model_btn.setEnabled(False)
        model_ops_layout.addWidget(self.load_model_btn)
        model_ops_layout.addWidget(self.unload_model_btn)
        model_ops_layout.addStretch()
        layout.addLayout(model_ops_layout)

        self.env_label = QLabel("环境未检测")
        layout.addWidget(self.env_label)
        self.model_state_label = QLabel("模型状态: 未加载")
        layout.addWidget(self.model_state_label)

        body_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("添加图片")
        self.add_btn.clicked.connect(self.add_images)
        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_images)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.clear_btn)
        left_layout.addLayout(btn_row)

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.ExtendedSelection)
        left_layout.addWidget(self.image_list)
        body_layout.addLayout(left_layout, stretch=1)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("固定 prompts（可编辑，每行一个）:"))
        self.prompts_edit = QTextEdit()
        self.prompts_edit.setPlainText("\n".join(DEFAULT_PROMPTS))
        right_layout.addWidget(self.prompts_edit)

        params_row_1 = QHBoxLayout()
        params_row_1.addWidget(QLabel("Steps:"))
        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 200)
        self.steps_input.setValue(30)
        params_row_1.addWidget(self.steps_input)
        params_row_1.addWidget(QLabel("CFG:"))
        self.cfg_input = QDoubleSpinBox()
        self.cfg_input.setRange(1.0, 30.0)
        self.cfg_input.setSingleStep(0.5)
        self.cfg_input.setValue(7.5)
        params_row_1.addWidget(self.cfg_input)
        params_row_1.addWidget(QLabel("Strength:"))
        self.strength_input = QDoubleSpinBox()
        self.strength_input.setRange(0.1, 1.0)
        self.strength_input.setSingleStep(0.05)
        self.strength_input.setValue(0.75)
        params_row_1.addWidget(self.strength_input)
        params_row_1.addWidget(QLabel("每提示图数:"))
        self.num_images_input = QSpinBox()
        self.num_images_input.setRange(1, 8)
        self.num_images_input.setValue(1)
        params_row_1.addWidget(self.num_images_input)
        right_layout.addLayout(params_row_1)

        params_row_2 = QHBoxLayout()
        params_row_2.addWidget(QLabel("Seed:"))
        self.seed_input = QLineEdit("-1")
        self.seed_input.setPlaceholderText("-1 表示随机")
        params_row_2.addWidget(self.seed_input)
        params_row_2.addWidget(QLabel("Negative Prompt:"))
        self.negative_prompt_input = QLineEdit()
        self.negative_prompt_input.setPlaceholderText("可选")
        params_row_2.addWidget(self.negative_prompt_input, stretch=1)
        right_layout.addLayout(params_row_2)

        right_layout.addWidget(QLabel("日志:"))
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        right_layout.addWidget(self.log_edit)
        body_layout.addLayout(right_layout, stretch=1)

        layout.addLayout(body_layout)

        bottom_layout = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setFormat("0/0")
        bottom_layout.addWidget(self.progress, stretch=1)
        self.start_btn = QPushButton("开始 z-image 编辑")
        self.start_btn.clicked.connect(self.start_edit)
        bottom_layout.addWidget(self.start_btn)
        layout.addLayout(bottom_layout)

    def on_model_changed(self):
        current_ref = self.model_input.text().strip()
        if self.loaded_pipe is not None and current_ref != self.loaded_model_ref:
            self.model_state_label.setText("模型状态: 已加载模型与当前输入不一致，请重新加载")
        self.load_model_btn.setEnabled(bool(current_ref))
        self.save_local_config()

    def load_local_config(self):
        model_ref = ""
        if os.path.isfile(Z_IMAGE_CONFIG_FILE):
            try:
                with open(Z_IMAGE_CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f) or {}
                model_ref = str(cfg.get("last_local_model_dir", "")).strip()
            except Exception:
                model_ref = ""
        if model_ref:
            self.model_input.setText(model_ref)

    def save_local_config(self):
        config = {
            "last_local_model_dir": self.model_input.text().strip()
        }
        try:
            with open(Z_IMAGE_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        except Exception:
            # 不阻断主流程，配置保存失败时静默跳过
            pass

    def pick_local_model_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择本地模型目录", "")
        if directory:
            self.model_input.setText(directory)
            self.run_full_check()

    def _build_env_text(self, env):
        return (
            f"PyTorch: {env['torch_version']} | "
            f"CUDA可用: {'是' if env['cuda_available'] else '否'} | "
            f"CUDA版本: {env['cuda_version']} | "
            f"GPU: {env['gpu_name']} | "
            f"diffusers: {'已安装' if env['diffusers_installed'] else '未安装'} | "
            f"可运行: {'是' if env['can_run_command'] else '否'}"
        )

    def add_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not files:
            return
        for file_path in files:
            if file_path not in self.image_paths:
                self.image_paths.append(file_path)
                item = QListWidgetItem(os.path.basename(file_path))
                item.setData(Qt.UserRole, file_path)
                self.image_list.addItem(item)
        self.log(f"已添加 {len(files)} 张图片")

    def clear_images(self):
        self.image_paths.clear()
        self.image_list.clear()
        self.progress.setValue(0)
        self.progress.setFormat("0/0")
        self.log_edit.clear()

    def run_env_check(self):
        env = detect_torch_cuda_env(self.model_input.text().strip())
        text = self._build_env_text(env)
        self.env_label.setText(text)
        self.log(text)
        for err in env["errors"]:
            self.log(f"[错误] {err}")
        for warn in env["warnings"]:
            self.log(f"[提示] {warn}")
        self.log(env["message"])

    def run_full_check(self):
        model_ref = self.model_input.text().strip()
        env = detect_torch_cuda_env(model_ref)
        text = self._build_env_text(env)
        self.env_label.setText(text)
        self.log(f"完整检测结果: {env['message']}")
        if model_ref:
            model_status = "本地路径可用" if env["model_local_exists"] else "本地路径不可用"
            self.log(f"模型输入: {model_ref} | 状态: {model_status}")
        for err in env["errors"]:
            self.log(f"[错误] {err}")
        for warn in env["warnings"]:
            self.log(f"[提示] {warn}")

    def load_model(self):
        model_ref = self.model_input.text().strip()
        if not model_ref:
            QMessageBox.warning(self, "提示", "请先填写 z-image 本地模型目录")
            return

        check = detect_torch_cuda_env(model_ref)
        if not check["can_run_command"]:
            msg = "\n".join(check["errors"]) or "当前环境不满足运行条件"
            QMessageBox.warning(self, "检测未通过", msg)
            self.run_full_check()
            return

        self._set_model_buttons_enabled(False)
        self.log("开始加载模型，请稍候...")
        self.load_worker = ModelLoadWorker(model_ref)
        self.load_worker.log.connect(self.log)
        self.load_worker.loaded.connect(self.on_model_loaded)
        self.load_worker.failed.connect(self.on_model_load_failed)
        self.load_worker.start()

    def on_model_loaded(self, pipe, model_ref, device, model_vram_mb):
        self.loaded_pipe = pipe
        self.loaded_model_ref = model_ref
        self.loaded_device = device
        self.model_vram_mb = model_vram_mb
        self.unload_model_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        if device == "cuda":
            self.model_state_label.setText(
                f"模型状态: 已加载 | 设备: {device} | 估算占用显存: {model_vram_mb:.1f} MB"
            )
        else:
            self.model_state_label.setText("模型状态: 已加载 | 设备: cpu | 当前未占用显存")
        self.log(f"模型加载完成: {model_ref}")

    def on_model_load_failed(self, error_text):
        self.load_model_btn.setEnabled(True)
        self.unload_model_btn.setEnabled(self.loaded_pipe is not None)
        self.log(f"模型加载失败: {error_text}")
        QMessageBox.warning(self, "模型加载失败", error_text)

    def unload_model(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "任务执行中，无法卸载模型")
            return
        if self.loaded_pipe is None:
            self.model_state_label.setText("模型状态: 未加载")
            self.unload_model_btn.setEnabled(False)
            return
        try:
            torch = importlib.import_module("torch")
        except Exception:
            torch = None

        before_alloc = 0
        if torch is not None and self.loaded_device == "cuda" and torch.cuda.is_available():
            try:
                before_alloc = int(torch.cuda.memory_allocated())
            except Exception:
                before_alloc = 0
        self.loaded_pipe = None
        self.loaded_model_ref = ""
        self.model_vram_mb = 0.0
        gc.collect()
        released_mb = 0.0
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                after_alloc = int(torch.cuda.memory_allocated())
                released_mb = max(0.0, (before_alloc - after_alloc) / (1024 ** 2))
            except Exception:
                released_mb = 0.0
        self.model_state_label.setText("模型状态: 未加载")
        self.unload_model_btn.setEnabled(False)
        self.load_model_btn.setEnabled(bool(self.model_input.text().strip()))
        if released_mb > 0:
            self.log(f"模型已卸载，估算释放显存: {released_mb:.1f} MB")
        else:
            self.log("模型已卸载")

    def _set_model_buttons_enabled(self, enabled):
        self.load_model_btn.setEnabled(enabled)
        self.unload_model_btn.setEnabled(enabled and self.loaded_pipe is not None)
        self.full_check_btn.setEnabled(enabled)
        self.env_btn.setEnabled(enabled)
        self.browse_model_btn.setEnabled(enabled)
        self.model_input.setEnabled(enabled)

    def log(self, text):
        self.log_edit.append(text)
        scrollbar = self.log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def start_edit(self):
        if not self.image_paths:
            QMessageBox.warning(self, "提示", "请先添加图片")
            return

        prompts = [line.strip() for line in self.prompts_edit.toPlainText().splitlines() if line.strip()]
        if not prompts:
            QMessageBox.warning(self, "提示", "请至少保留一个 prompt")
            return

        if self.loaded_pipe is None:
            QMessageBox.warning(self, "提示", "请先点击“加载模型”")
            return
        model_path_or_id = self.model_input.text().strip()
        if model_path_or_id != self.loaded_model_ref:
            QMessageBox.warning(self, "提示", "当前输入模型与已加载模型不一致，请重新加载模型")
            return

        seed_text = self.seed_input.text().strip() or "-1"
        try:
            seed = int(seed_text)
        except Exception:
            QMessageBox.warning(self, "提示", "Seed 必须是整数，-1 表示随机")
            return
        params = {
            "steps": self.steps_input.value(),
            "guidance_scale": self.cfg_input.value(),
            "strength": self.strength_input.value(),
            "num_images_per_prompt": self.num_images_input.value(),
            "negative_prompt": self.negative_prompt_input.text().strip(),
            "seed": seed
        }

        self.start_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.prompts_edit.setEnabled(False)
        self.steps_input.setEnabled(False)
        self.cfg_input.setEnabled(False)
        self.strength_input.setEnabled(False)
        self.num_images_input.setEnabled(False)
        self.seed_input.setEnabled(False)
        self.negative_prompt_input.setEnabled(False)
        self._set_model_buttons_enabled(False)

        total = len(self.image_paths) * len(prompts)
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(0)
        self.progress.setFormat(f"0/{total}")
        self.log("开始执行 z-image 编辑任务")

        self.worker = ZImageEditWorker(
            self.image_paths,
            prompts,
            model_path_or_id,
            self.loaded_pipe,
            self.loaded_device,
            params
        )
        self.worker.log.connect(self.log)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_progress(self, current, total):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(current)
        self.progress.setFormat(f"{current}/{total}")

    def on_finished(self, records):
        self.start_btn.setEnabled(True)
        self.add_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.prompts_edit.setEnabled(True)
        self.steps_input.setEnabled(True)
        self.cfg_input.setEnabled(True)
        self.strength_input.setEnabled(True)
        self.num_images_input.setEnabled(True)
        self.seed_input.setEnabled(True)
        self.negative_prompt_input.setEnabled(True)
        self._set_model_buttons_enabled(True)
        today = datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join("data", today, "z-image-edit")
        self.log(f"任务完成，共保存 {len(records)} 条结果")
        self.log(f"输出目录: {save_dir}")
        QMessageBox.information(self, "完成", f"任务完成。\n保存目录:\n{save_dir}")

    def on_failed(self, error_text):
        self.start_btn.setEnabled(True)
        self.add_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.prompts_edit.setEnabled(True)
        self.steps_input.setEnabled(True)
        self.cfg_input.setEnabled(True)
        self.strength_input.setEnabled(True)
        self.num_images_input.setEnabled(True)
        self.seed_input.setEnabled(True)
        self.negative_prompt_input.setEnabled(True)
        self._set_model_buttons_enabled(True)
        self.log(f"任务失败: {error_text}")
        QMessageBox.warning(self, "失败", error_text)
