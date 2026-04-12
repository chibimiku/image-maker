import os
import re
import json
import importlib
from datetime import datetime
from PIL import Image
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QLabel, QTextEdit, QMessageBox, QLineEdit, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


DEFAULT_PROMPTS = [
    "refine",
    "enhance details",
    "remove artifacts",
    "improve lighting"
]


def _safe_prompt_name(prompt: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", prompt.strip())
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = cleaned[:48].strip("-_")
    return cleaned or "edit"


def detect_torch_cuda_env():
    result = {
        "ok": False,
        "torch_version": "未安装",
        "cuda_available": False,
        "cuda_version": "未知",
        "gpu_name": "无",
        "diffusers_installed": False,
        "message": ""
    }
    try:
        torch = importlib.import_module("torch")
        result["torch_version"] = getattr(torch, "__version__", "未知")
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_version"] = str(getattr(torch.version, "cuda", "未知"))
        if result["cuda_available"]:
            result["gpu_name"] = str(torch.cuda.get_device_name(0))
        result["ok"] = True
    except Exception as e:
        result["message"] = f"未检测到可用的 PyTorch 环境: {e}"
    try:
        importlib.import_module("diffusers")
        result["diffusers_installed"] = True
    except Exception:
        result["diffusers_installed"] = False
    if not result["message"]:
        if result["ok"]:
            result["message"] = "环境检测完成"
        else:
            result["message"] = "环境检测失败"
    return result


class ZImageEditWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, image_paths, prompts, model_path_or_id):
        super().__init__()
        self.image_paths = image_paths
        self.prompts = prompts
        self.model_path_or_id = model_path_or_id

    def run(self):
        try:
            torch = importlib.import_module("torch")
            diffusers = importlib.import_module("diffusers")
            AutoPipelineForImage2Image = getattr(diffusers, "AutoPipelineForImage2Image")
        except Exception as e:
            self.failed.emit(f"缺少 z-image 运行依赖，请先安装可用环境: {e}")
            return

        if not self.model_path_or_id:
            self.failed.emit("请先填写 z-image 模型路径或 HuggingFace 模型 ID")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.log.emit(f"正在加载模型: {self.model_path_or_id}")
        self.log.emit(f"运行设备: {device}")

        try:
            pipe = AutoPipelineForImage2Image.from_pretrained(self.model_path_or_id, torch_dtype=dtype)
            pipe = pipe.to(device)
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
        except Exception as e:
            self.failed.emit(f"模型加载失败，请确认模型是否可用于图像编辑: {e}")
            return

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
                        result = pipe(prompt=prompt_clean, image=img)
                    images = getattr(result, "images", [])
                    if not images:
                        raise RuntimeError("模型未返回图片")
                    out_img = images[0]
                    ts = datetime.now().strftime("%H%M%S")
                    prompt_name = _safe_prompt_name(prompt_clean)
                    out_name = f"{ts}-{image_name}-{prompt_name}.png"
                    out_path = os.path.join(save_dir, out_name)
                    out_img.save(out_path)
                    metadata = {
                        "source_image": image_path,
                        "prompt": prompt_clean,
                        "model": self.model_path_or_id,
                        "device": device,
                        "saved_image": out_path,
                        "time": datetime.now().isoformat(timespec="seconds")
                    }
                    meta_name = f"{ts}-{image_name}-{prompt_name}.json"
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
        self.init_ui()
        self.run_env_check()

    def init_ui(self):
        layout = QVBoxLayout(self)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("z-image 模型:"))
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("例如: /path/to/local/model 或 HuggingFace 模型 ID")
        model_layout.addWidget(self.model_input, stretch=1)
        self.env_btn = QPushButton("检测 CUDA/PyTorch")
        self.env_btn.clicked.connect(self.run_env_check)
        model_layout.addWidget(self.env_btn)
        layout.addLayout(model_layout)

        self.env_label = QLabel("环境未检测")
        layout.addWidget(self.env_label)

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
        env = detect_torch_cuda_env()
        text = (
            f"PyTorch: {env['torch_version']} | "
            f"CUDA可用: {'是' if env['cuda_available'] else '否'} | "
            f"CUDA版本: {env['cuda_version']} | "
            f"GPU: {env['gpu_name']} | "
            f"diffusers: {'已安装' if env['diffusers_installed'] else '未安装'}"
        )
        self.env_label.setText(text)
        self.log(text)
        if not env["ok"]:
            self.log(env["message"])

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

        model_path_or_id = self.model_input.text().strip()
        if not model_path_or_id:
            QMessageBox.warning(self, "提示", "请填写 z-image 模型路径或模型 ID")
            return

        self.start_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.prompts_edit.setEnabled(False)
        self.model_input.setEnabled(False)
        self.env_btn.setEnabled(False)

        total = len(self.image_paths) * len(prompts)
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(0)
        self.progress.setFormat(f"0/{total}")
        self.log("开始执行 z-image 编辑任务")

        self.worker = ZImageEditWorker(self.image_paths, prompts, model_path_or_id)
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
        self.model_input.setEnabled(True)
        self.env_btn.setEnabled(True)
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
        self.model_input.setEnabled(True)
        self.env_btn.setEnabled(True)
        self.log(f"任务失败: {error_text}")
        QMessageBox.warning(self, "失败", error_text)
