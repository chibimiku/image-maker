import base64
import json
import os
from datetime import datetime

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from utils.webui_img2img_client import WebuiImg2ImgClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_IMAGE_FILE = os.path.join(BASE_DIR, "config-image.json")
TEMP_INPUT_DIR = os.path.join(BASE_DIR, "data")


class ImageDropLabel(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(180)
        self.setStyleSheet(
            "QLabel { border: 2px dashed #999; background: #fafafa; color: #666; padding: 8px; }"
        )
        self.setText("拖拽图片到这里\n或点击“选择图片”\n或按 Ctrl+V 粘贴图片")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if os.path.isfile(path):
            self.file_dropped.emit(path)


class Flux2ClientWorker(QThread):
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, request_data):
        super().__init__()
        self.request_data = request_data

    def run(self):
        try:
            client = WebuiImg2ImgClient(
                base_url=self.request_data["base_url"],
                timeout=self.request_data["timeout"],
            )
            result = client.img2img_image_file(
                image_path=self.request_data["image_path"],
                prompt=self.request_data["prompt"],
                negative_prompt=self.request_data["negative_prompt"],
                steps=self.request_data["num_inference_steps"],
                cfg_scale=self.request_data["guidance_scale"],
                denoising_strength=self.request_data["strength"],
                num_images=self.request_data["num_images"],
                seed=self.request_data["seed"],
                width=self.request_data["width"],
                height=self.request_data["height"],
                sampler_name=self.request_data["sampler_name"],
                scheduler=self.request_data["scheduler"],
                sd_model=self.request_data["sd_model"],
                sd_vae=self.request_data["sd_vae"],
                extra_payload=self.request_data["extra_payload"],
                return_base64=self.request_data["return_base64"],
                output_dir="data",
            )
            self.finished_ok.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class Flux2ClientWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.current_image_path = ""
        self.current_result = {}
        self.worker = None
        self.init_ui()
        self.load_config()
        self.update_preview()

    def init_ui(self):
        layout = QVBoxLayout(self)

        cfg_group = QGroupBox("WebUI Img2Img 参数")
        cfg_layout = QFormLayout()

        self.base_url_input = QLineEdit("http://127.0.0.1:7860")
        cfg_layout.addRow("WebUI 地址:", self.base_url_input)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(10, 3600)
        self.timeout_spin.setValue(600)
        self.timeout_spin.setSuffix(" 秒")
        cfg_layout.addRow("请求超时:", self.timeout_spin)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setMinimumHeight(70)
        cfg_layout.addRow("Prompt:", self.prompt_edit)

        self.negative_prompt_edit = QTextEdit()
        self.negative_prompt_edit.setMinimumHeight(55)
        cfg_layout.addRow("Negative Prompt:", self.negative_prompt_edit)

        grid = QGridLayout()

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 200)
        self.steps_spin.setValue(28)
        grid.addWidget(QLabel("步数"), 0, 0)
        grid.addWidget(self.steps_spin, 0, 1)

        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(0.0, 30.0)
        self.cfg_spin.setSingleStep(0.1)
        self.cfg_spin.setValue(4.0)
        grid.addWidget(QLabel("CFG"), 0, 2)
        grid.addWidget(self.cfg_spin, 0, 3)

        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.01, 1.0)
        self.strength_spin.setSingleStep(0.01)
        self.strength_spin.setValue(0.72)
        grid.addWidget(QLabel("Strength"), 1, 0)
        grid.addWidget(self.strength_spin, 1, 1)

        self.num_images_spin = QSpinBox()
        self.num_images_spin.setRange(1, 8)
        self.num_images_spin.setValue(1)
        grid.addWidget(QLabel("输出张数"), 1, 2)
        grid.addWidget(self.num_images_spin, 1, 3)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2147483647)
        self.seed_spin.setValue(-1)
        grid.addWidget(QLabel("Seed"), 2, 0)
        grid.addWidget(self.seed_spin, 2, 1)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(0, 8192)
        self.width_spin.setSpecialValueText("不指定")
        self.width_spin.setValue(0)
        grid.addWidget(QLabel("Width"), 2, 2)
        grid.addWidget(self.width_spin, 2, 3)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(0, 8192)
        self.height_spin.setSpecialValueText("不指定")
        self.height_spin.setValue(0)
        grid.addWidget(QLabel("Height"), 3, 0)
        grid.addWidget(self.height_spin, 3, 1)

        self.return_base64_check = QCheckBox("返回 base64（便于预览）")
        self.return_base64_check.setChecked(True)
        grid.addWidget(self.return_base64_check, 3, 2, 1, 2)

        cfg_layout.addRow(grid)

        self.sampler_input = QLineEdit("Euler a")
        cfg_layout.addRow("Sampler:", self.sampler_input)

        self.scheduler_input = QLineEdit("Automatic")
        cfg_layout.addRow("Scheduler:", self.scheduler_input)

        self.sd_model_input = QLineEdit("")
        self.sd_model_input.setPlaceholderText("可选：指定 sd_model_checkpoint")
        cfg_layout.addRow("Checkpoint:", self.sd_model_input)

        self.sd_vae_input = QLineEdit("Automatic")
        cfg_layout.addRow("VAE:", self.sd_vae_input)

        self.extra_payload_edit = QTextEdit()
        self.extra_payload_edit.setMinimumHeight(70)
        self.extra_payload_edit.setPlaceholderText("可选：附加 JSON，将合并到 img2img payload")
        cfg_layout.addRow("附加 Payload(JSON):", self.extra_payload_edit)

        cfg_group.setLayout(cfg_layout)
        layout.addWidget(cfg_group)

        image_group = QGroupBox("输入图片")
        image_layout = QVBoxLayout()
        row = QHBoxLayout()
        self.image_path_input = QLineEdit()
        self.image_path_input.setPlaceholderText("图片路径（可选择/拖拽/粘贴）")
        row.addWidget(self.image_path_input, stretch=1)
        self.pick_btn = QPushButton("选择图片")
        self.pick_btn.clicked.connect(self.pick_image)
        row.addWidget(self.pick_btn)
        image_layout.addLayout(row)

        self.drop_area = ImageDropLabel()
        self.drop_area.file_dropped.connect(self.set_image_path)
        image_layout.addWidget(self.drop_area)
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        btn_row = QHBoxLayout()
        self.health_btn = QPushButton("检查服务")
        self.health_btn.clicked.connect(self.check_health)
        btn_row.addWidget(self.health_btn)
        self.run_btn = QPushButton("执行编辑")
        self.run_btn.clicked.connect(self.run_edit)
        btn_row.addWidget(self.run_btn)
        layout.addLayout(btn_row)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        layout.addWidget(self.log_text)

        self.result_preview = QLabel("暂无返回图片")
        self.result_preview.setAlignment(Qt.AlignCenter)
        self.result_preview.setMinimumHeight(220)
        self.result_preview.setStyleSheet("QLabel { border: 1px solid #ccc; background: #f7f7f7; }")
        layout.addWidget(self.result_preview)

        self.result_json = QTextEdit()
        self.result_json.setReadOnly(True)
        self.result_json.setMinimumHeight(180)
        layout.addWidget(self.result_json)

        self.image_path_input.textChanged.connect(self.on_image_path_changed)
        self.base_url_input.editingFinished.connect(self.save_config)
        self.prompt_edit.textChanged.connect(self.save_config)
        self.negative_prompt_edit.textChanged.connect(self.save_config)
        self.timeout_spin.valueChanged.connect(lambda _: self.save_config())
        self.steps_spin.valueChanged.connect(lambda _: self.save_config())
        self.cfg_spin.valueChanged.connect(lambda _: self.save_config())
        self.strength_spin.valueChanged.connect(lambda _: self.save_config())
        self.num_images_spin.valueChanged.connect(lambda _: self.save_config())
        self.seed_spin.valueChanged.connect(lambda _: self.save_config())
        self.width_spin.valueChanged.connect(lambda _: self.save_config())
        self.height_spin.valueChanged.connect(lambda _: self.save_config())
        self.return_base64_check.stateChanged.connect(lambda _: self.save_config())
        self.sampler_input.editingFinished.connect(self.save_config)
        self.scheduler_input.editingFinished.connect(self.save_config)
        self.sd_model_input.editingFinished.connect(self.save_config)
        self.sd_vae_input.editingFinished.connect(self.save_config)
        self.extra_payload_edit.textChanged.connect(self.save_config)

        QShortcut(QKeySequence("Ctrl+V"), self, activated=self.paste_from_clipboard)

    def log(self, text):
        self.log_text.append(text)
        bar = self.log_text.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _load_full_image_config(self):
        if not os.path.exists(CONFIG_IMAGE_FILE):
            return {}
        try:
            with open(CONFIG_IMAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_full_image_config(self, data):
        os.makedirs(os.path.dirname(CONFIG_IMAGE_FILE), exist_ok=True)
        with open(CONFIG_IMAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load_config(self):
        full_config = self._load_full_image_config()
        config = full_config.get("webui_img2img", full_config.get("flux2_client", {}))
        self.base_url_input.setText(str(config.get("base_url", "http://127.0.0.1:7860")))
        self.timeout_spin.setValue(int(config.get("timeout", 600)))
        self.prompt_edit.setPlainText(str(config.get("prompt", "")))
        self.negative_prompt_edit.setPlainText(str(config.get("negative_prompt", "")))
        self.steps_spin.setValue(int(config.get("num_inference_steps", 28)))
        self.cfg_spin.setValue(float(config.get("guidance_scale", 4.0)))
        self.strength_spin.setValue(float(config.get("strength", 0.72)))
        self.num_images_spin.setValue(int(config.get("num_images", 1)))
        self.seed_spin.setValue(int(config.get("seed", -1)))
        self.width_spin.setValue(int(config.get("width", 0) or 0))
        self.height_spin.setValue(int(config.get("height", 0) or 0))
        self.return_base64_check.setChecked(bool(config.get("return_base64", True)))
        self.sampler_input.setText(str(config.get("sampler_name", "Euler a")))
        self.scheduler_input.setText(str(config.get("scheduler", "Automatic")))
        self.sd_model_input.setText(str(config.get("sd_model", "")))
        self.sd_vae_input.setText(str(config.get("sd_vae", "Automatic")))
        self.extra_payload_edit.setPlainText(str(config.get("extra_payload", "")))

        saved_image = str(config.get("image_path", "")).strip()
        if saved_image and os.path.isfile(saved_image):
            self.current_image_path = saved_image
            self.image_path_input.setText(saved_image)

    def save_config(self):
        existing = self._load_full_image_config()
        existing["webui_img2img"] = {
            "base_url": self.base_url_input.text().strip(),
            "timeout": int(self.timeout_spin.value()),
            "prompt": self.prompt_edit.toPlainText().strip(),
            "negative_prompt": self.negative_prompt_edit.toPlainText().strip(),
            "num_inference_steps": int(self.steps_spin.value()),
            "guidance_scale": float(self.cfg_spin.value()),
            "strength": float(self.strength_spin.value()),
            "num_images": int(self.num_images_spin.value()),
            "seed": int(self.seed_spin.value()),
            "width": int(self.width_spin.value()),
            "height": int(self.height_spin.value()),
            "return_base64": bool(self.return_base64_check.isChecked()),
            "sampler_name": self.sampler_input.text().strip(),
            "scheduler": self.scheduler_input.text().strip(),
            "sd_model": self.sd_model_input.text().strip(),
            "sd_vae": self.sd_vae_input.text().strip(),
            "extra_payload": self.extra_payload_edit.toPlainText().strip(),
            "image_path": self.current_image_path,
        }
        try:
            self._save_full_image_config(existing)
        except Exception:
            pass

    def pick_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择输入图片",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if file_path:
            self.set_image_path(file_path)

    def on_image_path_changed(self):
        text = self.image_path_input.text().strip()
        if os.path.isfile(text):
            self.current_image_path = text
            self.update_preview()
            self.save_config()

    def set_image_path(self, path):
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "提示", "图片路径无效")
            return
        self.current_image_path = path
        self.image_path_input.setText(path)
        self.update_preview()
        self.save_config()

    def _save_clipboard_image(self, qimage):
        today = datetime.now().strftime("%Y%m%d")
        out_dir = os.path.join(TEMP_INPUT_DIR, today, "webui-img2img-input")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"pasted-{datetime.now().strftime('%H%M%S')}.png")
        ok = qimage.save(out_path, "PNG")
        return out_path if ok else ""

    def paste_from_clipboard(self):
        clip = QApplication.clipboard()
        mime = clip.mimeData()
        if mime.hasUrls():
            for url in mime.urls():
                path = url.toLocalFile()
                if os.path.isfile(path):
                    self.set_image_path(path)
                    self.log(f"已从剪贴板文件粘贴: {path}")
                    return
        img = clip.image()
        if img and not img.isNull():
            out_path = self._save_clipboard_image(img)
            if out_path:
                self.set_image_path(out_path)
                self.log(f"已从剪贴板图片粘贴并保存: {out_path}")
                return
        self.log("剪贴板中没有可用图片")

    def update_preview(self):
        if not self.current_image_path or not os.path.isfile(self.current_image_path):
            self.drop_area.setText("拖拽图片到这里\n或点击“选择图片”\n或按 Ctrl+V 粘贴图片")
            self.result_preview.setText("暂无返回图片")
            return
        pix = QPixmap(self.current_image_path)
        if pix.isNull():
            self.drop_area.setText("图片加载失败")
        else:
            self.drop_area.setPixmap(
                pix.scaled(self.drop_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_preview()

    def _build_request_data(self):
        if not self.current_image_path or not os.path.isfile(self.current_image_path):
            raise ValueError("请先选择一张输入图片")
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            raise ValueError("Prompt 不能为空")
        extra_payload_text = self.extra_payload_edit.toPlainText().strip()
        extra_payload = None
        if extra_payload_text:
            try:
                loaded = json.loads(extra_payload_text)
                if not isinstance(loaded, dict):
                    raise ValueError("附加 Payload 必须是 JSON 对象")
                extra_payload = loaded
            except Exception as exc:
                raise ValueError(f"附加 Payload JSON 无效: {exc}") from exc
        return {
            "base_url": self.base_url_input.text().strip(),
            "timeout": int(self.timeout_spin.value()),
            "image_path": self.current_image_path,
            "prompt": prompt,
            "negative_prompt": self.negative_prompt_edit.toPlainText().strip(),
            "num_inference_steps": int(self.steps_spin.value()),
            "guidance_scale": float(self.cfg_spin.value()),
            "strength": float(self.strength_spin.value()),
            "num_images": int(self.num_images_spin.value()),
            "seed": int(self.seed_spin.value()),
            "width": int(self.width_spin.value()) or None,
            "height": int(self.height_spin.value()) or None,
            "sampler_name": self.sampler_input.text().strip() or "Euler a",
            "scheduler": self.scheduler_input.text().strip() or "Automatic",
            "sd_model": self.sd_model_input.text().strip(),
            "sd_vae": self.sd_vae_input.text().strip() or "Automatic",
            "extra_payload": extra_payload,
            "return_base64": bool(self.return_base64_check.isChecked()),
        }

    def set_running(self, running):
        self.run_btn.setEnabled(not running)
        self.health_btn.setEnabled(not running)
        self.pick_btn.setEnabled(not running)

    def check_health(self):
        try:
            client = WebuiImg2ImgClient(self.base_url_input.text().strip(), timeout=15)
            health = client.health()
            self.log(f"WebUI 可用: {json.dumps(health, ensure_ascii=False)}")
        except Exception as e:
            QMessageBox.warning(self, "检查失败", str(e))

    def run_edit(self):
        try:
            req = self._build_request_data()
        except Exception as e:
            QMessageBox.warning(self, "参数错误", str(e))
            return
        self.save_config()
        self.set_running(True)
        self.log("开始调用 WebUI img2img ...")
        self.worker = Flux2ClientWorker(req)
        self.worker.finished_ok.connect(self.on_edit_success)
        self.worker.failed.connect(self.on_edit_failed)
        self.worker.start()

    def on_edit_success(self, result):
        self.set_running(False)
        self.current_result = result or {}
        self.result_json.setPlainText(json.dumps(self.current_result, ensure_ascii=False, indent=2))
        self.log("请求完成")

        outputs = self.current_result.get("outputs", []) if isinstance(self.current_result, dict) else []
        if outputs:
            first = outputs[0]
            image_base64 = first.get("image_base64")
            file_path = first.get("file_path", "")
            if image_base64:
                try:
                    raw = base64.b64decode(image_base64)
                    pix = QPixmap()
                    pix.loadFromData(raw, "PNG")
                    self.result_preview.setPixmap(
                        pix.scaled(self.result_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    )
                except Exception:
                    self.result_preview.setText("base64 预览失败")
            elif file_path and os.path.isfile(file_path):
                pix = QPixmap(file_path)
                self.result_preview.setPixmap(
                    pix.scaled(self.result_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            else:
                self.result_preview.setText("无可预览图片")

    def on_edit_failed(self, error_text):
        self.set_running(False)
        self.log(f"请求失败: {error_text}")
        self.result_json.setPlainText(str(error_text))
        QMessageBox.warning(self, "调用失败", error_text)
