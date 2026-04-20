import os
import importlib

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from utils.image_upscale_runtime import (
    JpgAutoUpscaleThread,
    get_loaded_upscale_models_status,
    list_esrgan_models,
    normalize_upscale_options,
    release_loaded_upscale_models,
)


class UpscalerTabWidget(QWidget):
    def __init__(self, options_getter=None, options_changed_callback=None):
        super().__init__()
        self.get_options = options_getter
        self.on_options_changed = options_changed_callback
        self.image_files = []
        self.worker_thread = None
        self.initUI()
        self.reload_models()
        self.load_saved_options()

    def initUI(self):
        self.setAcceptDrops(True)
        main_layout = QVBoxLayout()

        drag_label = QLabel("请将图片拖拽至此（支持多图），输出为同名 -fixed.png")
        drag_label.setAlignment(Qt.AlignCenter)
        drag_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px dashed #aaa; padding: 16px; }")
        drag_label.setMinimumHeight(88)
        main_layout.addWidget(drag_label)

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.image_list.setMinimumHeight(140)
        main_layout.addWidget(self.image_list)

        list_btn_layout = QHBoxLayout()
        self.pick_btn = QPushButton("选择图片")
        self.pick_btn.clicked.connect(self.pick_images)
        self.remove_btn = QPushButton("移除选中")
        self.remove_btn.clicked.connect(self.remove_selected)
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.clicked.connect(self.clear_all)
        list_btn_layout.addWidget(self.pick_btn)
        list_btn_layout.addWidget(self.remove_btn)
        list_btn_layout.addWidget(self.clear_btn)
        main_layout.addLayout(list_btn_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("主模型:"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(False)
        self.model_combo.currentTextChanged.connect(self.persist_options)
        model_layout.addWidget(self.model_combo, stretch=1)
        self.reload_models_btn = QPushButton("刷新模型")
        self.reload_models_btn.clicked.connect(self.reload_models)
        model_layout.addWidget(self.reload_models_btn)
        main_layout.addLayout(model_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("放大模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("按倍率", 0)
        self.mode_combo.addItem("按目标尺寸", 1)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addWidget(QLabel("倍率:"))
        self.by_spin = QDoubleSpinBox()
        self.by_spin.setRange(1.0, 8.0)
        self.by_spin.setSingleStep(0.1)
        self.by_spin.setValue(2.0)
        self.by_spin.valueChanged.connect(self.persist_options)
        mode_layout.addWidget(self.by_spin)
        mode_layout.addWidget(QLabel("最大边(0=不限制):"))
        self.max_side_spin = QSpinBox()
        self.max_side_spin.setRange(0, 16384)
        self.max_side_spin.valueChanged.connect(self.persist_options)
        mode_layout.addWidget(self.max_side_spin)
        main_layout.addLayout(mode_layout)

        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("目标宽:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 16384)
        self.width_spin.valueChanged.connect(self.persist_options)
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("目标高:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 16384)
        self.height_spin.valueChanged.connect(self.persist_options)
        size_layout.addWidget(self.height_spin)
        self.crop_cb = QCheckBox("裁切到目标尺寸")
        self.crop_cb.toggled.connect(self.persist_options)
        size_layout.addWidget(self.crop_cb)
        main_layout.addLayout(size_layout)

        second_layout = QHBoxLayout()
        second_layout.addWidget(QLabel("次模型(可选):"))
        self.secondary_model_combo = QComboBox()
        self.secondary_model_combo.currentTextChanged.connect(self.persist_options)
        second_layout.addWidget(self.secondary_model_combo, stretch=1)
        second_layout.addWidget(QLabel("混合可见度:"))
        self.secondary_visibility_spin = QDoubleSpinBox()
        self.secondary_visibility_spin.setRange(0.0, 1.0)
        self.secondary_visibility_spin.setSingleStep(0.1)
        self.secondary_visibility_spin.valueChanged.connect(self.persist_options)
        second_layout.addWidget(self.secondary_visibility_spin)
        main_layout.addLayout(second_layout)

        misc_layout = QHBoxLayout()
        misc_layout.addWidget(QLabel("缓存大小:"))
        self.cache_size_spin = QSpinBox()
        self.cache_size_spin.setRange(1, 128)
        self.cache_size_spin.valueChanged.connect(self.persist_options)
        misc_layout.addWidget(self.cache_size_spin)
        misc_layout.addWidget(QLabel("WebP 目标体积(MB):"))
        self.webp_target_spin = QDoubleSpinBox()
        self.webp_target_spin.setRange(0.1, 100.0)
        self.webp_target_spin.setDecimals(1)
        self.webp_target_spin.setSingleStep(0.5)
        self.webp_target_spin.valueChanged.connect(self.persist_options)
        misc_layout.addWidget(self.webp_target_spin)
        misc_layout.addWidget(QLabel("推理设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU优先(推荐稳定)", "cpu")
        self.device_combo.addItem("自动CUDA(有GPU时启用)", "auto")
        self.device_combo.currentIndexChanged.connect(self.persist_options)
        misc_layout.addWidget(self.device_combo)
        misc_layout.addWidget(QLabel("下采样算法:"))
        self.downsample_combo = QComboBox()
        self.downsample_combo.addItem("Lanczos(锐利)", "lanczos")
        self.downsample_combo.addItem("Area/Box(稳妥抗锯齿)", "area")
        self.downsample_combo.addItem("Bicubic(平滑)", "bicubic")
        self.downsample_combo.addItem("Mitchell(Wand，可选)", "mitchell_wand")
        self.downsample_combo.currentIndexChanged.connect(self.persist_options)
        misc_layout.addWidget(self.downsample_combo)
        main_layout.addLayout(misc_layout)

        self.loading_label = QLabel("状态: 空闲")
        main_layout.addWidget(self.loading_label)
        self.model_cache_status_label = QLabel("模型状态: 未加载")
        main_layout.addWidget(self.model_cache_status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("进度: 0/0")
        main_layout.addWidget(self.progress_bar)

        action_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始优化")
        self.start_btn.clicked.connect(self.start_upscale)
        action_layout.addWidget(self.start_btn)
        self.release_models_btn = QPushButton("释放模型")
        self.release_models_btn.clicked.connect(self.release_loaded_models)
        action_layout.addWidget(self.release_models_btn)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_upscale)
        action_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(action_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)
        self.setLayout(main_layout)

    def _on_mode_changed(self):
        mode = self.mode_combo.currentData()
        is_size_mode = int(mode) == 1
        self.width_spin.setEnabled(is_size_mode)
        self.height_spin.setEnabled(is_size_mode)
        self.crop_cb.setEnabled(is_size_mode)
        self.by_spin.setEnabled(not is_size_mode)
        self.persist_options()

    def reload_models(self):
        models = list_esrgan_models()
        current_primary = self.model_combo.currentText().strip()
        current_secondary = self.secondary_model_combo.currentText().strip()
        self.model_combo.blockSignals(True)
        self.secondary_model_combo.blockSignals(True)
        self.model_combo.clear()
        self.secondary_model_combo.clear()
        self.secondary_model_combo.addItem("")
        self.model_combo.addItems(models)
        self.secondary_model_combo.addItems(models)
        if current_primary:
            self.model_combo.setCurrentText(current_primary)
        if current_secondary:
            self.secondary_model_combo.setCurrentText(current_secondary)
        self.model_combo.blockSignals(False)
        self.secondary_model_combo.blockSignals(False)
        if not models:
            self.log_msg("未找到 ESRGAN 模型，请将模型文件放到 data/models/ESRGAN 或 models/ESRGAN")
        self.persist_options()

    def load_saved_options(self):
        raw = self.get_options() if callable(self.get_options) else {}
        opts = normalize_upscale_options(raw)
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(1 if int(opts["upscale_mode"]) == 1 else 0)
        self.mode_combo.blockSignals(False)
        self.by_spin.setValue(float(opts["upscale_by"]))
        self.max_side_spin.setValue(int(opts["max_side_length"]))
        self.width_spin.setValue(int(opts["upscale_to_width"]))
        self.height_spin.setValue(int(opts["upscale_to_height"]))
        self.crop_cb.setChecked(bool(opts["upscale_crop"]))
        self.cache_size_spin.setValue(int(opts["cache_size"]))
        self.secondary_visibility_spin.setValue(float(opts["upscaler_2_visibility"]))
        self.webp_target_spin.setValue(float(opts["webp_target_mb"]))
        self.device_combo.setCurrentIndex(1 if str(opts.get("inference_device", "cpu")) == "auto" else 0)
        downsample = str(opts.get("downsample_method", "lanczos"))
        downsample_idx = self.downsample_combo.findData(downsample)
        self.downsample_combo.setCurrentIndex(0 if downsample_idx < 0 else downsample_idx)
        if opts["model_name"]:
            self.model_combo.setCurrentText(opts["model_name"])
        if opts["upscaler_2_name"]:
            self.secondary_model_combo.setCurrentText(opts["upscaler_2_name"])
        self._on_mode_changed()
        self._refresh_model_status_bar()

    def collect_options(self):
        model_name = self.model_combo.currentText().strip()
        mode = int(self.mode_combo.currentData())
        prev = normalize_upscale_options(self.get_options() if callable(self.get_options) else {})
        options = {
            "enabled": bool(prev.get("enabled", False)),
            "model_name": model_name,
            "upscale_mode": mode,
            "upscale_by": float(self.by_spin.value()),
            "max_side_length": int(self.max_side_spin.value()),
            "upscale_to_width": int(self.width_spin.value()),
            "upscale_to_height": int(self.height_spin.value()),
            "upscale_crop": bool(self.crop_cb.isChecked()),
            "upscaler_2_name": self.secondary_model_combo.currentText().strip(),
            "upscaler_2_visibility": float(self.secondary_visibility_spin.value()),
            "cache_size": int(self.cache_size_spin.value()),
            "webp_target_mb": float(self.webp_target_spin.value()),
            "inference_device": str(self.device_combo.currentData()),
            "downsample_method": str(self.downsample_combo.currentData()),
        }
        return normalize_upscale_options(options)

    def persist_options(self):
        if callable(self.on_options_changed):
            self.on_options_changed(self.collect_options())

    def log_msg(self, text):
        self.log_text.append(str(text))
        sb = self.log_text.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def add_images(self, paths):
        for path in paths:
            norm = os.path.abspath(path)
            if not os.path.isfile(norm):
                continue
            if not norm.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                continue
            if norm in self.image_files:
                continue
            self.image_files.append(norm)
            item = QListWidgetItem(os.path.basename(norm))
            item.setData(Qt.ItemDataRole.UserRole, norm)
            self.image_list.addItem(item)
        self._update_progress()

    def pick_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图片",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if files:
            self.add_images(files)

    def remove_selected(self):
        items = self.image_list.selectedItems()
        for item in items:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path in self.image_files:
                self.image_files.remove(path)
            self.image_list.takeItem(self.image_list.row(item))
        self._update_progress()

    def clear_all(self):
        self.image_files = []
        self.image_list.clear()
        self._update_progress()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.add_images(paths)

    def _set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.release_models_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.pick_btn.setEnabled(not running)
        self.remove_btn.setEnabled(not running)
        self.clear_btn.setEnabled(not running)
        self.reload_models_btn.setEnabled(not running)

    def start_upscale(self):
        if not self.image_files:
            QMessageBox.warning(self, "提示", "请先添加图片。")
            return
        options = self.collect_options()
        if not options.get("model_name"):
            QMessageBox.warning(self, "提示", "请先选择主模型。")
            return
        self.persist_options()
        self.loading_label.setText("状态: 处理中...")
        self.log_msg(f"开始处理，共 {len(self.image_files)} 张。")
        self._set_running(True)
        self.progress_bar.setRange(0, max(1, len(self.image_files)))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"进度: 0/{len(self.image_files)}")
        self.worker_thread = JpgAutoUpscaleThread(
            image_paths=list(self.image_files),
            options=options,
            task_name="Upscaler",
        )
        self.worker_thread.log_signal.connect(self.log_msg)
        self.worker_thread.progress_signal.connect(self._on_progress)
        self.worker_thread.finish_signal.connect(self.on_upscale_finished)
        self.worker_thread.start()
        self._log_loaded_models_status(prefix="启动前缓存状态")
        self._refresh_model_status_bar()

    def cancel_upscale(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.request_cancel()
            self.log_msg("已请求取消当前任务...")
        self._refresh_model_status_bar()

    def release_loaded_models(self):
        result = release_loaded_upscale_models()
        released = int(result.get("released_models", 0))
        self.log_msg(f"已手动释放放大模型缓存，释放数量: {released}")
        if result.get("cuda_cache_cleared"):
            self.log_msg("已尝试清理 CUDA 缓存。")
        cuda_error = str(result.get("cuda_error", "") or "").strip()
        if cuda_error:
            self.log_msg(f"清理 CUDA 缓存时出现提示: {cuda_error}")
        self._log_loaded_models_status(prefix="释放后缓存状态")
        self._refresh_model_status_bar()

    def on_upscale_finished(self, results):
        total = len(results or [])
        success_count = 0
        for idx, item in enumerate(results or [], start=1):
            self.progress_bar.setValue(idx)
            self.progress_bar.setFormat(f"进度: {idx}/{max(1, total)}")
            if item.get("fixed_png_path") and not item.get("error"):
                success_count += 1
        self._set_running(False)
        self.loading_label.setText("状态: 空闲")
        self.log_msg(f"处理完成，成功 {success_count}/{total}")
        self._log_loaded_models_status(prefix="任务完成后缓存状态")
        self._refresh_model_status_bar()

    def _on_progress(self, current, total):
        self.progress_bar.setRange(0, max(1, int(total)))
        self.progress_bar.setValue(int(current))
        self.progress_bar.setFormat(f"进度: {int(current)}/{int(total)}")

    def _update_progress(self):
        total = len(self.image_files)
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"进度: 0/{total}")

    def _log_loaded_models_status(self, prefix: str = "模型缓存状态"):
        status = get_loaded_upscale_models_status()
        if not status:
            self.log_msg(f"{prefix}: 当前无已加载模型")
            return
        self.log_msg(f"{prefix}: 已加载 {len(status)} 个模型")
        for model_name, meta in status.items():
            display_name = meta.get("model_name") or model_name
            mode = meta.get("mode", "cpu")
            device = meta.get("device", "unknown")
            arch = meta.get("arch", "unknown")
            native_scale = meta.get("native_scale", "unknown")
            self.log_msg(
                f"- {display_name}: mode={mode}, arch={arch}, native_scale={native_scale}, device={device}"
            )

    def _refresh_model_status_bar(self):
        self.model_cache_status_label.setText(self._build_model_status_text())

    def _build_model_status_text(self) -> str:
        status = get_loaded_upscale_models_status()
        if not status:
            return "模型状态: 未加载"

        loaded_count = len(status)
        devices = sorted({str(meta.get("device", "unknown")) for meta in status.values()})
        modes = sorted({str(meta.get("mode", "cpu")) for meta in status.values()})
        base_text = f"模型状态: 已加载 {loaded_count} 个 | mode={','.join(modes)} | device={','.join(devices)}"

        if "cuda" not in devices:
            return base_text

        try:
            torch = importlib.import_module("torch")
            if not torch.cuda.is_available():
                return base_text + " | CUDA不可用"
            mem_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
            mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            return f"{base_text} | GPU={gpu_name} | 显存 已分配/保留={mem_alloc:.0f}/{mem_reserved:.0f} MB"
        except Exception:
            return base_text + " | 显存信息不可用"
