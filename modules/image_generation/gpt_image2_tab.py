"""gpt-image-2 专用 Tab（同一界面切换 new.aigc2d / autodl 两个站点）。

- 站点下拉：
  - `new.aigc2d` → `apis.aigc-2d-gpt`，走 `generate_image_aigc2d_gpt`（尺寸仅 3 档）
  - `autodl` → `apis.autodl`，走 `generate_image_openai_image`（尺寸含 auto / 1792x1024）
- 模式下拉：生图 / 编辑
  - 生图：不挂参考图 → `/v1/images/generations`；挂了参考图 → 自动走 `/v1/images/edits` 垫图
  - 编辑：至少 1 张图
- 两个站点都支持 gpt-image-2，参考图上限 16 张
- 配置只从 conf/config-image.json 读取（`apis.*` + 顶层 `gpt_image2` 记忆），不受「设置」里全局 API 类型影响
"""
import json
import os

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from modules.others.api_backend import (
    CONFIG_DIR,
    GPT_IMAGE2_MAX_REFERENCE_IMAGES,
    GPT_IMAGE2_OUTPUT_FORMATS,
    GPT_IMAGE2_SITE_AIGC2D,
    GPT_IMAGE2_SITE_API_TYPES,
    GPT_IMAGE2_SITE_AUTODL,
    GPT_IMAGE2_SITE_DEFAULT_SIZES,
    GPT_IMAGE2_SITE_FILE_PREFIXES,
    GPT_IMAGE2_SITE_SAVE_SUB_DIRS,
    GPT_IMAGE2_SITE_SIZES,
    GPT_IMAGE2_SIZE_SQUARE,
    generate_image_aigc2d_gpt,
    generate_image_openai_image,
    get_api_config,
    normalize_gpt_image2_size,
)

# 站点/接口映射统一来自 api_backend（GUI 与 tools/gpt_image2_gen.py 共用一份，避免漂移）
SITE_AIGC2D = GPT_IMAGE2_SITE_AIGC2D
SITE_AUTODL = GPT_IMAGE2_SITE_AUTODL
API_TYPE_BY_SITE = GPT_IMAGE2_SITE_API_TYPES
SAVE_SUB_DIR_BY_SITE = GPT_IMAGE2_SITE_SAVE_SUB_DIRS
FILE_PREFIX_BY_SITE = GPT_IMAGE2_SITE_FILE_PREFIXES
DEFAULT_SIZE_BY_SITE = GPT_IMAGE2_SITE_DEFAULT_SIZES

SIZE_LABELS = {
    "auto": "自动 auto",
    "1024x1024": "方形 1024x1024",
    "1536x1024": "横向 1536x1024",
    "1792x1024": "宽幅 1792x1024",
    "1024x1536": "纵向 1024x1536",
}
# aigc2d 通道只认 3 档尺寸；autodl 通道含 auto / 1792x1024（清单在 api_backend）
SIZE_CHOICES_BY_SITE = {
    site: tuple((SIZE_LABELS.get(value, value), value) for value in sizes)
    for site, sizes in GPT_IMAGE2_SITE_SIZES.items()
}
QUALITY_CHOICES = (
    ("high 精细(较慢较贵)", "high"),
    ("medium 标准", "medium"),
    ("low 快速草稿", "low"),
    ("auto 服务端默认", "auto"),
)

CONFIG_IMAGE_FILE = os.path.join(CONFIG_DIR, "config-image.json")
CONFIG_NODE = "gpt_image2"

MODE_GENERATE = "生图"
MODE_EDIT = "编辑"
IMAGE_FILTER = "图片 (*.png *.jpg *.jpeg *.webp *.bmp)"


def _read_config_image() -> dict:
    try:
        with open(CONFIG_IMAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _set_combo_by_data(combo: QComboBox, value, fallback=None):
    for candidate in (value, fallback):
        if candidate is None:
            continue
        index = combo.findData(candidate)
        if index < 0:
            index = combo.findText(str(candidate))
        if index >= 0:
            combo.setCurrentIndex(index)
            return


class GptImage2Worker(QThread):
    """按站点调用对应后端；aigc2d 通道才支持取消与实时日志。"""

    log = pyqtSignal(str)
    done = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, backend, params: dict, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.params = dict(params or {})

    def run(self):
        params = dict(self.params)
        if self.backend is generate_image_aigc2d_gpt:
            params.setdefault("cancel_check", lambda: self.isInterruptionRequested())
            params.setdefault("log_callback", lambda message: self.log.emit(str(message)))
        try:
            files = self.backend(**params)
            self.done.emit(list(files or []))
        except Exception as exc:  # noqa: BLE001 - 线程内异常统一回传 UI
            self.error.emit(str(exc))


class GptImage2Widget(QWidget):
    """gpt-image-2 生图 / 编辑台（new.aigc2d 与 autodl 双站点）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_paths = []
        self._worker = None
        self._loading = False
        self.setAcceptDrops(True)
        self.initUI()
        self.load_defaults()
        self.refresh_api_hint()

    # ---------------- UI ----------------
    def initUI(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.site_combo = QComboBox()
        self.site_combo.addItem(SITE_AIGC2D, SITE_AIGC2D)
        self.site_combo.addItem(SITE_AUTODL, SITE_AUTODL)
        self.site_combo.currentIndexChanged.connect(lambda _index: self.on_site_changed(self.current_site()))
        form.addRow("站点:", self.site_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([MODE_GENERATE, MODE_EDIT])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        form.addRow("模式:", self.mode_combo)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItem("gpt-image-2")
        form.addRow("模型:", self.model_combo)

        self.size_combo = QComboBox()
        form.addRow("尺寸:", self.size_combo)

        self.quality_combo = QComboBox()
        for label, value in QUALITY_CHOICES:
            self.quality_combo.addItem(label, value)
        form.addRow("画质:", self.quality_combo)

        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(list(GPT_IMAGE2_OUTPUT_FORMATS))
        form.addRow("输出格式:", self.output_format_combo)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 10)
        self.n_spin.setValue(1)
        form.addRow("张数:", self.n_spin)
        layout.addLayout(form)

        self.api_hint = QLabel("")
        self.api_hint.setWordWrap(True)
        layout.addWidget(self.api_hint)

        layout.addWidget(QLabel("提示词 (prompt):"))
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("生图: 描述画面; 编辑: 说明要改什么、保留什么")
        self.prompt_edit.setMinimumHeight(90)
        layout.addWidget(self.prompt_edit)

        layout.addWidget(
            QLabel(
                f"参考图 (生图=垫图可留空; 编辑=必填; 最多 {GPT_IMAGE2_MAX_REFERENCE_IMAGES} 张，可拖拽图片进来):"
            )
        )
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(100)
        layout.addWidget(self.image_list)

        img_btn_row = QHBoxLayout()
        self.add_btn = QPushButton("添加图片")
        self.add_btn.clicked.connect(self.add_images)
        img_btn_row.addWidget(self.add_btn)
        self.clear_btn = QPushButton("清空图片")
        self.clear_btn.clicked.connect(self.clear_images)
        img_btn_row.addWidget(self.clear_btn)
        self.refresh_cfg_btn = QPushButton("刷新接口配置")
        self.refresh_cfg_btn.clicked.connect(self.refresh_api_hint)
        img_btn_row.addWidget(self.refresh_cfg_btn)
        img_btn_row.addStretch(1)
        layout.addLayout(img_btn_row)

        action_row = QHBoxLayout()
        self.generate_btn = QPushButton("开始生成")
        self.generate_btn.clicked.connect(self.generate)
        action_row.addWidget(self.generate_btn)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_generation)
        action_row.addWidget(self.cancel_btn)
        self.status_label = QLabel("就绪")
        action_row.addWidget(self.status_label, stretch=1)
        layout.addLayout(action_row)

        body_row = QHBoxLayout()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("运行日志")
        body_row.addWidget(self.log_view, stretch=1)
        self.preview_label = QLabel("预览")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(220, 220)
        self.preview_label.setStyleSheet("border: 1px solid #888;")
        body_row.addWidget(self.preview_label, stretch=1)
        layout.addLayout(body_row, stretch=1)

        self.result_list = QListWidget()
        self.result_list.setMaximumHeight(90)
        self.result_list.itemDoubleClicked.connect(self.open_item)
        layout.addWidget(self.result_list)

        self._populate_sizes(SITE_AIGC2D)
        self.on_mode_changed(self.mode_combo.currentText())

    def current_site(self) -> str:
        return str(self.site_combo.currentData() or SITE_AIGC2D)

    def on_site_changed(self, site: str):
        if self._loading:
            return
        self._apply_site_defaults(site)
        self.refresh_api_hint()
        self._append_log(f"已切换到站点: {site}（API类型 {API_TYPE_BY_SITE.get(site, '')}）")

    def on_mode_changed(self, mode: str):
        editing = str(mode) == MODE_EDIT
        self.prompt_edit.setPlaceholderText(
            "编辑: 说明要改什么、保留什么（例: 把背景换成日落海滩，保留人物姿势与服装）"
            if editing
            else "生图: 描述画面（挂了参考图就作为垫图参考）"
        )

    def _populate_sizes(self, site: str):
        self.size_combo.blockSignals(True)
        self.size_combo.clear()
        for label, value in SIZE_CHOICES_BY_SITE.get(site, SIZE_CHOICES_BY_SITE[SITE_AIGC2D]):
            self.size_combo.addItem(label, value)
        self.size_combo.blockSignals(False)

    def _populate_models(self, site: str, data: dict):
        api_type = API_TYPE_BY_SITE.get(site, "aigc-2d-gpt")
        api_cfg = ((data or {}).get("apis") or {}).get(api_type) or {}
        model = str(api_cfg.get("model") or "gpt-image-2")
        if self.model_combo.findText(model) < 0:
            self.model_combo.addItem(model)
        self.model_combo.setCurrentText(model)

    # ---------------- 配置 ----------------
    def refresh_api_hint(self):
        site = self.current_site()
        api_type = API_TYPE_BY_SITE.get(site, "")
        cfg = get_api_config(api_type=api_type)
        base_url = str(cfg.get("base_url") or "(未配置)")
        key_state = "已配置" if str(cfg.get("api_key") or "").strip() else "未配置(请到设置→图片生成 API 填写)"
        model = str(cfg.get("model") or "gpt-image-2")
        cancel_state = "支持取消" if site == SITE_AIGC2D else "不支持中断(请求返回后才结束)"
        self.api_hint.setText(
            f"站点: {site}  (API类型 {api_type})    接口: {base_url}    模型: {model}    "
            f"API Key: {key_state}    取消: {cancel_state}"
        )

    def _site_node(self, data: dict, site: str) -> dict:
        node = data.get(CONFIG_NODE) if isinstance(data.get(CONFIG_NODE), dict) else {}
        sites = node.get("sites") if isinstance(node.get("sites"), dict) else {}
        per_site = sites.get(site) if isinstance(sites.get(site), dict) else {}
        per_site = dict(per_site)
        if site == SITE_AIGC2D:
            # 兼容早期扁平结构 {"size": ..., "quality": ...}
            for key in ("model", "size", "quality", "output_format", "n"):
                if key not in per_site and key in node:
                    per_site[key] = node[key]
        return per_site

    def _apply_site_defaults(self, site: str):
        data = _read_config_image()
        self._populate_sizes(site)
        self._populate_models(site, data)
        per_site = self._site_node(data, site)
        fallback_size = DEFAULT_SIZE_BY_SITE.get(site, GPT_IMAGE2_SIZE_SQUARE)
        saved_size = per_site.get("size")
        if saved_size and site == SITE_AIGC2D:
            saved_size = normalize_gpt_image2_size(size=saved_size)
        _set_combo_by_data(self.size_combo, saved_size, fallback_size)
        _set_combo_by_data(self.quality_combo, per_site.get("quality"), "high")
        output_format = str(per_site.get("output_format") or "png").lower()
        if self.output_format_combo.findText(output_format) >= 0:
            self.output_format_combo.setCurrentText(output_format)
        try:
            saved_n = max(1, min(10, int(per_site.get("n") or 1)))
        except (TypeError, ValueError):
            saved_n = 1
        self.n_spin.setValue(saved_n)
        # autodl 通道一次一张，没有 n 参数
        self.n_spin.setEnabled(site != SITE_AUTODL)

    def load_defaults(self):
        data = _read_config_image()
        node = data.get(CONFIG_NODE) if isinstance(data.get(CONFIG_NODE), dict) else {}
        self._loading = True
        try:
            site = str(node.get("site") or SITE_AIGC2D)
            if self.site_combo.findData(site) < 0:
                site = SITE_AIGC2D
            self.site_combo.setCurrentIndex(self.site_combo.findData(site))
            self._apply_site_defaults(site)
            mode = str(node.get("mode") or MODE_GENERATE)
            if mode in (MODE_GENERATE, MODE_EDIT):
                self.mode_combo.setCurrentText(mode)
        finally:
            self._loading = False

    def save_defaults(self):
        data = _read_config_image()
        node = data.get(CONFIG_NODE) if isinstance(data.get(CONFIG_NODE), dict) else {}
        site = self.current_site()
        sites = node.get("sites") if isinstance(node.get("sites"), dict) else {}
        sites = dict(sites)
        sites[site] = {
            "model": self.model_combo.currentText().strip() or "gpt-image-2",
            "size": self.size_combo.currentData() or DEFAULT_SIZE_BY_SITE.get(site, GPT_IMAGE2_SIZE_SQUARE),
            "quality": self.quality_combo.currentData() or "high",
            "output_format": self.output_format_combo.currentText().strip() or "png",
            "n": int(self.n_spin.value()),
        }
        node["sites"] = sites
        node["site"] = site
        node["mode"] = self.mode_combo.currentText()
        data[CONFIG_NODE] = node
        try:
            with open(CONFIG_IMAGE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as exc:  # noqa: BLE001 - 记默认值失败不该阻断生图
            self._append_log(f"保存本 Tab 默认值失败(忽略): {exc}")

    # ---------------- 图片 ----------------
    def add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择参考图", "", IMAGE_FILTER)
        self._add_paths(paths or [])

    def _add_paths(self, paths):
        for path in paths:
            if not path or not os.path.isfile(path):
                continue
            if path in self.image_paths:
                continue
            if len(self.image_paths) >= GPT_IMAGE2_MAX_REFERENCE_IMAGES:
                QMessageBox.information(
                    self, "提示", f"最多 {GPT_IMAGE2_MAX_REFERENCE_IMAGES} 张参考图，多余的已忽略。"
                )
                break
            self.image_paths.append(path)
            self.image_list.addItem(QListWidgetItem(path))

    def clear_images(self):
        self.image_paths = []
        self.image_list.clear()

    def dragEnterEvent(self, event):  # noqa: N802 - Qt 接口
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802 - Qt 接口
        paths = []
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local:
                paths.append(local)
        self._add_paths(paths)
        event.acceptProposedAction()

    # ---------------- 生成 ----------------
    def build_request(self):
        """按当前站点组装 (后端函数, 调用参数)，供 Worker 调用（也便于测试）。"""
        site = self.current_site()
        prompt = self.prompt_edit.toPlainText().strip()
        model = self.model_combo.currentText().strip() or "gpt-image-2"
        size = self.size_combo.currentData() or DEFAULT_SIZE_BY_SITE.get(site, GPT_IMAGE2_SIZE_SQUARE)
        quality = self.quality_combo.currentData() or "high"
        output_format = self.output_format_combo.currentText().strip() or "png"
        mode = "edit" if self.mode_combo.currentText() == MODE_EDIT else "generate"

        if site == SITE_AUTODL:
            params = {
                "prompt": prompt,
                "image_paths": list(self.image_paths),
                "model": model,
                "aspect_ratio": "1:1",       # 尺寸由 size 直控
                "instructions": "",
                "api_type": API_TYPE_BY_SITE[site],
                "save_sub_dir": SAVE_SUB_DIR_BY_SITE[site],
                "file_prefix": FILE_PREFIX_BY_SITE[site],
                "return_metadata": False,
                "size": size,
                "quality": quality,
                "output_format": output_format,
            }
            return generate_image_openai_image, params

        params = {
            "prompt": prompt,
            "image_paths": list(self.image_paths),
            "model": model,
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "n": int(self.n_spin.value()),
            "mode": mode,
            "api_type": API_TYPE_BY_SITE[site],
            "save_sub_dir": SAVE_SUB_DIR_BY_SITE[site],
            "file_prefix": FILE_PREFIX_BY_SITE[site],
            "return_metadata": False,
        }
        return generate_image_aigc2d_gpt, params

    def generate(self):
        site = self.current_site()
        mode = self.mode_combo.currentText()
        if not self.prompt_edit.toPlainText().strip():
            QMessageBox.warning(self, "提示", "请输入提示词")
            return
        if mode == MODE_EDIT and not self.image_paths:
            QMessageBox.warning(self, "提示", "编辑模式至少需要 1 张图片")
            return
        api_type = API_TYPE_BY_SITE.get(site, "")
        if not str(get_api_config(api_type=api_type).get("api_key") or "").strip():
            QMessageBox.warning(
                self,
                "提示",
                f"conf/config-image.json 的 apis.{api_type} 缺少 api_key，请先在「设置 → 图片生成 API」里填写。",
            )
            return

        self.save_defaults()
        backend, params = self.build_request()

        self.log_view.clear()
        self._append_log(
            f"开始: 站点={site} 模式={mode} 尺寸={params.get('size')} 画质={params.get('quality')} "
            f"格式={params.get('output_format')} 张数={params.get('n', 1)} 参考图={len(params.get('image_paths') or [])}"
        )
        if site == SITE_AUTODL:
            self._append_log("autodl 通道：一次一张，且无法中途取消（请求返回后才结束）。")

        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(site == SITE_AIGC2D)
        self.status_label.setText("生成中...(high 画质可能要几分钟)")
        self.preview_label.setText("请求中...")

        self._worker = GptImage2Worker(backend, params, self)
        self._worker.log.connect(self._append_log)
        self._worker.done.connect(self.on_done)
        self._worker.error.connect(self.on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def cancel_generation(self):
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self.status_label.setText("已请求取消，等待当前请求返回...")

    def _on_worker_finished(self):
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()

    def on_done(self, paths):
        paths = [p for p in (paths or []) if p]
        if not paths:
            self.status_label.setText("完成(无图片, 看日志确认原因)")
            return
        self.status_label.setText(f"完成, {len(paths)} 张")
        for path in paths:
            item = QListWidgetItem(f"{os.path.basename(path)}  ->  {path}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.result_list.addItem(item)
        self.show_preview(paths[0])

    def on_error(self, message):
        self.status_label.setText("出错")
        self._append_log(f"错误: {message}")
        QMessageBox.critical(self, "生成错误", str(message)[:600])

    def show_preview(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.preview_label.setText("无法预览")
            return
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def open_item(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and os.path.exists(path):
            os.startfile(os.path.dirname(path))  # noqa: S606 - 打开所在目录

    def _append_log(self, message):
        self.log_view.appendPlainText(str(message))

    # ---------------- 生命周期 ----------------
    def showEvent(self, event):  # noqa: N802 - Qt 接口
        super().showEvent(event)
        self.refresh_api_hint()
