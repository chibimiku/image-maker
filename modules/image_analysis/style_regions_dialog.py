"""Inspectable automatic region proposals with direct manual curve editing."""
import copy
import json
from pathlib import Path
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QImage
from PyQt6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
                            QPushButton, QLabel, QTextEdit, QCheckBox, QDialogButtonBox)
from PIL import Image, ImageOps
from utils.style_face_metrics import digest, read_annotation
from utils.style_regions import propose_regions, save_annotation

TOOLS = {"面部轮廓": "face_outline", "头发内部区域": "hair_regions",
         "深色发丝路径": "hair_dark", "亮色发丝路径": "hair_light",
         "左眼上眼睑": "viewer_left.upper_lid", "左眼下眼睑": "viewer_left.lower_lid",
         "右眼上眼睑": "viewer_right.upper_lid", "右眼下眼睑": "viewer_right.lower_lid",
         "左眼虹膜": "viewer_left.iris", "右眼虹膜": "viewer_right.iris",
         "左眼睫毛": "viewer_left.lashes", "右眼睫毛": "viewer_right.lashes",
         "鼻尖中心": "nose", "嘴部中心": "mouth"}


class ProposalWorker(QThread):
    completed = pyqtSignal(object, str)
    def __init__(self, path, config, parent=None):
        super().__init__(parent)
        self.path, self.config = path, config
    def run(self):
        try:
            self.completed.emit(propose_regions(self.path, self.config), "")
        except Exception as exc:
            self.completed.emit({}, str(exc))


class RegionCanvas(QWidget):
    def __init__(self, path, dialog):
        super().__init__(dialog)
        self.dialog = dialog
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            raw = image.tobytes()
            self.image = QImage(raw, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888).copy()
        self.setMinimumSize(450, 400)
        self.pending = []
        self.zoom = 1.0
        self.pan = [0.0, 0.0]
        self.pan_start = None
    def rectangle(self):
        scale = min(self.width() / self.image.width(), self.height() / self.image.height()) * self.zoom
        w, h = self.image.width() * scale, self.image.height() * scale
        return (self.width() - w) / 2 + self.pan[0], (self.height() - h) / 2 + self.pan[1], w, h
    def paintEvent(self, event):
        from PyQt6.QtCore import QRectF
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        x, y, w, h = self.rectangle()
        painter.drawImage(QRectF(x, y, w, h), self.image)
        try:
            shapes = self.dialog.shapes() if self.dialog.show_annotations.isChecked() else []
        except (ValueError, KeyError, TypeError, AttributeError):
            shapes = []
        for index, (label, vertices) in enumerate(shapes + [("正在标注", self.pending)]):
            painter.setPen(QPen(QColor.fromHsv((index * 47) % 360, 255, 230), 2))
            from utils.style_face_metrics import points
            try:
                vertices = points(vertices, 1).tolist()
            except (ValueError, TypeError):
                continue
            converted = [QPointF(x + p[0] * w, y + p[1] * h) for p in vertices]
            for first, second in zip(converted[:-1], converted[1:]):
                painter.drawLine(first, second)
            for point in converted:
                painter.drawEllipse(point, 2, 2)
            selected = self.dialog.tool.currentData()
            show_label = label == "正在标注" or (selected == "face_outline" and label == "face") or (selected == "hair_regions" and label.startswith("hair ")) or (selected.startswith("hair_") and selected != "hair_regions" and label.startswith("strand ")) or ("." in selected and label == selected)
            if converted and show_label:
                painter.drawText(converted[0], label)
    def mousePressEvent(self, event):
        x, y, w, h = self.rectangle()
        point = event.position()
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_start = point
            return
        if event.button() == Qt.MouseButton.LeftButton and x <= point.x() <= x + w and y <= point.y() <= y + h:
            self.pending.append([round((point.x() - x) / w, 6), round((point.y() - y) / h, 6)])
            self.update()

    def wheelEvent(self, event):
        point = event.position()
        x, y, w, h = self.rectangle()
        u, v = (point.x() - x) / w, (point.y() - y) / h
        self.zoom = min(16.0, max(.5, self.zoom * (1.2 if event.angleDelta().y() > 0 else 1/1.2)))
        nx, ny, nw, nh = self.rectangle()
        self.pan[0] += point.x() - u * nw - nx
        self.pan[1] += point.y() - v * nh - ny
        self.update()
    def mouseMoveEvent(self, event):
        if self.pan_start is not None:
            delta = event.position() - self.pan_start
            self.pan[0] += delta.x()
            self.pan[1] += delta.y()
            self.pan_start = event.position()
            self.update()
    def mouseReleaseEvent(self, event):
        self.pan_start = None
    def fit(self):
        self.zoom, self.pan = 1.0, [0.0, 0.0]
        self.update()


class StyleRegionsDialog(QDialog):
    def __init__(self, path, config_getter=None, parent=None):
        super().__init__(parent)
        self.path, self.config_getter, self.worker = str(path), config_getter, None
        self.edit_buttons = []
        self.setWindowTitle("面部与发丝定位：自动候选 / 人工校正")
        self.resize(1040, 720)
        try:
            self.data = read_annotation(path) or {}
        except Exception:
            self.data = {}
        if not self.data:
            self.data = {"version": "style-regions/1", "image_sha256": digest(path), "confirmed": False,
                         "pose": "unknown", "eyes": {n: {"state": "unknown"} for n in ("viewer_left", "viewer_right")},
                         "face_outline": [], "hair_regions": [], "hair_strands": []}
        layout = QVBoxLayout(self)
        self.notice = QLabel("左右按看图者方向；滚轮放大、中键平移；沿可见轮廓逐点点击，完成路径。发丝只标头发内部的实际细线，至少3条。遮挡 / 闭眼不要补画。")
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        row = QHBoxLayout()
        self.auto_button = QPushButton("自动定位候选（文本模型调用）")
        self.auto_button.setEnabled(config_getter is not None)
        self.auto_button.clicked.connect(self.propose)
        row.addWidget(self.auto_button)
        self.tool = QComboBox()
        for label, key in TOOLS.items():
            self.tool.addItem(label, key)
        row.addWidget(self.tool)
        self.tool.currentIndexChanged.connect(self.reset_pending)
        button = QPushButton("完成路径")
        button.clicked.connect(self.commit)
        row.addWidget(button)
        self.edit_buttons.append(button)
        button = QPushButton("撤销点")
        button.clicked.connect(self.undo)
        row.addWidget(button)
        self.edit_buttons.append(button)
        button = QPushButton("清除此项")
        button.clicked.connect(self.clear_tool)
        row.addWidget(button)
        self.edit_buttons.append(button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.pose = QComboBox()
        for label, key in (("视角未知", "unknown"), ("正面", "frontal"), ("四分之三侧面", "three_quarter"), ("侧面", "profile")):
            self.pose.addItem(label, key)
        self.pose.setCurrentIndex(self.pose.findData(self.data.get("pose", "unknown")))
        self.pose.currentIndexChanged.connect(self.pose_changed)
        row.addWidget(self.pose)
        self.show_annotations = QCheckBox("叠加定位")
        self.show_annotations.setChecked(True)
        self.show_annotations.toggled.connect(lambda checked: self.canvas.update() if hasattr(self, "canvas") else None)
        row.addWidget(self.show_annotations)
        self.states = {}
        for name, label in (("viewer_left", "左眼"), ("viewer_right", "右眼")):
            row.addWidget(QLabel(label))
            combo = QComboBox()
            for text, key in (("未知", "unknown"), ("张开", "open"), ("闭合", "closed"), ("遮挡", "occluded")):
                combo.addItem(text, key)
            combo.setCurrentIndex(combo.findData(self.data.get("eyes", {}).get(name, {}).get("state", "unknown")))
            combo.currentIndexChanged.connect(lambda index, n=name: self.state_changed(n))
            row.addWidget(combo)
            self.states[name] = combo
        layout.addLayout(row)
        area = QHBoxLayout()
        self.canvas = RegionCanvas(path, self)
        area.addWidget(self.canvas, 2)
        self.json_editor = QTextEdit()
        self.json_editor.setToolTip("可直接修改 JSON；改动后需要重新确认定位。图中曲线以当前有效 JSON 为准。")
        self.json_editor.textChanged.connect(self.json_changed)
        area.addWidget(self.json_editor, 1)
        layout.addLayout(area, 1)
        self.confirm = QCheckBox("我已检查目标角色、眼睑与发丝定位（正式测量）")
        self.confirm.setChecked(self.data.get("confirmed") is True)
        layout.addWidget(self.confirm)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存定位")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("关闭")
        self.buttons.accepted.connect(self.save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.refresh()

    def refresh(self):
        self.json_editor.blockSignals(True)
        self.json_editor.setPlainText(json.dumps(self.data, ensure_ascii=False, indent=2))
        self.json_editor.blockSignals(False)
        self.canvas.update()

    def json_changed(self):
        self.confirm.setChecked(False)
        try:
            value = json.loads(self.json_editor.toPlainText())
            if isinstance(value, dict):
                self.data = value
                self.canvas.update()
        except (ValueError, TypeError):
            pass

    def shapes(self):
        result = []
        if self.data.get("face_outline"):
            result.append(("face", self.data["face_outline"]))
        for i, polygon in enumerate(self.data.get("hair_regions", [])):
            result.append((f"hair {i}", polygon))
        for i, strand in enumerate(self.data.get("hair_strands", [])):
            result.append((f"strand {i}", strand["points"]))
        for name, eye in self.data.get("eyes", {}).items():
            for key in ("upper_lid", "lower_lid", "iris"):
                if eye.get(key):
                    result.append((name + "." + key, eye[key]))
            for i, points in enumerate(eye.get("lashes", [])):
                result.append((name + f" lash {i}", points))
        for key in ("nose", "mouth"):
            if self.data.get(key):
                result.append((key, [self.data[key]]))
        return result

    def reset_pending(self):
        if hasattr(self, "canvas"):
            self.canvas.pending = []
            self.canvas.update()

    def undo(self):
        if self.canvas.pending:
            self.canvas.pending.pop()
            self.canvas.update()

    def commit(self):
        key, vertices = self.tool.currentData(), copy.deepcopy(self.canvas.pending)
        minimum = 1 if key in ("nose", "mouth") else 3 if key in ("face_outline", "hair_regions") or key.endswith("iris") else 2
        if len(vertices) < minimum:
            self.notice.setText(f"此项需要至少 {minimum} 个点。")
            return
        if key in ("nose", "mouth"):
            self.data[key] = vertices[-1]
        elif key in ("hair_dark", "hair_light"):
            self.data.setdefault("hair_strands", []).append({"points": vertices, "polarity": key.split("_")[1]})
        elif key == "hair_regions":
            self.data.setdefault(key, []).append(vertices)
        elif "." in key:
            name, feature = key.split(".")
            eye = self.data.setdefault("eyes", {}).setdefault(name, {"state": "unknown"})
            if feature == "lashes":
                eye.setdefault(feature, []).append(vertices)
            else:
                eye[feature] = vertices
        else:
            self.data[key] = vertices
        self.confirm.setChecked(False)
        self.reset_pending()
        self.refresh()

    def clear_tool(self):
        key = self.tool.currentData()
        if key.startswith("hair_") and key != "hair_regions":
            self.data["hair_strands"] = []
        elif "." in key:
            name, feature = key.split(".")
            self.data.setdefault("eyes", {}).setdefault(name, {}).pop(feature, None)
        else:
            self.data.pop(key, None)
        self.confirm.setChecked(False)
        self.reset_pending()
        self.refresh()

    def pose_changed(self):
        self.data["pose"] = self.pose.currentData()
        if hasattr(self, "confirm"):
            self.confirm.setChecked(False)
            self.refresh()

    def state_changed(self, name):
        self.data.setdefault("eyes", {}).setdefault(name, {})["state"] = self.states[name].currentData()
        self.confirm.setChecked(False)
        self.refresh()

    def propose(self):
        if self.worker:
            return
        try:
            config = self.config_getter()
        except Exception as exc:
            self.notice.setText(str(exc))
            return
        self.auto_button.setEnabled(False)
        for control in [self.canvas, self.json_editor, self.confirm, self.pose, self.tool, *self.states.values(), *self.edit_buttons]:
            control.setEnabled(False)
        self.buttons.setEnabled(False)
        self.notice.setText("正在请求候选定位；不会把自动结果标为人工确认。")
        self.worker = ProposalWorker(self.path, config, self)
        self.worker.completed.connect(self.proposed)
        self.worker.finished.connect(self.proposal_stopped)
        self.worker.start()

    def proposed(self, value, error):
        if error:
            self.notice.setText(error)
        else:
            self.data = value
            self.confirm.setChecked(value.get("confirmed") is True)
            self.pose.blockSignals(True)
            self.pose.setCurrentIndex(self.pose.findData(value.get("pose", "unknown")))
            self.pose.blockSignals(False)
            for name, combo in self.states.items():
                combo.blockSignals(True)
                combo.setCurrentIndex(combo.findData(value.get("eyes", {}).get(name, {}).get("state", "unknown")))
                combo.blockSignals(False)
            self.refresh()
            self.notice.setText("候选已保存；查看并校正曲线。未确认结果只作为暂定测量，不计正式均值。")

    def proposal_stopped(self):
        worker = self.worker
        self.worker = None
        self.auto_button.setEnabled(True)
        for control in [self.canvas, self.json_editor, self.confirm, self.pose, self.tool, *self.states.values(), *self.edit_buttons]:
            control.setEnabled(True)
        self.buttons.setEnabled(True)
        if worker:
            worker.deleteLater()

    def save(self):
        try:
            value = json.loads(self.json_editor.toPlainText())
            value.update(version="style-regions/1", image_sha256=digest(self.path), confirmed=self.confirm.isChecked())
            save_annotation(self.path, value)
            self.accept()
        except Exception as exc:
            self.notice.setText(str(exc))

    def reject(self):
        if self.worker:
            self.notice.setText("定位请求尚在进行；完成后可以关闭。")
            return
        super().reject()
