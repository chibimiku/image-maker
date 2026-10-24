"""Manual crop selection on EXIF-oriented style references; no model calls."""
from PIL import Image, ImageOps
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QRubberBand,
                             QVBoxLayout)

from utils.style_reference import file_hash


class ReferenceCropDialog(QDialog):
    def __init__(self, path, role="face_eyes", parent=None):
        super().__init__(parent)
        self.setWindowTitle("框选画风参考区域（只参考画法）")
        self.source_sha256 = file_hash(path)
        self.role = role
        self.box = None
        with Image.open(path) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            oriented.thumbnail((900, 650))
        w, h = oriented.size
        qimage = QImage(oriented.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("拖出矩形区域，保留足够上下文；裁剪仅作为附加参考，不会贴回成品。"))
        self.canvas = QLabel()
        self.canvas.setFixedSize(w, h)
        self.canvas.setPixmap(QPixmap.fromImage(qimage))
        layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignCenter)
        self.band = QRubberBand(QRubberBand.Shape.Rectangle, self.canvas)
        self.canvas.mousePressEvent = self._press
        self.canvas.mouseMoveEvent = self._move
        self.canvas.mouseReleaseEvent = self._release
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.origin = None

    def _point(self, event):
        point = event.position().toPoint()
        return QPoint(max(0, min(point.x(), self.canvas.width())),
                      max(0, min(point.y(), self.canvas.height())))

    def _press(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.origin = self._point(event)
            self.box = None
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            self.band.setGeometry(QRect(self.origin, self.origin))
            self.band.show()

    def _move(self, event):
        if self.origin is not None:
            self.band.setGeometry(QRect(self.origin, self._point(event)).normalized())

    def _release(self, event):
        if self.origin is None or event.button() != Qt.MouseButton.LeftButton:
            return
        end = self._point(event)
        x0, x1 = sorted((self.origin.x(), end.x()))
        y0, y1 = sorted((self.origin.y(), end.y()))
        self.origin = None
        if x1 - x0 >= 16 and y1 - y0 >= 16:
            self.box = [x0 / self.canvas.width(), y0 / self.canvas.height(),
                        x1 / self.canvas.width(), y1 / self.canvas.height()]
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

    def specification(self):
        if self.box is None:
            raise ValueError("尚未框选参考区域")
        return {"role": self.role, "box": self.box, "source_sha256": self.source_sha256}
