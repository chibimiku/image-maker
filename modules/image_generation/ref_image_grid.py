"""参考图缩略图网格控件（gpt-image-2 生图/编辑台使用，可被其它 Tab 复用）。

取代早期「QListWidget 里一行一个裸路径」的参考图列表：

- 缩略图卡片 + 序号：列表顺序就是提交给 `/v1/images/edits` 的 `image[]` 顺序
- 每张卡片右上角「×」删除按钮；右键菜单可预览 / 打开所在目录 / 移除
- 卡片之间可拖拽重排，落点卡片高亮提示
- 支持从资源管理器拖入图片或目录（目录会展开成其中的图片）
- 单击卡片预览，双击打开所在目录；完整路径与文件大小在悬浮提示里
"""
import os

from PyQt6.QtCore import QMimeData, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QDrag, QImageReader, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

# 拖拽重排时卡片之间传递的 mime 类型（值是源卡片下标）
CARD_MIME = "application/x-image-maker-ref-card"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
DEFAULT_THUMB_SIZE = 96

_STYLE_NORMAL = (
    "#RefImageCard { border: 1px solid #c8c8c8; border-radius: 6px; background: #fafafa; }"
)
_STYLE_DROP = (
    "#RefImageCard { border: 2px solid #2d8cf0; border-radius: 6px; background: #eaf4ff; }"
)
_DELETE_BTN_STYLE = (
    "QPushButton { border: none; border-radius: 9px; background: rgba(0, 0, 0, 150);"
    " color: white; font-weight: bold; font-size: 12px; padding: 0px; }"
    "QPushButton:hover { background: #d9534f; }"
)


def is_supported_image(path) -> bool:
    return str(path or "").lower().endswith(IMAGE_EXTENSIONS)


def collect_image_paths(mime_data) -> list:
    """从拖拽数据里取出图片路径（目录会展开成其中的图片），去重并保持原顺序。"""
    if mime_data is None or not mime_data.hasUrls():
        return []
    collected = []
    for url in mime_data.urls():
        if not url.isLocalFile():
            continue
        local = url.toLocalFile()
        if os.path.isdir(local):
            for root, _dirs, files in os.walk(local):
                for name in sorted(files):
                    candidate = os.path.join(root, name)
                    if is_supported_image(candidate):
                        collected.append(candidate)
        elif os.path.isfile(local) and is_supported_image(local):
            collected.append(local)
    return list(dict.fromkeys(os.path.normpath(item) for item in collected))


def load_thumbnail(path, size):
    """按目标尺寸直接解码缩略图，避免把 4K 原图整张读进内存。"""
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    original = reader.size()
    if original.isValid() and (original.width() > size or original.height() > size):
        reader.setScaledSize(original.scaled(QSize(size, size), Qt.AspectRatioMode.KeepAspectRatio))
    image = reader.read()
    if image.isNull():
        return QPixmap()
    return QPixmap.fromImage(image)


class RefImageCard(QFrame):
    """单张参考图的缩略图卡片（可拖拽重排 / 删除 / 预览）。"""

    delete_requested = pyqtSignal(object)
    activated = pyqtSignal(object)          # 单击
    opened = pyqtSignal(object)             # 双击或右键「打开所在目录」
    dropped = pyqtSignal(int, object)       # (源卡片下标, 本卡片)

    def __init__(self, path, index, thumb_size=DEFAULT_THUMB_SIZE, parent=None):
        super().__init__(parent)
        self.path = str(path)
        self.index = int(index)
        self.thumb_size = int(thumb_size)
        self._press_pos = None
        self._highlight = False

        self.setObjectName("RefImageCard")
        self.setStyleSheet(_STYLE_NORMAL)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        grid = QGridLayout(self)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(2)

        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(self.thumb_size, self.thumb_size)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = load_thumbnail(self.path, self.thumb_size)
        if pixmap.isNull():
            self.thumb_label.setText("无法预览")
        else:
            self.thumb_label.setPixmap(pixmap)
        grid.addWidget(self.thumb_label, 0, 0)

        self.delete_btn = QPushButton("×", self)
        self.delete_btn.setFixedSize(18, 18)
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.setToolTip("移除这张参考图")
        self.delete_btn.setStyleSheet(_DELETE_BTN_STYLE)
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(self))
        grid.addWidget(
            self.delete_btn,
            0,
            0,
            alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
        )

        self.caption_label = QLabel()
        self.caption_label.setFixedWidth(self.thumb_size)
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.caption_label, 1, 0)

        self.set_index(self.index)

    def set_index(self, index):
        self.index = int(index)
        name = os.path.basename(self.path)
        metrics = self.caption_label.fontMetrics()
        self.caption_label.setText(
            metrics.elidedText(
                f"{self.index + 1}. {name}", Qt.TextElideMode.ElideMiddle, self.thumb_size
            )
        )
        self.setToolTip(f"{self.path}\n{self._size_text()}（拖动可与其它参考图交换顺序）")

    def _size_text(self) -> str:
        try:
            return f"{os.path.getsize(self.path) / 1024:.0f} KB"
        except OSError:
            return "文件不存在"

    def set_drop_highlight(self, enabled: bool):
        if bool(enabled) == self._highlight:
            return
        self._highlight = bool(enabled)
        self.setStyleSheet(_STYLE_DROP if self._highlight else _STYLE_NORMAL)

    # ---------------- 拖拽重排 ----------------
    def mousePressEvent(self, event):  # noqa: N802 - Qt 接口
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            self.activated.emit(self)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt 接口
        if self._press_pos is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        moved = (event.position().toPoint() - self._press_pos).manhattanLength()
        if moved < QApplication.startDragDistance():
            return
        self._press_pos = None
        self._start_drag()

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt 接口
        self._press_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802 - Qt 接口
        if event.button() == Qt.MouseButton.LeftButton:
            self.opened.emit(self)

    def _start_drag(self):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(CARD_MIME, str(self.index).encode("utf-8"))
        drag.setMimeData(mime)
        pixmap = self.thumb_label.pixmap()
        if pixmap is not None and not pixmap.isNull():
            drag.setPixmap(pixmap)
            drag.setHotSpot(pixmap.rect().center())
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        drag.exec(Qt.DropAction.MoveAction)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def dragEnterEvent(self, event):  # noqa: N802 - Qt 接口
        if event.mimeData().hasFormat(CARD_MIME):
            self.set_drop_highlight(True)
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802 - Qt 接口
        if event.mimeData().hasFormat(CARD_MIME):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragLeaveEvent(self, event):  # noqa: N802 - Qt 接口
        self.set_drop_highlight(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):  # noqa: N802 - Qt 接口
        self.set_drop_highlight(False)
        payload = bytes(event.mimeData().data(CARD_MIME))
        try:
            source_index = int(payload.decode("utf-8"))
        except (TypeError, ValueError):
            event.ignore()
            return
        self.dropped.emit(source_index, self)
        event.acceptProposedAction()

    # ---------------- 右键菜单 ----------------
    def contextMenuEvent(self, event):  # noqa: N802 - Qt 接口
        menu = QMenu(self)
        preview_action = menu.addAction("预览这张图")
        open_action = menu.addAction("打开所在目录")
        menu.addSeparator()
        remove_action = menu.addAction("移除这张图")
        chosen = menu.exec(event.globalPos())
        if chosen is preview_action:
            self.activated.emit(self)
        elif chosen is open_action:
            self.opened.emit(self)
        elif chosen is remove_action:
            self.delete_requested.emit(self)


class RefImageGrid(QWidget):
    """参考图缩略图网格：本控件持有路径列表，顺序即提交顺序。"""

    images_changed = pyqtSignal(list)
    image_clicked = pyqtSignal(str)
    image_double_clicked = pyqtSignal(str)

    def __init__(self, max_images=16, thumb_size=DEFAULT_THUMB_SIZE, parent=None, compact_when_empty=False):
        """`compact_when_empty=True` 时，没有图片就不占缩略图高度（只留一行占位提示）。

        gpt-image-2 Tab 用它换回约 170px 的垂直空间（那个 Tab 上方控件多，
        否则 Qt 会把可拉伸的提示词编辑框压到最小高度）；其它 Tab 保持原来的固定高度。
        """
        super().__init__(parent)
        self.max_images = max(1, int(max_images))
        self.thumb_size = int(thumb_size)
        self.compact_when_empty = bool(compact_when_empty)
        self._paths = []
        self._cards = []

        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.placeholder = QLabel(
            "把图片拖到这里，或点「添加图片」；\n拖动缩略图可调整顺序，点右上角 × 删除"
        )
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setStyleSheet(
            "QLabel { border: 1px dashed #bbb; border-radius: 6px; color: #888; padding: 8px; }"
        )
        layout.addWidget(self.placeholder)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        self.row_layout = QHBoxLayout(container)
        self.row_layout.setContentsMargins(4, 4, 4, 4)
        self.row_layout.setSpacing(8)
        self.row_layout.addStretch(1)
        self.scroll_area.setWidget(container)
        layout.addWidget(self.scroll_area)

        self._sync_placeholder()
        self._apply_height_for_content()

    # ---------------- 查询 ----------------
    def _apply_height_for_content(self):
        """按是否有缩略图调整高度：compact 模式下空列表只占一行占位提示的高度。"""
        if self.compact_when_empty and not self._paths:
            self.setMinimumHeight(0)
            self.setMaximumHeight(self.placeholder.sizeHint().height() + 16)
        else:
            self.setMinimumHeight(self.thumb_size + 76)
            self.setMaximumHeight(self.thumb_size + 104)

    def paths(self) -> list:
        return list(self._paths)

    def count(self) -> int:
        return len(self._paths)

    def cards(self) -> list:
        """当前卡片列表（下标与路径列表一一对应），供测试与外部定位使用。"""
        return list(self._cards)

    def card_at(self, index):
        if 0 <= int(index) < len(self._cards):
            return self._cards[int(index)]
        return None

    def empty(self) -> bool:
        return not self._paths

    # ---------------- 增删改 ----------------
    def add_paths(self, paths):
        """追加图片，返回 (新增数量, 因超过上限被忽略的数量)。"""
        added = 0
        ignored = 0
        seen = {os.path.normcase(os.path.abspath(item)) for item in self._paths}
        for raw in paths or []:
            if not raw:
                continue
            path = os.path.normpath(str(raw))
            if not os.path.isfile(path):
                continue
            key = os.path.normcase(os.path.abspath(path))
            if key in seen:
                continue
            if len(self._paths) >= self.max_images:
                ignored += 1
                continue
            seen.add(key)
            self._paths.append(path)
            added += 1
        if added:
            self._rebuild()
            self.images_changed.emit(self.paths())
        return added, ignored

    def set_paths(self, paths):
        self._paths = []
        added, _ignored = self.add_paths(paths)
        if not added:
            self._rebuild()
            self.images_changed.emit([])

    def remove_index(self, index):
        if 0 <= int(index) < len(self._paths):
            del self._paths[int(index)]
            self._rebuild()
            self.images_changed.emit(self.paths())

    def remove_path(self, path):
        target = os.path.normpath(str(path))
        for index, item in enumerate(self._paths):
            if os.path.normcase(os.path.abspath(item)) == os.path.normcase(os.path.abspath(target)):
                self.remove_index(index)
                return

    def clear(self):
        changed = bool(self._paths)
        self._paths = []
        self._rebuild()
        if changed:
            self.images_changed.emit([])

    def move_image(self, source_index, target_index):
        """把 source_index 的图移动到 target_index 的位置（拖到某张图上=占用它的位置）。"""
        total = len(self._paths)
        if not (0 <= int(source_index) < total):
            return
        source_index = int(source_index)
        target_index = max(0, min(int(target_index), total - 1))
        if source_index == target_index:
            return
        path = self._paths.pop(source_index)
        self._paths.insert(target_index, path)
        self._rebuild()
        self.images_changed.emit(self.paths())

    # ---------------- 视图重建 ----------------
    def _rebuild(self):
        for card in self._cards:
            self.row_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self._cards = []
        for index, path in enumerate(self._paths):
            card = RefImageCard(path, index, self.thumb_size, self)
            card.delete_requested.connect(self._on_card_delete_requested)
            card.activated.connect(lambda _card, item=path: self.image_clicked.emit(item))
            card.opened.connect(lambda _card, item=path: self.image_double_clicked.emit(item))
            card.dropped.connect(self._on_card_dropped)
            self.row_layout.insertWidget(self.row_layout.count() - 1, card)
            self._cards.append(card)
        self._sync_placeholder()
        self._apply_height_for_content()

    def _sync_placeholder(self):
        has_images = bool(self._paths)
        self.scroll_area.setVisible(has_images)
        self.placeholder.setVisible(not has_images)

    def _index_of_card(self, card) -> int:
        try:
            return self._cards.index(card)
        except ValueError:
            return -1

    def _on_card_delete_requested(self, card):
        self.remove_index(self._index_of_card(card))

    def _on_card_dropped(self, source_index, target_card):
        target_index = self._index_of_card(target_card)
        if target_index < 0:
            return
        self.move_image(source_index, target_index)

    # ---------------- 外部拖入 ----------------
    def dragEnterEvent(self, event):  # noqa: N802 - Qt 接口
        mime = event.mimeData()
        if mime.hasFormat(CARD_MIME) or collect_image_paths(mime):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802 - Qt 接口
        self.dragEnterEvent(event)

    def dropEvent(self, event):  # noqa: N802 - Qt 接口
        mime = event.mimeData()
        if mime.hasFormat(CARD_MIME):
            try:
                source_index = int(bytes(mime.data(CARD_MIME)).decode("utf-8"))
            except (TypeError, ValueError):
                event.ignore()
                return
            self.move_image(source_index, len(self._paths) - 1)  # 拖到空白处 = 排到最后
            event.acceptProposedAction()
            return
        paths = collect_image_paths(mime)
        if not paths:
            event.ignore()
            return
        self.add_paths(paths)
        event.acceptProposedAction()
