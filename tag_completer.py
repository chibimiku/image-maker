import os
import json
import csv
from PyQt6.QtCore import Qt, QStringListModel, QObject, QEvent, QAbstractListModel, QModelIndex, QSize
from PyQt6.QtWidgets import QCompleter, QLineEdit, QTextEdit, QStyledItemDelegate, QStyleOptionViewItem, QApplication
from PyQt6.QtGui import QTextCursor, QColor, QPainter, QFontMetrics

# Danbooru tag types: 0=General, 1=Artist, 3=Copyright, 4=Character, 5=Meta
TAG_COLORS = {
    0: QColor("#000000"), # General (Black)
    1: QColor("#cc0000"), # Artist (Red)
    3: QColor("#c000c0"), # Copyright (Purple)
    4: QColor("#00aa00"), # Character (Green)
    5: QColor("#ff8800"), # Meta (Orange)
}

class TagItemDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        painter.save()
        
        # Get data from model
        tag_name = index.data(Qt.ItemDataRole.DisplayRole)
        tag_type = index.data(Qt.ItemDataRole.UserRole)
        post_count = index.data(Qt.ItemDataRole.UserRole + 1)
        
        # Draw background
        if option.state & QStyleOptionViewItem.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
            text_color = option.palette.highlightedText().color()
            count_color = option.palette.highlightedText().color()
        else:
            painter.fillRect(option.rect, option.palette.base())
            text_color = TAG_COLORS.get(tag_type, QColor("#000000"))
            count_color = QColor("#888888") # Gray for post count
            
        # Draw tag name
        painter.setPen(text_color)
        font = option.font
        painter.setFont(font)
        
        rect = option.rect
        padding = 4
        
        # Draw tag name on the left
        painter.drawText(rect.adjusted(padding, 0, -padding, 0), 
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, 
                         tag_name)
                         
        # Draw post count on the right
        if post_count is not None:
            painter.setPen(count_color)
            count_str = f"{post_count:,}"
            painter.drawText(rect.adjusted(padding, 0, -padding, 0), 
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, 
                             count_str)
                             
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        size = super().sizeHint(option, index)
        return QSize(size.width(), size.height() + 4) # Add a little padding

class TagListModel(QAbstractListModel):
    def __init__(self, tags_data, max_results=50, parent=None):
        super().__init__(parent)
        self.all_tags_data = tags_data # List of dicts: {'name': str, 'type': int, 'count': int}
        self.filtered_tags = []
        self.max_results = max_results

    def set_prefix(self, prefix):
        prefix = prefix.lower()
        if not prefix:
            self.beginResetModel()
            self.filtered_tags = []
            self.endResetModel()
            return

        exact_matches = []
        starts_matches = []
        contains_matches = []

        for item in self.all_tags_data:
            name = item['name'].lower()
            if name == prefix:
                if not exact_matches:
                    exact_matches.append(item)
            elif name.startswith(prefix):
                if len(starts_matches) < self.max_results:
                    starts_matches.append(item)
            elif prefix in name:
                if len(contains_matches) < self.max_results:
                    contains_matches.append(item)
                    
            if len(exact_matches) > 0 and len(starts_matches) >= self.max_results and len(contains_matches) >= self.max_results:
                break

        self.beginResetModel()
        combined = exact_matches + starts_matches + contains_matches
        self.filtered_tags = combined[:self.max_results]
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.filtered_tags)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.filtered_tags)):
            return None
            
        item = self.filtered_tags[index.row()]
        
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            return item['name']
        elif role == Qt.ItemDataRole.UserRole:
            return item['type']
        elif role == Qt.ItemDataRole.UserRole + 1:
            return item['count']
            
        return None

class MultiWordCompleter(QCompleter):
    def __init__(self, tags_data, max_results=50, parent=None):
        super().__init__(parent)
        self.model = TagListModel(tags_data, max_results, self)
        self.setModel(self.model)
        self.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setCompletionMode(QCompleter.CompletionMode.UnfilteredPopupCompletion)
        self.setMaxVisibleItems(15)
        
        # Set custom delegate for rendering
        self.popup().setItemDelegate(TagItemDelegate(self.popup()))

    def update_prefix(self, prefix):
        self.model.set_prefix(prefix)
        self.setCompletionPrefix(prefix)

    def pathFromIndex(self, index):
        completion = super().pathFromIndex(index)
        widget = self.widget()
        if isinstance(widget, QLineEdit):
            text = widget.text()
            cursor_pos = widget.cursorPosition()
            text_before_cursor = text[:cursor_pos]
            text_after_cursor = text[cursor_pos:]
            last_comma_idx = text_before_cursor.rfind(',')
            if last_comma_idx != -1:
                return text_before_cursor[:last_comma_idx+1] + " " + completion + ", " + text_after_cursor
            else:
                return completion + ", " + text_after_cursor
        return completion

class TextEditCompleter(QObject):
    def __init__(self, text_edit: QTextEdit, completer: MultiWordCompleter):
        super().__init__(text_edit)
        self.text_edit = text_edit
        self.completer = completer
        self.completer.setWidget(self.text_edit)
        self.completer.activated[str].connect(self.insert_completion)
        self.text_edit.textChanged.connect(self.on_text_changed)
        self.min_chars = 2

    def insert_completion(self, completion):
        self.text_edit.blockSignals(True)
        tc = self.text_edit.textCursor()
        text = self.text_edit.toPlainText()
        cursor_pos = tc.position()
        
        text_before_cursor = text[:cursor_pos]
        last_comma_idx = text_before_cursor.rfind(',')
        
        if last_comma_idx != -1:
            prefix_len = len(text_before_cursor) - last_comma_idx - 1
        else:
            prefix_len = len(text_before_cursor)
            
        # Remove the prefix
        tc.movePosition(QTextCursor.MoveOperation.Left, QTextCursor.MoveMode.KeepAnchor, prefix_len)
        tc.removeSelectedText()
        
        # Insert the completion
        if last_comma_idx != -1:
            tc.insertText(" " + completion + ", ")
        else:
            tc.insertText(completion + ", ")
        self.text_edit.setTextCursor(tc)
        self.text_edit.blockSignals(False)

    def on_text_changed(self):
        tc = self.text_edit.textCursor()
        text = self.text_edit.toPlainText()
        cursor_pos = tc.position()
        
        text_before_cursor = text[:cursor_pos]
        last_comma_idx = text_before_cursor.rfind(',')
        
        if last_comma_idx != -1:
            prefix = text_before_cursor[last_comma_idx + 1:].lstrip()
        else:
            prefix = text_before_cursor.lstrip()
            
        if len(prefix) < getattr(self, 'min_chars', 2):
            self.completer.popup().hide()
            return
            
        self.completer.update_prefix(prefix)
        
        if self.completer.model.rowCount() > 0:
            self.completer.popup().setCurrentIndex(self.completer.completionModel().index(0, 0))
            cr = self.text_edit.cursorRect()
            cr.setWidth(self.completer.popup().sizeHintForColumn(0)
                        + self.completer.popup().verticalScrollBar().sizeHint().width())
            self.completer.complete(cr)
        else:
            self.completer.popup().hide()

class LineEditCompleter(QObject):
    def __init__(self, line_edit: QLineEdit, completer: MultiWordCompleter):
        super().__init__(line_edit)
        self.line_edit = line_edit
        self.completer = completer
        self.completer.setWidget(self.line_edit)
        self.completer.activated[str].connect(self.insert_completion)
        self.line_edit.textChanged.connect(self.on_text_changed)
        self.min_chars = 2
        self._last_text = ""
        self._last_cursor_pos = 0

    def insert_completion(self, completion):
        # QCompleter already updated the text via pathFromIndex
        # completion is the full text returned by pathFromIndex
        text = self._last_text
        cursor_pos = self._last_cursor_pos
        text_before_cursor = text[:cursor_pos]
        last_comma_idx = text_before_cursor.rfind(',')
        
        # Find the position of the comma we just inserted
        # The new text before the cursor was: text_before_cursor[:last_comma_idx+1] + " " + tag + ", "
        # So we can just search for the first comma after last_comma_idx in the new text
        search_start = last_comma_idx + 1 if last_comma_idx != -1 else 0
        new_comma_idx = completion.find(',', search_start)
        if new_comma_idx != -1:
            self.line_edit.setCursorPosition(new_comma_idx + 2)
        else:
            self.line_edit.setCursorPosition(len(completion))

    def on_text_changed(self, text):
        self._last_text = text
        self._last_cursor_pos = self.line_edit.cursorPosition()
        cursor_pos = self.line_edit.cursorPosition()
        text_before_cursor = text[:cursor_pos]
        last_comma_idx = text_before_cursor.rfind(',')
        
        if last_comma_idx != -1:
            prefix = text_before_cursor[last_comma_idx + 1:].lstrip()
        else:
            prefix = text_before_cursor.lstrip()
            
        if len(prefix) < getattr(self, 'min_chars', 2):
            self.completer.popup().hide()
            return
            
        self.completer.update_prefix(prefix)
        
        if self.completer.model.rowCount() > 0:
            self.completer.popup().setCurrentIndex(self.completer.completionModel().index(0, 0))
            cr = self.line_edit.cursorRect()
            self.completer.complete(cr)
        else:
            self.completer.popup().hide()

class TagAutocompleteManager:
    def __init__(self, config_path="config-autocomplete.json"):
        self.config_path = config_path
        self.tags = []
        self.load_config()
        self.load_tags()

    def load_config(self):
        self.config = {
            "enable_autocomplete": True,
            "csv_path": "data/tags/danbooru.csv",
            "max_results": 50,
            "min_chars": 2
        }
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config.update(json.load(f))
            except Exception as e:
                print(f"Failed to load autocomplete config: {e}")

    def load_tags(self):
        if not self.config.get("enable_autocomplete", False):
            return

        csv_path = self.config.get("csv_path", "")
        if not os.path.exists(csv_path):
            print(f"Tag CSV file not found at {csv_path}")
            return

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or len(row) < 3:
                        continue
                    tag = row[0].strip()
                    if tag.lower() in ['name', 'tag', 'id']:
                        continue
                    if tag:
                        try:
                            tag_type = int(row[1].strip())
                            post_count = int(row[2].strip())
                        except ValueError:
                            tag_type = 0
                            post_count = 0
                            
                        self.tags.append({
                            'name': tag,
                            'type': tag_type,
                            'count': post_count
                        })
            
            # Sort tags by post count descending
            self.tags.sort(key=lambda x: x['count'], reverse=True)
            print(f"Loaded {len(self.tags)} tags for autocomplete.")
        except Exception as e:
            print(f"Failed to load tags from CSV: {e}")

    def setup_text_edit(self, text_edit: QTextEdit):
        if not self.tags or not self.config.get("enable_autocomplete", False):
            return None
        max_results = self.config.get("max_results", 50)
        completer = MultiWordCompleter(self.tags, max_results, text_edit)
        completer.setMaxVisibleItems(15)
        tc = TextEditCompleter(text_edit, completer)
        tc.min_chars = self.config.get("min_chars", 2)
        return tc

    def setup_line_edit(self, line_edit: QLineEdit):
        if not self.tags or not self.config.get("enable_autocomplete", False):
            return None
        max_results = self.config.get("max_results", 50)
        completer = MultiWordCompleter(self.tags, max_results, line_edit)
        completer.setMaxVisibleItems(15)
        lc = LineEditCompleter(line_edit, completer)
        lc.min_chars = self.config.get("min_chars", 2)
        return lc
