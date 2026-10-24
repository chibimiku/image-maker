"""Non-blocking, undoable prompt optimizer UI shared by two image tabs."""
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QPushButton

from utils.image_prompt_optimizer import optimize_image_prompt


class PromptOptimizeThread(QThread):
    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, brief, context, parent=None):
        super().__init__(parent)
        self.brief, self.context = brief, context

    def run(self):
        try:
            result = optimize_image_prompt(self.brief, self.context, log_callback=self.log_signal.emit)
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(f"提示词优化失败，原文保留：{exc}")


class PromptOptimizerControl(QObject):
    def __init__(self, owner, editor, context_getter, log_callback, supported=lambda: True,
                 generation_busy=lambda: False, state_getter=None):
        super().__init__(owner)
        self.owner, self.editor = owner, editor
        self.context_getter, self.log_callback = context_getter, log_callback
        self.supported, self.generation_busy = supported, generation_busy
        self.state_getter = state_getter or context_getter
        self.worker = None
        self.busy = False
        self.button = QPushButton("优化提示词", owner)
        self.button.setToolTip("调用当前文字分析 API（按配置使用备用端点），优化后回填；Ctrl+Z 撤销。不会自动出图。")
        self.button.clicked.connect(self.start)
        editor.textChanged.connect(self.refresh)

    def refresh(self, *args):
        self.button.setEnabled(not self.busy and self.supported() and not self.generation_busy()
                               and bool(self.editor.toPlainText().strip()))

    def start(self):
        if self.busy or self.generation_busy() or not self.supported():
            return
        self.original = self.editor.toPlainText()
        if not self.original.strip():
            return
        self.context = dict(self.context_getter())
        self.original_state = self.state_getter()
        self.busy = True
        self.button.setText("正在优化…")
        self.owner.generate_btn.setEnabled(False)
        if hasattr(self.owner, "_refresh_task_controls"):
            self.owner._refresh_task_controls()
        self.refresh()
        self.worker = PromptOptimizeThread(self.original, self.context, self)
        self.worker.succeeded.connect(self.apply_result)
        self.worker.failed.connect(self.log_callback)
        self.worker.log_signal.connect(self.log_callback)
        self.worker.finished.connect(self.finished)
        self.worker.start()

    def apply_result(self, text):
        if (self.editor.toPlainText() != self.original or dict(self.context_getter()) != self.context
                or self.state_getter() != self.original_state):
            self.log_callback("优化期间原文或选项已改变，未覆盖当前提示词。")
            return
        cursor = self.editor.textCursor()
        cursor.beginEditBlock()
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.insertText(text)
        cursor.endEditBlock()
        self.editor.setTextCursor(cursor)
        self.log_callback("提示词已优化并回填，可继续编辑；Ctrl+Z 可恢复原文。")

    def finished(self):
        self.busy = False
        self.button.setText("优化提示词")
        worker, self.worker = self.worker, None
        if worker:
            worker.deleteLater()
        self.owner.generate_btn.setEnabled(not self.generation_busy())
        if hasattr(self.owner, "_refresh_task_controls"):
            self.owner._refresh_task_controls()
        self.refresh()
