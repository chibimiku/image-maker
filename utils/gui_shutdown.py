"""Keep GUI event delivery alive until outstanding QThreads have finished.

Closing waits for existing work and saves its results; no forced termination or
system-wide process operations. Includes workers kept in Python containers.
"""
from PyQt6.QtCore import QObject, QThread, QTimer


def running_threads(root, retained=()):
    seen, threads = set(), []

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, QThread):
            try:
                if value.isRunning():
                    threads.append(value)
            except RuntimeError:  # Qt may already have deleted a finished worker.
                pass
        elif isinstance(value, QObject):
            try:
                for child in value.children():
                    visit(child)
                for item in vars(value).values():
                    if isinstance(item, (QObject, dict, list, tuple, set)):
                        visit(item)
            except RuntimeError:
                pass
        elif isinstance(value, dict):
            for key, item in list(value.items()):
                visit(key)
                visit(item)
        elif isinstance(value, (list, tuple, set)):
            for item in list(value):
                visit(item)

    visit(root)
    visit(retained)
    return threads


def defer_close_until_threads_finish(widget, event):
    retained = getattr(widget, '_shutdown_threads', ())
    active = running_threads(widget, retained)
    if not active:
        timer = getattr(widget, '_shutdown_timer', None)
        if timer:
            timer.stop()
        return False
    event.ignore()
    widget._shutdown_threads = active  # Preserve parentless workers through shutdown.
    if getattr(widget, '_shutdown_pending', False):
        return True
    widget._shutdown_pending = True
    widget.setEnabled(False)
    widget.setWindowTitle(widget.windowTitle() + '（等待正在运行的任务完成后关闭）')
    timer = QTimer(widget)
    widget._shutdown_timer = timer
    idle_ticks = 0

    def poll():
        nonlocal idle_ticks
        active = running_threads(widget, widget._shutdown_threads)
        widget._shutdown_threads = active
        if active:
            idle_ticks = 0
            return
        # Give queued result/finished callbacks a turn; they can launch a saved
        # next stage. Re-scan before accepting closure.
        idle_ticks += 1
        if idle_ticks >= 2:
            timer.stop()
            widget.close()

    timer.timeout.connect(poll)
    timer.start(100)
    return True
