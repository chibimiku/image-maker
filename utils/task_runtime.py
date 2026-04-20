import datetime
from PyQt5.QtCore import QObject, QTimer
from PyQt5.QtWidgets import QSystemTrayIcon, QStyle


class SystemNotifier:
    def __init__(self, host_widget):
        self._tray = None
        try:
            self._tray = QSystemTrayIcon(host_widget)
            self._tray.setIcon(host_widget.style().standardIcon(QStyle.SP_ComputerIcon))
            self._tray.setVisible(True)
        except Exception:
            self._tray = None

    def notify(self, title, message, timeout_ms=5000):
        try:
            if self._tray and QSystemTrayIcon.isSystemTrayAvailable():
                self._tray.showMessage(str(title), str(message), QSystemTrayIcon.Information, int(timeout_ms))
        except Exception:
            pass


class TaskCountdown(QObject):
    def __init__(self, parent=None, on_tick=None, on_timeout=None, interval_ms=1000):
        super().__init__(parent)
        self._on_tick = on_tick
        self._on_timeout = on_timeout
        self._deadline = None
        self._timer = QTimer(self)
        self._timer.setInterval(max(100, int(interval_ms)))
        self._timer.timeout.connect(self._tick)

    def start(self, timeout_seconds):
        seconds = max(1, int(timeout_seconds))
        self._deadline = datetime.datetime.now() + datetime.timedelta(seconds=seconds)
        if not self._timer.isActive():
            self._timer.start()
        self._tick()

    def stop(self):
        self._timer.stop()
        self._deadline = None

    def is_running(self):
        return self._timer.isActive() and (self._deadline is not None)

    def remaining_seconds(self):
        if self._deadline is None:
            return -1
        return int((self._deadline - datetime.datetime.now()).total_seconds())

    def _tick(self):
        if self._deadline is None:
            return
        remain = self.remaining_seconds()
        if callable(self._on_tick):
            self._on_tick(max(0, remain))
        if remain <= 0:
            self.stop()
            if callable(self._on_timeout):
                self._on_timeout()
