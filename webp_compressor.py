import sys
import os
import io
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PIL import Image


def compress_image_to_webp_best_quality(input_path, target_mb, output_path=None, max_iters=10):
    """
    将图片压缩为不超过 target_mb 的 WebP，并尽量保持最高 quality。
    返回: (ok, output_path, output_size_bytes, best_quality)
    """
    target_bytes = int(float(target_mb) * 1024 * 1024)
    if target_bytes <= 0:
        return False, "", 0, 0
    if not output_path:
        directory = os.path.dirname(input_path)
        name, _ = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(directory, f"{name}_compressed.webp")
    try:
        img = Image.open(input_path)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA") if "transparency" in img.info else img.convert("RGB")

        # 最高画质直接命中
        first_buffer = io.BytesIO()
        img.save(first_buffer, format="webp", quality=100)
        first_data = first_buffer.getvalue()
        if len(first_data) <= target_bytes:
            with open(output_path, "wb") as f:
                f.write(first_data)
            return True, output_path, len(first_data), 100

        low, high = 0, 100
        best_quality = 0
        best_data = b""
        for _ in range(max(1, int(max_iters))):
            mid = (low + high) // 2
            buf = io.BytesIO()
            img.save(buf, format="webp", quality=mid)
            data = buf.getvalue()
            size = len(data)
            if size <= target_bytes:
                best_quality = mid
                best_data = data
                low = mid + 1
            else:
                high = mid - 1

        if best_data:
            with open(output_path, "wb") as f:
                f.write(best_data)
            return True, output_path, len(best_data), best_quality

        # 兜底保存最低质量
        fallback = io.BytesIO()
        img.save(fallback, format="webp", quality=0)
        fallback_data = fallback.getvalue()
        with open(output_path, "wb") as f:
            f.write(fallback_data)
        return True, output_path, len(fallback_data), 0
    except Exception:
        return False, "", 0, 0


def get_unique_output_path(original_path):
    directory = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    name, _ = os.path.splitext(filename)

    base_new_name = f"{name}_compressed.webp"
    new_path = os.path.join(directory, base_new_name)

    counter = 1
    # 避免重名，叠加序号
    while os.path.exists(new_path):
        new_path = os.path.join(directory, f"{name}_compressed_{counter}.webp")
        counter += 1

    return new_path


class WebpCompressWorker(QObject):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(list)

    def __init__(self, file_paths, target_mb):
        super().__init__()
        self.file_paths = list(file_paths or [])
        self.target_mb = float(target_mb)

    def run(self):
        total = len(self.file_paths)
        results = []
        for index, img_path in enumerate(self.file_paths, start=1):
            output_path = get_unique_output_path(img_path)
            ok, saved_path, output_size_bytes, quality = compress_image_to_webp_best_quality(
                input_path=img_path,
                target_mb=self.target_mb,
                output_path=output_path,
                max_iters=10,
            )
            results.append(
                {
                    "input_path": img_path,
                    "ok": bool(ok),
                    "output_path": saved_path if ok else "",
                    "output_size_bytes": int(output_size_bytes or 0),
                    "quality": int(quality),
                }
            )
            self.progress_signal.emit(index, total, os.path.basename(img_path))
        self.finished_signal.emit(results)

class DragDropCompressor(QWidget):
    def __init__(self):
        super().__init__()
        self.worker_thread = None
        self.worker = None
        self.is_running = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle('WebP 图片定大小压缩工具')
        self.resize(400, 300)
        self.setAcceptDrops(True)

        layout = QVBoxLayout()

        # 目标大小输入区域
        self.size_label = QLabel('请输入目标大小 (MB):')
        self.size_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.size_label)

        self.size_input = QLineEdit()
        self.size_input.setText('10') # 默认 10MB
        self.size_input.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.size_input)

        # 拖拽提示区域
        self.drop_label = QLabel('请将图片拖拽到此窗口内\n(支持 JPG, PNG, BMP 等常见格式)')
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f9f9f9;
                font-size: 16px;
                color: #666;
            }
        """)
        layout.addWidget(self.drop_label)
        layout.setStretchFactor(self.drop_label, 1)

        self.status_label = QLabel("状态: 空闲")
        self.status_label.setStyleSheet("font-size: 13px; color: #444;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def dragEnterEvent(self, event):
        if self.is_running:
            event.ignore()
            return
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if self.is_running:
            QMessageBox.information(self, "处理中", "当前正在压缩，请稍候完成后再拖入。")
            return

        try:
            target_mb = float(self.size_input.text())
            if target_mb <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的大于0的数字作为目标大小！")
            return

        file_paths = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                file_paths.append(file_path)

        if not file_paths:
            QMessageBox.warning(self, "提示", "未检测到可处理的本地文件。")
            return

        self.start_batch_compress(file_paths, target_mb)

    def _set_running(self, running):
        self.is_running = bool(running)
        self.size_input.setEnabled(not running)
        if running:
            self.drop_label.setText("处理中，请稍候...\n(处理完成后会弹窗提示)")
            self.status_label.setText("状态: 处理中 0/0")
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            self.drop_label.setText("请将图片拖拽到此窗口内\n(支持 JPG, PNG, BMP 等常见格式)")
            self.status_label.setText("状态: 空闲")
            QApplication.restoreOverrideCursor()

    def start_batch_compress(self, file_paths, target_mb):
        self._set_running(True)
        total = len(file_paths)
        self.status_label.setText(f"状态: 处理中 0/{total}")

        self.worker_thread = QThread(self)
        self.worker = WebpCompressWorker(file_paths=file_paths, target_mb=target_mb)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress_signal.connect(self.on_worker_progress)
        self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.finished_signal.connect(self.worker_thread.quit)
        self.worker.finished_signal.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.on_worker_thread_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def on_worker_progress(self, current, total, current_name):
        self.status_label.setText(f"状态: 处理中 {current}/{total}")
        self.drop_label.setText(f"处理中：{current_name}\n进度 {current}/{total}")

    def on_worker_finished(self, results):
        self._set_running(False)
        success = [item for item in results if item.get("ok")]
        failed = [item for item in results if not item.get("ok")]

        if len(results) == 1:
            item = results[0]
            if item.get("ok"):
                QMessageBox.information(
                    self,
                    "压缩成功",
                    f"成功压缩至指定大小。\n"
                    f"Quality 参数: {item.get('quality')}\n"
                    f"保存路径:\n{item.get('output_path')}",
                )
            else:
                QMessageBox.critical(self, "错误", f"处理图片失败:\n{item.get('input_path')}")
            return

        message_lines = [f"处理完成：成功 {len(success)} / {len(results)}"]
        if failed:
            failed_names = [os.path.basename(item.get("input_path", "")) for item in failed[:5]]
            if failed_names:
                message_lines.append("失败文件(最多显示5个):")
                message_lines.extend(failed_names)
            if len(failed) > 5:
                message_lines.append(f"... 还有 {len(failed) - 5} 个失败文件")

        if failed:
            QMessageBox.warning(self, "压缩完成", "\n".join(message_lines))
        else:
            QMessageBox.information(self, "压缩完成", "\n".join(message_lines))

    def on_worker_thread_finished(self):
        self.worker = None
        self.worker_thread = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DragDropCompressor()
    ex.show()
    sys.exit(app.exec_())
