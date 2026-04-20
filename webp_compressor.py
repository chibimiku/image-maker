import sys
import os
import io
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
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

class DragDropCompressor(QWidget):
    def __init__(self):
        super().__init__()
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

        self.setLayout(layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        try:
            target_mb = float(self.size_input.text())
            if target_mb <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的大于0的数字作为目标大小！")
            return

        urls = event.mimeData().urls()
        for url in urls:
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                self.process_image(file_path, target_mb)

    def get_output_path(self, original_path):
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

    def process_image(self, img_path, target_mb):
        target_bytes = int(target_mb * 1024 * 1024)
        output_path = self.get_output_path(img_path)

        try:
            img = Image.open(img_path)
            # WebP 最好使用 RGB 或 RGBA 模式
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGBA') if 'transparency' in img.info else img.convert('RGB')

            # 初始检查：如果最高质量100已经小于目标大小，直接保存
            buffer = io.BytesIO()
            img.save(buffer, format="webp", quality=100)
            if len(buffer.getvalue()) <= target_bytes:
                with open(output_path, 'wb') as f:
                    f.write(buffer.getvalue())
                QMessageBox.information(self, "完成", f"原图体积已满足要求。\n已保存最高质量:\n{output_path}")
                return

            # 二分查找寻找最佳 quality
            low, high = 0, 100
            best_quality = 0
            best_buffer = None

            for _ in range(10): # 10次迭代对于 0-100 的区间足够精确
                mid = (low + high) // 2
                buffer = io.BytesIO()
                img.save(buffer, format="webp", quality=mid)
                size = len(buffer.getvalue())

                if size <= target_bytes:
                    best_quality = mid
                    best_buffer = buffer.getvalue()
                    low = mid + 1 # 尝试进一步提升画质
                else:
                    high = mid - 1 # 体积超标，降低画质

            # 写入结果
            if best_buffer:
                with open(output_path, 'wb') as f:
                    f.write(best_buffer)
                QMessageBox.information(self, "压缩成功", f"成功压缩至指定大小！\nQuality 参数: {best_quality}\n保存路径:\n{output_path}")
            else:
                # 极端情况：quality=0 依然超标（通常发生在极小目标体积下，如 0.001MB）
                buffer = io.BytesIO()
                img.save(buffer, format="webp", quality=0)
                with open(output_path, 'wb') as f:
                    f.write(buffer.getvalue())
                QMessageBox.warning(self, "提示", f"已降至最低画质，但体积可能仍大于目标大小。\n保存路径:\n{output_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理图片 {img_path} 时发生错误:\n{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DragDropCompressor()
    ex.show()
    sys.exit(app.exec_())
