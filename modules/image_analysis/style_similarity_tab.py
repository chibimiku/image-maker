"""Two-image UI using the extraction pipeline's shared metrics worker."""
import json
import os
from pathlib import Path
import uuid

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QFileDialog, QComboBox, QTableWidget,
                             QTableWidgetItem, QTextEdit, QTabWidget, QCheckBox)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from utils.style_similarity import ALL_METRICS, LABELS, LIMITS, atomic_json
from modules.image_analysis.style_deep_comparison import create_worker


class StyleSimilarityWidget(QWidget):
    def __init__(self, parent=None, config_getter=None):
        super().__init__(parent)
        self.config_getter = config_getter
        self.worker = None
        self.result = {}
        self.paths = []
        self.previews = []
        self.choose_buttons = []
        self.region_buttons = []
        layout = QVBoxLayout(self)
        images = QHBoxLayout()
        for title in ("待比较图片", "参考图片"):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            preview = QLabel("选择图片")
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setFixedHeight(140)
            column.addWidget(preview)
            path = QLineEdit()
            path.setPlaceholderText("图片路径")
            path.textChanged.connect(lambda value, i=len(self.paths): self.preview_image(i, value))
            column.addWidget(path)
            button = QPushButton("选择图片…")
            button.clicked.connect(lambda checked=False, i=len(self.paths): self.choose_image(i))
            actions = QHBoxLayout()
            actions.addWidget(button)
            region = QPushButton("面部 / 发丝定位…")
            region.clicked.connect(lambda checked=False, i=len(self.paths): self.edit_regions(i))
            actions.addWidget(region)
            column.addLayout(actions)
            self.region_buttons.append(region)
            self.choose_buttons.append(button)
            self.paths.append(path)
            self.previews.append(preview)
            images.addLayout(column)
        layout.addLayout(images)
        row = QHBoxLayout()
        self.device = QComboBox()
        for label, value in (("自动（NPU 优先）", "auto-npu"), ("自动", "auto"), ("CPU", "cpu"), ("CUDA", "cuda"), ("NPU", "npu")):
            self.device.addItem(label, value)
        row.addWidget(self.device)
        self.auto_regions = QCheckBox("自动面部定位（文本 API）")
        self.auto_regions.setChecked(config_getter is not None)
        self.auto_regions.setToolTip("首次请求定位候选并缓存；未确认的候选只显示暂定测量。关闭后只用已保存区域。")
        row.addWidget(self.auto_regions)
        self.run_button = QPushButton("计算画风相似程度")
        self.run_button.clicked.connect(self.start)
        row.addWidget(self.run_button)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        row.addWidget(self.cancel_button)
        self.export_button = QPushButton("导出 JSON…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        row.addWidget(self.export_button)
        layout.addLayout(row)
        self.status = QLabel("与多图画风提取使用同一计算入口。首次加载本地模型可能较慢。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 3)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setMinimumHeight(240)
        self.table.setHorizontalHeaderLabels(["指标", "数值", "状态"])
        self.metric_tabs = QTabWidget()
        self.metric_tabs.addTab(self.table, "整图八组")
        self.face_table = QTableWidget(0, 3)
        self.face_table.setHorizontalHeaderLabels(["面部 / 发丝指标", "贴近度", "定位 / 测量状态"])
        self.face_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metric_tabs.addTab(self.face_table, "面部与发丝细项")
        layout.addWidget(self.metric_tabs)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlainText(LIMITS)
        layout.addWidget(self.details, 1)

    def edit_regions(self, index):
        path = self.paths[index].text().strip()
        if not Path(path).is_file():
            self.status.setText("请先选择有效图片。")
            return
        from modules.image_analysis.style_regions_dialog import StyleRegionsDialog
        dialog = StyleRegionsDialog(path, self.config_getter, self)
        if dialog.exec():
            self.result = {}
            self.table.setRowCount(0)
            self.face_table.setRowCount(0)
            self.export_button.setEnabled(False)
            self.status.setText("区域已保存；重新计算将使用当前定位。未确认定位只显示暂定测量。")

    def choose_image(self, index):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片 (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
            self.paths[index].setText(path)

    def preview_image(self, index, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.previews[index].setText("无法预览")
        else:
            self.previews[index].setPixmap(pixmap.scaled(300, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.result = {}
        self.table.setRowCount(0)
        self.face_table.setRowCount(0)
        self.export_button.setEnabled(False)

    def start(self):
        if self.worker:
            return
        paths = [str(Path(field.text().strip()).resolve()) for field in self.paths]
        if not all(Path(p).is_file() for p in paths):
            self.status.setText("请选择两张有效图片。")
            return
        directory = Path(__file__).resolve().parents[2] / "cache/temp/style-similarity" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        state_path = directory / "state.json"
        atomic_json(state_path, {"dataset": {"images": [paths[1]]},
                                "test_images": {"pair": {"generated_files": {"input": [paths[0]]}}}})
        try:
            region_config = self.config_getter() if self.auto_regions.isChecked() and self.config_getter else None
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self.result = {}
        self.table.setRowCount(0)
        self.face_table.setRowCount(0)
        self.export_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.device.setEnabled(False)
        self.auto_regions.setEnabled(False)
        for field in self.paths + self.choose_buttons + self.region_buttons:
            field.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText("正在计算八组指标…")
        self.worker = create_worker(state_path, self.device.currentData(), self,
                                    locate_regions=self.auto_regions.isChecked(), region_config=region_config)
        self.worker.progress.connect(self.status.setText)
        self.worker.completed.connect(self.completed)
        self.worker.finished.connect(self.stopped)
        self.worker.start()

    def completed(self, result, error):
        if error:
            self.status.setText("计算失败")
            self.details.setPlainText(error)
            return
        self.result = result
        pair = result["rows"][0]["pairs"][0]
        self.table.setRowCount(len(ALL_METRICS))
        for row, metric in enumerate(ALL_METRICS):
            outcome = pair[metric]
            value = format(outcome["value"], ".8g") if outcome["value"] is not None else "缺失"
            for column, text in enumerate((LABELS[metric], value, outcome["status"])):
                self.table.setItem(row, column, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        from utils.style_face_metrics import LABELS as FACE_LABELS
        features = pair.get("face_features", {})
        self.face_table.setRowCount(len(FACE_LABELS))
        for row, (metric, label) in enumerate(FACE_LABELS.items()):
            outcome = features.get(metric, {"status": "unavailable", "value": None})
            value = format(outcome["value"], ".5g") if outcome["value"] is not None else "缺失"
            for column, text in enumerate((label, value, outcome["status"])):
                self.face_table.setItem(row, column, QTableWidgetItem(text))
        self.face_table.resizeColumnsToContents()
        backend = result["backend"]
        self.status.setText(f"整图 {result['status']} · 面部 {result.get('face_status', '未定位')} · {backend['actual']} / {backend['precision']}")
        self.details.setPlainText(json.dumps(result, ensure_ascii=False, indent=2))
        self.export_button.setEnabled(True)

    def stopped(self):
        worker = self.worker
        self.worker = None
        self.run_button.setEnabled(True)
        self.device.setEnabled(True)
        self.auto_regions.setEnabled(True)
        for field in self.paths + self.choose_buttons + self.region_buttons:
            field.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if worker:
            worker.deleteLater()

    def cancel(self):
        if self.worker:
            self.worker.requestInterruption()

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出指标", "style-similarity.json", "JSON (*.json)")
        if path and self.result:
            atomic_json(path, self.result)

    def closeEvent(self, event):
        if self.worker:
            self.cancel()
            self.worker.wait()
        super().closeEvent(event)
