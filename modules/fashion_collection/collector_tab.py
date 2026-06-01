from __future__ import annotations

import datetime
import os
import threading

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from utils.task_runtime import append_log_line, set_task_status

from .collection_service import FashionCollectionService
from .make_pic_bridge import export_bundle_manifest, export_make_pic_state, launch_make_pic
from .models import PART_DRESS, PART_SHOES, PART_SOCKS


SITE_OPTIONS = {
    "lolibrary": {
        "label": "Lolibrary",
        "search_url": "https://lolibrary.org/search?brands[]=angelic-pretty",
        "brand_placeholder": "例如 angelic-pretty",
        "brand_help": "Lolibrary 使用品牌 slug。",
    },
    "wear": {
        "label": "WEAR",
        "search_url": "https://wear.jp/women-category/onepiece/dress/",
        "brand_placeholder": "可留空；也可填品牌 key/名，例如 feepur",
        "brand_help": "WEAR 会从穿搭详情里拆取单品图，品牌条件可选。",
    },
}


class FashionCollectorWidget(QWidget):
    def __init__(self, project_root: str | None = None):
        super().__init__()
        self.project_root = os.path.abspath(project_root or os.path.join(os.path.dirname(__file__), "..", ".."))
        self.service = FashionCollectionService()
        self.latest_bundle = None
        self._worker = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        intro_label = QLabel(
            "独立采集模组：先从站点抓取服饰素材，再导出到 make-pic.py 的赛博暖暖状态文件。"
        )
        intro_label.setWordWrap(True)
        main_layout.addWidget(intro_label)

        config_group = QGroupBox("采集配置")
        form = QFormLayout()
        self.site_combo = QComboBox()
        self.site_combo.addItem("Lolibrary", "lolibrary")
        self.site_combo.addItem("WEAR", "wear")
        self.site_combo.currentIndexChanged.connect(self.on_site_changed)
        form.addRow("目标站点:", self.site_combo)

        self.search_url_input = QLineEdit("https://lolibrary.org/search?brands[]=angelic-pretty")
        form.addRow("示例搜索页:", self.search_url_input)

        self.brand_slug_input = QLineEdit("angelic-pretty")
        form.addRow("品牌 slug/key:", self.brand_slug_input)
        self.brand_help_label = QLabel("Lolibrary 使用品牌 slug。")
        self.brand_help_label.setWordWrap(True)
        form.addRow("品牌说明:", self.brand_help_label)

        self.page_count_input = QLineEdit("1")
        form.addRow("扫描页数:", self.page_count_input)

        self.aspect_ratio_input = QLineEdit("3:4")
        form.addRow("导出比例:", self.aspect_ratio_input)

        self.instructions_input = QTextEdit()
        self.instructions_input.setMaximumHeight(80)
        self.instructions_input.setPlaceholderText("这里填最终交给 make-pic.py 的整体画风/约束。")
        form.addRow("Instructions:", self.instructions_input)

        self.extra_prompt_input = QTextEdit()
        self.extra_prompt_input.setMaximumHeight(70)
        self.extra_prompt_input.setPlaceholderText("这里填局部补充，比如人物姿态、镜头、背景氛围等。")
        form.addRow("额外 Prompt:", self.extra_prompt_input)

        part_row = QHBoxLayout()
        self.dress_cb = QCheckBox("连衣裙")
        self.dress_cb.setChecked(True)
        self.shoes_cb = QCheckBox("鞋子")
        self.shoes_cb.setChecked(True)
        self.socks_cb = QCheckBox("袜子")
        self.socks_cb.setChecked(True)
        part_row.addWidget(self.dress_cb)
        part_row.addWidget(self.shoes_cb)
        part_row.addWidget(self.socks_cb)
        part_row.addStretch()
        form.addRow("采集部位:", part_row)

        config_group.setLayout(form)
        main_layout.addWidget(config_group)

        self.status_label = QLabel("状态: 就绪")
        main_layout.addWidget(self.status_label)

        button_row = QHBoxLayout()
        self.collect_btn = QPushButton("开始采集")
        self.collect_btn.clicked.connect(self.start_collection)
        button_row.addWidget(self.collect_btn)

        self.export_btn = QPushButton("导出到 make-pic")
        self.export_btn.clicked.connect(self.export_to_make_pic)
        self.export_btn.setEnabled(False)
        button_row.addWidget(self.export_btn)

        self.launch_btn = QPushButton("启动 make-pic.py")
        self.launch_btn.clicked.connect(self.launch_make_pic_ui)
        self.launch_btn.setEnabled(False)
        button_row.addWidget(self.launch_btn)
        button_row.addStretch()
        main_layout.addLayout(button_row)

        self.result_label = QLabel("尚未采集素材")
        self.result_label.setWordWrap(True)
        main_layout.addWidget(self.result_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(220)
        main_layout.addWidget(self.log_output, 1)
        self.setLayout(main_layout)
        self.on_site_changed()

    def current_site_key(self) -> str:
        return str(self.site_combo.currentData() or "lolibrary")

    def on_site_changed(self, *_args):
        site_key = self.current_site_key()
        config = SITE_OPTIONS.get(site_key, SITE_OPTIONS["lolibrary"])
        self.search_url_input.setText(config["search_url"])
        self.brand_slug_input.setPlaceholderText(config["brand_placeholder"])
        self.brand_help_label.setText(config["brand_help"])
        if site_key == "wear" and self.brand_slug_input.text().strip() == "angelic-pretty":
            self.brand_slug_input.clear()
        if site_key == "lolibrary" and not self.brand_slug_input.text().strip():
            self.brand_slug_input.setText("angelic-pretty")

    def _selected_parts(self) -> list[str]:
        parts = []
        if self.dress_cb.isChecked():
            parts.append(PART_DRESS)
        if self.shoes_cb.isChecked():
            parts.append(PART_SHOES)
        if self.socks_cb.isChecked():
            parts.append(PART_SOCKS)
        return parts

    def start_collection(self):
        if self._worker and self._worker.is_alive():
            QMessageBox.information(self, "提示", "采集任务仍在运行，请稍候。")
            return
        parts = self._selected_parts()
        if not parts:
            QMessageBox.warning(self, "提示", "请至少勾选一个目标部位。")
            return

        site_key = self.current_site_key()
        brand_slug = self.brand_slug_input.text().strip()
        if site_key == "lolibrary" and not brand_slug:
            QMessageBox.warning(self, "提示", "Lolibrary 模式请填写品牌 slug。")
            return

        try:
            max_pages = max(1, int(self.page_count_input.text().strip() or "1"))
        except ValueError:
            QMessageBox.warning(self, "提示", "扫描页数必须是整数。")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.project_root, "data", "fashion-collector", site_key, timestamp)
        append_log_line(
            self.log_output,
            f"开始采集站点 {SITE_OPTIONS.get(site_key, {}).get('label', site_key)}，品牌条件: {brand_slug or '未限制'}，目标部位: {', '.join(parts)}",
        )
        set_task_status(self.status_label, "running", "采集中")
        self.collect_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.launch_btn.setEnabled(False)

        def worker():
            try:
                bundle = self.service.collect_bundle(
                    site_key=site_key,
                    brand_slug=brand_slug,
                    output_dir=output_dir,
                    max_pages=max_pages,
                    preferred_parts=parts,
                )
            except Exception as exc:
                QTimer.singleShot(0, lambda: self._on_collect_error(str(exc)))
                return
            QTimer.singleShot(0, lambda b=bundle: self._on_collect_success(b))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _on_collect_success(self, bundle):
        self.latest_bundle = bundle
        self.collect_btn.setEnabled(True)
        self.export_btn.setEnabled(bool(bundle.assets))
        self.launch_btn.setEnabled(bool(bundle.assets))
        if bundle.missing_parts:
            set_task_status(self.status_label, "success", f"缺失: {', '.join(bundle.missing_parts)}")
        else:
            set_task_status(self.status_label, "success", "素材已齐")
        append_log_line(self.log_output, f"采集完成，共落地 {len(bundle.assets)} 张素材，目录: {bundle.output_dir}")
        for asset in bundle.assets:
            append_log_line(self.log_output, f"[{asset.part}] {asset.item.title} -> {asset.local_path}")
        if bundle.missing_parts:
            append_log_line(self.log_output, f"缺失部位: {', '.join(bundle.missing_parts)}")
        self.result_label.setText(self._build_result_text(bundle))

    def _on_collect_error(self, message: str):
        self.collect_btn.setEnabled(True)
        set_task_status(self.status_label, "error", "采集失败")
        append_log_line(self.log_output, f"采集失败: {message}")
        QMessageBox.warning(self, "采集失败", message)

    def _build_result_text(self, bundle) -> str:
        if not bundle.assets:
            return "未采集到可用素材。"
        lines = [f"{asset.part}: {asset.item.title}" for asset in bundle.assets]
        if bundle.missing_parts:
            lines.append(f"缺失: {', '.join(bundle.missing_parts)}")
        return "\n".join(lines)

    def export_to_make_pic(self):
        if not self.latest_bundle or not self.latest_bundle.assets:
            QMessageBox.information(self, "提示", "请先完成采集。")
            return

        cache_dir = os.path.join(self.project_root, "cache")
        manifest_path = export_bundle_manifest(self.latest_bundle, self.latest_bundle.output_dir)
        state_path = export_make_pic_state(
            self.latest_bundle,
            state_path=os.path.join(cache_dir, "last_state.json"),
            instructions=self.instructions_input.toPlainText(),
            extra_prompt=self.extra_prompt_input.toPlainText(),
            aspect_ratio=self.aspect_ratio_input.text().strip() or "3:4",
        )
        append_log_line(self.log_output, f"已导出清单: {manifest_path}")
        append_log_line(self.log_output, f"已写入 make-pic 状态文件: {state_path}")
        QMessageBox.information(
            self,
            "导出完成",
            f"素材清单已输出到:\n{manifest_path}\n\nmake-pic 状态文件已写入:\n{state_path}",
        )

    def launch_make_pic_ui(self):
        if not self.latest_bundle or not self.latest_bundle.assets:
            QMessageBox.information(self, "提示", "请先采集并导出素材。")
            return
        self.export_to_make_pic()
        launch_make_pic(self.project_root)
        append_log_line(self.log_output, "已启动 make-pic.py，可直接在赛博暖暖页检查导入结果。")
