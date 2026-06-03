from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import threading

from PyQt6.QtCore import QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap
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
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from modules.others.api_backend import generate_image_aigc2d, get_api_config
from utils.task_runtime import append_log_line, set_task_status

from .collection_service import FashionCollectionService
from .generation_plan import (
    PART_LABELS,
    build_reference_prompt,
    build_scene_and_character_description,
    load_styles_config,
    resolve_prompt_and_instructions,
    resolve_style_bundle,
)
from .make_pic_bridge import export_bundle_manifest, export_make_pic_state, launch_make_pic
from .models import PART_BAG, PART_DRESS, PART_HAIR_ACCESSORY, PART_SHOES, PART_SOCKS
from .theme_profiles import get_theme_profile


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
    "mayla": {
        "label": "MAYLA",
        "search_url": "https://mayla.jp/main/category/classification/?kn=b1",
        "brand_placeholder": "可留空；MAYLA 是单一品牌，无需填写",
        "brand_help": "MAYLA 是日本少女鞋履/配饰品牌，使用浏览器后台抓取 JS 渲染页面。",
    },
    "hybrid": {
        "label": "Hybrid",
        "search_url": "Lolibrary 主裙 + WEAR 鞋袜/发饰/包袋",
        "brand_placeholder": "可留空；主裙默认会优先尝试 angelic-pretty",
        "brand_help": "Hybrid 会优先用 Lolibrary 取主裙，再用 WEAR 补鞋袜、发饰和包袋。",
    },
}

AUTO_STYLE_TEXT = "自动(主题默认)"


class FashionCollectorWidget(QWidget):
    log_signal = pyqtSignal(str)
    collect_done = pyqtSignal(object)
    collect_error = pyqtSignal(str)
    generate_done = pyqtSignal(object)
    generate_error = pyqtSignal(str)
    analysis_done = pyqtSignal(dict)
    analysis_error = pyqtSignal(str)
    def __init__(self, project_root: str | None = None):
        super().__init__()
        self.project_root = os.path.abspath(project_root or os.path.join(os.path.dirname(__file__), "..", ".."))
        self.service = FashionCollectionService()
        self.styles_data = load_styles_config(os.path.join(self.project_root, "conf", "config-styles.json"))
        self.latest_bundle = None
        self._worker = None
        self._generate_worker = None
        self._analysis_worker = None
        self._char_analysis_worker = None
        self.char_image_path: str | None = None
        self.char_analysis: dict | None = None  # Vision analysis result for character ref
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        intro_label = QLabel(
            "独立采集模组：可采集服饰素材、导出到 make-pic.py，并直接调用 AIGC2D 生成少女图。"
        )
        intro_label.setWordWrap(True)
        main_layout.addWidget(intro_label)

        config_group = QGroupBox("采集配置")
        form = QFormLayout()
        self.site_combo = QComboBox()
        self.site_combo.addItem("Lolibrary", "lolibrary")
        self.site_combo.addItem("WEAR", "wear")
        self.site_combo.addItem("MAYLA", "mayla")
        self.site_combo.addItem("Hybrid", "hybrid")
        self.site_combo.currentIndexChanged.connect(self.on_site_changed)
        form.addRow("目标站点:", self.site_combo)

        self.search_url_input = QLineEdit("https://lolibrary.org/search?brands[]=angelic-pretty")
        self.search_url_input.setReadOnly(True)
        form.addRow("示例搜索页:", self.search_url_input)

        self.brand_slug_input = QLineEdit("angelic-pretty")
        form.addRow("品牌 slug/key:", self.brand_slug_input)
        self.brand_help_label = QLabel("Lolibrary 使用品牌 slug。")
        self.brand_help_label.setWordWrap(True)
        form.addRow("品牌说明:", self.brand_help_label)

        self.page_count_input = QLineEdit("1")
        form.addRow("扫描页数:", self.page_count_input)

        proxy_row = QHBoxLayout()
        self.proxy_input = QLineEdit("")
        self.proxy_input.setPlaceholderText("例如 http://127.0.0.1:7897")
        proxy_row.addWidget(self.proxy_input, stretch=1)
        self.proxy_cb = QCheckBox("启用代理")
        self.proxy_cb.toggled.connect(self._on_proxy_toggled)
        proxy_row.addWidget(self.proxy_cb)
        form.addRow("HTTP 代理:", proxy_row)
        self.proxy_input.textChanged.connect(lambda _: self._save_collector_config())

        self.theme_input = QLineEdit("甜美洛丽塔")
        self.theme_input.setPlaceholderText("例如 甜美洛丽塔、黑白学院风、夏日海边少女")
        form.addRow("采集主题:", self.theme_input)

        self.style_combo = QComboBox()
        self.style_combo.setEditable(True)
        self.style_combo.addItem(AUTO_STYLE_TEXT)
        for style_name in sorted(self.styles_data.keys()):
            self.style_combo.addItem(style_name)
        self.style_combo.setCurrentText(AUTO_STYLE_TEXT)
        form.addRow("画风 style:", self.style_combo)

        self.character_count_combo = QComboBox()
        self.character_count_combo.addItem("1 人", 1)
        self.character_count_combo.addItem("2 人", 2)
        form.addRow("主角人数:", self.character_count_combo)

        self.aspect_ratio_input = QLineEdit("3:4")
        form.addRow("导出比例:", self.aspect_ratio_input)

        self.resolution_input = QLineEdit("")
        self.resolution_input.setPlaceholderText("可选，例如 2K；为空时沿用配置")
        form.addRow("生成分辨率:", self.resolution_input)

        self.file_prefix_input = QLineEdit("fashion_gui")
        form.addRow("输出前缀:", self.file_prefix_input)

        self.instructions_input = QTextEdit()
        self.instructions_input.setMaximumHeight(90)
        self.instructions_input.setPlaceholderText("这里填额外的系统约束，例如构图、镜头、质感要求。")
        self.instructions_input.textChanged.connect(lambda: self._save_collector_config())
        form.addRow("Instructions:", self.instructions_input)

        self.extra_prompt_input = QTextEdit()
        self.extra_prompt_input.setMaximumHeight(80)
        self.extra_prompt_input.setPlaceholderText("这里填额外人物/镜头提示；系统会自动补场景和主角描述。")
        self.extra_prompt_input.textChanged.connect(lambda: self._save_collector_config())
        form.addRow("额外 Prompt:", self.extra_prompt_input)

        # --- Character reference image drop zone ---
        char_img_wrapper = QWidget()
        char_img_wrapper.setMaximumHeight(148)
        char_img_layout = QVBoxLayout()
        char_img_layout.setContentsMargins(0, 0, 0, 0)
        char_img_layout.setSpacing(2)
        char_img_top = QHBoxLayout()
        char_img_top.setContentsMargins(0, 0, 0, 0)
        self.char_img_label = self._DropLabel("拖入角色参考图\n（可选，用于固定女主角形象）", self)
        self.char_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.char_img_label.setFixedSize(160, 120)
        self.char_img_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.char_img_label.setStyleSheet(
            "QLabel { border: 2px dashed #888; border-radius: 6px; background: #f5f5f5; color: #666; font-size: 11px; }"
        )
        self.char_img_label.set_on_drop(self._set_character_image)
        self.char_img_clear_btn = QPushButton("清除")
        self.char_img_clear_btn.setMaximumWidth(50)
        self.char_img_clear_btn.clicked.connect(self._clear_character_image)
        self.char_img_clear_btn.setVisible(False)
        char_img_top.addWidget(self.char_img_label)
        char_img_top.addWidget(self.char_img_clear_btn, alignment=Qt.AlignmentFlag.AlignTop)
        char_img_layout.addLayout(char_img_top)
        self.char_img_status = QLabel("")
        self.char_img_status.setStyleSheet("color: #888; font-size: 10px;")
        self.char_img_status.setWordWrap(True)
        self.char_img_status.setMaximumHeight(28)
        char_img_layout.addWidget(self.char_img_status)
        char_img_wrapper.setLayout(char_img_layout)
        form.addRow("角色参考:", char_img_wrapper)

        part_row = QHBoxLayout()
        self.dress_cb = QCheckBox("连衣裙")
        self.dress_cb.setChecked(True)
        self.shoes_cb = QCheckBox("鞋子")
        self.shoes_cb.setChecked(True)
        self.socks_cb = QCheckBox("袜子")
        self.socks_cb.setChecked(True)
        self.hair_accessory_cb = QCheckBox("发饰")
        self.hair_accessory_cb.setChecked(True)
        self.bag_cb = QCheckBox("包袋/手持物")
        self.bag_cb.setChecked(True)
        part_row.addWidget(self.dress_cb)
        part_row.addWidget(self.shoes_cb)
        part_row.addWidget(self.socks_cb)
        part_row.addWidget(self.hair_accessory_cb)
        part_row.addWidget(self.bag_cb)
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

        self.generate_btn = QPushButton("生成少女图")
        self.generate_btn.clicked.connect(self.generate_image_from_bundle)
        self.generate_btn.setEnabled(False)
        button_row.addWidget(self.generate_btn)

        self.export_btn = QPushButton("导出到 make-pic")
        self.export_btn.clicked.connect(self.export_to_make_pic)
        self.export_btn.setEnabled(False)
        button_row.addWidget(self.export_btn)

        self.launch_btn = QPushButton("启动 make-pic.py")
        self.launch_btn.clicked.connect(self.launch_make_pic_ui)
        self.launch_btn.setEnabled(False)
        button_row.addWidget(self.launch_btn)
        button_row.addStretch()
        self.auto_gen_cb = QCheckBox("采集后自动生成图片")
        self.auto_gen_cb.toggled.connect(lambda _: self._save_collector_config())
        button_row.addWidget(self.auto_gen_cb)
        self.auto_analyze_cb = QCheckBox("出图后自动分析")
        self.auto_analyze_cb.setChecked(True)
        self.auto_analyze_cb.toggled.connect(lambda _: self._save_collector_config())
        button_row.addWidget(self.auto_analyze_cb)
        self.color_match_cb = QCheckBox("颜色协调适配")
        self.color_match_cb.setChecked(False)
        self.color_match_cb.setToolTip("启用后先选连衣裙，后续配件颜色需与裙子色调协调")
        self.color_match_cb.toggled.connect(lambda _: self._save_collector_config())
        button_row.addWidget(self.color_match_cb)
        main_layout.addLayout(button_row)

        self.result_label = QLabel("尚未采集素材")
        self.result_label.setWordWrap(True)
        main_layout.addWidget(self.result_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(220)
        main_layout.addWidget(self.log_output, 1)
        self.log_signal.connect(lambda msg: append_log_line(self.log_output, msg))
        self.collect_done.connect(self._on_collect_success)
        self.collect_error.connect(self._on_collect_error)
        self.generate_done.connect(self._on_generate_success)
        self.generate_error.connect(self._on_generate_error)
        self.analysis_done.connect(self._on_analysis_success)
        self.analysis_error.connect(self._on_analysis_error)
        self.setLayout(main_layout)
        self.on_site_changed()
        self._load_collector_config()

    # ---- character reference image ----

    class _DropLabel(QLabel):
        """QLabel subclass that accepts image drag-and-drop."""

        def __init__(self, text: str = "", parent=None):
            super().__init__(text, parent)
            self.setAcceptDrops(True)
            self._on_drop = None

        def set_on_drop(self, callback):
            self._on_drop = callback

        def dragEnterEvent(self, event):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
            else:
                event.ignore()

        def dragMoveEvent(self, event):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
            else:
                event.ignore()

        def dragLeaveEvent(self, event):
            event.accept()

        def dropEvent(self, event):
            urls = event.mimeData().urls()
            if urls and self._on_drop:
                for url in urls:
                    path = url.toLocalFile()
                    if path and os.path.isfile(path):
                        self._on_drop(path)
                        event.acceptProposedAction()
                        return
            event.ignore()

    def _set_character_image(self, image_path: str) -> None:
        if not os.path.isfile(image_path):
            return
        self.char_image_path = image_path
        # Show thumbnail
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(156, 116, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.char_img_label.setPixmap(pixmap)
            self.char_img_label.setStyleSheet(
                "QLabel { border: 2px solid #66b; border-radius: 6px; }"
            )
            self.char_img_clear_btn.setVisible(True)
            fname = os.path.basename(image_path)
            self.char_img_status.setText(f"已加载: {fname}"[:80])
        self._save_collector_config()
        # Start analysis in background
        self._analyze_character_image(image_path)

    def _clear_character_image(self) -> None:
        self.char_image_path = None
        self.char_analysis = None
        self.char_img_label.clear()
        self.char_img_label.setText("拖入角色参考图\n（可选，用于固定女主角形象）")
        self.char_img_label.setStyleSheet(
            "QLabel { border: 2px dashed #888; border-radius: 6px; background: #f5f5f5; color: #666; font-size: 11px; }"
        )
        self.char_img_clear_btn.setVisible(False)
        self.char_img_status.setText("")
        self._save_collector_config()
        append_log_line(self.log_output, "[角色图] 角色参考图已清除")

    def _analyze_character_image(self, image_path: str) -> None:
        append_log_line(self.log_output, f"[角色图] 开始分析角色外观: {os.path.basename(image_path)}")
        def worker():
            try:
                self.log_signal.emit("[角色图] 正在通过 Vision 分析角色特征...")
                from modules.others.api_backend import get_api_config
                cfg = get_api_config(api_type="aigc2d")
                api_base = str(cfg.get("base_url", "") or "").strip()
                api_key = cfg.get("api_key", "")
                if not api_key:
                    self.log_signal.emit("[角色图] api_key 缺失，跳过分析")
                    return
                if "/v1beta/models/" in api_base:
                    api_base = api_base.split("/v1beta/models/")[0] + "/v1"
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=api_base, timeout=60)
                # Use a specific prompt focused on character appearance
                char_system = (
                    "You are a character design analyst. Analyze the given anime/game character image "
                    "and output the character's physical appearance traits in JSON format. "
                    "Focus ONLY on visible traits — do not invent details.\n\n"
                    "IMPORTANT: Pay special attention to hair decorations (ribbons, clips, headbands, "
                    "hairpins, bows, flowers in hair, etc.) as these are key to character identity.\n\n"
                    'Output JSON: {"hair_color": "颜色(中英文均可)", "hair_length": "short/medium/long", '
                    '"hair_style": "发型描述(如:双马尾/单马尾/长发/短发/卷发/直发等)", '
                    '"hair_accessory": "头发上的饰物描述(蝴蝶结/发箍/发夹/头花等, 无则写无)", '
                    '"eye_color": "颜色", "skin_tone": "fair/pale/light/medium/tan/dark", '
                    '"distinctive_features": "其他显著特征(眼镜/泪痣/呆毛等, 无则写无)", '
                    '"overall_vibe": "整体气质(2-3词)", "character_description": "一句话中文角色外观描述,包含发型发色瞳色发饰"}'
                )
                import base64 as _b64
                with open(image_path, "rb") as f:
                    img_b64 = _b64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(image_path)[1].lower()
                mime = "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "image/jpeg"
                # Use fetch_llm_json with a simpler approach — just use the OpenAI client directly for vision
                self.log_signal.emit("[角色图] 正在请求 Vision 分析...")
                response = client.chat.completions.create(
                    model="gemini-2.5-flash",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": char_system},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                        ]
                    }],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or ""
                self.log_signal.emit(f"[角色图] Vision 原始返回: {raw[:300]}")
                result = json.loads(raw) if isinstance(raw, str) else (raw or {})
                if isinstance(result, dict) and result.get("hair_color"):
                    self.char_analysis = result
                    desc = result.get("character_description", "")
                    hair = result.get("hair_color", "")
                    eyes = result.get("eye_color", "")
                    self.log_signal.emit(
                        f"[角色图] 分析完成: 发色={hair}, 瞳色={eyes}, 描述={desc[:60]}"
                    )
                    if desc:
                        hair_acc = result.get("hair_accessory", "")
                        status = f"特征: {desc[:40]} 发={hair} 瞳={eyes}"
                        if hair_acc and hair_acc not in ("无", "无", "none", "None"):
                            status += f" 发饰={hair_acc[:15]}"
                        self.char_img_status.setText(status[:100])
                else:
                    self.log_signal.emit("[角色图] Vision 未返回有效分析结果")
            except Exception as exc:
                self.log_signal.emit(f"[角色图] 分析失败: {exc}")
        self._char_analysis_worker = threading.Thread(target=worker, daemon=True)
        self._char_analysis_worker.start()

    # ---- config persistence ----

    def _collector_config_path(self) -> str:
        return os.path.join(self.project_root, "conf", "config-collector.json")

    def _load_collector_config(self) -> None:
        path = self._collector_config_path()
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                proxy = str(data.get("proxy_url", "")).strip()
                if proxy:
                    self.proxy_input.setText(proxy)
                    self.proxy_cb.setChecked(bool(data.get("proxy_enabled", True)))
                    self._on_proxy_toggled(self.proxy_cb.isChecked())
                instructions = str(data.get("instructions", "")).strip()
                if instructions:
                    self.instructions_input.setPlainText(instructions)
                extra_prompt = str(data.get("extra_prompt", "")).strip()
                if extra_prompt:
                    self.extra_prompt_input.setPlainText(extra_prompt)
                self.auto_gen_cb.setChecked(bool(data.get("auto_generate", False)))
                self.auto_analyze_cb.setChecked(bool(data.get("auto_analyze", True)))
                self.color_match_cb.setChecked(bool(data.get("color_match", False)))
                char_img = str(data.get("char_image_path", "")).strip()
                if char_img and os.path.isfile(char_img):
                    self._set_character_image(char_img)
        except Exception:
            pass

    def _save_collector_config(self) -> None:
        data = {
            "proxy_url": self.proxy_input.text().strip(),
            "proxy_enabled": self.proxy_cb.isChecked(),
            "instructions": self.instructions_input.toPlainText(),
            "extra_prompt": self.extra_prompt_input.toPlainText(),
            "auto_generate": self.auto_gen_cb.isChecked(),
            "auto_analyze": self.auto_analyze_cb.isChecked(),
            "color_match": self.color_match_cb.isChecked(),
            "char_image_path": self.char_image_path or "",
        }
        path = self._collector_config_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _on_proxy_toggled(self, checked: bool) -> None:
        self.proxy_input.setEnabled(checked)
        self._save_collector_config()

    def _get_active_proxy_url(self) -> str | None:
        if not self.proxy_cb.isChecked():
            return None
        url = self.proxy_input.text().strip()
        return url if url else None

    # -------------------------

    def current_site_key(self) -> str:
        return str(self.site_combo.currentData() or "lolibrary")

    def on_site_changed(self, *_args):
        site_key = self.current_site_key()
        config = SITE_OPTIONS.get(site_key, SITE_OPTIONS["lolibrary"])
        self.search_url_input.setText(config["search_url"])
        self.brand_slug_input.setPlaceholderText(config["brand_placeholder"])
        self.brand_help_label.setText(config["brand_help"])
        if site_key in {"wear", "hybrid", "mayla"} and self.brand_slug_input.text().strip() == "angelic-pretty":
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
        if self.hair_accessory_cb.isChecked():
            parts.append(PART_HAIR_ACCESSORY)
        if self.bag_cb.isChecked():
            parts.append(PART_BAG)
        return parts

    def _is_busy(self) -> bool:
        return bool((self._worker and self._worker.is_alive()) or (self._generate_worker and self._generate_worker.is_alive()))

    def _current_style_value(self) -> str:
        value = self.style_combo.currentText().strip()
        return "" if value == AUTO_STYLE_TEXT else value

    def _build_generation_context(self, bundle) -> dict:
        theme_text = self.theme_input.text().strip()
        theme_profile = get_theme_profile(theme_text)
        resolved_style_names, style_text = resolve_style_bundle(
            self._current_style_value(),
            self.styles_data,
            theme_profile=theme_profile,
        )
        final_prompt_base, final_instructions = resolve_prompt_and_instructions(
            self.extra_prompt_input.toPlainText(),
            self.instructions_input.toPlainText(),
            theme_profile=theme_profile,
            style_text=style_text,
        )
        scene_text, character_text = build_scene_and_character_description(
            bundle,
            theme_profile=theme_profile,
            character_count=int(self.character_count_combo.currentData() or 1),
        )

        # ---- character reference image override ----
        if self.char_analysis and isinstance(self.char_analysis, dict):
            char_desc = self.char_analysis.get("character_description", "")
            hair_color = self.char_analysis.get("hair_color", "")
            eye_color = self.char_analysis.get("eye_color", "")
            hair_style = self.char_analysis.get("hair_style", "")
            hair_length = self.char_analysis.get("hair_length", "")
            hair_accessory = self.char_analysis.get("hair_accessory", "")
            skin = self.char_analysis.get("skin_tone", "")
            distinctive = self.char_analysis.get("distinctive_features", "")
            vibe = self.char_analysis.get("overall_vibe", "")

            # Build character description from Vision analysis
            char_parts = [char_desc] if char_desc else []
            if hair_color and hair_style:
                char_parts.append(f"发色: {hair_color}{hair_length}({hair_style})")
            elif hair_color:
                char_parts.append(f"发色: {hair_color}")
            if hair_accessory and hair_accessory not in ("无", "无", "none", "None"):
                char_parts.append(f"发饰: {hair_accessory}")
            if eye_color:
                char_parts.append(f"瞳色: {eye_color}")
            if skin and skin != "无":
                char_parts.append(f"肤色: {skin}")
            if distinctive and distinctive not in ("无", "无", "none", "None"):
                char_parts.append(f"特征: {distinctive}")
            if vibe:
                char_parts.append(f"气质: {vibe}")

            analysis_char_text = "\n".join(char_parts)
            if analysis_char_text:
                # Character identity — the reference image provides hair (style+color+accessory)
                # and eye color for recognizability; clothing comes from collected assets
                character_text = (
                    "主角描述: 最后一张参考图是该角色原型，必须严格复制以下辨识特征以让观众认出这个角色: "
                    f"{analysis_char_text}。"
                    "发型、发色、头发上的装饰物以及瞳色必须与参考图完全一致。"
                    "但该角色穿着的服装必须换成前面参考图所提供的新服饰套装。"
                )
                # Strip conflicting character descriptions
                conflict_patterns = [
                    r'粉色头发', r'金色头发', r'黑色长发', r'黑色短发', r'棕色头发',
                    r'蓝色头发', r'紫色头发', r'白色头发', r'红色头发', r'绿色头发',
                    r'粉发', r'金发', r'黑发', r'棕发', r'蓝发', r'紫发', r'白发', r'红发', r'绿发',
                    r'pink\s*hair', r'blonde?\s*hair', r'black\s*hair', r'brown\s*hair',
                    r'blue\s*hair', r'purple\s*hair', r'white\s*hair', r'red\s*hair',
                    r'碧眼', r'蓝瞳', r'绿瞳', r'红瞳', r'棕瞳', r'金瞳', r'紫瞳',
                    r'双马尾', r'单马尾', r'短发', r'长发', r'卷发', r'直发',
                ]
                for pat in conflict_patterns:
                    final_prompt_base = re.sub(pat, '', final_prompt_base, flags=re.IGNORECASE)
                final_prompt_base = re.sub(r'\n{3,}', '\n\n', final_prompt_base).strip()

                final_prompt_base = (
                    "角色原型设定: 最后一张参考图是女主角的原型图，必须100%复制以下物理特征以保证角色可辨识: "
                    f"{analysis_char_text}。"
                    "发型、发色、头发上的装饰物（蝴蝶结/发箍/发夹等）以及瞳色，必须与参考图严格一致，这是角色辨识的关键。"
                    "但是，全身服装穿搭必须全部替换为前面参考图中展示的连衣裙、鞋袜、发饰（服装类）、包袋等。"
                    "切勿复制参考图中的衣服、鞋子、袜子或包袋。\n\n"
                    + final_prompt_base
                )

                final_instructions = (
                    "角色辨识度要求: 最后一张参考图的用途仅限于角色外貌辨识——"
                    "必须严格保留其发型、发色、头发上的饰物以及瞳色，让观众一看就知道是同一个角色。"
                    "但该图中的服装穿搭完全忽略，角色必须穿着前面参考图提供的服饰套装"
                    "（连衣裙、鞋子、袜子、服装类发饰、包袋等）。"
                    "忽略Prompt中任何与参考图角色外貌冲突的头发/眼睛描述。\n\n"
                    + final_instructions
                )

        final_prompt = build_reference_prompt(
            final_prompt_base,
            bundle,
            scene_text=scene_text,
            character_text=character_text,
        )
        composed_extra_prompt = "\n\n".join(
            [text for text in [final_prompt_base, scene_text, character_text] if str(text or "").strip()]
        ).strip()
        return {
            "theme_profile": theme_profile,
            "theme_text": theme_text,
            "resolved_style_names": resolved_style_names,
            "style_text": style_text,
            "final_prompt_base": final_prompt_base,
            "final_instructions": final_instructions,
            "scene_text": scene_text,
            "character_text": character_text,
            "final_prompt": final_prompt,
            "composed_extra_prompt": composed_extra_prompt,
        }

    def _default_file_prefix(self) -> str:
        value = self.file_prefix_input.text().strip()
        return value or "fashion_gui"

    def _set_busy_state(self, busy: bool):
        self.collect_btn.setEnabled(not busy)
        self.generate_btn.setEnabled((not busy) and bool(self.latest_bundle and self.latest_bundle.assets))
        self.export_btn.setEnabled((not busy) and bool(self.latest_bundle and self.latest_bundle.assets))
        self.launch_btn.setEnabled((not busy) and bool(self.latest_bundle and self.latest_bundle.assets))

    def start_collection(self):
        if self._is_busy():
            QMessageBox.information(self, "提示", "当前已有任务在运行，请稍候。")
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

        proxy_url = self._get_active_proxy_url()
        if proxy_url:
            self.service.set_proxy_url(proxy_url)
            append_log_line(self.log_output, f"[代理] 已启用代理: {proxy_url}")
        else:
            self.service.set_proxy_url(None)
            append_log_line(self.log_output, "[代理] 未启用代理，使用直连")

        self.service.enable_color_match = self.color_match_cb.isChecked()
        if self.service.enable_color_match:
            append_log_line(self.log_output, "[颜色] 颜色协调适配已启用，配件将与连衣裙色调匹配")

        append_log_line(
            self.log_output,
            f"开始采集站点 {SITE_OPTIONS.get(site_key, {}).get('label', site_key)}，品牌条件: {brand_slug or '未限制'}，目标部位: {', '.join(parts)}，主题: {self.theme_input.text().strip() or '未设置'}",
        )
        set_task_status(self.status_label, "running", "采集中")
        self._set_busy_state(True)

        def worker():
            try:
                bundle = self.service.collect_bundle(
                    site_key=site_key,
                    brand_slug=brand_slug,
                    output_dir=output_dir,
                    max_pages=max_pages,
                    preferred_parts=parts,
                    theme=self.theme_input.text().strip(),
                    log_callback=lambda msg: self.log_signal.emit(msg),
                )
            except Exception as exc:
                self.collect_error.emit(str(exc))
                return
            self.collect_done.emit(bundle)

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _on_collect_success(self, bundle):
        self.latest_bundle = bundle
        self._set_busy_state(False)
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
        if self.auto_gen_cb.isChecked() and bundle.assets:
            QTimer.singleShot(200, self.generate_image_from_bundle)

    def _on_collect_error(self, message: str):
        self._set_busy_state(False)
        set_task_status(self.status_label, "error", "采集失败")
        append_log_line(self.log_output, f"采集失败: {message}")
        QMessageBox.warning(self, "采集失败", message)

    def _build_result_text(self, bundle) -> str:
        if not bundle.assets:
            return "未采集到可用素材。"
        context = self._build_generation_context(bundle)
        lines = [f"{PART_LABELS.get(asset.part, asset.part)}: {asset.item.title}" for asset in bundle.assets]
        if bundle.missing_parts:
            lines.append(f"缺失: {', '.join(bundle.missing_parts)}")
        lines.append("")
        lines.append(context["scene_text"])
        lines.append(context["character_text"])
        return "\n".join(lines)

    def export_to_make_pic(self):
        if not self.latest_bundle or not self.latest_bundle.assets:
            QMessageBox.information(self, "提示", "请先完成采集。")
            return

        context = self._build_generation_context(self.latest_bundle)
        cache_dir = os.path.join(self.project_root, "cache")
        manifest_path = export_bundle_manifest(self.latest_bundle, self.latest_bundle.output_dir)
        state_path = export_make_pic_state(
            self.latest_bundle,
            state_path=os.path.join(cache_dir, "last_state.json"),
            instructions=context["final_instructions"],
            extra_prompt=context["composed_extra_prompt"],
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

    def generate_image_from_bundle(self):
        if self._is_busy():
            QMessageBox.information(self, "提示", "当前已有任务在运行，请稍候。")
            return
        if not self.latest_bundle or not self.latest_bundle.assets:
            QMessageBox.information(self, "提示", "请先完成采集。")
            return

        bundle = self.latest_bundle
        context = self._build_generation_context(bundle)
        image_paths = [asset.local_path for asset in bundle.assets if os.path.isfile(asset.local_path)]
        if not image_paths:
            QMessageBox.warning(self, "提示", "当前没有可用的本地素材图。")
            return

        # Append character reference image at the end (clothing images first so
        # the model registers the outfit before seeing the character reference)
        if self.char_image_path and os.path.isfile(self.char_image_path):
            image_paths.append(self.char_image_path)
            append_log_line(self.log_output, f"[角色图] 参考图 1 张: {os.path.basename(self.char_image_path)}")

        self._set_busy_state(True)
        set_task_status(self.status_label, "running", "生成中")
        append_log_line(self.log_output, f"开始生成少女图，主角人数: {self.character_count_combo.currentData()}，主题: {context['theme_text'] or '未设置'}")
        if context["resolved_style_names"]:
            append_log_line(self.log_output, f"已应用画风预设: {', '.join(context['resolved_style_names'])}")
        append_log_line(self.log_output, context["scene_text"])
        append_log_line(self.log_output, context["character_text"])

        def worker():
            try:
                self.log_signal.emit("[生成] 正在发送 API 请求...")
                self.log_signal.emit(f"[生成] Prompt 长度: {len(context['final_prompt'])} chars, 参考图: {len(image_paths)} 张")
                result = generate_image_aigc2d(
                    prompt=context["final_prompt"],
                    image_paths=image_paths,
                    aspect_ratio=self.aspect_ratio_input.text().strip() or "3:4",
                    instructions=context["final_instructions"],
                    resolution=self.resolution_input.text().strip() or None,
                    api_type="aigc2d",
                    save_sub_dir="fashion-generate",
                    file_prefix=self._default_file_prefix(),
                    return_metadata=True,
                    log_callback=lambda msg: self.log_signal.emit(msg),
                )
                self.log_signal.emit("[生成] API 请求完成，处理结果中...")
            except Exception as exc:
                self.generate_error.emit(str(exc))
                return
            self.generate_done.emit(result)

        self._generate_worker = threading.Thread(target=worker, daemon=True)
        self._generate_worker.start()

    def _on_generate_success(self, result):
        self._set_busy_state(False)
        raw_path = ""
        saved_files = []
        if isinstance(result, dict):
            saved_files = result.get("saved_files") or []
            raw_path = os.path.join(self.latest_bundle.output_dir, f"{self._default_file_prefix()}_aigc2d_result.json")
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            saved_files = result or []

        if raw_path:
            append_log_line(self.log_output, f"AIGC2D 原始结果: {raw_path}")
        if saved_files:
            set_task_status(self.status_label, "success", "生成完成")
            append_log_line(self.log_output, "生成图片:")
            for path in saved_files:
                append_log_line(self.log_output, path)
            append_log_line(self.log_output, "少女图已生成并保存到 data 目录。")
            if self.auto_analyze_cb.isChecked():
                first_image = saved_files[0]
                QTimer.singleShot(300, lambda: self._start_auto_analysis(first_image))
            else:
                self._send_windows_notification("Image Maker", "少女图生成完成")
        else:
            set_task_status(self.status_label, "error", "生成失败")
            append_log_line(self.log_output, "AIGC2D 未返回可保存图片，生成失败。")

    def _on_generate_error(self, message: str):
        self._set_busy_state(False)
        set_task_status(self.status_label, "error", "生成失败")
        append_log_line(self.log_output, f"生成失败: {message}")

    # ---- auto-analysis ----

    def _start_auto_analysis(self, image_path: str) -> None:
        set_task_status(self.status_label, "running", "分析中")
        self._set_busy_state(True)
        append_log_line(self.log_output, f"[分析] 开始分析图片: {os.path.basename(image_path)}")

        def worker():
            try:
                self.log_signal.emit("[分析] 正在初始化 API 客户端...")
                cfg = get_api_config(api_type="aigc2d")
                api_base = str(cfg.get("base_url", "") or "").strip()
                api_key = cfg.get("api_key", "")
                if "/v1beta/models/" in api_base:
                    api_base = api_base.split("/v1beta/models/")[0] + "/v1"
                model = "gemini-2.5-flash"
                timeout_val = int(cfg.get("timeout", 120))

                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout_val)
                self.log_signal.emit(f"[分析] API: {api_base}, 模型: {model}, 超时: {timeout_val}s")

                from modules.image_analysis.single_analyzer import step_1_analyze_image
                self.log_signal.emit("[分析] 正在发送 Vision 分析请求...")
                result = step_1_analyze_image(
                    image_source=image_path,
                    client=client,
                    model_name=model,
                    log_callback=lambda msg: self.log_signal.emit(msg),
                    booru_tag_limit=30,
                    timeout_seconds=timeout_val,
                )
                self.log_signal.emit("[分析] Vision 请求完成")
                self.analysis_done.emit(result or {})
            except Exception as exc:
                self.analysis_error.emit(str(exc))

        self._analysis_worker = threading.Thread(target=worker, daemon=True)
        self._analysis_worker.start()

    def _on_analysis_success(self, result: dict) -> None:
        self._set_busy_state(False)
        set_task_status(self.status_label, "success", "分析完成")

        # Save analysis result
        if self.latest_bundle:
            analysis_path = os.path.join(self.latest_bundle.output_dir, f"{self._default_file_prefix()}_analysis.json")
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            append_log_line(self.log_output, f"[分析] 结果已保存: {analysis_path}")

        desc = str(result.get("english_description", "") or "")[:200]
        tags = result.get("pixiv_tags", [])
        if desc:
            append_log_line(self.log_output, f"[分析] 英文描述: {desc}...")
        if tags:
            append_log_line(self.log_output, f"[分析] Pixiv Tags: {', '.join(tags[:12])}")
        append_log_line(self.log_output, "[分析] 分析完成")
        self._send_windows_notification("Image Maker", "采集 → 生图 → 分析 全部完成")

    def _on_analysis_error(self, message: str) -> None:
        self._set_busy_state(False)
        set_task_status(self.status_label, "error", "分析失败")
        append_log_line(self.log_output, f"[分析] 失败: {message}")

    @staticmethod
    def _send_windows_notification(title: str, body: str) -> None:
        """Send a Windows toast notification via PowerShell (stdin pipe)."""
        if os.name != "nt":
            return
        try:
            # $1 = title, $2 = body
            ps_script = r'''
Add-Type -AssemblyName System.Windows.Forms
$balloon = New-Object System.Windows.Forms.NotifyIcon
$balloon.Icon = [System.Drawing.SystemIcons]::Information
$balloon.BalloonTipTitle = "{title}"
$balloon.BalloonTipText = "{body}"
$balloon.Visible = $true
$balloon.ShowBalloonTip(5000)
Start-Sleep -Seconds 7
$balloon.Dispose()
'''
            ps_script = ps_script.replace("{title}", title).replace("{body}", body)
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", "-"],
                input=ps_script, text=True, capture_output=True, timeout=15,
            )
        except Exception:
            pass  # notification is best-effort
