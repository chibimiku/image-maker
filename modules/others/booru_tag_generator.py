import csv
import os
import random
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QPushButton,
    QTextEdit,
    QMessageBox,
    QGroupBox,
    QGridLayout,
    QCheckBox,
    QComboBox,
)


class BooruTagGeneratorWidget(QWidget):
    CATEGORY_CONFIGS = [
        {
            "key": "sexual_positions",
            "label": "sexual positions",
            "hints": [
                "position",
                "sex_from",
                "mating_press",
                "doggystyle",
                "doggy_style",
                "missionary",
                "cowgirl",
                "reverse_cowgirl",
                "spooning",
                "standing_sex",
                "sixty_nine",
                "_69",
                "full_nelson",
                "from_behind",
                "anal_position",
            ],
            "fallback": [
                "missionary",
                "doggystyle",
                "cowgirl_position",
                "reverse_cowgirl_position",
                "sex_from_behind",
            ],
            "default_checked": True,
            "default_count": 1,
        },
        {
            "key": "decorations",
            "label": "装饰词",
            "hints": [
                "cum",
                "creampie",
                "ejaculation",
                "orgasm",
                "after_sex",
                "dripping",
                "drool",
                "saliva",
                "sweat",
                "messy",
                "panting",
                "wet",
                "body_fluids",
            ],
            "fallback": [
                "cum_inside",
                "creampie",
                "cumdrip",
                "cum_on_body",
                "facial",
            ],
            "default_checked": True,
            "default_count": 1,
        },
        {
            "key": "expressions",
            "label": "表情",
            "hints": [
                "smile",
                "blush",
                "embarrassed",
                "aroused",
                "happy",
                "laugh",
                "wink",
                "closed_eyes",
                "open_mouth",
                "tears",
                "crying",
                "grin",
                "surprised",
            ],
            "fallback": [
                "blush",
                "smile",
                "open_mouth",
                "embarrassed",
                "aroused",
            ],
            "default_checked": True,
            "default_count": 1,
        },
        {
            "key": "actions",
            "label": "其他动作",
            "hints": [
                "holding",
                "grabbing",
                "touching",
                "licking",
                "kissing",
                "hugging",
                "spreading",
                "pressing",
                "squeezing",
                "thrusting",
                "riding",
                "kneeling",
                "straddling",
                "pov",
            ],
            "fallback": [
                "kissing",
                "holding",
                "licking",
                "touching",
                "hugging",
            ],
            "default_checked": True,
            "default_count": 1,
        },
        {
            "key": "camera_composition",
            "label": "镜头构图",
            "hints": [
                "looking_at_viewer",
                "pov",
                "close_up",
                "upper_body",
                "full_body",
                "from_above",
                "from_below",
                "dutch_angle",
                "depth_of_field",
            ],
            "fallback": [
                "looking_at_viewer",
                "pov",
                "close_up",
                "from_above",
            ],
            "default_checked": False,
            "default_count": 1,
        },
        {
            "key": "clothing_props",
            "label": "服饰道具",
            "hints": [
                "lingerie",
                "stockings",
                "garter",
                "panties",
                "bra",
                "shirt",
                "skirt",
                "gloves",
                "collar",
                "choker",
                "ribbon",
                "hair_ornament",
            ],
            "fallback": [
                "lingerie",
                "stockings",
                "garter_belt",
                "shirt_lift",
                "hair_ornament",
            ],
            "default_checked": False,
            "default_count": 1,
        },
        {
            "key": "insect",
            "label": "insect",
            "hints": [
                "insect",
                "arthropod",
                "bee",
                "wasp",
                "moth",
                "butterfly",
                "beetle",
                "spider",
                "ant",
                "mantis",
                "dragonfly",
                "mosquito",
                "cockroach",
                "centipede",
                "scorpion",
            ],
            "fallback": [
                "insect",
                "bee",
                "butterfly",
                "moth",
                "spider",
            ],
            "default_checked": False,
            "default_count": 1,
        },
    ]

    def __init__(self):
        super().__init__()
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.csv_path = os.path.join(
            base_dir,
            "data",
            "tags",
            "danbooru_e621_merged.csv",
        )
        self.translation_path = os.path.join(
            base_dir,
            "data",
            "tags",
            "danbooru_e621_merged_zh.csv",
        )
        self.category_pools = {}
        self.category_widgets = {}
        self.zh_map = {}
        self._init_ui()
        self._load_translation_map()
        self._load_tag_pools()

    def _init_ui(self):
        layout = QVBoxLayout()
        form = QFormLayout()

        self.total_count_spin = QSpinBox()
        self.total_count_spin.setRange(1, 5000)
        self.total_count_spin.setValue(20)
        form.addRow("生成条数:", self.total_count_spin)
        layout.addLayout(form)

        category_group = QGroupBox("按分类组合（勾选并设置每类数量）")
        category_grid = QGridLayout()
        category_grid.addWidget(QLabel("启用"), 0, 0)
        category_grid.addWidget(QLabel("分类"), 0, 1)
        category_grid.addWidget(QLabel("每行数量"), 0, 2)
        category_grid.addWidget(QLabel("可用词数"), 0, 3)

        for idx, conf in enumerate(self.CATEGORY_CONFIGS, start=1):
            check = QCheckBox()
            check.setChecked(bool(conf.get("default_checked", False)))
            count_spin = QSpinBox()
            count_spin.setRange(0, 10)
            count_spin.setValue(int(conf.get("default_count", 1)))
            if not check.isChecked():
                count_spin.setEnabled(False)
            check.toggled.connect(count_spin.setEnabled)
            pool_label = QLabel("0")

            category_grid.addWidget(check, idx, 0)
            category_grid.addWidget(QLabel(str(conf["label"])), idx, 1)
            category_grid.addWidget(count_spin, idx, 2)
            category_grid.addWidget(pool_label, idx, 3)
            self.category_widgets[conf["key"]] = {
                "check": check,
                "count_spin": count_spin,
                "pool_label": pool_label,
            }

        category_group.setLayout(category_grid)
        layout.addWidget(category_group)

        insect_group = QGroupBox("insect 固定选择")
        insect_form = QFormLayout()
        self.insect_mode_combo = QComboBox()
        mode_items = ["fixed(固定)", "random(随机)"]
        self.insect_mode_combo.addItems(sorted(mode_items, key=lambda x: x.lower()))
        insect_form.addRow("模式:", self.insect_mode_combo)

        self.insect_fixed_combo = QComboBox()
        self.insect_fixed_combo.setEnabled(False)
        insect_form.addRow("固定 insect:", self.insect_fixed_combo)
        insect_group.setLayout(insect_form)
        layout.addWidget(insect_group)
        self.insect_mode_combo.currentTextChanged.connect(self._on_insect_mode_changed)

        self.info_label = QLabel()
        layout.addWidget(self.info_label)

        action_row = QHBoxLayout()
        self.reload_btn = QPushButton("重载词库")
        self.reload_btn.clicked.connect(self._on_reload_clicked)
        action_row.addWidget(self.reload_btn)

        self.generate_btn = QPushButton("生成组合")
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        action_row.addWidget(self.generate_btn)
        layout.addLayout(action_row)

        self.output_text = QTextEdit()
        self.output_text.setPlaceholderText("生成结果会显示在这里，每行一套组合。")
        layout.addWidget(self.output_text)
        self.setLayout(layout)

    def _load_translation_map(self):
        self.zh_map = {}
        if not os.path.isfile(self.translation_path):
            return
        try:
            with open(self.translation_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    en_tag = str(row[0]).strip().lower()
                    zh_tag = str(row[1]).strip()
                    if not en_tag or not zh_tag:
                        continue
                    self.zh_map[en_tag] = zh_tag
        except Exception:
            self.zh_map = {}

    def _load_tag_pools(self):
        if not os.path.isfile(self.csv_path):
            self.category_pools = {
                conf["key"]: list(conf.get("fallback", []))
                for conf in self.CATEGORY_CONFIGS
            }
            self._refresh_info_label(
                f"未找到词库文件，已启用内置词库。路径: {self.csv_path}"
            )
            self._refresh_category_pool_labels()
            self._refresh_insect_combo_items()
            return

        pools = {conf["key"]: set() for conf in self.CATEGORY_CONFIGS}
        try:
            with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    tag = str(row[0]).strip().lower()
                    if not tag:
                        continue
                    for conf in self.CATEGORY_CONFIGS:
                        if self._matches_category(tag, conf):
                            pools[conf["key"]].add(tag)
        except Exception as e:
            self.category_pools = {
                conf["key"]: list(conf.get("fallback", []))
                for conf in self.CATEGORY_CONFIGS
            }
            self._refresh_info_label(f"读取词库失败，已启用内置词库。错误: {e}")
            self._refresh_category_pool_labels()
            self._refresh_insect_combo_items()
            return

        normalized = {}
        for conf in self.CATEGORY_CONFIGS:
            key = conf["key"]
            values = pools.get(key) or set()
            if not values:
                values = set(conf.get("fallback", []))
            normalized[key] = sorted(values)
        self.category_pools = normalized
        self._refresh_info_label("词库加载完成。")
        self._refresh_category_pool_labels()
        self._refresh_insect_combo_items()

    def _refresh_info_label(self, prefix):
        enabled_count = 0
        for conf in self.CATEGORY_CONFIGS:
            key = conf["key"]
            widgets = self.category_widgets.get(key, {})
            check = widgets.get("check")
            if check is not None and check.isChecked():
                enabled_count += 1
        self.info_label.setText(
            f"{prefix} 已加载分类数: {len(self.CATEGORY_CONFIGS)}，当前启用: {enabled_count}"
        )

    def _refresh_category_pool_labels(self):
        for conf in self.CATEGORY_CONFIGS:
            key = conf["key"]
            size = len(self.category_pools.get(key, []))
            widgets = self.category_widgets.get(key, {})
            pool_label = widgets.get("pool_label")
            if pool_label is not None:
                pool_label.setText(str(size))

    def _matches_category(self, tag, conf):
        for token in conf.get("hints", []):
            if token in tag:
                return True
        return False

    def _on_insect_mode_changed(self, value):
        self.insect_fixed_combo.setEnabled(value.startswith("fixed"))

    def _refresh_insect_combo_items(self):
        insect_pool = sorted(self.category_pools.get("insect", []), key=lambda x: x.lower())
        display_rows = []
        for tag in insect_pool:
            zh = self.zh_map.get(tag, "")
            if zh:
                display_rows.append(f"{tag} ({zh})")
            else:
                display_rows.append(tag)
        self.insect_fixed_combo.blockSignals(True)
        self.insect_fixed_combo.clear()
        self.insect_fixed_combo.addItems(display_rows)
        self.insect_fixed_combo.blockSignals(False)

    def _sample_tags(self, pool, count):
        if count <= 0 or not pool:
            return []
        if count <= len(pool):
            return random.sample(pool, count)
        picked = random.sample(pool, len(pool))
        for _ in range(count - len(pool)):
            picked.append(random.choice(pool))
        return picked

    def _on_reload_clicked(self):
        self._load_translation_map()
        self._load_tag_pools()
        QMessageBox.information(self, "完成", "词库已重载。")

    def _on_generate_clicked(self):
        total = int(self.total_count_spin.value())
        enabled_categories = []
        for conf in self.CATEGORY_CONFIGS:
            key = conf["key"]
            widgets = self.category_widgets.get(key, {})
            check = widgets.get("check")
            count_spin = widgets.get("count_spin")
            if check is None or count_spin is None:
                continue
            if not check.isChecked():
                continue
            count = int(count_spin.value())
            if count <= 0:
                continue
            enabled_categories.append((key, count))

        if not enabled_categories:
            QMessageBox.warning(self, "提示", "请至少启用一个分类，并设置每行数量大于 0。")
            return

        lines = []
        for _ in range(total):
            tags = []
            for key, count in enabled_categories:
                if key == "insect" and self.insect_mode_combo.currentText().startswith("fixed"):
                    fixed_text = self.insect_fixed_combo.currentText().strip()
                    fixed_tag = fixed_text.split(" (", 1)[0].strip() if fixed_text else ""
                    if fixed_tag:
                        tags.extend([fixed_tag] * count)
                        continue
                pool = self.category_pools.get(key, [])
                tags.extend(self._sample_tags(pool, count))
            random.shuffle(tags)
            lines.append(", ".join(tags))

        self.output_text.setPlainText("\n".join(lines))
        self._refresh_info_label("生成完成。")
