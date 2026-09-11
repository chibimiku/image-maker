# -*- coding: utf-8 -*-
"""
tag 速查组件（资料 Tab 项）
========================
将自定义 booru 标签组数据（如 Danbooru wiki 的 Tag Group:Sexual Positions）
固化为 JSON 数据文件（modules/reference/data/*.json），本组件提供本地速查：
- 树状展示分组层级（体位 / 束缚专有 / 参考分组）
- 支持按 英文 tag / 中文翻译 / 说明 / 别名 实时过滤
- 右侧详情面板展示中文翻译、说明、别名与来源
- 一键复制标签（用于拼 prompt）、一键打开 Danbooru Wiki 来源页
"""

import os
import json
import html
import glob

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox, QLabel,
    QPushButton, QTreeWidget, QTreeWidgetItem, QTextBrowser, QSplitter,
    QApplication,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class TagQuickRefWidget(QWidget):
    """基于本地 JSON 数据文件的 booru 标签速查组件。"""

    def __init__(self, data_dir=None, parent=None):
        super().__init__(parent)
        self.data_dir = data_dir or DATA_DIR
        self.datasets = []          # 加载的每个 JSON 数据集 dict
        self._current_data = None
        self._current_node = None
        self._total_tags = 0
        self.init_ui()
        self.reload_datasets()

    # ------------------------------------------------------------------ UI
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)

        # 顶部：数据源选择 + 搜索
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("数据源:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_changed)
        top_row.addWidget(self.dataset_combo, stretch=1)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "搜索: 英文 tag / 中文翻译 / 说明 / 别名（如: cowgirl / 骑乘）"
        )
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self.apply_filter)
        top_row.addWidget(self.search_input, stretch=2)

        self.reload_btn = QPushButton("重新加载")
        self.reload_btn.setToolTip("重新从 JSON 数据文件加载（可手动编辑 data/*.json 后刷新）")
        self.reload_btn.clicked.connect(self.reload_datasets)
        top_row.addWidget(self.reload_btn)
        layout.addLayout(top_row)

        # 中部：左侧树 + 右侧详情
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(1)
        self.tree.itemSelectionChanged.connect(self.on_selection_changed)
        self.tree.itemDoubleClicked.connect(self.copy_current_tag)
        splitter.addWidget(self.tree)

        detail_widget = QWidget()
        detail_layout = QVBoxLayout()
        detail_layout.setContentsMargins(0, 0, 0, 0)

        btn_row = QHBoxLayout()
        self.copy_btn = QPushButton("复制标签")
        self.copy_btn.setToolTip("复制选中标签（多个标签时以英文逗号分隔），用于拼 prompt")
        self.copy_btn.clicked.connect(self.copy_current_tag)
        self.wiki_btn = QPushButton("打开来源 Wiki")
        self.wiki_btn.setToolTip("在浏览器中打开该标签对应的 Danbooru Wiki 页面")
        self.wiki_btn.clicked.connect(self.open_wiki_page)
        self.copy_btn.setEnabled(False)
        self.wiki_btn.setEnabled(False)
        btn_row.addWidget(self.copy_btn)
        btn_row.addWidget(self.wiki_btn)
        btn_row.addStretch()
        detail_layout.addLayout(btn_row)

        self.detail_browser = QTextBrowser()
        self.detail_browser.setOpenExternalLinks(True)
        self.detail_browser.setMinimumWidth(360)
        detail_layout.addWidget(self.detail_browser, stretch=1)
        detail_widget.setLayout(detail_layout)
        splitter.addWidget(detail_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        # 底部状态
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)
        self.show_hint()

    # ------------------------------------------------------------- 数据加载
    def reload_datasets(self):
        """扫描 data 目录下所有 *.json，作为独立数据源加入下拉框。"""
        self.datasets = []
        current_title = self.dataset_combo.currentData()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        json_files = sorted(glob.glob(os.path.join(self.data_dir, "*.json")))
        select_index = 0
        for idx, path in enumerate(json_files):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[tag速查] 加载数据文件失败 {path}: {e}")
                continue
            data["_file_path"] = path
            self.datasets.append(data)
            label = f"{data.get('title_zh', data.get('title', os.path.basename(path)))} ({data.get('title', os.path.basename(path))})"
            self.dataset_combo.addItem(label, data.get("title", os.path.basename(path)))
            if data.get("title") == current_title:
                select_index = idx
        if self.datasets:
            self.dataset_combo.setCurrentIndex(min(select_index, len(self.datasets) - 1))
        else:
            self._current_data = None
            self.tree.clear()
            self.status_label.setText("未找到数据文件: " + str(self.data_dir))
            self.show_hint()
        self.dataset_combo.blockSignals(False)
        if self.datasets:
            self.on_dataset_changed(self.dataset_combo.currentIndex())

    def on_dataset_changed(self, index):
        if index < 0 or index >= len(self.datasets):
            self._current_data = None
            return
        self._current_data = self.datasets[index]
        self.build_tree()
        self.apply_filter()
        self.tree.setCurrentItem(self.tree.topLevelItem(0))

    # ------------------------------------------------------------- 树构建
    def build_tree(self):
        self.tree.clear()
        self._total_tags = 0
        data = self._current_data
        if not data:
            self.show_hint()
            return
        for group in data.get("groups", []):
            group_item = QTreeWidgetItem([f"{group.get('name', '')} — {group.get('zh', '')}"])
            group_item.setData(0, Qt.ItemDataRole.UserRole, {
                "type": "group",
                "title": group.get("name", ""),
                "zh": group.get("zh", ""),
            })
            for node in group.get("items", []):
                self._add_node(group_item, node)
            self.tree.addTopLevelItem(group_item)
            group_item.setExpanded(True)
        self.status_label.setText(
            f"共 {self._total_tags} 个标签 · {len(data.get('groups', []))} 个分组 · 来源: {data.get('source_url', '')}"
        )

    def _add_node(self, parent_item, node):
        node_type = node.get("type", "tag")
        if node_type == "link":
            label = f"{node.get('title', '')} — {node.get('zh', '')}"
        else:
            tags = node.get("tags") or ([node["tag"]] if node.get("tag") else [])
            tag_text = " / ".join(tags)
            label = f"{tag_text} — {node.get('zh', '')}"
            self._total_tags += len(tags)
        item = QTreeWidgetItem([label])
        item.setData(0, Qt.ItemDataRole.UserRole, node)
        item.setToolTip(0, f"{label}\n双击复制标签")
        parent_item.addChild(item)
        for child in node.get("children", []):
            self._add_node(item, child)

    # ------------------------------------------------------------- 过滤
    def _node_matches(self, node, keyword):
        if not node:
            return False
        if node.get("type") == "link":
            parts = [node.get("title", ""), node.get("zh", ""), node.get("url", "")]
        else:
            parts = list(node.get("tags") or [])
            if not parts and node.get("tag"):
                parts = [node["tag"]]
            parts += [node.get("zh", ""), node.get("desc", "")] + list(node.get("aliases") or [])
        text = " ".join(str(p) for p in parts).lower()
        return keyword in text

    def apply_filter(self):
        keyword = self.search_input.text().strip().lower()

        def walk(item):
            node = item.data(0, Qt.ItemDataRole.UserRole)
            self_ok = bool(keyword) and self._node_matches(node, keyword)
            child_visible = False
            for i in range(item.childCount()):
                if walk(item.child(i)):
                    child_visible = True
            visible = (not keyword) or self_ok or child_visible
            item.setHidden(not visible)
            return visible

        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))

        if keyword:
            self.tree.expandAll()
        else:
            self.tree.collapseAll()
            for i in range(self.tree.topLevelItemCount()):
                self.tree.topLevelItem(i).setExpanded(True)

    # ------------------------------------------------------------- 详情
    def selected_node(self):
        items = self.tree.selectedItems()
        if not items:
            return None
        return items[0].data(0, Qt.ItemDataRole.UserRole)

    def on_selection_changed(self):
        node = self.selected_node()
        self._current_node = node
        is_tag = bool(node) and node.get("type", "tag") != "link"
        self.copy_btn.setEnabled(is_tag)
        self.wiki_btn.setEnabled(bool(node))
        self.render_detail(node)

    def render_detail(self, node):
        if not node:
            self.show_hint()
            return
        if node.get("type") == "link":
            title = html.escape(str(node.get("title", "")))
            zh = html.escape(str(node.get("zh", "")))
            url = html.escape(str(node.get("url", "")))
            self.detail_browser.setHtml(
                f"<h2>{title}</h2>"
                f"<p><b>资料分组:</b> {zh}</p>"
                f"<p><b>地址:</b> <a href='{url}'>{url}</a></p>"
                f"<p style='color:gray'>点击上方链接可在浏览器打开对应分组页面。</p>"
            )
            return

        tags = node.get("tags") or ([node["tag"]] if node.get("tag") else [])
        tag_html = "<br>".join(html.escape(str(t)) for t in tags)
        zh = html.escape(str(node.get("zh", "")))
        desc = html.escape(str(node.get("desc", ""))).replace("\n", "<br>")
        aliases = node.get("aliases") or []
        aliases_html = html.escape("、".join(str(a) for a in aliases)) if aliases else "<span style='color:gray'>无</span>"

        path = []
        items = self.tree.selectedItems()
        if items:
            cur = items[0].parent()
            while cur is not None:
                path.insert(0, cur.text(0))
                cur = cur.parent()
        path_html = " → ".join(html.escape(p) for p in path) if path else ""

        wiki_url = ""
        primary = tags[0] if tags else ""
        if primary:
            from urllib.parse import quote
            wiki_url = "https://danbooru.donmai.us/wiki_pages/" + quote(primary.replace(" ", "_"))

        parts = [f"<h2>{tag_html}</h2>"]
        parts.append(f"<p><b>中文翻译:</b> {zh}</p>")
        if desc:
            parts.append(f"<p><b>说明:</b> {desc}</p>")
        parts.append(f"<p><b>别名:</b> {aliases_html}</p>")
        if path_html:
            parts.append(f"<p><b>所属分类:</b> {path_html}</p>")
        if wiki_url:
            parts.append(f"<p><b>Wiki 页面:</b> <a href='{wiki_url}'>{wiki_url}</a></p>")
        parts.append(
            "<p style='color:gray'>可点击左上方「复制标签」把标签复制到剪贴板，用于拼接生图提示词。</p>"
        )
        self.detail_browser.setHtml("".join(parts))

    def show_hint(self):
        self.detail_browser.setHtml(
            "<p style='color:gray'>请在左侧选择条目查看详情。</p>"
            "<p style='color:gray'>顶部搜索框支持按 英文 tag / 中文翻译 / 说明 / 别名 过滤。</p>"
        )

    # ------------------------------------------------------------- 动作
    def copy_current_tag(self):
        node = self._current_node if self._current_node is not None else self.selected_node()
        if not node or node.get("type") == "link":
            return
        tags = node.get("tags") or ([node["tag"]] if node.get("tag") else [])
        if not tags:
            return
        text = ", ".join(tags)
        QApplication.clipboard().setText(text)
        self.status_label.setText(f"已复制: {text}")

    def open_wiki_page(self):
        node = self._current_node if self._current_node is not None else self.selected_node()
        if not node:
            return
        if node.get("type") == "link":
            url = node.get("url", "")
        else:
            tags = node.get("tags") or ([node["tag"]] if node.get("tag") else [])
            if not tags:
                return
            from urllib.parse import quote
            url = "https://danbooru.donmai.us/wiki_pages/" + quote(str(tags[0]).replace(" ", "_"))
        if not url:
            return
        QDesktopServices.openUrl(QUrl(url))
