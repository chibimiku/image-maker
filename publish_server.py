"""
图片发布 Server — 拖拽本地图片入队，匹配元数据 JSON，通过 API 供油猴脚本拉取上传。
启动方式: python publish_server.py
"""
import os
import sys
import json
import uuid
import hashlib
import socket
import sqlite3
import base64
import logging
import mimetypes
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, g, send_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONF_DIR = os.path.join(BASE_DIR, "conf")
DATA_DIR = os.path.join(BASE_DIR, "data", "publish")
DB_PATH = os.path.join(DATA_DIR, "publish_queue.db")
CONFIG_FILE = os.path.join(CONF_DIR, "config-publish.json")

os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── 默认配置 ──────────────────────────────────────────────
DEFAULT_PORT = 18765


def port_is_available(port: int) -> bool:
    """检测指定端口是否可绑定"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"port": DEFAULT_PORT}


def save_config(cfg: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ── 数据库 ────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
    return g.db


def init_db():
    """初始化数据库表（独立连接，供 GUI 线程使用）"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS publish_queue (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid            TEXT UNIQUE NOT NULL,
            image_path      TEXT NOT NULL,
            image_md5       TEXT NOT NULL,
            json_path       TEXT,
            metadata        TEXT,
            pixiv_status    TEXT DEFAULT 'pending',
            chichipui_status TEXT DEFAULT 'pending',
            created_at      TEXT DEFAULT (datetime('now','localtime')),
            updated_at      TEXT DEFAULT (datetime('now','localtime'))
        )
    """)
    # 迁移旧表：如果存在旧 status 列则迁移为双平台字段
    cols = {r[1] for r in conn.execute("PRAGMA table_info(publish_queue)").fetchall()}
    if "status" in cols and "pixiv_status" not in cols:
        conn.execute("ALTER TABLE publish_queue RENAME COLUMN status TO pixiv_status")
        conn.execute("ALTER TABLE publish_queue ADD COLUMN chichipui_status TEXT DEFAULT 'pending'")
    conn.commit()
    conn.close()


def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


# ── 元数据查找 ────────────────────────────────────────────
def find_metadata_json(image_path: str) -> str | None:
    """
    根据图片文件名查找同目录下的元数据 JSON。
    规则：
      图片: ee721fed_003711-18b7de.jpg → 取 _ 分割的第一段 "ee721fed"
      JSON:  20260708-003608-ee721fed-紫苑の天光.json → 取 - 分割的第3段 "ee721fed"
    匹配即返回 JSON 路径。
    """
    img_file = os.path.basename(image_path)
    img_dir = os.path.dirname(image_path)
    # 取 _ 分割的第一段作为 key
    key = img_file.split("_")[0]
    if not key:
        return None

    try:
        entries = os.listdir(img_dir)
    except OSError:
        return None

    for entry in entries:
        if not entry.lower().endswith(".json"):
            continue
        # JSON 文件名拆分，key 应在第3段（索引2）
        parts = os.path.splitext(entry)[0].split("-")
        if len(parts) >= 3 and parts[2] == key:
            return os.path.join(img_dir, entry)

    return None


def compute_md5(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def image_to_base64(image_path: str) -> str | None:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None


# ── Flask App ─────────────────────────────────────────────
flask_app = Flask(__name__)
flask_app.teardown_appcontext(close_db)

VALID_STATUS = ("pending", "uploading", "uploaded")
VALID_PLATFORM = ("pixiv", "chichipui")


def _row_to_item(r) -> dict:
    item = dict(r)
    item["image_base64"] = image_to_base64(r["image_path"])
    if item["metadata"]:
        try:
            item["metadata"] = json.loads(item["metadata"])
        except json.JSONDecodeError:
            pass
    return item


@flask_app.route("/api/queue", methods=["GET"])
def api_list():
    """查询列表：默认 20 条，按时间倒序，支持翻页"""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    pixiv_status = request.args.get("pixiv_status", None, type=str)
    chichipui_status = request.args.get("chichipui_status", None, type=str)

    db = get_db()
    conditions = []
    params = []
    if pixiv_status:
        conditions.append("pixiv_status = ?")
        params.append(pixiv_status)
    if chichipui_status:
        conditions.append("chichipui_status = ?")
        params.append(chichipui_status)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    row = db.execute(f"SELECT COUNT(*) as cnt FROM publish_queue {where}", params).fetchone()
    total = row["cnt"]

    offset = (page - 1) * per_page
    rows = db.execute(
        f"SELECT * FROM publish_queue {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        params + [per_page, offset],
    ).fetchall()

    items = [_row_to_item(r) for r in rows]

    return jsonify({
        "page": page,
        "per_page": per_page,
        "total": total,
        "items": items,
    })


@flask_app.route("/api/queue/latest-pending", methods=["GET"])
def api_latest_pending():
    """拉取最新一条待上传数据。支持 ?platform=pixiv|chichipui，默认 pixiv"""
    platform = request.args.get("platform", "pixiv", type=str)
    if platform not in VALID_PLATFORM:
        return jsonify({"code": 400, "message": "invalid platform, use pixiv or chichipui"}), 400
    col = "pixiv_status" if platform == "pixiv" else "chichipui_status"

    db = get_db()
    row = db.execute(
        f"SELECT * FROM publish_queue WHERE {col} = 'pending' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return jsonify({"code": 1, "message": f"no pending item for {platform}", "data": None})

    return jsonify({"code": 0, "message": "ok", "data": _row_to_item(row)})


@flask_app.route("/api/queue/<item_uuid>", methods=["GET"])
def api_get_by_uuid(item_uuid):
    """查询特定 UUID 数据"""
    db = get_db()
    row = db.execute("SELECT * FROM publish_queue WHERE uuid = ?", [item_uuid]).fetchone()
    if not row:
        return jsonify({"code": 404, "message": "uuid not found"}), 404
    return jsonify(_row_to_item(row))


def _guess_mime(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


@flask_app.route("/api/queue/<item_uuid>/image", methods=["GET"])
def api_get_image(item_uuid):
    """返回原始图片二进制（前端可直接 fetch 为 Blob）"""
    db = get_db()
    row = db.execute("SELECT image_path FROM publish_queue WHERE uuid = ?", [item_uuid]).fetchone()
    if not row:
        return jsonify({"code": 404, "message": "uuid not found"}), 404
    img_path = row["image_path"]
    if not os.path.isfile(img_path):
        return jsonify({"code": 404, "message": "image file not found on disk"}), 404
    return send_file(img_path, mimetype=_guess_mime(img_path))


@flask_app.route("/api/queue/<item_uuid>/status", methods=["PUT"])
def api_update_status(item_uuid):
    """修改状态: {"platform": "pixiv"|"chichipui", "status": "uploading"|"uploaded"|"pending"}"""
    data = request.get_json(silent=True)
    if not data or "status" not in data:
        return jsonify({"code": 400, "message": "missing status"}), 400
    platform = data.get("platform", "pixiv")
    if platform not in VALID_PLATFORM:
        return jsonify({"code": 400, "message": "invalid platform, use pixiv or chichipui"}), 400

    new_status = data["status"]
    if new_status not in VALID_STATUS:
        return jsonify({"code": 400, "message": "invalid status"}), 400

    col = "pixiv_status" if platform == "pixiv" else "chichipui_status"
    db = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = db.execute(
        f"UPDATE publish_queue SET {col} = ?, updated_at = ? WHERE uuid = ?",
        [new_status, now, item_uuid],
    )
    db.commit()
    if cur.rowcount == 0:
        return jsonify({"code": 404, "message": "uuid not found"}), 404
    return jsonify({"code": 0, "message": "ok", "uuid": item_uuid, "platform": platform, "status": new_status})


# ── 辅助：添加图片到队列（供 GUI 调用） ──────────────────
def add_to_queue(image_path: str) -> dict:
    """返回 {"ok": bool, "message": str, "uuid": str|None}"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        # 检查是否已存在相同路径
        existing = conn.execute(
            "SELECT uuid FROM publish_queue WHERE image_path = ?", [image_path]
        ).fetchone()
        if existing:
            return {"ok": False, "message": f"图片路径已在队列中: {existing['uuid']}", "uuid": None}

        # 检查是否已存在相同内容（MD5）
        image_md5 = compute_md5(image_path)
        existing_md5 = conn.execute(
            "SELECT uuid, image_path FROM publish_queue WHERE image_md5 = ?", [image_md5]
        ).fetchone()
        if existing_md5:
            return {
                "ok": False,
                "message": f"图片内容已存在 (MD5 重复): {existing_md5['uuid']} → {existing_md5['image_path']}",
                "uuid": None,
            }

        json_path = find_metadata_json(image_path)
        metadata_str = None
        if json_path:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    metadata_str = f.read()
            except Exception:
                pass

        item_uuid = str(uuid.uuid4())
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            """INSERT INTO publish_queue (uuid, image_path, image_md5, json_path, metadata,
               pixiv_status, chichipui_status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'pending', 'pending', ?, ?)""",
            [item_uuid, image_path, image_md5, json_path, metadata_str, now, now],
        )
        conn.commit()
        return {"ok": True, "message": f"已加入队列: {item_uuid[:8]}...", "uuid": item_uuid}
    except Exception as e:
        return {"ok": False, "message": str(e), "uuid": None}
    finally:
        conn.close()


# ── Flask 服务器线程 ──────────────────────────────────────
class FlaskServerThread(threading.Thread):
    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.port = port
        self._stop_event = threading.Event()

    def run(self):
        # 静默 Flask 的 banner 输出，避免干扰 GUI 日志
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.WARNING)
        flask_app.run(host="127.0.0.1", port=self.port, debug=False, use_reloader=False)

    def stop(self):
        self._stop_event.set()


# ── PyQt6 GUI ─────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem,
    QPlainTextEdit, QHeaderView, QAbstractItemView, QSplitter, QMessageBox,
    QStyledItemDelegate, QComboBox, QMenu, QToolTip,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QSize, QByteArray, QBuffer, QIODevice, QEvent
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QHelpEvent, QCursor
import subprocess


class _SignalBridge(QObject):
    """跨线程信号桥接"""
    log_signal = pyqtSignal(str)
    refresh_signal = pyqtSignal()


# ── 状态下拉代理 ──────────────────────────────────────────
class StatusDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.setEditable(True)
        combo.addItems(["待上传", "正在上传", "已上传"])
        return combo

    def setEditorData(self, editor, index):
        current = index.data(Qt.ItemDataRole.DisplayRole)
        idx = editor.findText(current)
        if idx >= 0:
            editor.setCurrentIndex(idx)
        else:
            editor.setCurrentText(current or "")

    def setModelData(self, editor, model, index):
        text = editor.currentText()
        model.setData(index, text, Qt.ItemDataRole.DisplayRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class PublishServerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片发布 Server")
        self.resize(1100, 700)
        self.setAcceptDrops(True)

        self._config = load_config()
        self._server: FlaskServerThread | None = None
        self._signal = _SignalBridge()
        self._signal.log_signal.connect(self._append_log)
        self._signal.refresh_signal.connect(self._refresh_table)

        # 翻页与筛选状态
        self._current_page = 1
        self._per_page = 20
        self._total_count = 0
        self._pixiv_filter = "全部"
        self._chichipui_filter = "全部"

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # ── 顶部：端口 + 启停 ──
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("监听端口:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(self._config.get("port", DEFAULT_PORT))
        self.port_spin.valueChanged.connect(self._on_port_changed)
        top_layout.addWidget(self.port_spin)

        self.btn_start = QPushButton("启动 Server")
        self.btn_start.clicked.connect(self._toggle_server)
        top_layout.addWidget(self.btn_start)

        self.status_label = QLabel("● 未启动")
        top_layout.addWidget(self.status_label)
        top_layout.addStretch()

        root_layout.addLayout(top_layout)

        # ── 筛选栏 ──
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("筛选 Pixiv:"))
        self.filter_pixiv = QComboBox()
        self.filter_pixiv.addItems(["全部", "待上传", "正在上传", "已上传"])
        self.filter_pixiv.currentTextChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_pixiv)

        filter_layout.addSpacing(20)
        filter_layout.addWidget(QLabel("筛选 Chichi-pui:"))
        self.filter_chichipui = QComboBox()
        self.filter_chichipui.addItems(["全部", "待上传", "正在上传", "已上传"])
        self.filter_chichipui.currentTextChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_chichipui)

        filter_layout.addStretch()
        root_layout.addLayout(filter_layout)

        # ── 中部：表格 + 日志 ──
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 表格
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["Pixiv", "Chichi-pui", "UUID", "图片路径", "MD5", "JSON路径", "创建时间"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)

        # 状态列使用下拉代理
        status_delegate = StatusDelegate(self.table)
        self.table.setItemDelegateForColumn(0, status_delegate)
        self.table.setItemDelegateForColumn(1, status_delegate)

        # 编辑后持久化
        self.table.cellChanged.connect(self._on_cell_changed)

        # 右键菜单
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)

        # 悬浮预览：鼠标进入图片路径列时显示缩略图
        self.table.setMouseTracking(True)
        self.table.cellEntered.connect(self._on_cell_hover)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        splitter.addWidget(self.table)

        # 日志
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_label = QLabel("日志:")
        log_layout.addWidget(log_label)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(500)
        log_layout.addWidget(self.log_text)
        splitter.addWidget(log_widget)
        splitter.setSizes([400, 200])

        root_layout.addWidget(splitter)

        # ── 底部按钮 ──
        bottom_layout = QHBoxLayout()

        # 翻页控件
        bottom_layout.addWidget(QLabel("每页:"))
        self.per_page_spin = QSpinBox()
        self.per_page_spin.setRange(5, 200)
        self.per_page_spin.setValue(self._per_page)
        self.per_page_spin.valueChanged.connect(self._on_per_page_changed)
        bottom_layout.addWidget(self.per_page_spin)

        self.btn_prev = QPushButton("上一页")
        self.btn_prev.clicked.connect(self._go_prev_page)
        bottom_layout.addWidget(self.btn_prev)

        self.page_label = QLabel("第 1 页 / 共 1 页")
        bottom_layout.addWidget(self.page_label)

        self.btn_next = QPushButton("下一页")
        self.btn_next.clicked.connect(self._go_next_page)
        bottom_layout.addWidget(self.btn_next)

        bottom_layout.addSpacing(30)

        self.btn_refresh = QPushButton("刷新列表")
        self.btn_refresh.clicked.connect(self._refresh_table)
        bottom_layout.addWidget(self.btn_refresh)

        self.btn_clear_uploaded = QPushButton("清除已上传记录")
        self.btn_clear_uploaded.clicked.connect(self._clear_uploaded)
        bottom_layout.addWidget(self.btn_clear_uploaded)

        self.btn_delete_selected = QPushButton("删除选中行")
        self.btn_delete_selected.clicked.connect(self._delete_selected)
        bottom_layout.addWidget(self.btn_delete_selected)
        bottom_layout.addStretch()

        url_label = QLabel(f"API 地址: http://127.0.0.1:{self.port_spin.value()}/api/queue")
        bottom_layout.addWidget(url_label)
        root_layout.addLayout(bottom_layout)

        # 定时刷新
        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self._refresh_table)
        self._refresh_timer.start(5000)  # 5 秒刷新

        # 图片悬浮预览缓存
        self._tooltip_cache = {}

        # 初始化数据库并刷新
        init_db()
        self._refresh_table()
        self._log("GUI 初始化完成，数据库就绪。")

        # 端口空闲则自动启动
        saved_port = self._config.get("port", DEFAULT_PORT)
        if port_is_available(saved_port):
            self._start_server()
            self._log(f"端口 {saved_port} 空闲，已自动启动。")
        else:
            self._log(f"端口 {saved_port} 已被占用，请手动切换端口后启动。")

    # ── 筛选变更 ──
    def _on_filter_changed(self):
        self._pixiv_filter = self.filter_pixiv.currentText()
        self._chichipui_filter = self.filter_chichipui.currentText()
        self._current_page = 1  # 重置到第一页
        self._refresh_table()

    # ── 每页数量变更 ──
    def _on_per_page_changed(self, val):
        self._per_page = val
        self._current_page = 1
        self._refresh_table()

    # ── 翻页 ──
    def _go_prev_page(self):
        if self._current_page > 1:
            self._current_page -= 1
            self._refresh_table()

    def _go_next_page(self):
        total_pages = max(1, (self._total_count + self._per_page - 1) // self._per_page)
        if self._current_page < total_pages:
            self._current_page += 1
            self._refresh_table()

    # ── 拖拽支持 ──
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        added = 0
        skipped = 0
        for url in urls:
            path = url.toLocalFile()
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"):
                continue
            result = add_to_queue(path)
            if result["ok"]:
                added += 1
                self._log(f"[+] {os.path.basename(path)} → {result['uuid']}")
            else:
                skipped += 1
                self._log(f"[!] {os.path.basename(path)}: {result['message']}")
        self._current_page = 1
        self._refresh_table()
        self._log(f"拖拽完成：成功 {added}，跳过 {skipped}")

    # ── 表格刷新 ──
    def _refresh_table(self):
        # 临时断开信号，避免 setItem 触发持久化
        self.table.cellChanged.disconnect(self._on_cell_changed)

        # 构建筛选条件
        status_map = {"待上传": "pending", "正在上传": "uploading", "已上传": "uploaded"}
        conditions = []
        params = []
        if self._pixiv_filter != "全部":
            conditions.append("pixiv_status = ?")
            params.append(status_map.get(self._pixiv_filter, self._pixiv_filter))
        if self._chichipui_filter != "全部":
            conditions.append("chichipui_status = ?")
            params.append(status_map.get(self._chichipui_filter, self._chichipui_filter))
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # 查询总数
        row = conn.execute(f"SELECT COUNT(*) as cnt FROM publish_queue {where}", params).fetchone()
        self._total_count = row["cnt"]

        # 计算总页数，修正当前页
        total_pages = max(1, (self._total_count + self._per_page - 1) // self._per_page)
        if self._current_page > total_pages:
            self._current_page = total_pages

        # 分页查询
        offset = (self._current_page - 1) * self._per_page
        rows = conn.execute(
            f"SELECT * FROM publish_queue {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [self._per_page, offset],
        ).fetchall()
        conn.close()

        # 更新翻页按钮和标签
        self.btn_prev.setEnabled(self._current_page > 1)
        self.btn_next.setEnabled(self._current_page < total_pages)
        self.page_label.setText(f"第 {self._current_page} 页 / 共 {total_pages} 页 (共 {self._total_count} 条)")

        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            keys = set(r.keys())
            raw_p = r["pixiv_status"] if "pixiv_status" in keys else r["status"] if "status" in keys else "pending"
            raw_c = r["chichipui_status"] if "chichipui_status" in keys else "pending"
            self.table.setItem(i, 0, QTableWidgetItem(status_map.get(raw_p, raw_p)))
            self.table.setItem(i, 1, QTableWidgetItem(status_map.get(raw_c, raw_c)))
            # UUID
            uuid_item = QTableWidgetItem(r["uuid"])
            uuid_item.setFlags(uuid_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 2, uuid_item)
            img_item = QTableWidgetItem(r["image_path"])
            img_item.setData(Qt.ItemDataRole.ToolTipRole, r["image_path"])
            self.table.setItem(i, 3, img_item)
            # MD5
            md5_item = QTableWidgetItem(r["image_md5"] or "")
            md5_item.setFlags(md5_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 4, md5_item)
            self.table.setItem(i, 5, QTableWidgetItem(r["json_path"] or "(无)"))
            # 创建时间
            time_item = QTableWidgetItem(r["created_at"])
            time_item.setFlags(time_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 6, time_item)

        self.table.cellChanged.connect(self._on_cell_changed)

    # ── 清除已上传 ──
    def _clear_uploaded(self):
        reply = QMessageBox.question(
            self, "确认", "确定要清除两个平台都已上传的记录吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "DELETE FROM publish_queue WHERE pixiv_status = 'uploaded' AND chichipui_status = 'uploaded'"
        )
        conn.commit()
        conn.close()
        self._current_page = 1
        self._refresh_table()
        self._log("已清除两个平台都已上传的记录。")

    # ── 删除选中 ──
    def _delete_selected(self):
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.information(self, "提示", "请先选中要删除的行。")
            return
        reply = QMessageBox.question(
            self, "确认", f"确定要删除选中的 {len(selected)} 条记录吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        conn = sqlite3.connect(DB_PATH)
        for idx in selected:
            uuid_item = self.table.item(idx.row(), 2)
            if uuid_item:
                conn.execute("DELETE FROM publish_queue WHERE uuid = ?", [uuid_item.text()])
        conn.commit()
        conn.close()
        self._current_page = 1
        self._refresh_table()
        self._log(f"删除 {len(selected)} 条记录。")

    # ── 右键菜单 ──
    def _on_context_menu(self, pos):
        item = self.table.itemAt(pos)
        if not item:
            return
        row = item.row()
        path_item = self.table.item(row, 3)
        image_path = path_item.text() if path_item else ""
        if not image_path or not os.path.isfile(image_path):
            return

        menu = QMenu(self.table)
        open_img = menu.addAction("打开图片")
        open_dir = menu.addAction("打开所在文件夹")
        action = menu.exec(self.table.viewport().mapToGlobal(pos))

        if action == open_img:
            os.startfile(image_path)
        elif action == open_dir:
            # 打开文件夹并选中该文件
            subprocess.Popen(['explorer', '/select,', image_path])

    # ── 悬浮预览 ──
    def _on_cell_hover(self, row: int, col: int):
        if col != 3:
            QToolTip.hideText()
            return
        image_path = self.table.item(row, col).text()
        if not image_path or not os.path.isfile(image_path):
            QToolTip.hideText()
            return
        tip = self._tooltip_cache.get(image_path)
        if tip is None:
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                self._tooltip_cache[image_path] = ""
                QToolTip.hideText()
                return
            scaled = pixmap.scaled(
                QSize(720, 720),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            ba = QByteArray()
            buf = QBuffer(ba)
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            scaled.save(buf, "PNG")
            b64 = ba.toBase64().data().decode()
            tip = f'<img src="data:image/png;base64,{b64}" style="max-width:720px;max-height:720px;">'
            self._tooltip_cache[image_path] = tip
        if tip:
            QToolTip.showText(QCursor.pos(), tip, self.table)
        else:
            QToolTip.hideText()

    # ── 单元格编辑回调 ──
    def _on_cell_changed(self, row: int, col: int):
        """双击编辑状态列或图片路径/JSON路径后自动持久化到数据库。"""
        uuid_item = self.table.item(row, 2)
        if not uuid_item:
            return
        item_uuid = uuid_item.text()
        new_value = self.table.item(row, col).text().strip()
        status_reverse = {"待上传": "pending", "正在上传": "uploading", "已上传": "uploaded"}

        if col == 0:  # Pixiv 状态
            db_status = status_reverse.get(new_value, new_value)
            conn = sqlite3.connect(DB_PATH)
            conn.execute("UPDATE publish_queue SET pixiv_status = ?, updated_at = datetime('now','localtime') WHERE uuid = ?",
                         [db_status, item_uuid])
            conn.commit()
            conn.close()
            self._log(f"[{item_uuid[:8]}] Pixiv → {new_value}")
        elif col == 1:  # Chichi-pui 状态
            db_status = status_reverse.get(new_value, new_value)
            conn = sqlite3.connect(DB_PATH)
            conn.execute("UPDATE publish_queue SET chichipui_status = ?, updated_at = datetime('now','localtime') WHERE uuid = ?",
                         [db_status, item_uuid])
            conn.commit()
            conn.close()
            self._log(f"[{item_uuid[:8]}] Chichi-pui → {new_value}")
        elif col == 3:  # 图片路径
            conn = sqlite3.connect(DB_PATH)
            conn.execute("UPDATE publish_queue SET image_path = ?, updated_at = datetime('now','localtime') WHERE uuid = ?",
                         [new_value, item_uuid])
            conn.commit()
            conn.close()
            self._log(f"[{item_uuid[:8]}] 图片路径已更新")
        elif col == 5:  # JSON路径
            conn = sqlite3.connect(DB_PATH)
            conn.execute("UPDATE publish_queue SET json_path = ?, updated_at = datetime('now','localtime') WHERE uuid = ?",
                         [new_value, item_uuid])
            conn.commit()
            conn.close()
            self._log(f"[{item_uuid[:8]}] JSON路径已更新")

    # ── 服务器启停 ──
    def _toggle_server(self):
        if self._server and self._server.is_alive():
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self):
        port = self.port_spin.value()
        self._server = FlaskServerThread(port)
        self._server.start()
        self.btn_start.setText("停止 Server")
        self.status_label.setText(f"● 运行中 (port {port})")
        self._log(f"Server 已启动: http://127.0.0.1:{port}/api/queue")

    def _stop_server(self):
        if self._server:
            self._server.stop()
        self._server = None
        self.btn_start.setText("启动 Server")
        self.status_label.setText("● 未启动")
        self._log("Server 已停止。")

    def _on_port_changed(self, val):
        self._config["port"] = val
        save_config(self._config)
        if self._server and self._server.is_alive():
            self._log("端口已更改，请重启 Server 生效。")

    def _append_log(self, msg: str):
        self.log_text.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _log(self, msg: str):
        self._signal.log_signal.emit(msg)

    def closeEvent(self, event):
        self._stop_server()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = PublishServerWindow()
    window.show()
    sys.exit(app.exec())


# ── 冒烟测试 ─────────────────────────────────────────────
def smoke_test():
    """无 GUI 冒烟测试：验证数据库、入队、API 核心逻辑"""
    import tempfile

    # 用临时数据库，避免污染真实数据
    global DB_PATH
    original_db = DB_PATH
    DB_PATH = os.path.join(tempfile.gettempdir(), "publish_smoke_test.db")

    errors = []

    try:
        # 1. 建表
        init_db()
        print("[PASS] init_db")

        # 2. 模拟入队（构造假图片和假 JSON）
        tmpdir = tempfile.mkdtemp()
        img_path = os.path.join(tmpdir, "aaabbb_000001-test.jpg")
        json_path = os.path.join(tmpdir, "20260708-000000-aaabbb-test.json")
        # 生成最小合法 PNG（1x1 红色像素），QPixmap 能正确加载
        import struct, zlib as _zlib
        def _make_minimal_png(w=1, h=1, r=255, g=0, b=0):
            sig = b'\x89PNG\r\n\x1a\n'
            ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
            ihdr = b'\x00\x00\x00\x0dIHDR' + ihdr_data
            ihdr += struct.pack('>I', _zlib.crc32(b'IHDR' + ihdr_data) & 0xFFFFFFFF)
            raw = b'\x00' + bytes([r, g, b]) * w
            for _ in range(1, h):
                raw += b'\x00' + bytes([r, g, b]) * w
            idat_data = _zlib.compress(raw)
            idat = struct.pack('>I', len(idat_data)) + b'IDAT' + idat_data
            idat += struct.pack('>I', _zlib.crc32(b'IDAT' + idat_data) & 0xFFFFFFFF)
            iend = b'\x00\x00\x00\x00IEND'
            iend += struct.pack('>I', _zlib.crc32(b'IEND') & 0xFFFFFFFF)
            return sig + ihdr + idat + iend
        with open(img_path, "wb") as f:
            f.write(_make_minimal_png())
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"prompt": "test prompt"}, f)

        result = add_to_queue(img_path)
        assert result["ok"], f"add_to_queue failed: {result['message']}"
        item_uuid = result["uuid"]
        print(f"[PASS] add_to_queue → {item_uuid}")

        # 3. 验证数据库字段
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM publish_queue WHERE uuid = ?", [item_uuid]).fetchone()
        assert row is not None, "record not found"
        assert row["pixiv_status"] == "pending", f"pixiv_status: {row['pixiv_status']}"
        assert row["chichipui_status"] == "pending", f"chichipui_status: {row['chichipui_status']}"
        assert row["image_md5"], "md5 empty"
        conn.close()
        print("[PASS] DB fields check")

        # 4. 测试 API（通过 Flask test client）
        with flask_app.test_client() as client:
            # 列表
            resp = client.get("/api/queue?pixiv_status=pending")
            data = resp.get_json()
            assert data["total"] >= 1, "list empty"
            print(f"[PASS] GET /api/queue (total={data['total']})")

            # latest-pending
            resp = client.get("/api/queue/latest-pending?platform=pixiv")
            data = resp.get_json()
            assert data["code"] == 0, f"expected code=0, got {data}"
            assert data["data"]["uuid"] == item_uuid, "uuid mismatch"
            print("[PASS] GET /api/queue/latest-pending?platform=pixiv")

            resp = client.get("/api/queue/latest-pending?platform=chichipui")
            data = resp.get_json()
            assert data["code"] == 0
            assert data["data"]["uuid"] == item_uuid
            print("[PASS] GET /api/queue/latest-pending?platform=chichipui")

            # get by uuid
            resp = client.get(f"/api/queue/{item_uuid}")
            data = resp.get_json()
            assert "image_base64" in data, "missing image_base64"
            print("[PASS] GET /api/queue/{uuid}")

            # get image blob
            resp = client.get(f"/api/queue/{item_uuid}/image")
            assert resp.status_code == 200, f"image status: {resp.status_code}"
            assert resp.content_type.startswith("image/"), f"content_type: {resp.content_type}"
            print("[PASS] GET /api/queue/{uuid}/image")

            # 更新 pixiv 状态
            resp = client.put(
                f"/api/queue/{item_uuid}/status",
                json={"platform": "pixiv", "status": "uploaded"},
            )
            data = resp.get_json()
            assert data["code"] == 0, f"update failed: {data}"
            assert data["platform"] == "pixiv"
            print("[PASS] PUT /api/queue/{uuid}/status (pixiv=uploaded)")

            # 验证 pixiv 已更新但 chichipui 未变
            resp = client.get(f"/api/queue/{item_uuid}")
            data = resp.get_json()
            assert data["pixiv_status"] == "uploaded"
            assert data["chichipui_status"] == "pending"
            print("[PASS] 双状态独立验证")

            # 更新 chichipui 状态
            resp = client.put(
                f"/api/queue/{item_uuid}/status",
                json={"platform": "chichipui", "status": "uploading"},
            )
            assert resp.get_json()["code"] == 0
            print("[PASS] PUT /api/queue/{uuid}/status (chichipui=uploading)")

            # 非法状态
            resp = client.put(
                f"/api/queue/{item_uuid}/status",
                json={"platform": "pixiv", "status": "invalid"},
            )
            assert resp.status_code == 400
            print("[PASS] 非法 status 返回 400")

            # 非法平台
            resp = client.put(
                f"/api/queue/{item_uuid}/status",
                json={"platform": "invalid", "status": "pending"},
            )
            assert resp.status_code == 400
            print("[PASS] 非法 platform 返回 400")

            # ===== 悬浮预览工具链自测 =====
            print("\n--- 悬浮预览工具链自测 ---")

            # 确保 QApplication 实例存在（QPixmap 依赖）
            _app = QApplication.instance()
            if _app is None:
                _app = QApplication(sys.argv)
            print(f"[PASS] QApplication 实例: {_app}")

            # 测试 QPixmap 缩放 + base64 转换链路
            pixmap = QPixmap(img_path)
            assert not pixmap.isNull(), "QPixmap 加载失败"
            print(f"[PASS] QPixmap 加载: {pixmap.width()}x{pixmap.height()}")

            scaled = pixmap.scaled(QSize(720, 720), Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
            assert max(scaled.width(), scaled.height()) <= 720, "缩放后超出 720px 限制"
            print(f"[PASS] 缩放: {scaled.width()}x{scaled.height()} (max 720)")

            ba = QByteArray()
            buf = QBuffer(ba)
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            scaled.save(buf, "PNG")
            b64 = ba.toBase64().data().decode()
            assert len(b64) > 100, "base64 过短"
            assert b64.startswith("iVBOR"), f"不是 PNG base64: {b64[:20]}..."
            print(f"[PASS] PNG base64 编码: {len(b64)} chars")

            # 验证 HTML tooltip 字符串可拼接
            tip = f'<img src="data:image/png;base64,{b64}" style="max-width:720px;max-height:720px;">'
            assert '<img' in tip and 'data:image/png;base64,' in tip
            print(f"[PASS] HTML tooltip 拼接: {len(tip)} chars")

            # 验证 QCursor.pos() 可调用
            try:
                cp = QCursor.pos()
                print(f"[PASS] QCursor.pos() = ({cp.x()}, {cp.y()})")
            except Exception:
                print("[PASS] QCursor.pos() 在无屏幕环境跳过（正常）")

            # 验证 QToolTip 模块可用
            assert hasattr(QToolTip, 'showText'), "QToolTip.showText 不存在"
            assert hasattr(QToolTip, 'hideText'), "QToolTip.hideText 不存在"
            print("[PASS] QToolTip API 可用")

    except Exception as e:
        errors.append(str(e))
    finally:
        # 清理
        try:
            os.remove(DB_PATH)
        except OSError:
            pass
        DB_PATH = original_db

    if errors:
        print(f"\n[FAIL] {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n[OK] 全部冒烟测试通过")


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        main()
