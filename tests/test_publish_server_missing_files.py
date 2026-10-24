"""Missing queued images must be quarantined without blocking the next upload."""
import json
import os
import sqlite3

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
import publish_server as ps


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "DB_PATH", str(tmp_path / "queue.db"))
    ps.init_db()
    def insert(name, pixiv="pending", chichi="pending", exists=True):
        image = tmp_path / (name + ".png")
        if exists:
            image.write_bytes(b"test image")
        with sqlite3.connect(ps.DB_PATH) as conn:
            conn.execute("""INSERT INTO publish_queue
                (uuid,image_path,image_md5,metadata,pixiv_status,chichipui_status)
                VALUES (?,?,?,?,?,?)""", [name, str(image), name, json.dumps({"title": name}), pixiv, chichi])
        return image
    return insert


def row(name):
    with sqlite3.connect(ps.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute("SELECT * FROM publish_queue WHERE uuid=?", [name]).fetchone())


def test_startup_checks_all_unfinished_images_preserves_uploaded_history(queue):
    queue("both-missing", exists=False)
    queue("partly-uploaded", "uploaded", "pending", exists=False)
    queue("in-flight", "uploading", "uploaded", exists=False)
    queue("finished", "uploaded", "uploaded", exists=False)
    queue("valid")
    assert set(ps.init_db()) == {"both-missing", "partly-uploaded", "in-flight"}
    assert row("both-missing")["pixiv_status"] == row("both-missing")["chichipui_status"] == "error"
    assert row("partly-uploaded")["pixiv_status"] == "uploaded"
    assert row("partly-uploaded")["chichipui_status"] == "error"
    assert row("in-flight")["pixiv_status"] == "error"
    assert row("finished")["pixiv_status"] == row("finished")["chichipui_status"] == "uploaded"
    assert row("valid")["pixiv_status"] == "pending"
    assert "图片文件不存在" in row("both-missing")["error_message"]
    assert ps.init_db() == []  # The quarantine is idempotent.


@pytest.mark.parametrize("platform", ["pixiv", "chichipui"])
def test_delete_after_startup_skips_missing_head_and_returns_next(queue, platform):
    queue("valid")
    queue("missing-1").unlink()
    queue("missing-2").unlink()
    response = ps.flask_app.test_client().get("/api/queue/latest-pending?platform=" + platform)
    assert response.status_code == 200
    assert response.json["data"]["uuid"] == "valid"
    assert response.json["data"]["image_base64"]
    assert row("missing-1")["pixiv_status"] == "error"
    assert row("missing-2")["chichipui_status"] == "error"


def test_no_valid_pending_returns_empty_instead_of_missing_record(queue):
    queue("missing", exists=False)
    response = ps.flask_app.test_client().get("/api/queue/latest-pending")
    assert response.json["code"] == 1 and response.json["data"] is None
    assert row("missing")["pixiv_status"] == "error"


def test_image_deleted_between_queue_pull_and_binary_fetch(queue):
    image = queue("late-delete")
    client = ps.flask_app.test_client()
    assert client.get("/api/queue/latest-pending").json["data"]["uuid"] == "late-delete"
    image.unlink()
    assert client.get("/api/queue/late-delete/image").status_code == 404
    assert row("late-delete")["pixiv_status"] == "error"
    assert client.get("/api/queue/latest-pending").json["data"] is None


def test_unreadable_or_empty_image_does_not_block_next(queue, monkeypatch):
    queue("valid")
    queue("unreadable")
    original = ps.image_to_base64
    monkeypatch.setattr(ps, "image_to_base64", lambda path: None if os.path.basename(path) == "unreadable.png" else original(path))
    client = ps.flask_app.test_client()
    assert client.get("/api/queue/latest-pending").json["data"]["uuid"] == "valid"
    assert "无法读取" in row("unreadable")["error_message"]


def test_restore_then_manually_reset_pending(queue):
    image = queue("restore", "uploaded", "pending", exists=False)
    ps.init_db()
    client = ps.flask_app.test_client()
    data = {"platform": "chichipui", "status": "pending"}
    assert client.put("/api/queue/restore/status", json=data).status_code == 409
    image.write_bytes(b"restored image")
    assert client.put("/api/queue/restore/status", json=data).status_code == 200
    assert row("restore")["error_message"] == ""
    assert row("restore")["pixiv_status"] == "uploaded"
    assert client.get("/api/queue/latest-pending?platform=chichipui").json["data"]["uuid"] == "restore"


def test_error_status_api_filter_and_direct_lookup(queue):
    queue("manual-error")
    client = ps.flask_app.test_client()
    assert client.put("/api/queue/manual-error/status", json={"platform": "pixiv", "status": "error"}).status_code == 200
    listing = client.get("/api/queue?pixiv_status=error").json
    assert listing["total"] == 1
    assert listing["items"][0]["uuid"] == "manual-error"
    assert client.get("/api/queue/manual-error").json["pixiv_status"] == "error"


def test_legacy_schema_adds_error_message_without_losing_data(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "DB_PATH", str(tmp_path / "legacy.db"))
    with sqlite3.connect(ps.DB_PATH) as conn:
        conn.execute("""CREATE TABLE publish_queue (id INTEGER PRIMARY KEY, uuid TEXT,
            image_path TEXT,image_md5 TEXT,json_path TEXT,metadata TEXT,status TEXT,
            created_at TEXT,updated_at TEXT)""")
        conn.execute("INSERT INTO publish_queue VALUES(1,'legacy',?,'hash',NULL,?,'pending','old','old')",
                     [str(tmp_path / "missing.png"), json.dumps({"title": "history"})])
    assert ps.init_db() == ["legacy"]
    result = row("legacy")
    assert result["pixiv_status"] == result["chichipui_status"] == "error"
    assert json.loads(result["metadata"])["title"] == "history"


def test_gui_error_filter_reason_and_missing_reset_guard(queue, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    queue("missing", exists=False)
    monkeypatch.setattr(ps, "CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(ps, "port_is_available", lambda port: False)
    window = ps.PublishServerWindow()
    try:
        assert window.filter_pixiv.findText("错误") >= 0
        assert window.table.item(0, 0).text() == "错误"
        assert "图片文件不存在" in window.table.item(0, 0).toolTip()
        window.table.item(0, 0).setText("待上传")
        assert row("missing")["pixiv_status"] == "error"
        assert window.table.item(0, 0).text() == "错误"
    finally:
        window._refresh_timer.stop()
        window.close()
        window.deleteLater()
        app.processEvents()
