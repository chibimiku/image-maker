import json
import sqlite3


def test_source_directory_gemini_metadata_uses_latest_exact_source(tmp_path):
    import publish_server as ps

    image = tmp_path / "Gemini_Generated_Image_9rk5jl9rk5jl9rk5.jpg"
    image.write_bytes(b"image")
    other = tmp_path / "Gemini_Generated_Image_other.jpg"
    other.write_bytes(b"other")
    for name, source in (
        ("20261003-213908-Gemini-old.json", image),
        ("20261004-015352-Gemini-latest.json", image),
        ("20261004-020000-Gemini-other.json", other),
    ):
        (tmp_path / name).write_text(
            json.dumps({"source_image_path": str(source)}), encoding="utf-8"
        )

    assert ps.find_metadata_json(str(image)) == str(
        tmp_path / "20261004-015352-Gemini-latest.json"
    )


def test_shared_gemini_prefix_never_matches_another_source(tmp_path):
    import publish_server as ps

    image = tmp_path / "Gemini_Generated_Image_missing.jpg"
    image.write_bytes(b"image")
    metadata = tmp_path / "20261004-020000-Gemini-other.json"
    metadata.write_text(json.dumps({
        "source_image_path": str(tmp_path / "Gemini_Generated_Image_other.jpg")
    }), encoding="utf-8")

    assert ps.find_metadata_json(str(image)) is None


def test_exact_source_match_does_not_require_filename_key(tmp_path):
    import publish_server as ps

    image = tmp_path / "Original_Image.jpg"
    image.write_bytes(b"image")
    metadata = tmp_path / "20261004-020000-12345678-title.json"
    metadata.write_text(json.dumps({"source_image_path": str(image)}), encoding="utf-8")

    assert ps.find_metadata_json(str(image)) == str(metadata)


def test_legacy_filename_key_is_case_insensitive(tmp_path):
    import publish_server as ps

    image = tmp_path / "Gemini_Generated_Image_legacy.jpg"
    image.write_bytes(b"image")
    metadata = tmp_path / "20261004-020000-Gemini-title.json"
    metadata.write_text(json.dumps({"title": "legacy"}), encoding="utf-8")

    assert ps.find_metadata_json(str(image)) == str(metadata)


def test_final_style_prefix_finds_analysis_json_by_embedded_hash(tmp_path):
    import publish_server as ps

    image = tmp_path / "ajicoma-960cddea-quality-refine_105842-1e7c96-final.jpg"
    image.write_bytes(b"image")
    metadata = tmp_path / "20260926-105642-960cddea-title.json"
    metadata.write_text(json.dumps({"source_image_path": "elsewhere.jpg"}), encoding="utf-8")

    assert ps.find_metadata_json(str(image)) == str(metadata)


def test_init_db_repairs_existing_rows_without_metadata(tmp_path, monkeypatch):
    import publish_server as ps

    image = tmp_path / "ajicoma-960cddea-quality-refine-final.jpg"
    image.write_bytes(b"image")
    metadata = tmp_path / "20260926-105642-960cddea-title.json"
    metadata.write_text(json.dumps({"title": "linked"}), encoding="utf-8")
    db_path = tmp_path / "queue.db"
    monkeypatch.setattr(ps, "DB_PATH", str(db_path))
    ps.init_db()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO publish_queue (uuid,image_path,image_md5,json_path,metadata) VALUES (?,?,?,?,?)",
        ["row-1", str(image), "md5", None, None],
    )
    conn.commit()
    conn.close()

    ps.init_db()

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT json_path, metadata FROM publish_queue WHERE uuid='row-1'").fetchone()
    conn.close()
    assert row[0] == str(metadata)
    assert json.loads(row[1])["title"] == "linked"
