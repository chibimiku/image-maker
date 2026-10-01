"""Publishing queue must never accept an image without usable submission metadata."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import publish_server as ps


class PublishQueueMetadataTests(unittest.TestCase):
    def test_rejects_missing_and_invalid_json_before_inserting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "publish_queue.db"
            image = root / "65ebea5b_final.jpg"
            image.write_bytes(b"image")
            metadata = root / "20260928-221533-65ebea5b-title.json"
            with patch.object(ps, "DB_PATH", str(db_path)):
                ps.init_db()
                missing = ps.add_to_queue(str(image))
                self.assertFalse(missing["ok"])
                self.assertEqual(missing["error"], "missing_metadata")

                metadata.write_text("{invalid", encoding="utf-8")
                invalid = ps.add_to_queue(str(image))
                self.assertFalse(invalid["ok"])
                self.assertEqual(invalid["error"], "invalid_metadata")

                connection = sqlite3.connect(db_path)
                try:
                    count = connection.execute("SELECT COUNT(*) FROM publish_queue").fetchone()[0]
                finally:
                    connection.close()
                self.assertEqual(count, 0)

                metadata.write_text(json.dumps({"title": "ready"}), encoding="utf-8")
                added = ps.add_to_queue(str(image))
                self.assertTrue(added["ok"], added["message"])
                connection = sqlite3.connect(db_path)
                try:
                    row = connection.execute(
                        "SELECT json_path, metadata FROM publish_queue WHERE uuid = ?",
                        (added["uuid"],),
                    ).fetchone()
                finally:
                    connection.close()
                self.assertEqual(row[0], str(metadata))
                self.assertEqual(json.loads(row[1])["title"], "ready")


if __name__ == "__main__":
    unittest.main()
