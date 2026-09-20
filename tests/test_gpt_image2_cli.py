"""tools/gpt_image2_gen.py CLI 测试（不发真实请求）。"""
import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.others import api_backend

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


def load_cli_module():
    spec = importlib.util.spec_from_file_location("gpt_image2_gen_cli", REPO_ROOT / "tools" / "gpt_image2_gen.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli():
    return load_cli_module()


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config-image.json"
    path.write_text(
        json.dumps(
            {
                "apis": {
                    "aigc-2d-gpt": {
                        "base_url": "https://example.invalid/v1",
                        "api_key": "test-key",
                        "model": "gpt-image-2",
                        "timeout": 30,
                        "max_retries": 0,
                    },
                    "autodl": {
                        "base_url": "https://autodl.invalid/api/v1",
                        "api_key": "autodl-key",
                        "model": "gpt-image-2",
                        "timeout": 30,
                        "max_retries": 0,
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return str(path)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.encoding = "utf-8"

    def json(self):
        return self._payload

    @property
    def text(self):
        return json.dumps(self._payload)

    def raise_for_status(self):
        pass


def test_list_sites_prints_both_sites(cli, config_path, capsys):
    assert cli.main(["--config", config_path, "--list-sites"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "new.aigc2d" in out and "autodl" in out
    assert "aigc-2d-gpt" in out
    assert "1024x1536" in out
    assert "常用模型" in out and "gpt-image-2.5-flare" in out


def test_list_models_prints_only_gpt_image_family(cli, config_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "list_available_models",
        lambda **_kwargs: ["Kimi-K3", "gpt-image-2", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"],
    )

    assert cli.main(["--config", config_path, "--list-models"]) == cli.EXIT_OK

    out = capsys.readouterr().out
    assert "gpt-image-2.5-flare" in out
    assert "gpt-image-2.5-sunburst" in out
    assert "ChatGPT Images 2.5" in out
    assert "Kimi-K3" not in out
    assert "另有 1 个文本/视频等模型未列出" in out


def test_list_models_without_api_key_is_usage_error(cli, config_path, tmp_path, capsys):
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    payload["apis"]["aigc-2d-gpt"]["api_key"] = ""
    broken = tmp_path / "config-no-key.json"
    broken.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    assert cli.main(["--config", str(broken), "--list-models"]) == cli.EXIT_USAGE
    assert "缺少 api_key" in capsys.readouterr().out


def test_dry_run_aigc2d_shows_generations_endpoint_and_payload(cli, config_path, capsys):
    code = cli.main(["--config", config_path, "--prompt", "一只猫", "--size", "auto", "--dry-run"])
    assert code == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "https://example.invalid/v1/images/generations" in out
    assert '"size": "1024x1024"' in out      # auto 被收敛
    assert "DRY-RUN" in out


def test_dry_run_with_image_switches_to_edits(cli, config_path, tmp_path, capsys):
    image = tmp_path / "ref.png"
    image.write_bytes(TINY_PNG)
    code = cli.main(["--config", config_path, "--prompt", "改背景", "--image", str(image), "--dry-run"])
    assert code == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "https://example.invalid/v1/images/edits" in out


def test_dry_run_autodl_uses_v1_images_path(cli, config_path, capsys):
    code = cli.main(["--config", config_path, "--site", "autodl", "--prompt", "一只猫", "--size", "1792x1024", "--dry-run"])
    assert code == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "https://autodl.invalid/api/v1/images/generations" in out
    assert "1792x1024" in out


def test_missing_prompt_is_usage_error(cli, config_path, capsys):
    assert cli.main(["--config", config_path]) == cli.EXIT_USAGE
    assert "需要 --prompt" in capsys.readouterr().out


def test_edit_mode_requires_image(cli, config_path, capsys):
    code = cli.main(["--config", config_path, "--mode", "edit", "--prompt", "改一下"])
    assert code == cli.EXIT_USAGE
    assert "至少需要 1 张" in capsys.readouterr().out


def test_missing_image_is_usage_error(cli, config_path, tmp_path, capsys):
    code = cli.main(["--config", config_path, "--prompt", "x", "--image", str(tmp_path / "nope.png")])
    assert code == cli.EXIT_USAGE
    assert "参考图不存在" in capsys.readouterr().out


def test_run_saves_image_via_aigc2d_channel(cli, config_path, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse({"data": [{"b64_json": base64.b64encode(TINY_PNG).decode()}], "output_format": "png"})

    monkeypatch.setattr(api_backend.requests, "post", fake_post)

    code = cli.main([
        "--config", config_path,
        "--prompt", "一位少女站在海边",
        "--size", "1024x1536",
        "--quality", "medium",
        "--output-subdir", "cli-test",
        "--prefix", "unitcli",
        "--json",
    ])

    assert code == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "SAVED " in out
    assert captured["url"] == "https://example.invalid/v1/images/generations"
    assert captured["kwargs"]["json"]["size"] == "1024x1536"
    assert captured["kwargs"]["json"]["quality"] == "medium"
    saved = list((tmp_path / "data").rglob("unitcli_*.png"))
    assert len(saved) == 1
    assert saved[0].read_bytes() == TINY_PNG
    assert '"saved_files"' in out


def test_prompt_file_supports_size_override(cli, config_path, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    captured = []

    def fake_post(url, **kwargs):
        captured.append(kwargs)
        return _FakeResponse({"data": [{"b64_json": base64.b64encode(TINY_PNG).decode()}], "output_format": "png"})

    monkeypatch.setattr(api_backend.requests, "post", fake_post)

    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "# 注释行\n1024x1536|纵向构图\n1536x1024|横向构图\n普通一行\n",
        encoding="utf-8",
    )

    code = cli.main(["--config", config_path, "--prompt-file", str(prompt_file), "--output-subdir", "cli-batch"])

    assert code == cli.EXIT_OK
    sizes = [item["json"]["size"] for item in captured]
    assert sizes == ["1024x1536", "1536x1024", "1024x1536"]  # 最后一行用站点默认
    assert len(list((tmp_path / "data").rglob("*.png"))) == 3


def test_run_failure_returns_exit_1(cli, config_path, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def fake_post(url, **kwargs):
        return _FakeResponse({"error": {"message": "moderation_blocked"}})

    monkeypatch.setattr(api_backend.requests, "post", fake_post)

    code = cli.main(["--config", config_path, "--prompt", "x", "--output-subdir", "cli-fail"])

    assert code == cli.EXIT_FAIL
    assert "失败" in capsys.readouterr().out


def test_timeout_override_writes_temp_config(cli, config_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_post(url, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return _FakeResponse({"data": [{"b64_json": base64.b64encode(TINY_PNG).decode()}], "output_format": "png"})

    monkeypatch.setattr(api_backend.requests, "post", fake_post)

    code = cli.main(["--config", config_path, "--prompt", "x", "--timeout", "123", "--output-subdir", "cli-timeout"])

    assert code == cli.EXIT_OK
    assert seen["timeout"] == 123
    # 原配置文件不被改写
    assert json.loads(Path(config_path).read_text(encoding="utf-8"))["apis"]["aigc-2d-gpt"]["timeout"] == 30
