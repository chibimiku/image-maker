from pathlib import Path

import pytest

from utils import prompt_loader


MOCK_PROMPTS_DIR = Path(__file__).resolve().parent / "mock_data" / "prompts"


def test_get_prompt_path_requires_relative_path():
    with pytest.raises(ValueError):
        prompt_loader.get_prompt_path("")


def test_get_prompt_path_normalizes_separators(monkeypatch):
    monkeypatch.setattr(prompt_loader, "PROMPTS_DIR", str(MOCK_PROMPTS_DIR))
    actual = Path(prompt_loader.get_prompt_path(r"nested\demo.txt"))
    expected = MOCK_PROMPTS_DIR / "nested" / "demo.txt"
    assert actual == expected


def test_read_prompt_file_reads_mock_data(monkeypatch):
    monkeypatch.setattr(prompt_loader, "PROMPTS_DIR", str(MOCK_PROMPTS_DIR))
    content = prompt_loader.read_prompt_file("sample_prompt.txt")
    assert "Hello {name}!" in content


def test_render_prompt_file_replaces_placeholders(monkeypatch):
    monkeypatch.setattr(prompt_loader, "PROMPTS_DIR", str(MOCK_PROMPTS_DIR))
    rendered = prompt_loader.render_prompt_file(
        "sample_prompt.txt",
        {"name": "Alice", "kind": "demo"},
    )
    assert rendered == "Hello Alice!\nPrompt type: demo\n"


def test_find_missing_prompt_files_returns_missing_paths(monkeypatch):
    monkeypatch.setattr(prompt_loader, "PROMPTS_DIR", str(MOCK_PROMPTS_DIR))
    missing = prompt_loader.find_missing_prompt_files(
        ["sample_prompt.txt", "missing_prompt.txt"]
    )
    assert missing == [str(MOCK_PROMPTS_DIR / "missing_prompt.txt")]
