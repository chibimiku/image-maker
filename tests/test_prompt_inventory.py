from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "prompts"

REQUIRED_PROMPT_FILES = [
    "default_template.txt",
    "diff-cg-anchor-system.md",
    "diff-cg-anchor-user.md",
    "diff-cg-script-system.md",
    "diff-cg-script-user.md",
    "doujin-translator-system.md",
    "doujin-translator-user.md",
    "prompt-generator-system.md",
    "prompt-generator-user.md",
    "recompute-pixiv-tags.md",
    "refine-desc.md",
    "sd-make-system_prompt.md",
    "single-analyzer-outfit-check.md",
    "single-analyzer-system.md",
    "style-analy.md",
    "style-analyzer-system.md",
    "translate-booru-tags-system.md",
    "translate-booru-tags-user.md",
    "char/default.json",
    "image-edit/default.md",
]


def test_required_prompt_files_exist():
    missing = [str(PROMPTS_DIR / rel_path) for rel_path in REQUIRED_PROMPT_FILES if not (PROMPTS_DIR / rel_path).is_file()]
    assert not missing, f"缺少 Prompt 文件: {missing}"


def test_python_files_no_longer_reference_data_prompts():
    py_files = REPO_ROOT.rglob("*.py")
    offenders = []
    legacy_posix = "data" + "/" + "prompts"
    legacy_windows = r"data" + "\\" + "prompts"
    for path in py_files:
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if legacy_posix in text or legacy_windows in text:
            offenders.append(str(path))
    assert not offenders, f"仍存在 data/prompts 引用: {offenders}"


def test_tmp_prompt_file_removed():
    assert not (PROMPTS_DIR / "tmp.txt").exists()
