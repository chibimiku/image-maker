from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOTS = [
    REPO_ROOT,
    REPO_ROOT / "modules",
    REPO_ROOT / "utils",
]

ROOT_FILES = {
    "app.py",
    "make-pic.py",
    "sd-make-pic.py",
    "doujin_translator.py",
    "translate_booru_tags.py",
}

LEGACY_PATTERNS = [
    r"QMessageBox\.(Yes|No)\b",
    r"QLineEdit\.PasswordEchoOnEdit\b",
    r"QAbstractItemView\.(SingleSelection|ExtendedSelection)\b",
    r"QListWidget\.(ExtendedSelection|IconMode|Adjust)\b",
    r"QFrame\.StyledPanel\b",
    r"Qt\.(AlignCenter|AlignTop|AlignLeft|AlignRight|AlignBottom)\b",
    r"Qt\.(Horizontal|Vertical|CopyAction)\b",
    r"Qt\.(UserRole|Checked|Unchecked|ItemIsUserCheckable)\b",
    r"Qt\.(KeepAspectRatio|SmoothTransformation)\b",
    r"Qt\.(CaseInsensitive|MatchContains)\b",
    r"Qt\.(LeftButton|RightButton|ControlModifier|Key_V)\b",
    r"Qt\.(white|red|green|yellow|WaitCursor)\b",
    r"exec_\(",
]


def iter_source_files():
    yielded = set()

    for path in sorted(REPO_ROOT.glob("*.py")):
        if path.name in ROOT_FILES:
            yielded.add(path.resolve())
            yield path

    for root in SOURCE_ROOTS[1:]:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            resolved = path.resolve()
            if resolved in yielded:
                continue
            yielded.add(resolved)
            yield path


def test_source_files_no_longer_import_pyqt5():
    offenders = []
    legacy_import = "from " + "PyQt5"

    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        if legacy_import in text or "import PyQt5" in text:
            offenders.append(str(path))

    assert not offenders, f"仍存在 PyQt5 导入: {offenders}"


def test_source_files_no_longer_use_legacy_qt_patterns():
    offenders = {}

    for path in iter_source_files():
        text = path.read_text(encoding="utf-8")
        matches = [pattern for pattern in LEGACY_PATTERNS if re.search(pattern, text)]
        if matches:
            offenders[str(path)] = matches

    assert not offenders, f"仍存在旧式 Qt 写法: {offenders}"
