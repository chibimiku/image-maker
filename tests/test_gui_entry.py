import types

from utils import gui_entry


def test_warm_up_optional_module_imports_when_not_skipped(monkeypatch):
    calls = []

    def fake_import_module(name):
        calls.append(name)
        return object()

    monkeypatch.setattr(gui_entry.importlib, "import_module", fake_import_module)

    assert gui_entry.warm_up_optional_module("onnxruntime") is True
    assert calls == ["onnxruntime"]


def test_warm_up_optional_module_respects_skip_env(monkeypatch):
    calls = []

    def fake_import_module(name):
        calls.append(name)
        return object()

    monkeypatch.setattr(gui_entry.importlib, "import_module", fake_import_module)
    monkeypatch.setenv("IMAGE_MAKER_SKIP_ONNXRUNTIME_PRELOAD", "1")

    assert (
        gui_entry.warm_up_optional_module(
            "onnxruntime",
            skip_env_var="IMAGE_MAKER_SKIP_ONNXRUNTIME_PRELOAD",
        )
        is False
    )
    assert calls == []


def test_configure_qt_application_attributes_sets_supported_flags():
    applied = []

    class FakeApplication:
        @staticmethod
        def setAttribute(attr, value):
            applied.append((attr, value))

    fake_qt = types.SimpleNamespace(
        ApplicationAttribute=types.SimpleNamespace(
            AA_EnableHighDpiScaling="dpi-scaling",
            AA_UseHighDpiPixmaps="dpi-pixmaps",
        )
    )

    gui_entry.configure_qt_application_attributes(FakeApplication, fake_qt)

    assert applied == [("dpi-scaling", True), ("dpi-pixmaps", True)]
