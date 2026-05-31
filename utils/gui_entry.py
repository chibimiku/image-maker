import importlib
import os
import sys


def redirect_stdio_for_windows_gui_entry():
    """Avoid pythonw/frozen GUI startup deadlocks caused by blocked stderr output."""
    should_redirect = False
    if getattr(sys, "frozen", False):
        should_redirect = True
    else:
        executable = str(getattr(sys, "executable", "") or "").lower()
        should_redirect = "pythonw" in executable
    if not should_redirect:
        return False

    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    return True


def warm_up_optional_module(module_name: str, skip_env_var: str | None = None):
    if skip_env_var and os.environ.get(skip_env_var):
        return False
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def configure_qt_application_attributes(qapplication_cls, qt_namespace):
    app_attr = getattr(qt_namespace, "ApplicationAttribute", None)
    if app_attr is None:
        return

    if hasattr(app_attr, "AA_EnableHighDpiScaling"):
        qapplication_cls.setAttribute(app_attr.AA_EnableHighDpiScaling, True)
    if hasattr(app_attr, "AA_UseHighDpiPixmaps"):
        qapplication_cls.setAttribute(app_attr.AA_UseHighDpiPixmaps, True)
