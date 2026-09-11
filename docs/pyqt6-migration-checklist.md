# PyQt6 Migration Stage Summary

## Current Stage

The project has completed the first major PyQt6 migration phase:

- Most legacy Qt enum/API usages have been replaced with PyQt6-style names.
- Main source files have been switched from `PyQt5` imports to `PyQt6`.
- Window and tab construction smoke tests now run under PyQt6.
- Key interaction smoke tests cover drag/drop, clipboard paste, task selection, template switching, and UI state roundtrip.
- GUI entry bootstrap logic has been centralized in `utils/gui_entry.py`.
- Remaining `PyQt5` string matches are only in test assertions that guard against regressions.

Current automated validation status:

- `pytest`: passing
- `py_compile`: passing for recently touched files
- PyQt6 smoke tests: passing

## Closure Status

Current closure state for the main migration line:

- Completed: source imports are on `PyQt6`, legacy `exec_()` usage is gone, and remaining `PyQt5` matches are only test guard strings.
- Completed: automated smoke coverage now includes main window construction, top-level tab switching, drag/drop, clipboard paste, UI state roundtrip, and tray notification no-crash branches.
- Completed: `app.py` has been launched once on the normal desktop backend with `IMAGE_MAKER_SKIP_ONNXRUNTIME_PRELOAD=1`, and it stayed alive until manually stopped.
- Pending manual confirmation: native Windows desktop drag gesture, native clipboard image paste, visible tray balloon behavior on the current shell, and repeated human-operated tab switching across `图片分析` / `图片生成` / `其他` / `设置`.

## Entry Points

Current GUI/script entry points found in the repo:

- `app.py`
- `make-pic.py`
- `sd-make-pic.py`
- `doujin_translator.py`
- `modules/image_generation/webp_compressor.py`

Current CLI/script-like entry points:

- `translate_booru_tags.py`
- `utils/pic_cate.py`
- `modules/others/api_backend.py`
- `test.py`

## Entry Adaptation Assessment

### `app.py`

Status:

- Main integrated GUI entry is already on `PyQt6`.
- Uses shared GUI entry bootstrap helpers from `utils/gui_entry.py`.
- Keeps optional `onnxruntime` warm-up, now controllable via environment variable.

Assessment:

- Good: already considers `pythonw.exe` and frozen app behavior.
- Risk: eager `onnxruntime` preload can complicate packaged startup and increase failure surface.
- Risk: this file imports many tabs, so packaging must include a broad dependency set.

Recommendation:

- Keep shared bootstrap helpers as the single source of truth for GUI entry startup behavior.
- Use `IMAGE_MAKER_SKIP_ONNXRUNTIME_PRELOAD=1` when desktop smoke verification should avoid local ONNX startup side effects.
- Treat `app.py` as the primary packaging entry.

### `make-pic.py`

Status:

- Independent GUI entry migrated to `PyQt6`.
- Applies high-DPI application attributes before `QApplication`.

Assessment:

- Good standalone entry for manual runtime verification.
- No visible frozen-specific handling yet.

Recommendation:

- If kept as a standalone distributed tool, give it its own packaging profile.
- Otherwise consider reducing long-term packaging targets by gradually merging or deprecating standalone entries.

### `sd-make-pic.py`

Status:

- Already uses `PyQt6`.
- Still acts as a standalone SD workflow entry.

Assessment:

- Functionally independent, but increases packaging surface.
- Reads its own SD config path and prompt assets.

Recommendation:

- Keep out of the first packaged target unless it is a release requirement.
- If future plan is integration into main GUI, avoid investing in a separate frozen build first.

### `doujin_translator.py`

Status:

- Already switched to `PyQt6`.
- Separate GUI entry.

Assessment:

- Lower dependency complexity than `app.py`, but still an extra packaging target.

Recommendation:

- Decide whether it remains a separate shipped executable or becomes a module-only tool.

## Packaging Chain Assessment

## Current State

No existing packaging configuration was found for:

- `PyInstaller`
- `Nuitka`
- `cx_Freeze`
- `pyproject.toml`
- `.spec` files

This means the current task is not "migrate an existing packaging chain", but rather:

- identify PyQt6 packaging risks,
- choose a packaging tool,
- define the first supported packaged target,
- then add packaging config.

Additional gap:

- The repo currently has `requirements-dev.txt`, but no dedicated runtime dependency manifest such as `requirements.txt`.

Impact:

- Packaging reproducibility is weaker than it should be.
- A first packaging pass should be accompanied by a runtime dependency file or equivalent lock/manifest.

## Main Packaging Risks

### Dynamic Imports

The following dynamic-import-heavy areas will need explicit packaging attention:

- `app.py`: `onnxruntime`
- `modules/image_generation/z_image_edit_tab.py`: `torch`, `diffusers`, `transformers`, `huggingface_hub`
- `utils/image_upscale_runtime.py`: `torch`, `spandrel`, `spandrel_extra_arches`, `wand.image`, `numpy`, `onnxruntime`
- `utils/wd14_tagger.py`: `numpy`, `onnxruntime`
- `modules/image_generation/diff_cg_tab.py`: `cv2`, `numpy`, `torch`

Impact:

- PyInstaller/Nuitka may not automatically discover all required hidden imports.
- These modules should be treated as hidden-import candidates or optional runtime extras.

Recommended hidden-import inventory by exact module name:

- Core startup-sensitive:
  - `onnxruntime`

- `z-image` related:
  - `torch`
  - `diffusers`
  - `transformers`
  - `huggingface_hub`

- Upscaler / local inference related:
  - `numpy`
  - `onnxruntime`
  - `torch`
  - `spandrel`
  - `spandrel_extra_arches`
  - `wand.image`

- Diff-CG related:
  - `cv2`
  - `numpy`
  - `torch`

- WD14 local tagger related:
  - `numpy`
  - `onnxruntime`

Dynamic import sources currently found in code:

- [app.py](file:///d:/code/image-maker/app.py): `onnxruntime`
- [z_image_edit_tab.py](file:///d:/code/image-maker/modules/image_generation/z_image_edit_tab.py):
  - `torch`
  - `diffusers`
  - `transformers`
  - `huggingface_hub`
- [image_upscale_runtime.py](file:///d:/code/image-maker/utils/image_upscale_runtime.py):
  - `wand.image`
  - `numpy`
  - `torch`
  - `spandrel`
  - `spandrel_extra_arches`
  - `onnxruntime`
- [upscaler_tab.py](file:///d:/code/image-maker/modules/image_generation/upscaler_tab.py): `torch`
- [diff_cg_tab.py](file:///d:/code/image-maker/modules/image_generation/diff_cg_tab.py):
  - `cv2`
  - `numpy`
  - `torch`
- [wd14_tagger.py](file:///d:/code/image-maker/utils/wd14_tagger.py):
  - `numpy`
  - `onnxruntime`

### Qt Runtime Pieces

Potential packaging-sensitive Qt features already in use:

- `QSystemTrayIcon` in `utils/task_runtime.py`
- `QShortcut`
- clipboard / image handling via Qt + Pillow
- desktop services / URL open behavior

Impact:

- Packaged builds need correct Qt platform plugins and image format plugins.
- Tray availability may differ across environments.

### Heavy Native Dependencies

Native/runtime-heavy libraries already used:

- `onnxruntime`
- `torch`
- `diffusers`
- `transformers`
- `opencv-python`
- `numpy`
- `Pillow`

Impact:

- These are the most likely packaged startup and distribution-size risks.
- They also increase the chance of missing DLLs or platform plugin mismatches.

### `pythonw` / Frozen Behavior

The project now has shared GUI bootstrap logic for:

- `pythonw.exe`
- `getattr(sys, "frozen", False)`

Impact:

- This is now standardized across the main GUI entry flow.
- Standalone GUI entries can reuse the same helper instead of carrying divergent startup code.

## Recommended Packaging Strategy

### First Target

Use `app.py` as the first packaged GUI target.

Reason:

- It is the integrated main window.
- It already has the most advanced PyQt6 migration work and smoke validation.
- It includes `pythonw` / frozen handling.

### Tool Choice

Preferred first attempt:

- `PyInstaller`

Reason:

- Faster to establish a first working frozen build.
- Easier to iterate hidden imports and data files.
- Better suited for "get one GUI target running first" than jumping straight to more aggressive optimization.

### Initial Scope Control

First packaged target should avoid forcing all optional heavy local-inference features on day one.

Suggested scope:

- Package `app.py`
- Keep hidden `z-image` path lazy and non-blocking
- Allow optional runtime failure messaging for unavailable local-inference extras

## Final PyQt6 Closure Checklist

### A. Entry and Runtime

- Confirm `app.py` launches correctly in a real desktop environment, not only `offscreen`.
- Verify tray notifications behave acceptably when tray is unavailable.
- Verify clipboard paste and drag-drop in real Windows desktop session.
- Verify `pythonw` launch path still suppresses problematic stderr output without hiding actionable fatal errors.

### A1. Real Desktop Smoke For `app.py`

Suggested launch commands in PowerShell:

```powershell
python app.py
```

If local `onnxruntime` warm-up should be skipped during manual GUI verification:

```powershell
$env:IMAGE_MAKER_SKIP_ONNXRUNTIME_PRELOAD=1
python app.py
```

Manual verification checklist:

1. Launch `app.py` and confirm the main window opens normally without immediate crash.
2. Switch across the main tab groups and confirm tab content renders without blank panes or frozen UI.
3. In `单图分析`, drag a local image into the widget and confirm preview, button state, and logs update.
4. In `单图分析`, copy an image to the clipboard and press `Ctrl+V`, then confirm preview and logs update.
5. Trigger one non-destructive notification-producing path and confirm tray notification behavior is acceptable; if the tray is unavailable, confirm there is no crash.
6. Switch repeatedly between analysis, generation, and other tabs to confirm no focus glitches, repaint issues, or unexpected resets.

Pass criteria:

- No crash during startup or tab switching.
- Drag-drop works in the desktop session.
- Clipboard paste works in the desktop session.
- Tray notification path does not crash the app.
- Main tabs remain responsive after repeated switching.

Latest execution record:

- 2026-05-31: agent launched `app.py` with `IMAGE_MAKER_SKIP_ONNXRUNTIME_PRELOAD=1` in the normal desktop backend and confirmed the process stayed alive until manually stopped, with no immediate startup crash.
- 2026-05-31: automated supplement passed via `python -m pytest tests/test_pyqt6_smoke.py tests/test_pyqt6_inventory.py tests/test_gui_entry.py -q` (`24 passed`), including:
  - main `app.py` window tab switching smoke,
  - `SingleAnalyzerWidget` drag-drop and clipboard paste smoke,
  - tray notification no-crash smoke for both tray-available and tray-unavailable branches,
  - `SingleAnalyzerWidget` notification bridge forwarding to `SystemNotifier`.
- Remaining manual confirmation in a real Windows session:
  - actual OS drag-drop gesture into `单图分析`,
  - actual clipboard image paste via `Ctrl+V`,
  - visible tray balloon behavior on the current desktop shell,
  - repeated human-operated switching across analysis / generation / others / settings tabs.

Suggested manual record template:

```md
### app.py Manual Smoke Record

- Date:
- Environment: Windows desktop / whether `IMAGE_MAKER_SKIP_ONNXRUNTIME_PRELOAD=1` was used
- Startup: pass / fail
- Multi-tab switching: pass / issue
- Single analyzer drag-drop: pass / issue
- Single analyzer clipboard paste: pass / issue
- Tray notification: pass / not visible but stable / issue
- Notes:
```

### B. Packaging Preparation

- Add a runtime dependency manifest for packaged builds.
- Create a first packaging config for `app.py`.
- Enumerate hidden imports for dynamic modules:
  - `onnxruntime`
  - `torch`
  - `diffusers`
  - `transformers`
  - `huggingface_hub`
  - `spandrel`
  - `spandrel_extra_arches`
  - `wand.image`
  - `cv2`
  - `numpy`
- Enumerate bundled data directories:
  - `conf/`
  - `prompts/`
  - `models/` placeholders or model lookup docs as needed

### C. Packaging Validation

- Run a frozen `app.py` build locally.
- Validate startup without console deadlock.
- Validate prompt loading from bundled paths.
- Validate config read/write paths under frozen mode.
- Validate image preview, drag-drop, clipboard paste, and tab switching.

### D. Dependency Policy

- Decide which heavy local-inference features are first-class in packaged builds.
- Decide whether `z-image` remains excluded/lazy in the packaged main app.
- Decide whether standalone entries remain separate release targets.

### E. Cleanup

- Remove any remaining transitional assumptions that only existed for mixed PyQt5/PyQt6 migration.
- Decide whether to remove obsolete PyQt5-era packaging notes, if any are added later.

## Suggested Next Execution Order

1. Add packaging config for `app.py` only.
2. Run one local frozen build.
3. Fix hidden imports and bundled data paths.
4. Perform real desktop smoke verification.
5. Reassess whether `make-pic.py`, `sd-make-pic.py`, and `doujin_translator.py` deserve separate packaged targets.
