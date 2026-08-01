import csv
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QSpinBox, QCheckBox, QFileDialog, QListWidget, QListWidgetItem,
                             QMessageBox, QFrame, QAbstractItemView, QListView, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QIcon

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

# ---------------------------------------------------------------------------
#  Perceptual hash for image deduplication
# ---------------------------------------------------------------------------

def _is_image_file(filepath):
    return os.path.splitext(str(filepath))[1].lower() in SUPPORTED_EXTENSIONS


def _compute_image_hash(image_path, hash_size=32):
    """Compute a simple perceptual hash (average hash) for an image."""
    try:
        img = Image.open(image_path)
        img = img.convert('L').resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = ''.join('1' if p > avg else '0' for p in pixels)
        hex_hash = hex(int(bits, 2))[2:].zfill(hash_size * hash_size // 4)
        return hex_hash
    except Exception:
        return None


def _hamming_distance_hex(h1, h2):
    """Compute hamming distance between two hex string hashes."""
    if not h1 or not h2 or len(h1) != len(h2):
        return 999999
    try:
        b1 = int(h1, 16)
        b2 = int(h2, 16)
        xor = b1 ^ b2
        return xor.bit_count()
    except Exception:
        return 999999


# ---------------------------------------------------------------------------
#  Lightweight person detection via WD14 ONNX model
# ---------------------------------------------------------------------------

# Cache for the WD14 model and person-related label indices (loaded once)
_WD14_PERSON_CACHE = None
_WD14_PERSON_CACHE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
#  Analyzed image history (persistent to avoid re-processing same images)
# ---------------------------------------------------------------------------

_HISTORY_LOCK = threading.Lock()
_HISTORY_CACHE = None  # set of absolute paths, loaded lazily


def _get_history_path():
    return _resolve_project_path("conf/analyzed_history.json")


def _get_history_set():
    """Load the analyzed history set (lazy, thread-safe, cached in memory)."""
    global _HISTORY_CACHE
    if _HISTORY_CACHE is not None:
        return _HISTORY_CACHE
    with _HISTORY_LOCK:
        if _HISTORY_CACHE is not None:
            return _HISTORY_CACHE
        try:
            path = _get_history_path()
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    _HISTORY_CACHE = set(data.get("paths", []))
            else:
                _HISTORY_CACHE = set()
        except Exception:
            _HISTORY_CACHE = set()
        return _HISTORY_CACHE


def _invalidate_history_cache():
    """Force re-read of the history file next time _get_history_set is called."""
    global _HISTORY_CACHE
    with _HISTORY_LOCK:
        _HISTORY_CACHE = None


def add_to_analyzed_history(paths):
    """Record image paths as analyzed (persisted to conf/analyzed_history.json).

    Args:
        paths: iterable of str — absolute file paths to record.
    """
    if not paths:
        return
    abs_paths = []
    for p in paths:
        ap = os.path.abspath(p)
        if os.path.isfile(ap):
            abs_paths.append(ap)
    if not abs_paths:
        return

    existing = _get_history_set().copy()
    existing.update(abs_paths)

    try:
        hist_path = _get_history_path()
        os.makedirs(os.path.dirname(hist_path), exist_ok=True)
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump({"paths": sorted(list(existing))}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    _invalidate_history_cache()

# Person-related danbooru tag patterns (matches count + gender, and special tags)
_PERSON_TAG_PATTERNS = re.compile(
    r'^(\d+\+?(girl|boy|woman|man|person)s?)$'   # 1girl, 2boys, multiple_girls, 3+girls, etc.
    r'|^(solo|multiple_girls|multiple_boys)$'    # solo / multiple
    r'|^(female|male|girl|boy|woman|man)$'       # standalone gender tags
    , re.IGNORECASE
)
# Tags that explicitly indicate NO humans
_NO_HUMANS_TAG_PATTERNS = re.compile(
    r'^(no_humans|nobody|scenery|landscape)$', re.IGNORECASE
)

# WD14 category constants
_CAT_GENERAL = 0
_CAT_CHARACTER = 4
_CAT_RATING = 9


def _resolve_project_path(rel_path):
    """Resolve a path relative to the project root (two levels up from this file)."""
    if os.path.isabs(rel_path):
        return rel_path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(project_root, rel_path)


def _find_wd14_paths():
    """Find the WD14 model.onnx and selected_tags.csv paths."""
    candidates = [
        "models/wd14/model.onnx",
        "data/models/wd14/model.onnx",
        "wd14_tagger_model/model.onnx",
    ]
    for c in candidates:
        p = _resolve_project_path(c)
        if os.path.isfile(p):
            model_path = p
            tags_path = os.path.join(os.path.dirname(p), "selected_tags.csv")
            if os.path.isfile(tags_path):
                return model_path, tags_path
    return None, None


# Thresholds mirroring config defaults
_PERSON_GENERAL_THRESHOLD = 0.35
_PERSON_CHARACTER_THRESHOLD = 0.35

# Batch size for person detection (GPU-friendly)
_BATCH_SIZE = 8


def _select_onnx_providers():
    """Select best available ONNX execution providers, GPU-first."""
    providers = []
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        # DirectML — Windows GPU via DirectX 12, most universal
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        # CUDA — NVIDIA GPU
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        # CoreML — macOS GPU
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
    except Exception:
        pass
    providers.append("CPUExecutionProvider")
    return providers


def _load_wd14_person_detector():
    """Load WD14 ONNX session (GPU-accelerated if available) and extract person-related label indices (cached).

    Returns a dict with keys:
        session: onnxruntime.InferenceSession
        person_indices: list[int] — indices of person-related tags
        person_categories: list[int] — category (0/4/9) for each person tag
        target_size: int — required input size for the model
        provider: str — name of the active execution provider
    """
    global _WD14_PERSON_CACHE
    if _WD14_PERSON_CACHE is not None:
        return _WD14_PERSON_CACHE

    with _WD14_PERSON_CACHE_LOCK:
        if _WD14_PERSON_CACHE is not None:
            return _WD14_PERSON_CACHE
        try:
            import numpy as _np  # noqa: F811
            import onnxruntime as ort
        except Exception:
            _WD14_PERSON_CACHE = False
            return None

        model_path, tags_path = _find_wd14_paths()
        if not model_path or not tags_path:
            _WD14_PERSON_CACHE = False
            return None

        providers = _select_onnx_providers()
        provider_name = providers[0]
        try:
            session = ort.InferenceSession(
                model_path,
                providers=providers,
            )
            actual_providers = session.get_providers()
            provider_name = actual_providers[0] if actual_providers else "CPUExecutionProvider"
        except Exception:
            # Fallback to CPU only
            try:
                session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                provider_name = "CPUExecutionProvider"
            except Exception:
                _WD14_PERSON_CACHE = False
                return None

        # Parse labels CSV: tag_id, name, category
        person_indices = []
        person_categories = []
        try:
            with open(tags_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    name = str(row.get("name", "")).strip().lower().replace(" ", "_")
                    cat_str = str(row.get("category", "0")).strip()
                    cat = int(cat_str) if cat_str.isdigit() else 0
                    if _PERSON_TAG_PATTERNS.match(name) and not _NO_HUMANS_TAG_PATTERNS.match(name):
                        person_indices.append(idx)
                        person_categories.append(cat)
        except Exception:
            _WD14_PERSON_CACHE = False
            return None

        if not person_indices:
            _WD14_PERSON_CACHE = False
            return None

        # Determine target input size
        input_shape = session.get_inputs()[0].shape
        positive_dims = [d for d in input_shape if isinstance(d, int) and d > 4]
        target_size = max(positive_dims) if positive_dims else 448

        cache = {
            "session": session,
            "person_indices": person_indices,
            "person_categories": person_categories,
            "target_size": target_size,
            "provider": provider_name,
        }
        _WD14_PERSON_CACHE = cache
        return cache


def _rgb_img_to_array(img, target_size):
    """Convert a PIL RGB image to a (target_size, target_size, 3) float32 array, BGR."""
    import numpy as np

    w, h = img.size
    scale = float(target_size) / float(max(w, h))
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    resized = img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    canvas.paste(resized, ((target_size - resized_w) // 2, (target_size - resized_h) // 2))
    arr = np.asarray(canvas, dtype=np.float32)
    return arr[:, :, ::-1]  # RGB -> BGR


def _check_person_probs(probs, person_indices, person_categories):
    """Check a single probability vector for person-related tag scores."""
    for idx, cat in zip(person_indices, person_categories):
        if idx >= len(probs):
            break
        score = float(probs[idx])
        threshold = _PERSON_CHARACTER_THRESHOLD if cat == _CAT_CHARACTER else _PERSON_GENERAL_THRESHOLD
        if score >= threshold:
            return True
    return False


def _batch_detect_person(image_paths, progress_callback=None):
    """Detect people in a list of images using batched GPU-accelerated WD14 inference.

    Returns:
        list of bool: True if person detected, False otherwise (same length as input).
        None: model not available.
    """
    cache = _load_wd14_person_detector()
    if cache is None:
        return None

    import numpy as np

    session = cache["session"]
    person_indices = cache["person_indices"]
    person_categories = cache["person_categories"]
    target_size = cache["target_size"]
    input_name = session.get_inputs()[0].name
    total = len(image_paths)
    results = [True] * total  # conservative default

    for batch_start in range(0, total, _BATCH_SIZE):
        batch_end = min(batch_start + _BATCH_SIZE, total)
        batch_paths = image_paths[batch_start:batch_end]
        batch_count = len(batch_paths)

        # Preprocess batch — pad to uniform size, stack into (N, H, W, C)
        batch_tensor = np.zeros((batch_count, target_size, target_size, 3), dtype=np.float32)
        valid_mask = [True] * batch_count
        for i, path in enumerate(batch_paths):
            try:
                img = Image.open(path)
                if img.mode == "RGBA":
                    base = Image.new("RGBA", img.size, (255, 255, 255, 255))
                    base.alpha_composite(img)
                    img = base.convert("RGB")
                elif img.mode != "RGB":
                    img = img.convert("RGB")
                batch_tensor[i] = _rgb_img_to_array(img, target_size)
            except Exception:
                valid_mask[i] = False

        # Run ONNX inference on the batch
        try:
            output = session.run(None, {input_name: batch_tensor})[0]
        except Exception:
            # Fallback: skip batch on error, keep conservative defaults
            if progress_callback:
                for i in range(batch_count):
                    progress_callback(batch_start + i + 1, total)
            continue

        if len(output.shape) == 2:
            probs_batch = output  # (N, num_tags)
        elif len(output.shape) == 3:
            probs_batch = output[:, 0, :]  # (N, 1, num_tags) -> (N, num_tags)
        else:
            probs_batch = output

        for i in range(batch_count):
            if not valid_mask[i]:
                continue
            try:
                results[batch_start + i] = _check_person_probs(
                    probs_batch[i], person_indices, person_categories
                )
            except Exception:
                pass  # keep conservative default

        if progress_callback:
            progress_callback(batch_end, total)

    return results


# ---------------------------------------------------------------------------
#  Background scan thread
# ---------------------------------------------------------------------------

class ImageScanThread(QThread):
    """Background thread to scan directory, compute hashes, and optionally detect people."""
    progress_signal = pyqtSignal(int, int)   # current, total
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(list)        # list of (path, hash)

    # Subdirectory sampling
    SUBDIR_SAMPLE_THRESHOLD = 5
    SUBDIR_SAMPLE_RATIO = 0.5

    # Parallelism
    HASH_WORKERS = 6

    def __init__(self, directory, recursive=True, filter_person=False, exclude_analyzed=False):
        super().__init__()
        self.directory = directory
        self.recursive = recursive
        self.filter_person = filter_person
        self.exclude_analyzed = exclude_analyzed

    def run(self):
        # ---- Load analyzed history if needed ----
        analyzed_set = set()
        if self.exclude_analyzed:
            analyzed_set = _get_history_set()
            if analyzed_set:
                self.status_signal.emit(f"历史记录已加载 {len(analyzed_set)} 条已分析路径。")

        all_files = []
        if self.recursive:
            sampled_dir_count = 0
            total_subdirs_seen = 0
            for root, dirs, files in os.walk(self.directory):
                if len(dirs) > self.SUBDIR_SAMPLE_THRESHOLD:
                    keep_count = max(1, int(len(dirs) * self.SUBDIR_SAMPLE_RATIO))
                    total_subdirs_seen += len(dirs)
                    sampled_dir_count += 1
                    kept = random.sample(dirs, keep_count)
                    dirs[:] = kept

                for f in files:
                    fpath = os.path.join(root, f)
                    if _is_image_file(fpath):
                        all_files.append(fpath)

            if sampled_dir_count > 0:
                self.status_signal.emit(
                    f"递归扫描: 在 {sampled_dir_count} 个层级中随机采样子目录 "
                    f"(共遍历 {total_subdirs_seen} 个子目录) 以提高多样性..."
                )
        else:
            try:
                for f in os.listdir(self.directory):
                    fpath = os.path.join(self.directory, f)
                    if os.path.isfile(fpath) and _is_image_file(fpath):
                        all_files.append(fpath)
            except Exception:
                pass

        # ---- Filter out previously analyzed images (before hash computation) ----
        if analyzed_set:
            before = len(all_files)
            all_files = [f for f in all_files if os.path.abspath(f) not in analyzed_set]
            skipped = before - len(all_files)
            if skipped > 0:
                self.status_signal.emit(f"已排除 {skipped} 张历史已分析的图片。")

        total = len(all_files)
        self.status_signal.emit(f"找到 {total} 张图片，正在并行计算特征...")
        self.progress_signal.emit(0, max(total, 1))

        # ---- Pass 1: parallel hash computation ----
        results = []
        completed = 0
        with ThreadPoolExecutor(max_workers=self.HASH_WORKERS) as executor:
            future_to_path = {executor.submit(self._hash_one, fpath): fpath for fpath in all_files}
            for future in as_completed(future_to_path):
                if self.isInterruptionRequested():
                    for f in future_to_path.values():
                        f.cancel()
                    break
                path = future_to_path[future]
                try:
                    img_hash = future.result()
                    if img_hash:
                        results.append((path, img_hash))
                except Exception:
                    pass
                completed += 1
                if completed % 20 == 0 or completed == total:
                    self.progress_signal.emit(completed, total)
                    self.status_signal.emit(f"特征计算中... {completed}/{total}")

        if self.isInterruptionRequested():
            return

        self.status_signal.emit(f"特征计算完成，共 {len(results)} 张可用图片。")

        # ---- Pass 2: person filter (optional, GPU-accelerated batch) ----
        if self.filter_person and results:
            detector_cache = _load_wd14_person_detector()
            if detector_cache is None:
                self.status_signal.emit("⚠ 人物过滤未启用：WD14 模型未找到或加载失败，跳过人物检测。")
            else:
                provider = detector_cache.get("provider", "CPU")
                total_person = len(results)
                paths_only = [p for p, _ in results]

                self.status_signal.emit(
                    f"正在批量检测人物 ({provider} 加速, 每批 {_BATCH_SIZE} 张)... 0/{total_person}"
                )
                self.progress_signal.emit(0, max(total_person, 1))

                # Progress callback for the batched detector
                last_report = [0]
                def _batch_progress(done, _total):
                    if self.isInterruptionRequested():
                        return
                    if done - last_report[0] >= 10 or done >= _total:
                        last_report[0] = done
                        self.progress_signal.emit(done, _total)
                        self.status_signal.emit(
                            f"人物检测中... {done}/{_total}"
                        )

                person_flags = _batch_detect_person(paths_only, progress_callback=_batch_progress)

                if self.isInterruptionRequested():
                    return

                if person_flags is None:
                    self.status_signal.emit("⚠ 人物检测失败，保留全部图片。")
                else:
                    filtered = []
                    filtered_count = 0
                    for (path, h), has_person in zip(results, person_flags):
                        if has_person:
                            filtered.append((path, h))
                        else:
                            filtered_count += 1
                    self.status_signal.emit(
                        f"人物过滤完成 ({provider}): {len(filtered)} 张保留, {filtered_count} 张已过滤（无人）"
                    )
                    results = filtered

        self.finished_signal.emit(results)

    @staticmethod
    def _hash_one(fpath):
        """Hash a single image file. Static method for safe pickling with ThreadPoolExecutor."""
        if os.path.isfile(fpath):
            return _compute_image_hash(fpath)
        return None


# ---------------------------------------------------------------------------
#  Dialog
# ---------------------------------------------------------------------------

class DirectoryBatchSelectorDialog(QDialog):
    """Dialog for batch-selecting images from a directory with dedup and person filter."""

    THUMB_SIZE = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("从目录批量选择图片")
        self.resize(1100, 750)
        self.setMinimumSize(780, 500)

        # State
        self._all_scanned = []          # list of (path, hash)
        self._selected_paths = []       # currently selected paths
        self._selected_hashes = []      # corresponding hashes
        self._candidate_pool = []       # remaining candidates (not selected)
        self._scan_thread = None

        self._init_ui()
        self._update_buttons()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # ---- Row 1: Directory selection ----
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("目录:"))
        self.dir_label = QLabel("未选择目录")
        self.dir_label.setStyleSheet("QLabel { border: 1px solid #ccc; padding: 4px; background: #fafafa; }")
        dir_layout.addWidget(self.dir_label, stretch=1)
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self._browse_directory)
        dir_layout.addWidget(self.browse_btn)
        layout.addLayout(dir_layout)

        # ---- Row 2: Count + recursive + person filter ----
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("选取图片数量:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 9999)
        self.count_spin.setValue(10)
        params_layout.addWidget(self.count_spin)

        self.recursive_cb = QCheckBox("递归查找子目录")
        self.recursive_cb.setChecked(True)
        params_layout.addWidget(self.recursive_cb)

        self.filter_person_cb = QCheckBox("仅保留含人物的图片")
        self.filter_person_cb.setToolTip("使用本地 WD14 模型检测图片中是否包含人物，自动过滤风景/物品等不含人物的图片。")
        params_layout.addWidget(self.filter_person_cb)

        self.exclude_analyzed_cb = QCheckBox("排除已分析过的图片")
        self.exclude_analyzed_cb.setToolTip("跳过此前已加入分析队列的图片，避免重复分析。")
        params_layout.addWidget(self.exclude_analyzed_cb)

        params_layout.addStretch()

        self.scan_btn = QPushButton("扫描目录")
        self.scan_btn.setFixedHeight(32)
        self.scan_btn.clicked.connect(self._start_scan)
        params_layout.addWidget(self.scan_btn)
        layout.addLayout(params_layout)

        # ---- Progress bar ----
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # ---- Status label ----
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)

        # ---- Separator ----
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # ---- Thumbnail grid ----
        self.list_label = QLabel("已选择的图片 (双击查看原图，可多选后移除，路径悬浮显示):")
        layout.addWidget(self.list_label)

        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListView.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(self.THUMB_SIZE, self.THUMB_SIZE))
        self.list_widget.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_widget.setFlow(QListView.Flow.LeftToRight)
        self.list_widget.setWrapping(True)
        self.list_widget.setSpacing(6)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.setStyleSheet(
            "QListWidget::item { padding: 4px; }"
            "QListWidget::item:selected { background-color: #cce5ff; }"
        )
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.list_widget.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self.list_widget, stretch=1)

        # ---- Action buttons row ----
        action_layout = QHBoxLayout()
        self.remove_btn = QPushButton("移除选中")
        self.remove_btn.clicked.connect(self._remove_selected)
        action_layout.addWidget(self.remove_btn)

        self.remove_all_btn = QPushButton("全部移除")
        self.remove_all_btn.clicked.connect(self._remove_all)
        action_layout.addWidget(self.remove_all_btn)

        action_layout.addStretch()

        self.refill_btn = QPushButton("重新查找补足")
        self.refill_btn.clicked.connect(self._refill)
        action_layout.addWidget(self.refill_btn)

        self.rescan_btn = QPushButton("重新扫描")
        self.rescan_btn.clicked.connect(self._start_scan)
        action_layout.addWidget(self.rescan_btn)
        layout.addLayout(action_layout)

        # ---- Bottom: Confirm / Cancel ----
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(self.cancel_btn)

        self.confirm_btn = QPushButton("确认加入分析队列")
        self.confirm_btn.setFixedHeight(36)
        self.confirm_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        self.confirm_btn.clicked.connect(self._confirm)
        bottom_layout.addWidget(self.confirm_btn)
        layout.addLayout(bottom_layout)

        self.setLayout(layout)

    def _browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if directory:
            self.dir_label.setText(os.path.abspath(directory))
            self._start_scan()

    def _start_scan(self):
        directory = self.dir_label.text()
        if not directory or directory == "未选择目录":
            QMessageBox.warning(self, "提示", "请先选择一个目录。")
            return
        if not os.path.isdir(directory):
            QMessageBox.warning(self, "提示", f"目录不存在: {directory}")
            return

        # Cancel existing scan
        if self._scan_thread and self._scan_thread.isRunning():
            self._scan_thread.requestInterruption()
            self._scan_thread.wait(1000)

        self._all_scanned = []
        self._selected_paths = []
        self._selected_hashes = []
        self._candidate_pool = []
        self.list_widget.clear()

        self.scan_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.recursive_cb.setEnabled(False)
        self.filter_person_cb.setEnabled(False)
        self.exclude_analyzed_cb.setEnabled(False)
        self.count_spin.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self._scan_thread = ImageScanThread(
            directory,
            self.recursive_cb.isChecked(),
            self.filter_person_cb.isChecked(),
            self.exclude_analyzed_cb.isChecked(),
        )
        self._scan_thread.progress_signal.connect(self._on_scan_progress)
        self._scan_thread.status_signal.connect(self.status_label.setText)
        self._scan_thread.finished_signal.connect(self._on_scan_finished)
        self._scan_thread.start()

    def _on_scan_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_scan_finished(self, results):
        self._all_scanned = results
        self.progress_bar.setVisible(False)
        self.scan_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.recursive_cb.setEnabled(True)
        self.filter_person_cb.setEnabled(True)
        self.exclude_analyzed_cb.setEnabled(True)
        self.count_spin.setEnabled(True)

        if not self._all_scanned:
            self.status_label.setText("未找到任何图片文件（或被人物过滤全部排除）。")
            self._update_buttons()
            return

        self.status_label.setText(f"扫描完成，共 {len(self._all_scanned)} 张可用图片。正在自动选择...")
        self._auto_select()

        # If count is below target after dedup and there are remaining candidates,
        # auto-refill up to 3 rounds (each round tries different shuffled order).
        target = self.count_spin.value()
        retry = 0
        while len(self._selected_paths) < target and self._candidate_pool and retry < 3:
            self.status_label.setText(
                f"已选 {len(self._selected_paths)}/{target} 张，自动补足中 (第{retry + 1}轮)..."
            )
            self._refill()
            retry += 1

        if self._candidate_pool:
            # Some candidates still available — user can manually refill later
            self.status_label.setText(
                f"已选择 {len(self._selected_paths)}/{target} 张（候选池剩余 {len(self._candidate_pool)} 张），可手动补足或确认。"
            )
        else:
            self.status_label.setText(
                f"已选择 {len(self._selected_paths)}/{target} 张，候选池已空。"
            )

        self._refresh_list()
        self._update_buttons()

    def _auto_select(self):
        """Select images using dedup algorithm. Avoids selecting similar images."""
        target_count = self.count_spin.value()
        if target_count <= 0 or not self._all_scanned:
            return

        pool = list(self._all_scanned)
        random.shuffle(pool)

        selected = []
        selected_hashes = []

        for path, h in pool:
            if len(selected) >= target_count:
                break
            too_similar = False
            for existing_h in selected_hashes:
                dist = _hamming_distance_hex(h, existing_h)
                if dist <= 60:
                    too_similar = True
                    break
            if not too_similar:
                selected.append((path, h))
                selected_hashes.append(h)

        self._selected_paths = [p for p, _ in selected]
        self._selected_hashes = selected_hashes
        selected_set = set(self._selected_paths)
        self._candidate_pool = [(p, h) for p, h in self._all_scanned if p not in selected_set]

        if len(self._selected_paths) < target_count:
            self.status_label.setText(
                '已选择 {} 张（目标 {} 张），已尽量去重，自动补足中...'.format(
                    len(self._selected_paths), target_count
                )
            )
        else:
            self.status_label.setText(f"已选择 {len(self._selected_paths)} 张，去重完成。")

    @staticmethod
    def _truncate_text(text, max_chars=28):
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."

    def _refresh_list(self):
        """Refresh the list widget with current selection as a grid of large thumbnails."""
        self.list_widget.clear()
        if not self._selected_paths:
            return

        icon_size = self.THUMB_SIZE
        for path in self._selected_paths:
            # Load directly via QPixmap(filepath) to avoid PIL ImagingCore type issues
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    icon_size, icon_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            else:
                pixmap = QPixmap()

            item = QListWidgetItem()
            if not pixmap.isNull():
                item.setIcon(QIcon(pixmap))
            filename = os.path.basename(path)
            item.setText(self._truncate_text(filename))
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.list_widget.addItem(item)

    def _update_buttons(self):
        has_items = self.list_widget.count() > 0
        self.remove_btn.setEnabled(has_items and len(self.list_widget.selectedItems()) > 0)
        self.remove_all_btn.setEnabled(has_items)
        self.refill_btn.setEnabled(bool(self._candidate_pool))
        self.confirm_btn.setEnabled(has_items)

    def _remove_selected(self):
        items = self.list_widget.selectedItems()
        if not items:
            return
        remove_paths = set()
        for item in items:
            path = item.data(Qt.ItemDataRole.UserRole)
            remove_paths.add(path)

        kept_paths = []
        kept_hashes = []
        for i, p in enumerate(self._selected_paths):
            if p not in remove_paths:
                kept_paths.append(p)
                kept_hashes.append(self._selected_hashes[i])
            else:
                h = self._selected_hashes[i]
                self._candidate_pool.append((p, h))

        self._selected_paths = kept_paths
        self._selected_hashes = kept_hashes
        self._refresh_list()
        self._update_buttons()

    def _remove_all(self):
        removed = list(zip(self._selected_paths, self._selected_hashes))
        self._candidate_pool.extend(removed)
        self._selected_paths = []
        self._selected_hashes = []
        self._refresh_list()
        self._update_buttons()

    def _refill(self):
        target_count = self.count_spin.value()
        if len(self._selected_paths) >= target_count:
            self.status_label.setText("已满足目标数量，无需补足。")
            return
        if not self._candidate_pool:
            self.status_label.setText("候选池已空，请重新扫描。")
            return

        random.shuffle(self._candidate_pool)

        needed = target_count - len(self._selected_paths)
        added = 0
        new_selected = list(self._selected_paths)
        new_hashes = list(self._selected_hashes)
        new_pool = []

        for path, h in self._candidate_pool:
            if added >= needed:
                new_pool.append((path, h))
                continue
            too_similar = False
            for existing_h in new_hashes:
                dist = _hamming_distance_hex(h, existing_h)
                if dist <= 60:
                    too_similar = True
                    break
            if too_similar:
                new_pool.append((path, h))
                continue
            new_selected.append(path)
            new_hashes.append(h)
            added += 1

        self._selected_paths = new_selected
        self._selected_hashes = new_hashes
        self._candidate_pool = new_pool

        if added > 0:
            self.status_label.setText(f"已补充 {added} 张，当前共 {len(self._selected_paths)} 张。")
        else:
            self.status_label.setText(
                f"未能补充新图片（候选池中剩余图片均与已选图片高度相似），当前共 {len(self._selected_paths)} 张。"
            )

        self._refresh_list()
        self._update_buttons()

    def _on_item_double_clicked(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and os.path.isfile(path):
            os.startfile(path)

    def _confirm(self):
        if not self._selected_paths:
            QMessageBox.warning(self, "提示", "请至少选择一张图片。")
            return
        self.accept()

    def get_selected_paths(self):
        """Return the list of selected image paths after dialog is accepted."""
        return list(self._selected_paths)
