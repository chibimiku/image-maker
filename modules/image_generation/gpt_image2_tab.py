"""gpt-image-2 专用 Tab（同一界面切换 new.aigc2d / autodl 两个站点）。

- 站点下拉：
  - `new.aigc2d` → `apis.aigc-2d-gpt`，走 `generate_image_aigc2d_gpt`（尺寸仅 3 档）
  - `autodl` → `apis.autodl`，走 `generate_image_openai_image`（尺寸含 auto / 1792x1024）
- 模式下拉：生图 / 编辑
  - 生图：不挂参考图 → `/v1/images/generations`；挂了参考图 → 自动走 `/v1/images/edits` 垫图
  - 编辑：至少 1 张图
- 两个站点都支持 gpt-image-2，参考图上限 16 张
- 模型下拉列出该站点的 gpt-image 家族（aigc2d 含 2.5 的 flare / sunburst 及其 `-c` 计费版，
  autodl 只有 gpt-image-2）；下拉仍可手输，另有「刷新模型列表」按 `GET /v1/models` 现拉
- 参考图用缩略图网格（`ref_image_grid.RefImageGrid`）展示：可点 × 删除、拖动缩略图调整顺序，
  单击预览、双击打开所在目录；顺序即提交给 `/images/edits` 的 `image[]` 顺序
- 配置只从 conf/config.json 读取（`apis.*` + 顶层 `gpt_image2` 记忆），不受「设置」里全局 API 类型影响
"""
import json
import logging
import os
import time
import traceback

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from modules.image_generation.ref_image_grid import RefImageGrid, collect_image_paths
from modules.others.api_backend import (
    CONFIG_DIR,
    GPT_IMAGE2_MAX_REFERENCE_IMAGES,
    GPT_IMAGE2_MODEL_DEFAULT,
    GPT_IMAGE2_MODEL_NOTES,
    GPT_IMAGE2_OUTPUT_FORMATS,
    GPT_IMAGE2_SITE_AIGC2D,
    GPT_IMAGE2_SITE_API_TYPES,
    GPT_IMAGE2_SITE_AUTODL,
    GPT_IMAGE2_SITE_DEFAULT_SIZES,
    GPT_IMAGE2_SITE_FILE_PREFIXES,
    GPT_IMAGE2_SITE_MODELS,
    GPT_IMAGE2_SITE_SAVE_SUB_DIRS,
    GPT_IMAGE2_SITE_SIZES,
    GPT_IMAGE2_SIZE_PORTRAIT,
    GPT_IMAGE2_SIZE_SQUARE,
    generate_image_aigc2d_gpt,
    generate_image_openai_image,
    generate_image_repaint,
    get_api_config,
    is_gpt_image_model,
    list_available_models,
    normalize_gpt_image2_size,
    pick_gpt_image_models,
    resolve_images_endpoint,
    resolve_models_endpoint,
)
from utils.gpt_image_optimize import (
    ASPECT_RATIO_AUTO,
    ASPECT_RATIO_OPTIONS as REPAINT_DEFAULT_ASPECT_OPTIONS,
    DEFAULTS as REPAINT_DEFAULTS,
    build_repaint_prompt,
    image_aspect_ratio_label,
    is_auto_aspect_ratio,
    load_config as load_repaint_config,
    plan_output as plan_repaint_output,
    resolve_aspect_ratio,
    save_config as save_repaint_config,
)

# 站点/接口映射统一来自 api_backend（GUI 与 tools/gpt_image2_gen.py 共用一份，避免漂移）
SITE_AIGC2D = GPT_IMAGE2_SITE_AIGC2D
SITE_AUTODL = GPT_IMAGE2_SITE_AUTODL
API_TYPE_BY_SITE = GPT_IMAGE2_SITE_API_TYPES
SAVE_SUB_DIR_BY_SITE = GPT_IMAGE2_SITE_SAVE_SUB_DIRS
CONFIG_STYLES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "conf", "config-styles.json")
STYLE_NONE = "默认(无附加)"
FILE_PREFIX_BY_SITE = GPT_IMAGE2_SITE_FILE_PREFIXES
DEFAULT_SIZE_BY_SITE = GPT_IMAGE2_SITE_DEFAULT_SIZES

SIZE_LABELS = {
    "auto": "自动 auto",
    "1024x1024": "方形 1024x1024",
    "1536x1024": "横向 1536x1024",
    "1792x1024": "宽幅 1792x1024",
    "1024x1536": "纵向 1024x1536",
}
# aigc2d 通道只认 3 档尺寸；autodl 通道含 auto / 1792x1024（清单在 api_backend）
SIZE_CHOICES_BY_SITE = {
    site: tuple((SIZE_LABELS.get(value, value), value) for value in sizes)
    for site, sizes in GPT_IMAGE2_SITE_SIZES.items()
}
QUALITY_CHOICES = (
    ("high 精细(较慢较贵)", "high"),
    ("medium 标准", "medium"),
    ("low 快速草稿", "low"),
    ("auto 服务端默认", "auto"),
)

CONFIG_IMAGE_FILE = os.path.join(CONFIG_DIR, "config.json")   # 图片配置已并入统一 config.json
CONFIG_NODE = "gpt_image2"

MODE_GENERATE = "生图"
MODE_EDIT = "编辑"
# 重绘：以「已有产物」为输入走 Gemini 重绘提线（见 docs/gpt-image-optimize/）
MODE_REPAINT = "重绘(Gemini优化产物)"
MODE_CHOICES = (MODE_GENERATE, MODE_EDIT, MODE_REPAINT)
IMAGE_FILTER = "图片 (*.png *.jpg *.jpeg *.webp *.bmp)"
# 尺寸=自动：按参考图实际比例在 1536x1024 / 1024x1536 / 1024x1024 里挑（宽图不会输出竖图）
SIZE_FOLLOW_INPUT = "auto-follow-input"


def _read_config_image() -> dict:
    try:
        with open(CONFIG_IMAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _set_combo_by_data(combo: QComboBox, value, fallback=None):
    for candidate in (value, fallback):
        if candidate is None:
            continue
        index = combo.findData(candidate)
        if index < 0:
            index = combo.findText(str(candidate))
        if index >= 0:
            combo.setCurrentIndex(index)
            return


def mask_secret(value: str) -> str:
    """把 key/token 打成 sk-abc…wxyz，便于在日志里确认用的是哪把 key 而不泄露。"""
    text = str(value or "")
    if not text:
        return "(空)"
    if len(text) <= 12:
        return text[:2] + "***"
    return f"{text[:6]}…{text[-4:]}(len={len(text)})"


class BackendLogHandler(logging.Handler):
    """把 api_backend 的 logger 记录转发到 Tab 日志栏（含请求 URL / payload / 服务器原始返回 / 错误）。"""

    def __init__(self, sink):
        super().__init__(level=logging.INFO)
        self._sink = sink

    def emit(self, record):
        try:
            self._sink(self.format(record))
        except Exception:
            pass


def format_request_dump(params: dict, api_cfg: dict, site: str, api_type: str, mode: str) -> list:
    """生成提交前的详细日志行（不含明文 key）。"""
    images = list(params.get("image_paths") or [])
    base_url = str(api_cfg.get("base_url") or "")
    endpoint = resolve_images_endpoint(base_url, has_images=bool(images), api_type=api_type)
    lines = [
        f"[请求] 站点={site}  API类型={api_type}  模式={mode}",
        f"[请求] 端点={endpoint}  （{'参考图≥1 → /images/edits' if images else '无参考图 → /images/generations'}）",
        f"[请求] 模型={params.get('model')}  尺寸={params.get('size')}  画质={params.get('quality') or '(配置默认)'}  "
        f"格式={params.get('output_format') or '(配置默认)'}  张数={params.get('n', 1)}",
        f"[请求] base_url={base_url or '(未配置)'}  api_key={mask_secret(api_cfg.get('api_key'))}  "
        f"timeout={api_cfg.get('timeout', '(默认)')}s  max_retries={api_cfg.get('max_retries', 1)}",
    ]
    prompt = str(params.get("prompt") or "")
    lines.append(f"[请求] prompt（{len(prompt)} 字）: {prompt[:200]}{'…' if len(prompt) > 200 else ''}")
    if images:
        lines.append(f"[请求] 参考图 {len(images)} 张:")
        for path in images:
            size_text = f"{os.path.getsize(path)} bytes" if os.path.isfile(path) else "文件不存在!"
            lines.append(f"        - {path}  ({size_text})")
    else:
        lines.append("[请求] 参考图: 无")
    return lines


class PricingWorker(QThread):
    """后台拉价目表（/api/pricing），避免首屏卡在网络请求上。"""
    done = pyqtSignal(bool)

    def run(self):
        try:
            from utils.cost_estimate import fetch_pricing
            pricing = fetch_pricing()
            ok = bool((pricing or {}).get("models"))
        except Exception:  # noqa: BLE001
            ok = False
        self.done.emit(ok)


class PostProcessWorker(QThread):
    """后台跑勾选的后处理工序（结构线叠加 / 局部重绘+羽化贴回）。"""
    log = pyqtSignal(str)
    done = pyqtSignal(list)

    def __init__(self, paths, steps, firmware=None, parent=None):
        super().__init__(parent)
        self.paths = list(paths or [])
        self.steps = steps or {}
        self.firmware = firmware

    def run(self):
        try:
            from utils.post_process import run_pipeline
            result = run_pipeline(self.paths, self.steps, firmware=self.firmware,
                                  log_callback=self.log.emit)
        except Exception as exc:  # noqa: BLE001 - 后处理失败不能让结果列表空掉
            self.log.emit(f"[后处理] 失败，改为使用重绘/生图原始产物：{type(exc).__name__}: {exc}")
            result = list(self.paths)
        self.done.emit(result)


class GptImage2Worker(QThread):
    """按站点调用对应后端；aigc2d 通道才支持取消与实时日志。

    当 `repaint_params` 非空时走「生图 → 重绘」链：先出图，再逐张送去 Gemini 重绘提线，
    **以重绘产物作为最终产物**（原始产物路径仍会写进日志，便于回溯 / 对比）。
    """

    log = pyqtSignal(str)
    done = pyqtSignal(list)
    error = pyqtSignal(str)
    raw_done = pyqtSignal(list)   # 链式模式下先回传「未重绘的原始产物」

    def __init__(self, backend, params: dict, parent=None, repaint_params: dict = None):
        super().__init__(parent)
        self.backend = backend
        self.params = dict(params or {})
        self.repaint_params = dict(repaint_params) if repaint_params else None

    def run(self):
        params = dict(self.params)
        backend_name = getattr(self.backend, "__name__", str(self.backend))
        if self.backend is generate_image_aigc2d_gpt:
            params.setdefault("cancel_check", lambda: self.isInterruptionRequested())
            params.setdefault("log_callback", lambda message: self.log.emit(str(message)))

        # 把 api_backend 的 logger 接到日志栏：请求 URL / 请求体 / 服务器原始返回 / 错误原因都会出现
        handler = BackendLogHandler(lambda message: self.log.emit(message))
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
        backend_logger = logging.getLogger("whatai_logger")
        backend_logger.addHandler(handler)
        started = time.perf_counter()
        self.log.emit(f"[后端] 调用 {backend_name}(...) ...")
        try:
            files = self.backend(**params)
            elapsed = time.perf_counter() - started
            if isinstance(files, dict):
                files = files.get("saved_files") or []
            saved = [p for p in (files or []) if p]
            if saved:
                self.log.emit(f"[后端] 成功返回 {len(saved)} 张，耗时 {elapsed:.1f}s")
            else:
                self.log.emit(
                    f"[后端] 返回 0 张图片，耗时 {elapsed:.1f}s —— 失败原因见上方后端日志"
                    "（HTTP 状态码 / 服务器原始返回 / data 节点缺失 / 审核拦截 / 网络异常）"
                )
                self.done.emit([])
                return
            if not self.repaint_params:
                self.done.emit(saved)
                return

            self.raw_done.emit(list(saved))
            rp = dict(self.repaint_params)
            rp.setdefault("cancel_check", lambda: self.isInterruptionRequested())
            rp.setdefault("log_callback", lambda message: self.log.emit(str(message)))
            rp["return_metadata"] = False
            rp_started = time.perf_counter()
            self.log.emit(f"===== 进入重绘阶段：{len(saved)} 张待优化（模型 {rp.get('model')} @{rp.get('resolution')}）=====")
            repainted = generate_image_repaint(source_paths=saved, **rp)
            repainted = [p for p in (repainted or []) if p]
            self.log.emit(
                f"[重绘] 完成 {len(repainted)}/{len(saved)} 张，耗时 {time.perf_counter() - rp_started:.1f}s"
            )
            if not repainted:
                self.log.emit("[重绘] 未产出重绘结果，已回退为以原始生图产物作为结果。")
                self.done.emit(list(saved))
                return
            self.done.emit(list(repainted))
        except Exception as exc:  # noqa: BLE001 - 线程内异常统一回传 UI
            elapsed = time.perf_counter() - started
            self.log.emit(f"[后端] 抛异常（耗时 {elapsed:.1f}s）: {type(exc).__name__}: {exc}")
            self.log.emit(traceback.format_exc())
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            backend_logger.removeHandler(handler)


class ModelListWorker(QThread):
    """后台拉取站点模型列表（`GET /v1/models`），避免刷新按钮卡住界面。"""

    done = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, api_type: str, config_path: str = None, timeout: int = 20, parent=None):
        super().__init__(parent)
        self.api_type = api_type
        self.config_path = config_path
        self.timeout = timeout

    def run(self):
        try:
            models = list_available_models(
                api_type=self.api_type, config_path=self.config_path, timeout=self.timeout
            )
            self.done.emit(list(models or []))
        except Exception as exc:  # noqa: BLE001 - 线程内异常统一回传 UI
            self.error.emit(f"{type(exc).__name__}: {exc}")


class GptImage2Widget(QWidget):
    SIZE_FOLLOW_INPUT = SIZE_FOLLOW_INPUT          # 尺寸=自动（跟随参考图比例）
    """gpt-image-2 生图 / 编辑台（new.aigc2d 与 autodl 双站点）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._model_worker = None
        self._live_models = {}
        self._loading = False
        # 重绘模式会把固件正文填进提示词框；这里记录"上次填的是哪份固件"和"进入前的原文"
        self._repaint_last_preset = ""
        self._mode_prompt_backup = ""
        self._mode_prompt_active = ""
        self.setAcceptDrops(True)
        self.initUI()
        self.load_defaults()
        self.refresh_api_hint()
        # 长度护栏 + 成本估算（先用内置价目快照显示，首次显示窗口时再拉一次网络价目刷新）
        self._connect_budget_signals()
        self._refresh_budget()

    def showEvent(self, event):  # noqa: N802 - Qt 命名
        super().showEvent(event)
        self._start_pricing_refresh()

    def _connect_budget_signals(self):
        """勾选框 / 画质 / 尺寸 / 参考图变化都会影响长度与成本估算，统一挂到刷新上。"""
        for widget, signal in ((getattr(self, "repaint_check", None), "toggled"),
                               (getattr(self, "structure_check", None), "toggled"),
                               (getattr(self, "local_repaint_check", None), "toggled"),
                               (getattr(self, "quality_combo", None), "currentIndexChanged"),
                               (getattr(self, "size_combo", None), "currentIndexChanged"),
                               (getattr(self, "site_combo", None), "currentIndexChanged")):
            if widget is not None:
                getattr(widget, signal).connect(lambda *_: self._refresh_budget())

    @property
    def image_paths(self) -> list:
        """当前参考图路径（顺序即提交给 `/images/edits` 的 image[] 顺序）。"""
        return self.image_grid.paths()

    # ---------------- UI ----------------
    def initUI(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.site_combo = QComboBox()
        self.site_combo.addItem(SITE_AIGC2D, SITE_AIGC2D)
        self.site_combo.addItem(SITE_AUTODL, SITE_AUTODL)
        self.site_combo.currentIndexChanged.connect(lambda _index: self.on_site_changed(self.current_site()))
        form.addRow("站点:", self.site_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(MODE_CHOICES))
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        self.mode_combo.setToolTip(
            "生图：gpt-image 出图（挂参考图则自动走 edits 垫图）\n"
            "编辑：对已挂的图做 gpt-image 编辑\n"
            "重绘：用 Gemini 3 Pro Image 对已有产物做『重绘提线』——修手、连通发丝、\n"
            "      保住蕾丝/褶皱结构（提示词固件在 prompts/gpt-image-optimize/）"
        )
        form.addRow("模式:", self.mode_combo)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)     # 下拉选常见模型，也允许手输新模型名
        self.model_combo.setToolTip(
            "下拉列出该站点可用的 gpt-image 系列（含 2.5 的 flare / sunburst 与 -c 计费版）；\n"
            "也可以直接手输模型名；点「刷新模型列表」按站点接口现拉一次。"
        )
        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.addWidget(self.model_combo, stretch=1)
        self.refresh_models_btn = QPushButton("刷新模型列表")
        self.refresh_models_btn.setToolTip("GET {base_url}/v1/models，按站点拉取可用模型")
        self.refresh_models_btn.clicked.connect(self.refresh_models)
        model_row.addWidget(self.refresh_models_btn)
        form.addRow("模型:", model_row)

        self.size_combo = QComboBox()
        form.addRow("尺寸:", self.size_combo)
        self.size_info_label = QLabel("")
        self.size_info_label.setStyleSheet("color: #666; font-size: 11px;")
        form.addRow("", self.size_info_label)

        self.quality_combo = QComboBox()
        for label, value in QUALITY_CHOICES:
            self.quality_combo.addItem(label, value)
        form.addRow("画质:", self.quality_combo)

        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(list(GPT_IMAGE2_OUTPUT_FORMATS))
        form.addRow("输出格式:", self.output_format_combo)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 10)
        self.n_spin.setValue(1)
        form.addRow("张数:", self.n_spin)
        layout.addLayout(form)

        # ---------------- 产物优化（Gemini 重绘提线） ----------------
        # 布局取舍：**功能开关（勾选框）常驻顶层可见**，只有 4 行参数收进可折叠区。
        # 原因：整块常驻会给 Tab 增加 ~270px 最小高度，一旦超过窗口高度，Qt 就会把可拉伸的
        # prompt_edit 压到最小值（输入框看起来过小）——所以折叠的是参数，不是功能本身。
        self._repaint_config = load_repaint_config()
        self.repaint_check = QCheckBox("出图后立即重绘优化（以重绘结果作为产物）")
        self.repaint_check.setToolTip(
            "勾选后：gpt-image 出图/编辑 → 每张自动送去 Gemini 重绘提线 → 结果列表里只有重绘产物。\n"
            "生图和编辑两种模式都支持（gpt-image 出的分辨率都偏低，重绘顺带提分辨率与细节）。\n"
            "原始产物路径会写进日志，方便回头对比。\n"
            "参数与提示词固件：prompts/gpt-image-optimize/（理论见 docs/gpt-image-optimize/）。"
        )
        self.repaint_check.toggled.connect(self.on_repaint_toggled)
        layout.addWidget(self.repaint_check)

        # ---------------- 后处理流水线（每道工序一个勾选框，可任意组合） ----------------
        # 结构线叠加 = 纯本地像素精修（不调模型）；局部重绘 = 按区域裁切重绘再羽化贴回（一次 Gemini 调用）。
        # 实测见 docs/gpt-image-tid-style/README.md §4.20 / §4.21。
        from utils.post_process import DEFAULT_FEATHER, DEFAULT_STRUCTURE_STRENGTH, REGION_LABELS
        self.structure_check = QCheckBox("结构线叠加（纯本地，不改内容；线条更连贯）")
        self.structure_check.setToolTip(
            "抽出画面里的长结构边，按局部色调整色后按强度叠回：纯本地、几毫秒、零 API 成本。\n"
            "实测三族平均骨架段长 +6~22%、端点密度最多 −16%，配色与饱和度几乎不动。"
        )
        layout.addWidget(self.structure_check)

        self.local_repaint_check = QCheckBox("局部重绘 + 羽化贴回（按区域强化头发 / 脸 / 裙子）")
        self.local_repaint_check.setToolTip(
            "按区域裁切 → 放大 → 走 Gemini 重绘（当前固件 + 区域强调句）→ 羽化贴回。\n"
            "头发糊就选「头发」，脸部用「脸部」；Gemini 没有 mask，这是最接近遮罩式局部重绘的做法。"
        )
        layout.addWidget(self.local_repaint_check)

        self.post_toggle_btn = QPushButton("▸ 后处理参数（结构线强度 / 局部区域）")
        self.post_toggle_btn.setCheckable(True)
        self.post_toggle_btn.setToolTip("展开/收起后处理参数；开关本身在上面两个勾选框。")
        self.post_toggle_btn.toggled.connect(lambda on: self.post_panel.setVisible(bool(on)))
        layout.addWidget(self.post_toggle_btn)

        self.post_panel = QWidget()
        self.post_panel.setVisible(False)
        post_form = QFormLayout(self.post_panel)
        post_form.setContentsMargins(16, 0, 0, 0)
        self.structure_strength_spin = QDoubleSpinBox()
        self.structure_strength_spin.setRange(0.05, 1.0)
        self.structure_strength_spin.setSingleStep(0.05)
        self.structure_strength_spin.setValue(DEFAULT_STRUCTURE_STRENGTH)
        self.structure_strength_spin.setToolTip("结构线不透明度；0.35 轻微、0.50 推荐、0.70 以上偏硬。")
        post_form.addRow("结构线强度:", self.structure_strength_spin)
        self.local_region_combo = QComboBox()
        for key, label in REGION_LABELS.items():
            self.local_region_combo.addItem(label, key)
        self.local_region_combo.setCurrentIndex(max(0, self.local_region_combo.findData("hair")))
        self.local_region_combo.setToolTip(
            "局部重绘区域：\n"
            "  头发 / 脸部 / 头部 / 裙子 / 上半身 —— 固定比例区\n"
            "  整个人物 —— 自动检测人物框（GrabCut 前景 + 人脸锚点，裁到不切发梢裙摆）\n"
            "  人物（不含面部）—— 同样的人物框，但**脸区保留原像素**（贴回时反向羽化排除），适合只想修衣服/头发又不想动脸\n"
            "  整张 —— 等于全图重绘（不如勾「出图后立即重绘优化」划算）"
        )
        post_form.addRow("局部重绘区域:", self.local_region_combo)
        self.local_feather_spin = QSpinBox()
        self.local_feather_spin.setRange(8, 200)
        self.local_feather_spin.setValue(DEFAULT_FEATHER)
        self.local_feather_spin.setToolTip("贴回时的羽化半径（像素），越大接缝越柔和、改动越局部。")
        post_form.addRow("贴回羽化:", self.local_feather_spin)
        layout.addWidget(self.post_panel)
        # 重绘双参考（源图 + 线锚图）：§⑲ 实测线条最连贯的配方
        self.post_dual_check = QCheckBox("重绘用双参考（源图+线锚图，线条更连贯）")
        self.post_dual_check.setChecked(True)
        self.post_dual_check.setToolTip(
            "重绘时把源图与自动生成的线锚图（白底长结构线）一起作为参考送入；\n"
            "实测线条连通性明显好于单参考（§⑲ E 组配方）。"
        )
        layout.addWidget(self.post_dual_check)
        # 色调校准 + 线条加墨
        self.post_tone_check = QCheckBox("色调校准（按参考图匹配亮度/饱和度）")
        self.post_tone_check.setToolTip(
            "目标选画风参考图时画面更深更饱和、线条更清楚（画质优先）；选输入照片更接近照片原色。"
        )
        self.post_tone_target = QComboBox()
        self.post_tone_target.addItem("目标=画风参考图（画质优先）", "style")
        self.post_tone_target.addItem("目标=输入照片（色彩保真）", "photo")
        self.post_ink_check = QCheckBox("线条加墨（解决线条稀碎）")
        self.post_ink_check.setChecked(True)
        layout.addWidget(self.post_tone_check)
        layout.addWidget(self.post_tone_target)
        layout.addWidget(self.post_ink_check)

        # 后处理失败重试：读取 pipeline-steps/pipeline-manifest.json，从失败节点继续
        post_retry_row = QHBoxLayout()
        self.post_retry_btn = QPushButton("重试失败步骤（从失败节点继续）")
        self.post_retry_btn.setEnabled(False)
        self.post_retry_btn.setToolTip(
            "后处理按「重绘提线 → 结构线叠加 → 局部重绘」执行，每步产物与状态记录在\n"
            "<产物目录>/pipeline-steps/pipeline-manifest.json；\n"
            "某步失败时点这里会跳过已成功的步骤，只重跑失败的那一步。"
        )
        self.post_retry_btn.clicked.connect(self.retry_failed_post_steps)
        post_retry_row.addWidget(self.post_retry_btn)
        post_retry_row.addStretch(1)
        self.post_retry_row = QWidget()
        self.post_retry_row.setLayout(post_retry_row)
        layout.addWidget(self.post_retry_row)

        self.repaint_toggle_btn = QPushButton("▸ 重绘参数（模型 / 分辨率 / 比例 / 次数）")
        self.repaint_toggle_btn.setCheckable(True)
        self.repaint_toggle_btn.setToolTip(
            "展开/收起重绘参数；不用的平时收起，输入框才有正常高度。\n"
            "重绘模型默认 gemini-3-pro-image-preview（Nano Banana Pro）。"
        )
        self.repaint_toggle_btn.toggled.connect(self.on_repaint_panel_toggled)
        layout.addWidget(self.repaint_toggle_btn)

        self.repaint_panel = QWidget()
        repaint_panel_layout = QVBoxLayout(self.repaint_panel)
        repaint_panel_layout.setContentsMargins(16, 0, 0, 0)   # 缩进表示从属于上面那个勾选框
        repaint_panel_layout.setSpacing(4)

        repaint_form = QFormLayout()
        self.repaint_model_combo = QComboBox()
        self.repaint_model_combo.setEditable(True)
        for name in (self._repaint_config.get("model_options") or REPAINT_DEFAULTS["model_options"]):
            self.repaint_model_combo.addItem(str(name))
        saved_repaint_model = str(self._repaint_config.get("model") or REPAINT_DEFAULTS["model"])
        if self.repaint_model_combo.findText(saved_repaint_model) < 0:
            self.repaint_model_combo.addItem(saved_repaint_model)
        self.repaint_model_combo.setCurrentText(saved_repaint_model)
        self.repaint_model_combo.setToolTip(
            "重绘模型（走 apis.aigc2d 的 /v1beta/models/{model}:generateContent）。\n"
            "gemini-3-pro-image-preview = Nano Banana Pro，重绘质量最好；flash 更快但细节弱一些。"
        )
        repaint_form.addRow("重绘模型:", self.repaint_model_combo)

        self.repaint_resolution_combo = QComboBox()
        for value in (self._repaint_config.get("resolution_options") or REPAINT_DEFAULTS["resolution_options"]):
            self.repaint_resolution_combo.addItem(str(value))
        saved_resolution = str(self._repaint_config.get("resolution") or REPAINT_DEFAULTS["resolution"])
        if self.repaint_resolution_combo.findText(saved_resolution) < 0:
            self.repaint_resolution_combo.addItem(saved_resolution)
        self.repaint_resolution_combo.setCurrentText(saved_resolution)
        repaint_form.addRow("重绘分辨率:", self.repaint_resolution_combo)

        self.repaint_aspect_combo = QComboBox()
        for value in (self._repaint_config.get("aspect_ratio_options") or REPAINT_DEFAULT_ASPECT_OPTIONS):
            self.repaint_aspect_combo.addItem(
                "auto（跟随源图比例）" if is_auto_aspect_ratio(value) else value, value
            )
        saved_aspect = resolve_aspect_ratio(self._repaint_config, ASPECT_RATIO_AUTO)
        _set_combo_by_data(self.repaint_aspect_combo, saved_aspect, ASPECT_RATIO_AUTO)
        self.repaint_aspect_combo.setToolTip(
            "auto（默认）= 不下发 aspectRatio 字段，由模型按输入图比例输出；\n"
            "换任意比例的源图都不会被拉伸或裁切（官方默认行为）。\n"
            "只有确实要强制输出比例时才选具体值。"
        )
        repaint_form.addRow("输出宽高比:", self.repaint_aspect_combo)
        # 选了"强制比例"就该立刻出现提醒，所以要跟着刷新提示
        self.repaint_aspect_combo.currentIndexChanged.connect(lambda _i: self._refresh_repaint_hint())

        self.repaint_repeat_spin = QSpinBox()
        self.repaint_repeat_spin.setRange(1, 4)
        self.repaint_repeat_spin.setValue(max(1, int(self._repaint_config.get("repeat") or 1)))
        self.repaint_repeat_spin.setToolTip("重绘是抽卡：同一张源图多跑几次，挑手/发丝最好的那张。")
        repaint_form.addRow("每张重绘次数:", self.repaint_repeat_spin)
        repaint_panel_layout.addLayout(repaint_form)

        # 只在**异常/警告**时显示的提示（正常情况下界面上的控件自己就能说明一切，不再重复复述状态）
        self.repaint_notice = QLabel("")
        self.repaint_notice.setWordWrap(True)
        self.repaint_notice.setMinimumWidth(420)
        self.repaint_notice.setMaximumHeight(64)
        self.repaint_notice.setStyleSheet("color: #b26a00;")
        self.repaint_notice.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        repaint_panel_layout.addWidget(self.repaint_notice)
        layout.addWidget(self.repaint_panel)
        # 默认收起参数；勾选框（功能开关）始终可见
        self.repaint_panel.setVisible(False)
        self.repaint_toggle_btn.setChecked(False)
        self._refresh_repaint_hint()

        self.api_hint = QLabel("")
        self.api_hint.setWordWrap(True)
        layout.addWidget(self.api_hint)

        layout.addWidget(QLabel("提示词 (prompt):"))
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("生图: 描述画面; 编辑: 说明要改什么、保留什么")
        # 给足最小高度：这一块是布局里唯一可拉伸的编辑区，最小高度太小就会被挤成一条
        self.prompt_edit.setMinimumHeight(140)
        self.prompt_edit.setToolTip("提示词编辑区（可拖动下方边框调整高度）")
        layout.addWidget(self.prompt_edit)

        # 长度护栏 + 成本估算：常驻显示，避免用户写出"太长会爆 / 让画风参考图失效"的提示词
        self.budget_label = QLabel("")
        self.budget_label.setWordWrap(True)
        self.budget_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.budget_label.setToolTip(
            "长度：软上限 2000 字符（超过它画风参考图基本失效）、硬上限 15000 字符"
            "（本中转站实测 15662 字符直接断连），官网上限 32000。\n"
            "成本：按中转站公开价目 + 本项目实测 token 用量估算（new-api 口径），"
            "分组倍率取实测路由分组（gpt-image → Openai-Gpt-1，Gemini 图片 → Discounted-Banana-1）。"
        )
        layout.addWidget(self.budget_label)
        self.prompt_edit.textChanged.connect(self._refresh_budget)

        self.ref_label = QLabel("")
        self.ref_label.setWordWrap(True)
        self.ref_label.setToolTip(
            "顺序就是提交给接口的 image[] 顺序：先拖入/添加，再拖动缩略图排序。\n"
            "单击缩略图预览，双击打开所在目录，点右上角 ×（或右键菜单）删除。"
        )
        layout.addWidget(self.ref_label)

        # 画风选择：gpt-image 通道用 config-styles.json 里的 prompt_gpt 短版 + 画风的参考图
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("画风:"))
        self.style_combo = QComboBox()
        self.style_combo.setMinimumWidth(220)
        self.style_combo.setToolTip(
            "选择画风后：提示词自动带上该画风的「gpt-image 专用短版说明」(prompt_gpt)，\n"
            "并把画风的参考图追加到参考图列表**最后一张**（内容图在前、画风图在后），\n"
            "同时在提示词里写明两者的职责分工——避免把画风参考图的角色/服装/构图搬进来。\n"
            "要改画风说明就编辑 conf/config-styles.json 的 prompt_gpt 字段（或跑 tools/convert_styles_gpt.py）。"
        )
        self.style_combo.currentIndexChanged.connect(self._on_style_changed)
        style_row.addWidget(self.style_combo)
        self.style_info_label = QLabel("")
        self.style_info_label.setWordWrap(True)
        style_row.addWidget(self.style_info_label, stretch=1)
        layout.addLayout(style_row)
        self.reload_styles()

        layout.addWidget(QLabel("画风参考图:"))
        self.image_grid = RefImageGrid(
            max_images=GPT_IMAGE2_MAX_REFERENCE_IMAGES,
            parent=self,
            compact_when_empty=True,   # 空列表时不吃缩略图高度，把空间让给提示词编辑框
        )
        self.image_grid.images_changed.connect(self._on_images_changed)
        self.image_grid.image_clicked.connect(self.show_preview)
        self.image_grid.image_double_clicked.connect(self.open_containing_dir)
        layout.addWidget(self.image_grid)
        self._on_images_changed([])

        img_btn_row = QHBoxLayout()
        self.add_btn = QPushButton("添加图片")
        self.add_btn.clicked.connect(self.add_images)
        img_btn_row.addWidget(self.add_btn)
        self.clear_btn = QPushButton("清空图片")
        self.clear_btn.clicked.connect(self.clear_images)
        img_btn_row.addWidget(self.clear_btn)
        self.refresh_cfg_btn = QPushButton("刷新接口配置")
        self.refresh_cfg_btn.clicked.connect(self.refresh_api_hint)
        img_btn_row.addWidget(self.refresh_cfg_btn)
        self.copy_log_btn = QPushButton("复制日志")
        self.copy_log_btn.clicked.connect(self.copy_log)
        img_btn_row.addWidget(self.copy_log_btn)
        img_btn_row.addStretch(1)
        layout.addLayout(img_btn_row)

        action_row = QHBoxLayout()
        self.generate_btn = QPushButton("开始生成")
        self.generate_btn.clicked.connect(self.generate)
        action_row.addWidget(self.generate_btn)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_generation)
        action_row.addWidget(self.cancel_btn)
        self.status_label = QLabel("就绪")
        action_row.addWidget(self.status_label, stretch=1)
        layout.addLayout(action_row)

        body_row = QHBoxLayout()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("运行日志")
        body_row.addWidget(self.log_view, stretch=1)
        self.preview_label = QLabel("预览")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(220, 220)
        self.preview_label.setStyleSheet("border: 1px solid #888;")
        body_row.addWidget(self.preview_label, stretch=1)
        layout.addLayout(body_row, stretch=1)

        self.result_list = QListWidget()
        self.result_list.setMaximumHeight(90)
        self.result_list.itemDoubleClicked.connect(self.open_item)
        layout.addWidget(self.result_list)

        result_btn_row = QHBoxLayout()
        self.add_results_to_ref_btn = QPushButton("把结果加入参考图")
        self.add_results_to_ref_btn.setToolTip(
            "把结果列表里的图片加成参考图，随后用「重绘」模式再优化一轮（迭代重绘）。"
        )
        self.add_results_to_ref_btn.clicked.connect(self.add_results_to_reference)
        result_btn_row.addWidget(self.add_results_to_ref_btn)
        self.repaint_results_btn = QPushButton("重绘结果列表的图")
        self.repaint_results_btn.setToolTip("直接对结果列表里的图跑一轮 Gemini 重绘（不需要先加参考图）。")
        self.repaint_results_btn.clicked.connect(self.repaint_selected_results)
        result_btn_row.addWidget(self.repaint_results_btn)
        self.clear_results_btn = QPushButton("清空结果")
        self.clear_results_btn.clicked.connect(self.result_list.clear)
        result_btn_row.addWidget(self.clear_results_btn)
        result_btn_row.addStretch(1)
        layout.addLayout(result_btn_row)

        self._populate_sizes(SITE_AIGC2D)
        self.on_mode_changed(self.mode_combo.currentText())

    def current_site(self) -> str:
        return str(self.site_combo.currentData() or SITE_AIGC2D)

    def on_site_changed(self, site: str):
        if self._loading:
            return
        self._apply_site_defaults(site)
        self.refresh_api_hint()
        self._append_log(f"已切换到站点: {site}（API类型 {API_TYPE_BY_SITE.get(site, '')}）")

    def _looks_like_preset(self, text: str, preset: str) -> bool:
        """粗判提示词框里是不是固件原文（忽略空白差异）。只用于决定切模式时要不要保留。"""
        a = " ".join(str(text or "").split())
        b = " ".join(str(preset or "").split())
        return bool(b) and bool(a) and a == b

    def _repaint_preset_text(self) -> str:
        """当前固件（主 prompt + 细节后缀）的完整正文，用于填进重绘模式的提示词框。"""
        try:
            return build_repaint_prompt(self.current_repaint_config())
        except Exception as exc:  # noqa: BLE001 - 装载失败不该让界面起不来
            self._append_log(f"[重绘] 固件装载失败: {type(exc).__name__}: {exc}")
            return ""

    def on_mode_changed(self, mode: str):
        mode_text = str(mode)
        if mode_text == MODE_REPAINT:
            # 进入重绘模式：**总是把固件正文填进提示词框**。
            # 这样用户能直接看到实际在用的重绘提示词（含修手/保蕾丝等约束），
            # 也就明白"只填自己的描述"为什么体现不出修复效果。
            # 生图/编辑那边原本的文字先存起来，切回去时还原。
            previous = self.prompt_edit.toPlainText()
            if previous and not self._looks_like_preset(previous, self._repaint_last_preset):
                self._mode_prompt_backup = previous
            self._mode_prompt_active = "repaint"
            preset = self._repaint_preset_text()
            if preset:
                self._repaint_last_preset = preset
                self.prompt_edit.setPlainText(preset)
            self.prompt_edit.setPlaceholderText("已填入重绘固件（可直接修改）；清空则回退为使用固件原文")
        else:
            if self._mode_prompt_active == "repaint":
                # 离开重绘模式：把生图/编辑那边原来的文字还给用户，避免固件串到生图框
                self.prompt_edit.setPlainText(self._mode_prompt_backup or "")
                self._mode_prompt_active = ""
            if mode_text == MODE_EDIT:
                self.prompt_edit.setPlaceholderText(
                    "编辑: 说明要改什么、保留什么（例: 把背景换成日落海滩，保留人物姿势与服装）"
                )
            else:
                self.prompt_edit.setPlaceholderText("生图: 描述画面（挂了参考图就作为垫图参考）")
        self._refresh_repaint_hint()
        self._on_images_changed(self.image_paths)

    # ---------------- 产物优化（重绘） ----------------
    def on_repaint_panel_toggled(self, expanded: bool):
        """展开/收起**重绘参数**（模型/分辨率/比例/次数）；开关本身常驻可见，不在折叠区里。"""
        if hasattr(self, "repaint_panel"):
            self.repaint_panel.setVisible(bool(expanded))
        arrow = "▾" if expanded else "▸"
        self.repaint_toggle_btn.setText(f"{arrow} 重绘参数（模型 / 分辨率 / 比例 / 次数）")

    def repaint_enabled(self) -> bool:
        """本次是否会走重绘。

        - 重绘模式：恒定走重绘（勾选框无意义，界面上置灰）
        - 生图 / 编辑模式：由「出图后立即重绘优化」勾选框决定。
          编辑模式同样支持——gpt-image 的 edits 出的图分辨率同样偏低，重绘一步到位。
        - 重绘模式下再叠一层重绘没有意义，所以那种情况不看勾选框。
        """
        mode_text = self.mode_combo.currentText()
        if mode_text == MODE_REPAINT:
            return True
        if mode_text in (MODE_GENERATE, MODE_EDIT):
            return bool(self.repaint_check.isChecked())
        return False

    def current_repaint_config(self) -> dict:
        conf = dict(self._repaint_config)
        conf["model"] = self.repaint_model_combo.currentText().strip() or REPAINT_DEFAULTS["model"]
        conf["resolution"] = self.repaint_resolution_combo.currentText().strip() or REPAINT_DEFAULTS["resolution"]
        conf["aspect_ratio"] = str(
            self.repaint_aspect_combo.currentData() or self.repaint_aspect_combo.currentText() or ASPECT_RATIO_AUTO
        ).strip() or ASPECT_RATIO_AUTO
        conf["repeat"] = int(self.repaint_repeat_spin.value())
        return conf

    def _refresh_repaint_hint(self):
        """只显示**异常/需要提醒**的信息。

        界面上的控件（勾选框、模型/分辨率/比例/次数下拉）自己就说明了当前设置，
        再复述一遍「当前：…」纯属噪音，已删除；这里只留三类提醒：
        1) 提示词固件装载失败；2) 重绘 API key 未配置；3) 输出比例被强制成非 auto。
        """
        conf = self._repaint_config
        notices = []
        try:
            build_repaint_prompt(conf)
        except Exception as exc:  # noqa: BLE001 - 提示词文件缺失不该让界面起不来
            notices.append(f"⚠ 重绘提示词装载失败：{type(exc).__name__}: {exc}")
        api_type = str(conf.get("api_type") or "aigc2d")
        if not str(get_api_config(api_type=api_type).get("api_key") or "").strip():
            notices.append(f"⚠ 重绘不可用：{self._key_hint(api_type)}")
        if hasattr(self, "repaint_aspect_combo"):
            chosen = self.current_repaint_config().get("aspect_ratio")
            if not is_auto_aspect_ratio(chosen):
                sources = list(self.image_paths)
                labels = [image_aspect_ratio_label(p) for p in sources[:3]]
                labels = [x for x in labels if x]
                source_state = f"（源图实际 {'、'.join(labels)}）" if labels else ""
                notices.append(f"⚠ 输出比例被强制为 {chosen}{source_state}，与源图不一致时可能被拉伸")
        if hasattr(self, "repaint_notice"):
            self.repaint_notice.setText("；".join(notices))
            self.repaint_notice.setVisible(bool(notices))

        mode_text = self.mode_combo.currentText() if hasattr(self, "mode_combo") else MODE_GENERATE
        if hasattr(self, "repaint_check"):
            # 生图 / 编辑模式都由这个勾选框决定是否链式重绘；重绘模式恒定走重绘，勾选框无意义 → 置灰仍可见
            self.repaint_check.setEnabled(mode_text != MODE_REPAINT)
        # 参数在三种模式下都可调（编辑模式也会走重绘；重绘模式更不用说）
        for widget in (self.repaint_model_combo, self.repaint_resolution_combo,
                       self.repaint_aspect_combo, self.repaint_repeat_spin):
            widget.setEnabled(True)

    def on_repaint_toggled(self, _checked: bool):
        self._refresh_repaint_hint()
        self.save_repaint_defaults()

    def save_repaint_defaults(self):
        """把重绘选项写回 prompts/gpt-image-optimize/config.json（用户改了就不用每次重设）。"""
        conf = dict(self._repaint_config)
        conf["enabled"] = bool(self.repaint_check.isChecked())
        conf["model"] = self.repaint_model_combo.currentText().strip() or REPAINT_DEFAULTS["model"]
        conf["resolution"] = self.repaint_resolution_combo.currentText().strip() or REPAINT_DEFAULTS["resolution"]
        conf["aspect_ratio"] = str(
            self.repaint_aspect_combo.currentData() or self.repaint_aspect_combo.currentText() or ASPECT_RATIO_AUTO
        ).strip() or ASPECT_RATIO_AUTO
        conf["repeat"] = int(self.repaint_repeat_spin.value())
        try:
            save_repaint_config(conf)
            self._repaint_config = load_repaint_config()
        except Exception as exc:  # noqa: BLE001 - 存默认值失败不该阻断生图
            self._append_log(f"保存重绘默认值失败(忽略): {exc}")

    def build_repaint_request(self, sources) -> dict:
        """组装 `generate_image_repaint` 的参数（供 Worker 调用，也便于测试）。"""
        conf = self.current_repaint_config()
        override = self.prompt_edit.toPlainText().strip()
        params = {
            "api_type": str(conf.get("api_type") or REPAINT_DEFAULTS["api_type"]),
            "model": str(conf.get("model") or REPAINT_DEFAULTS["model"]),
            "resolution": str(conf.get("resolution") or REPAINT_DEFAULTS["resolution"]),
            # auto 时后端不下发 aspectRatio，由模型按输入图比例输出
            "aspect_ratio": resolve_aspect_ratio(conf, ASPECT_RATIO_AUTO),
            "repeat": int(conf.get("repeat") or 1),
            "save_sub_dir": str(conf.get("save_sub_dir") or REPAINT_DEFAULTS["save_sub_dir"]),
            "return_metadata": False,
        }
        if override:
            params["prompt"] = override
        return params

    def add_results_to_reference(self):
        paths = [self.result_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.result_list.count())]
        paths = [p for p in paths if p]
        if not paths:
            QMessageBox.information(self, "提示", "结果列表为空。")
            return
        self._add_paths(paths)
        self.mode_combo.setCurrentText(MODE_REPAINT)
        self._append_log(f"[重绘] 已把 {len(paths)} 张结果加入参考图并切到重绘模式，点「开始生成」即可迭代重绘。")

    def repaint_selected_results(self):
        """不用先加参考图：直接对结果列表里的图跑一轮重绘。"""
        paths = [self.result_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.result_list.count())]
        paths = [p for p in paths if p]
        if not paths:
            QMessageBox.information(self, "提示", "结果列表为空（先生成，或先添加图片到参考图）。")
            return
        conf = self.current_repaint_config()
        if not str(get_api_config(api_type=str(conf.get("api_type") or REPAINT_DEFAULTS["api_type"])).get("api_key") or "").strip():
            QMessageBox.warning(self, "提示", f"apis.{conf.get('api_type')} 缺少 api_key，无法重绘。")
            return
        self.save_repaint_defaults()
        params = self.build_repaint_request(paths)
        params["source_paths"] = paths
        self.log_view.clear()
        self._append_log(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} 重绘结果列表 {len(paths)} 张 =====")
        self._append_log(f"[重绘] 模型 {params['model']} @{params['resolution']}，每张 {params['repeat']} 次")
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("重绘中...")
        self.preview_label.setText("重绘中...")
        self._worker = GptImage2Worker(generate_image_repaint, params, self)
        self._worker.log.connect(self._append_log)
        self._worker.done.connect(self.on_done)
        self._worker.error.connect(self.on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _populate_sizes(self, site: str):
        self.size_combo.blockSignals(True)
        self.size_combo.clear()
        if site == SITE_AIGC2D:
            self.size_combo.addItem("自动（跟随参考图比例）", SIZE_FOLLOW_INPUT)
        for label, value in SIZE_CHOICES_BY_SITE.get(site, SIZE_CHOICES_BY_SITE[SITE_AIGC2D]):
            self.size_combo.addItem(label, value)
        self.size_combo.blockSignals(False)

    def _populate_models(self, site: str, data: dict = None, live_models=None):
        """按站点填模型下拉：站点常用清单 + 该站点已保存的模型 + 刷新拿到的实时清单。"""
        api_type = API_TYPE_BY_SITE.get(site, "")
        api_cfg = ((data or {}).get("apis") or {}).get(api_type) or {}
        saved = str(api_cfg.get("model") or "").strip()
        if live_models is None:
            live_models = self._live_models.get(site)
        self._set_model_choices(
            pick_gpt_image_models(site=site, models=live_models, extra=[saved]),
            saved or GPT_IMAGE2_MODEL_DEFAULT,
        )

    def _set_model_choices(self, models, current: str):
        """重建模型下拉（模型名原样作为条目文本，说明放悬浮提示）。"""
        self.model_combo.blockSignals(True)
        try:
            self.model_combo.clear()
            for name in models or []:
                self.model_combo.addItem(name)
                note = GPT_IMAGE2_MODEL_NOTES.get(name)
                if note:
                    self.model_combo.setItemData(
                        self.model_combo.count() - 1, f"{name}：{note}", Qt.ItemDataRole.ToolTipRole
                    )
            if current:
                self.model_combo.setCurrentText(current)
        finally:
            self.model_combo.blockSignals(False)

    def current_model(self) -> str:
        return self.model_combo.currentText().strip() or GPT_IMAGE2_MODEL_DEFAULT

    def refresh_models(self):
        """按当前站点 GET /v1/models，把 gpt-image 系列填进下拉（后台线程，不卡界面）。"""
        site = self.current_site()
        api_type = API_TYPE_BY_SITE.get(site, "")
        cfg = get_api_config(api_type=api_type)
        if not str(cfg.get("api_key") or "").strip():
            QMessageBox.warning(
                self, "提示", self._key_hint(api_type) + "，无法获取模型列表。"
            )
            return
        if self._model_worker is not None and self._model_worker.isRunning():
            return
        endpoint = resolve_models_endpoint(str(cfg.get("base_url") or ""), api_type=api_type)
        self._append_log(f"[模型] 获取 {site} 模型列表: GET {endpoint}")
        self.refresh_models_btn.setEnabled(False)
        self.refresh_models_btn.setText("获取中...")
        self._model_worker = ModelListWorker(api_type=api_type, config_path=CONFIG_IMAGE_FILE, parent=self)
        self._model_worker.done.connect(self._on_models_fetched)
        self._model_worker.error.connect(self._on_models_fetch_error)
        self._model_worker.finished.connect(self._on_model_worker_finished)
        self._model_worker.start()

    def _on_models_fetched(self, models):
        site = self.current_site()
        self._live_models[site] = list(models or [])
        current = self.current_model()
        choices = pick_gpt_image_models(site=site, models=self._live_models[site], extra=[current])
        self._set_model_choices(choices, current)
        family = [name for name in choices if is_gpt_image_model(name)]
        self._append_log(
            f"[模型] {site} 返回 {len(models or [])} 个模型，其中 gpt-image/dall-e 系列 {len(family)} 个已入下拉: "
            f"{', '.join(family)}"
        )
        self.status_label.setText(f"模型列表已更新（{len(family)} 个）")

    def _on_models_fetch_error(self, message):
        self._append_log(f"[模型] 获取模型列表失败: {message}")
        QMessageBox.warning(
            self, "获取失败", f"获取模型列表失败：{message}\n请检查该站点的 base_url / api_key / 网络。"
        )

    def _on_model_worker_finished(self):
        self.refresh_models_btn.setEnabled(True)
        self.refresh_models_btn.setText("刷新模型列表")
        worker = self._model_worker
        self._model_worker = None
        if worker is not None:
            worker.deleteLater()

    def _key_hint(self, api_type: str) -> str:
        """缺 key 时的提示文案：说明两种来源（配置文件 / 环境变量），不暴露任何密钥值。

        节点配置要连着取（`_env_key_names` 需要 `env_slug`）——`aigc-2d-gpt` 的密钥变量
        可能与 `aigc2d` 共用（`IMAGE_MAKER_AIGC2D_API_KEY`），只按节点名提示会指向一个
        根本不存在的变量名。
        """
        from modules.others.api_backend import _env_key_names

        try:
            cfg = get_api_config(api_type=api_type)
        except Exception:  # noqa: BLE001 - 提示文案不该因为读配置失败而崩
            cfg = {}
        names = _env_key_names(api_type, cfg)
        env_hint = f"，或设环境变量 {names[0]}" if names else ""
        return f"apis.{api_type} 缺少 api_key（可在 conf/config.json 填写{env_hint}）"

    # ---------------- 配置 ----------------
    def refresh_api_hint(self):
        site = self.current_site()
        api_type = API_TYPE_BY_SITE.get(site, "")
        cfg = get_api_config(api_type=api_type)
        base_url = str(cfg.get("base_url") or "(未配置)")
        key_state = {
            "config": "已配置(conf/config.json)",
            "none": "未配置(请到设置→图片生成 API 填写，或设环境变量)",
        }.get(str(cfg.get("_api_key_source") or ""), "")
        if not key_state:
            key_state = f"已配置({cfg.get('_api_key_source')})"   # env:VARNAME，只暴露变量名
        model = str(cfg.get("model") or "gpt-image-2")
        cancel_state = "支持取消" if site == SITE_AIGC2D else "不支持中断(请求返回后才结束)"
        self.api_hint.setText(
            f"站点: {site}  (API类型 {api_type})    接口: {base_url}    模型: {model}    "
            f"API Key: {key_state}    取消: {cancel_state}"
        )

    def _site_node(self, data: dict, site: str) -> dict:
        node = data.get(CONFIG_NODE) if isinstance(data.get(CONFIG_NODE), dict) else {}
        sites = node.get("sites") if isinstance(node.get("sites"), dict) else {}
        per_site = sites.get(site) if isinstance(sites.get(site), dict) else {}
        per_site = dict(per_site)
        if site == SITE_AIGC2D:
            # 兼容早期扁平结构 {"size": ..., "quality": ...}
            for key in ("model", "size", "quality", "output_format", "n"):
                if key not in per_site and key in node:
                    per_site[key] = node[key]
        return per_site

    def _apply_site_defaults(self, site: str):
        data = _read_config_image()
        self._populate_sizes(site)
        self._populate_models(site, data)
        per_site = self._site_node(data, site)
        fallback_size = DEFAULT_SIZE_BY_SITE.get(site, GPT_IMAGE2_SIZE_SQUARE)
        # 没有历史选择时，aigc2d 默认「自动（跟随参考图比例）」——固定竖图会让宽图输入输出成竖图
        if site == SITE_AIGC2D and not per_site.get("size"):
            fallback_size = SIZE_FOLLOW_INPUT
        saved_size = per_site.get("size")
        if saved_size and site == SITE_AIGC2D:
            if str(saved_size) != SIZE_FOLLOW_INPUT:
                saved_size = normalize_gpt_image2_size(size=saved_size)
        _set_combo_by_data(self.size_combo, saved_size, fallback_size)
        _set_combo_by_data(self.quality_combo, per_site.get("quality"), "high")
        output_format = str(per_site.get("output_format") or "png").lower()
        if self.output_format_combo.findText(output_format) >= 0:
            self.output_format_combo.setCurrentText(output_format)
        try:
            saved_n = max(1, min(10, int(per_site.get("n") or 1)))
        except (TypeError, ValueError):
            saved_n = 1
        self.n_spin.setValue(saved_n)
        # autodl 通道一次一张，没有 n 参数
        self.n_spin.setEnabled(site != SITE_AUTODL)

    def load_defaults(self):
        data = _read_config_image()
        node = data.get(CONFIG_NODE) if isinstance(data.get(CONFIG_NODE), dict) else {}
        self._loading = True
        try:
            site = str(node.get("site") or SITE_AIGC2D)
            if self.site_combo.findData(site) < 0:
                site = SITE_AIGC2D
            self.site_combo.setCurrentIndex(self.site_combo.findData(site))
            self._apply_site_defaults(site)
            mode = str(node.get("mode") or MODE_GENERATE)
            if mode in MODE_CHOICES:
                self.mode_combo.setCurrentText(mode)
            if hasattr(self, "post_dual_check") and "post_dual" in node:
                self.post_dual_check.setChecked(bool(node.get("post_dual")))
            saved_style = node.get("style")
            if saved_style and hasattr(self, "style_combo"):
                idx = self.style_combo.findText(str(saved_style))
                if idx >= 0:
                    self.style_combo.setCurrentIndex(idx)
            repaint_node = node.get("repaint") if isinstance(node.get("repaint"), dict) else {}
            if hasattr(self, "repaint_check"):
                saved_enabled = repaint_node.get("enabled")
                if saved_enabled is None:
                    saved_enabled = bool(load_repaint_config().get("enabled"))
                self.repaint_check.setChecked(bool(saved_enabled))
                if repaint_node.get("model"):
                    self.repaint_model_combo.setCurrentText(str(repaint_node["model"]))
                if repaint_node.get("resolution"):
                    self.repaint_resolution_combo.setCurrentText(str(repaint_node["resolution"]))
                if repaint_node.get("aspect_ratio") is not None and hasattr(self, "repaint_aspect_combo"):
                    _set_combo_by_data(self.repaint_aspect_combo, repaint_node.get("aspect_ratio"), ASPECT_RATIO_AUTO)
                try:
                    self.repaint_repeat_spin.setValue(max(1, int(repaint_node.get("repeat") or 1)))
                except (TypeError, ValueError):
                    self.repaint_repeat_spin.setValue(1)
            post_node = node.get("post_process") if isinstance(node.get("post_process"), dict) else {}
            if post_node and hasattr(self, "structure_check"):
                self.structure_check.setChecked(bool(post_node.get("structure_enabled")))
                try:
                    self.structure_strength_spin.setValue(float(post_node.get("structure_strength") or 0.5))
                except (TypeError, ValueError):
                    pass
                self.local_repaint_check.setChecked(bool(post_node.get("local_enabled")))
                region = str(post_node.get("local_region") or "hair")
                idx = self.local_region_combo.findData(region)
                if idx >= 0:
                    self.local_region_combo.setCurrentIndex(idx)
                try:
                    self.local_feather_spin.setValue(max(8, int(post_node.get("local_feather") or 48)))
                except (TypeError, ValueError):
                    pass
                if post_node.get("panel_open"):
                    self.post_toggle_btn.setChecked(True)
                    self.post_panel.setVisible(True)
        finally:
            self._loading = False

    def save_defaults(self):
        data = _read_config_image()
        node = data.get(CONFIG_NODE) if isinstance(data.get(CONFIG_NODE), dict) else {}
        site = self.current_site()
        sites = node.get("sites") if isinstance(node.get("sites"), dict) else {}
        sites = dict(sites)
        sites[site] = {
            "model": self.current_model(),
            "size": self.size_combo.currentData() or DEFAULT_SIZE_BY_SITE.get(site, GPT_IMAGE2_SIZE_SQUARE),
            "quality": self.quality_combo.currentData() or "high",
            "output_format": self.output_format_combo.currentText().strip() or "png",
            "n": int(self.n_spin.value()),
        }
        node["sites"] = sites
        node["site"] = site
        node["mode"] = self.mode_combo.currentText()
        if hasattr(self, "style_combo"):
            node["style"] = self.style_combo.currentText()
        if hasattr(self, "post_dual_check"):
            node["post_dual"] = bool(self.post_dual_check.isChecked())
        if hasattr(self, "repaint_check"):
            node["repaint"] = {
                "enabled": bool(self.repaint_check.isChecked()),
                "model": self.repaint_model_combo.currentText().strip(),
                "resolution": self.repaint_resolution_combo.currentText().strip(),
                "aspect_ratio": self.current_repaint_config().get("aspect_ratio") or ASPECT_RATIO_AUTO,
                "repeat": int(self.repaint_repeat_spin.value()),
            }
        if hasattr(self, "structure_check"):
            node["post_process"] = {
                "structure_enabled": bool(self.structure_check.isChecked()),
                "structure_strength": float(self.structure_strength_spin.value()),
                "local_enabled": bool(self.local_repaint_check.isChecked()),
                "local_region": str(self.local_region_combo.currentData() or "hair"),
                "local_feather": int(self.local_feather_spin.value()),
                "panel_open": bool(self.post_toggle_btn.isChecked()),
            }
        data[CONFIG_NODE] = node
        try:
            with open(CONFIG_IMAGE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as exc:  # noqa: BLE001 - 记默认值失败不该阻断生图
            self._append_log(f"保存本 Tab 默认值失败(忽略): {exc}")

    # ---------------- 图片 ----------------
    def _on_images_changed(self, paths):
        count = len(paths or [])
        mode_hint = "编辑=至少 1 张" if self.mode_combo.currentText() == MODE_EDIT else "生图=可留空(作为垫图)"
        self.ref_label.setText(
            f"参考图 {count}/{GPT_IMAGE2_MAX_REFERENCE_IMAGES}"
            f"（{mode_hint}；拖入图片、拖动缩略图排序、点 × 删除，顺序即提交顺序）:"
        )
        # 缩略图出现/消失会改变网格高度，让布局把富余空间重新分给提示词编辑框
        if hasattr(self, "image_grid"):
            self.image_grid.updateGeometry()
        self._refresh_repaint_hint()

    def add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择参考图", "", IMAGE_FILTER)
        self._add_paths(paths or [])

    def _add_paths(self, paths):
        """追加参考图，返回新增数量（超上限的会提示并忽略）。"""
        added, ignored = self.image_grid.add_paths(paths)
        if added:
            self._append_log(f"已添加 {added} 张参考图，当前 {self.image_grid.count()} 张")
        if ignored:
            QMessageBox.information(
                self, "提示", f"最多 {GPT_IMAGE2_MAX_REFERENCE_IMAGES} 张参考图，超出的 {ignored} 张已忽略。"
            )
        return added

    def clear_images(self):
        self.image_grid.clear()

    def dragEnterEvent(self, event):  # noqa: N802 - Qt 接口
        if collect_image_paths(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):  # noqa: N802 - Qt 接口
        paths = collect_image_paths(event.mimeData())
        if not paths:
            event.ignore()
            return
        self._add_paths(paths)
        event.acceptProposedAction()

    # ---------------- 生成 ----------------
    def resolve_request_size(self):
        """把「自动」解析成实际尺寸：有参考图就按参考图比例，否则用站点默认。"""
        raw = self.size_combo.currentData() or DEFAULT_SIZE_BY_SITE.get(self.current_site(), GPT_IMAGE2_SIZE_SQUARE)
        if str(raw) != SIZE_FOLLOW_INPUT:
            return str(raw)
        from modules.others.api_backend import pick_gpt_image2_size_for_images
        grid = getattr(self, "image_grid", None) or getattr(self, "ref_grid", None)
        images = []
        if grid is not None:
            for method in ("paths", "get_paths", "get_images"):
                if hasattr(grid, method):
                    value = getattr(grid, method)
                    images = [p for p in (value() if callable(value) else value) if p]
                    break
        picked = pick_gpt_image2_size_for_images(
            images, fallback=DEFAULT_SIZE_BY_SITE.get(self.current_site(), GPT_IMAGE2_SIZE_PORTRAIT))
        self.size_info_label.setText(f"自动尺寸 → {picked}（按参考图比例）" if images else f"自动尺寸 → {picked}（无参考图）")
        return picked

    def build_request(self):
        """按当前站点组装 (后端函数, 调用参数)，供 Worker 调用（也便于测试）。"""
        site = self.current_site()
        raw_prompt = self.prompt_edit.toPlainText().strip()
        model = self.current_model()
        size = self.resolve_request_size() if hasattr(self, "resolve_request_size") else (
            self.size_combo.currentData() or DEFAULT_SIZE_BY_SITE.get(site, GPT_IMAGE2_SIZE_SQUARE))
        quality = self.quality_combo.currentData() or "high"
        output_format = self.output_format_combo.currentText().strip() or "png"
        mode_text = self.mode_combo.currentText()
        mode = "edit" if mode_text == MODE_EDIT else "generate"

        # 画风：gpt-image 通道用 prompt_gpt 短版；画风参考图追加到内容图**之后**（最后一张）
        from utils.styles import compose_style_prompt, ordered_reference_images
        style_text, style_ref = self.current_style_block()
        content_images = list(self.image_paths)
        prompt = compose_style_prompt(style_text, raw_prompt,
                                      style_ref_attached=bool(style_ref),
                                      content_image_count=len(content_images))
        image_paths = ordered_reference_images(content_images, style_ref, style_ref_attached=bool(style_ref))
        if style_text:
            self._append_log(f"[画风] {self.current_style_name()}：说明 {len(style_text)} 字符"
                             + (f" + 参考图 {os.path.basename(style_ref)}（提交顺序最后一张）" if style_ref
                                else "（该画风无参考图）"))

        if site == SITE_AUTODL:
            params = {
                "prompt": prompt,
                "image_paths": image_paths,
                "model": model,
                "aspect_ratio": "1:1",       # 尺寸由 size 直控
                "instructions": "",
                "api_type": API_TYPE_BY_SITE[site],
                "save_sub_dir": SAVE_SUB_DIR_BY_SITE[site],
                "file_prefix": FILE_PREFIX_BY_SITE[site],
                "return_metadata": False,
                "size": size,
                "quality": quality,
                "output_format": output_format,
            }
            return generate_image_openai_image, params

        params = {
            "prompt": prompt,
            "image_paths": image_paths,
            "model": model,
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "n": int(self.n_spin.value()),
            "mode": mode,
            "api_type": API_TYPE_BY_SITE[site],
            "save_sub_dir": SAVE_SUB_DIR_BY_SITE[site],
            "file_prefix": FILE_PREFIX_BY_SITE[site],
            "return_metadata": False,
        }
        return generate_image_aigc2d_gpt, params

    def build_repaint_backend_request(self):
        """重绘模式下的 (后端函数, 参数)：直接对当前参考图（= 产物）走 Gemini 重绘。"""
        params = self.build_repaint_request(list(self.image_paths))
        params["source_paths"] = list(self.image_paths)
        return generate_image_repaint, params

    def generate(self):
        site = self.current_site()
        mode = self.mode_combo.currentText()
        api_type = API_TYPE_BY_SITE.get(site, "")

        # ---- 分支 1：重绘模式（输入=已拖入的产物）----
        if mode == MODE_REPAINT:
            if not self.image_paths:
                QMessageBox.warning(
                    self, "提示",
                    "重绘模式需要至少 1 张图片：把 gpt-image 产物拖进「参考图」区，或先点「添加图片」。\n"
                    "（也可以先正常生图，再点「重绘结果列表的图」）",
                )
                return
            repaint_conf = self.current_repaint_config()
            repaint_api_type = str(repaint_conf.get("api_type") or REPAINT_DEFAULTS["api_type"])
            if not str(get_api_config(api_type=repaint_api_type).get("api_key") or "").strip():
                QMessageBox.warning(
                    self, "提示",
                    self._key_hint(repaint_api_type) + "，无法重绘。",
                )
                return
            self.save_repaint_defaults()
            backend, params = self.build_repaint_backend_request()
            self.log_view.clear()
            self._append_log(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} 开始重绘（源图 {len(self.image_paths)} 张）=====")
            self._append_log(
                f"[重绘] 模型 {params['model']} @{params['resolution']} ，每张 {params['repeat']} 次；"
                f"提示词来源: {'界面输入(覆盖)' if params.get('prompt') else str(repaint_conf.get('system_prompt'))}"
            )
            self._append_log("[状态] 已发出重绘请求，等待服务器返回（2K 单张通常 30~90 秒）...")
            self.generate_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.status_label.setText("重绘中...(2K 单张约 30~90s)")
            self.preview_label.setText("重绘中...")
            self._worker = GptImage2Worker(backend, params, self)
            self._worker.log.connect(self._append_log)
            self._worker.done.connect(self.on_done)
            self._worker.error.connect(self.on_error)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker.start()
            return

        # ---- 分支 2/3：生图（可勾选接着重绘） / gpt-image 编辑 ----
        if not self.prompt_edit.toPlainText().strip():
            QMessageBox.warning(self, "提示", "请输入提示词")
            return
        if mode == MODE_EDIT and not self.image_paths:
            QMessageBox.warning(self, "提示", "编辑模式至少需要 1 张图片")
            return
        if not str(get_api_config(api_type=api_type).get("api_key") or "").strip():
            QMessageBox.warning(
                self,
                "提示",
                self._key_hint(api_type) + "，请先在「设置 → 图片生成 API」里填写。",
            )
            return
        # 生图与编辑都支持出图后接着重绘（gpt-image 两种模式出的分辨率都偏低）
        chain_repaint = bool(self.repaint_check.isChecked()) and mode in (MODE_GENERATE, MODE_EDIT)
        if chain_repaint:
            repaint_api_type = str(self.current_repaint_config().get("api_type") or REPAINT_DEFAULTS["api_type"])
            if not str(get_api_config(api_type=repaint_api_type).get("api_key") or "").strip():
                QMessageBox.warning(
                    self, "提示",
                    "已勾选「出图后立即重绘优化」，但 " + self._key_hint(repaint_api_type) + "。",
                )
                return
            self.save_repaint_defaults()

        self.save_defaults()
        backend, params = self.build_request()
        repaint_params = None
        if chain_repaint:
            repaint_params = self.build_repaint_request(list(self.image_paths))
            repaint_params.pop("source_paths", None)

        self.log_view.clear()
        self._append_log(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} 开始生成 =====")
        for line in format_request_dump(params, get_api_config(api_type=api_type), site, api_type, mode if mode != MODE_REPAINT else "edit"):
            self._append_log(line)
        self._append_log(f"[后端] 将要调用: {getattr(backend, '__name__', backend)}(...)")
        if chain_repaint:
            self._append_log(
                f"[链路] 生图 → 重绘（{repaint_params.get('model')} @{repaint_params.get('resolution')}，"
                f"每张 {repaint_params.get('repeat')} 次），最终产物为重绘结果。"
            )
        if site == SITE_AUTODL:
            self._append_log("[提示] autodl 通道一次只出一张，且无法中途取消（请求返回后才结束）。")
        post_steps = self.post_pipeline_steps()
        active_post = [name for name, cfg in post_steps.items() if isinstance(cfg, dict) and cfg.get("enabled")]
        if active_post:
            labels = {"structure": "结构线叠加", "local": "局部重绘+羽化贴回"}
            self._append_log("[链路] 出图" + (" → 重绘" if chain_repaint else "")
                             + " → " + " → ".join(labels.get(n, n) for n in active_post)
                             + "，最终产物为最后一道工序的输出。")
        self._append_log("[状态] 已发出请求，等待服务器返回（high/max 画质单张可能需要 1~5 分钟）...")

        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(site == SITE_AIGC2D)
        self.status_label.setText("生成中...(high 画质可能要几分钟)")
        self.preview_label.setText("请求中...")

        self._worker = GptImage2Worker(backend, params, self, repaint_params=repaint_params)
        self._worker.log.connect(self._append_log)
        self._worker.done.connect(self.on_done)
        self._worker.raw_done.connect(self._on_raw_products)
        self._worker.error.connect(self.on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_raw_products(self, paths):
        """链式模式下先记录未重绘的原始产物（只写日志，不进结果列表）。"""
        for path in paths or []:
            self._append_log(f"[原始产物] {path}")
        self._append_log(
            f"[原始产物] 共 {len(paths or [])} 张，重绘完成后结果列表里只会出现重绘产物；"
            "想对比可点「打开所在目录」。"
        )

    def cancel_generation(self):
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self.status_label.setText("已请求取消，等待当前请求返回...")

    def _on_worker_finished(self):
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()

    def on_done(self, paths):
        paths = [p for p in (paths or []) if p]
        if not paths:
            self.status_label.setText("完成(无图片) —— 详见下方日志")
            self._append_log("[结果] 服务器没有返回可用图片。排查顺序：")
            self._append_log("       1) 上方『请求 URL / 请求数据』是否发到了预期端点与字段")
            self._append_log("       2) 上方『服务器原始返回信息』里的 error/message（审核拦截 moderation_blocked、余额不足、参数不支持等）")
            self._append_log("       3) 上方『服务器返回 JSON 已保存到: …』给出的文件，含完整原文")
            self._append_log("       4) 本机日志文件 log/<日期>.log，内容相同且含重试过程")
            return
        steps = self.post_pipeline_steps()
        if any(cfg.get("enabled") for cfg in steps.values()):
            self._start_post_process(paths)
            return
        self._show_results(paths)

    def _refresh_post_retry(self, paths):
        """根据 pipeline-steps 里的清单决定「重试失败步骤」是否可用。"""
        if not hasattr(self, "post_retry_btn"):
            return
        from utils.post_process import pipeline_failures
        work_dir = os.path.join(os.path.dirname(os.path.abspath(paths[0])), "pipeline-steps") if paths else ""
        failures = pipeline_failures(work_dir) if work_dir else []
        self.post_retry_btn.setEnabled(bool(failures))
        if failures:
            self.status_label.setText(f"部分完成：{len(failures)} 个后处理步骤失败（可点「重试失败步骤」）")

    def _show_results(self, paths):
        paths = [p for p in (paths or []) if p]
        if not paths:
            self.status_label.setText("完成(后处理无产物) —— 详见下方日志")
            return
        self.status_label.setText(f"完成, {len(paths)} 张")
        for path in paths:
            item = QListWidgetItem(f"{os.path.basename(path)}  ->  {path}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.result_list.addItem(item)
            self._append_log(f"[结果] 已保存: {path}")
        self.show_preview(paths[0])

    def reload_styles(self):
        """从 conf/config-styles.json 加载画风列表（第一项 = 默认(无附加)）。"""
        if not hasattr(self, "style_combo"):
            return
        current = self.style_combo.currentText()
        self._styles_data = {}
        try:
            import json as _json
            with open(CONFIG_STYLES_FILE, encoding="utf-8") as f:
                self._styles_data = _json.load(f) or {}
        except Exception as exc:  # noqa: BLE001 - 画风表读不到不该影响生图
            self._append_log(f"[画风] 读取 {CONFIG_STYLES_FILE} 失败（忽略）: {exc}")
        self.style_combo.blockSignals(True)
        self.style_combo.clear()
        self.style_combo.addItem(STYLE_NONE)
        for name in self._styles_data:
            self.style_combo.addItem(str(name))
        idx = self.style_combo.findText(current)
        self.style_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.style_combo.blockSignals(False)
        self._refresh_style_info()

    def current_style_name(self) -> str:
        if not hasattr(self, "style_combo"):
            return ""
        text = self.style_combo.currentText().strip()
        return "" if text == STYLE_NONE else text

    def current_style_block(self) -> tuple:
        """返回 (画风短版文本, 画风参考图路径)。gpt-image 通道优先用 prompt_gpt。"""
        from utils.styles import ref_image_valid, style_prompt, style_prompt_gpt, style_ref_image
        name = self.current_style_name()
        if not name:
            return "", ""
        gpt_text = style_prompt_gpt(self._styles_data, name).strip()
        text = gpt_text or style_prompt(self._styles_data, name).strip()
        ref = style_ref_image(self._styles_data, name)
        return text, (ref if ref_image_valid(ref) else "")

    def _on_style_changed(self, _index=None):
        self._refresh_style_info()
        self._refresh_budget()

    def _refresh_style_info(self):
        if not hasattr(self, "style_info_label"):
            return
        name = self.current_style_name()
        if not name:
            self.style_info_label.setText("未附加画风说明（只用提示词框里的文字）")
            return
        text, ref = self.current_style_block()
        src = "prompt_gpt（gpt-image 专用短版）" if str(
            __import__("utils.styles", fromlist=["x"]).style_prompt_gpt(self._styles_data, name)).strip() \
            else "prompt（全量说明，未转换 prompt_gpt）"
        parts = [f"{len(text)} 字符 / {src}"]
        parts.append("画风参考图: " + (os.path.basename(ref) if ref else "无（该画风没配参考图）"))
        if not self._styles_data.get(name, {}).get("prompt_gpt"):
            parts.append("⚠ 缺 prompt_gpt，建议跑 tools/convert_styles_gpt.py --only " + name)
        self.style_info_label.setText(" ｜ ".join(parts))
        self.style_info_label.setStyleSheet("color: #b26a00;" if not ref else "")

    def _refresh_budget(self):
        """刷新提示词长度护栏 + 单张全工序成本估算（常驻显示）。"""
        if not hasattr(self, "budget_label"):
            return
        try:
            from utils.cost_estimate import (format_budget, format_pipeline_cost, estimate_pipeline,
                                             prompt_budget)
        except Exception as exc:  # noqa: BLE001 - 估算不可用不该影响生图
            self.budget_label.setText(f"（长度/成本估算不可用: {exc}）")
            return
        text = self.prompt_edit.toPlainText() if hasattr(self, "prompt_edit") else ""
        style_text = self.current_style_block()[0] if hasattr(self, "style_combo") else ""
        budget = prompt_budget(len(text), style_chars=len(style_text))
        steps = self.post_pipeline_steps() if hasattr(self, "structure_check") else {}
        chain_repaint = bool(self.repaint_check.isChecked()) if hasattr(self, "repaint_check") else False
        try:
            est = estimate_pipeline(
                max(1, len(text) + len(style_text)),
                size=str(self.resolve_request_size() if hasattr(self, "resolve_request_size") else
                         self.size_combo.currentData() or DEFAULT_SIZE_BY_SITE.get(self.current_site(),
                                                                                  GPT_IMAGE2_SIZE_SQUARE)),
                quality=str(self.quality_combo.currentData() or "medium"),
                ref_images=len(self.image_paths or []),
                include_repaint=chain_repaint,
                include_structure=bool(steps.get("structure", {}).get("enabled")),
                include_local=bool(steps.get("local", {}).get("enabled")),
            )
        except Exception as exc:  # noqa: BLE001
            est = None
            self._append_log(f"[估算] 成本估算失败（忽略）: {exc}")
        lines = [format_budget(budget)]
        if est:
            detail = " + ".join(f"{s['label'].split(' ')[0]} ${s['usd']:.4f}" for s in est["steps"])
            lines.append(f"单张全工序估算 ≈ ${est['total_usd']:.3f}（≈¥{est['total_cny']:.2f}） ｜ {detail}")
        self.budget_label.setText("\n".join(lines))
        if budget["over_hard"]:
            color = "#c62828"
        elif budget["over_soft"]:
            color = "#b26a00"
        else:
            color = "#3a6b35"
        self.budget_label.setStyleSheet(f"color: {color};")

    def _start_pricing_refresh(self):
        """后台拉一次价目表（首次显示用内置快照，拿到网络价目后再刷新一次）。"""
        if getattr(self, "_pricing_worker", None) is not None:
            return
        self._pricing_worker = PricingWorker(self)
        self._pricing_worker.done.connect(self._on_pricing_ready)
        self._pricing_worker.start()

    def _on_pricing_ready(self, ok):
        worker = getattr(self, "_pricing_worker", None)
        self._pricing_worker = None
        if worker is not None:
            worker.deleteLater()
        if ok:
            self._refresh_budget()

    def post_pipeline_steps(self):
        """按界面勾选框组装后处理流水线（未勾选的工序不会执行）。"""
        from utils.post_process import default_pipeline
        steps = default_pipeline()
        steps["structure"]["enabled"] = bool(self.structure_check.isChecked())
        steps["structure"]["strength"] = float(self.structure_strength_spin.value())
        steps["local"]["enabled"] = bool(self.local_repaint_check.isChecked())
        steps["local"]["region"] = str(self.local_region_combo.currentData() or "hair")
        steps["local"]["feather"] = int(self.local_feather_spin.value())
        steps["local"]["resolution"] = str(self.repaint_resolution_combo.currentText() or "2K")
        return steps

    def _last_post_outputs(self):
        """最近一次生成/重绘的产物路径（重试用）。"""
        paths = []
        for i in range(self.result_list.count()):
            item = self.result_list.item(i)
            p = item.data(Qt.ItemDataRole.UserRole)
            if p and os.path.isfile(p):
                paths.append(p)
        return paths

    def retry_failed_post_steps(self):
        """从失败节点继续跑后处理（已成功的步骤会被跳过）。"""
        from utils.post_process import pipeline_failures
        sources = self._last_post_outputs()
        if not sources:
            QMessageBox.information(self, "提示", "结果列表里没有可用产物，先跑一次生成。")
            return
        work_dir = os.path.join(os.path.dirname(os.path.abspath(sources[0])), "pipeline-steps")
        failures = pipeline_failures(work_dir)
        if not failures:
            QMessageBox.information(self, "提示", "没有发现失败的后处理步骤。")
            return
        names = "、".join(sorted({str(f.get("step")) for f in failures}))
        self._append_log(f"[后处理] 重试失败步骤: {names}")
        self._start_post_process(sources)

    def _on_post_progress(self, message):
        """把后处理进度同步到状态栏（避免"生成返回就算完成"的错觉）。"""
        text = str(message or "")
        if "[工序]" in text:
            self.status_label.setText("后处理中… " + text.replace("[工序]", "").strip()[:60])

    def _start_post_process(self, paths):
        """在后台线程跑勾选的后处理工序（局部重绘要走网络，不能卡 UI）。"""
        from utils.post_process import run_pipeline
        steps = self.post_pipeline_steps()
        self.status_label.setText("后处理中...(局部重绘可能要 1 分钟)")
        firmware = None
        if steps["local"]["enabled"]:
            firmware = str(self.current_repaint_config().get("system_prompt") or "")
        self._post_worker = PostProcessWorker(paths, steps, firmware, self)
        self._post_worker.log.connect(self._append_log)
        self._post_worker.log.connect(self._on_post_progress)
        self._post_worker.done.connect(self._on_post_done)
        self._post_worker.start()

    def _on_post_done(self, paths):
        worker = getattr(self, "_post_worker", None)
        self._post_worker = None
        if worker is not None:
            worker.deleteLater()
        self._show_results(paths)

    def on_error(self, message):
        self.status_label.setText("出错")
        self._append_log(f"[结果] 调用失败: {message}")
        self._append_log("       排查: 确认 conf/config.json 中该站点的 base_url / api_key / model；"
                         "网络异常与 429/5xx 会自动重试，4xx（参数/审核）直接把服务器原文落盘")
        QMessageBox.critical(self, "生成错误", str(message)[:600])

    def show_preview(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.preview_label.setText("无法预览")
            return
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def open_containing_dir(self, path):
        if path and os.path.exists(path):
            os.startfile(os.path.dirname(path))  # noqa: S606 - 打开所在目录

    def open_item(self, item):
        self.open_containing_dir(item.data(Qt.ItemDataRole.UserRole))

    def _append_log(self, message):
        self.log_view.appendPlainText(str(message))
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def copy_log(self):
        QApplication.clipboard().setText(self.log_view.toPlainText())
        self.status_label.setText("日志已复制到剪贴板")

    # ---------------- 生命周期 ----------------
    def showEvent(self, event):  # noqa: N802 - Qt 接口
        super().showEvent(event)
        self.refresh_api_hint()
