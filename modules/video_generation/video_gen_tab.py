# -*- coding: utf-8 -*-
"""视频生成 Tab: MiniMax-H3 (autodl 站点).

提供: 模型/生成模式(文生视频/首帧/首尾帧/多模态参考图·视频·音频)/分辨率/时长/比例/
callback_url/aigc_watermark/任务提交/轮询进度/自动下载, 以及"提示词与图片引用"帮助面板.
配置持久化到 conf/config-video.json.
"""
import os
import json
import threading
from datetime import datetime

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
                             QPlainTextEdit, QCheckBox, QFileDialog, QListWidget,
                             QListWidgetItem, QMessageBox, QProgressBar, QSplitter,
                             QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from modules.video_generation import ai_video_backend as vbk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONF_PATH = os.path.join(PROJECT_ROOT, "conf", "config-video.json")

# ------------------------------------------------------------------ 示例提示词
EXAMPLE_PROMPTS = {
    "text": ("""无附件·文生视频（必须选具体比例，如 16:9；不可 adaptive）
英文示例：
Cinematic wide shot, a girl in a blue dress sits in a small white paper boat floating on a
pastel-purple starry sea. The Milky Way reflects on gentle ripples. Soft moonlight, slow camera
push-in, hair and dress drift in the breeze. [Shot 1] At 00:03.000, the camera tilts to reveal a
crescent moon. [Shot 2] At 00:07.000, a close-up of her eyes; a faint dark silhouette appears.

中文示例：
广角电影镜头，蓝裙少女坐在纸船里漂在浅紫色星海，银河倒映、波纹轻漾，柔和月光，镜头缓慢推进，
衣袂与发丝随风轻扬。00:03 镜头抬起露出弯月；00:07 特写她的眼睛，远处黑暗里隐隐浮现一个黑色身影。"""),
    "first": ("""首帧模式 I2VA（1 张图 = 0 秒端点锚点）
对齐表述 + 动作路径（首帧锚点→动作开始→连续发展→结果/反应）：
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is
fully referenced.
[Shot 1] Keep the girl's identity, blue dress and bow consistent with <Picture 1>. A medium
tracking shot follows her standing in the paper boat; the moonlit star sea reflects as the boat
drifts; wind moves her hair. 00:01.5 she raises her head to gaze into the distance, camera slowly
pushes in."""),
    "first_last": ("""首尾帧模式 FL2VA（图1=首帧，图2=尾帧；模型补全中间连续变化）对齐表述：
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the
0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the 8.00-second mark of
the target video.
动作路径（首帧状态→中间变化→逐步收敛→尾帧状态）：
[Shot 1] Starting from <Picture 1>: the girl in the blue dress sinks slowly, hair and skirt drift
upward, bubbles rise. She realizes she can breathe and opens her eyes. Gradually the scene converges
to <Picture 2> — a calm hover with soft rays."""),
    "reference": ("""多模态参考 Ref2VA（图/视频/音频做 人物·产品·场景·风格 参考；非精确端点；图片≤9张）
素材职责绑定 + 人物锁定：
<Subject 1> is the girl whose appearance and blue dress come from <Picture 1>; preserve her facial
identity, long blue hair, white blouse and black bead necklace.
<Picture 2> only provides the pastel-purple starry sea palette and lighting.
<Video 1> provides the gentle boat-rocking movement and pace.
<Audio 1> provides the soft ambient underwater tone.
[Shot 1] A slow push-in follows <Subject 1> drifting underwater; identities stay locked, priority:
<Picture 1> for character, <Picture 2> for mood."""),
}


# ------------------------------------------------------------------ 帮助文本
HELP_TEXT = """【MiniMax H3 提示词与图片引用指南】

■ 模式与素材职责（一次请求只选一种模式；首/尾帧与多模态参考互斥）
· 文生视频：仅文字；比例必填（16:9 等），不能用 adaptive。
· 图生视频-首帧 I2VA：1 图，图片是第 0 秒的像素级端点锚点。
· 图生视频-首尾帧 FL2VA：2 图，模型补全两点间的连续变化。
· 图生视频-尾帧 L2VA：1 图作尾帧，视频应逐步收敛到该图。
· 多模态参考 Ref2VA：图/视频/音频做参考（人物/产品/场景/风格/动作），最多 9 张参考图；
  音频不能作为唯一参考，必须同时有至少一张图片或一段视频。

■ 如何引用图片附件（前端 @图片1 会被编译为 <Picture 1> 锚点）
· 首帧/尾帧是"端点锚点"，照抄对齐表述即可：
  I2VA:  For the target video, at 0.00 seconds into the target video, <Picture 1>
         (from [Shot 1]) is fully referenced.
  FL2VA: How the reference pictures align with the target video — Picture 1 (from Shot 1)
         aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N)
         aligns with the S.SS-second mark of the target video.
· 参考图在正文里用 <Picture N> 指代，并声明职责与优先级：
  "<Subject 1> is the girl whose appearance and blue dress come from <Picture 1>;
   preserve her facial identity, hair, dress and accessories."
  "图1严格负责人物身份与服装；图2只负责场景色调与光线；冲突部分以图1为准。"
· 人物锁定（不变量只写一次）：身份、五官、发型发色、体型、服装版型、配饰数量；
  允许变化的通常只有表情、视线、姿势、呼吸与自然的头发/衣料运动。

■ 提示词六层控制（先定不变量→事件→摄影，再补质感）
1) 目的  这条视频为什么存在（展示/剧情钩子/视觉包装）
2) 不变量  什么绝不能漂移（人物身份/产品结构/品牌字样/UI布局）
3) 事件  画面实际发生什么（动作顺序→状态变化→结尾落点）
4) 摄影  观众如何看见（景别/机位/构图/运镜/对焦/切镜）
5) 质感  它属于什么世界（光线/色彩/材质/媒介/后期纹理）
6) 声音  与音视频参考绑定：对白/音乐/环境声/动作音效
只写"高级、电影感、炸裂"不能形成可执行镜头。

■ 镜头术语（用机械正确的词）
Zoom=变焦（机位不动）  Push=推机（机位前移）  Pan=摇（机头扫动）  Truck=横移
Tilt=俯仰  Pedestal=升降。摄影词要绑定主体行为，如"人物转身时镜头保持眼平并横向跟移"。

■ 动作路径
I2VA：首帧锚点→动作开始→连续发展→结果/反应   （保持身份/服装/色彩/空间关系）
FL2VA：首帧状态→可观察的中间变化→逐步缩小差异→尾帧状态（官方偏好单镜头）
L2VA：合理先前状态→明确动作与过渡→最终镜头逐步收敛→落到尾帧

■ 转场：写触发物与切换时刻，别只写"丝滑转场"。
例："遮挡最满时切到下一场景""甩动转场""闪光转场""声音重拍处切换"。

■ 常见失败：素材不分工、风格词过多、动作过载、约束堆砌、重复锁定、
文字失控（用白名单并限制出现次数）、将编辑写成重拍、伪完整 Prompt（"可自行补充"等占位语）。

■ 参数与模型约束（对照官方文档）
· MiniMax-H3：分辨率 768P/2K；时长 4~15s；支持文生/图生(首/尾/首尾)/多模态参考。
· MiniMax-H3-Max：仅 480P/768P（不支持 2K）；时长 5~15s；仅文生与首/尾帧，不支持多模态参考。
· 输入媒体：图片 JPG/PNG/WEBP/HEIC(≤30MB, [256,5760]px, 宽高比[0.4,2.5])；
  参考视频 MP4/MOV(≤50MB, 2~15s)；音频 WAV/MP3(≤15MB)。请求体总大小 ≤64MB（本工具用 base64，
  大文件建议压缩后使用）。
· 附图（可选）：callback_url（任务回调，需 3 秒内回传 challenge 验证）、
  aigc_watermark（生成视频加 AIGC 标识水印，默认 false）。
"""


class VideoWorker(QThread):
    progress = pyqtSignal(str, str)                # kind, msg
    done = pyqtSignal(str, object)                 # out_path 或 None, last_status
    error = pyqtSignal(str)

    def __init__(self, args, config, parent=None):
        super().__init__(parent)
        self.args = args
        self.config = config
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            out_path = self.args.get("out_path")
            task_id, _, _ = vbk.create_task(**self.args["create"], config=self.config)
            self.progress.emit("created", task_id)
            out, status = vbk.poll_and_download(
                task_id, out_path, config=self.config,
                progress_cb=lambda s, task, sec: self.progress.emit(
                    "poll", f"status={s} elapsed={sec}s "
                            f"{json.dumps(task.get('error') or '', ensure_ascii=False)[:120]}"),
                cancel_check=self._cancel.is_set)
            self.done.emit(out, status)
        except Exception as e:
            self.error.emit(repr(e)[:400])


class VideoGenWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._last_task_id = ""
        self.initUI()
        self.load_config()

    # ---------------------------------------------------------------- UI
    def initUI(self):
        layout = QVBoxLayout(self)

        # API 配置
        cfg_box = QFormLayout()
        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("https://www.autodl.art/api/v1/minimax")
        cfg_box.addRow("Base URL:", self.base_url_edit)
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        cfg_box.addRow("API Key:", self.key_edit)
        self.save_cfg_btn = QPushButton("保存视频API配置")
        self.save_cfg_btn.clicked.connect(self.save_config)
        cfg_box.addRow("", self.save_cfg_btn)
        layout.addLayout(cfg_box)

        self.sub_tabs = QTabWidget()
        self.sub_tabs.addTab(self._build_gen_panel(), "生成参数")
        self.sub_tabs.addTab(self._build_help_panel(), "提示词与图片引用帮助")
        layout.addWidget(self.sub_tabs, stretch=1)

    def _build_gen_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        opt_grid = QGridLayout()
        opt_grid.addWidget(QLabel("模型:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(vbk.MODELS)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        opt_grid.addWidget(self.model_combo, 0, 1)
        opt_grid.addWidget(QLabel("生成模式:"), 0, 2)
        self.mode_combo = QComboBox()
        for key, label in vbk.MODES:
            self.mode_combo.addItem(label, key)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        opt_grid.addWidget(self.mode_combo, 0, 3)
        opt_grid.addWidget(QLabel("分辨率:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(vbk.RESOLUTIONS)
        opt_grid.addWidget(self.resolution_combo, 1, 1)
        opt_grid.addWidget(QLabel("时长(4-15s):"), 1, 2)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(4, 15)
        self.duration_spin.setValue(8)
        opt_grid.addWidget(self.duration_spin, 1, 3)
        opt_grid.addWidget(QLabel("比例:"), 2, 0)
        self.ratio_combo = QComboBox()
        self.ratio_combo.setEditable(True)
        self.ratio_combo.addItems(vbk.RATIOS)
        opt_grid.addWidget(self.ratio_combo, 2, 1)
        self.aigc_watermark_check = QCheckBox("AIGC水印(aigc_watermark)")
        opt_grid.addWidget(self.aigc_watermark_check, 2, 2)
        self.callback_url_edit = QLineEdit()
        self.callback_url_edit.setPlaceholderText("可选(callback_url, 回调需回传challenge)")
        opt_grid.addWidget(self.callback_url_edit, 2, 3)
        layout.addLayout(opt_grid)

        self.hint_label = QLabel("")
        self.hint_label.setStyleSheet("color:#b07a2a;")
        layout.addWidget(self.hint_label)

        # 输入区
        in_grid = QGridLayout()
        in_grid.addWidget(QLabel("首帧/参考图:"), 0, 0)
        self.first_frame_edit = QLineEdit()
        self.first_frame_edit.setPlaceholderText("图生视频首帧 / 参考图(可多张,用分号分隔)")
        self.first_frame_btn = QPushButton("浏览...")
        self.first_frame_btn.clicked.connect(lambda: self._browse_into(self.first_frame_edit))
        in_grid.addWidget(self.first_frame_edit, 0, 1)
        in_grid.addWidget(self.first_frame_btn, 0, 2)
        in_grid.addWidget(QLabel("尾帧图:"), 1, 0)
        self.last_frame_edit = QLineEdit()
        self.last_frame_edit.setPlaceholderText("仅首尾帧模式")
        self.last_frame_btn = QPushButton("浏览...")
        self.last_frame_btn.clicked.connect(lambda: self._browse_into(self.last_frame_edit))
        in_grid.addWidget(self.last_frame_edit, 1, 1)
        in_grid.addWidget(self.last_frame_btn, 1, 2)
        in_grid.addWidget(QLabel("参考视频:"), 2, 0)
        self.ref_video_edit = QLineEdit()
        self.ref_video_edit.setPlaceholderText("多模态参考模式: mp4/mov, 2~15s")
        self.ref_video_btn = QPushButton("浏览...")
        self.ref_video_btn.clicked.connect(lambda: self._browse_into(self.ref_video_edit, "视频 (*.mp4 *.mov)"))
        in_grid.addWidget(self.ref_video_edit, 2, 1)
        in_grid.addWidget(self.ref_video_btn, 2, 2)
        in_grid.addWidget(QLabel("参考音频:"), 3, 0)
        self.audio_edit = QLineEdit()
        self.audio_edit.setPlaceholderText("多模态参考模式: mp3/wav")
        self.audio_btn = QPushButton("浏览...")
        self.audio_btn.clicked.connect(lambda: self._browse_into(self.audio_edit, "音频 (*.mp3 *.wav)"))
        in_grid.addWidget(self.audio_edit, 3, 1)
        in_grid.addWidget(self.audio_btn, 3, 2)
        layout.addLayout(in_grid)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText(
            "提示词(prompt, 必填)。描述画面/运镜/氛围；想引用图/视频/音频时用 <Picture N>/<Video N>/<Audio N>。\n"
            "可切到右侧『提示词帮助』看写法与一键示例。")
        self.prompt_edit.setMinimumHeight(90)
        layout.addWidget(self.prompt_edit)

        row = QHBoxLayout()
        self.submit_btn = QPushButton("提交生成视频")
        self.submit_btn.clicked.connect(self._submit)
        row.addWidget(self.submit_btn)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        row.addWidget(self.cancel_btn)
        self.query_btn = QPushButton("查询状态")
        self.query_btn.clicked.connect(self._query)
        row.addWidget(self.query_btn)
        self.status_label = QLabel("就绪")
        row.addWidget(self.status_label, stretch=1)
        layout.addLayout(row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.task_id_label = QLabel("task_id: (未提交)")
        layout.addWidget(self.task_id_label)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.result_list = QListWidget()
        self.result_list.itemDoubleClicked.connect(self._open_item_path)
        split.addWidget(self.result_list)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("任务日志")
        split.addWidget(self.log_edit)
        split.setSizes([500, 500])
        layout.addWidget(split, stretch=1)

        self._on_mode_changed(self.mode_combo.currentIndex())
        self._on_model_changed(self.model_combo.currentText())
        return panel

    def _build_help_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        row = QHBoxLayout()
        row.addWidget(QLabel("示例一键填入（会覆盖右侧提示词框，之后可按需改）:"))
        for label, key in [("文生视频", "text"), ("首帧", "first"), ("首尾帧", "first_last"), ("多模态参考", "reference")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, k=key: self.prompt_edit.setPlainText(EXAMPLE_PROMPTS[k]))
            row.addWidget(btn)
        layout.addLayout(row)
        help_edit = QPlainTextEdit()
        help_edit.setReadOnly(True)
        help_edit.setPlainText(HELP_TEXT)
        layout.addWidget(help_edit, stretch=1)
        return panel

    # ---------------------------------------------------------------- 交互
    def _browse_into(self, edit, pattern="图片 (*.png *.jpg *.jpeg *.webp)"):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", pattern)
        if path:
            edit.setText(path)

    def _on_mode_changed(self, index):
        mode = self.mode_combo.currentData()
        need_first = mode in ("first", "first_last", "reference")
        self.first_frame_edit.setEnabled(need_first)
        self.first_frame_btn.setEnabled(need_first)
        self.last_frame_edit.setEnabled(mode == "first_last")
        self.last_frame_btn.setEnabled(mode == "first_last")
        self.ref_video_edit.setEnabled(mode == "reference")
        self.ref_video_btn.setEnabled(mode == "reference")
        self.audio_edit.setEnabled(mode == "reference")
        self.audio_btn.setEnabled(mode == "reference")
        self._update_hint()

    def _on_model_changed(self, text):
        self._update_hint()

    def _update_hint(self):
        model = self.model_combo.currentText().strip()
        mode = self.mode_combo.currentData()
        notes = []
        if "Max" in model or model == "MiniMax-H3-Max":
            notes.append("H3-Max: 仅 480P/768P, 时长 5~15s, 不支持多模态参考")
        if mode == "text":
            notes.append("文生视频: 比例必须选具体值(如16:9), 不能用 adaptive")
        self.hint_label.setText(" ; ".join(notes))

    def _log(self, text):
        self.log_edit.appendPlainText(text)

    def _open_item_path(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and os.path.exists(path):
            os.startfile(os.path.dirname(path))  # noqa: S606

    def _collect_create_args(self):
        mode = self.mode_combo.currentData()
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            raise ValueError("prompt 不能为空")
        ref_imgs = [p.strip() for p in self.first_frame_edit.text().split(";") if p.strip()]
        args = {
            "mode": mode,
            "prompt": prompt,
            "model": self.model_combo.currentText().strip() or "MiniMax-H3",
            "resolution": self.resolution_combo.currentText().strip() or "768P",
            "duration": int(self.duration_spin.value()),
            "ratio": self.ratio_combo.currentText().strip() or "adaptive",
            "callback_url": self.callback_url_edit.text().strip() or None,
            "aigc_watermark": self.aigc_watermark_check.isChecked(),
        }
        if mode == "first":
            if not ref_imgs:
                raise ValueError("首帧模式需要选择 首帧图")
            args["first_frame"] = ref_imgs[0]
        elif mode == "first_last":
            if not ref_imgs or not self.last_frame_edit.text().strip():
                raise ValueError("首尾帧模式需要 首帧图 + 尾帧图")
            args["first_frame"] = ref_imgs[0]
            args["last_frame"] = self.last_frame_edit.text().strip()
        elif mode == "reference":
            if not ref_imgs:
                raise ValueError("多模态参考模式需要至少一张参考图")
            args["reference_images"] = ref_imgs
            if self.ref_video_edit.text().strip():
                args["reference_videos"] = [self.ref_video_edit.text().strip()]
            if self.audio_edit.text().strip():
                args["reference_audio"] = self.audio_edit.text().strip()
        return args

    def _submit(self):
        try:
            create_args = self._collect_create_args()
        except ValueError as e:
            QMessageBox.warning(self, "参数错误", str(e))
            return
        if create_args["mode"] == "text" and create_args["ratio"].lower() == "adaptive":
            QMessageBox.warning(self, "参数错误", "文生视频必须指定具体比例(如 16:9)，不能用 adaptive")
            return
        cfg = self._current_config()
        out_dir = self.out_dir_edit.text().strip() or cfg.get("output_dir", "data/video_generation")
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(PROJECT_ROOT, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path = os.path.join(out_dir, name)
        self._worker = VideoWorker({"create": create_args, "out_path": out_path}, cfg, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self.submit_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.show()
        self._worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
            self._log("请求取消...")

    def _on_progress(self, kind, msg):
        if kind == "created":
            self._last_task_id = msg
            self.task_id_label.setText(f"task_id: {msg}")
        self._log(msg)
        self.status_label.setText(msg)

    def _on_done(self, out_path, last_status):
        self.submit_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.hide()
        if out_path and os.path.exists(out_path):
            self.status_label.setText("完成: " + os.path.basename(out_path))
            item = QListWidgetItem(os.path.basename(out_path))
            item.setData(Qt.ItemDataRole.UserRole, out_path)
            self.result_list.addItem(item)
            self._log(f"已保存: {out_path}")
        else:
            self.status_label.setText(f"未完成: {last_status}")
            self._log(f"任务结束, 状态: {last_status}")

    def _on_error(self, msg):
        self.submit_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.hide()
        self._log("ERROR: " + msg)
        self.status_label.setText("出错: " + msg[:80])

    def _query(self):
        if not self._last_task_id:
            self._log("还没有 task_id")
            return
        cfg = self._current_config()
        try:
            task = vbk.query_task(self._last_task_id, cfg)
        except Exception as e:
            self._log("查询失败: " + repr(e)[:200])
            return
        self._log(json.dumps(task, ensure_ascii=False)[:500])
        self.status_label.setText(f"status={task.get('status')}")

    # ---------------------------------------------------------------- 配置
    def _current_config(self):
        return {
            "base_url": self.base_url_edit.text().strip(),
            "api_key": self.key_edit.text().strip(),
            "model": self.model_combo.currentText().strip() or "MiniMax-H3",
            "resolution": self.resolution_combo.currentText().strip() or "768P",
            "duration": int(self.duration_spin.value()),
            "ratio": self.ratio_combo.currentText().strip() or "adaptive",
            "output_dir": self.out_dir_edit.text().strip() or "data/video_generation",
        }

    def load_config(self):
        if os.path.exists(CONF_PATH):
            try:
                with open(CONF_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self.base_url_edit.setText(cfg.get("base_url", ""))
                self.key_edit.setText(cfg.get("api_key", ""))
                if cfg.get("model") and self.model_combo.findText(cfg["model"]) >= 0:
                    self.model_combo.setCurrentText(cfg["model"])
                if cfg.get("resolution") and self.resolution_combo.findText(cfg["resolution"]) >= 0:
                    self.resolution_combo.setCurrentText(cfg["resolution"])
                self.duration_spin.setValue(int(cfg.get("duration", 8)))
                if cfg.get("ratio"):
                    self.ratio_combo.setCurrentText(cfg["ratio"])
                self.out_dir_edit.setText(cfg.get("output_dir", "data/video_generation"))
            except Exception as e:
                print(f"加载 {CONF_PATH} 失败: {e}")

    def save_config(self):
        cfg = self._current_config()
        try:
            with open(CONF_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "已保存", f"视频配置已保存到 {CONF_PATH}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", repr(e))
