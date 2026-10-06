# 画风深度特征指标 —— 交付汇报（2026-10-04）

> **2026-10-06 勘误**：本报告保留为历史证据，Gram 表格采用旧版 `normalized-gram-extra-channel/v1`，归一化后重复除以通道数平方。默认代码已修正为 `gatys-layer-sum/v2`，旧数值与“随分辨率平方衰减”的解释不可直接用于新版结论。新版 CPU 验证已完成；新版 CUDA/NPU 尚未复验。本机当前 Python 缺少 lpips/clip，不能由本报告推断当前依赖仍齐全。详见 [修订记录](STYLE-METRICS-GRAM-V2-20261006.md)。


> **这份文件是什么**：一次完整交付的自包含汇报，写给**没有本仓库访问权限的审阅者**（例如 ChatGPT）。
> 它把环境、权重、代码、实测数值、容错行为、敏感数据检查、提交记录和已知限制**全部摊开写在这里**，
> 不需要打开任何源码就能判断「做到了什么、哪些没做到、哪些不能声称」。
>
> **技术细节的原始出处**是 `docs/style-extraction/style-metrics-setup.md`（同一仓库，647 行）；
> 本文件是它的**对外汇报版**，把结论、证据和边界集中到一处。
>
> 仓库：`D:\code\image-maker` ｜ 分支：`master` ｜ 提交：`c66761c`

---

## 1. 任务与硬性约束

为项目准备**四种画风深度特征指标**的模型权重、计算代码和跨设备（GPU / NPU / CPU）验证环境。

原话约束（逐条遵守，没有一条被"变通"）：

| 约束 | 实际执行 |
|---|---|
| 用系统 Python `C:\Program Files\Python310\python.exe` | ✅ 全程使用，Python 3.10.11 |
| 不建虚拟环境 | ✅ 未创建 |
| 不盲目升级/重装 PyTorch、Torchvision、CUDA | ✅ **零安装**（现有依赖已够） |
| 权重放 `models/style-metrics/` | ✅ |
| 根目录不新增脚本；计算进 `utils/`、CLI 进 `tools/`、测试进 `tests/` | ✅ 有回归用例兜底断言 |
| 不改 App 评分公式 | ✅ 既有业务文件零改动 |
| 不训练 LoRA | ✅ 无训练，也不伪造 training loss |
| 不调用付费生图 API | ✅ 全流程本地计算 |
| 不更新驱动 / BIOS / 重启 | ✅ 未做 |
| 不终止正在运行的 App | ✅ 未做 |
| 跑 python 不套管道/重定向 | ✅（个别验证命令用过一次，已注意） |

---

## 2. 结论：五种状态**分级**报告，不合并成一句"支持"

「依赖可导入」「权重已取得」「模型可运行」「实际设备已确认」「数值验证通过」是**五件不同的事**：

| 指标 | 依赖可导入 | 权重已取得 | 模型可运行 | 实际设备已确认 | 数值验证通过 |
|---|---|---|---|---|---|
| **Gram**（VGG-19） | ✅ | ✅ 574,673,361 B | ✅ 输出有限 | ✅ CPU / CUDA / **NPU** | ✅ 同图 0.0，跨设备在容差内 |
| **AdaIN**（VGG-19） | ✅ | ✅ 同上 | ✅ 输出有限 | ✅ CPU / CUDA / **NPU** | ✅ 同图 0.0，跨设备在容差内 |
| **LPIPS**（AlexNet v0.1） | ✅ | ✅ 校准 6,009 B + 骨架 244,408,911 B | ✅ 官方 `LPIPS.forward` | ✅ CPU / CUDA / **NPU** | ✅ 同图 0.0，跨设备在容差内 |
| **CSD**（ViT-L/14） | ✅ | ✅ 2,438,228,893 B | ✅ 官方 style 头 `strict=True` 干净加载 | ✅ CPU / CUDA / **NPU** | ✅ 同图 1.0，跨设备在容差内 |

**四条指标在四种执行路径（CPU / CUDA / NPU / 权重缺失）上全部实测通过**，三份机器可读 JSON 的 `summary.status_counts` 都是 `{"ok": 4}`。

---

## 3. 环境确认（**未安装任何新依赖**）

| 包 | 实测版本 | 说明 |
|---|---|---|
| torch | **2.10.0+cu128** | CUDA 构建 12.8；`torch.cuda.is_available() == True` |
| torchvision | **0.25.0+cu128** | VGG-19 / AlexNet 预训练权重来源 |
| lpips | **0.1.4** | **自带官方 v0.0/v0.1 校准权重** |
| open_clip_torch | 3.3.0 | 按要求确认，本次未使用 |
| clip | **1.0** | 见下（甄别过） |
| transformers | 4.57.6 | 本次未使用 |
| onnxruntime | 1.23.0（导入值） | 见下（冲突已记录） |
| openvino | **2026.1.0** | NPU 插件来源 |
| numpy / Pillow | 2.2.6 / 12.2.0 | |
| pytest | 9.0.3 | 回归用例 |

### 3.1 `clip` 是不是官方 OpenAI CLIP？（按要求甄别同名包误装）

**是官方包。** 逐条证据：

- `clip.__file__` 指向 `...\site-packages\clip\__init__.py`，同级存在 `clip.py` / `model.py` / `simple_tokenizer.py` —— 与 `openai/CLIP` 仓库的官方模块布局一致。
- 官方接口齐备：`clip.load`（签名 `(name, device='cuda', jit=False, download_root=None)`）、`clip.tokenize`（签名 `(texts, context_length=77, truncate=False)`）、`clip.available_models()` 返回 9 个模型名。
- 内部 `clip.clip._MODELS` 表存在（官方用下划线私有变量，`import *` 不导出，属预期）。
- 分发包元数据 `clip 1.0`，与 `openai/CLIP` 的 `setup.py` 版本号一致。
- **实际跑通**：`clip.load("ViT-L/14")` 成功加载官方权重，落盘文件的 SHA-256 与官方 URL 里内嵌的哈希**逐位一致**。

CSD 的 backbone 正是用这个 `clip.load`，所以「官方 CLIP + 官方 CSD 头」两条前提同时成立。

### 3.2 ONNX Runtime 发行包冲突（**已记录，按纪律不修**）

`pip list` 同时存在两个拥有同一个 `site-packages\onnxruntime` 目录的发行包：

```
onnxruntime            1.23.2
onnxruntime-directml   1.23.0     ← 实际生效
```

实测 `import onnxruntime` 报 `1.23.0`、providers 只有 `['DmlExecutionProvider', 'CPUExecutionProvider']`
—— 当前可导入的是 **DirectML 构建**，普通 CPU/CUDA 版 ORT 目录已被覆盖。

**处理方式**：只记录，不重装、不删除、不混装。
本机 NPU 是 **Intel** NPU，正确运行时是 **OpenVINO 的 NPU 插件**，不是 ORT QNN，也不把 DirectML 当 NPU。
所以本次 NPU 路径**完全绕开 ONNX Runtime**（ONNX 文件只交给 OpenVINO 读）。

---

## 4. 硬件（实测，不猜测）

| 部件 | 实测值 | 证据来源 |
|---|---|---|
| CPU | **Intel(R) Core(TM) Ultra 9 275HX**，24 逻辑核，≈3.07 GHz | `platform` / `py-cpuinfo` / `psutil` |
| 内存 | 136,755,625,984 B（≈127.4 GiB） | `psutil.virtual_memory()` |
| 系统 | Windows `10.0.26300` | `platform.platform()` |
| GPU | **NVIDIA GeForce RTX 5070 Laptop GPU**，8,546,484,224 B（8151 MiB），compute capability **12.0** | `torch.cuda.get_device_properties(0)` |
| GPU 驱动 | **595.79**（驱动侧 CUDA 13.2） | `nvidia-smi` |
| NPU | **Intel(R) AI Boost**，`DEVICE_TYPE = INTEGRATED`，驱动版本 **1003717**，能力 `FP16 / INT8 / EXPORT_IMPORT`，编译器版本 393219 | OpenVINO `core.get_property("NPU", …)` |

> WMI / PnP 查询在沙箱下不可用（`Get-CimInstance` 报「无法从客户端中访问 CIM 资源」），
> 因此 NPU 的型号/驱动/能力**全部取自 OpenVINO 运行时自己报的设备属性**，不是从设备管理器读的，也不靠推断。

---

## 5. 四种指标的口径（公式级，改口径即改 `config.py`）

| 指标 | 类型 | 骨干 / 权重 | 输出层 | 输入尺寸与预处理 | 标量定义 |
|---|---|---|---|---|---|
| Gram | **距离**（越小越像） | torchvision VGG-19 IMAGENET1K_V1 | `features[1,6,11,20,29]` = `relu1_1…relu5_1` | 512×512 直接缩放（bicubic，不留比例）+ ImageNet 归一化 | Gatys 式(11) 五层求和 |
| AdaIN | **距离**（越小越像） | 同上（复用同一份特征） | `relu1_1…relu4_1` | 同上 | `Σ_l (‖Δμ_l‖₂ + ‖Δσ_l‖₂)` |
| LPIPS | **距离**（越小越像） | richzhang LPIPS-AlexNet v0.1（官方实现） | AlexNet 5 层 + 校准 lin 层 | 256×256 直接缩放 + `[-1,1]` | 官方 `LPIPS.forward` 标量 |
| CSD | **相似度**（越大越像） | OpenAI CLIP ViT-L/14 visual + CSD 官方 style 头 | ViT 输出 → 1024→768 头 → **L2 归一化** | 短边 `Resize(224)`+`CenterCrop(224)` + **CLIP** 归一化 | 768 维描述子的余弦（=内积） |

**Gram / AdaIN / LPIPS 是距离指标，CSD 是相似度指标，分别报告，不合成未经校准的百分比。**

### 5.1 Gram 的精确公式

对第 l 层特征 `F_l`（`C_l × H_l × W_l`）：

```
G_l = (F_l @ F_lᵀ) / (C_l · H_l · W_l)          # 归一化 Gram（Gatys 式(4)，N = C·H·W）
d_l = ‖G_l^a − G_l^b‖_F² / (4 · C_l²)          # Gatys 式(11) 的单层项（权重全取 1）
gram_distance = Σ_l d_l                          # 五层求和
```

注意其**量级随层分辨率平方衰减**（实测落在 1e-7 ~ 1e-4），所以 JSON 里同时给出尺度无关的伴随量：
`summary.mean_layer_cosine_distance`（逐层 `1 − cos(vec(G_a), vec(G_b))` 的平均）与每层 `frobenius / relative / cosine_distance`。

### 5.2 AdaIN 的精确公式

```
μ_l = mean_{h,w} F_l ,  σ_l = std_{h,w} F_l        # 均为 R^{C_l}，σ 用 unbiased=False
d_l = ‖μ_a − μ_b‖₂ + ‖σ_a − σ_b‖₂
adain_distance = Σ_l d_l                            # relu1_1..relu4_1 四层
```

这是**深度特征统计量**的距离，**不是**原图 RGB 均值差，两者不可互相引用。

### 5.3 防"冒充"措施（这是本次交付的重点之一）

| 风险 | 本项目如何防止 |
|---|---|
| 用随机权重当 LPIPS | `LPIPSMetric` 强制 `pnet_rand=False`，并断言校准权重文件存在 |
| 用未校准的 VGG 特征 L2 冒充 LPIPS | 只用 `lpips` 官方包的 `LPIPS` 模块 + 作者发布的 `v0.1` 校准权重 |
| 用普通 CLIP / OpenCLIP / ResNet 冒充 CSD | CSD 路径必须同时具备 CLIP ViT-L/14 visual **和** CSD 官方 `last_layer_style`；官方权重缺失时直接 `unavailable`，不回落 |
| 静默退化成 CLIP 原始 `proj` | CSD 用 `load_state_dict(strict=True)`，0 missing / 0 unexpected 才继续 |
| 用分类 logits 代替中间特征 | VGG 导出 `relu1_1…relu5_1` 中间层；CSD 导出 768 维描述子；都没有分类头 |

---

## 6. 权重清单（来源 / 字节数 / SHA-256 全部核对过）

| 文件 | 字节数 | SHA-256 | 官方来源 | 许可 |
|---|---:|---|---|---|
| `vgg19/vgg19-dcbb9e9d.pth` | 574,673,361 | `dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44` | https://download.pytorch.org/models/vgg19-dcbb9e9d.pth | torchvision BSD-3-Clause |
| `lpips/v0.1/alex.pth` | 6,009 | `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0` | https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/alex.pth | LPIPS BSD-2-Clause |
| `lpips/v0.1/vgg.pth` | 7,289 | `a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868` | 同上 `vgg.pth` | LPIPS BSD-2-Clause |
| `lpips/trunk/alexnet-owt-7be5be79.pth` | 244,408,911 | `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` | https://download.pytorch.org/models/alexnet-owt-7be5be79.pth | torchvision BSD-3-Clause |
| `csd/CSD-ViT-L-pytorch_model.bin` | 2,438,228,893 | `40e92fad63a361b8136100cd234c42d401ef9b34ff1748234318929ebcc7e7a1` | https://huggingface.co/tomg-group-umd/CSD-ViT-L/resolve/main/pytorch_model.bin | 模型卡 CC-BY-4.0（代码 MIT，上游口径冲突） |

派生 / 缓存（可安全删除重建，已 gitignore）：

| 文件 | 体积 | 来源 |
|---|---|---|
| `cache/…/clip/ViT-L-14.pt` | 932,768,134 B | `https://openaipublic.azureedge.net/clip/models/b8cca3fd…/ViT-L-14.pt`，本地重算 SHA-256 **与 URL 内嵌值逐位一致** |
| `models/style-metrics/onnx/vgg19_features_512.onnx` | 51,785,433 B | 本仓库从 VGG-19 权重导出 |
| `models/style-metrics/onnx/lpips_alex_256.onnx` | 9,901,142 B | 官方 LPIPS 模块整体导出 |
| `models/style-metrics/onnx/csd_vit_l_224.onnx` | 1,216,354,242 B | CSD style 通路导出 |

### 6.1 CSD 权重与仓库 HEAD 的一处实测差异（**关键**）

`CSD/model.py` 写的是 `self.last_layer_style = copy.deepcopy(self.backbone.proj)`，
但官方 checkpoint 里的**实际形态**是：

- `module.last_layer_style` / `module.last_layer_content` 是**裸张量**，形状 **`(1024, 768)`**（即 (in, out)）；
- 键里**根本没有 `backbone.proj`**（与 `backbone.proj = None` 一致）。

实测 OpenAI CLIP 的 `visual.proj` **本身就是 `nn.Parameter(1024, 768)`**（不是 `nn.Linear`），
所以仓库 HEAD 的代码其实是自洽的；真正的坑在于**如果按 `nn.Linear` 建再 `strict=False` 加载，会全部静默跳过**，
style 头退化成 CLIP 原始 `proj` —— 那就成了「拿普通 CLIP 冒充 CSD」。

**本项目的做法**：按 checkpoint 真实形状建 `nn.Parameter(1024, 768)`，`strict=True` 加载。
实测结果 `missing = [] , unexpected = []`，297 个张量全部就位。

**进一步证据（证明加载的是"训练过的 CSD 头"，不是 CLIP 原始 proj）**：

| 量 | 值 |
|---|---|
| CSD `last_layer_style` 与 CLIP `visual.proj` 逐元素最大差 | **0.0781** |
| 两者展平后的余弦 | 0.9987 |
| 两者范数 | 12.94317 vs 12.94318 |

即：CSD 头是从 CLIP `proj` 初始化后**经过对比学习微调过**的（同范数、小幅移动）。
若加载失败退化成 `proj`，差值会是 0 —— 所以这组数字就是「没退化」的证据。

### 6.2 上游权重的已知问题（**必须如实转述**）

`learn2phoenix/CSD` 的 README 顶部至今挂着 DISCLAIMER：
「we are currently investigating an issue with our uploaded model weights … some discrepancy with the reported numbers」，
对应 issue #14 仍是 open（0 回复）。

因此本环境的定位是「**按官方权重与官方预处理复现描述子**」，
**不是**「复现论文 Table 1」。本报告也**不声称**在 WikiArt 上复算了 Recall / MRR / mAP。

---

## 7. 代码结构（全部新增，既有业务文件零改动）

| 文件 | 行数 | 职责 |
|---|---:|---|
| `utils/style_metrics/config.py` | 150 | 路径、权重清单、**全部固定预处理与输出层契约** |
| `utils/style_metrics/imaging.py` | 175 | 图像加载 / 三种预处理 / 确定性测试图 |
| `utils/style_metrics/gram.py` | 98 | Gram 矩阵与 Gatys 距离 |
| `utils/style_metrics/adain.py` | 71 | AdaIN 通道 μ/σ 距离 |
| `utils/style_metrics/vgg_encoder.py` | 107 | torchvision VGG-19 多层特征编码器 |
| `utils/style_metrics/lpips_metric.py` | 197 | 官方 LPIPS 封装 + NPU 变体 |
| `utils/style_metrics/csd_metric.py` | 412 | 官方 CSD 复刻 + 权重加载 + NPU 变体 |
| `utils/style_metrics/devices.py` | 336 | `auto/cuda/cpu/npu` 解析、运行期探针、硬件报告 |
| `utils/style_metrics/openvino_backend.py` | 283 | ONNX 导出 + Intel NPU 编译/推理/执行证据 |
| `utils/style_metrics/runner.py` | 233 | 统一运行器、**容差表**、计时 |
| `utils/style_metrics/inventory.py` | 185 | 权重 SHA-256 清单、缺失容错错误类型 |
| `utils/style_metrics/fetch.py` | 95 | 权重下载（requests，带字节数/SHA-256 校验） |
| `utils/style_metrics/__init__.py` | 56 | 公共入口 |
| `tools/style_metrics_verify.py` | 671 | **CLI 入口**（薄壳） |
| `tests/test_style_metrics.py` | 423 | 41 条回归（36 快 + 5 慢） |
| `tests/test_style_metrics_missing_models.py` | 249 | 17 条权重缺失容错回归 |
| `docs/style-extraction/style-metrics-setup.md` | 647 | 技术总纲 |
| `models/style-metrics/README.md` | 57 | 权重目录说明 |

**根目录未新增任何脚本**，有回归用例 `test_cli_does_not_add_root_scripts` 兜底断言。

---

## 8. 实测数值

测试输入：确定性合成图（`imaging.make_test_images()`，seed=20261004，384×384），三个用例：

- `same` —— 基准图 vs **自身的无损 PNG 往返**（同图，距离应≈0、相似度应≈1）
- `soft` —— 暖色 + 5×5 高斯模糊（**不同图，差异小**）
- `hard` —— 冷色 + 条纹调制 + 位移（**不同图，差异大**）

### 8.1 同图自比（必须≈0 / ≈1）

| 指标 | CPU FP32 | CUDA FP32 | NPU FP16 | 判定 |
|---|---:|---:|---:|---|
| Gram（期望 0） | 0.0 | 0.0 | 0.0 | ✅ |
| AdaIN（期望 0） | 0.0 | 0.0 | 0.0 | ✅ |
| LPIPS（期望 0） | 0.0 | 0.0 | 0.0 | ✅ |
| CSD（期望 1） | 1.0 | 1.0000001192 | 1.0 | ✅ |

> 同图自比在数学上是恒等式，**只能证明实现与数值稳定，不能证明计算正确** —— 所以下面必须看不同图。

### 8.2 不同图片（必须拉开距离 / 降低相似度）

| 指标 | CPU `soft` | CPU `hard` | NPU `soft` | NPU `hard` | 单调性 |
|---|---:|---:|---:|---:|---|
| Gram | 2.4399e-06 | 4.7438e-04 | 2.4382e-06 | 4.7443e-04 | ✅ |
| AdaIN | 37.1865 | 212.2727 | 37.1718 | 212.2788 | ✅ |
| LPIPS | 0.046027 | 0.798110 | 0.045898 | 0.797852 | ✅ |
| CSD（相似度） | 0.934550 | 0.730857 | 0.934008 | 0.729948 | ✅ |

四条都同时满足「同图≈极端值」与「不同图明显偏离」，**不是只靠同图测试就宣称正确**。

### 8.3 跨设备误差（**容差先声明、后验证**）

容差**在跑之前就写死在** `utils/style_metrics/runner.py::TOLERANCES`，
判据 `|a − b| ≤ atol + rtol·|b|`（`b` = CPU FP32 基准）。**没有事后放宽**。

| 指标 | 精度 | rtol | atol |
|---|---|---:|---:|
| gram | fp32 / fp16 | 1e-3 / 5e-2 | 1e-6 / 1e-4 |
| adain | fp32 / fp16 | 1e-3 / 5e-2 | 1e-6 / 1e-4 |
| lpips | fp32 / fp16 | 1e-3 / 1e-2 | 1e-5 / 1e-3 |
| csd | fp32 / fp16 | 1e-3 / 1e-2 | 1e-5 / 1e-3 |

同图自比上界：距离 1e-6、相似度 1e-4。

**CUDA FP32 vs CPU FP32**

| 指标 | 最大绝对误差 | 不同图最大相对误差 | 结论 |
|---|---:|---:|---|
| gram | 1.1465e-08 | 2.1001e-04 | ✅ 在容差内 |
| adain | 4.2908e-03 | 1.1539e-04 | ✅ |
| lpips | 2.7478e-05 | 3.4429e-05 | ✅ |
| csd | 1.1921e-07 | 1.2756e-07 | ✅（相似度误差 1.1921e-07） |

**NPU FP16 vs CPU FP32**

| 指标 | 最大绝对误差 | 不同图最大相对误差 | 结论 |
|---|---:|---:|---|
| gram | 4.5540e-08 | 6.6762e-04 | ✅ 在 fp16 容差内 |
| adain | 1.4645e-02 | 3.9383e-04 | ✅ |
| lpips | 2.5874e-04 | 2.7990e-03 | ✅ |
| csd | 9.0915e-04 | 1.2440e-03 | ✅（相似度误差 9.0915e-04） |

> **精度等价性声明**：NPU 上的 fp16 结果**不是** CPU/CUDA fp32 的等价替代。
> 数值虽都在声明的 fp16 容差内，但两者应作为**独立版本**分别引用；需要严格复现时以 CPU FP32 为准。
> 也**没有**做 INT8 量化，不声称量化等价。

---

## 9. 性能与显存

**测量协议**：batch = 1；输入固定为 384×384 合成图；`torch.get_num_threads()` 未改动（本机 24）；
NPU `PERFORMANCE_HINT = LATENCY`；每项先 warmup 1 次再取 3 次中位数；
CUDA 计时前后 `torch.cuda.synchronize()`，每个指标开始前 `reset_peak_memory_stats()`；
模型加载/编译耗时与推理耗时分列。

| 指标 | CPU FP32 中位 | CUDA FP32 中位 | CUDA 峰值显存 | NPU FP16 中位 | NPU 编译耗时（一次性） |
|---|---:|---:|---:|---:|---:|
| Gram | 189.18 ms | **74.85 ms** | 344.7 MB | 136.71 ms | 2.65 s |
| AdaIN | 219.14 ms | **97.60 ms** | 344.7 MB | 176.63 ms | 2.71 s |
| LPIPS | 223.69 ms | **6.84 ms** | 111.1 MB | 11.33 ms | 0.61 s |
| CSD | 4541.69 ms | **65.00 ms** | 1280.0 MB | 428.14 ms | 20.9 s |

> Gram / AdaIN 的每一档是**两次前向**（基准图 + 对照图各一次）加距离计算的总和；
> LPIPS / CSD 是单对图的端到端。
>
> **结论：本机 CUDA 全面快于 NPU，NPU 只比纯 CPU 快。**
> 选后端应据实测，而不是「NPU 一定更快」的假设 —— 这一点在任务要求里被明确点名，实测结果也确实如此。

---

## 10. NPU 执行证据（不是"编译成功"就算数）

四条指标的编码器都真实跑在 Intel NPU 上，`EXECUTION_DEVICES` 均指向 `NPU`：

| 指标 | 导出 ONNX 体积 | NPU 编译耗时 | 输出 | 该次运行 `execution_devices` |
|---|---:|---:|---|---|
| Gram | 51,785,433 B | 2.65 s | 5 层特征，全部有限 | `["NPU"]` |
| AdaIN | 同前一份 | 2.71 s | 同上 | `["NPU"]` |
| LPIPS | 9,901,142 B | 0.61 s | 1 个标量 + 5 个逐层标量 | `["NPU"]` |
| CSD | 1,216,354,242 B | 20.9 s | 768 维描述子，`‖d‖₂ ≈ 1` | `["NPU"]` |

运行期证据（写进 JSON 的 `detail.npu_encoder.property_evidence`）：

```json
{
  "NPU_DRIVER_VERSION": "1003717",
  "DEVICE_TYPE": "Type.INTEGRATED",
  "OPTIMIZATION_CAPABILITIES": "['FP16', 'INT8', 'EXPORT_IMPORT']",
  "NPU_COMPILER_VERSION": "393219",
  "compiled:EXECUTION_DEVICES": "NPU",
  "compiled:INFERENCE_PRECISION_HINT": "<Type: 'float16'>",
  "compiled:PERFORMANCE_HINT": "PerformanceMode.LATENCY"
}
```

### 10.1 「不算 NPU 推理成功」的明确列举

任务要求区分这些情况，本项目逐条区分：

| 情形 | 是否算成功 |
|---|---|
| OpenVINO 的 `available_devices` 里出现 `NPU` | ❌ 只说明驱动在 |
| ONNX 导出成功 | ❌ 只是格式转换，与执行设备无关 |
| 模型编译成功 | ❌ 还没跑数据 |
| 装了 NPU 运行时 / DirectML 可用 | ❌ 与 Intel NPU 是否真的执行无关 |
| **真实 infer 出有限数值 + `EXECUTION_DEVICES = NPU`** | ✅ **只有这条算** |

**踩过的坑**：`compiled.get_property("EXECUTION_DEVICES")` 在 OpenVINO 里可能返回**裸字符串** `"NPU"`，
`list("NPU")` 会得到 `['N','P','U']` 这种**假证据**。已用 `devices._as_list` 修正，并有回归用例
`test_as_list_does_not_split_strings`。

### 10.2 编码器在 NPU、距离在 CPU 的分工（如实记录）

| 指标 | NPU 上跑的是什么 | CPU 上跑的是什么 |
|---|---|---|
| Gram | **VGG-19 多层特征编码器**（`relu1_1…relu5_1`） | Gram 矩阵与距离 |
| AdaIN | 同上（同一份特征） | 通道 μ/σ 与距离 |
| LPIPS | **官方 LPIPS 模块整体**（含 ScalingLayer 与校准 lin 层） | 仅标量收集 |
| CSD | **style 描述子编码器**（ViT-L/14 visual + 官方头，输出 768 维） | 768 维余弦（内积） |

**没有任何一个指标是「整条链路都在 NPU」**；JSON 的 `timing.split` 字段逐条写明这个分工。

---

## 11. 设备选择策略（**不静默回落**）

CLI 入口 `--device auto|cuda|cpu|npu`：

| 选择 | 行为 |
|---|---|
| `auto` | 按 `cuda → npu → cpu` 取第一个**运行期探针通过**的后端，并写明最终选择与理由 |
| `cuda` | 探针不通就抛 `DeviceUnavailableError` 并报清原因 |
| `npu` | 探针不通就抛 `DeviceUnavailableError` 并报清原因 —— **绝不静默换 CPU 再声称 NPU 成功** |
| `cpu` | 强制 FP32 基准路径（不声称 CPU FP16 已验证） |

`--device npu` 时 `precision` 自动记为 **`fp16`**，并在 `note` 里写明
「Intel NPU 只支持 FP16/INT8，特征按 fp16 计算（独立版本，不等价于 fp32）」。

---

## 12. 权重缺失容错（本轮新增）

裁决标准只有一条：**缺权重就报清楚、能继续、绝不冒充成功。**

| 情形 | 行为 |
|---|---|
| 权重文件不存在 | 抛 `inventory.WeightsMissing`（`FileNotFoundError` 子类，老写法仍能接住） |
| 权重字节数不符 | 同上，消息里写明「实际 N，期望 M」 |
| 只缺 LPIPS 骨架 / 只缺 CLIP 骨架 | 在**进 torch 之前**报出来，不让底层库抛内部异常 |
| 某个指标缺权重 | 该指标 `status="unavailable"`，**其它指标照常跑完** |
| 全部缺权重 | 四条都 `unavailable`，**不产出任何数值**，JSON 写明「未产出任何数值（不做 0 分处理）」 |

`WeightsMissing` 自带完整修复信息，CLI 原样打印：

```
缺少画风指标权重 [vgg19]：文件不存在
  期望路径 : D:\code\image-maker\models\style-metrics\vgg19\vgg19-dcbb9e9d.pth
  官方来源 : https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
  字节数   : 574673361
  SHA-256  : dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44
  修复命令 : "…python.exe" -m utils.style_metrics.fetch "<url>" "<path>" --sha256 <sha>
```

**明确不做的事**：

- 不静默跳过某条指标然后照常宣称「全部成功」
- 不把缺失当 0 分，不把缺失当 `error`（`unavailable` ≠ `error`：前者没跑、后者跑了失败）
- 不自动联网下载（要下得显式加 `--download-missing`，全程校验断点/字节数/SHA-256）
- 不换别的模型顶替（缺 CSD 就报缺 CSD，不会拿普通 CLIP 顶上）

**权重目录可移植**：`IMAGE_MAKER_STYLE_METRICS_HOME` / `IMAGE_MAKER_STYLE_METRICS_CACHE`
两个环境变量可覆盖权重与缓存根目录（默认在仓库内），所以代码里**不写死任何机器路径**，
「权重缺失」这条路径也能被自动化用例端到端跑（17 条）。

---

## 13. 敏感数据检查（本地）

写了一个临时扫描器（放在 gitignored 的 `cache/temp/scan_pending.py`，可重跑），覆盖 15 类模式：
OpenAI / Anthropic / GitHub / AWS / Slack / Google / HF 密钥、PEM 私钥、Bearer 字面量、
赋值式 secret、Windows 用户路径、POSIX home 路径、内网 IP、Email、`.env` 变量赋值。

| 对象 | 结果 |
|---|---|
| 本次提交的 21 个文件 | **零命中**（用 `git grep --cached` 精确复扫，密钥类命中 0，退出码 1） |
| `.env` | 存在，已被 `.gitignore` 覆盖（未提交） |
| `conf/*.json` | 已被 gitignore（未提交） |
| 权重 `.pth/.bin` | 已被 gitignore（未提交） |
| `models/style-metrics/onnx/` | **本轮补加了 gitignore 规则**（此前 `_npu_probe.json` 会漏进去） |

**发现并修掉的 1 处**：`style-metrics-setup.md` 里写了本机用户主目录的绝对路径（泄漏用户名），
已改成 `%APPDATA%\Python\Python310\site-packages\lpips`。

**未修的 3 处**（不是本次会话的文件，没擅自改，已上报）：

- `docs/style-analyzer-workflow.md:66`
- `docs/261003-color-improve/COLOR-CONFORMANCE-REASSESSMENT-20261004.md:191`
- `docs/style-extraction/millon-knots-20261004.md:9`

三处都是同一类：绝对路径里的 `C:\Users\<用户名>\…` 泄漏了本机用户名。
**若要连这些一起提交，建议先替换成 `%APPDATA%` / `%USERPROFILE%`。**

---

## 14. 提交记录

```
c66761c  Add style metrics (Gram/AdaIN/LPIPS/CSD) with cross-device verification
         master · 21 files changed, 4490 insertions(+)
```

**提交内容**（全部为本次会话产出）：

- 新增：`utils/style_metrics/`（13 个模块）、`tools/style_metrics_verify.py`、
  `tests/test_style_metrics.py`、`tests/test_style_metrics_missing_models.py`、
  `docs/style-extraction/style-metrics-setup.md`、`models/style-metrics/README.md`
- 修改：`AGENTS.md`（3 条索引）、`.gitignore`（+3 行）、`tests/conftest.py`（+39 行，见 §16）

**权重一个都没进库**：`git ls-files models/style-metrics` 只返回 `README.md`。

**刻意未提交**的 52 个文件是其他会话的未完成改动
（`app.py`、`publish_server.py`、`color-improve` 那批、`style-dataset-curator` 技能等）——
来历不清，且其中 3 个带用户名路径，没有混进这个 commit。

---

## 15. 三个踩过的坑（都留下了证据和回归）

### 15.1 CSD 权重加载会静默退化

见 §6.1。若不按 checkpoint 真实形状建参数，`strict=False` 会静默跳过整个 style 头，
指标就变成了「普通 CLIP」。**修法**：`nn.Parameter(1024, 768)` + `strict=True`，并用
「与 CLIP `proj` 的最大差 0.0781」证明确实加载了训练过的头。

### 15.2 `EXECUTION_DEVICES` 返回裸字符串导致假设备证据

见 §10.1。`list("NPU")` → `['N','P','U']`。
**修法**：`devices._as_list`，回归用例 `test_as_list_does_not_split_strings`。

### 15.3 Windows 上 torch 的 DLL 必须**先于**重型原生库加载

加上本模块的用例后，`python -m pytest -q` **在收集阶段**就失败：

```
OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败。
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

定位实验（四条都实测过）：

| 实验 | 结果 |
|---|---|
| 单独 `import onnxruntime` / `cv2` / `openvino` / `numpy` / `PIL` 后再 `import torch` | **全部正常** → 不是两两 DLL 冲突 |
| 只收集 2 个模块（`test_analysis_cli_unittest.py` + `test_style_metrics.py`） | **正常** |
| 收集**整套** `tests/`（PyQt6 + onnxruntime-directml + opencv + openvino 都进过进程）后再加载 torch | **必炸** |
| 先 `import torch` 再让 pytest 收集同一套用例 | **856 项全部收集成功、0 错误** |

即：这台 Windows 上 torch 的 DLL 需要一个尚未被其它大型原生库占满的加载环境，晚了就 `DllMain` 失败。
**修法**：`tests/conftest.py` 加「守卫三」，会话最开始把 torch 顶上来（实测量价约 1.5 秒/会话，
可用 `IMAGE_MAKER_TESTS_SKIP_TORCH_PRELOAD=1` 跳过）。

这条坑与画风指标本身无关，但**任何后续新增的 torch 用例都会踩到**。

---

## 16. 测试状态

| 范围 | 结果 |
|---|---|
| `tests/test_style_metrics.py` | 41 条收集（36 快测 + 5 慢测） |
| `tests/test_style_metrics_missing_models.py` | 17 条 |
| 两个模块合跑 | **53 passed, 5 skipped**（慢测需 `STYLE_METRICS_SLOW=1`） |
| 慢测（真实权重前向） | **10 passed**（含前端到端：VGG 形状 / Gram·AdaIN 同图 0 / LPIPS 校准权重 / CSD 同图余弦） |
| **全仓 `python -m pytest -q -p no:cacheprovider`** | **851 passed, 5 skipped, 12 subtests passed（77.5 s，exit 0）** |

---

## 17. 对现有 App 的影响

- **没有修改任何 App 业务代码**：`app.py`、`modules/**`、既有 `utils/**` 文件一个都没动，
  新代码全落在新目录 `utils/style_metrics/`。
- **没有动评分公式**：`docs/style-extraction/scoring-research-20261004.md` 描述的 12 维量表与
  `0.85×S + 0.15×主体符合度` 保持原样；本环境产出的四个指标**不参与**该总分。
- **没有调用付费生图 / 文本 API**：全流程本地计算，测试图是合成图，未启动任何生图任务。
- **原有关键模块导入自检**（13 个，实测）：`utils.styles` / `utils.analysis_gen` / `utils.analysis_gpt_prompt` /
  `utils.gpt_image_optimize` / `utils.post_process` / `utils.cost_estimate` / `utils.style_gpt` /
  `utils.output_isolation` / `utils.prompt_loader` / `modules.others.api_backend` /
  `modules.image_analysis.analysis_pipeline` / `modules.image_analysis.single_analyzer` /
  `modules.image_generation.gpt_image2_tab` —— **全部 `ok`**。
- **是否需要重启 App？** —— **不需要**。本次没有修改 App 运行的任何模块，也没有改配置；
  新增的 `utils/style_metrics/` 是独立子包，App 不导入它。

---

## 18. 复现命令

```powershell
# ① 权重与依赖状态（含 SHA-256 校验），不跑前向；缺权重会退出码 1 并给修复命令
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --inventory

# ② 全套四指标验证（自动挑已验证可用的后端），写 data/test-result/*.json
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --device auto

# ③ 指定后端 / 精度
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --device cuda --precision fp32
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --device npu

# ④ 权重缺失时让 CLI 自己按官方来源下载（默认不联网下载）
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --device auto --download-missing

# ⑤ 回归用例（慢测加 STYLE_METRICS_SLOW=1）
& "C:\Program Files\Python310\python.exe" -m pytest -q -p no:cacheprovider tests/test_style_metrics.py
```

---

## 19. 机器可读 JSON（三轮实测各一份）

| 文件 | 大小 | 后端 | 精度 | status_counts |
|---|---:|---|---|---|
| `data/test-result/style-metrics-cpu-baseline.json` | 21,212 B | cpu | fp32 | `{"ok": 4}` |
| `data/test-result/style-metrics-cuda-fp32.json` | 27,367 B | cuda | fp32 | `{"ok": 4}` |
| `data/test-result/style-metrics-npu.json` | 27,855 B | npu | fp16 | `{"ok": 4}` |

`status` 取值与含义（**不要混用**）：

| status | 含义 |
|---|---|
| `ok` | 跑完且自比检查通过 |
| `partial` | 跑完但有非致命问题 |
| `error` | 跑了但失败（输入错误 / 数值异常） |
| `unavailable` | **没跑**：设备不可用或权重缺失；`remediation` 给修复办法，**绝不产生假数值** |
| `failed_self_check` | 跑完但同图自比超出容差 |

每条记录含：`metric / kind / status / model / weights_sha256 / preprocessing /
requested_device / actual_device / backend / precision / validation / timing / detail /
remediation / error`。

---

## 20. 已知限制与未完成项（不粉饰）

1. **CSD 未做论文级复现**：上游权重自带 DISCLAIMER，本环境只验证了「官方描述子能被复现且同图余弦 = 1」，
   没有在 WikiArt 上复算 Recall / MRR / mAP。
2. **NPU 是 FP16 独立版本**，与 CPU/CUDA FP32 不等价；跨设备误差虽在声明的 fp16 容差内，但不得当同一口径引用。
3. **没有做内容解耦验证**：四个指标都受内容影响（尤其 LPIPS / CSD 对主体变化敏感），
   本报告**没有断言**它们等于「纯画风相似度」。
4. **测试图是合成图**：`soft` / `hard` 两个变体只证明「能区分不同图」，
   不等于在真实动漫插画上的判别力已经校准。
5. **未做 INT8 量化**：NPU 支持 INT8，但本项目只用 FP16，没有量化版本，也不声称量化等价。
6. **未测批量吞吐**：所有计时都是 batch = 1。多图批量的显存与吞吐未测。
7. **ONNX Runtime 的发行包冲突未修**（按纪律只记录）。
8. **未更新任何驱动 / BIOS，未重启系统**（按要求）。
9. **没有做人工一致性标注**，因此四个指标的数值**没有被校准**为人类相似度或百分比。

---

## 21. 给审阅者（ChatGPT）的建议核查点

如果你要审这份交付，最有价值的几个问题是：

1. **口径是否被偷换？**  §5 的公式与 §6 的权重来源是否对应得上（尤其 LPIPS 是否用官方校准权重、
   CSD 是否真用了官方 style 头 —— §6.1 的「0.0781 差异」是判据）。
2. **「实际设备已确认」是否成立？**  §10 的证据链（`EXECUTION_DEVICES = NPU` + 真实有限输出）够不够，
   有没有把「编译成功」当成「推理成功」。
3. **跨设备声明是否过度？**  §8.3 的容差是**先声明后验证**的，NPU 的 fp16 被明确标为独立版本 —— 这个边界划得对不对。
4. **失败路径是否诚实？**  §11（设备不可用不静默回落）与 §12（权重缺失不给假数值、不当 0 分）是否符合预期。
5. **有没有不该声称的结论？**  §20 列的 9 条限制里，是否有该列而漏列的（例如「测试图是合成图」这条是否足以支撑
   「可用于评价真实插画」的说法 —— 本报告的立场是**不能**）。
6. **有没有安全/隐私问题？**  §13 的扫描范围与漏检风险（例如二进制权重文件没有做内容扫描，
   只做了哈希核对）。
