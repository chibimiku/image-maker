# 画风深度特征指标：权重、代码与跨设备（GPU / NPU / CPU）验证环境

> **2026-10-06 口径修正**：默认 Gram 已改为 `gatys-layer-sum/v2`；下文环境及跨设备数值表为 10 月 4 日历史记录，Gram 属于旧版 v1，不能当作新版验证。首轮未完成 LPIPS/CSD 验证，后续真实 CPU 双图八组完整验证见 [共享画风相似度说明](STYLE-SIMILARITY-20261006.md)。新版 CPU 验证与真实配对修复见 `STYLE-METRICS-GRAM-V2-20261006.md`。

> 目标：在**不重装 PyTorch、不建虚拟环境、不改 App 评分公式、不训练 LoRA、不调用付费生图 API** 的前提下，
> 为四种画风指标（Gram / AdaIN / LPIPS / CSD）准备可复现的权重、计算代码与跨设备验证环境。
>
> 本次所有结论都来自本机实测。**未安装任何新依赖**（现有包已足够），**未更新驱动 / BIOS / 重启**，
> **未终止正在运行的 App**。

---

## 0. 状态分层（一张表说清「做到哪一步」）

「依赖可导入」「权重已取得」「模型可运行」「实际设备已确认」「数值验证通过」是**五件不同的事**，
本项目分级记录，不合并成一句「支持」：

| 指标 | 依赖可导入 | 权重已取得 | 模型可运行 | 实际设备已确认 | 数值验证通过 |
|---|---|---|---|---|---|
| **Gram**（VGG-19） | ✅ lpips/torchvision | ✅ 574,673,361 B | ✅ 前向有限 | ✅ CPU / CUDA / **NPU** | ✅ 同图 0、跨设备在容差内 |
| **AdaIN**（VGG-19） | ✅ 同上 | ✅ 同上 | ✅ 前向有限 | ✅ CPU / CUDA / **NPU** | ✅ 同图 0、跨设备在容差内 |
| **LPIPS**（AlexNet v0.1） | ✅ `lpips` 0.1.4 | ✅ 校准权重 6,009 B + 骨架 244,408,911 B | ✅ 官方 `LPIPS.forward` | ✅ CPU / CUDA / **NPU** | ✅ 同图 0、跨设备在容差内 |
| **CSD**（ViT-L/14） | ✅ OpenAI `clip` 1.0 | ✅ 2,438,228,893 B | ✅ 官方 style 头 `strict=True` 干净加载 | ✅ CPU / CUDA / **NPU** | ✅ 同图 1.0、跨设备在容差内 |

**明确未完成 / 不声称的部分**：

- 没有把任何一个指标的数值**校准**成人类相似度或百分比，也没有动 App 的 `style_score` 公式。
- CSD 上游权重存在**作者自己声明的已知问题**（README DISCLAIMER + issue #14 仍 open），
  本环境只声称「按官方权重与官方预处理复现描述子」，**不声称复现论文指标**。
- LPIPS/CSD 的 **NPU 版本是 FP16 独立版本**，与 CPU/CUDA 的 FP32 结果**不等价**（差异已在 §5 量化），
  不允许把两者当同一口径混用。
- 项目没有 LoRA 训练过程，因此**不存在真实的 LoRA training loss**，本文件不计算也不伪造。

---

## 1. 四种指标的口径（固定契约）

全部契约集中在 `utils/style_metrics/config.py`，改它就是改口径。

### 1.1 一览

| 指标 | 类型 | 骨干 / 权重 | 输出层 | 输入尺寸与预处理 | 标量定义 |
|---|---|---|---|---|---|
| Gram | **距离**（越小越像） | torchvision VGG-19 IMAGENET1K_V1 | `features[1,6,11,20,29]` = `relu1_1…relu5_1` | 512×512 直接缩放（bicubic，不留比例）+ ImageNet 归一化 | Gatys 单层归一化、五层等权求和（v2） |
| AdaIN | **距离**（越小越像） | 同上（复用同一份特征） | `relu1_1…relu4_1` | 同上 | `Σ_l (‖Δμ_l‖₂ + ‖Δσ_l‖₂)` |
| LPIPS | **距离**（越小越像） | richzhang LPIPS-AlexNet v0.1（官方实现） | AlexNet 5 层 + 校准 lin 层 | 256×256 直接缩放 + `[-1,1]` | 官方 `LPIPS.forward` 标量 |
| CSD | **相似度**（越大越像） | OpenAI CLIP ViT-L/14 visual + CSD 官方 style 头 | ViT 输出 → 1024→768 头 → **L2 归一化** | 短边 Resize(224)+CenterCrop(224) + **CLIP** 归一化 | 768 维描述子的余弦（=内积） |

**Gram 与 AdaIN 是距离指标，CSD 是相似度指标，分别报告，不合成未校准的百分比。**

### 1.2 Gram 的精确公式

对第 l 层特征 `F_l`（`C_l × H_l × W_l`）：

```
G_l = (F_l @ F_lᵀ) / (C_l · H_l · W_l)    # 项目使用归一化 Gram
 d_l = ‖G_l^a − G_l^b‖_F² / 4               # 已含 C²(HW)²，不再除 C²
 gram_distance = Σ_l d_l                    # 五层权重全部为 1
```

等价于对原始 Gram 使用 `‖ΔG_raw‖² / (4 C² (HW)²)`。单层归一化对应 [Gatys 原论文](https://arxiv.org/pdf/1508.06576) v2 式(3)–(5)，总层权重为项目约定，不宣称完全复现论文训练损失。
旧版 `normalized-gram-extra-channel/v1` 重复除以 C²，只可显式用于历史复算；旧表数值保留。不同层通道数不同，不能按一个常数换算旧总分。
归一化统计量没有统一的“随分辨率平方衰减”规律；重复相同空间样本的距离保持不变。JSON 同时保留逐层 Frobenius、relative、cosine_distance，不能将这些量合成未经校准的百分比。

### 1.3 AdaIN 的精确公式

```
μ_l = mean_{h,w} F_l ,  σ_l = std_{h,w} F_l        # 均为 R^{C_l}，σ 用 unbiased=False
d_l = ‖μ_a − μ_b‖₂ + ‖σ_a − σ_b‖₂
adain_distance = Σ_l d_l                            # relu1_1..relu4_1 四层
```

这是**深度特征统计量**的距离，**不是**原图 RGB 均值差，两者不可互相引用。

### 1.4 为什么「不能用别的东西冒充」

| 风险 | 本项目如何防止 |
|---|---|
| 用随机权重当 LPIPS | `LPIPSMetric` 强制 `pnet_rand=False`，并断言校准权重文件存在（`models/style-metrics/lpips/v0.1/alex.pth`） |
| 用未校准的 VGG 特征 L2 冒充 LPIPS | 只用 `lpips` 官方包的 `LPIPS` 模块 + 作者发布的 `v0.1` 校准权重 |
| 用普通 CLIP / OpenCLIP / ResNet 冒充 CSD | CSD 路径必须同时具备 CLIP ViT-L/14 visual **和** CSD 官方 `last_layer_style`；官方权重缺失时直接 `status="unavailable"`，不回落 |
| 静默退化成 CLIP 原始 `proj` | CSD 用 `load_state_dict(strict=True)`；0 missing / 0 unexpected 才继续（详见 §3.2） |
| 用分类 logits 代替中间特征 | VGG 导出的是 `relu1_1…relu5_1` 中间层；CSD 导出的是 768 维描述子；都没有分类头 |

---

## 2. 依赖与环境（已确认，未做任何安装）

全部使用系统 Python：`C:\Program Files\Python310\python.exe` → **Python 3.10.11** (MSC v.1929 64 bit AMD64)。
**没有创建虚拟环境，没有升级/重装 PyTorch、Torchvision 或 CUDA，没有安装任何新包。**

| 包 | 实测版本 | 说明 |
|---|---|---|
| torch | **2.10.0+cu128** | CUDA 构建 12.8；`torch.cuda.is_available() == True` |
| torchvision | **0.25.0+cu128** | VGG-19 / AlexNet 预训练权重来源 |
| lpips | **0.1.4** | 用户级 site-packages 下的 `lpips/`（`%APPDATA%\Python\Python310\site-packages\lpips`）；**自带官方 v0.0/v0.1 校准权重** |
| open_clip_torch | **3.3.0** | `import open_clip` 正常，`__version__ = 3.3.0`（本任务未使用，仅按要求确认） |
| clip | **1.0** | 见下 |
| transformers | 4.57.6 | 本任务未使用 |
| onnxruntime | 1.23.0（导入值） | 见下 |
| openvino | **2026.1.0** | NPU 插件来源 |
| numpy / Pillow | 2.2.6 / 12.2.0 | |
| pytest | 9.0.3 | 回归用例 |

### 2.1 `clip` 是不是官方 OpenAI CLIP？（按要求甄别同名包）

**是官方包，不是同名误装。** 证据（全部运行期实测）：

- `clip.__file__` = `...\site-packages\clip\__init__.py`，同级存在 `clip.py` / `model.py` / `simple_tokenizer.py`
  —— 与 `openai/CLIP` 仓库的官方模块布局一致。
- 官方接口齐备：`clip.load`、`clip.tokenize`、`clip.available_models`。
  - `clip.load` 签名：`(name: str, device: Union[str, torch.device] = 'cuda', jit: bool = False, download_root: str = None)`
  - `clip.tokenize` 签名：`(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False)`
  - `clip.available_models()` = `['RN50','RN101','RN50x4','RN50x16','RN50x64','ViT-B/32','ViT-B/16','ViT-L/14','ViT-L/14@336px']`
- 内部 `clip.clip._MODELS` 表存在（官方实现用下划线私有变量，`import *` 不会导出，属预期）。
- 分发包元数据：`clip 1.0`，与 `openai/CLIP` 的 `setup.py` 版本号一致。
- **实际跑通**：`clip.load("ViT-L/14")` 成功下载并加载官方 ViT-L/14 权重（见 §3.4 哈希核对）。

CSD 的 backbone 正是用这个 `clip.load`，所以「官方 CLIP + 官方 CSD 头」两条前提都成立。

### 2.2 ONNX Runtime 的发行包冲突（**已记录，按纪律不修**）

`pip list` 同时存在两个拥有同一个 `site-packages\onnxruntime` 目录的发行包：

```
onnxruntime            1.23.2
onnxruntime-directml   1.23.0     ← 实际生效
```

实测 `import onnxruntime` 报 `__version__ = 1.23.0`，`get_available_providers()` = `['DmlExecutionProvider', 'CPUExecutionProvider']`
—— 即当前可导入的是 **DirectML 构建**，普通 CPU/CUDA 版 ORT 的目录已被覆盖。

**处理方式**：只记录，不重装、不删除、不混装。本机 NPU 是 **Intel** NPU，
正确的 NPU 运行时是 **OpenVINO 的 NPU 插件**，不是 ORT QNN，也不把 DirectML 当 NPU。
因此本次 NPU 路径完全绕开 ONNX Runtime（ONNX 文件只是交给 OpenVINO 读）。

---

## 3. 权重：来源、路径、SHA-256

全部落在 `models/style-metrics/`（`.gitignore` 已忽略 `*.pth` / `*.pt` / `*.bin` / `*.onnx`，**不入库**）。
下载缓存落在工作区 `cache/style-metrics-download/`。

### 3.1 清单

| 文件 | 字节数 | SHA-256 | 官方来源 | 许可 |
|---|---:|---|---|---|
| `vgg19/vgg19-dcbb9e9d.pth` | 574,673,361 | `dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44` | `https://download.pytorch.org/models/vgg19-dcbb9e9d.pth` | torchvision BSD-3-Clause |
| `lpips/v0.1/alex.pth` | 6,009 | `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0` | `https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/alex.pth` | LPIPS BSD-2-Clause |
| `lpips/v0.1/vgg.pth` | 7,289 | `a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868` | 同上 `vgg.pth` | LPIPS BSD-2-Clause |
| `lpips/trunk/alexnet-owt-7be5be79.pth` | 244,408,911 | `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` | `https://download.pytorch.org/models/alexnet-owt-7be5be79.pth` | torchvision BSD-3-Clause |
| `csd/CSD-ViT-L-pytorch_model.bin` | 2,438,228,893 | `40e92fad63a361b8136100cd234c42d401ef9b34ff1748234318929ebcc7e7a1` | `https://huggingface.co/tomg-group-umd/CSD-ViT-L/resolve/main/pytorch_model.bin` | 模型卡 CC-BY-4.0（代码 MIT，上游口径冲突） |

派生 / 缓存（可安全删除重建）：

| 文件 | 位置 | 来源与校验 |
|---|---|---|
| OpenAI CLIP ViT-L/14 骨架 | `cache/style-metrics-download/clip/ViT-L-14.pt` | `https://openaipublic.azureedge.net/clip/models/b8cca3fd…/ViT-L-14.pt`，932,768,134 B |
| VGG-19 特征编码器 ONNX | `models/style-metrics/onnx/vgg19_features_512.onnx` | 51,785,433 B，由本仓库从上面的 VGG-19 权重导出 |
| LPIPS ONNX | `models/style-metrics/onnx/lpips_alex_256.onnx` | 9,901,142 B，官方 LPIPS 模块整体导出 |
| CSD 描述子 ONNX | `models/style-metrics/onnx/csd_vit_l_224.onnx` | 1,216,354,242 B，CSD style 通路导出 |

### 3.2 CSD 权重与仓库 HEAD 的一处实测差异（**必须知道，否则会静默退化**）

`CSD/model.py` 里写的是：

```python
self.last_layer_style = copy.deepcopy(self.backbone.proj)   # 看起来像 nn.Linear
...
style_output = feature @ self.last_layer_style
```

但官方 checkpoint 里的实际形态是：

- `module.last_layer_style` / `module.last_layer_content` 是**裸张量**，形状 **`(1024, 768)`**（即 (in, out)）；
- 键里**根本没有 `backbone.proj`**（与 `backbone.proj = None` 一致）；
- 实测 OpenAI CLIP 的 `visual.proj` **本身就是 `nn.Parameter(1024, 768)`**，不是 `nn.Linear`
  —— 所以仓库 HEAD 的代码其实是自洽的，真正的坑在于**如果按 `nn.Linear` 建再 `strict=False` 加载，会全部静默跳过**。

**本项目的做法**：按 checkpoint 的真实形状建 `nn.Parameter(1024, 768)`，用 **`strict=True`** 加载。
实测结果：`missing = [] , unexpected = []`，297 个张量全部就位。

**进一步证据（证明加载的是「训练过的 CSD 头」而不是 CLIP 原始 `proj`）**：

| 量 | 值 |
|---|---|
| CSD `last_layer_style` 与 CLIP `visual.proj` 逐元素最大差 | **0.0781** |
| 两者展平后的余弦 | 0.9987 |
| 两者范数 | 12.94317 vs 12.94318 |

即：CSD 头是从 CLIP `proj` 初始化后**经过对比学习微调过**的（同范数、小幅移动），
这正是「真 CSD 头」应有的样子；若加载失败退化成 `proj`，差值会是 0。

### 3.3 上游权重许可证与已知问题

- 代码：`learn2phoenix/CSD` 的 `LICENSE` 是 **MIT**；但 `main_sim.py` / `CSD/utils.py` 保留 FAIR **Apache-2.0** 头（MoCo/DINO 来源），属混合来源。
- 权重：HF 模型卡写 **CC-BY-4.0**，而 `CSD/model.py` 里的 `PyTorchModelHubMixin(license="mit")` 写 MIT —— **上游两处口径冲突**，本项目**并列记录，不替上游裁决**。
- **上游已声明的问题**：README 顶部至今挂着
  “we are currently investigating an issue with our uploaded model weights … some discrepancy with the reported numbers”，
  对应 issue #14 仍是 open。故本环境的定位是「复现官方描述子」，**不是**「复现论文 Table 1」。

### 3.4 CLIP 骨架的哈希核对

下载 URL 路径里内嵌的 sha256 就是 OpenAI 发布的校验值：

```
b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836
```

本地重算 `cache/style-metrics-download/clip/ViT-L-14.pt` 的 SHA-256 → **逐位一致**。

### 3.5 权重校验命令

```powershell
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --inventory
```

会对 VGG-19 / LPIPS 两套权重重算 SHA-256 并与清单比对；CSD 的 2.44 GB 哈希在下载时已逐字节算过并与 HF LFS 元数据核对，
清单里直接引用常量以免每次重算。**有任何一项缺失或字节数不符，退出码为 1 并打印修复命令。**

### 3.6 权重缺失时的容错（明确不做什么）

裁决标准只有一条：**缺权重就报清楚、能继续、绝不冒充成功。**

| 情形 | 行为 |
|---|---|
| 权重文件不存在 | 抛 `inventory.WeightsMissing`（是 `FileNotFoundError` 的子类，老写法仍能接住） |
| 权重字节数不符 | 同上，消息里写明「实际 N，期望 M」 |
| 只缺 LPIPS 骨架 / 只缺 CLIP 骨架 | 同样在**进 torch 之前**报出来，不让底层库抛内部异常 |
| 某个指标缺权重 | 该指标 `status="unavailable"`，**其它指标照常跑完** |
| 全部缺权重 | 四条都 `unavailable`，**不产出任何数值**，JSON 里 `validation.note` 写明「未产出任何数值（不做 0 分处理）」 |

`WeightsMissing` 自带完整修复信息，CLI 会原样打印：

```
缺少画风指标权重 [vgg19]：文件不存在
  期望路径 : D:\code\image-maker\models\style-metrics\vgg19\vgg19-dcbb9e9d.pth
  官方来源 : https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
  字节数   : 574673361
  SHA-256  : dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44
  修复命令 : "C:\Program Files\Python310\python.exe" -m utils.style_metrics.fetch "<url>" "<path>" --sha256 <sha>
```

**明确不做的事**：

- 不静默跳过某条指标然后照常宣称「全部成功」；
- 不把缺失当 0 分、不把缺失当 `error`（`unavailable` ≠ `error`，前者是「没跑」，后者是「跑了但失败」）；
- 不自动联网下载（要下得显式加 `--download-missing`，下载断点、字节数、SHA-256 全程校验）；
- 不换别的模型顶替（缺 CSD 就报缺 CSD，不会拿普通 CLIP 顶上）。

**权重目录可移植**：`IMAGE_MAKER_STYLE_METRICS_HOME` / `IMAGE_MAKER_STYLE_METRICS_CACHE`
两个环境变量可以覆盖权重与缓存根目录（默认在仓库内），因此代码里不写死任何机器路径，
「权重缺失」这条路径也能被自动化用例端到端跑（见 `tests/test_style_metrics_missing_models.py`，17 条）。

---

## 4. 硬件与后端（实测，不猜测）

### 4.1 本机硬件

| 部件 | 实测值 | 证据来源 |
|---|---|---|
| CPU | **Intel(R) Core(TM) Ultra 9 275HX**，24 逻辑核，≈3.07 GHz | `platform` / `py-cpuinfo` / `psutil` |
| 内存 | 136,755,625,984 B（≈127.4 GiB） | `psutil.virtual_memory()` |
| 系统 | Windows `10.0.26300` | `platform.platform()` |
| GPU | **NVIDIA GeForce RTX 5070 Laptop GPU**，8,546,484,224 B（8151 MiB），compute capability **12.0** | `torch.cuda.get_device_properties(0)` |
| GPU 驱动 | **595.79**（驱动侧 CUDA 13.2），`nvidia-smi` 无运行中进程 | `nvidia-smi` |
| NPU | **Intel(R) AI Boost**，`DEVICE_TYPE = INTEGRATED`，驱动版本 **1003717**，能力 `FP16 / INT8 / EXPORT_IMPORT`，编译器版本 393219 | OpenVINO `core.get_property("NPU", …)` |

> WMI / PnP 查询在本次沙箱下不可用（`Get-CimInstance` 返回“无法从客户端中访问 CIM 资源”），
> 因此 NPU 的型号/驱动/能力全部取自 **OpenVINO 运行时自己报的设备属性**，不是从设备管理器读的，也不靠推断。

### 4.2 三个后端与「可用」的定义

| 后端 | 运行时 | 「可用」的判据（本项目的定义） |
|---|---|---|
| `cpu` | PyTorch CPU | 永远可用（基准回退路径） |
| `cuda` | PyTorch CUDA | `torch.cuda.is_available()` **且**真的做一次 256×256 矩阵乘并 `synchronize()` 拿到有限值 |
| `npu` | OpenVINO NPU 插件（Intel AI Boost） | 在 NPU 上**真实编译并跑完一次前向**、拿到有限输出，且 `EXECUTION_DEVICES` 指向 NPU |

**明确不算「NPU 推理成功」的情形**（本项目把它们与成功区分开）：

- 「OpenVINO 的 `available_devices` 里出现了 `NPU`」→ 只说明驱动在，不算成功；
- 「ONNX 导出成功」→ 只是格式转换，与执行设备无关；
- 「模型编译成功」→ 还没跑数据；
- 「装了 NPU 运行时」/「DirectML 可用」→ 与 Intel NPU 是否真的执行无关。

### 4.3 设备选择策略

CLI 入口 `--device auto|cuda|cpu|npu`：

| 选择 | 行为 |
|---|---|
| `auto` | 按 `cuda → npu → cpu` 取第一个**运行期探针通过**的后端，并在输出里写明最终选择与理由 |
| `cuda` | 探针不通就抛 `DeviceUnavailableError` 并报清原因 |
| `npu` | 探针不通就抛 `DeviceUnavailableError` 并报清原因 —— **绝不静默换 CPU 再声称 NPU 成功** |
| `cpu` | 强制 FP32 基准路径（不声称 CPU FP16 已验证） |

`--device npu` 时 `precision` 会由 `fp32` 自动记为 **`fp16`**，并在 `note` 里写明
「Intel NPU 只支持 FP16/INT8，特征按 fp16 计算（独立版本，不等价于 fp32）」。

### 4.4 编码器在 NPU、距离在 CPU 的分工（如实记录）

| 指标 | NPU 上跑的是什么 | CPU 上跑的是什么 |
|---|---|---|
| Gram | **VGG-19 多层特征编码器**（`relu1_1…relu5_1`） | Gram 矩阵与距离 |
| AdaIN | 同上（同一份特征） | 通道 μ/σ 与距离 |
| LPIPS | **官方 LPIPS 模块整体**（含 ScalingLayer 与校准 lin 层） | 仅标量收集 |
| CSD | **style 描述子编码器**（ViT-L/14 visual + 官方头，输出 768 维） | 768 维余弦（内积） |

也就是说，**没有任何一个指标是「整条链路都在 NPU」**；JSON 的 `timing.split` 字段逐条写明这个分工。

### 4.5 不假定 NPU 比 GPU 快（实测果然如此）

本机三后端耗时（同一输入、同权重、同预处理，预热后中位数）：

| 指标 | CPU FP32 | CUDA FP32 | NPU FP16 |
|---|---:|---:|---:|
| Gram | 189.2 ms | **74.9 ms** | 136.7 ms |
| AdaIN | 219.1 ms | **97.6 ms** | 176.6 ms |
| LPIPS | 223.7 ms | **6.8 ms** | 11.3 ms |
| CSD | 4541.7 ms | **65.0 ms** | 428.1 ms |

结论：**本机 CUDA 全面快于 NPU**，NPU 只比纯 CPU 快。选后端应据实测，而不是「NPU 一定更快」的假设。

---

## 5. 实测数值与跨设备验证

测试输入：`utils.style_metrics.imaging.make_test_images()`（确定性合成图，seed=20261004，384×384），三个用例：

- `same` —— 基准图 vs 自身的无损 PNG 往返（**同图**，距离应≈0、相似度应≈1）
- `soft` —— 暖色 + 5×5 高斯模糊（**不同图，差异小**）
- `hard` —— 冷色 + 条纹调制 + 位移（**不同图，差异大**）

### 5.1 同图自比（必须≈0 / ≈1）

| 指标 | CPU FP32 | CUDA FP32 | NPU FP16 | 判定 |
|---|---:|---:|---:|---|
| Gram（期望 0） | 0.0 | 0.0 | 0.0 | ✅ |
| AdaIN（期望 0） | 0.0 | 0.0 | 0.0 | ✅ |
| LPIPS（期望 0） | 0.0 | 0.0 | 0.0 | ✅ |
| CSD（期望 1） | 1.0 | 1.0000001192 | 1.0 | ✅（FP32 上略 >1 属浮点项，容差 1e-4） |

> 同图自比在数学上是恒等式，**只能证明实现与数值稳定，不能证明计算正确** —— 所以下面必须看不同图。

### 5.2 不同图片（必须拉开距离 / 降低相似度）

| 指标 | CPU FP32 `soft` | CPU FP32 `hard` | 单调性 |
|---|---:|---:|---|
| Gram | 2.4398528920413813e-06 | 4.7438320669965673e-04 | ✅ hard > soft > 0 |
| AdaIN | 37.186454224794474 | 212.27273658961408 | ✅ |
| LPIPS | 0.04602726921439171 | 0.79811030626297 | ✅ |
| CSD（相似度） | 0.9345495700836182 | 0.730856716632843 | ✅ hard < soft < 1 |

四个指标都同时满足「同图≈极端值」与「不同图明显偏离」，**不是只靠同图测试就宣称正确**。

### 5.3 跨设备误差（先声明容差，后验证）

**容差在跑之前就写死在 `utils/style_metrics/runner.py::TOLERANCES`**，判据 `|a − b| ≤ atol + rtol·|b|`，
`b` 为 CPU FP32 基准。**没有事后放宽。**

| 指标 | 精度 | rtol | atol |
|---|---|---:|---:|
| gram | fp32 / fp16 | 1e-3 / 5e-2 | 1e-6 / 1e-4 |
| adain | fp32 / fp16 | 1e-3 / 5e-2 | 1e-6 / 1e-4 |
| lpips | fp32 / fp16 | 1e-3 / 1e-2 | 1e-5 / 1e-3 |
| csd | fp32 / fp16 | 1e-3 / 1e-2 | 1e-5 / 1e-3 |

**测试图自比上界**：距离 1e-6、相似度 1e-4。

#### CUDA FP32 vs CPU FP32

| 指标 | 最大绝对误差 | 不同图最大相对误差 | 结论 |
|---|---:|---:|---|
| gram | 1.1465e-08 | 2.1001e-04 | ✅ 在容差内 |
| adain | 4.2908e-03 | 1.1539e-04 | ✅ |
| lpips | 2.7478e-05 | 3.4429e-05 | ✅ |
| csd | 1.1921e-07 | 1.2756e-07 | ✅（相似度误差 1.1921e-07） |

#### NPU FP16 vs CPU FP32

| 指标 | 最大绝对误差 | 不同图最大相对误差 | 结论 |
|---|---:|---:|---|
| gram | 4.5540e-08 | 6.6762e-04 | ✅ 在 fp16 容差内 |
| adain | 1.4645e-02 | 3.9383e-04 | ✅ |
| lpips | 2.5874e-04 | 2.7990e-03 | ✅ |
| csd | 9.0915e-04 | 1.2440e-03 | ✅（相似度误差 9.0915e-04） |

> **精度等价性声明**：NPU 上的 fp16 结果**不是** CPU/CUDA fp32 的等价替代。
> 虽然数值都在声明的 fp16 容差内，但两者应作为**独立版本**分别引用；
> 需要严格复现时以 CPU FP32 为准。

#### NPU FP16 的实测数值

| 指标 | same | soft | hard |
|---|---:|---:|---:|
| gram | 0.0 | 2.438223988177705e-06 | 4.7442874682841307e-04 |
| adain | 0.0 | 37.17180922372333 | 212.27879743072995 |
| lpips | 0.0 | 0.0458984375 | 0.7978515625 |
| csd（相似度） | 1.0 | 0.9340081810951233 | 0.729947566986084 |

### 5.4 性能与显存（固定输入 / batch=1 / 线程固定 / 预热后中位数）

**测量协议**：输入固定为上面的 384×384 合成图；batch = 1；`torch.get_num_threads()` 未改动（本机 24）；
NPU 的 `PERFORMANCE_HINT = LATENCY`；每次指标先 warmup 1 次再取 3 次中位数；
CUDA 计时前后都 `torch.cuda.synchronize()`，并在每个指标开始前 `reset_peak_memory_stats()`；
**模型加载/编译耗时与推理耗时分列**（见 `timing.compile_seconds` 与 `timing.median_ms`）。

| 指标 | CPU FP32 中位 | CUDA FP32 中位 | CUDA 峰值显存 | NPU FP16 中位 | NPU 编译耗时（一次性） |
|---|---:|---:|---:|---:|---:|
| Gram | 189.18 ms | 74.85 ms | 344.7 MB | 136.71 ms | 2.65 s |
| AdaIN | 219.14 ms | 97.60 ms | 344.7 MB | 176.63 ms | 2.71 s |
| LPIPS | 223.69 ms | 6.84 ms | 111.1 MB | 11.33 ms | 0.61 s |
| CSD | 4541.69 ms | 65.00 ms | 1280.0 MB | 428.14 ms | 20.9 s |

> 说明：`Gram` / `AdaIN` 的每一档都是**两次前向**（基准图 + 对照图各一次）加距离计算的总和；
> LPIPS / CSD 是单对图的端到端。CSD 的 ONNX 体积 1.22 GB，NPU 首次编译 20.9 s，属一次性成本；
> VGG-19 的 ONNX 是 51.8 MB、LPIPS 的 ONNX 是 9.9 MB。
> NPU 的三行数字里，编码器在 NPU、距离/相似度在 CPU（见 §4.4）。

### 5.5 一次实测的完整证据链（NPU）

NPU 的「实际设备已确认」由运行期属性给出，写进了 JSON 的 `detail.npu_encoder.property_evidence`
（四个指标各自一份，下面是 VGG-19 编码器那一份）：

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

四个指标在 NPU 上都跑通了，`EXECUTION_DEVICES` 均指向 `NPU`：

| 指标 | 导出 ONNX 体积 | NPU 编译耗时 | 输出 | 该次运行的 `execution_devices` |
|---|---:|---:|---|---|
| Gram | 51,785,433 B | 2.65 s | 5 层特征，全部有限 | `["NPU"]` |
| AdaIN | 51,785,433 B（同前一份） | 2.71 s | 同上 | `["NPU"]` |
| LPIPS | 9,901,142 B | 0.61 s | 1 个标量 + 5 个逐层标量 | `["NPU"]` |
| CSD | 1,216,354,242 B | 20.9 s | 768 维描述子，`‖d‖₂ ≈ 1` | `["NPU"]` |

`compiled:EXECUTION_DEVICES = NPU` 说明**编译产物确实被指派给 NPU 执行**，
配合「真实 infer 出有限数值」才算通过（探针同时要求 `probe_output_shape` 与有限性）。

> 踩过的坑：`compiled.get_property("EXECUTION_DEVICES")` 在 OpenVINO 里可能返回裸字符串 `"NPU"`，
> `list("NPU")` 会得到 `['N','P','U']` 这种假证据。已用 `devices._as_list` 修正并有回归用例
> `test_as_list_does_not_split_strings`。

---

## 6. 复现命令（一条命令即可）

```powershell
# ① 权重与依赖状态（含 SHA-256 校验），不跑前向；缺权重会退出码 1 并给修复命令
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --inventory

# ①b 权重缺失时让 CLI 自己按官方来源下载（默认不联网下载）
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --device auto --download-missing

# ② 全套四指标验证（自动挑已验证可用的后端），并写 data/test-result/*.json
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --device auto

# ③ 指定后端 / 精度
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --device cuda --precision fp32
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --device cpu
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --device npu

# ④ 只跑部分指标 / 用真实图片对
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --device cuda --metrics gram,adain
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --device cuda --pair a.png b.png

# ⑤ 回归用例（快测；慢测加环境变量 STYLE_METRICS_SLOW=1）
& "C:\Program Files\Python310\python.exe" -m pytest -q -p no:cacheprovider tests/test_style_metrics.py
```

> 本项目的纪律：跑 python 命令**不套管道 / 不重定向**（沙箱会拒绝 spawn），日志由 Python 自己写文件。

### 6.1 代码落位（遵守 `PROJECT_REQUIREMENTS.md` 的目录边界）

| 文件 | 职责 |
|---|---|
| `utils/style_metrics/config.py` | 路径、权重清单、**全部固定预处理与输出层契约** |
| `utils/style_metrics/imaging.py` | 图像加载与三种预处理（VGG / LPIPS / CSD），确定性测试图生成 |
| `utils/style_metrics/vgg_encoder.py` | torchvision VGG-19 多层特征编码器 |
| `utils/style_metrics/gram.py` | Gram 矩阵与 Gatys 距离 |
| `utils/style_metrics/adain.py` | AdaIN 通道 μ/σ 距离 |
| `utils/style_metrics/lpips_metric.py` | 官方 LPIPS 封装 + NPU 变体 |
| `utils/style_metrics/csd_metric.py` | 官方 CSD 复刻 + 权重加载 + NPU 变体 |
| `utils/style_metrics/devices.py` | `auto/cuda/cpu/npu` 解析、运行期探针、硬件报告 |
| `utils/style_metrics/openvino_backend.py` | ONNX 导出 + NPU 编译/推理/证据 |
| `utils/style_metrics/runner.py` | 统一运行器、**容差表**、计时 |
| `utils/style_metrics/inventory.py` | 权重 SHA-256 清单与校验 |
| `utils/style_metrics/fetch.py` | 权重下载（requests） |
| `tools/style_metrics_verify.py` | CLI 入口（薄壳） |
| `tests/test_style_metrics.py` | 回归用例（口径/容差/设备解析/CLI 契约；慢测要 `STYLE_METRICS_SLOW=1`） |
| `tests/test_style_metrics_missing_models.py` | **权重缺失容错用例**（17 条：错误对象、预检、指标降级、CLI 端到端） |

**根目录未新增任何脚本**（有回归用例 `test_cli_does_not_add_root_scripts` 兜底）。

---

## 7. 机器可读 JSON

输出到 `data/test-result/`，本次三份：

| 文件 | 内容 |
|---|---|
| `style-metrics-cpu-baseline.json` | CPU FP32 全四指标 |
| `style-metrics-cuda-fp32.json` | CUDA FP32 + 内嵌 CPU 基准对比（`--verify-weights`） |
| `style-metrics-npu.json` | NPU FP16 + 内嵌 CPU 基准对比 |

字段结构（`schema = "style-metrics-verification/1"`）：

```jsonc
{
  "schema": "style-metrics-verification/1",
  "generated_at": "...",
  "requested_device": "cuda",
  "tolerance_policy": { "rule": "|actual - cpu_baseline| <= atol + rtol*|cpu_baseline|",
                        "declared_before_running": true, "table": { ... }, "self_check": { ... } },
  "weights":      { "<key>": { "path", "bytes", "expected_sha256", "sha256", "sha256_matches",
                               "source_url", "license", "origin" } },
  "hardware":     { "cpu", "cuda", "openvino", "npu", "nvidia_smi" },
  "backend":      { "requested", "actual", "backend", "device_name", "precision",
                    "fallback", "note", "evidence" },
  "results": [
    {
      "metric": "gram",                     // gram | adain | lpips | csd
      "kind": "distance",                   // distance | similarity
      "status": "ok",                       // ok | partial | error | failed_self_check | unavailable | not_run
      "model": { "name", "weights_file", "weights_origin", "source_url", "license", ... },
      "weights_sha256": { ... },
      "preprocessing": { "resize", "color", "normalize", "output_layers", ... },
      "requested_device": "cuda",
      "actual_device": "cuda",
      "backend": "torch-cuda",
      "precision": "fp32",
      "validation": {
        "self_distance_or_similarity": { "value", "expected", "tolerance", "passed" },
        "different_images": { "soft", "hard", "monotonic_separation_passed" },
        "finite": true,
        "cross_device": { "baseline_backend", "tolerance", "per_case": { "same|soft|hard": {
                            "actual", "baseline", "abs_error", "rel_error",
                            "tolerance_limit", "within_tolerance" } },
                          "max_abs_error", "max_rel_error_on_different_images",
                          "within_tolerance", "cases_compared" }
      },
      "timing": { "warmup_runs", "repeat", "median_ms", "min_ms", "max_ms", "samples_ms",
                  "split", "peak_gpu_memory_mb", "compile_seconds" },
      "detail": { "per_layer": ..., "npu_encoder": { "execution_devices", "property_evidence", ... } },
      "remediation": [ { "reason": "weights_missing", "key", "expected_path", "source_url",
                         "expected_bytes", "expected_sha256", "remedy" } ],
      "error": null
    }
  ],
  "summary": { "metrics": [...], "status_counts": { "ok": 4 }, "all_ok": true },
  "weights_preflight": { "<metric>": [ { "key", "expected_path", "source_url", "remedy" } ] },
  "notes": []
}
```

`status` 取值与含义（**不要混用**）：

| status | 含义 |
|---|---|
| `ok` | 跑完且自比检查通过 |
| `partial` | 跑完但有非致命问题（如某后端未实现该指标） |
| `error` | 跑了但失败（输入错误 / 数值异常） |
| `unavailable` | **没跑**：设备不可用或权重缺失；`remediation` 里给修复办法，**绝不产生假数值** |
| `failed_self_check` | 跑完但同图自比超出容差 |

---

## 8. 对现有 App 的影响

- **没有修改任何 App 业务代码**：`app.py`、`modules/**`、`utils/**` 的既有文件一个都没动，
  新代码全部落在新目录 `utils/style_metrics/`。
- **没有动评分公式**：`docs/style-extraction/scoring-research-20261004.md` 描述的 12 维量表与
  `0.85×S + 0.15×主体符合度` 保持原样；本环境产出的四个指标**不参与**该总分。
- **没有调用付费生图 / 文本 API**：全流程本地计算，测试图是合成图，未启动任何生图任务。
- 原有关键模块导入自检（本机实测）：`utils.styles` / `utils.analysis_gen` / `utils.analysis_gpt_prompt` /
  `utils.gpt_image_optimize` / `utils.post_process` / `utils.cost_estimate` / `utils.style_gpt` /
  `utils.output_isolation` / `utils.prompt_loader` / `modules.others.api_backend` /
  `modules.image_analysis.analysis_pipeline` / `modules.image_analysis.single_analyzer` /
  `modules.image_generation.gpt_image2_tab` —— **全部 `ok`**。
- **是否需要重启 App？** —— **不需要**。本次没有修改 App 运行的任何模块，也没有改配置；
  新增的 `utils/style_metrics/` 是独立子包，App 不导入它。
  （仅在「想在自己的代码里调用这些指标」时才需要重新导入模块；这不属于重启 App。）

> 一次观察记录（**与本次改动无关，事前就存在**）：
> 跑 `python -m pytest tests/test_analysis_cli_unittest.py` 时，faulthandler 会打印一条
> `Windows fatal exception: access violation`，栈顶是 `utils/gui_entry.py:26 → import onnxruntime`。
> 该用例本身 **1 passed**，`app.py` 里的导入被 `warm_up_optional_module` 的 `try/except` 吞掉，
> 全量 `pytest` 也以 **exit code 0** 跑完 100%。这与 §2.2 记的 ONNX Runtime 发行包冲突（DirectML 构建覆盖了普通构建）
> 相互印证，属**环境既有现象**；本次未触碰 `utils/gui_entry.py` / `app.py` / ORT，故按纪律只记录、不修。
>
> 另一次观察：在 CPU 全量验证进程同时加载 3 GB 权重的瞬间，另一个进程的一次 `import torch`
> 曾报 `OSError: [WinError 1114] … c10.dll`；该进程结束后单独复测 `import torch` 与 `import utils.style_metrics.runner`
> 均正常。判定为并发加载下的**一次性资源争用**，非代码缺陷；未做任何环境改动。

### 8.1 全量 pytest 的一个 Windows 加载顺序坑（已修）

加上本模块的用例后，`python -m pytest -q` 一度在**收集阶段**就失败：

```
OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败。
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

定位（四条都实测过，结论明确）：

| 实验 | 结果 |
|---|---|
| 单独 `import onnxruntime` / `cv2` / `openvino` / `numpy` / `PIL` 后再 `import torch` | **全部正常** → 不是两两 DLL 冲突 |
| 只收集 2 个模块（`test_analysis_cli_unittest.py` + `test_style_metrics.py`） | **正常** |
| 收集**整套** `tests/`（PyQt6 + onnxruntime-directml + opencv + openvino 都进过进程）后再加载 torch | **必炸** |
| 先 `import torch` 再让 pytest 收集同一套用例 | **856 项全部收集成功、0 错误** |

即：这台 Windows 上 torch 的 DLL 需要一个尚未被其它大型原生库占满的加载环境，晚了就 `DllMain` 失败。
**修法**：在 `tests/conftest.py` 里加「守卫三」，会话最开始就把 torch 顶上来（实测量价约 1.5 秒/会话，
可用 `IMAGE_MAKER_TESTS_SKIP_TORCH_PRELOAD=1` 跳过）。修后全量结果为
**851 passed / 5 skipped / 0 failed（77.5 s）**。

这条坑与画风指标本身无关，但**任何后续新增的 torch 用例都会踩到**，所以记在这里。

---

## 9. 已知限制与未完成项（不粉饰）

1. **CSD 未做论文级复现**：上游权重自带 DISCLAIMER，本环境只验证了「官方描述子能被复现且同图余弦=1」，
   没有在 WikiArt 上复算 Recall/MRR/mAP。
2. **NPU 是 FP16 独立版本**，与 CPU/CUDA FP32 不等价；跨设备误差虽在声明的 fp16 容差内，
   但不得把两者当同一口径引用。
3. **没有做内容解耦验证**：四个指标都受内容影响（尤其 LPIPS / CSD 对主体变化敏感），
   本文件没有断言它们等于「纯画风相似度」。
4. **测试图是合成图**：`soft` / `hard` 两个变体只证明「能区分不同图」，
   不等于在真实动漫插画上的判别力已经校准。
5. **未做 INT8 量化**：NPU 支持 INT8，但本项目只用 FP16，没有量化版本，也不声称量化等价。
6. **未测批量吞吐**：所有计时都是 batch = 1。多图批量的显存与吞吐未测。
7. **ONNX Runtime 的发行包冲突未修**（按纪律只记录）：当前 `onnxruntime` 可导入的是 DirectML 构建；
   本项目 NPU 路径不依赖 ORT，故未受影响。
8. **未更新任何驱动 / BIOS，未重启系统**（按要求）。
