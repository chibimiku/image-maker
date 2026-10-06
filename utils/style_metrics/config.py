"""画风指标的路径、权重清单与固定预处理契约。

**任何**预处理参数、输入尺寸、输出层或精度都写在这里，不允许散落到各指标实现。
改动本文件等于改动指标口径，必须同步更新
``docs/style-extraction/style-metrics-setup.md`` 与 ``models/style-metrics/README.md``。
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: 权重根目录。默认在仓库内；可用环境变量指到别处（例如权重放在另一块盘），
#: 这样既不写死机器路径，也让「权重缺失」的行为可以端到端测试。
MODEL_ROOT = Path(
    os.environ.get("IMAGE_MAKER_STYLE_METRICS_HOME") or (PROJECT_ROOT / "models" / "style-metrics")
)
DOWNLOAD_CACHE = Path(
    os.environ.get("IMAGE_MAKER_STYLE_METRICS_CACHE")
    or (PROJECT_ROOT / "cache" / "style-metrics-download")
)
ONNX_CACHE = MODEL_ROOT / "onnx"
RESULT_DIR = PROJECT_ROOT / "data" / "test-result"


def torch_home() -> Path:
    """torchvision / torch.hub 的下载根目录（留在工作区，不污染用户主目录）。"""
    return DOWNLOAD_CACHE / "torch"


# --------------------------------------------------------------------------- #
# 权重文件与官方来源
# --------------------------------------------------------------------------- #

WEIGHTS: dict[str, dict] = {
    "vgg19": {
        "path": MODEL_ROOT / "vgg19" / "vgg19-dcbb9e9d.pth",
        "sha256": "dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44",
        "bytes": 574673361,
        "source": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
        "license": "torchvision BSD-3-Clause（VGG-19 结构与 ImageNet 预训练权重）",
        "origin": "torchvision VGG19_Weights.IMAGENET1K_V1",
    },
    "lpips_alex": {
        "path": MODEL_ROOT / "lpips" / "v0.1" / "alex.pth",
        "sha256": "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
        "bytes": 6009,
        "source": (
            "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
            "master/lpips/weights/v0.1/alex.pth"
        ),
        "license": "LPIPS BSD-2-Clause（作者官方校准权重 v0.1）",
        "origin": "richzhang/PerceptualSimilarity lpips/weights/v0.1/alex.pth",
    },
    "lpips_vgg": {
        "path": MODEL_ROOT / "lpips" / "v0.1" / "vgg.pth",
        "sha256": "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868",
        "bytes": 7289,
        "source": (
            "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
            "master/lpips/weights/v0.1/vgg.pth"
        ),
        "license": "LPIPS BSD-2-Clause（作者官方校准权重 v0.1）",
        "origin": "richzhang/PerceptualSimilarity lpips/weights/v0.1/vgg.pth",
    },
    "lpips_alex_trunk": {
        "path": MODEL_ROOT / "lpips" / "trunk" / "alexnet-owt-7be5be79.pth",
        "sha256": "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
        "bytes": 244408911,
        "source": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        "license": "torchvision BSD-3-Clause（AlexNet ImageNet 骨架）",
        "origin": "torchvision AlexNet_Weights.IMAGENET1K_V1",
    },
    "csd": {
        "path": MODEL_ROOT / "csd" / "CSD-ViT-L-pytorch_model.bin",
        "sha256": "40e92fad63a361b8136100cd234c42d401ef9b34ff1748234318929ebcc7e7a1",
        "bytes": 2438228893,
        "source": (
            "https://huggingface.co/tomg-group-umd/CSD-ViT-L/resolve/main/pytorch_model.bin"
        ),
        "license": "HF 模型卡 CC-BY-4.0（仓库代码 MIT，上游口径冲突，两者并列）",
        "origin": "learn2phoenix/CSD 官方 ViT-L 训练权重（HF tomg-group-umd/CSD-ViT-L）",
    },
    "clip_vit_l14": {
        "path": DOWNLOAD_CACHE / "clip" / "ViT-L-14.pt",
        "sha256": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
        "bytes": 932768134,
        "source": (
            "https://openaipublic.azureedge.net/clip/models/"
            "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
        ),
        "license": "OpenAI CLIP MIT（ViT-L/14 骨架；URL 内嵌 sha256 即官方校验值）",
        "origin": "CSD backbone：OpenAI clip.load(\"ViT-L/14\") 的视觉塔",
    },
}

# --------------------------------------------------------------------------- #
# 固定预处理契约
# --------------------------------------------------------------------------- #

#: VGG-19 特征（Gram / AdaIN）统一输入尺寸；两图必须同尺寸才能比较 Gram。
VGG_INPUT_SIZE = 512
VGG_RESIZE_MODE = "bicubic"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: torchvision VGG19.features 的下标 → relu 层名（与 Gatys / AdaIN 论文一致）。
VGG19_LAYERS = {
    "relu1_1": 1,
    "relu2_1": 6,
    "relu3_1": 11,
    "relu4_1": 20,
    "relu5_1": 29,
}
#: Gram 距离默认参与层（沿用 Gatys 的 conv1_1..conv5_1 五层）。
VGG19_GRAM_LAYERS = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
# Scalar formula versions; archived v1 results must never be mixed with v2.
GRAM_FORMULA_VERSION = "gatys-layer-sum/v2"
GRAM_LEGACY_VERSION = "normalized-gram-extra-channel/v1"
#: AdaIN 距离默认参与层（AdaIN 论文使用 relu1_1..relu4_1 的统计量）。
VGG19_ADAIN_LAYERS = ("relu1_1", "relu2_1", "relu3_1", "relu4_1")

#: LPIPS 官方输入契约：RGB，[-1, 1] 归一化，默认 256x256。
LPIPS_INPUT_SIZE = 256
LPIPS_VERSION = "0.1"
LPIPS_NET = "alex"

#: CSD 输入契约（官方代码默认，见 csd_metric.py 顶部注释）。
CSD_INPUT_SIZE = 224
CSD_NORMALIZE = "imagenet"


def layer_channels(vgg_name: str) -> int:
    """VGG-19 各层输出通道数（用于 Gram 归一化分母）。"""
    return {
        "relu1_1": 64,
        "relu2_1": 128,
        "relu3_1": 256,
        "relu4_1": 512,
        "relu5_1": 512,
    }[vgg_name]


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")
