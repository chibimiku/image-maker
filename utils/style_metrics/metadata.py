"""Shared model and preprocessing provenance."""
from . import csd_metric as csd_mod
from utils.style_metrics.config import (  # noqa: E402
    CSD_INPUT_SIZE,
    GRAM_FORMULA_VERSION,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LPIPS_INPUT_SIZE,
    LPIPS_VERSION,
    RESULT_DIR,
    VGG19_ADAIN_LAYERS,
    VGG19_GRAM_LAYERS,
    VGG_INPUT_SIZE,
    WEIGHTS,
)

def preprocessing_for(metric: str) -> dict:
    if metric in ("gram", "adain"):
        return {
            "resize": f"直接缩放到 {VGG_INPUT_SIZE}x{VGG_INPUT_SIZE}（bicubic, 不保持长宽比）",
            "color": "PIL convert('RGB')，丢弃 alpha",
            "to_tensor": "[0,1] float32",
            "normalize": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD), "scheme": "ImageNet"},
            "output_layers": list(VGG19_GRAM_LAYERS if metric == "gram" else VGG19_ADAIN_LAYERS),
        }
    if metric == "lpips":
        return {
            "resize": f"直接缩放到 {LPIPS_INPUT_SIZE}x{LPIPS_INPUT_SIZE}（bicubic, 不保持长宽比）",
            "color": "PIL convert('RGB')",
            "range": "[-1, 1]（tensor*2-1，官方 normalize=False 分支）",
            "normalize": "由官方 ScalingLayer 内部完成（shift/scale 缓冲）",
            "output_layers": ["relu1", "relu2", "relu3", "relu4", "relu5"],
        }
    if metric == "csd":
        return {
            "resize": f"Resize({CSD_INPUT_SIZE}, BICUBIC) 短边 + CenterCrop({CSD_INPUT_SIZE})",
            "color": "PIL convert('RGB')",
            "to_tensor": "[0,1] float32",
            "normalize": {"mean": list(csd_mod.CSD_MEAN), "std": list(csd_mod.CSD_STD), "scheme": "CLIP"},
            "output_layers": ["style head（last_layer_style 之后 L2 归一化）"],
            "descriptor_dim": 768,
        }
    raise KeyError(metric)


def model_for(metric: str) -> dict:
    if metric in ("gram", "adain"):
        return {
            "name": "torchvision VGG-19 (features)",
            "weights_file": str(WEIGHTS["vgg19"]["path"]),
            "weights_origin": WEIGHTS["vgg19"]["origin"],
            "source_url": WEIGHTS["vgg19"]["source"],
            "license": WEIGHTS["vgg19"]["license"],
        }
    if metric == "lpips":
        return {
            "name": "LPIPS-AlexNet (richzhang/PerceptualSimilarity)",
            "weights_file": str(WEIGHTS["lpips_alex"]["path"]),
            "weights_origin": WEIGHTS["lpips_alex"]["origin"],
            "source_url": WEIGHTS["lpips_alex"]["source"],
            "license": WEIGHTS["lpips_alex"]["license"],
            "trunk_file": str(WEIGHTS["lpips_alex_trunk"]["path"]),
            "trunk_origin": WEIGHTS["lpips_alex_trunk"]["origin"],
            "version": LPIPS_VERSION,
        }
    return {
        "name": "CSD ViT-L (learn2phoenix/CSD)",
        "weights_file": str(csd_mod.DEFAULT_CSD_WEIGHTS),
        "weights_origin": "Hugging Face tomg-group-umd/CSD-ViT-L",
        "source_url": "https://huggingface.co/tomg-group-umd/CSD-ViT-L",
        "license": "CC-BY-4.0（模型卡）；GitHub 仓库代码见 LICENSE",
        "backbone": "OpenAI CLIP ViT-L/14（clip.load 官方实现）",
    }


