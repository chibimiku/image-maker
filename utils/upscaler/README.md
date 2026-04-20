# upscaler

`upscaler/` 是从 Extras 放大流程抽离出的独立模块，设计目标是脱离 WebUI 也可运行。

## 目录结构

- `upscaler/core.py`: 放大参数解析、尺寸限制、裁切、缓存 key 生成。
- `upscaler/cache.py`: 图片缓存（LRU 风格）。
- `upscaler/types.py`: 对外类型定义（请求、provider 协议、upscaler handle）。
- `upscaler/extras_pipeline.py`: 主流程（主 upscaler、次 upscaler 混合、信息回填）。
- `upscaler/webui_provider.py`: WebUI 适配层（可选，不是独立运行必需）。

## 独立运行原则

- `upscaler/` 本身不依赖 WebUI 的 UI 流程，不要求修改 `scripts/`。
- 核心模块不强绑定模型路径，模型放置与加载由你实现的 `UpscalerProvider` 决定。
- 只要你能提供 `name -> upscale(image, scale)` 映射，就可以使用完整放大流程。

## 模型放置与配置（独立项目）

- 推荐目录：`<你的项目根目录>/models/ESRGAN`。
- 推荐后缀：`.pt`、`.pth`、`.safetensors`（与原项目加载约定一致）。
- 推荐做法：在你自己的 `Provider` 里持有 `model_dir` 配置，启动时扫描目录并注册可用模型名称。
- 核心点：`upscaler/` 不规定“必须放在哪”，路径是业务层配置项，而不是框架硬编码。

示例（仅示意目录配置，不含真实模型推理）：

```python
from pathlib import Path
from PIL import Image
from upscaler.extras_pipeline import ExtrasUpscalePipeline
from upscaler.types import UpscalerHandle

class MyProvider:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)

    def resolve_primary(self, name: str | None):
        if name is None:
            return None
        model_path = self.model_dir / f"{name}.pth"
        if not model_path.exists():
            raise AssertionError(f"could not find upscaler model: {model_path}")

        # 这里替换成你的真实模型推理函数
        return UpscalerHandle(
            name=name,
            upscale=lambda image, scale: image.resize(
                (int(image.width * scale), int(image.height * scale))
            ),
        )

    def resolve_secondary(self, name: str | None):
        return None

pipeline = ExtrasUpscalePipeline(
    provider=MyProvider(model_dir="./models/ESRGAN"),
    cache_size_getter=lambda: 8,
)
```

## 调用示例（与 WebUI 无关）

```python
from PIL import Image
from upscaler.extras_pipeline import ExtrasUpscalePipeline

# provider 见上面的 MyProvider
pipeline = ExtrasUpscalePipeline(
    provider=MyProvider(model_dir="./models/ESRGAN"),
    cache_size_getter=lambda: 8,
)

img = Image.open("input.png").convert("RGB")
info = {}

out = pipeline.upscale_image(
    image=img,
    info=info,
    upscale_mode=0,               # 0: 按倍率, 1: 按目标尺寸
    upscale_by=2.0,
    max_side_length=0,
    upscale_to_width=1024,
    upscale_to_height=1024,
    upscale_crop=False,
    upscaler_1_name="MyModelX4",
    upscaler_2_name=None,
    upscaler_2_visibility=0.0,
)

out.save("output.png")
print(info)
```

## 参数行为说明

- `upscale_mode=0`: 按 `upscale_by` 放大，可被 `max_side_length` 限制自动切换为目标尺寸模式。
- `upscale_mode=1`: 按 `upscale_to_width/upscale_to_height` 放大，倍率自动计算。
- `upscaler_2_visibility > 0`: 且存在第二 upscaler 时，执行双模型混合。
- `pipeline.clear_cache()`: 清理缓存，建议在输入图变化后调用。

## Python 环境要求

- Python: 建议 `3.10+`（与当前仓库主环境保持一致更稳妥）。
- 必需依赖: `Pillow`、`numpy`。
- 可选依赖: 你的模型推理后端（如 `torch`、`onnxruntime`、`ncnn` 等），由你的 `Provider` 决定。

最小安装示例：

```bash
pip install pillow numpy
```

## CPU/GPU 说明

- `upscaler/` 核心流程本身不强依赖 GPU，纯 CPU 可以运行。
- 是否使用 GPU 取决于你在 `UpscalerProvider` 里接入的推理实现。
- 如果你的 `Provider` 用的是 CPU 推理后端，那就是纯 CPU 跑。
- 如果你的 `Provider` 用的是 CUDA/DirectML 后端（如 `torch` CUDA），才会走 GPU。
- 结论：你理解得对，当前这套封装默认可以按 CPU 独立运行，不需要强制 GPU。

## 生产建议

- **模型预热**: 服务启动时先跑一次小图，避免首请求冷启动抖动。
- **模型常驻**: 在 `Provider` 层缓存模型对象，避免每次请求重复加载权重。
- **并发控制**: 为单模型实例加锁或队列，避免并发导致显存/内存峰值失控。
- **缓存策略**: 根据业务流量设置 `cache_size_getter`，并在图像切换时调用 `clear_cache()`。
- **超大图保护**: 配置 `max_side_length`，并限制输入分辨率，防止 OOM。
- **失败回退**: 模型加载失败时提供降级路径（如 Lanczos/双三次）和明确错误日志。
- **可观测性**: 记录每次放大耗时、输入输出尺寸、模型名、失败原因，便于压测与排障。
- **版本固定**: 锁定推理框架版本（如 `torch`）并保留 `requirements.txt`，避免部署漂移。

## WebUI 适配（可选）

如果你在本仓库内使用 `WebUIUpscalerProvider`（`upscaler/webui_provider.py`）：

- 默认 ESRGAN 目录是 `models/ESRGAN`。
- 可通过启动参数修改：`--esrgan-models-path "<你的目录>"`。
- 这是适配层行为，不影响 `upscaler/` 独立运行能力。
