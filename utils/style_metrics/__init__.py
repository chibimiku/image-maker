"""画风深度特征指标（style metrics）公共入口。

四种指标：
- ``gram``  : Gatys 多层 VGG-19 Gram 矩阵风格距离（距离，越小越像）
- ``adain`` : AdaIN 风格统计距离（逐层通道均值/标准差，距离，越小越像）
- ``lpips`` : LPIPS AlexNet v0.1 校准感知距离（距离，越小越像）
- ``csd``   : CSD 对比式画风描述子余弦相似度（相似度，越大越像）

指标之间的数值尺度不可通用，禁止把两者合成百分比。详见
``docs/style-extraction/style-metrics-setup.md``。

本包只做**评价**，不参与生图，不修改 App 评分公式。

代码落位（遵守 PROJECT_REQUIREMENTS 的目录边界）：

- ``config.py``            路径、权重清单、全部固定预处理与输出层契约
- ``imaging.py``           图像加载 / 三种预处理 / 确定性测试图
- ``gram.py`` / ``adain.py``  Gram 矩阵与 Gatys 距离 / AdaIN 通道统计距离
- ``vgg_encoder.py``       torchvision VGG-19 多层特征编码器
- ``lpips_metric.py``      官方 LPIPS 封装 + NPU 变体
- ``csd_metric.py``        官方 CSD 复刻 + 权重加载 + NPU 变体
- ``devices.py``           auto/cuda/cpu/npu 解析、运行期探针、硬件报告
- ``openvino_backend.py``  ONNX 导出 + Intel NPU 编译/推理/执行证据
- ``runner.py``            统一运行器、容差表、计时
- ``inventory.py``         权重 SHA-256 清单与校验
- ``fetch.py``             权重下载（requests）
"""

from __future__ import annotations

__all__ = [
    "METRIC_NAMES",
    "DEVICE_CHOICES",
    "SimilarityResult",
    "compare_images",
    "list_inventory",
]

METRIC_NAMES = ("gram", "adain", "lpips", "csd")
DISTANCE_METRICS = ("gram", "adain", "lpips")
SIMILARITY_METRICS = ("csd",)

#: 允许的输入设备；“npu” 指厂商运行时后端（当前机器为 Intel NPU / OpenVINO）。
DEVICE_CHOICES = ("auto", "cuda", "cpu", "npu")


def __getattr__(name):  # 惰性导入，避免仅取常量时拉起 torch
    if name in ("SimilarityResult", "compare_images"):
        from . import runner

        return getattr(runner, name)
    if name == "list_inventory":
        from . import inventory

        return inventory.list_inventory
    raise AttributeError(name)
