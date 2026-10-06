# 画风深度特征指标 —— 模型权重目录

> 本目录由 `utils/style_metrics/` 与 `tools/style_metrics_verify.py` 使用。
> **权重不入库**（`.gitignore` 已忽略 `models/**/*.pth`、`*.pt`、`*.bin`、`*.onnx`）。
> 重建方式见 `docs/style-extraction/style-metrics-setup.md`。

## 目录结构

```
models/style-metrics/
├── vgg19/
│   └── vgg19-dcbb9e9d.pth            # torchvision VGG19 IMAGENET1K_V1（Gram / AdaIN 共用）
├── lpips/
│   ├── v0.1/alex.pth                 # LPIPS 作者官方校准权重（v0.1 = 最新，alex 骨干）
│   ├── v0.1/vgg.pth                  # 同上，vgg 骨干（备用，默认不用）
│   └── trunk/alexnet-owt-7be5be79.pth # LPIPS 的 AlexNet 骨架（ImageNet IMAGENET1K_V1）
├── csd/
│   └── CSD-ViT-L-pytorch_model.bin   # CSD 官方权重（HF tomg-group-umd/CSD-ViT-L）
└── onnx/                             # 派生文件（ONNX 导出与 NPU 编译缓存，可安全删除重建）
```

## 权重清单、来源与校验

| 文件 | 字节数 | SHA-256 | 官方来源 | 许可 |
|---|---|---|---|---|
| `vgg19/vgg19-dcbb9e9d.pth` | 574673361 | `dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44` | https://download.pytorch.org/models/vgg19-dcbb9e9d.pth | torchvision BSD-3-Clause |
| `lpips/v0.1/alex.pth` | 6009 | `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0` | https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/alex.pth | LPIPS BSD-2-Clause |
| `lpips/v0.1/vgg.pth` | 7289 | `a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868` | https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth | LPIPS BSD-2-Clause |
| `lpips/trunk/alexnet-owt-7be5be79.pth` | 244408911 | `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` | https://download.pytorch.org/models/alexnet-owt-7be5be79.pth | torchvision BSD-3-Clause |
| `csd/CSD-ViT-L-pytorch_model.bin` | 2438228893 | `40e92fad63a361b8136100cd234c42d401ef9b34ff1748234318929ebcc7e7a1` | https://huggingface.co/tomg-group-umd/CSD-ViT-L/resolve/main/pytorch_model.bin | 模型卡 CC-BY-4.0（代码 MIT，上游口径冲突） |

派生/缓存（可重建，不入库）：

| 文件 | 位置 | 来源 |
|---|---|---|
| OpenAI CLIP ViT-L/14 骨架 | `cache/style-metrics-download/clip/ViT-L-14.pt` | https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt （932768134 B，SHA-256 与 URL 内嵌值逐位一致） |
| VGG-19 特征编码器 ONNX | `models/style-metrics/onnx/vgg19_features_512.onnx` | 由本仓库 `openvino_backend.export_vgg19_onnx()` 从上面的 VGG-19 权重导出 |

校验命令：

```powershell
& "C:\Program Files\Python310\python.exe" tools\style_metrics_verify.py --inventory
```

`--inventory` 会对需要校验的文件重算 SHA-256 并与上表比对；不符会直接报错，不会「大概是这份」继续跑。

## 三个必须注意的口径问题

1. **LPIPS 的校准权重很小（6 KB）**，真正的大头是 AlexNet 骨架（244 MB）。两者缺一不可；
   只有骨架没有校准权重 = 未校准特征距离，**不算 LPIPS**。
2. **CSD 权重里没有 `backbone.proj`**，`last_layer_style` 是裸 `(1024, 768)` 张量。
   按仓库 `model.py` 建 `nn.Linear` 再 `strict=False` 加载会静默跳过、退化成普通 CLIP。
   本仓库实测确认 OpenAI CLIP 的 `visual.proj` 本身就是 `nn.Parameter(1024, 768)`，
   因此按 `nn.Parameter` 建并用 `strict=True` 加载，0 missing / 0 unexpected。
3. **上游 CSD 权重有已声明的已知问题**：官方 README 顶部 DISCLAIMER 仍在，
   `learn2phoenix/CSD` issue #14 至今 open —— 上传权重与论文报告数字有出入。
   本目录只保证「拿到官方权重、按官方预处理复现描述子」，**不声称复现论文指标**。

## 2026-10-06 Gram 版本

默认 `gatys-layer-sum/v2`：归一化 Gram `FFᵀ/(C·H·W)` 的平方差只除以 4，五层等权。旧 `normalized-gram-extra-channel/v1` 仅供显式历史复算，数值不能混用。此次不改变权重、输入预处理或特征编码器，无须仅为标量公式修正重新导出 ONNX。App 深度比较使用独立 v2 缓存文件；旧缓存和报告保留。新版实测与局限见 `docs/style-extraction/STYLE-METRICS-GRAM-V2-20261006.md`。
