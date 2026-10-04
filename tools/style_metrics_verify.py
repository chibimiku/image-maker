#!/usr/bin/env python
"""画风深度特征指标验证 CLI（Gram / AdaIN / LPIPS / CSD）。

用法示例::

    # 全套验证 + 跨设备对比（自动挑选已验证可用的后端）
    python tools/style_metrics_verify.py --device auto

    # 指定后端
    python tools/style_metrics_verify.py --device cuda --precision fp32
    python tools/style_metrics_verify.py --device cpu
    python tools/style_metrics_verify.py --device npu

    # 只看权重清单（含 SHA-256 校验），不跑前向
    python tools/style_metrics_verify.py --inventory

    # 用真实图片对做验证（可重复传 --pair A.png B.png）
    python tools/style_metrics_verify.py --device cuda --pair a.png b.png

输出：终端摘要 + ``data/test-result/style-metrics-<时间戳>.json``（机器可读）。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from utils.style_metrics import devices, imaging, inventory, runner  # noqa: E402
from utils.style_metrics import csd_metric as csd_mod  # noqa: E402
from utils.style_metrics.config import (  # noqa: E402
    CSD_INPUT_SIZE,
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

SCHEMA = "style-metrics-verification/1"


# --------------------------------------------------------------------------- #
# 预处理 / 模型描述（写进 JSON，保证口径可追溯）
# --------------------------------------------------------------------------- #


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


def weights_sha256_for(metric: str) -> dict:
    if metric in ("gram", "adain"):
        return {"vgg19": WEIGHTS["vgg19"]["sha256"]}
    if metric == "lpips":
        return {
            "lpips_alex_calibrated": WEIGHTS["lpips_alex"]["sha256"],
            "alexnet_trunk": WEIGHTS["lpips_alex_trunk"]["sha256"],
        }
    # CSD 权重 2.44 GB，哈希已在下载时逐字节算出并与 HF LFS 元数据核对一致；
    # 这里引用常量，避免每次跑验证都重算一遍 2.4 GB。
    return {
        "csd_weights": csd_mod.CSD_WEIGHTS_SHA256,
        "csd_weights_bytes": csd_mod.CSD_WEIGHTS_BYTES,
        "csd_weights_sha256_verified_at_download": True,
        "openai_clip_vit_l14": csd_mod.CLIP_VIT_L14_SHA256,
    }


#: 兼容旧引用（原先用于存放 CLIP 骨架哈希）
CLIP_SHA256: dict = {"sha256": csd_mod.CLIP_VIT_L14_SHA256}


# --------------------------------------------------------------------------- #
# 单指标在单后端上的评估
# --------------------------------------------------------------------------- #


def _vgg_features(images, backend, npu_encoder=None):
    if backend.actual == "npu":
        from utils.style_metrics.runner import vgg_features_npu

        return vgg_features_npu(npu_encoder, images)
    return runner.vgg_features_torch(images, backend)


def evaluate_metric(metric: str, backend, pair_a: Image.Image, pair_b: Image.Image, images: dict) -> dict:
    """返回 ``{'values': {...}, 'features_mae': float|None, 'timing': {...}, 'detail': {...}}``。"""
    out: dict = {"values": {}, "features_mae": None, "timing": {}, "detail": {}, "error": None}
    # 显存峰值按指标分别统计（否则后面的指标只会看到累加的历史最大值）
    if backend.actual == "cuda":
        try:
            import torch

            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    npu_encoder = None
    if backend.actual == "npu":
        from utils.style_metrics.openvino_backend import NPUEncoder, export_vgg19_onnx
        from utils.style_metrics.config import ONNX_CACHE

        onnx_path = export_vgg19_onnx(ONNX_CACHE / f"vgg19_features_{VGG_INPUT_SIZE}.onnx")
        npu_encoder = NPUEncoder(onnx_path)
        npu_encoder.compile()
        out["detail"]["npu_encoder"] = npu_encoder.describe()

    if metric in ("gram", "adain"):
        f_base = _vgg_features([images["base"]], backend, npu_encoder)
        f_same = _vgg_features([images["base_png"]], backend, npu_encoder)
        f_soft = _vgg_features([images["variant_soft"]], backend, npu_encoder)
        f_hard = _vgg_features([images["variant_hard"]], backend, npu_encoder)
        run = runner.run_gram if metric == "gram" else runner.run_adain
        for name, feats in (("same", f_same), ("soft", f_soft), ("hard", f_hard)):
            res = run(f_base, feats)
            out["values"][name] = res.value
            if res.error:
                out["error"] = res.error
            if name == "hard":
                out["detail"]["per_layer"] = res.detail
        out["features_mae"] = float((f_base["relu4_1"] - f_same["relu4_1"]).abs().max())

        def _call():
            f = _vgg_features([images["base"]], backend, npu_encoder)
            run(f, f_hard)

        out["timing"] = runner.time_call(_call, warmup=1, repeat=3, sync=runner.cuda_sync(backend))
        out["timing"]["split"] = (
            "特征编码器在 NPU（OpenVINO），Gram/AdaIN 距离在 CPU"
            if backend.actual == "npu"
            else f"特征编码器与距离计算都在 {backend.actual}"
        )

    elif metric == "lpips":
        if backend.actual == "npu":
            from utils.style_metrics.lpips_metric import NPULPIPSMetric

            npu_lpips = NPULPIPSMetric()
            out["detail"]["npu_encoder"] = npu_lpips.describe()
            for name, img in (
                ("same", images["base_png"]),
                ("soft", images["variant_soft"]),
                ("hard", images["variant_hard"]),
            ):
                dist, per_layer = npu_lpips.distance_images(images["base"], img)
                out["values"][name] = dist
                if name == "hard":
                    out["detail"]["per_layer"] = {"values": per_layer}

            def _call():
                npu_lpips.distance_images(images["base"], images["variant_hard"])

            out["timing"] = runner.time_call(_call, warmup=1, repeat=3)
            out["timing"]["split"] = (
                "官方 LPIPS 模块整体编译到 NPU（含 ScalingLayer 与校准 lin 层），"
                "输入准备在 CPU"
            )
            return out
        for name, img in (("same", images["base_png"]), ("soft", images["variant_soft"]), ("hard", images["variant_hard"])):
            res = runner.run_lpips(images["base"], img, backend)
            out["values"][name] = res.value
            if name == "hard":
                out["detail"]["per_layer"] = res.detail
            if res.error:
                out["error"] = res.error

        def _call():
            runner.run_lpips(images["base"], images["variant_hard"], backend)

        out["timing"] = runner.time_call(_call, warmup=1, repeat=3, sync=runner.cuda_sync(backend))
        out["timing"]["split"] = f"端到端（LPIPS 官方实现在 {backend.actual}）"

    elif metric == "csd":
        if backend.actual == "npu":
            from utils.style_metrics.csd_metric import NPUCSDMetric

            npu_csd = NPUCSDMetric()
            out["detail"]["npu_encoder"] = npu_csd.describe()
            for name, img in (
                ("same", images["base_png"]),
                ("soft", images["variant_soft"]),
                ("hard", images["variant_hard"]),
            ):
                dist, _, _ = npu_csd.cosine(images["base"], img)
                out["values"][name] = dist

            def _call():
                npu_csd.cosine(images["base"], images["variant_hard"])

            out["timing"] = runner.time_call(_call, warmup=1, repeat=3)
            out["timing"]["split"] = "描述子编码器在 NPU，余弦相似度在 CPU"
            return out
        from utils.style_metrics.runner import run_csd

        for name, img in (("same", images["base_png"]), ("soft", images["variant_soft"]), ("hard", images["variant_hard"])):
            res = run_csd(images["base"], img, backend)
            out["values"][name] = res.value
            if res.error:
                out["error"] = res.error
            if name == "hard":
                out["detail"]["model"] = {
                    k: v for k, v in res.detail.items() if k not in ("layers",)
                }

        def _call():
            runner.run_csd(images["base"], images["variant_hard"], backend)

        out["timing"] = runner.time_call(_call, warmup=1, repeat=3, sync=runner.cuda_sync(backend))
        out["timing"]["split"] = f"端到端（CSD 官方实现在 {backend.actual}）"

    if backend.actual == "cuda":
        out["timing"]["peak_gpu_memory_mb"] = runner.cuda_peak_memory_mb(backend)
    return out


# --------------------------------------------------------------------------- #
# 跨设备比较
# --------------------------------------------------------------------------- #


def compare_to_baseline(metric: str, current: dict, baseline: dict, precision: str) -> dict:
    tol = runner.tolerance_for(metric, precision)
    per_case = {}
    max_abs = 0.0
    max_rel = 0.0
    all_ok = True
    limited = False
    for case in ("same", "soft", "hard"):
        a = current.get(case)
        b = baseline.get(case)
        if a is None or b is None:
            per_case[case] = {"note": "该后端未产出该用例数值"}
            limited = True
            continue
        ok, limit = runner.within_tolerance(a, b, tol)
        abs_err = abs(a - b)
        rel_err = abs_err / (abs(b) + 1e-12)
        max_abs = max(max_abs, abs_err)
        if case != "same":
            max_rel = max(max_rel, rel_err)
        per_case[case] = {
            "actual": a,
            "baseline": b,
            "abs_error": abs_err,
            "rel_error": rel_err,
            "tolerance_limit": limit,
            "within_tolerance": ok,
            "note": "同图自比基准为 0，相对误差无意义，只看绝对误差" if case == "same" else None,
        }
        all_ok = all_ok and ok
    result = {
        "baseline_backend": "cpu",
        "baseline_precision": "fp32",
        "baseline_note": "同权重、同输入、同预处理，仅执行设备/精度不同",
        "tolerance": tol,
        "per_case": per_case,
        "max_abs_error": max_abs,
        "max_rel_error_on_different_images": max_rel,
        "within_tolerance": all_ok,
        "cases_compared": sum(1 for c in per_case.values() if "abs_error" in c),
        "partial": limited,
    }
    fama = current.get("features_mae")
    if fama is not None:
        result["feature_max_abs_error"] = fama
    # 相似度指标额外报告「相似度误差」
    if metric == "csd":
        vals = [c["abs_error"] for c in per_case.values() if "abs_error" in c]
        result["similarity_error"] = max(vals) if vals else None
    return result


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def download_missing(report: dict) -> dict:
    """按预检结果下载缺失权重（仅当显式传 --download-missing 时才走这里）。"""
    from utils.style_metrics.fetch import download

    done: dict = {}
    seen: set[str] = set()
    for problems in report.values():
        for p in problems:
            key = p["key"]
            if key in seen:
                continue
            seen.add(key)
            print(f"[下载] {key} ← {p['source_url']}")
            try:
                done[key] = download(
                    p["source_url"],
                    p["expected_path"],
                    expected_sha256=p["expected_sha256"],
                    expected_bytes=p["expected_bytes"],
                    timeout=900,
                )
            except Exception as exc:
                done[key] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                print(f"[下载失败] {key}: {done[key]['error']}")
    return done


def run(args) -> dict:
    images = imaging.make_test_images()
    pairs = [(Path(a), Path(b)) for a, b in (args.pair or [])]
    if pairs:
        missing_imgs = [str(p) for pair in pairs for p in pair if not p.exists()]
        if missing_imgs:
            raise FileNotFoundError("--pair 指定的图片不存在：\n  " + "\n  ".join(missing_imgs))
        images = {
            "base": imaging.load_rgb(pairs[0][0]),
            "base_png": imaging.load_rgb(pairs[0][1]),
            "variant_soft": imaging.load_rgb(pairs[-1][0]),
            "variant_hard": imaging.load_rgb(pairs[-1][1]),
        }

    payload: dict = {
        "schema": SCHEMA,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "requested_device": args.device,
        "precision_argument": args.precision,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "tolerance_policy": {
            "rule": "|actual - cpu_baseline| <= atol + rtol * |cpu_baseline|",
            "declared_before_running": True,
            "table": {f"{k[0]}|{k[1]}": v for k, v in runner.TOLERANCES.items()},
            "self_check": runner.SELF_TOLERANCE,
        },
        "weights": inventory.list_inventory(verify=args.verify_weights),
        "hardware": devices.hardware_report(),
        "input_images": {
            "source": "真实图片对" if pairs else "确定性合成图（imaging.make_test_images，seed=20261004）",
            "size": list(images["base"].size),
            "cases": ["same（base vs base 无损 PNG 往返）", "soft（暖色+模糊）", "hard（冷色+条纹+位移）"],
        },
        "results": [],
        "notes": [],
    }

    # 跑前权重预检：缺什么先说清楚，并给出可直接执行的修复命令。
    pre = inventory.preflight(args.metrics)
    if pre:
        print("[权重预检] 发现缺失：")
        print(inventory.format_preflight(pre))
        if args.download_missing:
            payload["notes"].append("按 --download-missing 尝试下载缺失权重")
            payload["downloads"] = download_missing(pre)
            pre = inventory.preflight(args.metrics)
        payload["weights_preflight"] = pre
        if pre:
            payload["notes"].append(
                "以下指标因权重缺失未运行（状态标记为 unavailable，不是计算失败，也不是 0 分）："
                + ", ".join(sorted(pre))
            )

    try:
        backend = devices.resolve_device(
            args.device, precision=args.precision, probe_npu=not args.no_probe
        )
    except devices.DeviceUnavailableError as exc:
        payload["notes"].append(f"设备解析失败: {exc}")
        payload["backend"] = {"requested": args.device, "actual": None, "error": str(exc)}
        payload["results"] = blocked_records(args.metrics, pre, args, reason=str(exc))
        payload["summary"] = summarize(payload["results"])
        return payload

    payload["backend"] = backend.as_dict()
    print(f"[设备] 请求={backend.requested} 实际={backend.actual} 后端={backend.backend} 精度={backend.precision}")
    if backend.note:
        print(f"       {backend.note}")

    # CPU 基准（同权重、同输入、同预处理）
    baseline: dict = {}
    if backend.actual != "cpu" and args.compare_cpu:
        runnable = [m for m in args.metrics if m not in pre]
        if runnable:
            print("[基准] 正在计算 CPU fp32 基准 …")
            try:
                cpu_backend = devices.resolve_device("cpu")
                for metric in runnable:
                    baseline[metric] = evaluate_metric(metric, cpu_backend, None, None, images)
            except Exception as exc:
                payload["notes"].append(f"CPU 基准失败: {type(exc).__name__}: {exc}")

    for metric in args.metrics:
        record: dict = {
            "metric": metric,
            "kind": runner.METRIC_KIND[metric],
            "status": "not_run",
            "model": model_for(metric),
            "weights_sha256": weights_sha256_for(metric),
            "preprocessing": preprocessing_for(metric),
            "requested_device": args.device,
            "actual_device": backend.actual,
            "backend": backend.backend,
            "precision": backend.precision,
            "validation": {},
            "timing": {},
            "error": None,
        }
        if metric in pre:
            record["status"] = "unavailable"
            record["error"] = "权重缺失，未运行：" + "; ".join(p["key"] for p in pre[metric])
            record["remediation"] = pre[metric]
            record["validation"] = {"note": "权重未就位，未产出任何数值（不做 0 分处理）"}
            print(f"[{metric}] 跳过：权重缺失 {[p['key'] for p in pre[metric]]}")
            payload["results"].append(record)
            continue
        print(f"[{metric}] 计算中 …")
        try:
            cur = evaluate_metric(metric, backend, None, None, images)
            record["error"] = cur.get("error")
            record["timing"] = cur.get("timing") or {}
            record["detail"] = cur.get("detail") or {}
            if cur.get("detail", {}).get("reason") == "weights_missing":
                record["remediation"] = [cur["detail"]]

            same = cur["values"].get("same")
            soft = cur["values"].get("soft")
            hard = cur["values"].get("hard")
            self_tol = runner.SELF_TOLERANCE[runner.METRIC_KIND[metric]]
            if runner.METRIC_KIND[metric] == "distance":
                same_ok = same is not None and abs(same) <= self_tol
                sep_ok = hard is not None and hard > (abs(soft) if soft is not None else 0.0)
            else:
                same_ok = same is not None and abs(same - 1.0) <= self_tol
                sep_ok = hard is not None and hard < (soft if soft is not None else 1.0)
            record["validation"] = {
                "self_distance_or_similarity": {
                    "value": same,
                    "expected": 0.0 if runner.METRIC_KIND[metric] == "distance" else 1.0,
                    "tolerance": self_tol,
                    "passed": same_ok,
                },
                "different_images": {
                    "soft": soft,
                    "hard": hard,
                    "monotonic_separation_passed": sep_ok,
                    "note": "不同图片必须给出与同图明显不同的数值；只靠同图自比不能证明计算正确",
                },
                "finite": all(v is not None and v == v and abs(v) != float("inf") for v in (same, soft, hard)),
            }
            if metric in baseline:
                record["validation"]["cross_device"] = compare_to_baseline(
                    metric, cur["values"], baseline[metric]["values"], backend.precision
                )
            else:
                record["validation"]["cross_device"] = {
                    "note": "本次未计算 CPU 基准（--no-compare-cpu 或该指标未运行）"
                }
            if cur.get("error"):
                record["status"] = "partial"
            elif not record["validation"]["finite"]:
                record["status"] = "error"
                record["error"] = record["error"] or "输出不是有限数值"
            elif not record["validation"]["self_distance_or_similarity"]["passed"]:
                record["status"] = "failed_self_check"
            else:
                record["status"] = "ok"
            print(
                f"  same={same} soft={soft} hard={hard} → status={record['status']}"
                + (f" err={record['error']}" if record["error"] else "")
            )
        except inventory.WeightsMissing as exc:
            # 运行中才发现权重缺失（例如只缺骨架/预热后才读到）→ 仍按 unavailable 处理
            record["status"] = "unavailable"
            record["error"] = str(exc)
            record["remediation"] = [exc.as_dict()]
            record["validation"] = {"note": "权重未就位，未产出任何数值（不做 0 分处理）"}
            print(f"  [缺权重] {record['error'].splitlines()[0]}")
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()[-2000:]
            print(f"  [失败] {record['error']}")
        payload["results"].append(record)

    payload["summary"] = summarize(payload["results"])
    return payload


def blocked_records(metrics, pre: dict, args, reason: str) -> list[dict]:
    """设备不可用时的记录：逐条写清「为什么没跑」，不产生任何假数值。"""
    records = []
    for metric in metrics:
        records.append(
            {
                "metric": metric,
                "kind": runner.METRIC_KIND[metric],
                "status": "unavailable" if metric in pre else "error",
                "model": model_for(metric),
                "weights_sha256": weights_sha256_for(metric),
                "preprocessing": preprocessing_for(metric),
                "requested_device": args.device,
                "actual_device": None,
                "backend": None,
                "precision": None,
                "validation": {"note": "设备不可用，未产出任何数值"},
                "timing": {},
                "error": reason if metric not in pre else f"权重缺失且设备不可用：{reason}",
                "remediation": pre.get(metric),
            }
        )
    return records


def summarize(results: list[dict]) -> dict:
    statuses: dict[str, int] = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {
        "metrics": [r["metric"] for r in results],
        "status_counts": statuses,
        "all_ok": all(r["status"] == "ok" for r in results) and bool(results),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="画风深度特征指标验证（Gram/AdaIN/LPIPS/CSD）")
    ap.add_argument("--device", default="auto", choices=list(("auto", "auto-npu", "cuda", "cpu", "npu")))
    ap.add_argument("--precision", default="fp32", choices=list(("fp32", "fp16")))
    ap.add_argument(
        "--metrics",
        default="gram,adain,lpips,csd",
        help="逗号分隔的子集，默认全跑",
    )
    ap.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"), help="真实图片对")
    ap.add_argument("--comparison-state", help="对画风提取结果的全部候选与全部源图计算本地指标并生成报告")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（默认 data/test-result/…）")
    ap.add_argument("--no-compare-cpu", action="store_true", help="跳过 CPU 基准对比")
    ap.add_argument("--compare-cpu", action="store_true", default=True, help=argparse.SUPPRESS)
    ap.add_argument("--verify-weights", action="store_true", help="对权重做 SHA-256 校验（大文件较慢）")
    ap.add_argument(
        "--download-missing",
        action="store_true",
        help="跑之前把缺失的权重按官方 URL 下载到 models/style-metrics/（默认不联网下载）",
    )
    ap.add_argument("--no-probe", action="store_true", help="跳过 NPU 运行期探针")
    ap.add_argument("--inventory", action="store_true", help="只打印权重清单与依赖状态")
    args = ap.parse_args(argv)

    if args.comparison_state:
        from modules.image_analysis.style_deep_comparison import compute
        try:
            result = compute(args.comparison_state, args.device,
                             progress=lambda message: print("PROGRESS " + message, flush=True))
            print(f"{result['status']}: {result['report_path']}")
            return 0
        except Exception:
            traceback.print_exc()
            return 1
    if args.device == "auto-npu":
        ap.error("auto-npu 用于 --comparison-state；部署验证请明确选择 npu/cuda/cpu 或 auto")

    args.metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in args.metrics if m not in runner.METRIC_KIND]
    if bad:
        print(f"未知指标: {bad}")
        return 2

    if args.inventory:
        inv = inventory.list_inventory(verify=True)
        extra = {
            "csd": csd_mod.cli_report(),
            "npu_probe": devices.npu_probe(use_cache=False),
            "cuda": devices.cuda_available(),
            "openvino": devices.openvino_available(),
        }
        print(json.dumps({"weights": inv, "environment": extra}, ensure_ascii=False, indent=1))
        missing = {
            k: v for k, v in inv.items() if not v.get("exists") or not v.get("size_matches")
        }
        if missing:
            print("\n[警告] 以下权重缺失或字节数不符：")
            print(inventory.format_preflight(
                {"<inventory>": [inventory.WeightsMissing(k).as_dict() for k in missing]}
            ))
            return 1
        return 0

    try:
        payload = run(args)
    except FileNotFoundError as exc:
        # 输入图片不存在之类的早期错误：直接报清楚，不写半个 JSON
        print(f"[输入错误] {exc}")
        return 2
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULT_DIR / f"style-metrics-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print("\n=== 摘要 ===")
    for r in payload["results"]:
        first_line = (r["error"] or "").splitlines()[0] if r["error"] else ""
        print(f"  {r['metric']:<6} status={r['status']:<18} {first_line}")
    summary = payload.get("summary", {})
    if summary.get("status_counts", {}).get("unavailable"):
        print("\n[缺权重] 修复命令（也可直接加 --download-missing 让 CLI 自己下）：")
        for r in payload["results"]:
            for item in r.get("remediation") or []:
                print(f"  [{r['metric']}] {item.get('remedy')}")
    print(f"JSON: {out}")
    return 0 if summary.get("all_ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
