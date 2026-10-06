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
import os
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

SCHEMA = "style-metrics-verification/2"


# --------------------------------------------------------------------------- #
# 预处理 / 模型描述（写进 JSON，保证口径可追溯）
# --------------------------------------------------------------------------- #


from utils.style_metrics.metadata import preprocessing_for, model_for


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


def run_pair_mode(args, pairs):
    from utils.style_metrics.pairs import compare_pairs
    payload = {"schema": SCHEMA, "mode": "real-pairs", "generated_at": datetime.now().isoformat(timespec="seconds"),
               "metric_versions": {"gram": GRAM_FORMULA_VERSION}, "results": [],
               "notes": ["只读真实图片配对；不修改图片、不合成百分比、不判配色成功。"],
               "weights": inventory.list_inventory(verify=args.verify_weights)}
    # No hidden downloads in pair mode; opt-in downloads still use the same helper.
    pre = inventory.preflight(args.metrics)
    if pre and args.download_missing:
        payload["downloads"] = download_missing(pre)
        pre = inventory.preflight(args.metrics)
    try:
        backend = devices.resolve_device(args.device, precision=args.precision, probe_npu=not args.no_probe)
    except devices.DeviceUnavailableError as exc:
        payload["results"] = blocked_records(args.metrics, pre, args, str(exc))
        payload["summary"] = summarize(payload["results"])
        return payload
    payload["backend"] = backend.as_dict()
    payload["results"] = compare_pairs(pairs, args.metrics, backend)
    for row in payload["results"]:
        row["model"] = model_for(row["metric"])
        row["preprocessing"] = preprocessing_for(row["metric"])
        row["weights_sha256"] = weights_sha256_for(row["metric"])
    if backend.actual != "cpu" and args.compare_cpu:
        baseline = compare_pairs(pairs, args.metrics, devices.resolve_device("cpu"))
        for row, reference in zip(payload["results"], baseline):
            if row["status"] == reference["status"] == "ok":
                tolerance = runner.tolerance_for(row["metric"], backend.precision)
                passed, limit = runner.within_tolerance(row["value"], reference["value"], tolerance)
                row["validation"]["cross_device"] = {"baseline": reference["value"], "baseline_device": "cpu", "baseline_precision": "fp32", "within_tolerance": passed, "limit": limit}
                if not passed:
                    row.update(status="partial", error="Pair cross-device comparison outside declared tolerance")
            else:
                row["validation"]["cross_device"] = {"within_tolerance": None, "baseline_status": reference["status"], "note": "Comparison unavailable; not a tolerance pass"}
                if row["status"] == "ok":
                    row.update(status="partial", error="CPU baseline unavailable for requested comparison")
    payload["summary"] = summarize(payload["results"])
    return payload


def run(args) -> dict:
    pairs = [(Path(a), Path(b)) for a, b in (args.pair or [])]
    if pairs:
        missing_imgs = [str(p) for pair in pairs for p in pair if not p.exists()]
        if missing_imgs:
            raise FileNotFoundError("--pair 指定的图片不存在：\n  " + "\n  ".join(missing_imgs))
        return run_pair_mode(args, pairs)

    images = imaging.make_test_images()

    payload: dict = {
        "schema": SCHEMA,
        "metric_versions": {"gram": GRAM_FORMULA_VERSION},
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
            "formula_version": GRAM_FORMULA_VERSION if metric == "gram" else None,
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
            "formula_version": GRAM_FORMULA_VERSION if metric == "gram" else None,
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
    ap.add_argument("--locate-regions", action="store_true", help="为待比较图和参考图调用文本模型生成面部/发丝定位候选（未人工确认）")
    ap.add_argument("--regions", action="append", default=[], help="导入与图片 hash 绑定的区域 JSON，可重复")
    ap.add_argument("--similarity", nargs=2, metavar=("IMAGE", "REFERENCE"), help="共享八组画风指标双图计算")
    ap.add_argument("--reference", action="append", default=[], help="追加参考图（与 --similarity 使用，逐张计算）")
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
    args.compare_cpu = not args.no_compare_cpu

    if args.reference and not args.similarity:
        ap.error("--reference 需要 --similarity")
    if args.similarity and args.comparison_state:
        ap.error("--similarity 与 --comparison-state 不能同时使用")
    if (args.locate_regions or args.regions) and not (args.similarity or args.comparison_state):
        ap.error("定位选项需要 --similarity 或 --comparison-state")
    if args.locate_regions or args.regions:
        from utils.style_regions import propose_regions, save_annotation
        from utils.style_face_metrics import digest
        if args.similarity:
            paths = [args.similarity[0], args.similarity[1], *args.reference]
        else:
            from modules.image_analysis.style_deep_comparison import input_manifest
            manifest = input_manifest(json.loads(Path(args.comparison_state).read_text(encoding="utf-8")))
            paths = [r["path"] for r in manifest["references"] + manifest["candidates"]]
        try:
            by_hash = {digest(path): path for path in paths}
            for filename in args.regions:
                annotation = json.loads(Path(filename).read_text(encoding="utf-8"))
                image_path = by_hash.get(annotation.get("image_sha256"))
                if not image_path:
                    raise ValueError("区域 JSON 不属于本次输入图片：" + filename)
                save_annotation(image_path, annotation)
            if args.locate_regions:
                from utils.analysis_gpt_prompt import load_text_api_config
                config = load_text_api_config()
                config = {**config, **{key: os.environ[name] for key, name in (
                    ("base_url", "IMAGE_MAKER_REGIONS_BASE_URL"), ("api_key", "IMAGE_MAKER_REGIONS_API_KEY"),
                    ("model", "IMAGE_MAKER_REGIONS_MODEL")) if name in os.environ}}
                for path in dict.fromkeys(paths):
                    try:
                        propose_regions(path, (config["base_url"], config["api_key"], config["model"]))
                        print("REGIONS 候选定位已缓存：" + str(path), flush=True)
                    except Exception as exc:
                        print("REGIONS 未完成：" + str(path) + " · " + str(exc), flush=True)
        except Exception:
            traceback.print_exc()
            return 1
    if args.similarity:
        from utils.style_similarity import image_manifest, compare_images, atomic_json, ALL_METRICS
        from modules.image_analysis.style_deep_comparison import write_report
        try:
            inputs = image_manifest([args.similarity[0]], [args.similarity[1], *args.reference])
            output = Path(args.out).resolve() if args.out else PROJECT_ROOT / "cache/temp/style-similarity" / datetime.now().strftime("%Y%m%d-%H%M%S-%f") / "result.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            result = compare_images(inputs, args.device, lambda message: print("PROGRESS " + message, flush=True))
            result["report_path"] = write_report(result, output.parent)
            atomic_json(output, result)
            for row in result["rows"]:
                for m in ALL_METRICS:
                    print(f"{m}: {row['summary'][m]}")
                for m, summary in row.get("face_summary", {}).items():
                    print(f"face.{m}: {summary}")
            print(f"整图 {result['status']} / 面部 {result.get('face_status', '未定位')}: {output}")
            return 0 if result["status"] == "ok" else 1
        except Exception:
            traceback.print_exc()
            return 1
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
    if any(row.get("remediation") for row in payload["results"]):
        print("\n[缺权重] 修复命令（也可直接加 --download-missing 让 CLI 自己下）：")
        for r in payload["results"]:
            for item in r.get("remediation") or []:
                print(f"  [{r['metric']}] {item.get('remedy')}")
    print(f"JSON: {out}")
    return 0 if summary.get("all_ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
