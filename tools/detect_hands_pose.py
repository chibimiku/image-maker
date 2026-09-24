# -*- coding: utf-8 -*-
"""独立进程里的手部检测（YOLO pose 手腕/肘 → 手部框）。被 utils/post_process 以子进程调用。

用法: python tools/detect_hands_pose.py --image <图> --out <json> [--subject x0,y0,x1,y1]
输出: {"hands": [[x0,y0,x1,y1], ...], "ok": true, "source": "pose"}
"""
import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subject", default="")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    from utils.post_process import HAND_POSE_MODEL

    result = {"hands": [], "ok": False, "source": "pose"}
    try:
        os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(HAND_POSE_MODEL), "ultralytics"))
        import cv2
        import numpy as np
        from ultralytics import YOLO

        img = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        subject = None
        if args.subject:
            vals = [int(float(v)) for v in args.subject.split(",")]
            if len(vals) == 4:
                subject = tuple(vals)
        os.makedirs(os.path.dirname(HAND_POSE_MODEL), exist_ok=True)
        model = YOLO(HAND_POSE_MODEL)
        res = model.predict(img, verbose=False, conf=0.3)
        hands = []
        if res and res[0].keypoints is not None and len(res[0].keypoints):
            kp = res[0].keypoints.xy.cpu().numpy()[0]
            kc = res[0].keypoints.conf.cpu().numpy()[0] if res[0].keypoints.conf is not None else None
            for wrist_idx, elbow_idx in ((9, 7), (10, 8)):
                if wrist_idx >= len(kp):
                    continue
                c = float(kc[wrist_idx]) if kc is not None else 1.0
                if c < args.conf:
                    continue
                wx, wy = float(kp[wrist_idx][0]), float(kp[wrist_idx][1])
                if wx <= 0 and wy <= 0:
                    continue
                if elbow_idx < len(kp) and kc is not None and float(kc[elbow_idx]) >= args.conf:
                    ex, ey = float(kp[elbow_idx][0]), float(kp[elbow_idx][1])
                    fx, fy = wx - ex, wy - ey
                    norm = max(1e-3, (fx * fx + fy * fy) ** 0.5)
                    hx, hy = wx + fx / norm * norm * 0.45, wy + fy / norm * norm * 0.45
                    half = max(0.22 * norm, 0.02 * h)
                else:
                    hx, hy, half = wx, wy, 0.03 * h
                half = float(min(max(half, 0.015 * h), 0.09 * h))
                box = [int(hx - half), int(hy - half), int(hx + half), int(hy + half)]
                if subject:
                    sx0, sy0, sx1, sy1 = subject
                    box = [max(sx0, box[0]), max(sy0, box[1]), min(sx1, box[2]), min(sy1, box[3])]
                if box[2] - box[0] >= 6 and box[3] - box[1] >= 6:
                    hands.append(box)
        result["hands"] = hands
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
