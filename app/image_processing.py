# app/image_processing.py
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from PIL import Image

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency at runtime
    YOLO = None
try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency at runtime
    np = None
try:
    import cv2
except Exception:  # pragma: no cover - optional dependency at runtime
    cv2 = None

from app.storage import download_media


_YOLO_MODEL = None
_WHEEL_FORK_MODEL = None


def _ensure_yolo_model_path(
    model_path_env: str = "YOLOV8_MODEL_PATH",
    bucket_env: str = "YOLO_BUCKET_NAME",
    name_env: str = "YOLO_MODEL_NAME",
    fallback_bucket_env: str = "YOLO_BUCKET_NAME",
) -> Optional[str]:
    model_path = os.getenv(model_path_env, "").strip()
    if model_path:
        return model_path if os.path.exists(model_path) else None

    bucket_name = os.getenv(bucket_env, "").strip()
    model_name = os.getenv(name_env, "").strip()

    if bucket_name.endswith(".pt") and not model_name:
        model_name = bucket_name
        bucket_name = os.getenv(fallback_bucket_env, "").strip()

    if not bucket_name or not model_name:
        return None

    local_dir = "/tmp/yolo"
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, model_name)

    if os.path.exists(local_path):
        print("INFO image_processing: using cached YOLO model at", local_path)
        return local_path

    try:
        print("INFO image_processing: downloading YOLO model from", bucket_name, model_name)
        data = download_media(bucket_name, model_name)
        with open(local_path, "wb") as f:
            f.write(data)
        print("INFO image_processing: YOLO model saved to", local_path)
        return local_path
    except Exception as exc:
        print("WARN image_processing: failed to download YOLO model:", exc)
        return None


def _load_yolo_model() -> Optional["YOLO"]:
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    if YOLO is None:
        return None

    model_path = _ensure_yolo_model_path()
    if not model_path:
        return None

    _YOLO_MODEL = YOLO(model_path)
    return _YOLO_MODEL


def _ensure_wheel_fork_model_path() -> Optional[str]:
    return _ensure_yolo_model_path(
        model_path_env="WHEEL_FORK_MODEL_PATH",
        bucket_env="WHEEL_FORK_BUCKET_NAME",
        name_env="WHEEL_FORK_MODEL_NAME",
        fallback_bucket_env="YOLO_BUCKET_NAME",
    )


def _load_wheel_fork_model() -> Optional["YOLO"]:
    global _WHEEL_FORK_MODEL
    if _WHEEL_FORK_MODEL is not None:
        return _WHEEL_FORK_MODEL

    if YOLO is None:
        return None

    model_path = _ensure_wheel_fork_model_path()
    if not model_path:
        return None

    _WHEEL_FORK_MODEL = YOLO(model_path)
    return _WHEEL_FORK_MODEL


@dataclass(frozen=True)
class EllipseFit:
    center: tuple[float, float]
    axes: tuple[float, float]
    angle_deg: float


TARGET_LABELS = {
    "wheel",
    "front suspension fork",
    "front fork",
    "fork",
}

EDGE_CANNY_LOW = 50
EDGE_CANNY_HIGH = 150


def _normalize_points(points: "np.ndarray"):
    mean = points.mean(axis=0)
    dists = np.linalg.norm(points - mean, axis=1)
    scale = np.sqrt(2.0) / max(np.mean(dists), 1e-6)
    t = np.array(
        [
            [scale, 0.0, -scale * mean[0]],
            [0.0, scale, -scale * mean[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    pts_h = np.column_stack([points, np.ones(len(points))])
    norm = (t @ pts_h.T).T[:, :2]
    return norm, t


def _conic_matrix(coeffs: "np.ndarray"):
    a, b, c, d, e, f = coeffs
    return np.array(
        [
            [a, b * 0.5, d * 0.5],
            [b * 0.5, c, e * 0.5],
            [d * 0.5, e * 0.5, f],
        ],
        dtype=float,
    )


def _conic_coeffs_from_matrix(mat: "np.ndarray"):
    return np.array(
        [
            mat[0, 0],
            2.0 * mat[0, 1],
            mat[1, 1],
            2.0 * mat[0, 2],
            2.0 * mat[1, 2],
            mat[2, 2],
        ],
        dtype=float,
    )


def _unnormalize_conic(coeffs: "np.ndarray", t: "np.ndarray"):
    c_norm = _conic_matrix(coeffs)
    c = t.T @ c_norm @ t
    return _conic_coeffs_from_matrix(c)


def _conic_to_ellipse(coeffs: "np.ndarray") -> Optional[EllipseFit]:
    a, b, c, d, e, f = coeffs
    if b * b - 4.0 * a * c >= 0.0:
        return None

    denom = b * b - 4.0 * a * c
    x0 = (2.0 * c * d - b * e) / denom
    y0 = (2.0 * a * e - b * d) / denom

    numerator = 2.0 * (a * x0 * x0 + b * x0 * y0 + c * y0 * y0 - f)
    term = np.sqrt((a - c) ** 2 + b * b)
    a2 = numerator / (a + c + term)
    b2 = numerator / (a + c - term)
    if a2 <= 0.0 or b2 <= 0.0:
        return None

    ra = float(np.sqrt(a2))
    rb = float(np.sqrt(b2))
    angle = 0.5 * float(np.degrees(np.arctan2(b, a - c)))

    if rb > ra:
        ra, rb = rb, ra
        angle += 90.0

    return EllipseFit(center=(float(x0), float(y0)), axes=(ra, rb), angle_deg=angle)


def fit_ellipse_direct(points: "np.ndarray") -> Optional[EllipseFit]:
    if len(points) < 5:
        return None
    pts = np.asarray(points, dtype=float)
    pts_norm, t = _normalize_points(pts)
    x = pts_norm[:, 0][:, None]
    y = pts_norm[:, 1][:, None]

    d1 = np.hstack([x * x, x * y, y * y])
    d2 = np.hstack([x, y, np.ones_like(x)])
    s1 = d1.T @ d1
    s2 = d1.T @ d2
    s3 = d2.T @ d2
    try:
        t_mat = -np.linalg.inv(s3) @ s2.T
    except np.linalg.LinAlgError:
        return None
    m = s1 + s2 @ t_mat
    c1 = np.array([[0.0, 0.0, 2.0], [0.0, -1.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(c1) @ m)
    except np.linalg.LinAlgError:
        return None

    cond = 4.0 * eigvecs[0] * eigvecs[2] - eigvecs[1] ** 2
    if not np.any(cond > 0):
        return None
    a1 = eigvecs[:, np.where(cond > 0)[0][0]]
    a2 = t_mat @ a1
    coeffs = np.concatenate([a1, a2]).flatten()
    coeffs = _unnormalize_conic(coeffs, t)
    return _conic_to_ellipse(coeffs)


def ellipse_to_conic(fit: EllipseFit) -> "np.ndarray":
    cx, cy = fit.center
    a, b = fit.axes
    angle = np.radians(fit.angle_deg)
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    a2 = a * a
    b2 = b * b
    a_mat = np.array(
        [
            [cos_t * cos_t / a2 + sin_t * sin_t / b2, cos_t * sin_t * (1 / a2 - 1 / b2)],
            [cos_t * sin_t * (1 / a2 - 1 / b2), sin_t * sin_t / a2 + cos_t * cos_t / b2],
        ],
        dtype=float,
    )
    d = -2.0 * (a_mat @ np.array([cx, cy]))
    f = (a_mat[0, 0] * cx * cx + 2 * a_mat[0, 1] * cx * cy + a_mat[1, 1] * cy * cy - 1.0)
    return np.array([a_mat[0, 0], 2.0 * a_mat[0, 1], a_mat[1, 1], d[0], d[1], f], dtype=float)


def _conic_residuals(coeffs: "np.ndarray", points: "np.ndarray"):
    a, b, c, d, e, f = coeffs
    x = points[:, 0]
    y = points[:, 1]
    num = a * x * x + b * x * y + c * y * y + d * x + e * y + f
    den = np.sqrt((2 * a * x + b * y + d) ** 2 + (b * x + 2 * c * y + e) ** 2)
    den = np.maximum(den, 1e-6)
    return np.abs(num) / den


def sampson_distance(points: "np.ndarray", fit: EllipseFit) -> float:
    if len(points) == 0:
        return float("inf")
    coeffs = ellipse_to_conic(fit)
    residuals = _conic_residuals(coeffs, np.asarray(points, dtype=float))
    return float(np.mean(residuals))


def fit_ellipse_ransac(
    points: "np.ndarray",
    iters: int = 200,
    thresh: float = 2.5,
    min_inliers: int = 20,
) -> Optional[EllipseFit]:
    if len(points) < 5:
        return None
    pts = np.asarray(points, dtype=float)
    best_inliers = None
    rng = np.random.default_rng(0)
    for _ in range(iters):
        sample_idx = rng.choice(len(pts), size=5, replace=False)
        fit = fit_ellipse_direct(pts[sample_idx])
        if fit is None:
            continue
        coeffs = ellipse_to_conic(fit)
        residuals = _conic_residuals(coeffs, pts)
        inliers = residuals < thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < min_inliers:
        return None
    fit = fit_ellipse_direct(pts[best_inliers])
    if fit is None:
        return None
    return EllipseFit(center=fit.center, axes=fit.axes, angle_deg=fit.angle_deg)


def fit_ellipse(points: Iterable[Iterable[float]], method: str = "ransac-direct") -> Optional[EllipseFit]:
    pts = np.asarray(list(points), dtype=float)
    if method == "direct":
        return fit_ellipse_direct(pts)
    if method == "ransac-direct":
        return fit_ellipse_ransac(pts)
    raise ValueError(f"Unknown ellipse fit method: {method}")


def canonical_label(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").split())


def label_group(name: str) -> Optional[str]:
    label = canonical_label(name)
    if label in TARGET_LABELS:
        if "fork" in label:
            return "fork"
        if "wheel" in label:
            return "wheel"
    if "fork" in label and "rear" not in label:
        return "fork"
    if "wheel" in label:
        return "wheel"
    return None


def auto_detect_rim_perspective_ellipses(
    image: Image.Image,
    conf: float = 0.25,
    ellipse_method: str = "ransac-direct",
) -> tuple[dict[str, dict], Optional[str], dict[str, dict]]:
    if YOLO is None:
        return {}, "Wheel detection unavailable; missing ultralytics.", {}
    if np is None or cv2 is None:
        return {}, "Wheel detection unavailable; missing numpy or opencv.", {}

    model = _load_wheel_fork_model()
    if model is None:
        return {}, "Wheel detection unavailable; missing wheel/fork model.", {}

    if torch is not None:
        try:
            torch.backends.nnpack.enabled = False
        except Exception:
            pass

    result_list = model(image, conf=conf, verbose=False)
    if not result_list:
        return {}, "No detections from wheel/fork model.", {}
    result = result_list[0]

    if not hasattr(result, "boxes") or result.boxes is None or len(result.boxes) == 0:
        return {}, "No wheel detections.", {}

    names = result.names
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()

    wheel_detections: list[tuple[float, float, float, float, float, str]] = []
    fork_detections: list[tuple[float, float, float, float, float, str]] = []

    for cls_id, confv, box in zip(cls_ids, confs, xyxy):
        label = names.get(cls_id, "")
        group = label_group(label)
        if group is None:
            continue
        x1, y1, x2, y2 = map(float, box)
        if group == "wheel":
            wheel_detections.append((x1, y1, x2, y2, float(confv), label))
        elif group == "fork":
            fork_detections.append((x1, y1, x2, y2, float(confv), label))

    if not wheel_detections:
        return {}, "No wheel detections.", {}

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, EDGE_CANNY_LOW, EDGE_CANNY_HIGH)
    border_margin = 40
    if border_margin > 0:
        edges[:border_margin, :] = 0
        edges[-border_margin:, :] = 0
        edges[:, :border_margin] = 0
        edges[:, -border_margin:] = 0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    def sample_grad(x: float, y: float):
        h, w = grad_mag.shape
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            return 0.0, 0.0, 0.0
        gx = float(grad_x[yi, xi])
        gy = float(grad_y[yi, xi])
        gm = float(grad_mag[yi, xi])
        return gx, gy, gm

    def snap_along_radius(
        center: tuple[float, float],
        dx: float,
        dy: float,
        r0: float,
        window_in: float,
        window_out: float,
        prefer_outward: bool = True,
        align_thresh: float = 0.7,
        grad_thresh: float = 15.0,
        bounds: tuple[float, float, float, float] | None = None,
    ):
        if prefer_outward:
            r_min = max(2.0, r0 - window_in)
            r_max = r0 + window_out
        else:
            r_min = max(2.0, r0 - window_out)
            r_max = r0 + window_in
        h, w = edges.shape
        candidates = []
        steps = int(window_in + window_out) * 2 + 1
        for r in np.linspace(r_min, r_max, steps):
            x = center[0] + dx * r
            y = center[1] + dy * r
            xi = int(round(x))
            yi = int(round(y))
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                continue
            if bounds is not None:
                bx1, by1, bx2, by2 = bounds
                if x < bx1 or x > bx2 or y < by1 or y > by2:
                    continue
            if border_margin > 0:
                if x < border_margin or x > (w - border_margin) or y < border_margin or y > (h - border_margin):
                    continue
            if edges[yi, xi] == 0:
                continue
            gx, gy, gm = sample_grad(x, y)
            if gm < grad_thresh:
                continue
            dot = (gx * dx + gy * dy) / (gm + 1e-6)
            alignment = abs(dot)
            if alignment < align_thresh:
                continue
            candidates.append((r, gm, alignment, x, y))
        if not candidates:
            return None
        chooser = max if prefer_outward else min
        r, gm, alignment, x, y = chooser(candidates, key=lambda c: c[0])
        return float(x), float(y), gm, alignment

    def collect_rim_points(
        center: tuple[float, float],
        a: float,
        b: float,
        angle_deg: float,
        prefer_outward: bool = True,
        bounds: tuple[float, float, float, float] | None = None,
        edge_margin: float = 0.0,
    ):
        samples = 180
        theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        cos_a = np.cos(np.radians(angle_deg))
        sin_a = np.sin(np.radians(angle_deg))
        rim_points = []
        for t in theta:
            local_x = a * np.cos(t)
            local_y = b * np.sin(t)
            x = center[0] + local_x * cos_a - local_y * sin_a
            y = center[1] + local_x * sin_a + local_y * cos_a

            dx = x - center[0]
            dy = y - center[1]
            r0 = float(np.hypot(dx, dy))
            if r0 <= 1.0:
                continue
            dx /= r0
            dy /= r0
            window_in = max(4.0, 0.03 * r0)
            if dy > 0.35:
                window_out = max(4.0, 0.05 * r0)
            else:
                window_out = max(8.0, 0.12 * r0)
            snapped = snap_along_radius(
                center,
                dx,
                dy,
                r0,
                window_in,
                window_out,
                prefer_outward=prefer_outward,
                bounds=bounds,
            )
            if snapped is not None:
                sx, sy = snapped[:2]
                if bounds is not None and edge_margin > 0.0:
                    bx1, by1, bx2, by2 = bounds
                    if (
                        abs(sx - bx1) < edge_margin
                        or abs(sx - bx2) < edge_margin
                        or abs(sy - by1) < edge_margin
                        or abs(sy - by2) < edge_margin
                    ):
                        continue
                if dy > 0.35:
                    r_snap = float(np.hypot(sx - center[0], sy - center[1]))
                    if r_snap > r0 * 1.06:
                        continue
                rim_points.append((sx, sy))
        return rim_points

    def bounds_for_box(bx1: float, by1: float, bx2: float, by2: float):
        pad = 6.0
        h, w = edges.shape
        return (
            max(border_margin, bx1 - pad),
            max(border_margin, by1 - pad),
            min(w - border_margin, bx2 + pad),
            min(h - border_margin, by2 + pad),
        )

    def fit_from_seed(seed_cx: float, seed_cy: float, seed_hw: float, seed_hh: float, seed_bounds):
        rim_points = collect_rim_points(
            (seed_cx, seed_cy),
            seed_hw,
            seed_hh,
            0.0,
            bounds=seed_bounds,
            edge_margin=max(4.0, 0.02 * (seed_hw * 2.0)),
        )
        if not rim_points:
            return None, None, None
        rim = np.array(rim_points, dtype=float)
        fit = fit_ellipse(rim, method=ellipse_method)
        if fit is None:
            return rim, None, None
        score = sampson_distance(rim, fit)
        return rim, fit, score

    fork_centers = []
    for fx1, fy1, fx2, fy2, _, _ in fork_detections:
        fork_centers.append(((fx1 + fx2) * 0.5, (fy1 + fy2) * 0.5))

    front_idx = None
    bike_faces_right = None
    front_ratio = None
    if fork_centers:
        best = None
        for idx, (x1, y1, x2, y2, confv, label) in enumerate(wheel_detections):
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            dist = min((cx - fx) ** 2 + (cy - fy) ** 2 for fx, fy in fork_centers)
            if best is None or dist < best[0]:
                best = (dist, idx)
        front_idx = best[1] if best is not None else None
        if front_idx is not None and len(wheel_detections) > 1:
            rear_idx = 1 - front_idx if len(wheel_detections) == 2 else None
            if rear_idx is not None:
                front_cx = (wheel_detections[front_idx][0] + wheel_detections[front_idx][2]) * 0.5
                rear_cx = (wheel_detections[rear_idx][0] + wheel_detections[rear_idx][2]) * 0.5
                bike_faces_right = front_cx > rear_cx
                fx1, fy1, fx2, fy2 = wheel_detections[front_idx][:4]
                fh = fy2 - fy1
                if fh > 0:
                    front_ratio = (fx2 - fx1) / fh

    fits_by_role: dict[str, EllipseFit] = {}

    for idx, (x1, y1, x2, y2, confv, label) in enumerate(wheel_detections):
        orig_x1, orig_y1, orig_x2, orig_y2 = x1, y1, x2, y2
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        half_w = (x2 - x1) * 0.5
        half_h = (y2 - y1) * 0.5

        wheel_role = "rear"
        if front_idx is not None:
            wheel_role = "front" if idx == front_idx else "rear"
        elif len(wheel_detections) == 2:
            # assume leftmost is rear, rightmost is front
            centers = [((wx1 + wx2) * 0.5, i) for i, (wx1, wy1, wx2, wy2, _, _) in enumerate(wheel_detections)]
            centers.sort(key=lambda item: item[0])
            rear_idx = centers[0][1]
            front_idx_guess = centers[1][1]
            wheel_role = "front" if idx == front_idx_guess else "rear"

        if wheel_role == "rear" and bike_faces_right is not None:
            ratio = front_ratio if front_ratio is not None else 1.0
            half_w = half_h * ratio
            rear_edge = x1 if bike_faces_right else x2
            cx = rear_edge + half_w if bike_faces_right else rear_edge - half_w
            x1 = rear_edge
            x2 = rear_edge + 2.0 * half_w if bike_faces_right else rear_edge - 2.0 * half_w

        seed_a = (
            (orig_x1 + orig_x2) * 0.5,
            (orig_y1 + orig_y2) * 0.5,
            (orig_x2 - orig_x1) * 0.5,
            (orig_y2 - orig_y1) * 0.5,
        )
        seed_b = (cx, cy, half_w, half_h)
        bounds_a = bounds_for_box(orig_x1, orig_y1, orig_x2, orig_y2)
        bounds_b = bounds_for_box(x1, y1, x2, y2)
        rim_a, fit_a, score_a = fit_from_seed(*seed_a, bounds_a)
        rim_b, fit_b, score_b = fit_from_seed(*seed_b, bounds_b)

        best_fit = None
        if fit_a is None and fit_b is None:
            best_fit = None
        elif fit_a is None:
            best_fit = fit_b
        elif fit_b is None:
            best_fit = fit_a
        else:
            best_fit = fit_a if score_a <= score_b else fit_b

        if best_fit is not None:
            fits_by_role[wheel_role] = best_fit

    boxes_out: dict[str, dict] = {}
    if front_idx is not None:
        fx1, fy1, fx2, fy2, fconf, _ = wheel_detections[front_idx]
        boxes_out["front_wheel"] = {
            "x1": float(fx1),
            "y1": float(fy1),
            "x2": float(fx2),
            "y2": float(fy2),
            "conf": float(fconf),
        }
        if len(wheel_detections) > 1:
            rear_idx = None
            if len(wheel_detections) == 2:
                rear_idx = 1 - front_idx
            if rear_idx is not None:
                rx1, ry1, rx2, ry2, rconf, _ = wheel_detections[rear_idx]
                boxes_out["rear_wheel"] = {
                    "x1": float(rx1),
                    "y1": float(ry1),
                    "x2": float(rx2),
                    "y2": float(ry2),
                    "conf": float(rconf),
                }
    elif len(wheel_detections) == 2:
        centers = [(((wx1 + wx2) * 0.5), i) for i, (wx1, wy1, wx2, wy2, _, _) in enumerate(wheel_detections)]
        centers.sort(key=lambda item: item[0])
        rear_idx = centers[0][1]
        front_idx_guess = centers[1][1]
        rx1, ry1, rx2, ry2, rconf, _ = wheel_detections[rear_idx]
        fx1, fy1, fx2, fy2, fconf, _ = wheel_detections[front_idx_guess]
        boxes_out["rear_wheel"] = {
            "x1": float(rx1),
            "y1": float(ry1),
            "x2": float(rx2),
            "y2": float(ry2),
            "conf": float(rconf),
        }
        boxes_out["front_wheel"] = {
            "x1": float(fx1),
            "y1": float(fy1),
            "x2": float(fx2),
            "y2": float(fy2),
            "conf": float(fconf),
        }

    if fork_detections:
        fx1, fy1, fx2, fy2, fconf, _ = max(fork_detections, key=lambda d: d[4])
        boxes_out["fork"] = {
            "x1": float(fx1),
            "y1": float(fy1),
            "x2": float(fx2),
            "y2": float(fy2),
            "conf": float(fconf),
        }

    ellipses_out: dict[str, dict] = {}
    for role, fit in fits_by_role.items():
        ellipses_out[role] = {
            "cx": float(fit.center[0]),
            "cy": float(fit.center[1]),
            "rx": float(fit.axes[0]),
            "ry": float(fit.axes[1]),
            "angle_deg": float(fit.angle_deg),
        }

    if not ellipses_out:
        return {}, "No rim fits found.", boxes_out

    return ellipses_out, None, boxes_out


def detect_single_bike_bbox(image: Image.Image) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[str]]:
    """Return a single bicycle bbox (x1, y1, x2, y2) or a warning string."""
    model = _load_yolo_model()
    if model is None:
        return None, "Bike detection unavailable; saved original image."

    if torch is not None:
        try:
            torch.backends.nnpack.enabled = False
        except Exception:
            pass

    results = model(image, verbose=False)
    if not results:
        return None, "No bike detected; saved original image."

    result = results[0]
    if not hasattr(result, "boxes") or result.boxes is None:
        return None, "No bike detected; saved original image."

    names = getattr(model, "names", {})
    bikes = []
    for box in result.boxes:
        cls_id = int(box.cls[0]) if hasattr(box, "cls") else None
        label = names.get(cls_id, None) if cls_id is not None else None
        if label != "bicycle":
            continue
        conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
        xyxy = box.xyxy[0].tolist()
        bikes.append((conf, xyxy))

    if len(bikes) == 0:
        return None, "No bike detected; saved original image."
    if len(bikes) > 1:
        return None, "Multiple bikes detected; saved original image."

    _, xyxy = bikes[0]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]

    pad = 0.08
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    x1 = int(round(x1 - bw * pad))
    y1 = int(round(y1 - bh * pad))
    x2 = int(round(x2 + bw * pad))
    y2 = int(round(y2 + bh * pad))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)

    return (x1, y1, x2, y2), None


def crop_and_resize_webp(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    long_edge_px: Optional[int],
    quality: int = 85,
) -> bytes:
    """Crop to bbox and return a WebP (resized if long_edge_px set)."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, image.width))
    x2 = max(0, min(x2, image.width))
    y1 = max(0, min(y1, image.height))
    y2 = max(0, min(y2, image.height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop box from detector.")

    cropped = image.crop((x1, y1, x2, y2))

    if long_edge_px:
        long_edge = max(cropped.width, cropped.height)
        if long_edge > long_edge_px:
            scale = long_edge_px / long_edge
            new_w = max(1, int(round(cropped.width * scale)))
            new_h = max(1, int(round(cropped.height * scale)))
            cropped = cropped.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    cropped.save(buf, format="WEBP", quality=quality, method=6)
    return buf.getvalue()


def open_image_from_bytes(content: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(content))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
