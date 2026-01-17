import math
from typing import Iterable, Optional

import numpy as np


def _ellipse_extrema_points(ellipse: dict) -> Optional[dict]:
    if not ellipse:
        return None
    cx = float(ellipse.get("cx"))
    cy = float(ellipse.get("cy"))
    rx = float(ellipse.get("rx"))
    ry = float(ellipse.get("ry"))
    angle_deg = float(ellipse.get("angle_deg", 0.0))
    if not all(math.isfinite(v) for v in (cx, cy, rx, ry, angle_deg)):
        return None

    rot = math.radians(angle_deg)
    cos_a = math.cos(rot)
    sin_a = math.sin(rot)

    tx = math.atan2(-ry * sin_a, rx * cos_a)
    ty = math.atan2(ry * cos_a, rx * sin_a)
    ts = [tx, tx + math.pi, ty, ty + math.pi]

    points = []
    for t in ts:
        x = cx + rx * math.cos(t) * cos_a - ry * math.sin(t) * sin_a
        y = cy + rx * math.cos(t) * sin_a + ry * math.sin(t) * cos_a
        points.append({"x": x, "y": y})

    by_x = sorted(points, key=lambda p: p["x"])
    by_y = sorted(points, key=lambda p: p["y"])
    return {
        "west": by_x[0],
        "east": by_x[-1],
        "north": by_y[0],
        "south": by_y[-1],
        "center": {"x": cx, "y": cy},
    }


def _normalize_points(points: Iterable[dict]) -> Optional[tuple[list[dict], np.ndarray]]:
    pts = list(points)
    if not pts:
        return None

    xs = np.array([p["x"] for p in pts], dtype=float)
    ys = np.array([p["y"] for p in pts], dtype=float)
    cx = float(xs.mean())
    cy = float(ys.mean())
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    mean_d = float(d.mean())
    if mean_d <= 0:
        return None

    scale = math.sqrt(2) / mean_d
    T = np.array(
        [
            [scale, 0, -scale * cx],
            [0, scale, -scale * cy],
            [0, 0, 1],
        ],
        dtype=float,
    )
    out = []
    for p in pts:
        x = scale * (p["x"] - cx)
        y = scale * (p["y"] - cy)
        out.append({"x": x, "y": y})
    return out, T


def _solve_homography_from_pairs(src_pts: list[dict], dst_pts: list[dict]) -> Optional[np.ndarray]:
    norm_src = _normalize_points(src_pts)
    norm_dst = _normalize_points(dst_pts)
    if not norm_src or not norm_dst:
        return None
    src_norm, T_src = norm_src
    dst_norm, T_dst = norm_dst

    A = []
    for s, d in zip(src_norm, dst_norm):
        x, y = s["x"], s["y"]
        u, v = d["x"], d["y"]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A, dtype=float)

    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    h = vt[-1, :]
    if abs(h[-1]) < 1e-12:
        return None
    Hn = h.reshape(3, 3)

    try:
        T_dst_inv = np.linalg.inv(T_dst)
    except np.linalg.LinAlgError:
        return None
    H = T_dst_inv @ Hn @ T_src
    if abs(H[2, 2]) < 1e-12:
        return None
    H = H / H[2, 2]
    return H


def apply_homography(H: np.ndarray, x: float, y: float) -> Optional[tuple[float, float]]:
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    if abs(w) < 1e-12:
        return None
    nx = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
    ny = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
    return float(nx), float(ny)


def _rectify_params(rear_ellipse: Optional[dict], front_ellipse: Optional[dict]) -> Optional[dict]:
    basis = rear_ellipse or front_ellipse
    if not basis:
        return None
    avg_r = (basis["rx"] + basis["ry"]) / 2 or 1
    return {"scale": float(avg_r), "tx": float(basis["cx"]), "ty": float(basis["cy"])}


def compute_homography_from_ellipses(
    rear_ellipse: Optional[dict],
    front_ellipse: Optional[dict],
    mode: str = "both_ls",
) -> Optional[dict]:
    if mode == "front":
        rear_ellipse = None
    if not rear_ellipse and not front_ellipse:
        return None

    if not rear_ellipse or not front_ellipse:
        ellipse = rear_ellipse or front_ellipse
        c = _ellipse_extrema_points(ellipse)
        if not c:
            return None
        src = [c["north"], c["east"], c["south"], c["west"]]
        dst = [
            {"x": 0, "y": -1},
            {"x": 1, "y": 0},
            {"x": 0, "y": 1},
            {"x": -1, "y": 0},
        ]
        H = _solve_homography_from_pairs(src, dst)
        if H is None:
            return None
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        rectify = _rectify_params(rear_ellipse, front_ellipse)
        return {"H": H, "H_inv": H_inv, "rectify": rectify}

    rear_r = (rear_ellipse["rx"] + rear_ellipse["ry"]) / 2
    front_r = (front_ellipse["rx"] + front_ellipse["ry"]) / 2
    avg_r = (rear_r + front_r) / 2 or 1

    center_dist = math.hypot(
        front_ellipse["cx"] - rear_ellipse["cx"],
        front_ellipse["cy"] - rear_ellipse["cy"],
    )
    d = center_dist / avg_r

    rear_scale = rear_r / avg_r
    front_scale = front_r / avg_r
    rear_c = _ellipse_extrema_points(rear_ellipse)
    front_c = _ellipse_extrema_points(front_ellipse)
    if not rear_c or not front_c:
        return None

    src = [
        rear_c["north"],
        rear_c["east"],
        rear_c["south"],
        rear_c["west"],
        front_c["north"],
        front_c["east"],
        front_c["south"],
        front_c["west"],
    ]
    dst = [
        {"x": 0, "y": -rear_scale},
        {"x": rear_scale, "y": 0},
        {"x": 0, "y": rear_scale},
        {"x": -rear_scale, "y": 0},
        {"x": d, "y": -front_scale},
        {"x": d + front_scale, "y": 0},
        {"x": d, "y": front_scale},
        {"x": d - front_scale, "y": 0},
    ]

    H = _solve_homography_from_pairs(src, dst)
    if H is None:
        return None
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    rectify = _rectify_params(rear_ellipse, front_ellipse)
    return {"H": H, "H_inv": H_inv, "rectify": rectify}
