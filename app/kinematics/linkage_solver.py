# app/kinematics/linkage_solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import math
from pydantic import BaseModel

from app.schemas import BikePoint, RigidBody  # reuse your existing models


# ------------ Internal edge representation ------------


@dataclass
class LinkEdge:
    ia: int         # index of start point
    ib: int         # index of end point
    L0: float       # rest length at zero travel (in same units as points)
    is_shock: bool  # True if this edge is the driver shock


# ------------ Public solver output models ------------


class SolverStep(BaseModel):
    """
    State of the linkage at one shock-stroke step.
    """
    step_index: int
    shock_stroke: float            # shock stroke at this step (same units as driver_stroke)
    shock_length: float            # eye-to-eye at this step (same units as points)
    rear_travel: Optional[float]   # vertical rear axle travel vs initial (same units as points)
    leverage_ratio: Optional[float]  # d(rear_travel)/d(shock_stroke)
    anti_squat: Optional[float] = None
    anti_rise: Optional[float] = None
    shock_spring_rate: Optional[float] = None
    rear_wheel_force: Optional[float] = None
    points: Dict[str, Tuple[float, float]]  # point.id -> (x, y)


class SolverResult(BaseModel):
    """
    Full sweep from zero to full shock stroke.
    """
    steps: List[SolverStep]
    rear_axle_point_id: Optional[str]
    # Optional debug payload for the frontend
    debug: Optional[Dict[str, object]] = None
    # Optional full steps including pre-roll (negative stroke)
    full_steps: Optional[List[SolverStep]] = None
    # Optional scaled outputs for UI tables/plots
    scaled_outputs: Optional[Dict[str, object]] = None


# ------------ Converter: BikePoint + RigidBody → edges + flags ------------


def _build_internal_model(
    points: List[BikePoint],
    bodies: List[RigidBody],
    *,
    lock_point_types: bool = True,
    include_fixed_body_rigid_groups: bool = False,
) -> tuple[
    List[LinkEdge],
    List[bool],         # fixed flags per point index
    Optional[int],      # rear_axle_idx
    int,                # driver_edge_idx
    float,              # driver_L0
    float,              # driver_stroke
    List[dict],         # rigid groups for shape matching
]:
    """
    Convert your domain objects into an internal form for the solver:
    - edges: list of LinkEdge (bars + shock)
    - fixed: list of booleans for each point (locked vs free)
    - rear_axle_idx: optional index of rear axle point
    - driver_edge_idx: which edge is the shock driver
    - driver_L0: rest length of the shock at zero travel
    - driver_stroke: total stroke of the shock
    """
    if not points:
        raise ValueError("No points defined for this bike.")
    if not bodies:
        raise ValueError("No rigid bodies defined for this bike.")

    # Map point id -> index
    idx: Dict[str, int] = {p.id: i for i, p in enumerate(points)}
    if len(idx) != len(points):
        raise ValueError("BikePoint IDs must be unique.")

    # Initial coordinates
    x0 = [p.x for p in points]
    y0 = [p.y for p in points]

    # Fixed points: type == "fixed" or "bb" in the standard solver path.
    fixed = [lock_point_types and p.type in ("fixed", "bb") for p in points]

    # Rear axle index (optional)
    rear_axle_idx = next(
        (i for i, p in enumerate(points) if p.type == "rear_axle"),
        None,
    )

    edges: List[LinkEdge] = []
    rigid_groups: List[dict] = []
    driver_edge_idx: Optional[int] = None
    driver_L0: Optional[float] = None
    driver_stroke: Optional[float] = None

    # for body in bodies:
    #     pids = body.point_ids or []
    #     if len(pids) < 2:
    #         continue

    #     # Mark frame clusters as fixed (extra safety)
    #     if body.type == "fixed":
    #         for pid in pids:
    #             if pid not in idx:
    #                 raise ValueError(f"RigidBody {body.id!r} references unknown point {pid!r}.")
    #             fixed[idx[pid]] = True

    #     # Build segments (start/end) from ordered point_ids
    #     seg_pairs: List[Tuple[str, str]] = list(zip(pids, pids[1:]))
    #     if body.closed and len(pids) > 2:
    #         seg_pairs.append((pids[-1], pids[0]))

    #     for a_id, b_id in seg_pairs:
    #         if a_id not in idx or b_id not in idx:
    #             raise ValueError(f"RigidBody {body.id!r} references unknown point.")
    #         ia, ib = idx[a_id], idx[b_id]
    #         dx = x0[ib] - x0[ia]
    #         dy = y0[ib] - y0[ia]
    #         L0_geom = math.hypot(dx, dy)

    #         if body.type == "shock":
    #             # Shock body → this segment is the driver shock
    #             L0 = body.length0 if getattr(body, "length0", None) is not None else L0_geom
    #             stroke_val = getattr(body, "stroke", None)
    #             if stroke_val is None:
    #                 raise ValueError(f"Shock body {body.id!r} must define 'stroke'.")
    #             edge = LinkEdge(ia=ia, ib=ib, L0=L0, is_shock=True)
    #             edges.append(edge)
    #             if driver_edge_idx is not None:
    #                 raise ValueError("Multiple shock bodies found; only one driver is supported.")
    #             driver_edge_idx = len(edges) - 1
    #             driver_L0 = L0
    #             driver_stroke = float(stroke_val)
    #         else:
    #             # Normal bar (moving) or fixed body segment
    #             edges.append(LinkEdge(ia=ia, ib=ib, L0=L0_geom, is_shock=False))
    for body in bodies:
        pids = body.point_ids or []
        if len(pids) < 2:
            continue

        # Fixed body: lock all its points
        if body.type == "fixed" and lock_point_types:
            for pid in pids:
                if pid not in idx:
                    raise ValueError(f"RigidBody {body.id!r} references unknown point {pid!r}.")
                fixed[idx[pid]] = True

        # Shock: allow 2 points or 3 points with extension
        if body.type == "shock":
            if len(pids) not in (2, 3):
                raise ValueError(f"Shock body {body.id!r} must have 2 or 3 point_ids.")
            if len(pids) == 2:
                a_id, b_id = pids
                extra_ids: list[str] = []
            else:
                # Use the first two point_ids as the driver edge. Any remaining point
                # is treated as a rigid extension off the driver end.
                a_id, b_id = pids[0], pids[1]
                extra_ids = [pid for pid in pids if pid not in (a_id, b_id)]
            if a_id not in idx or b_id not in idx:
                raise ValueError(f"Shock body {body.id!r} references unknown point.")
            ia, ib = idx[a_id], idx[b_id]
            dx = x0[ib] - x0[ia]
            dy = y0[ib] - y0[ia]
            L0_geom = math.hypot(dx, dy)
            L0 = body.length0 if getattr(body, "length0", None) is not None else L0_geom
            if getattr(body, "stroke", None) is None:
                raise ValueError(f"Shock body {body.id!r} must define 'stroke'.")
            edges.append(LinkEdge(ia=ia, ib=ib, L0=L0, is_shock=True))
            if driver_edge_idx is not None:
                raise ValueError("Multiple shock bodies found; only one driver is supported.")
            driver_edge_idx = len(edges) - 1
            driver_L0 = L0
            driver_stroke = body.stroke
            # Add rigid extension edges for remaining points.
            if extra_ids:
                for extra_id in extra_ids:
                    if extra_id not in idx:
                        continue
                    ie = idx[extra_id]
                    dx = x0[ie] - x0[ib]
                    dy = y0[ie] - y0[ib]
                    L0_ext = math.hypot(dx, dy)
                    edges.append(LinkEdge(ia=ib, ib=ie, L0=L0_ext, is_shock=False))
            continue

        # Non-shock bodies: rigidify with hidden diagonals (solver-only)
        for a_id, b_id in _rigid_pairs(pids, closed=bool(body.closed)):
            if a_id not in idx or b_id not in idx:
                raise ValueError(f"RigidBody {body.id!r} references unknown point.")
            ia, ib = idx[a_id], idx[b_id]
            dx = x0[ib] - x0[ia]
            dy = y0[ib] - y0[ia]
            L0_geom = math.hypot(dx, dy)
            edges.append(LinkEdge(ia=ia, ib=ib, L0=L0_geom, is_shock=False))

        # Rigid body shape matching (>=3 points, non-shock, non-fixed)
        if body.type != "fixed" or include_fixed_body_rigid_groups:
            group_indices = [idx[pid] for pid in pids if pid in idx]
            if len(group_indices) >= 3:
                rest = [(x0[i], y0[i]) for i in group_indices]
                cx = sum(p[0] for p in rest) / len(rest)
                cy = sum(p[1] for p in rest) / len(rest)
                rigid_groups.append(
                    {
                        "indices": group_indices,
                        "rest": rest,
                        "rest_cx": cx,
                        "rest_cy": cy,
                    }
                )

    if driver_edge_idx is None or driver_L0 is None or driver_stroke is None:
        raise ValueError("No shock body found; need exactly one RigidBody with type='shock'.")

    return edges, fixed, rear_axle_idx, driver_edge_idx, driver_L0, driver_stroke, rigid_groups


def _rigid_pairs(point_ids: list[str], closed: bool) -> list[tuple[str, str]]:
    """Return constraint pairs for a rigid point set (complete graph)."""
    pids = [pid for pid in point_ids if pid]
    n = len(pids)
    if n < 2:
        return []

    pairs: list[tuple[str, str]] = []
    for i in range(n - 1):
        a = pids[i]
        for j in range(i + 1, n):
            pairs.append((a, pids[j]))
    return pairs

# ------------ Core PBD solver ------------


def _solve_with_edges(
    points: List[BikePoint],
    edges: List[LinkEdge],
    fixed: List[bool],
    rear_axle_idx: Optional[int],
    driver_edge_idx: int,
    driver_L0: float,
    driver_stroke: float,
    rigid_groups: List[dict],
    n_steps: int,
    iterations: int,
    pre_steps: int = 0,
    axis_constraints: Optional[Dict[int, Dict[str, float]]] = None,
) -> SolverResult:
    """
    Position-based distance-constraint solver for the linkage.
    - Treats each edge as a distance constraint.
    - For the driver shock edge, target length = L0 - shock_stroke.
    """
    n = len(points)
    x = [p.x for p in points]
    y = [p.y for p in points]

    def _apply_axis_constraints() -> None:
        if not axis_constraints:
            return
        for idx, constraint in axis_constraints.items():
            if idx < 0 or idx >= len(x):
                continue
            target_x = constraint.get("x")
            target_y = constraint.get("y")
            if target_x is not None:
                x[idx] = float(target_x)
            if target_y is not None:
                y[idx] = float(target_y)

    def _project_distance_constraints(target_shock_len: float) -> None:
        for ei, edge in enumerate(edges):
            ia, ib = edge.ia, edge.ib

            if ei == driver_edge_idx and edge.is_shock:
                target_L = max(1e-6, target_shock_len)
            else:
                target_L = edge.L0

            dx = x[ib] - x[ia]
            dy = y[ib] - y[ia]
            dist = math.hypot(dx, dy) or 1e-9
            diff = (dist - target_L) / dist

            fa = fixed[ia]
            fb = fixed[ib]

            if fa and fb:
                continue
            elif fa and not fb:
                x[ib] -= dx * diff
                y[ib] -= dy * diff
            elif fb and not fa:
                x[ia] += dx * diff
                y[ia] += dy * diff
            else:
                half = 0.5
                x[ia] += dx * diff * half
                y[ia] += dy * diff * half
                x[ib] -= dx * diff * half
                y[ib] -= dy * diff * half

    # Rear axle initial vertical position
    rear_y0 = y[rear_axle_idx] if rear_axle_idx is not None else None

    steps: List[SolverStep] = []
    n_steps = max(1, n_steps)
    iterations = max(1, iterations)

    total_steps = n_steps + max(0, pre_steps)
    # Negative extension length based on pre-roll steps.
    pre_roll_len = 0.0
    if pre_steps > 0:
        step_stroke = driver_stroke / n_steps
        pre_roll_len = step_stroke * pre_steps

    _apply_axis_constraints()

    for step_i in range(total_steps + 1):
        # Shock stroke used at this step (negative → extension, if pre_steps > 0)
        if pre_steps > 0 and step_i < pre_steps:
            # Linearly extend up to pre_roll_len before zero travel.
            s = -pre_roll_len * ((pre_steps - step_i) / pre_steps)
        else:
            s = driver_stroke * ((step_i - pre_steps) / n_steps)
        target_shock_len = driver_L0 - s  # shorten shock with positive stroke

        # Iterative constraint projection
        for _ in range(iterations):
            _project_distance_constraints(target_shock_len)

            # Rigid body shape matching (prevents collinear "bending")
            for group in rigid_groups or []:
                indices = group["indices"]
                if not indices:
                    continue
                if all(fixed[i] for i in indices):
                    continue

                rest = group["rest"]
                rest_cx = group["rest_cx"]
                rest_cy = group["rest_cy"]

                cur_cx = sum(x[i] for i in indices) / len(indices)
                cur_cy = sum(y[i] for i in indices) / len(indices)

                a = 0.0
                b = 0.0
                for (rx, ry), i in zip(rest, indices):
                    qx = rx - rest_cx
                    qy = ry - rest_cy
                    px = x[i] - cur_cx
                    py = y[i] - cur_cy
                    a += px * qx + py * qy
                    b += py * qx - px * qy

                denom = math.hypot(a, b)
                if denom < 1e-9:
                    continue
                cos_t = a / denom
                sin_t = b / denom

                for (rx, ry), i in zip(rest, indices):
                    if fixed[i]:
                        continue
                    qx = rx - rest_cx
                    qy = ry - rest_cy
                    tx = cos_t * qx - sin_t * qy
                    ty = sin_t * qx + cos_t * qy
                    x[i] = cur_cx + tx
                    y[i] = cur_cy + ty

            _apply_axis_constraints()

        # Final length projection after the last rigid-group solve keeps the
        # recorded zero-travel pose on the target shock eye-to-eye length.
        for _ in range(3):
            _project_distance_constraints(target_shock_len)
            _apply_axis_constraints()

        # --- Record this step ---
        # Rear axle travel (vertical, positive "up" from initial)
        rear_travel: Optional[float] = None
        if rear_axle_idx is not None and rear_y0 is not None:
            rear_travel = rear_y0 - y[rear_axle_idx]

        # Actual shock length after solve
        driver_edge = edges[driver_edge_idx]
        ia, ib = driver_edge.ia, driver_edge.ib
        dx = x[ib] - x[ia]
        dy = y[ib] - y[ia]
        shock_len = math.hypot(dx, dy)

        # Point positions
        positions = {p.id: (x[i], y[i]) for i, p in enumerate(points)}

        # Leverage via finite difference
        leverage: Optional[float] = None
        if step_i > 0 and rear_travel is not None and steps[-1].rear_travel is not None:
            ds = s - steps[-1].shock_stroke
            if abs(ds) > 1e-9:
                dr = rear_travel - steps[-1].rear_travel
                leverage = dr / ds

        steps.append(
            SolverStep(
                step_index=step_i,
                shock_stroke=s,
                shock_length=shock_len,
                rear_travel=rear_travel,
                leverage_ratio=leverage,
                points=positions,
            )
        )

    full_steps = steps
    if pre_steps > 0 and len(steps) > pre_steps:
        trimmed: List[SolverStep] = []
        for s in steps[pre_steps:]:
            trimmed.append(
                s.copy(update={"step_index": s.step_index - pre_steps})
            )
        steps = trimmed

    rear_axle_id = points[rear_axle_idx].id if rear_axle_idx is not None else None

    # --- Debug payload for the frontend ---
    debug_data: Dict[str, object] = {
        "edges": [
            {
                "ia": e.ia,
                "ib": e.ib,
                "L0": e.L0,
                "is_shock": e.is_shock,
                "a_id": points[e.ia].id,
                "b_id": points[e.ib].id,
            }
            for e in edges
        ],
        "fixed_point_ids": [p.id for i, p in enumerate(points) if fixed[i]],
        "rear_axle_point_id": rear_axle_id,
        "driver_edge_index": driver_edge_idx,
        "driver_L0": driver_L0,
        "driver_stroke": driver_stroke,
        "pre_steps": pre_steps,
        "axis_constraints": {
            points[idx].id: {
                key: float(value)
                for key, value in constraint.items()
                if key in {"x", "y"} and value is not None
            }
            for idx, constraint in (axis_constraints or {}).items()
            if 0 <= idx < len(points)
        },
    }

    return SolverResult(
        steps=steps,
        rear_axle_point_id=rear_axle_id,
        debug=debug_data,
        full_steps=full_steps,
    )


# ------------ Public API ------------


def solve_bike_linkage(
    points: List[BikePoint],
    bodies: List[RigidBody],
    n_steps: int = 1000,
    iterations: int = 1000,
    pre_steps: int = 0,
) -> SolverResult:
    """
    Public entrypoint used by your router.

    Expects:
      - BikePoint.type ∈ {"bb","rear_axle","front_axle","free","fixed"}
      - Exactly one RigidBody with type="shock", with >=2 point_ids and stroke set.
      - RigidBody.length0 and RigidBody.stroke in the same units as point coords,
        or at least in a consistent arbitrary unit system.
    """
    (
        edges,
        fixed,
        rear_axle_idx,
        driver_edge_idx,
        driver_L0,
        driver_stroke,
        rigid_groups,
    ) = _build_internal_model(points, bodies)

    return _solve_with_edges(
        points=points,
        edges=edges,
        fixed=fixed,
        rear_axle_idx=rear_axle_idx,
        driver_edge_idx=driver_edge_idx,
        driver_L0=driver_L0,
        driver_stroke=driver_stroke,
        rigid_groups=rigid_groups,
        n_steps=n_steps,
        iterations=iterations,
        pre_steps=pre_steps,
    )


def solve_bike_rest_pose(
    points: List[BikePoint],
    bodies: List[RigidBody],
    *,
    point_constraints: Dict[str, Dict[str, float]],
    iterations: int = 800,
) -> tuple[List[BikePoint], Dict[str, object]]:
    (
        edges,
        fixed,
        _rear_axle_idx,
        driver_edge_idx,
        driver_L0,
        _driver_stroke,
        rigid_groups,
    ) = _build_internal_model(
        points,
        bodies,
        lock_point_types=False,
        include_fixed_body_rigid_groups=True,
    )

    # In rest-pose solving the frame reference points must move as a single
    # rigid carrier. Otherwise the BB/fixed points can satisfy axis
    # constraints independently and "shear" relative to one another.
    point_index_by_id = {p.id: idx for idx, p in enumerate(points)}
    frame_anchor_ids: set[str] = {
        str(p.id)
        for p in points
        if str(getattr(p, "type", "") or "").strip().lower() in {"fixed", "bb"}
    }
    for body in bodies or []:
        if str(getattr(body, "type", "") or "").strip().lower() != "fixed":
            continue
        for pid in getattr(body, "point_ids", None) or []:
            if pid:
                frame_anchor_ids.add(str(pid))

    frame_anchor_indices = [
        idx for pid, idx in point_index_by_id.items() if pid in frame_anchor_ids
    ]
    if len(frame_anchor_indices) >= 2:
        existing_pairs = {
            tuple(sorted((edge.ia, edge.ib)))
            for edge in edges
        }
        for i in range(len(frame_anchor_indices) - 1):
            ia = frame_anchor_indices[i]
            for j in range(i + 1, len(frame_anchor_indices)):
                ib = frame_anchor_indices[j]
                pair = tuple(sorted((ia, ib)))
                if pair in existing_pairs:
                    continue
                dx = points[ib].x - points[ia].x
                dy = points[ib].y - points[ia].y
                edges.append(LinkEdge(ia=ia, ib=ib, L0=math.hypot(dx, dy), is_shock=False))
                existing_pairs.add(pair)
    if len(frame_anchor_indices) >= 3:
        rest = [(points[i].x, points[i].y) for i in frame_anchor_indices]
        cx = sum(p[0] for p in rest) / len(rest)
        cy = sum(p[1] for p in rest) / len(rest)
        rigid_groups.append(
            {
                "indices": frame_anchor_indices,
                "rest": rest,
                "rest_cx": cx,
                "rest_cy": cy,
            }
        )

    axis_constraints: Dict[int, Dict[str, float]] = {}
    for point_id, constraint in (point_constraints or {}).items():
        idx = point_index_by_id.get(str(point_id))
        if idx is None:
            continue
        axis_constraints[idx] = {
            key: float(value)
            for key, value in constraint.items()
            if key in {"x", "y"} and value is not None
        }

    result = _solve_with_edges(
        points=points,
        edges=edges,
        fixed=fixed,
        rear_axle_idx=None,
        driver_edge_idx=driver_edge_idx,
        driver_L0=driver_L0,
        driver_stroke=0.0,
        rigid_groups=rigid_groups,
        n_steps=1,
        iterations=max(1, iterations),
        pre_steps=0,
        axis_constraints=axis_constraints,
    )

    solved_positions = result.steps[-1].points if result.steps else {}
    solved_points: List[BikePoint] = []
    max_constraint_error = 0.0
    for point in points:
        coords = solved_positions.get(point.id)
        if coords is None:
            solved_points.append(point)
            continue
        x_val = float(coords[0])
        y_val = float(coords[1])
        constraint = point_constraints.get(point.id) or {}
        if constraint.get("x") is not None:
            max_constraint_error = max(max_constraint_error, abs(x_val - float(constraint["x"])))
        if constraint.get("y") is not None:
            max_constraint_error = max(max_constraint_error, abs(y_val - float(constraint["y"])))
        solved_points.append(point.copy(update={"x": x_val, "y": y_val}))

    debug = dict(result.debug or {})
    debug.update(
        {
            "mode": "rest_pose",
            "point_constraints": point_constraints,
            "max_constraint_error_px": max_constraint_error,
        }
    )
    return solved_points, debug
