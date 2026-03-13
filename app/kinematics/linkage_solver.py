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
    x0 = [p.x for p in points]
    y0 = [p.y for p in points]

    def _apply_axis_constraints(state_x: List[float], state_y: List[float]) -> None:
        if not axis_constraints:
            return
        for idx, constraint in axis_constraints.items():
            if idx < 0 or idx >= len(state_x):
                continue
            target_x = constraint.get("x")
            target_y = constraint.get("y")
            if target_x is not None:
                state_x[idx] = float(target_x)
            if target_y is not None:
                state_y[idx] = float(target_y)

    def _project_distance_constraints(
        state_x: List[float],
        state_y: List[float],
        target_shock_len: float,
    ) -> None:
        for ei, edge in enumerate(edges):
            ia, ib = edge.ia, edge.ib

            if ei == driver_edge_idx and edge.is_shock:
                target_L = max(1e-6, target_shock_len)
            else:
                target_L = edge.L0

            dx = state_x[ib] - state_x[ia]
            dy = state_y[ib] - state_y[ia]
            dist = math.hypot(dx, dy) or 1e-9
            diff = (dist - target_L) / dist

            fa = fixed[ia]
            fb = fixed[ib]

            if fa and fb:
                continue
            elif fa and not fb:
                state_x[ib] -= dx * diff
                state_y[ib] -= dy * diff
            elif fb and not fa:
                state_x[ia] += dx * diff
                state_y[ia] += dy * diff
            else:
                half = 0.5
                state_x[ia] += dx * diff * half
                state_y[ia] += dy * diff * half
                state_x[ib] -= dx * diff * half
                state_y[ib] -= dy * diff * half

    def _solve_state_to_stroke(state_x: List[float], state_y: List[float], s: float) -> None:
        target_shock_len = driver_L0 - s  # shorten shock with positive stroke

        for _ in range(iterations):
            _project_distance_constraints(state_x, state_y, target_shock_len)

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

                cur_cx = sum(state_x[i] for i in indices) / len(indices)
                cur_cy = sum(state_y[i] for i in indices) / len(indices)

                a = 0.0
                b = 0.0
                for (rx, ry), i in zip(rest, indices):
                    qx = rx - rest_cx
                    qy = ry - rest_cy
                    px = state_x[i] - cur_cx
                    py = state_y[i] - cur_cy
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
                    state_x[i] = cur_cx + tx
                    state_y[i] = cur_cy + ty

            _apply_axis_constraints(state_x, state_y)

        # Final length projection after the last rigid-group solve keeps the
        # recorded pose on the target shock eye-to-eye length.
        for _ in range(3):
            _project_distance_constraints(state_x, state_y, target_shock_len)
            _apply_axis_constraints(state_x, state_y)

    def _build_step_from_state(
        state_x: List[float],
        state_y: List[float],
        *,
        step_index: int,
        shock_stroke: float,
        previous_step: Optional[SolverStep] = None,
    ) -> SolverStep:
        rear_travel: Optional[float] = None
        if rear_axle_idx is not None and rear_y0 is not None:
            rear_travel = rear_y0 - state_y[rear_axle_idx]

        driver_edge = edges[driver_edge_idx]
        ia, ib = driver_edge.ia, driver_edge.ib
        dx = state_x[ib] - state_x[ia]
        dy = state_y[ib] - state_y[ia]
        shock_len = math.hypot(dx, dy)

        positions = {p.id: (state_x[i], state_y[i]) for i, p in enumerate(points)}

        leverage: Optional[float] = None
        if previous_step is not None and rear_travel is not None and previous_step.rear_travel is not None:
            ds = shock_stroke - previous_step.shock_stroke
            if abs(ds) > 1e-9:
                dr = rear_travel - previous_step.rear_travel
                leverage = dr / ds

        return SolverStep(
            step_index=step_index,
            shock_stroke=shock_stroke,
            shock_length=shock_len,
            rear_travel=rear_travel,
            leverage_ratio=leverage,
            points=positions,
        )

    # Rear axle initial vertical position
    rear_y0 = y0[rear_axle_idx] if rear_axle_idx is not None else None

    steps: List[SolverStep] = []
    n_steps = max(1, n_steps)
    iterations = max(1, iterations)

    step_stroke = driver_stroke / n_steps
    pre_roll_len = step_stroke * max(0, pre_steps)

    pos_x = x0.copy()
    pos_y = y0.copy()
    _apply_axis_constraints(pos_x, pos_y)
    zero_step = _build_step_from_state(
        pos_x,
        pos_y,
        step_index=0,
        shock_stroke=0.0,
    )
    steps.append(zero_step)
    previous_step = zero_step
    for step_i in range(1, n_steps + 1):
        s = driver_stroke * (step_i / n_steps)
        _solve_state_to_stroke(pos_x, pos_y, s)
        step = _build_step_from_state(
            pos_x,
            pos_y,
            step_index=step_i,
            shock_stroke=s,
            previous_step=previous_step,
        )
        steps.append(step)
        previous_step = step

    full_steps = steps
    if pre_steps > 0:
        neg_x = x0.copy()
        neg_y = y0.copy()
        _apply_axis_constraints(neg_x, neg_y)
        neg_steps_return: List[SolverStep] = []
        neg_previous = None

        # Pre-roll by extending to the most negative stroke first, then walk
        # back toward zero. This keeps the samples nearest L0 warm-started and
        # restores the smoother endpoint gradient behaviour.
        if pre_roll_len > 0:
            _solve_state_to_stroke(neg_x, neg_y, -pre_roll_len)

        for step_i in range(pre_steps, 0, -1):
            s = -(step_stroke * step_i)
            if step_i != pre_steps:
                _solve_state_to_stroke(neg_x, neg_y, s)
            step = _build_step_from_state(
                neg_x,
                neg_y,
                step_index=(pre_steps - step_i),
                shock_stroke=s,
                previous_step=neg_previous,
            )
            neg_steps_return.append(step)
            neg_previous = step

        full_steps = []
        ordered_full = neg_steps_return + steps
        for step_index, step in enumerate(ordered_full):
            full_steps.append(step.copy(update={"step_index": step_index}))

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
    point_by_id = {str(point.id): point for point in points if str(point.id)}
    if not point_by_id:
        return [], {
            "mode": "rest_pose",
            "point_constraints": point_constraints,
            "max_constraint_error_px": 0.0,
            "max_body_fit_error_px": 0.0,
            "iterations_used": 0,
            "body_count": 0,
        }

    constraints_by_id: Dict[str, Dict[str, float]] = {}
    for point_id, constraint in (point_constraints or {}).items():
        point_id_str = str(point_id or "").strip()
        if not point_id_str or point_id_str not in point_by_id or not isinstance(constraint, dict):
            continue
        normalized = {
            key: float(value)
            for key, value in constraint.items()
            if key in {"x", "y"} and value is not None
        }
        if normalized:
            constraints_by_id[point_id_str] = normalized

    body_models: List[Dict[str, object]] = []

    def _append_body_model(body_id: str, point_ids: List[str], ref_positions: List[Tuple[float, float]], body_type: str) -> None:
        normalized_ids = [str(pid or "").strip() for pid in point_ids if str(pid or "").strip()]
        if len(normalized_ids) < 2 or len(normalized_ids) != len(ref_positions):
            return
        body_models.append(
            {
                "id": body_id,
                "type": body_type,
                "point_ids": normalized_ids,
                "ref_positions": [(float(x), float(y)) for x, y in ref_positions],
            }
        )

    for body in bodies or []:
        point_ids = [str(pid or "").strip() for pid in (getattr(body, "point_ids", None) or []) if str(pid or "").strip()]
        if len(point_ids) < 2:
            continue
        ref_positions = []
        missing_point = False
        for point_id in point_ids:
            point = point_by_id.get(point_id)
            if point is None:
                missing_point = True
                break
            ref_positions.append((float(point.x), float(point.y)))
        if missing_point:
            continue

        body_type = str(getattr(body, "type", "") or "").strip().lower()
        if body_type == "shock" and len(ref_positions) >= 2:
            base_ax, base_ay = ref_positions[0]
            base_bx, base_by = ref_positions[1]
            shock_length = getattr(body, "length0", None)
            if shock_length is not None:
                dx = base_bx - base_ax
                dy = base_by - base_ay
                dist = math.hypot(dx, dy)
                if dist > 1e-9:
                    ux = dx / dist
                    uy = dy / dist
                else:
                    ux, uy = 1.0, 0.0
                normalized_bx = base_ax + ux * float(shock_length)
                normalized_by = base_ay + uy * float(shock_length)
                offset_x = normalized_bx - base_bx
                offset_y = normalized_by - base_by
                adjusted_positions = [(base_ax, base_ay), (normalized_bx, normalized_by)]
                for extra_x, extra_y in ref_positions[2:]:
                    adjusted_positions.append((extra_x + offset_x, extra_y + offset_y))
                ref_positions = adjusted_positions

        _append_body_model(
            str(getattr(body, "id", "") or f"body_{len(body_models) + 1}"),
            point_ids,
            ref_positions,
            body_type,
        )

    fixed_body_point_ids = [
        str(point.id)
        for point in points
        if str(getattr(point, "type", "") or "").strip().lower() in {"fixed", "bb"}
    ]
    if len(fixed_body_point_ids) >= 2:
        ref_positions = [
            (float(point_by_id[point_id].x), float(point_by_id[point_id].y))
            for point_id in fixed_body_point_ids
            if point_id in point_by_id
        ]
        if len(ref_positions) == len(fixed_body_point_ids):
            _append_body_model("__fixed_frame__", fixed_body_point_ids, ref_positions, "fixed_frame")

    current_positions: Dict[str, Tuple[float, float]] = {
        point_id: (float(point.x), float(point.y))
        for point_id, point in point_by_id.items()
    }
    for point_id, constraint in constraints_by_id.items():
        cur_x, cur_y = current_positions[point_id]
        current_positions[point_id] = (
            float(constraint.get("x", cur_x)),
            float(constraint.get("y", cur_y)),
        )

    def _fit_body(ref_positions: List[Tuple[float, float]], target_positions: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        if len(ref_positions) != len(target_positions) or not ref_positions:
            return 1.0, 0.0, 0.0, 0.0
        if len(ref_positions) == 1:
            qx, qy = ref_positions[0]
            px, py = target_positions[0]
            return 1.0, 0.0, px - qx, py - qy

        ref_cx = sum(x for x, _ in ref_positions) / len(ref_positions)
        ref_cy = sum(y for _, y in ref_positions) / len(ref_positions)
        target_cx = sum(x for x, _ in target_positions) / len(target_positions)
        target_cy = sum(y for _, y in target_positions) / len(target_positions)

        a = 0.0
        b = 0.0
        for (qx, qy), (px, py) in zip(ref_positions, target_positions):
            qcx = qx - ref_cx
            qcy = qy - ref_cy
            pcx = px - target_cx
            pcy = py - target_cy
            a += pcx * qcx + pcy * qcy
            b += pcy * qcx - pcx * qcy

        denom = math.hypot(a, b)
        if denom < 1e-9:
            cos_t = 1.0
            sin_t = 0.0
        else:
            cos_t = a / denom
            sin_t = b / denom

        tx = target_cx - (cos_t * ref_cx - sin_t * ref_cy)
        ty = target_cy - (sin_t * ref_cx + cos_t * ref_cy)
        return cos_t, sin_t, tx, ty

    def _apply_body_transform(
        ref_positions: List[Tuple[float, float]],
        transform: Tuple[float, float, float, float],
    ) -> List[Tuple[float, float]]:
        cos_t, sin_t, tx, ty = transform
        out: List[Tuple[float, float]] = []
        for x_val, y_val in ref_positions:
            out.append(
                (
                    cos_t * x_val - sin_t * y_val + tx,
                    sin_t * x_val + cos_t * y_val + ty,
                )
            )
        return out

    iterations_used = 0
    max_body_fit_error = 0.0
    max_step_delta = 0.0
    for _ in range(max(1, iterations)):
        iterations_used += 1
        predicted_by_point: Dict[str, List[Tuple[float, float]]] = {}
        max_body_fit_error = 0.0

        for body_model in body_models:
            body_point_ids = body_model["point_ids"]  # type: ignore[assignment]
            ref_positions = body_model["ref_positions"]  # type: ignore[assignment]
            target_positions = [current_positions[point_id] for point_id in body_point_ids]
            transform = _fit_body(ref_positions, target_positions)
            predicted_positions = _apply_body_transform(ref_positions, transform)
            for point_id, target_position, predicted_position in zip(body_point_ids, target_positions, predicted_positions):
                predicted_by_point.setdefault(point_id, []).append(predicted_position)
                dx = predicted_position[0] - target_position[0]
                dy = predicted_position[1] - target_position[1]
                max_body_fit_error = max(max_body_fit_error, math.hypot(dx, dy))

        next_positions: Dict[str, Tuple[float, float]] = {}
        max_step_delta = 0.0
        for point_id, current_position in current_positions.items():
            predictions = predicted_by_point.get(point_id)
            if predictions:
                avg_x = sum(x for x, _ in predictions) / len(predictions)
                avg_y = sum(y for _, y in predictions) / len(predictions)
            else:
                avg_x, avg_y = current_position

            constraint = constraints_by_id.get(point_id)
            if constraint:
                if constraint.get("x") is not None:
                    avg_x = float(constraint["x"])
                if constraint.get("y") is not None:
                    avg_y = float(constraint["y"])

            dx = avg_x - current_position[0]
            dy = avg_y - current_position[1]
            max_step_delta = max(max_step_delta, math.hypot(dx, dy))
            next_positions[point_id] = (avg_x, avg_y)

        current_positions = next_positions
        if max_step_delta <= 1e-6 and max_body_fit_error <= 1e-6:
            break

    solved_points: List[BikePoint] = []
    max_constraint_error = 0.0
    for point in points:
        coords = current_positions.get(point.id)
        if coords is None:
            solved_points.append(point)
            continue
        x_val = float(coords[0])
        y_val = float(coords[1])
        constraint = constraints_by_id.get(point.id) or {}
        if constraint.get("x") is not None:
            max_constraint_error = max(max_constraint_error, abs(x_val - float(constraint["x"])))
        if constraint.get("y") is not None:
            max_constraint_error = max(max_constraint_error, abs(y_val - float(constraint["y"])))
        solved_points.append(point.copy(update={"x": x_val, "y": y_val}))

    debug = {
        "mode": "rest_pose",
        "solver": "rigid_body_projection_v1",
        "point_constraints": constraints_by_id,
        "max_constraint_error_px": max_constraint_error,
        "max_body_fit_error_px": max_body_fit_error,
        "max_step_delta_px": max_step_delta,
        "iterations_used": iterations_used,
        "body_count": len(body_models),
        "body_ids": [str(body_model.get("id") or "") for body_model in body_models],
    }
    return solved_points, debug
