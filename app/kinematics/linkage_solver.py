# app/kinematics/linkage_solver.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import math
from pydantic import BaseModel

from app.schemas import BikePoint, RigidBody   # reuse your existing models


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
    shock_stroke: float            # "units" of shock stroke at this step (usually mm)
    shock_length: float            # eye-to-eye at this step (same units)
    rear_travel: Optional[float]   # vertical rear axle travel vs initial (same units)
    leverage_ratio: Optional[float]  # d(rear_travel)/d(shock_stroke)
    points: Dict[str, Tuple[float, float]]  # point.id -> (x, y)


class SolverResult(BaseModel):
    """
    Full sweep from zero to full shock stroke.
    """
    steps: List[SolverStep]
    rear_axle_point_id: Optional[str]


# ------------ Converter: BikePoint + RigidBody → edges + flags ------------

def _build_internal_model(
    points: List[BikePoint],
    bodies: List[RigidBody],
) -> tuple[
    List[LinkEdge],
    List[bool],         # fixed flags per point index
    Optional[int],      # rear_axle_idx
    int,                # driver_edge_idx
    float,              # driver_L0
    float,              # driver_stroke
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

    # Fixed points: type == "fixed" or "bb"
    fixed = [p.type in ("fixed", "bb") for p in points]

    # Rear axle index (optional)
    rear_axle_idx = next(
        (i for i, p in enumerate(points) if p.type == "rear_axle"),
        None,
    )

    edges: List[LinkEdge] = []
    driver_edge_idx: Optional[int] = None
    driver_L0: Optional[float] = None
    driver_stroke: Optional[float] = None

    for body in bodies:
        pids = body.point_ids or []
        if len(pids) < 2:
            continue

        # Mark frame clusters as fixed (extra safety)
        if body.type == "fixed":
            for pid in pids:
                if pid not in idx:
                    raise ValueError(f"RigidBody {body.id!r} references unknown point {pid!r}.")
                fixed[idx[pid]] = True

        # Build segments (start/end) from ordered point_ids
        seg_pairs: List[Tuple[str, str]] = list(zip(pids, pids[1:]))
        if body.closed and len(pids) > 2:
            seg_pairs.append((pids[-1], pids[0]))

        for a_id, b_id in seg_pairs:
            if a_id not in idx or b_id not in idx:
                raise ValueError(f"RigidBody {body.id!r} references unknown point.")
            ia, ib = idx[a_id], idx[b_id]
            dx = x0[ib] - x0[ia]
            dy = y0[ib] - y0[ia]
            L0_geom = math.hypot(dx, dy)

            if body.type == "shock":
                # Shock body → this segment is the driver shock
                L0 = body.length0 if body.length0 is not None else L0_geom
                if body.stroke is None:
                    raise ValueError(f"Shock body {body.id!r} must define 'stroke'.")
                edge = LinkEdge(ia=ia, ib=ib, L0=L0, is_shock=True)
                edges.append(edge)

                if driver_edge_idx is not None:
                    raise ValueError("Multiple shock bodies found; only one driver is supported.")
                driver_edge_idx = len(edges) - 1
                driver_L0 = L0
                driver_stroke = body.stroke
            else:
                # Normal bar (moving) or fixed body segment
                edges.append(LinkEdge(ia=ia, ib=ib, L0=L0_geom, is_shock=False))

    if driver_edge_idx is None or driver_L0 is None or driver_stroke is None:
        raise ValueError("No shock body found; need exactly one RigidBody with type='shock'.")

    return edges, fixed, rear_axle_idx, driver_edge_idx, driver_L0, driver_stroke


# ------------ Core PBD solver ------------

def _solve_with_edges(
    points: List[BikePoint],
    edges: List[LinkEdge],
    fixed: List[bool],
    rear_axle_idx: Optional[int],
    driver_edge_idx: int,
    driver_L0: float,
    driver_stroke: float,
    n_steps: int,
    iterations: int,
) -> SolverResult:
    """
    Position-based distance-constraint solver for the linkage.

    - Treats each edge as a distance constraint.
    - For the driver shock edge, target length = L0 - stroke_fraction.
    """
    n = len(points)
    x = [p.x for p in points]
    y = [p.y for p in points]

    # Rear axle initial vertical position
    rear_y0 = y[rear_axle_idx] if rear_axle_idx is not None else None

    steps: List[SolverStep] = []

    n_steps = max(1, n_steps)
    iterations = max(1, iterations)

    for step_i in range(n_steps + 1):
        # Shock stroke used at this step (0 → full)
        s = driver_stroke * (step_i / n_steps)
        target_shock_len = driver_L0 - s  # shorten shock with positive stroke

        # Iterative constraint projection
        for _ in range(iterations):
            for ei, edge in enumerate(edges):
                ia, ib = edge.ia, edge.ib

                # Target length for this constraint
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
                    # Move only B
                    x[ib] -= dx * diff
                    y[ib] -= dy * diff
                elif fb and not fa:
                    # Move only A
                    x[ia] += dx * diff
                    y[ia] += dy * diff
                else:
                    # Both free → split correction
                    half = 0.5
                    x[ia] += dx * diff * half
                    y[ia] += dy * diff * half
                    x[ib] -= dx * diff * half
                    y[ib] -= dy * diff * half

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

    rear_axle_id = points[rear_axle_idx].id if rear_axle_idx is not None else None
    return SolverResult(steps=steps, rear_axle_point_id=rear_axle_id)


# ------------ Public API ------------

def solve_bike_linkage(
    points: List[BikePoint],
    bodies: List[RigidBody],
    n_steps: int = 80,
    iterations: int = 100,
) -> SolverResult:
    """
    Public entrypoint used by your router.

    Expects:
      - BikePoint.type ∈ {"bb","rear_axle","front_axle","free","fixed"}
      - Exactly one RigidBody with type="shock", with 2 point_ids and stroke set.

    Returns:
      SolverResult with positions, rear_travel, leverage vs shock stroke.
    """
    (
        edges,
        fixed,
        rear_axle_idx,
        driver_edge_idx,
        driver_L0,
        driver_stroke,
    ) = _build_internal_model(points, bodies)

    return _solve_with_edges(
        points=points,
        edges=edges,
        fixed=fixed,
        rear_axle_idx=rear_axle_idx,
        driver_edge_idx=driver_edge_idx,
        driver_L0=driver_L0,
        driver_stroke=driver_stroke,
        n_steps=n_steps,
        iterations=iterations,
    )