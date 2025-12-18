# app/routers/bikes.py
import math
from datetime import datetime
from typing import Optional, List
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Response
from pydantic import BaseModel
from bson import ObjectId
from app.schemas import (
    BikeCreate,
    BikePoint,
    BikePointsUpdate,
    RigidBody,
    BikeBodiesOut,
    BikeBodiesUpdate,
    RearCenterUpdate,
    ScaleSourceUpdate,
    WheelbaseUpdate,
    BikeOut,
    BikeGeometry,
    BikeKinematics,
)
from app.kinematics.linkage_solver import solve_bike_linkage, SolverResult

from app.db import bikes_col#, media_items_col
# from app.storage import generate_signed_url
from .auth import get_current_user
from app.utils_media import resolve_hero_url

router = APIRouter(prefix="/bikes", tags=["bikes"])

def bike_doc_to_out(doc, hero_url: Optional[str] = None) -> BikeOut:
    # normalise points if present
    raw_points = doc.get("points") or []
    points: list[BikePoint] = []
    for p in raw_points:
        try:
            points.append(BikePoint(**p))
        except Exception as exc:
            logging.warning("Skipping invalid point on bike %s: %r (%s)", doc.get("_id"), p, exc)
    
    raw_bodies = doc.get("bodies") or []
    bodies: list[RigidBody] = []
    for b in raw_bodies:
        try:
            bodies.append(RigidBody(**b))
        except Exception:
            continue

    # 👇 NEW: geometry block
    geometry_raw = doc.get("geometry") or {}
    geometry: Optional[BikeGeometry] = None
    if geometry_raw:
        try:
            geometry = BikeGeometry(**geometry_raw)
        except Exception as exc:
            logging.warning(
                "Skipping invalid geometry on bike %s: %r (%s)",
                doc.get("_id"), geometry_raw, exc
            )

    # 👇 NEW: kinematics block
    kin_raw = doc.get("kinematics") or {}
    kinematics: Optional[BikeKinematics] = None
    if kin_raw:
        try:
            kinematics = BikeKinematics(**kin_raw)
        except Exception as exc:
            logging.warning(
                "Skipping invalid kinematics on bike %s: %r (%s)",
                doc.get("_id"), kin_raw, exc
            )

    return BikeOut(
        id=str(doc["_id"]),
        name=doc["name"],
        brand=doc["brand"],
        model_year=doc.get("model_year"),
        # Avoid "None" string if user_id is missing on old docs
        user_id=str(doc["user_id"]) if doc.get("user_id") is not None else "",
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        hero_media_id=(str(doc["hero_media_id"]) if doc.get("hero_media_id") else None),
        # prefer the explicit hero_url passed in, fall back to any stored value
        hero_url=hero_url if hero_url is not None else doc.get("hero_url"),
        points=points or None,
        bodies=bodies or None,
        geometry=geometry,
        kinematics=kinematics, 
    )


def _extract_user_oid(current_user) -> ObjectId:
    """Helper to robustly extract a Mongo ObjectId for the current user."""
    raw_id = None

    # If get_current_user returned a dict-like object
    if isinstance(current_user, dict):
        raw_id = current_user.get("id") or current_user.get("_id")
    else:
        # Pydantic model / object with attributes
        raw_id = getattr(current_user, "id", None) or getattr(current_user, "_id", None)

    if raw_id is None:
        # Fail loudly instead of silently writing user_id=None
        print("DEBUG: current_user has no id:", repr(current_user))
        raise HTTPException(status_code=500, detail="Could not determine current user id")

    if isinstance(raw_id, str):
        try:
            return ObjectId(raw_id)
        except Exception:
            # If id is already e.g. a stringified ObjectId you still want it in Mongo
            # but for safety we keep it as ObjectId when possible.
            print("DEBUG: failed to convert raw_id to ObjectId:", raw_id)
            raise HTTPException(status_code=500, detail="Invalid current user id format")

    # raw_id is already an ObjectId or similar
    return raw_id


@router.post("", response_model=BikeOut, status_code=status.HTTP_201_CREATED)
async def create_bike(
    bike_in: BikeCreate,
    current_user=Depends(get_current_user),
):
    now = datetime.utcnow()

    user_oid = _extract_user_oid(current_user)

    doc = {
        "user_id": user_oid,
        "name": bike_in.name,
        "brand": bike_in.brand,
        "model_year": bike_in.model_year,
        "created_at": now,
        "updated_at": now,
    }
    bikes = bikes_col()
    result = await bikes.insert_one(doc)
    doc["_id"] = result.inserted_id
    return bike_doc_to_out(doc)


@router.get("", response_model=List[BikeOut])
async def list_my_bikes(current_user=Depends(get_current_user)):
    user_oid = _extract_user_oid(current_user)
    bikes = bikes_col()
    cursor = bikes.find({"user_id": user_oid})
    docs = await cursor.to_list(length=1000)

    out: list[BikeOut] = []
    for d in docs:
        hero_id = d.get("hero_media_id")
        hero_url = await resolve_hero_url(hero_id)
        out.append(bike_doc_to_out(d, hero_url=hero_url))

    return out


@router.get("/{bike_id}", response_model=BikeOut)
async def get_bike(
    bike_id: str,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()
    doc = await bikes.find_one({"_id": oid, "user_id": user_oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Bike not found")

    hero_id = doc.get("hero_media_id")
    hero_url = await resolve_hero_url(hero_id)

    return bike_doc_to_out(doc, hero_url=hero_url)

def _ensure_unique_point_ids(points: list[BikePoint]) -> None:
    ids = [p.id for p in points if p.id]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise HTTPException(
            status_code=400,
            detail=f"Duplicate point ids: {sorted(dupes)}",
        )
    
@router.put("/{bike_id}/points", response_model=BikeOut)
async def update_bike_points(
    bike_id: str,
    payload: BikePointsUpdate,
    current_user=Depends(get_current_user),
):
    """Update the annotated geometry points for a bike."""
    user_oid = _extract_user_oid(current_user)

    _ensure_unique_point_ids(payload.points)

    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()
    doc = await bikes.find_one({"_id": oid, "user_id": user_oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Bike not found")

    now = datetime.utcnow()

    await bikes.update_one(
        {"_id": oid, "user_id": user_oid},
        {
            "$set": {
                "points": [p.dict() for p in payload.points],
                "updated_at": now,
            }
        },
    )

    updated = await bikes.find_one({"_id": oid, "user_id": user_oid})
    hero_id = updated.get("hero_media_id")
    hero_url = await resolve_hero_url(hero_id)
    return bike_doc_to_out(updated, hero_url=hero_url)


@router.get("/{bike_id}/bodies", response_model=BikeBodiesOut)
async def get_bodies(
    bike_id: str,
    current_user=Depends(get_current_user),
):
    """
    Return all rigid bodies for a given bike.

    Data is stored inside the bike document as doc["bodies"] = [...]
    but exposed via its own endpoint so we don't touch the points logic.
    """
    user_oid = _extract_user_oid(current_user)

    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()
    doc = await bikes.find_one(
        {"_id": oid, "user_id": user_oid},
        {"bodies": 1, "_id": 0},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Bike not found")

    raw_bodies = doc.get("bodies") or []
    bodies: List[RigidBody] = []
    for b in raw_bodies:
        try:
            bodies.append(RigidBody(**b))
        except Exception:
            # Skip malformed entries rather than crashing older docs
            continue

    return BikeBodiesOut(bodies=bodies)

def _ensure_unique_body_ids(bodies):
    ids = [b.id for b in bodies if b.id]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise HTTPException(
            status_code=400,
            detail=f"Duplicate body ids: {sorted(dupes)}",
        )
    
@router.put("/{bike_id}/bodies", response_model=BikeBodiesOut, status_code=status.HTTP_200_OK)
async def update_bodies(
    bike_id: str,
    payload: BikeBodiesUpdate,
    current_user=Depends(get_current_user),
):
    """
    Replace the bodies list for this bike.

    This DOES NOT touch points or anything else on the bike document.
    """
    user_oid = _extract_user_oid(current_user)

    _ensure_unique_body_ids(payload.bodies)

    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()

    result = await bikes.update_one(
        {"_id": oid, "user_id": user_oid},
        {
            "$set": {
                "bodies": [b.dict() for b in payload.bodies],
                "updated_at": datetime.utcnow(),
            }
        },
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Bike not found")

    # Echo back what we just stored
    return BikeBodiesOut(bodies=payload.bodies)


@router.put("/{bike_id}/rear_center", response_model=BikeOut)
async def update_rear_center(
    bike_id: str,
    payload: RearCenterUpdate,
    current_user=Depends(get_current_user),
):
    """Set rear-centre [mm] and compute/store a scale factor (mm per px)."""
    user_oid = _extract_user_oid(current_user)

    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()

    bike = await bikes.find_one({"_id": oid, "user_id": user_oid})
    if not bike:
        raise HTTPException(status_code=404, detail="Bike not found")

    points = bike.get("points", []) or []
    bb = next((p for p in points if p.get("type") in ("bb", "bottom_bracket")), None)
    rear_axle = next((p for p in points if p.get("type") == "rear_axle"), None)

    if not bb or not rear_axle:
        raise HTTPException(
            status_code=400,
            detail="Cannot compute scale: need BB and rear_axle points",
        )

    dx = rear_axle["x"] - bb["x"]
    dy = rear_axle["y"] - bb["y"]
    d_px = math.hypot(dx, dy)
    if d_px <= 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot compute scale: BB and rear axle coincide",
        )

    rear_center_mm = float(payload.rear_center_mm)
    scale_mm_per_px = rear_center_mm / d_px

    await bikes.update_one(
        {"_id": oid, "user_id": user_oid},
        {"$set": {
            "geometry.rear_center_mm": rear_center_mm,
            "geometry.scale_mm_per_px": scale_mm_per_px,
            "geometry.scale_source": "rear_center",
            "updated_at": datetime.utcnow(),
        }},
    )

    updated = await bikes.find_one({"_id": oid, "user_id": user_oid})
    hero_id = updated.get("hero_media_id")
    hero_url = await resolve_hero_url(hero_id)
    return bike_doc_to_out(updated, hero_url=hero_url)



@router.put("/{bike_id}/front_center", response_model=BikeOut)
async def update_front_center(
    bike_id: str,
    payload: FrontCenterUpdate,
    current_user=Depends(get_current_user),
):
    """Set front-centre [mm] and compute/store a scale factor (mm per px)."""
    user_oid = _extract_user_oid(current_user)

    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()

    bike = await bikes.find_one({"_id": oid, "user_id": user_oid})
    if not bike:
        raise HTTPException(status_code=404, detail="Bike not found")

    points = bike.get("points", []) or []
    bb = next((p for p in points if p.get("type") in ("bb", "bottom_bracket")), None)
    front_axle = next((p for p in points if p.get("type") == "front_axle"), None)

    if not bb or not front_axle:
        raise HTTPException(
            status_code=400,
            detail="Cannot compute scale: need BB and front_axle points",
        )

    dx = front_axle["x"] - bb["x"]
    dy = front_axle["y"] - bb["y"]
    d_px = math.hypot(dx, dy)
    if d_px <= 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot compute scale: BB and front axle coincide",
        )

    front_center_mm = float(payload.front_center_mm)
    scale_mm_per_px = front_center_mm / d_px

    await bikes.update_one(
        {"_id": oid, "user_id": user_oid},
        {"$set": {
            "geometry.front_center_mm": front_center_mm,
            "geometry.scale_mm_per_px": scale_mm_per_px,
            "geometry.scale_source": "front_center",
            "updated_at": datetime.utcnow(),
        }},
    )

    updated = await bikes.find_one({"_id": oid, "user_id": user_oid})
    hero_id = updated.get("hero_media_id")
    hero_url = await resolve_hero_url(hero_id)
    return bike_doc_to_out(updated, hero_url=hero_url)


@router.put("/{bike_id}/wheelbase", response_model=BikeOut)
async def update_wheelbase(
    bike_id: str,
    payload: WheelbaseUpdate,
    current_user=Depends(get_current_user),
):
    """Set wheelbase [mm] and compute/store scale factor (mm per px)."""
    user_oid = _extract_user_oid(current_user)

    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()

    bike = await bikes.find_one({"_id": oid, "user_id": user_oid})
    if not bike:
        raise HTTPException(status_code=404, detail="Bike not found")

    points = bike.get("points", []) or []
    rear_axle = next((p for p in points if p.get("type") == "rear_axle"), None)
    front_axle = next((p for p in points if p.get("type") == "front_axle"), None)

    if not rear_axle or not front_axle:
        raise HTTPException(
            status_code=400,
            detail="Cannot compute scale: need rear_axle and front_axle points",
        )

    dx = front_axle["x"] - rear_axle["x"]
    dy = front_axle["y"] - rear_axle["y"]
    d_px = math.hypot(dx, dy)
    if d_px <= 0:
        raise HTTPException(status_code=400, detail="Cannot compute scale: axles coincide")

    wheelbase_mm = float(payload.wheelbase_mm)
    scale_mm_per_px = wheelbase_mm / d_px

    await bikes.update_one(
        {"_id": oid, "user_id": user_oid},
        {"$set": {
            "geometry.wheelbase_mm": wheelbase_mm,
            "geometry.scale_mm_per_px": scale_mm_per_px,
            "geometry.scale_source": "wheelbase",
            "updated_at": datetime.utcnow(),
        }},
    )

    updated = await bikes.find_one({"_id": oid, "user_id": user_oid})
    hero_id = updated.get("hero_media_id")
    hero_url = await resolve_hero_url(hero_id)
    return bike_doc_to_out(updated, hero_url=hero_url)


@router.put("/{bike_id}/scale_source", response_model=BikeOut)
async def update_scale_source(
    bike_id: str,
    payload: ScaleSourceUpdate,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()
    bike = await bikes.find_one({"_id": oid, "user_id": user_oid})
    if not bike:
        raise HTTPException(status_code=404, detail="Bike not found")

    await bikes.update_one(
        {"_id": oid, "user_id": user_oid},
        {"$set": {
            "geometry.scale_source": payload.scale_source,
            "updated_at": datetime.utcnow(),
        }},
    )

    updated = await bikes.find_one({"_id": oid, "user_id": user_oid})
    hero_id = updated.get("hero_media_id")
    hero_url = await resolve_hero_url(hero_id)
    return bike_doc_to_out(updated, hero_url=hero_url)


@router.get("/{bike_id}/kinematics", response_model=SolverResult)
async def compute_bike_kinematics(
    bike_id: str,
    steps: int = 80,
    iterations: int = 100,
    current_user=Depends(get_current_user),
):
    """
    Run the 2D linkage solver for this bike.

    Side effects:
    - Writes `coords` onto each point in doc["points"] so that coords[i]
      matches kinematics step i (coords are in image pixels).
    - Writes a compact `kinematics` summary (per-step stroke, travel, leverage),
      all in mm.
    """
    user_oid = _extract_user_oid(current_user)

    # Parse bike_id → ObjectId
    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()
    doc = await bikes.find_one({"_id": oid, "user_id": user_oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Bike not found")

    # ---- Extract / validate points ----
    raw_points = doc.get("points") or []
    points: List[BikePoint] = []
    for p in raw_points:
        try:
            points.append(BikePoint(**p))
        except Exception as exc:
            logging.warning(
                "Skipping invalid point on bike %s: %r (%s)",
                doc.get("_id"), p, exc
            )
    if not points:
        raise HTTPException(status_code=400, detail="Bike has no valid points defined")

    # ---- Extract / validate bodies ----
    raw_bodies = doc.get("bodies") or []
    bodies: List[RigidBody] = []
    for b in raw_bodies:
        try:
            bodies.append(RigidBody(**b))
        except Exception as exc:
            logging.warning(
                "Skipping invalid body on bike %s: %r (%s)",
                doc.get("_id"), b, exc
            )
    if not bodies:
        raise HTTPException(status_code=400, detail="Bike has no rigid bodies defined")

    # ---- Get scale_mm_per_px from geometry (needed to convert stroke mm → px) ----
    geom = doc.get("geometry") or {}
    raw_scale = geom.get("scale_mm_per_px")
    if raw_scale is None:
        raise HTTPException(
            status_code=400,
            detail="Cannot run kinematics: rear_center / scale_mm_per_px not set.",
        )
    try:
        scale_mm_per_px = float(raw_scale)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Cannot run kinematics: invalid scale_mm_per_px in geometry.",
        )
    if scale_mm_per_px <= 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot run kinematics: scale_mm_per_px must be > 0.",
        )

    # ---- Convert shock stroke from mm → px for the solver ----
    bodies_for_solver: List[RigidBody] = []
    for b in bodies:
        if b.type == "shock" and b.stroke is not None:
            stroke_mm = float(b.stroke)
            stroke_px = stroke_mm / scale_mm_per_px
            # clone body with stroke in px
            b_px = b.copy(update={"stroke": stroke_px})
            bodies_for_solver.append(b_px)
        else:
            bodies_for_solver.append(b)

    # ---- Run solver (all geometry in px) ----
    try:
        result = solve_bike_linkage(
            points=points,
            bodies=bodies_for_solver,  # NOTE: stroke now in px
            n_steps=steps,
            iterations=iterations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # --------------------------------------------------------
    # 1) Scale stroke / length / travel to mm, keep coords in px
    # --------------------------------------------------------
    solver_steps = sorted(result.steps, key=lambda s: s.step_index)
    scale = scale_mm_per_px  # mm per px

    for s in solver_steps:
        # These came out of the solver in px
        s.shock_stroke = s.shock_stroke * scale      # → mm
        s.shock_length = s.shock_length * scale      # → mm
        if s.rear_travel is not None:
            s.rear_travel = s.rear_travel * scale    # → mm

    # --------------------------------------------------------
    # 2) Build coords per point_id: coords[i] = (x,y) at step i (STILL px)
    # --------------------------------------------------------
    coords_map: dict[str, list[dict]] = {}
    for step in solver_steps:
        for pid, (x, y) in step.points.items():
            coords_map.setdefault(pid, []).append({"x": float(x), "y": float(y)})

    # Rebuild points with coords attached (coords are image pixels)
    new_points: list[dict] = []
    for p in raw_points:
        pid = p.get("id")
        if not pid:
            continue
        p_copy = dict(p)
        p_copy["coords"] = coords_map.get(pid, [])
        new_points.append(p_copy)

    # --------------------------------------------------------
    # 3) Build compact kinematics summary (now in mm)
    # --------------------------------------------------------
    kin_steps: list[dict] = []
    for s in solver_steps:
        kin_steps.append(
            {
                "step_index": s.step_index,
                "shock_stroke": s.shock_stroke,      # mm
                "shock_length": s.shock_length,      # mm
                "rear_travel": s.rear_travel,        # mm
                "leverage_ratio": s.leverage_ratio,  # dimensionless
            }
        )

    kin_doc = {
        "rear_axle_point_id": result.rear_axle_point_id,
        "n_steps": len(solver_steps),
        # Already in mm after scaling above
        "driver_stroke": solver_steps[-1].shock_stroke if solver_steps else None,
        "steps": kin_steps,
    }

    # --------------------------------------------------------
    # 4) Persist into Mongo
    # --------------------------------------------------------
    await bikes.update_one(
        {"_id": oid, "user_id": user_oid},
        {
            "$set": {
                "points": new_points,
                "kinematics": kin_doc,
                "updated_at": datetime.utcnow(),
            }
        },
    )

    # Front-end gets the scaled result (mm stroke/travel, px coords)
    result.steps = solver_steps
    return result


@router.get("/{bike_id}/kinematics_cached", response_model=BikeKinematics)
async def get_cached_bike_kinematics(
    bike_id: str,
    current_user=Depends(get_current_user),
):
    """
    Return cached kinematics from the bike document (no solver run).
    This reads doc["kinematics"] written by /{bike_id}/kinematics.
    """
    user_oid = _extract_user_oid(current_user)

    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()
    doc = await bikes.find_one(
        {"_id": oid, "user_id": user_oid},
        {"kinematics": 1, "_id": 0},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Bike not found")

    kin = doc.get("kinematics")
    if not kin:
        # No cached run yet
        return BikeKinematicsOut(
            rear_axle_point_id=None,
            n_steps=0,
            driver_stroke=None,
            steps=[],
        )

    # Normalize a bit so the frontend always gets predictable fields
    return BikeKinematics(
        rear_axle_point_id=kin.get("rear_axle_point_id"),
        n_steps=kin.get("n_steps") or (len(kin.get("steps") or [])),
        driver_stroke=kin.get("driver_stroke"),
        steps=kin.get("steps") or [],
    )


@router.delete("/{bike_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_bike(
    bike_id: str,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)

    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()

    # Optional: fetch first if you want to delete media too
    doc = await bikes.find_one({"_id": oid, "user_id": user_oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Bike not found")

    # TODO later: delete associated media (hero_media_id etc.) if desired.

    await bikes.delete_one({"_id": oid, "user_id": user_oid})
    return Response(status_code=status.HTTP_204_NO_CONTENT)