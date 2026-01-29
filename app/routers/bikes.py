# app/routers/bikes.py
import math
import os
from datetime import datetime
from typing import Optional, List
import logging
import numpy as np

from fastapi import APIRouter, Depends, HTTPException, status, Response
from pydantic import BaseModel
from bson import ObjectId
from app.schemas import (
    BikeCreate,
    BikePoint,
    BikePointsUpdate,
    RigidBody,
    BikeBodiesOut,
    # BikeGeometryUpdate,
    BikeBodiesUpdate,
    # RearCenterUpdate,
    # FrontCenterUpdate,
    # ScaleSourceUpdate,
    # WheelbaseUpdate,
    BikeOut,
    BikeGeometry,
    BikeKinematics,
    RimEllipse,
    BikePageSettingsPayload,
    BikePageSettingsOut,
)
from app.kinematics.linkage_solver import solve_bike_linkage, SolverResult
from app.kinematics.homography import compute_homography_from_ellipses, apply_homography

from app.db import bikes_col, media_items_col, bike_page_settings_col
from app.storage import delete_media_prefix, GCS_BUCKET_NAME
# from app.storage import generate_signed_url
from .auth import get_current_user
from app.utils_media import resolve_hero_url, resolve_hero_variant_url
from app.settings import settings

router = APIRouter(prefix="/bikes", tags=["bikes"])

def bike_doc_to_out(
    doc,
    hero_url: Optional[str] = None,
    hero_thumb_url: Optional[str] = None,
    hero_perspective_ellipses: Optional[dict] = None,
    hero_perspective_homography: Optional[dict] = None,
    hero_detection_boxes: Optional[dict] = None,
) -> BikeOut:
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

    perspective_ellipses: Optional[dict[str, RimEllipse]] = None
    if hero_perspective_ellipses is not None:
        perspective_ellipses = {}
        for key, raw in hero_perspective_ellipses.items():
            try:
                perspective_ellipses[key] = RimEllipse(**raw)
            except Exception as exc:
                logging.warning(
                    "Skipping invalid perspective ellipse on bike %s: %r (%s)",
                    doc.get("_id"), raw, exc
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
        hero_thumb_url=hero_thumb_url if hero_thumb_url is not None else doc.get("hero_thumb_url"),
        hero_perspective_ellipses=perspective_ellipses or None,
        hero_perspective_homography=hero_perspective_homography,
        hero_detection_boxes=hero_detection_boxes,
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


def _load_perspective_homography(entry: dict | None) -> Optional[dict]:
    if not isinstance(entry, dict):
        return None
    H = entry.get("H")
    H_inv = entry.get("H_inv")
    if H is None or H_inv is None:
        return None
    if isinstance(H, list) and len(H) == 3 and all(isinstance(r, list) for r in H):
        H = [v for row in H for v in row]
    if isinstance(H_inv, list) and len(H_inv) == 3 and all(isinstance(r, list) for r in H_inv):
        H_inv = [v for row in H_inv for v in row]
    try:
        H_mat = np.array(H, dtype=float)
        H_inv_mat = np.array(H_inv, dtype=float)
        if H_mat.size == 9:
            H_mat = H_mat.reshape((3, 3))
        if H_inv_mat.size == 9:
            H_inv_mat = H_inv_mat.reshape((3, 3))
    except Exception:
        return None
    rectify = entry.get("rectify") if isinstance(entry.get("rectify"), dict) else None
    return {"H": H_mat, "H_inv": H_inv_mat, "rectify": rectify}


def _default_page_settings() -> dict:
    return {
        "perspective_mode": "off",
        "show_measurements": True,
        "show_ellipses": True,
        "show_detection_boxes": False,
        "front_wheel_size": "29",
        "rear_wheel_size": "29",
    }


def _page_settings_doc_to_out(doc) -> BikePageSettingsOut:
    return BikePageSettingsOut(
        bike_id=str(doc["bike_id"]),
        user_id=str(doc["user_id"]),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        settings=doc.get("settings", {}),
    )


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
        hero_thumb_url = await resolve_hero_variant_url(hero_id, "low")
        out.append(bike_doc_to_out(d, hero_url=hero_url, hero_thumb_url=hero_thumb_url))

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
    hero_perspective_ellipses = None
    hero_perspective_homography = None
    hero_detection_boxes = None
    if hero_id:
        media_doc = await media_items_col().find_one({"_id": hero_id, "bike_id": oid})
        if media_doc:
            hero_perspective_ellipses = media_doc.get("perspective_ellipses")
            hero_perspective_homography = media_doc.get("perspective_homography")
            hero_detection_boxes = media_doc.get("detection_boxes")

    hero_thumb_url = await resolve_hero_variant_url(hero_id, "low")
    return bike_doc_to_out(
        doc,
        hero_url=hero_url,
        hero_thumb_url=hero_thumb_url,
        hero_perspective_ellipses=hero_perspective_ellipses,
        hero_perspective_homography=hero_perspective_homography,
        hero_detection_boxes=hero_detection_boxes,
    )


@router.get("/{bike_id}/page_settings", response_model=BikePageSettingsOut)
async def get_page_settings(
    bike_id: str,
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

    settings_col = bike_page_settings_col()
    doc = await settings_col.find_one({"bike_id": oid, "user_id": user_oid})
    if not doc:
        now = datetime.utcnow()
        payload = _default_page_settings()
        doc = {
            "bike_id": oid,
            "user_id": user_oid,
            "created_at": now,
            "updated_at": now,
            "settings": payload,
        }
        await settings_col.insert_one(doc)

    return _page_settings_doc_to_out(doc)


@router.put("/{bike_id}/page_settings", response_model=BikePageSettingsOut)
async def update_page_settings(
    bike_id: str,
    payload: BikePageSettingsPayload,
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

    settings_col = bike_page_settings_col()
    patch = payload.settings or {}
    now = datetime.utcnow()

    doc = await settings_col.find_one({"bike_id": oid, "user_id": user_oid})
    if not doc:
        base = _default_page_settings()
        base.update(patch)
        doc = {
            "bike_id": oid,
            "user_id": user_oid,
            "created_at": now,
            "updated_at": now,
            "settings": base,
        }
        await settings_col.insert_one(doc)
        return _page_settings_doc_to_out(doc)

    await settings_col.update_one(
        {"_id": doc["_id"]},
        {"$set": {"settings": {**doc.get("settings", {}), **patch}, "updated_at": now}},
    )
    updated = await settings_col.find_one({"_id": doc["_id"]})
    return _page_settings_doc_to_out(updated)

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
    hero_thumb_url = await resolve_hero_variant_url(hero_id, "low")
    return bike_doc_to_out(updated, hero_url=hero_url, hero_thumb_url=hero_thumb_url)


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

# bikes/routes.py (or wherever your bikes router lives)
# from datetime import datetime
# import math
# from fastapi import APIRouter, Depends, HTTPException, status
# from bson import ObjectId

# router = APIRouter()  # <-- IMPORTANT: keep prefix out of here if you include_router(prefix="/bikes")

# Assumes you already have:
# - bikes_col()
# - get_current_user
# - _extract_user_oid
# - resolve_hero_url
# - bike_doc_to_out
# - BikeGeometry, BikeOut

def _find_point(points: list[dict], ptype: str):
    def _point_type(point):
        if isinstance(point, dict):
            return point.get("type")
        return getattr(point, "type", None)

    return next((p for p in points if _point_type(p) == ptype), None)

def _resolve_shock_segment(points: list[dict], bodies: list[dict]) -> tuple[dict, dict] | None:
    def _body_type(body):
        if isinstance(body, dict):
            return body.get("type")
        return getattr(body, "type", None)

    shock = next((b for b in bodies if _body_type(b) == "shock"), None)
    if not shock:
        return None
    if isinstance(shock, dict):
        ids = [pid for pid in (shock.get("point_ids") or []) if pid]
    else:
        ids = [pid for pid in (getattr(shock, "point_ids", None) or []) if pid]
    if len(ids) < 2:
        return None
    def _point_id(point):
        if isinstance(point, dict):
            return point.get("id")
        return getattr(point, "id", None)

    def _point_coords(point):
        if isinstance(point, dict):
            return point.get("x"), point.get("y")
        return getattr(point, "x", None), getattr(point, "y", None)

    if len(ids) == 2:
        p1 = next((p for p in points if _point_id(p) == ids[0]), None)
        p2 = next((p for p in points if _point_id(p) == ids[1]), None)
        if not p1 or not p2:
            return None
        return (p1, p2)

    anchor = None
    for pid in ids:
        p = next((pt for pt in points if _point_id(pt) == pid), None)
        if not p:
            continue
        ptype = p.get("type") if isinstance(p, dict) else getattr(p, "type", None)
        if ptype in ("fixed", "bb", "bottom_bracket"):
            anchor = p
            break
    if not anchor:
        anchor = next((pt for pt in points if _point_id(pt) == ids[0]), None)
    if not anchor:
        return None

    ax, ay = _point_coords(anchor)
    if ax is None or ay is None:
        return None

    nearest = None
    nearest_dist = math.inf
    for pid in ids:
        if pid == _point_id(anchor):
            continue
        p = next((pt for pt in points if _point_id(pt) == pid), None)
        if not p:
            continue
        px, py = _point_coords(p)
        dx = float(px or 0) - float(ax)
        dy = float(py or 0) - float(ay)
        d = math.hypot(dx, dy)
        if d < nearest_dist:
            nearest_dist = d
            nearest = p
    if not nearest:
        return None
    return (anchor, nearest)


def _compute_scale_mm_per_px(
    points: list[dict],
    bodies: list[dict],
    source: str,
    mm_value: float,
) -> float:
    def _point_coords(point):
        if isinstance(point, dict):
            return point.get("x"), point.get("y")
        return getattr(point, "x", None), getattr(point, "y", None)

    if mm_value <= 0:
        raise HTTPException(status_code=400, detail="Measurement must be > 0")

    if source == "rear_center":
        a = _find_point(points, "bb") or _find_point(points, "bottom_bracket")
        b = _find_point(points, "rear_axle")
        if not a or not b:
            raise HTTPException(status_code=400, detail="Cannot compute scale: need bb and rear_axle points")
        ax, ay = _point_coords(a)
        bx, by = _point_coords(b)
        d_px = math.hypot(float(bx) - float(ax), float(by) - float(ay))

    elif source == "front_center":
        a = _find_point(points, "bb") or _find_point(points, "bottom_bracket")
        b = _find_point(points, "front_axle")
        if not a or not b:
            raise HTTPException(status_code=400, detail="Cannot compute scale: need bb and front_axle points")
        ax, ay = _point_coords(a)
        bx, by = _point_coords(b)
        d_px = math.hypot(float(bx) - float(ax), float(by) - float(ay))

    elif source == "wheelbase":
        a = _find_point(points, "rear_axle")
        b = _find_point(points, "front_axle")
        if not a or not b:
            raise HTTPException(status_code=400, detail="Cannot compute scale: need rear_axle and front_axle points")
        ax, ay = _point_coords(a)
        bx, by = _point_coords(b)
        d_px = math.hypot(float(bx) - float(ax), float(by) - float(ay))

    elif source == "shock_eye":
        seg = _resolve_shock_segment(points, bodies or [])
        if not seg:
            raise HTTPException(status_code=400, detail="Cannot compute scale: need shock body points")
        a, b = seg
        ax, ay = _point_coords(a)
        bx, by = _point_coords(b)
        d_px = math.hypot(float(bx) - float(ax), float(by) - float(ay))

    else:
        raise HTTPException(status_code=400, detail="Unknown scale_source")

    if d_px <= 0:
        raise HTTPException(status_code=400, detail="Cannot compute scale: pixel distance is zero")

    return float(mm_value) / float(d_px)


@router.put("/{bike_id}/geometry", response_model=BikeOut)
async def update_geometry(
    bike_id: str,
    payload: BikeGeometry,
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

    geometry = bike.get("geometry") or {}
    patch = payload.model_dump(exclude_unset=True)

    # If caller is selecting a scale source, compute scale from the corresponding mm field.
    if "scale_source" in patch and patch["scale_source"] is not None:
        src = patch["scale_source"]

        # Prefer the value in THIS request; fallback to already-stored geometry.
        value_field = f"{src}_mm"
        mm_value = patch.get(value_field, geometry.get(value_field))
        if mm_value is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"scale_source='{src}' requires '{value_field}' to be set",
            )

        points = bike.get("points", []) or []
        bodies = bike.get("bodies", []) or []
        scale = _compute_scale_mm_per_px(points, bodies, src, float(mm_value))

        patch["scale_mm_per_px"] = scale
        patch["scale_source"] = src  # ensure stored

    geometry.update(patch)

    await bikes.update_one(
        {"_id": oid, "user_id": user_oid},
        {"$set": {"geometry": geometry, "updated_at": datetime.utcnow()}},
    )

    updated = await bikes.find_one({"_id": oid, "user_id": user_oid})
    hero_id = updated.get("hero_media_id")
    hero_url = await resolve_hero_url(hero_id)
    hero_thumb_url = await resolve_hero_variant_url(hero_id, "low")
    return bike_doc_to_out(updated, hero_url=hero_url, hero_thumb_url=hero_thumb_url)


# @router.put("/{bike_id}/rear_center", response_model=BikeOut)
# async def update_rear_center(
#     bike_id: str,
#     payload: RearCenterUpdate,
#     current_user=Depends(get_current_user),
# ):
#     """Set rear-centre [mm] and compute/store a scale factor (mm per px)."""
#     user_oid = _extract_user_oid(current_user)

#     try:
#         oid = ObjectId(bike_id)
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid bike_id")

#     bikes = bikes_col()

#     bike = await bikes.find_one({"_id": oid, "user_id": user_oid})
#     if not bike:
#         raise HTTPException(status_code=404, detail="Bike not found")

#     points = bike.get("points", []) or []
#     bb = next((p for p in points if p.get("type") in ("bb", "bottom_bracket")), None)
#     rear_axle = next((p for p in points if p.get("type") == "rear_axle"), None)

#     if not bb or not rear_axle:
#         raise HTTPException(
#             status_code=400,
#             detail="Cannot compute scale: need BB and rear_axle points",
#         )

#     dx = rear_axle["x"] - bb["x"]
#     dy = rear_axle["y"] - bb["y"]
#     d_px = math.hypot(dx, dy)
#     if d_px <= 0:
#         raise HTTPException(
#             status_code=400,
#             detail="Cannot compute scale: BB and rear axle coincide",
#         )

#     rear_center_mm = float(payload.rear_center_mm)
#     scale_mm_per_px = rear_center_mm / d_px

#     await bikes.update_one(
#         {"_id": oid, "user_id": user_oid},
#         {"$set": {
#             "geometry.rear_center_mm": rear_center_mm,
#             "geometry.scale_mm_per_px": scale_mm_per_px,
#             "geometry.scale_source": "rear_center",
#             "updated_at": datetime.utcnow(),
#         }},
#     )

#     updated = await bikes.find_one({"_id": oid, "user_id": user_oid})
#     hero_id = updated.get("hero_media_id")
#     hero_url = await resolve_hero_url(hero_id)
#     return bike_doc_to_out(updated, hero_url=hero_url)



# @router.put("/{bike_id}/front_center", response_model=BikeOut)
# async def update_front_center(
#     bike_id: str,
#     payload: FrontCenterUpdate,
#     current_user=Depends(get_current_user),
# ):
#     """Set front-centre [mm] and compute/store a scale factor (mm per px)."""
#     user_oid = _extract_user_oid(current_user)

#     try:
#         oid = ObjectId(bike_id)
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid bike_id")

#     bikes = bikes_col()

#     bike = await bikes.find_one({"_id": oid, "user_id": user_oid})
#     if not bike:
#         raise HTTPException(status_code=404, detail="Bike not found")

#     points = bike.get("points", []) or []
#     bb = next((p for p in points if p.get("type") in ("bb", "bottom_bracket")), None)
#     front_axle = next((p for p in points if p.get("type") == "front_axle"), None)

#     if not bb or not front_axle:
#         raise HTTPException(
#             status_code=400,
#             detail="Cannot compute scale: need BB and front_axle points",
#         )

#     dx = front_axle["x"] - bb["x"]
#     dy = front_axle["y"] - bb["y"]
#     d_px = math.hypot(dx, dy)
#     if d_px <= 0:
#         raise HTTPException(
#             status_code=400,
#             detail="Cannot compute scale: BB and front axle coincide",
#         )

#     front_center_mm = float(payload.front_center_mm)
#     scale_mm_per_px = front_center_mm / d_px

#     await bikes.update_one(
#         {"_id": oid, "user_id": user_oid},
#         {"$set": {
#             "geometry.front_center_mm": front_center_mm,
#             "geometry.scale_mm_per_px": scale_mm_per_px,
#             "geometry.scale_source": "front_center",
#             "updated_at": datetime.utcnow(),
#         }},
#     )

#     updated = await bikes.find_one({"_id": oid, "user_id": user_oid})
#     hero_id = updated.get("hero_media_id")
#     hero_url = await resolve_hero_url(hero_id)
#     return bike_doc_to_out(updated, hero_url=hero_url)


# @router.put("/{bike_id}/wheelbase", response_model=BikeOut)
# async def update_wheelbase(
#     bike_id: str,
#     payload: WheelbaseUpdate,
#     current_user=Depends(get_current_user),
# ):
#     """Set wheelbase [mm] and compute/store scale factor (mm per px)."""
#     user_oid = _extract_user_oid(current_user)

#     try:
#         oid = ObjectId(bike_id)
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid bike_id")

#     bikes = bikes_col()

#     bike = await bikes.find_one({"_id": oid, "user_id": user_oid})
#     if not bike:
#         raise HTTPException(status_code=404, detail="Bike not found")

#     points = bike.get("points", []) or []
#     rear_axle = next((p for p in points if p.get("type") == "rear_axle"), None)
#     front_axle = next((p for p in points if p.get("type") == "front_axle"), None)

#     if not rear_axle or not front_axle:
#         raise HTTPException(
#             status_code=400,
#             detail="Cannot compute scale: need rear_axle and front_axle points",
#         )

#     dx = front_axle["x"] - rear_axle["x"]
#     dy = front_axle["y"] - rear_axle["y"]
#     d_px = math.hypot(dx, dy)
#     if d_px <= 0:
#         raise HTTPException(status_code=400, detail="Cannot compute scale: axles coincide")

#     wheelbase_mm = float(payload.wheelbase_mm)
#     scale_mm_per_px = wheelbase_mm / d_px

#     await bikes.update_one(
#         {"_id": oid, "user_id": user_oid},
#         {"$set": {
#             "geometry.wheelbase_mm": wheelbase_mm,
#             "geometry.scale_mm_per_px": scale_mm_per_px,
#             "geometry.scale_source": "wheelbase",
#             "updated_at": datetime.utcnow(),
#         }},
#     )

#     updated = await bikes.find_one({"_id": oid, "user_id": user_oid})
#     hero_id = updated.get("hero_media_id")
#     hero_url = await resolve_hero_url(hero_id)
#     return bike_doc_to_out(updated, hero_url=hero_url)


# @router.put("/{bike_id}/scale_source", response_model=BikeOut)
# async def update_scale_source(
#     bike_id: str,
#     payload: ScaleSourceUpdate,
#     current_user=Depends(get_current_user),
# ):
#     user_oid = _extract_user_oid(current_user)
#     try:
#         oid = ObjectId(bike_id)
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid bike_id")

#     bikes = bikes_col()
#     bike = await bikes.find_one({"_id": oid, "user_id": user_oid})
#     if not bike:
#         raise HTTPException(status_code=404, detail="Bike not found")

#     await bikes.update_one(
#         {"_id": oid, "user_id": user_oid},
#         {"$set": {
#             "geometry.scale_source": payload.scale_source,
#             "updated_at": datetime.utcnow(),
#         }},
#     )

#     updated = await bikes.find_one({"_id": oid, "user_id": user_oid})
#     hero_id = updated.get("hero_media_id")
#     hero_url = await resolve_hero_url(hero_id)
#     return bike_doc_to_out(updated, hero_url=hero_url)


@router.get("/{bike_id}/kinematics", response_model=SolverResult)
async def compute_bike_kinematics(
    bike_id: str,
    steps: int = 250,
    iterations: int = 250,
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

    settings_doc = await bike_page_settings_col().find_one({"bike_id": oid, "user_id": user_oid})
    settings = settings_doc.get("settings") if settings_doc else {}
    if not isinstance(settings, dict):
        settings = {}
    perspective_mode = settings.get("perspective_mode", "off")
    settings_found = settings_doc is not None

    H = None
    H_inv = None
    rectify = None
    homography_applied = False
    ellipses_found = False
    media_doc_found = False
    ellipses_keys: list[str] = []
    if perspective_mode != "off":
        hero_id = doc.get("hero_media_id")
        if hero_id:
            media_doc = await media_items_col().find_one({"_id": hero_id, "bike_id": oid})
            media_doc_found = media_doc is not None
            ellipses = media_doc.get("perspective_ellipses") if media_doc else None
            rear_ellipse = ellipses.get("rear") if isinstance(ellipses, dict) else None
            front_ellipse = ellipses.get("front") if isinstance(ellipses, dict) else None
            ellipses_found = bool(rear_ellipse or front_ellipse)
            if isinstance(ellipses, dict):
                ellipses_keys = list(ellipses.keys())

            mode_key = perspective_mode
            if mode_key in ("both_ls", "both_avg"):
                mode_key = "both"
            stored_homographies = (
                media_doc.get("perspective_homography") if isinstance(media_doc, dict) else None
            )
            if isinstance(stored_homographies, dict) and mode_key in stored_homographies:
                loaded = _load_perspective_homography(stored_homographies.get(mode_key))
                if loaded:
                    H = loaded["H"]
                    H_inv = loaded["H_inv"]
                    rectify = loaded.get("rectify")

            if H is None:
                homography = compute_homography_from_ellipses(
                    rear_ellipse,
                    front_ellipse,
                    mode_key,
                )
                if homography:
                    H = homography["H"]
                    H_inv = homography["H_inv"]
                    rectify = homography.get("rectify")

    if H is not None:
        rectified_points: List[BikePoint] = []
        scale = rectify.get("scale") if isinstance(rectify, dict) else None
        tx = rectify.get("tx") if isinstance(rectify, dict) else None
        ty = rectify.get("ty") if isinstance(rectify, dict) else None
        for p in points:
            mapped = apply_homography(H, p.x, p.y)
            if not mapped:
                rectified_points.append(p)
                continue
            x, y = mapped
            if scale and tx is not None and ty is not None:
                x = x * scale + tx
                y = y * scale + ty
            rectified_points.append(p.copy(update={"x": x, "y": y}))
        points = rectified_points
        homography_applied = True

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
    scale_source = geom.get("scale_source")

    # If we rectified points, recompute scale in rectified space from the same source.
    if H is not None and scale_source:
        value_field = f"{scale_source}_mm"
        mm_value = geom.get(value_field)
        if mm_value is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"scale_source='{scale_source}' requires '{value_field}' to be set",
            )
        scale_mm_per_px = _compute_scale_mm_per_px(
            points,
            bodies,
            str(scale_source),
            float(mm_value),
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
            pre_steps=5,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # --------------------------------------------------------
    # 1) Scale stroke / length / travel to mm, keep coords in px
    # --------------------------------------------------------
    solver_steps = sorted(result.steps, key=lambda s: s.step_index)
    scale_mm = scale_mm_per_px  # mm per px

    for s in solver_steps:
        # These came out of the solver in px
        s.shock_stroke = s.shock_stroke * scale_mm   # → mm
        s.shock_length = s.shock_length * scale_mm   # → mm
        if s.rear_travel is not None:
            s.rear_travel = s.rear_travel * scale_mm # → mm

    # --------------------------------------------------------
    # 2) Build coords per point_id: coords[i] = (x,y) at step i (STILL px)
    # --------------------------------------------------------
    coords_map: dict[str, list[dict]] = {}
    rectify_scale = None
    if isinstance(rectify, dict) and rectify.get("scale") is not None:
        try:
            rectify_scale = float(rectify.get("scale"))
        except Exception:
            rectify_scale = None
    for step in solver_steps:
        for pid, (x, y) in step.points.items():
            if H_inv is not None and rectify_scale and tx is not None and ty is not None:
                rx = (float(x) - tx) / rectify_scale
                ry = (float(y) - ty) / rectify_scale
                mapped = apply_homography(H_inv, rx, ry)
                if mapped:
                    coords_map.setdefault(pid, []).append({"x": mapped[0], "y": mapped[1]})
                    continue
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

    # Front-end gets the scaled result (mm stroke/travel, px coords)
    result.steps = solver_steps
    rear_axle_steps = []
    if result.rear_axle_point_id:
        rear_coords = coords_map.get(result.rear_axle_point_id, [])
        for idx, coord in enumerate(rear_coords):
            raw_x = float(coord.get("x", 0.0))
            raw_y = float(coord.get("y", 0.0))
            scaled = None
            rectified = None
            if H is not None:
                mapped = apply_homography(H, raw_x, raw_y)
                if mapped:
                    scaled = {"x": mapped[0], "y": mapped[1]}
                    if (
                        isinstance(rectify, dict)
                        and rectify.get("scale") is not None
                        and rectify.get("tx") is not None
                        and rectify.get("ty") is not None
                    ):
                        rectified = {
                            "x": mapped[0] * float(rectify["scale"]) + float(rectify["tx"]),
                            "y": mapped[1] * float(rectify["scale"]) + float(rectify["ty"]),
                        }
            rear_axle_steps.append(
                {
                    "step_index": idx,
                    "raw": {"x": raw_x, "y": raw_y},
                    "scaled": scaled,
                    "rectified": rectified,
                }
            )
    result.debug = {
        "perspective_mode": perspective_mode,
        "settings_found": settings_found,
        "ellipses_found": ellipses_found,
        "ellipses_keys": ellipses_keys,
        "media_doc_found": media_doc_found,
        "hero_media_id": str(doc.get("hero_media_id")) if doc.get("hero_media_id") else None,
        "homography_applied": homography_applied,
        "rectify": rectify,
        "scale_mm_per_px": scale_mm_per_px,
        "point_count": len(points),
        "body_count": len(bodies),
        "rear_axle_steps": rear_axle_steps,
    }
    rear_axle_relative_mm: list[list[float]] = []
    leverage_ratio_series: list[Optional[float]] = []
    shock_stroke_mm_series: list[Optional[float]] = []
    if result.rear_axle_point_id:
        source_steps = result.full_steps or solver_steps
        trim_index = 0
        if source_steps:
            for idx, step in enumerate(source_steps):
                if step.shock_stroke is not None and step.shock_stroke >= -1e-9:
                    trim_index = idx
                    break

        origin = None
        if source_steps:
            for step in source_steps[trim_index:]:
                coords = step.points.get(result.rear_axle_point_id)
                if coords:
                    origin = (float(coords[0]), float(coords[1]))
                    break
        if origin:
            ox, oy = origin
            for step in source_steps[trim_index:]:
                coords = step.points.get(result.rear_axle_point_id)
                if not coords:
                    rear_axle_relative_mm.append([0.0, 0.0])
                else:
                    dx = (float(coords[0]) - ox) * scale_mm_per_px
                    dy = (float(coords[1]) - oy) * scale_mm_per_px
                    rear_axle_relative_mm.append([dx, dy])
                shock_stroke_mm_series.append(step.shock_stroke)

        # Compute leverage ratio with a smooth numerical gradient
        if source_steps:
            travel_series = [
                (float(s.rear_travel) if s.rear_travel is not None else np.nan)
                for s in source_steps
            ]
            stroke_series = [
                (float(s.shock_stroke) if s.shock_stroke is not None else np.nan)
                for s in source_steps
            ]
            try:
                with np.errstate(all="ignore"):
                    grad_full = np.gradient(travel_series, stroke_series)
                if trim_index > 0 and len(grad_full) > trim_index:
                    grad = grad_full[trim_index:]
                else:
                    grad = grad_full
                # Trim to series length if needed.
                if len(grad) > len(shock_stroke_mm_series):
                    grad = grad[: len(shock_stroke_mm_series)]
                # Endpoint smoothing: use neighboring value to avoid sharp kink at step 0
                if len(grad) >= 2:
                    grad[0] = grad[1]
                    grad[-1] = grad[-2]
                leverage_ratio_series = [
                    (float(val) if np.isfinite(val) else None) for val in grad
                ]
            except Exception:
                leverage_ratio_series = []

    scaled_outputs = {
        "rear_axle_relative_mm": rear_axle_relative_mm,
        "leverage_ratio": leverage_ratio_series,
        "shock_stroke_mm": shock_stroke_mm_series,
    }

    kin_doc = {
        "rear_axle_point_id": result.rear_axle_point_id,
        "n_steps": len(solver_steps),
        # Already in mm after scaling above
        "driver_stroke": solver_steps[-1].shock_stroke if solver_steps else None,
        "steps": kin_steps,
        "scaled_outputs": scaled_outputs,
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

    result.scaled_outputs = scaled_outputs
    return result


@router.get("/{bike_id}/debug")
async def debug_bike(
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
    media_doc = None
    if hero_id:
        media_doc = await media_items_col().find_one({"_id": hero_id, "bike_id": oid})

    settings_doc = await bike_page_settings_col().find_one({"bike_id": oid, "user_id": user_oid})
    settings_payload = settings_doc.get("settings") if settings_doc else None

    return {
        "db_name": settings.mongodb_db_name,
        "bike_id": str(doc["_id"]),
        "user_id": str(doc.get("user_id")),
        "hero_media_id": str(hero_id) if hero_id else None,
        "points_count": len(doc.get("points") or []),
        "bodies_count": len(doc.get("bodies") or []),
        "geometry_keys": sorted(list((doc.get("geometry") or {}).keys())),
        "page_settings_found": bool(settings_doc),
        "page_settings_keys": sorted(list(settings_payload.keys())) if isinstance(settings_payload, dict) else [],
        "media_doc_found": bool(media_doc),
        "media_ellipses_keys": sorted(list((media_doc.get("perspective_ellipses") or {}).keys()))
        if isinstance(media_doc, dict)
        else [],
    }


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
        scaled_outputs=kin.get("scaled_outputs"),
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
    media_items = media_items_col()

    doc = await bikes.find_one({"_id": oid, "user_id": user_oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Bike not found")

    media_cursor = media_items.find({"bike_id": oid, "user_id": user_oid})
    bucket_names = {os.getenv("GCS_MEDIA_BUCKET", GCS_BUCKET_NAME)}
    async for media_doc in media_cursor:
        bucket_name = media_doc.get("bucket", os.getenv("GCS_MEDIA_BUCKET", GCS_BUCKET_NAME))
        bucket_names.add(bucket_name)
        await media_items.delete_one({"_id": media_doc["_id"]})

    prefix = f"users/{user_oid}/bikes/{bike_id}/"
    for bucket_name in bucket_names:
        try:
            delete_media_prefix(bucket_name, prefix)
        except Exception as exc:
            logging.warning(
                "Failed to delete media prefix=%s bucket=%s for bike %s: %s",
                prefix,
                bucket_name,
                bike_id,
                exc,
            )

    await bikes.delete_one({"_id": oid, "user_id": user_oid})
    return Response(status_code=status.HTTP_204_NO_CONTENT)
