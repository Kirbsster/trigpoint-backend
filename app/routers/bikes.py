# app/routers/bikes.py
import math
import os
import re
import json
import hashlib
from datetime import datetime
from typing import Optional, List
import logging
import numpy as np

from fastapi import APIRouter, Depends, HTTPException, status, Response
from pydantic import BaseModel
from bson import ObjectId
from app.schemas import (
    BikeCreate,
    BikeAccessUpdate,
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
    KinematicsAnimationCache,
    KinematicsPlotCache,
    RimEllipse,
    BikeUpdate,
    BikePageSettingsPayload,
    BikePageSettingsOut,
    BikeVariantCreate,
    BikeVariantHydrateOut,
    BikeVariantOut,
    BikeVariantUpdate,
    ShockPresetCreate,
    ShockPresetOut,
    ShockModel,
)
from app.kinematics.linkage_solver import (
    solve_bike_linkage,
    solve_bike_rest_pose,
    SolverResult,
    SolverStep,
)
from app.kinematics.homography import compute_homography_from_ellipses, apply_homography

from app.db import (
    bike_page_settings_col,
    bike_variants_col,
    bikes_col,
    media_items_col,
    shock_presets_col,
    users_col,
)
from app.storage import delete_media_prefix, GCS_BUCKET_NAME
# from app.storage import generate_signed_url
from .auth import get_current_user
from app.utils_media import resolve_hero_url, resolve_hero_variant_url
from app.settings import settings
from pymongo.errors import DuplicateKeyError

router = APIRouter(prefix="/bikes", tags=["bikes"])

_BIKE_SHARED_SETTINGS_DEFAULTS = {
    "front_wheel_size": "29",
    "rear_wheel_size": "29",
    "brake_rotor_front_mm": 203,
    "brake_rotor_rear_mm": 203,
    "rear_brake_ic_body_id": "",
    "frame_cg_x_mm": None,
    "frame_cg_y_mm": None,
    "frame_mass_kg": None,
}

_PAGE_SETTINGS_DEFAULTS = {
    "perspective_mode": "off",
    "show_measurements": True,
    "show_ellipses": True,
    "show_detection_boxes": False,
    "show_ground_plane": False,
    "show_drivetrain": True,
    "show_center_of_gravity": True,
    "show_anti_squat": True,
    "show_anti_rise": True,
    "show_instant_center": True,
    "show_shock_overlay": True,
    "drivetrain_chainring_teeth": None,
    "drivetrain_cassette_teeth": None,
    "drivetrain_idler_teeth": None,
    "flip_chip_mode_id": "base",
    "flip_chip_selected_state_ids_by_variant": {},
    "active_variant_id": "",
    "rider_cg_x_mm": None,
    "rider_cg_y_mm": None,
    "rider_mass_kg": None,
    "shock_target_sag_pct": 30,
    "image_greyscale": False,
    "image_opacity": 1.0,
    "solver_step_index": 0,
    "point_origin_id": "",
}


def _round_to_nearest_10_mm(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    rounded = int(round(parsed / 10.0) * 10)
    if rounded < 0:
        rounded = 0
    return rounded


def _extract_max_travel_from_relative_series(series) -> Optional[float]:
    if not isinstance(series, list):
        return None
    max_value: Optional[float] = None
    for row in series:
        if not (isinstance(row, (list, tuple)) and len(row) >= 2):
            continue
        try:
            value = abs(float(row[1]))
        except Exception:
            continue
        if not math.isfinite(value):
            continue
        if max_value is None or value > max_value:
            max_value = value
    return max_value


def _derive_max_rear_travel_mm(doc: dict) -> Optional[int]:
    direct = _round_to_nearest_10_mm(doc.get("max_rear_travel_mm"))
    if direct is not None:
        return direct
    kin = doc.get("kinematics_plot_cache")
    if not isinstance(kin, dict):
        return None
    scaled = kin.get("scaled_outputs")
    if not isinstance(scaled, dict):
        return None
    max_trim = _extract_max_travel_from_relative_series(scaled.get("rear_axle_relative_mm"))
    max_full = _extract_max_travel_from_relative_series(scaled.get("rear_axle_relative_mm_full"))
    candidates = [v for v in (max_trim, max_full) if v is not None]
    if not candidates:
        return None
    return _round_to_nearest_10_mm(max(candidates))


def _stale_base_kinematics_cache_patch() -> dict:
    return {
        "kinematics_plot_cache": None,
        "kinematics_animation_cache": None,
        "kinematics_cache_fingerprint": None,
        "kinematics_cache_status": "stale",
        "max_rear_travel_mm": None,
    }

def bike_doc_to_out(
    doc,
    hero_url: Optional[str] = None,
    hero_thumb_url: Optional[str] = None,
    hero_perspective_ellipses: Optional[dict] = None,
    hero_perspective_homography: Optional[dict] = None,
    hero_detection_boxes: Optional[dict] = None,
    creator_shareable_id: Optional[str] = None,
    can_edit: Optional[bool] = None,
    include_plot_cache: bool = False,
) -> BikeOut:
    owner_raw = doc.get("owner_user_id") if doc.get("owner_user_id") is not None else doc.get("user_id")
    visibility = str(doc.get("visibility") or "private").strip().lower()
    if visibility not in {"private", "public"}:
        visibility = "private"
    is_verified = bool(doc.get("is_verified", False))
    if is_verified:
        visibility = "public"

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

    plot_cache_raw = doc.get("kinematics_plot_cache") or {}
    kinematics_plot_cache: Optional[KinematicsPlotCache] = None
    if include_plot_cache and plot_cache_raw:
        try:
            kinematics_plot_cache = KinematicsPlotCache(**plot_cache_raw)
        except Exception as exc:
            logging.warning(
                "Skipping invalid kinematics_plot_cache on bike %s: %r (%s)",
                doc.get("_id"), plot_cache_raw, exc
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
        bike_size=doc.get("bike_size"),
        front_wheel_size=doc.get("front_wheel_size"),
        rear_wheel_size=doc.get("rear_wheel_size"),
        brake_rotor_front_mm=doc.get("brake_rotor_front_mm"),
        brake_rotor_rear_mm=doc.get("brake_rotor_rear_mm"),
        rear_brake_ic_body_id=doc.get("rear_brake_ic_body_id"),
        frame_cg_x_mm=doc.get("frame_cg_x_mm"),
        frame_cg_y_mm=doc.get("frame_cg_y_mm"),
        frame_mass_kg=doc.get("frame_mass_kg"),
        # Avoid "None" string if user_id is missing on old docs
        user_id=str(doc["user_id"]) if doc.get("user_id") is not None else "",
        owner_user_id=str(owner_raw) if owner_raw is not None else "",
        creator_shareable_id=(
            creator_shareable_id
            if creator_shareable_id is not None
            else doc.get("creator_shareable_id")
        ),
        can_edit=bool(can_edit) if can_edit is not None else False,
        max_rear_travel_mm=_derive_max_rear_travel_mm(doc),
        visibility=visibility,  # type: ignore[arg-type]
        is_verified=is_verified,
        verified_by_user_id=(
            str(doc.get("verified_by_user_id"))
            if doc.get("verified_by_user_id") is not None
            else None
        ),
        verified_at=doc.get("verified_at"),
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
        kinematics_plot_cache=kinematics_plot_cache,
        kinematics_cache_fingerprint=doc.get("kinematics_cache_fingerprint"),
        kinematics_cache_status=str(doc.get("kinematics_cache_status") or "stale"),
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


def _extract_user_role(current_user) -> str:
    if isinstance(current_user, dict):
        role = current_user.get("role")
    else:
        role = getattr(current_user, "role", None)
    normalized = str(role or "user").strip().lower()
    return normalized if normalized else "user"


def _is_admin_user(current_user) -> bool:
    return _extract_user_role(current_user) == "admin"


def _owner_filter(user_oid: ObjectId) -> dict:
    return {
        "$or": [
            {"owner_user_id": user_oid},
            {"owner_user_id": {"$exists": False}, "user_id": user_oid},
        ]
    }


def _is_bike_owner(doc: dict, user_oid: ObjectId) -> bool:
    owner_value = doc.get("owner_user_id")
    if owner_value is None:
        owner_value = doc.get("user_id")
    return owner_value == user_oid


def _is_bike_public(doc: dict) -> bool:
    if bool(doc.get("is_verified", False)):
        return True
    visibility = str(doc.get("visibility") or "private").strip().lower()
    return visibility == "public"


def _can_view_bike(doc: dict, user_oid: ObjectId, is_admin: bool) -> bool:
    if is_admin:
        return True
    if _is_bike_owner(doc, user_oid):
        return True
    return _is_bike_public(doc)


def _normalize_shareable_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
    return cleaned[:32] if cleaned else None


def _owner_value_from_doc(doc: dict):
    owner = doc.get("owner_user_id")
    if owner is None:
        owner = doc.get("user_id")
    return owner


def _owner_oid_from_doc(doc: dict) -> Optional[ObjectId]:
    owner = _owner_value_from_doc(doc)
    if isinstance(owner, ObjectId):
        return owner
    if isinstance(owner, str):
        try:
            return ObjectId(owner)
        except Exception:
            return None
    return None


def _combine_with_and(*queries: dict) -> dict:
    parts = [q for q in queries if q]
    if not parts:
        return {}
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


async def _creator_filter_query(creator: Optional[str]) -> dict:
    normalized = _normalize_shareable_id(creator)
    if not normalized:
        return {}

    owner_docs = await users_col().find({"shareable_id": normalized}, {"_id": 1}).to_list(length=200)
    owner_oids = [d["_id"] for d in owner_docs if d.get("_id") is not None]
    clauses: list[dict] = [{"creator_shareable_id": normalized}]
    if owner_oids:
        clauses.extend(
            [
                {"owner_user_id": {"$in": owner_oids}},
                {"owner_user_id": {"$exists": False}, "user_id": {"$in": owner_oids}},
            ]
        )
    return {"$or": clauses}


async def _creator_map_for_docs(docs: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    unresolved: list[ObjectId] = []
    seen: set[ObjectId] = set()

    for d in docs:
        owner_oid = _owner_oid_from_doc(d)
        if owner_oid is None:
            continue
        owner_key = str(owner_oid)
        creator = d.get("creator_shareable_id")
        if isinstance(creator, str) and creator.strip():
            mapping[owner_key] = creator.strip().lower()
            continue
        if owner_oid not in seen:
            unresolved.append(owner_oid)
            seen.add(owner_oid)

    if unresolved:
        users = await users_col().find(
            {"_id": {"$in": unresolved}},
            {"shareable_id": 1},
        ).to_list(length=len(unresolved))
        for user in users:
            shareable = user.get("shareable_id")
            if isinstance(shareable, str) and shareable.strip():
                mapping[str(user["_id"])] = shareable.strip().lower()

    return mapping


def _creator_for_doc(doc: dict, creator_by_owner: dict[str, str]) -> Optional[str]:
    explicit = doc.get("creator_shareable_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()
    owner_oid = _owner_oid_from_doc(doc)
    if owner_oid is None:
        return None
    return creator_by_owner.get(str(owner_oid))


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
    return dict(_PAGE_SETTINGS_DEFAULTS)


def _default_bike_shared_settings() -> dict:
    return dict(_BIKE_SHARED_SETTINGS_DEFAULTS)


def _bike_shared_settings_from_doc(doc: Optional[dict]) -> dict:
    settings = _default_bike_shared_settings()
    if not isinstance(doc, dict):
        return settings
    for key in _BIKE_SHARED_SETTINGS_DEFAULTS.keys():
        if key in doc:
            settings[key] = doc.get(key)
    return settings


def _legacy_bike_shared_settings_patch(
    bike_doc: Optional[dict],
    raw_settings: Optional[dict],
) -> dict:
    if not isinstance(bike_doc, dict) or not isinstance(raw_settings, dict):
        return {}
    patch: dict[str, object] = {}
    for key in _BIKE_SHARED_SETTINGS_DEFAULTS.keys():
        if bike_doc.get(key) is not None:
            continue
        value = raw_settings.get(key)
        if value is None:
            continue
        if key == "rear_brake_ic_body_id" and not str(value).strip():
            continue
        patch[key] = value
    return patch


def _normalize_page_settings(raw_settings: Optional[dict]) -> dict:
    settings = _default_page_settings()
    if isinstance(raw_settings, dict):
        for key in _PAGE_SETTINGS_DEFAULTS.keys():
            if key in raw_settings:
                settings[key] = raw_settings.get(key)
    return settings


def _page_settings_patch_from_payload(raw_settings: Optional[dict]) -> dict:
    if not isinstance(raw_settings, dict):
        return {}
    return {
        key: raw_settings.get(key)
        for key in _PAGE_SETTINGS_DEFAULTS.keys()
        if key in raw_settings
    }


def _slugify_variant_name(value: Optional[str]) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", str(value or "").strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
    return cleaned[:64] if cleaned else "variant"


def _slugify_shock_preset_name(value: Optional[str]) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", str(value or "").strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
    return cleaned[:64] if cleaned else "shock-preset"


async def _next_shock_preset_id(name: Optional[str], brand: Optional[str] = None) -> str:
    base_parts = [str(brand or "").strip(), str(name or "").strip()]
    base = _slugify_shock_preset_name("-".join(part for part in base_parts if part))
    candidate = base
    suffix = 2
    presets = shock_presets_col()
    while await presets.find_one({"preset_id": candidate}, {"_id": 1}):
        suffix_str = f"-{suffix}"
        candidate = f"{base[: max(1, 64 - len(suffix_str))]}{suffix_str}"
        suffix += 1
    return candidate


def _variant_doc_to_out(doc: dict) -> BikeVariantOut:
    return BikeVariantOut(
        id=str(doc["_id"]),
        bike_id=str(doc["bike_id"]),
        name=str(doc.get("name") or "Variant"),
        slug=str(doc.get("slug") or "variant"),
        is_base=bool(doc.get("is_base", False)),
        sort_order=int(doc.get("sort_order", 0) or 0),
        overrides=doc.get("overrides") or {},
        solver_policy=doc.get("solver_policy") or {},
        kinematics_cache_fingerprint=doc.get("kinematics_cache_fingerprint"),
        status=str(doc.get("status") or "ready"),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


async def _load_bike_or_404(bike_id: str) -> dict:
    try:
        bike_oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")
    bike_doc = await bikes_col().find_one({"_id": bike_oid})
    if not bike_doc:
        raise HTTPException(status_code=404, detail="Bike not found")
    return bike_doc


async def _load_variant_or_404(variant_id: str) -> dict:
    try:
        variant_oid = ObjectId(variant_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid variant_id")
    variant_doc = await bike_variants_col().find_one({"_id": variant_oid})
    if not variant_doc:
        raise HTTPException(status_code=404, detail="Variant not found")
    return variant_doc


async def _ensure_base_variant_for_bike(bike_doc: dict) -> dict:
    variants = bike_variants_col()
    existing = await variants.find_one({"bike_id": bike_doc["_id"], "is_base": True})
    if existing:
        return existing

    now = datetime.utcnow()
    base_doc = {
        "bike_id": bike_doc["_id"],
        "name": "Base",
        "slug": "base",
        "is_base": True,
        "sort_order": 0,
        "overrides": {},
        "solver_policy": {},
        "kinematics_cache_fingerprint": None,
        "kinematics_plot_cache": None,
        "kinematics_animation_cache": None,
        "status": "stale",
        "created_at": now,
        "updated_at": now,
    }
    try:
        result = await variants.insert_one(base_doc)
        base_doc["_id"] = result.inserted_id
        return base_doc
    except DuplicateKeyError:
        existing = await variants.find_one({"bike_id": bike_doc["_id"], "is_base": True})
        if existing:
            return existing
        raise


def _merge_variant_overrides_into_settings(
    base_settings: Optional[dict],
    overrides: Optional[dict],
) -> dict:
    merged = dict(base_settings or {})
    for key, value in (overrides or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def _apply_variant_overrides_to_geometry(
    geometry: Optional[dict],
    overrides: Optional[dict],
) -> dict:
    geom = dict(geometry or {})
    if not isinstance(overrides, dict):
        return geom

    if overrides.get("shock_type") is not None:
        geom["shock_type"] = overrides.get("shock_type")
    if overrides.get("shock_model") is not None:
        geom["shock_model"] = overrides.get("shock_model")
    if overrides.get("shock_eye_mm") is not None:
        geom["shock_eye_mm"] = overrides.get("shock_eye_mm")
    return geom


def _apply_variant_point_overrides_to_points(
    points: List[BikePoint],
    overrides: Optional[dict],
) -> List[BikePoint]:
    if not points:
        return []
    if not isinstance(overrides, dict):
        return [point.copy(deep=True) for point in points]
    point_overrides = overrides.get("point_overrides")
    if not isinstance(point_overrides, dict):
        return [point.copy(deep=True) for point in points]

    out: List[BikePoint] = []
    for point in points:
        point_id = str(point.id or "")
        override = point_overrides.get(point_id) if point_id else None
        if not isinstance(override, dict):
            out.append(point.copy(deep=True))
            continue
        update: dict[str, float] = {}
        x_value = override.get("x")
        y_value = override.get("y")
        if x_value is not None:
            update["x"] = float(x_value)
        if y_value is not None:
            update["y"] = float(y_value)
        out.append(point.copy(update=update) if update else point.copy(deep=True))
    return out


def _resolve_shock_length0_px(
    points: List[BikePoint],
    bodies: List[RigidBody],
    geom: Optional[dict],
    scale_mm_per_px: float,
) -> Optional[float]:
    shock_eye_mm = _parse_optional_finite((geom or {}).get("shock_eye_mm"))
    if shock_eye_mm is not None and scale_mm_per_px > 0:
        return shock_eye_mm / scale_mm_per_px

    point_by_id = {str(point.id or ""): point for point in points if str(point.id or "")}
    for body in bodies or []:
        if body.type != "shock":
            continue
        if body.length0 is not None:
            try:
                return float(body.length0)
            except (TypeError, ValueError):
                pass
        point_ids = [str(pid).strip() for pid in (body.point_ids or []) if str(pid).strip()]
        if len(point_ids) < 2:
            continue
        point_a = point_by_id.get(point_ids[0])
        point_b = point_by_id.get(point_ids[1])
        if point_a is None or point_b is None:
            continue
        return math.hypot(float(point_b.x) - float(point_a.x), float(point_b.y) - float(point_a.y))
    return None


def _build_variant_point_constraints(
    target_points: List[BikePoint],
    overrides: Optional[dict],
) -> dict[str, dict[str, float]]:
    if not target_points:
        return {}
    if not isinstance(overrides, dict):
        return {}
    point_overrides = overrides.get("point_overrides")
    if not isinstance(point_overrides, dict):
        return {}

    target_by_id = {str(point.id or ""): point for point in target_points if str(point.id or "")}
    constraints: dict[str, dict[str, float]] = {}
    for point_id, override in point_overrides.items():
        point_id_str = str(point_id or "").strip()
        target_point = target_by_id.get(point_id_str)
        if not point_id_str or target_point is None or not isinstance(override, dict):
            continue
        constraint: dict[str, float] = {}
        if override.get("x") is not None:
            constraint["x"] = float(target_point.x)
        if override.get("y") is not None:
            constraint["y"] = float(target_point.y)
        if constraint:
            constraints[point_id_str] = constraint
    return constraints


def _apply_variant_overrides_to_bodies(
    bodies: List[RigidBody],
    overrides: Optional[dict],
    scale_mm_per_px: float,
) -> List[RigidBody]:
    if not bodies:
        return []
    if not isinstance(overrides, dict):
        return [body.copy(deep=True) for body in bodies]

    shock_eye_mm = _parse_optional_finite(overrides.get("shock_eye_mm"))
    shock_stroke_mm = _parse_optional_finite(overrides.get("shock_stroke_mm"))
    flip_chip_selected_state_ids = (
        overrides.get("flip_chip_selected_state_ids")
        if isinstance(overrides.get("flip_chip_selected_state_ids"), dict)
        else {}
    )

    out: List[RigidBody] = []
    for body in bodies:
        body_type = str(getattr(body, "type", "") or "").strip().lower()
        update: dict = {}

        if body_type == "shock":
            if shock_eye_mm is not None and scale_mm_per_px > 0:
                update["length0"] = shock_eye_mm / scale_mm_per_px
            if shock_stroke_mm is not None:
                update["stroke"] = shock_stroke_mm
        elif body_type == "flip_chip":
            body_id = str(getattr(body, "id", "") or "").strip()
            override_state_id = (
                str(flip_chip_selected_state_ids.get(body_id) or "").strip()
                if body_id
                else ""
            )
            if override_state_id:
                update["selected_state_point_id"] = override_state_id

        out.append(body.copy(update=update) if update else body.copy(deep=True))
    return out


def _solver_rigid_bodies_only(bodies: List[RigidBody]) -> List[RigidBody]:
    out: List[RigidBody] = []
    for body in bodies or []:
        body_type = str(getattr(body, "type", "") or "").strip().lower()
        if body_type == "flip_chip":
            continue
        out.append(body.copy(deep=True))
    return out


def _build_flip_chip_pose_constraints(
    points: List[BikePoint],
    bodies: List[RigidBody],
) -> dict[str, dict[str, float]]:
    if not points or not bodies:
        return {}

    point_by_id = {
        str(point.id or "").strip(): point
        for point in points
        if str(point.id or "").strip()
    }
    linkage_point_ids: set[str] = set()
    for body in bodies:
        body_type = str(getattr(body, "type", "") or "").strip().lower()
        if body_type == "flip_chip":
            continue
        for point_id in getattr(body, "point_ids", None) or []:
            point_id_str = str(point_id or "").strip()
            if point_id_str:
                linkage_point_ids.add(point_id_str)

    constraints: dict[str, dict[str, float]] = {}
    for body in bodies:
        body_type = str(getattr(body, "type", "") or "").strip().lower()
        if body_type != "flip_chip":
            continue

        body_id = str(getattr(body, "id", "") or "").strip()
        state_point_ids = [
            str(point_id or "").strip()
            for point_id in (getattr(body, "point_ids", None) or [])
            if str(point_id or "").strip()
        ]
        if len(state_point_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Flip chip '{body_id or 'unknown'}' requires at least two state points.",
            )

        driver_point_id = str(getattr(body, "driver_point_id", "") or "").strip()
        if not driver_point_id:
            driver_point_id = next(
                (point_id for point_id in state_point_ids if point_id in linkage_point_ids),
                "",
            )
        if not driver_point_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Flip chip '{body_id or 'unknown'}' must reference a linkage-connected "
                    "driver point before solving."
                ),
            )
        if driver_point_id not in point_by_id:
            raise HTTPException(
                status_code=400,
                detail=f"Flip chip '{body_id or 'unknown'}' driver point is missing from bike points.",
            )

        selected_state_point_id = str(getattr(body, "selected_state_point_id", "") or "").strip()
        if not selected_state_point_id:
            selected_state_point_id = driver_point_id
        if selected_state_point_id not in state_point_ids:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Flip chip '{body_id or 'unknown'}' selected state "
                    f"'{selected_state_point_id}' is not one of its state points."
                ),
            )
        if selected_state_point_id not in point_by_id:
            raise HTTPException(
                status_code=400,
                detail=f"Flip chip '{body_id or 'unknown'}' selected state point is missing from bike points.",
            )

        driver_point = point_by_id[driver_point_id]
        state_point = point_by_id[selected_state_point_id]
        dx = float(state_point.x) - float(driver_point.x)
        dy = float(state_point.y) - float(driver_point.y)
        if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
            continue
        constraints[driver_point_id] = {
            "x": float(state_point.x),
            "y": float(state_point.y),
        }

    return constraints


def _variant_fingerprint(
    bike_doc: dict,
    variant_doc: dict,
    overrides: Optional[dict],
    settings: Optional[dict],
    *,
    steps: int,
    iterations: int,
    solver_version: str = "variant_pose_v1",
) -> str:
    payload = {
        "bike_id": str(bike_doc.get("_id") or ""),
        "bike_updated_at": bike_doc.get("updated_at").isoformat() if bike_doc.get("updated_at") else None,
        "variant_id": str(variant_doc.get("_id") or ""),
        "variant_updated_at": variant_doc.get("updated_at").isoformat() if variant_doc.get("updated_at") else None,
        "overrides": overrides or {},
        "settings": settings or {},
        "steps": int(steps),
        "iterations": int(iterations),
        "solver_version": solver_version,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _build_variant_point_overrides(
    base_points: List[BikePoint],
    variant_points: List[BikePoint],
    tolerance_px: float = 1e-6,
) -> dict[str, dict[str, float]]:
    base_by_id = {str(point.id): point for point in base_points if point.id}
    variant_by_id = {str(point.id): point for point in variant_points if point.id}
    if set(base_by_id.keys()) != set(variant_by_id.keys()):
        raise HTTPException(
            status_code=400,
            detail="Variant point overrides can only move existing points.",
        )

    overrides: dict[str, dict[str, float]] = {}
    for point_id, base_point in base_by_id.items():
        variant_point = variant_by_id[point_id]
        dx = float(variant_point.x) - float(base_point.x)
        dy = float(variant_point.y) - float(base_point.y)
        if abs(dx) <= tolerance_px and abs(dy) <= tolerance_px:
            continue
        overrides[point_id] = {
            "x": float(variant_point.x),
            "y": float(variant_point.y),
        }
    return overrides


def _page_settings_doc_to_out(doc) -> BikePageSettingsOut:
    return BikePageSettingsOut(
        bike_id=str(doc["bike_id"]),
        user_id=str(doc["user_id"]),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        settings=_normalize_page_settings(doc.get("settings")),
    )


@router.post("", response_model=BikeOut, status_code=status.HTTP_201_CREATED)
async def create_bike(
    bike_in: BikeCreate,
    current_user=Depends(get_current_user),
):
    now = datetime.utcnow()

    user_oid = _extract_user_oid(current_user)
    creator_shareable_id = None
    if isinstance(current_user, dict):
        creator_shareable_id = _normalize_shareable_id(current_user.get("shareable_id"))
    else:
        creator_shareable_id = _normalize_shareable_id(getattr(current_user, "shareable_id", None))

    doc = {
        "owner_user_id": user_oid,
        "user_id": user_oid,
        "creator_shareable_id": creator_shareable_id,
        "visibility": "private",
        "is_verified": False,
        "verified_by_user_id": None,
        "verified_at": None,
        "name": bike_in.name,
        "brand": bike_in.brand,
        "model_year": bike_in.model_year,
        "bike_size": bike_in.bike_size,
        **_default_bike_shared_settings(),
        "kinematics_plot_cache": None,
        "kinematics_animation_cache": None,
        "kinematics_cache_fingerprint": None,
        "kinematics_cache_status": "stale",
        "created_at": now,
        "updated_at": now,
    }
    bikes = bikes_col()
    result = await bikes.insert_one(doc)
    doc["_id"] = result.inserted_id
    await _ensure_base_variant_for_bike(doc)
    return bike_doc_to_out(doc, can_edit=True)


@router.get("/{bike_id}/variants", response_model=List[BikeVariantOut])
async def list_bike_variants(
    bike_id: str,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    is_admin = _is_admin_user(current_user)
    bike_doc = await _load_bike_or_404(bike_id)
    if not _can_view_bike(bike_doc, user_oid, is_admin):
        raise HTTPException(status_code=404, detail="Bike not found")

    await _ensure_base_variant_for_bike(bike_doc)
    docs = await bike_variants_col().find({"bike_id": bike_doc["_id"]}).sort(
        [("sort_order", 1), ("created_at", 1)]
    ).to_list(length=200)
    return [_variant_doc_to_out(doc) for doc in docs]


@router.post("/{bike_id}/variants", response_model=BikeVariantOut, status_code=status.HTTP_201_CREATED)
async def create_bike_variant(
    bike_id: str,
    payload: BikeVariantCreate,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    is_admin = _is_admin_user(current_user)
    bike_doc = await _load_bike_or_404(bike_id)
    if not (is_admin or _is_bike_owner(bike_doc, user_oid)):
        raise HTTPException(status_code=404, detail="Bike not found")

    await _ensure_base_variant_for_bike(bike_doc)
    now = datetime.utcnow()
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Variant name is required")
    slug = _slugify_variant_name(payload.slug or name)
    sort_order = payload.sort_order
    if sort_order is None:
        last_variant = await bike_variants_col().find({"bike_id": bike_doc["_id"]}).sort(
            [("sort_order", -1), ("created_at", -1)]
        ).limit(1).to_list(length=1)
        sort_order = int(last_variant[0].get("sort_order", 0) or 0) + 1 if last_variant else 1

    variant_doc = {
        "bike_id": bike_doc["_id"],
        "name": name,
        "slug": slug,
        "is_base": False,
        "sort_order": int(sort_order),
        "overrides": payload.overrides or {},
        "solver_policy": payload.solver_policy or {},
        "kinematics_cache_fingerprint": None,
        "kinematics_plot_cache": None,
        "kinematics_animation_cache": None,
        "status": "stale",
        "created_at": now,
        "updated_at": now,
    }
    try:
        result = await bike_variants_col().insert_one(variant_doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Variant slug already exists for this bike")
    variant_doc["_id"] = result.inserted_id
    return _variant_doc_to_out(variant_doc)


@router.get("/variants/{variant_id}", response_model=BikeVariantOut)
async def get_bike_variant(
    variant_id: str,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    is_admin = _is_admin_user(current_user)
    variant_doc = await _load_variant_or_404(variant_id)
    bike_doc = await bikes_col().find_one({"_id": variant_doc["bike_id"]})
    if not bike_doc or not _can_view_bike(bike_doc, user_oid, is_admin):
        raise HTTPException(status_code=404, detail="Variant not found")
    return _variant_doc_to_out(variant_doc)


@router.put("/variants/{variant_id}", response_model=BikeVariantOut)
async def update_bike_variant(
    variant_id: str,
    payload: BikeVariantUpdate,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    is_admin = _is_admin_user(current_user)
    variant_doc = await _load_variant_or_404(variant_id)
    bike_doc = await bikes_col().find_one({"_id": variant_doc["bike_id"]})
    if not bike_doc or not (is_admin or _is_bike_owner(bike_doc, user_oid)):
        raise HTTPException(status_code=404, detail="Variant not found")

    update_doc: dict = {}
    if payload.name is not None:
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Variant name is required")
        update_doc["name"] = name
    if payload.slug is not None:
        if variant_doc.get("is_base"):
            raise HTTPException(status_code=400, detail="Base variant slug cannot be changed")
        update_doc["slug"] = _slugify_variant_name(payload.slug)
    if payload.sort_order is not None:
        update_doc["sort_order"] = int(payload.sort_order)
    if payload.overrides is not None:
        update_doc["overrides"] = payload.overrides
        update_doc["kinematics_plot_cache"] = None
        update_doc["kinematics_animation_cache"] = None
        update_doc["kinematics_cache_fingerprint"] = None
        update_doc["status"] = "stale"
    if payload.solver_policy is not None:
        update_doc["solver_policy"] = payload.solver_policy
        update_doc["kinematics_plot_cache"] = None
        update_doc["kinematics_animation_cache"] = None
        update_doc["kinematics_cache_fingerprint"] = None
        update_doc["status"] = "stale"
    if payload.kinematics_cache_fingerprint is not None:
        update_doc["kinematics_cache_fingerprint"] = payload.kinematics_cache_fingerprint
    if payload.kinematics_plot_cache is not None:
        update_doc["kinematics_plot_cache"] = payload.kinematics_plot_cache
    if payload.kinematics_animation_cache is not None:
        update_doc["kinematics_animation_cache"] = payload.kinematics_animation_cache
    if payload.status is not None:
        update_doc["status"] = payload.status

    if not update_doc:
        return _variant_doc_to_out(variant_doc)

    update_doc["updated_at"] = datetime.utcnow()
    try:
        await bike_variants_col().update_one({"_id": variant_doc["_id"]}, {"$set": update_doc})
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Variant slug already exists for this bike")
    merged = {**variant_doc, **update_doc}
    return _variant_doc_to_out(merged)


@router.delete("/variants/{variant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_bike_variant(
    variant_id: str,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    is_admin = _is_admin_user(current_user)
    variant_doc = await _load_variant_or_404(variant_id)
    bike_doc = await bikes_col().find_one({"_id": variant_doc["bike_id"]})
    if not bike_doc or not (is_admin or _is_bike_owner(bike_doc, user_oid)):
        raise HTTPException(status_code=404, detail="Variant not found")
    if bool(variant_doc.get("is_base")):
        raise HTTPException(status_code=400, detail="Base variant cannot be deleted")

    await bike_variants_col().delete_one({"_id": variant_doc["_id"]})
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/variants/{variant_id}/hydrate", response_model=BikeVariantHydrateOut)
async def hydrate_bike_variant(
    variant_id: str,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    is_admin = _is_admin_user(current_user)
    variant_doc = await _load_variant_or_404(variant_id)
    bike_doc = await bikes_col().find_one({"_id": variant_doc["bike_id"]})
    if not bike_doc or not _can_view_bike(bike_doc, user_oid, is_admin):
        raise HTTPException(status_code=404, detail="Variant not found")

    return BikeVariantHydrateOut(
        variant=_variant_doc_to_out(variant_doc),
        kinematics_plot_cache=variant_doc.get("kinematics_plot_cache"),
        kinematics_cache_fingerprint=variant_doc.get("kinematics_cache_fingerprint"),
        status=str(variant_doc.get("status") or "ready"),
    )


@router.get("", response_model=List[BikeOut])
async def list_my_bikes(
    creator: Optional[str] = None,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    bikes = bikes_col()
    creator_filter = await _creator_filter_query(creator)
    cursor = bikes.find(_combine_with_and(_owner_filter(user_oid), creator_filter))
    docs = await cursor.to_list(length=1000)
    creator_by_owner = await _creator_map_for_docs(docs)

    out: list[BikeOut] = []
    for d in docs:
        hero_id = d.get("hero_media_id")
        hero_url = await resolve_hero_url(hero_id)
        hero_thumb_url = await resolve_hero_variant_url(hero_id, "low")
        out.append(
            bike_doc_to_out(
                d,
                hero_url=hero_url,
                hero_thumb_url=hero_thumb_url,
                creator_shareable_id=_creator_for_doc(d, creator_by_owner),
                can_edit=_is_bike_owner(d, user_oid),
            )
        )

    return out


@router.get("/community", response_model=List[BikeOut])
async def list_community_bikes(
    creator: Optional[str] = None,
    current_user=Depends(get_current_user),
):
    # Auth-gated for now; visibility/segregation can be relaxed later if needed.
    user_oid = _extract_user_oid(current_user)
    creator_filter = await _creator_filter_query(creator)
    cursor = bikes_col().find(
        _combine_with_and(
            {"visibility": "public", "$or": [{"is_verified": {"$exists": False}}, {"is_verified": False}]},
            creator_filter,
        )
    ).sort("updated_at", -1)
    docs = await cursor.to_list(length=1000)
    creator_by_owner = await _creator_map_for_docs(docs)

    out: list[BikeOut] = []
    for d in docs:
        hero_id = d.get("hero_media_id")
        hero_url = await resolve_hero_url(hero_id)
        hero_thumb_url = await resolve_hero_variant_url(hero_id, "low")
        out.append(
            bike_doc_to_out(
                d,
                hero_url=hero_url,
                hero_thumb_url=hero_thumb_url,
                creator_shareable_id=_creator_for_doc(d, creator_by_owner),
                can_edit=_is_bike_owner(d, user_oid),
            )
        )
    return out


@router.get("/official", response_model=List[BikeOut])
async def list_official_bikes(
    creator: Optional[str] = None,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    creator_filter = await _creator_filter_query(creator)
    cursor = bikes_col().find(
        _combine_with_and({"is_verified": True}, creator_filter)
    ).sort("updated_at", -1)
    docs = await cursor.to_list(length=1000)
    creator_by_owner = await _creator_map_for_docs(docs)

    out: list[BikeOut] = []
    for d in docs:
        hero_id = d.get("hero_media_id")
        hero_url = await resolve_hero_url(hero_id)
        hero_thumb_url = await resolve_hero_variant_url(hero_id, "low")
        out.append(
            bike_doc_to_out(
                d,
                hero_url=hero_url,
                hero_thumb_url=hero_thumb_url,
                creator_shareable_id=_creator_for_doc(d, creator_by_owner),
                can_edit=_is_bike_owner(d, user_oid),
            )
    )
    return out


@router.get("/visible", response_model=List[BikeOut])
async def list_visible_bikes(
    creator: Optional[str] = None,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    is_admin = _is_admin_user(current_user)
    creator_filter = await _creator_filter_query(creator)

    if is_admin:
        visibility_query = {}
    else:
        visibility_query = {
            "$or": [
                _owner_filter(user_oid),
                {"visibility": "public"},
                {"is_verified": True},
            ]
        }

    cursor = bikes_col().find(
        _combine_with_and(visibility_query, creator_filter)
    ).sort([("brand", 1), ("model_year", -1), ("updated_at", -1)])
    docs = await cursor.to_list(length=2000)
    creator_by_owner = await _creator_map_for_docs(docs)

    out: list[BikeOut] = []
    for d in docs:
        hero_id = d.get("hero_media_id")
        hero_url = await resolve_hero_url(hero_id)
        hero_thumb_url = await resolve_hero_variant_url(hero_id, "low")
        out.append(
            bike_doc_to_out(
                d,
                hero_url=hero_url,
                hero_thumb_url=hero_thumb_url,
                creator_shareable_id=_creator_for_doc(d, creator_by_owner),
                can_edit=_is_bike_owner(d, user_oid),
            )
        )
    return out


@router.get("/shock-presets", response_model=List[ShockPresetOut])
async def list_shock_presets(current_user=Depends(get_current_user)):
    # Auth-gated so we can evolve this into user/team-owned presets later.
    _extract_user_oid(current_user)
    await _ensure_default_shock_presets()
    cursor = shock_presets_col().find({}).sort(
        [("sort_order", 1), ("category", 1), ("name", 1), ("preset_id", 1)]
    )
    docs = await cursor.to_list(length=500)
    out: list[ShockPresetOut] = []
    for doc in docs:
        try:
            out.append(_shock_preset_doc_to_out(doc))
        except Exception:
            continue
    return out


@router.post("/shock-presets", response_model=ShockPresetOut, status_code=status.HTTP_201_CREATED)
async def create_shock_preset(
    payload: ShockPresetCreate,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    name = str(payload.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Shock preset name is required")

    shock_type = str(payload.shock_type or "air").strip().lower()
    if shock_type not in {"air", "coil"}:
        shock_type = "air"

    model = _coerce_full_shock_model_doc(payload.shock_model.model_dump())
    preset_id = await _next_shock_preset_id(name, payload.brand)
    now = datetime.utcnow()
    doc = {
        "preset_id": preset_id,
        "name": name,
        "brand": str(payload.brand).strip() if payload.brand is not None else None,
        "category": str(payload.category).strip() if payload.category is not None else None,
        "shock_type": shock_type,
        "shock_model": model,
        "sort_order": 1000,
        "created_at": now,
        "updated_at": now,
        "created_by_user_id": user_oid,
    }
    try:
        result = await shock_presets_col().insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Shock preset id already exists")
    doc["_id"] = result.inserted_id
    return _shock_preset_doc_to_out(doc)


@router.get("/{bike_id}", response_model=BikeOut)
async def get_bike(
    bike_id: str,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    is_admin = _is_admin_user(current_user)
    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()
    doc = await bikes.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Bike not found")
    if not _can_view_bike(doc, user_oid, is_admin):
        raise HTTPException(status_code=403, detail="Not allowed to view this bike")
    if _is_bike_owner(doc, user_oid):
        settings_doc = await bike_page_settings_col().find_one({"bike_id": oid, "user_id": user_oid})
        legacy_patch = _legacy_bike_shared_settings_patch(
            doc,
            settings_doc.get("settings") if settings_doc else None,
        )
        if legacy_patch:
            migrated_updated_at = datetime.utcnow()
            await bikes.update_one(
                {"_id": oid, **_owner_filter(user_oid)},
                {"$set": {**legacy_patch, "updated_at": migrated_updated_at}},
            )
            doc.update(legacy_patch)
            doc["updated_at"] = migrated_updated_at

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
        can_edit=_is_bike_owner(doc, user_oid),
        include_plot_cache=True,
    )


@router.put("/{bike_id}", response_model=BikeOut)
async def update_bike(
    bike_id: str,
    payload: BikeUpdate,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    update_data = payload.dict(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    bikes = bikes_col()
    result = await bikes.update_one(
        {"_id": oid, **_owner_filter(user_oid)},
        {"$set": {**update_data, **_stale_base_kinematics_cache_patch(), "updated_at": datetime.utcnow()}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Bike not found")

    doc = await bikes.find_one({"_id": oid, **_owner_filter(user_oid)})
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
        can_edit=True,
    )


@router.put("/{bike_id}/access", response_model=BikeOut)
async def update_bike_access(
    bike_id: str,
    payload: BikeAccessUpdate,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    is_admin = _is_admin_user(current_user)
    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()
    doc = await bikes.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Bike not found")

    is_owner = _is_bike_owner(doc, user_oid)
    if not (is_owner or is_admin):
        raise HTTPException(status_code=403, detail="Not allowed to change bike access")

    patch: dict = {}
    has_visibility_change = payload.visibility is not None
    has_verified_change = payload.is_verified is not None
    if not (has_visibility_change or has_verified_change):
        raise HTTPException(status_code=400, detail="No access fields to update")

    current_visibility = str(doc.get("visibility") or "private").strip().lower()
    if current_visibility not in {"private", "public"}:
        current_visibility = "private"
    current_is_verified = bool(doc.get("is_verified", False))

    if has_verified_change:
        if not is_admin:
            raise HTTPException(status_code=403, detail="Only admin can change verified status")
        next_verified = bool(payload.is_verified)
        patch["is_verified"] = next_verified
        if next_verified:
            patch["visibility"] = "public"
            patch["verified_at"] = datetime.utcnow()
            patch["verified_by_user_id"] = user_oid
        else:
            patch["verified_at"] = None
            patch["verified_by_user_id"] = None

    next_verified_state = (
        bool(payload.is_verified) if has_verified_change else current_is_verified
    )
    if has_visibility_change:
        next_visibility = str(payload.visibility)
        if is_admin and next_visibility == "public" and not next_verified_state:
            patch["is_verified"] = True
            patch["verified_at"] = datetime.utcnow()
            patch["verified_by_user_id"] = user_oid
            next_verified_state = True
        if next_verified_state:
            next_visibility = "public"
        patch["visibility"] = next_visibility

    if current_is_verified and not is_admin and patch.get("visibility") == "private":
        raise HTTPException(status_code=403, detail="Verified bikes must stay public")

    patch["updated_at"] = datetime.utcnow()
    await bikes.update_one({"_id": oid}, {"$set": {**patch, **_stale_base_kinematics_cache_patch()}})
    updated = await bikes.find_one({"_id": oid})
    if not updated:
        raise HTTPException(status_code=404, detail="Bike not found")

    hero_id = updated.get("hero_media_id")
    hero_url = await resolve_hero_url(hero_id)
    hero_thumb_url = await resolve_hero_variant_url(hero_id, "low")
    return bike_doc_to_out(
        updated,
        hero_url=hero_url,
        hero_thumb_url=hero_thumb_url,
        can_edit=_is_bike_owner(updated, user_oid),
    )


@router.get("/{bike_id}/page_settings", response_model=BikePageSettingsOut)
async def get_page_settings(
    bike_id: str,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    is_admin = _is_admin_user(current_user)
    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()
    bike = await bikes.find_one({"_id": oid})
    if not bike:
        raise HTTPException(status_code=404, detail="Bike not found")
    if not _can_view_bike(bike, user_oid, is_admin):
        raise HTTPException(status_code=403, detail="Not allowed to view this bike")

    settings_col = bike_page_settings_col()
    doc = await settings_col.find_one({"bike_id": oid, "user_id": user_oid})
    if doc:
        if _is_bike_owner(bike, user_oid):
            legacy_patch = _legacy_bike_shared_settings_patch(bike, doc.get("settings"))
            if legacy_patch:
                migrated_updated_at = datetime.utcnow()
                await bikes.update_one(
                    {"_id": oid, **_owner_filter(user_oid)},
                    {"$set": {**legacy_patch, "updated_at": migrated_updated_at}},
                )
                bike.update(legacy_patch)
                bike["updated_at"] = migrated_updated_at
        normalized_settings = _normalize_page_settings(doc.get("settings"))
        if normalized_settings != doc.get("settings"):
            await settings_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {"settings": normalized_settings}},
            )
            doc["settings"] = normalized_settings
        return _page_settings_doc_to_out(doc)

    payload = _default_page_settings()
    if not _is_bike_owner(bike, user_oid):
        now = datetime.utcnow()
        return BikePageSettingsOut(
            bike_id=bike_id,
            user_id=str(user_oid),
            created_at=now,
            updated_at=now,
            settings=payload,
        )

    now = datetime.utcnow()
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
    bike = await bikes.find_one({"_id": oid})
    if not bike:
        raise HTTPException(status_code=404, detail="Bike not found")
    if not _is_bike_owner(bike, user_oid):
        raise HTTPException(status_code=403, detail="Not allowed to edit settings for this bike")

    settings_col = bike_page_settings_col()
    patch = _page_settings_patch_from_payload(payload.settings)
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

    merged_settings = _normalize_page_settings(doc.get("settings"))
    merged_settings.update(patch)
    await settings_col.update_one(
        {"_id": doc["_id"]},
        {"$set": {"settings": merged_settings, "updated_at": now}},
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
    doc = await bikes.find_one({"_id": oid, **_owner_filter(user_oid)})
    if not doc:
        raise HTTPException(status_code=404, detail="Bike not found")

    now = datetime.utcnow()

    await bikes.update_one(
        {"_id": oid, **_owner_filter(user_oid)},
        {
            "$set": {
                "points": [p.dict() for p in payload.points],
                **_stale_base_kinematics_cache_patch(),
                "updated_at": now,
            }
        },
    )

    updated = await bikes.find_one({"_id": oid, **_owner_filter(user_oid)})
    hero_id = updated.get("hero_media_id")
    hero_url = await resolve_hero_url(hero_id)
    hero_thumb_url = await resolve_hero_variant_url(hero_id, "low")
    return bike_doc_to_out(updated, hero_url=hero_url, hero_thumb_url=hero_thumb_url, can_edit=True)


@router.put("/variants/{variant_id}/points", response_model=BikeVariantOut)
async def update_variant_points(
    variant_id: str,
    payload: BikePointsUpdate,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    is_admin = _is_admin_user(current_user)

    _ensure_unique_point_ids(payload.points)

    variant_doc = await _load_variant_or_404(variant_id)
    bike_doc = await bikes_col().find_one({"_id": variant_doc["bike_id"]})
    if not bike_doc or not (is_admin or _is_bike_owner(bike_doc, user_oid)):
        raise HTTPException(status_code=404, detail="Variant not found")

    base_points: List[BikePoint] = []
    for raw_point in bike_doc.get("points") or []:
        try:
            base_points.append(BikePoint(**raw_point))
        except Exception as exc:
            logging.warning(
                "Skipping invalid base point on bike %s while saving variant points: %r (%s)",
                bike_doc.get("_id"),
                raw_point,
                exc,
            )
    if not base_points:
        raise HTTPException(status_code=400, detail="Bike has no valid base points defined")

    point_overrides = _build_variant_point_overrides(base_points, payload.points)
    next_overrides = (
        dict(variant_doc.get("overrides") or {})
        if isinstance(variant_doc.get("overrides"), dict)
        else {}
    )
    if point_overrides:
        next_overrides["point_overrides"] = point_overrides
    else:
        next_overrides.pop("point_overrides", None)

    update_doc = {
        "overrides": next_overrides,
        "kinematics_plot_cache": None,
        "kinematics_animation_cache": None,
        "kinematics_cache_fingerprint": None,
        "status": "stale",
        "updated_at": datetime.utcnow(),
    }
    await bike_variants_col().update_one({"_id": variant_doc["_id"]}, {"$set": update_doc})
    merged = {**variant_doc, **update_doc}
    return _variant_doc_to_out(merged)


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
    is_admin = _is_admin_user(current_user)

    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()
    bike_doc = await bikes.find_one(
        {"_id": oid},
        {"bodies": 1, "_id": 1, "owner_user_id": 1, "user_id": 1, "visibility": 1, "is_verified": 1},
    )
    if not bike_doc:
        raise HTTPException(status_code=404, detail="Bike not found")
    if not _can_view_bike(bike_doc, user_oid, is_admin):
        raise HTTPException(status_code=403, detail="Not allowed to view this bike")

    raw_bodies = bike_doc.get("bodies") or []
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
        {"_id": oid, **_owner_filter(user_oid)},
        {
            "$set": {
                "bodies": [b.dict() for b in payload.bodies],
                **_stale_base_kinematics_cache_patch(),
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


def _find_idler_point_id(points: list) -> Optional[str]:
    def _point_type(point):
        if isinstance(point, dict):
            return point.get("type")
        return getattr(point, "type", None)

    def _point_name(point):
        if isinstance(point, dict):
            return point.get("name")
        return getattr(point, "name", None)

    for point in points or []:
        ptype = str(_point_type(point) or "").strip().lower()
        if ptype in {"idler", "idler_gear", "chain_idler", "pulley"}:
            pid = point.get("id") if isinstance(point, dict) else getattr(point, "id", None)
            pid_s = str(pid or "").strip()
            if pid_s:
                return pid_s
    for point in points or []:
        name = str(_point_name(point) or "").strip().lower()
        if "idler" not in name:
            continue
        pid = point.get("id") if isinstance(point, dict) else getattr(point, "id", None)
        pid_s = str(pid or "").strip()
        if pid_s:
            return pid_s
    return None

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


def _list_rear_unsprung_body_candidates(
    bodies: list[RigidBody],
    rear_axle_point_id: Optional[str],
) -> list[tuple[RigidBody, list[str]]]:
    if not rear_axle_point_id:
        return []
    rear_axle_id = str(rear_axle_point_id)
    candidates: list[tuple[RigidBody, list[str]]] = []
    for body in bodies or []:
        if not isinstance(body, RigidBody):
            continue
        if body.type in ("shock", "fixed"):
            continue
        pids = [str(pid) for pid in (body.point_ids or []) if pid]
        if rear_axle_id in pids:
            candidates.append((body, pids))
    return candidates


def _pick_default_rear_unsprung_candidate_index(
    candidates: list[tuple[RigidBody, list[str]]],
    rear_axle_point_id: Optional[str],
    points: list[BikePoint],
) -> int:
    """Default IC body heuristic for split-pivot bikes.

    We prefer the upper member (usually seatstay) by taking the lowest mean `y`
    (image coordinates: smaller y is higher on the image).
    """
    if not candidates:
        return -1
    if len(candidates) == 1:
        return 0

    rear_axle_id = str(rear_axle_point_id or "")
    point_y_by_id: dict[str, float] = {}
    for point in points or []:
        pid = getattr(point, "id", None)
        if not pid:
            continue
        try:
            point_y_by_id[str(pid)] = float(getattr(point, "y"))
        except Exception:
            continue

    rear_y = point_y_by_id.get(rear_axle_id)
    best_idx = 0
    best_score = (float("inf"), float("-inf"), "")
    for idx, (body, pids) in enumerate(candidates):
        ys = [
            point_y_by_id[pid]
            for pid in pids
            if pid != rear_axle_id and pid in point_y_by_id
        ]
        if not ys and rear_y is not None:
            ys = [rear_y]
        avg_y = sum(ys) / len(ys) if ys else float("inf")
        # tie-breaker prefers richer bodies, then stable id ordering
        score = (avg_y, -float(len(pids)), str(getattr(body, "id", "")))
        if score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _pick_rear_body_point_ids(
    bodies: list[RigidBody],
    rear_axle_point_id: Optional[str],
    points: list[BikePoint],
    preferred_body_id: Optional[str] = None,
) -> list[str]:
    """Pick unsprung body point ids used for rear-wheel IC in anti-squat/anti-rise.

    Priority:
    1) Explicit caliper-attached body (if valid caliper point is attached)
    2) Default split-pivot heuristic (upper/seatstay candidate)
    """
    candidates = _list_rear_unsprung_body_candidates(bodies, rear_axle_point_id)
    if not candidates:
        return []

    point_type_by_id: dict[str, str] = {}
    for point in points or []:
        pid = getattr(point, "id", None)
        if pid:
            point_type_by_id[str(pid)] = str(getattr(point, "type", "") or "")

    preferred_id = str(preferred_body_id or "")
    if preferred_id:
        for body, pids in candidates:
            if str(getattr(body, "id", "")) == preferred_id:
                return pids

    for body, pids in candidates:
        caliper_id = getattr(body, "brake_caliper_point_id", None)
        if not caliper_id:
            continue
        if point_type_by_id.get(str(caliper_id)) == "brake_caliper":
            return pids

    idx = _pick_default_rear_unsprung_candidate_index(
        candidates,
        rear_axle_point_id,
        points,
    )
    if idx < 0:
        return []
    return candidates[idx][1]


def _pick_rear_brake_caliper_point_id(
    bodies: list[RigidBody],
    rear_axle_point_id: Optional[str],
    points: list[BikePoint],
    preferred_body_id: Optional[str] = None,
) -> Optional[str]:
    """Pick brake caliper point id from the unsprung body carrying the rear axle."""
    if not rear_axle_point_id:
        return None

    point_type_by_id: dict[str, str] = {}
    for point in points or []:
        pid = getattr(point, "id", None)
        ptype = getattr(point, "type", None)
        if pid:
            point_type_by_id[str(pid)] = str(ptype or "")

    candidate_entries = _list_rear_unsprung_body_candidates(bodies, rear_axle_point_id)
    if not candidate_entries:
        return None

    preferred_id = str(preferred_body_id or "")
    preferred_entry = None
    if preferred_id:
        for entry in candidate_entries:
            body, _ = entry
            if str(getattr(body, "id", "")) == preferred_id:
                preferred_entry = entry
                break

    default_entry = None
    if preferred_entry is None:
        default_idx = _pick_default_rear_unsprung_candidate_index(
            candidate_entries,
            rear_axle_point_id,
            points,
        )
        default_entry = (
            candidate_entries[default_idx]
            if 0 <= default_idx < len(candidate_entries)
            else None
        )

    # Primary: explicit body attachment field.
    # Prefer the default selected body first, then all others.
    ordered_entries = (
        ([preferred_entry] if preferred_entry else [])
        + ([default_entry] if default_entry else [])
        + [
            entry
            for entry in candidate_entries
            if entry is not preferred_entry and entry is not default_entry
        ]
    )
    for entry in ordered_entries:
        if not entry:
            continue
        body, _ = entry
        caliper_id = getattr(body, "brake_caliper_point_id", None)
        if not caliper_id:
            continue
        caliper_id = str(caliper_id)
        if point_type_by_id.get(caliper_id) == "brake_caliper":
            return caliper_id

    # Fallback: first brake_caliper point inside selected/default body, then others.
    for entry in ordered_entries:
        if not entry:
            continue
        _, pids = entry
        for pid in pids:
            pid_s = str(pid)
            if point_type_by_id.get(pid_s) == "brake_caliper":
                return pid_s

    # Final fallback: unique global brake_caliper point.
    global_calipers = [
        pid for pid, ptype in point_type_by_id.items() if str(ptype) == "brake_caliper"
    ]
    if len(global_calipers) == 1:
        return global_calipers[0]

    # If multiple global calipers exist, pick the one closest to rear axle.
    rear_axle_point = next(
        (
            p
            for p in (points or [])
            if str(getattr(p, "id", "")) == str(rear_axle_point_id or "")
        ),
        None,
    )
    if rear_axle_point and global_calipers:
        try:
            rear_x = float(getattr(rear_axle_point, "x"))
            rear_y = float(getattr(rear_axle_point, "y"))
            best_id: Optional[str] = None
            best_d2 = float("inf")
            for pid in global_calipers:
                point = next(
                    (p for p in (points or []) if str(getattr(p, "id", "")) == str(pid)),
                    None,
                )
                if point is None:
                    continue
                x = float(getattr(point, "x"))
                y = float(getattr(point, "y"))
                d2 = (x - rear_x) * (x - rear_x) + (y - rear_y) * (y - rear_y)
                if d2 < best_d2:
                    best_d2 = d2
                    best_id = str(pid)
            if best_id:
                return best_id
        except Exception:
            pass
    return None


def _compute_instant_center_series(
    steps: list[SolverStep],
    body_point_ids: list[str],
) -> list[dict[str, Optional[float]]]:
    """
    Compute instantaneous center per step from point trajectories of one rigid body.

    Returns a list with one entry per step:
    - {"x": float, "y": float} when resolvable
    - {"x": None, "y": None} when near-pure translation / underconstrained
    """
    def _extract_matched_body_coords(
        step_a: SolverStep,
        step_b: SolverStep,
        ids_local: list[str],
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        coords_a: list[tuple[float, float]] = []
        coords_b: list[tuple[float, float]] = []
        for pid in ids_local:
            point_a = step_a.points.get(pid)
            point_b = step_b.points.get(pid)
            if not point_a or not point_b:
                continue
            try:
                ax = float(point_a[0])
                ay = float(point_a[1])
                bx = float(point_b[0])
                by = float(point_b[1])
            except Exception:
                continue
            if not (math.isfinite(ax) and math.isfinite(ay) and math.isfinite(bx) and math.isfinite(by)):
                continue
            coords_a.append((ax, ay))
            coords_b.append((bx, by))
        return coords_a, coords_b

    def _finite_rotation_center(
        src_pts: list[tuple[float, float]],
        dst_pts: list[tuple[float, float]],
    ) -> Optional[tuple[float, float]]:
        if len(src_pts) != len(dst_pts) or len(src_pts) < 2:
            return None
        try:
            P = np.asarray(src_pts, dtype=float)
            Q = np.asarray(dst_pts, dtype=float)
            cP = P.mean(axis=0)
            cQ = Q.mean(axis=0)
            P0 = P - cP
            Q0 = Q - cQ
            H = P0.T @ Q0
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1.0
                R = Vt.T @ U.T
            t = cQ - (R @ cP)
            A = np.eye(2, dtype=float) - R
            detA = float(np.linalg.det(A))
            if not math.isfinite(detA) or abs(detA) < 1e-9:
                return None
            center = np.linalg.solve(A, t)
            x_ic = float(center[0])
            y_ic = float(center[1])
            if not (math.isfinite(x_ic) and math.isfinite(y_ic)):
                return None
            return (x_ic, y_ic)
        except Exception:
            return None

    out: list[dict[str, Optional[float]]] = []
    if not steps:
        return out
    ids = [pid for pid in (body_point_ids or []) if pid]
    if len(ids) < 2:
        return [{"x": None, "y": None} for _ in steps]

    n_steps = len(steps)
    if n_steps == 1:
        return [{"x": None, "y": None}]

    for i in range(n_steps):
        if i == 0:
            src_step = steps[0]
            dst_step = steps[1]
        elif i == n_steps - 1:
            src_step = steps[n_steps - 2]
            dst_step = steps[n_steps - 1]
        else:
            src_step = steps[i - 1]
            dst_step = steps[i + 1]

        src_pts, dst_pts = _extract_matched_body_coords(src_step, dst_step, ids)
        if len(src_pts) < 2:
            out.append({"x": None, "y": None})
            continue

        ic = _finite_rotation_center(src_pts, dst_pts)
        if ic is None:
            out.append({"x": None, "y": None})
            continue
        out.append({"x": float(ic[0]), "y": float(ic[1])})

    return out


_WHEEL_BSD_MM: dict[str, float] = {
    "24": 507.0,
    "26": 559.0,
    "27_5": 584.0,
    "29": 622.0,
}
_TYRE_THICKNESS_MM = 60.0
_CHAIN_PITCH_MM = 12.7
_PSI_TO_PA = 6894.757293168
_DEFAULT_SHOCK_MODEL: dict[str, float] = {
    "air_chamber_diameter_mm": 50.8,
    "body_eyelet_gap_mm": 50.8,
    "shaft_eyelet_gap_mm": 50.8,
    "air_chamber_length_mm": 70,
    "air_negative_chamber_length_mm": 20.0,
    "air_piston_head_thickness_mm": 5.0,
    "air_shaft_diameter_mm": 25.4,
    "air_positive_shaft_diameter_mm": 8.0,
    "air_initial_pressure_psi": 175.0,
    "air_reference_temp_c": 20.0,
    "air_cold_temp_c": 5.0,
    "air_hot_temp_c": 45.0,
    "coil_rate_n_per_mm": 70.0,
    "coil_preload_n": 0.0,
}
_DEFAULT_SHOCK_VISUAL_MODEL: dict[str, dict[str, float] | float] = {
    "body_end_gap_mm": 50.8,
    "shaft_end_gap_mm": 50.8,
    "body_eyelet": {
        "outer_diameter_mm": 19.9,
        "bore_diameter_mm": 12.7,
    },
    "body_end_solid": {
        "length_mm": 0.0,
        "diameter_mm": 50.8,
    },
    "body_end_positive_chamber": {
        "length_mm": 0.0,
        "diameter_mm": 50.8,
    },
    "positive_annular_chamber": {
        "length_mm": 0.0,
        "inner_diameter_mm": 50.8,
        "outer_diameter_mm": 50.8,
    },
    "swept_air_chamber": {
        "length_mm": 70.0,
        "diameter_mm": 50.8,
    },
    "negative_chamber_extension": {
        "length_mm": 20.0,
        "diameter_mm": 50.8,
    },
    "negative_annular_chamber": {
        "length_mm": 0.0,
        "inner_diameter_mm": 50.8,
        "outer_diameter_mm": 50.8,
    },
    "damper_shaft": {
        "diameter_mm": 25.4,
    },
    "positive_chamber_shaft": {
        "diameter_mm": 8.0,
    },
    "piston": {
        "diameter_mm": 50.8,
        "thickness_mm": 5.0,
    },
    "shaft_eyelet": {
        "outer_diameter_mm": 19.9,
        "bore_diameter_mm": 12.7,
    },
}
_DEFAULT_SHOCK_PRESETS: list[dict] = [
    {
        "preset_id": "default_xc_air",
        "name": "Default XC Air",
        "brand": "Generic",
        "category": "xc",
        "shock_type": "air",
        "sort_order": 10,
    },
    {
        "preset_id": "default_trail_enduro_air",
        "name": "Default Trail/Enduro Air",
        "brand": "Generic",
        "category": "trail_enduro",
        "shock_type": "air",
        "sort_order": 20,
    },
    {
        "preset_id": "default_dh_air",
        "name": "Default DH Air",
        "brand": "Generic",
        "category": "dh",
        "shock_type": "air",
        "sort_order": 30,
    },
]


def _parse_optional_finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _copy_shock_visual_model_defaults() -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in _DEFAULT_SHOCK_VISUAL_MODEL.items():
        out[key] = dict(value) if isinstance(value, dict) else float(value)
    return out


def _default_shock_visual_model_for_model(model: dict[str, object]) -> dict[str, object]:
    visual = _copy_shock_visual_model_defaults()
    chamber_d = max(
        1.0,
        float(model.get("air_chamber_diameter_mm", _DEFAULT_SHOCK_MODEL["air_chamber_diameter_mm"])),
    )
    damper_shaft_d = max(
        1.0,
        float(model.get("air_shaft_diameter_mm", _DEFAULT_SHOCK_MODEL["air_shaft_diameter_mm"])),
    )
    pos_shaft_d = max(
        0.0,
        float(
            model.get(
                "air_positive_shaft_diameter_mm",
                _DEFAULT_SHOCK_MODEL["air_positive_shaft_diameter_mm"],
            )
        ),
    )
    piston_t = max(
        0.0,
        float(
            model.get(
                "air_piston_head_thickness_mm",
                _DEFAULT_SHOCK_MODEL["air_piston_head_thickness_mm"],
            )
        ),
    )
    swept_len = max(
        1e-3,
        float(model.get("air_chamber_length_mm", _DEFAULT_SHOCK_MODEL["air_chamber_length_mm"])),
    )
    neg_ext_len = max(
        0.0,
        float(
            model.get(
                "air_negative_chamber_length_mm",
                _DEFAULT_SHOCK_MODEL["air_negative_chamber_length_mm"],
            )
        ),
    )
    body_gap = max(
        0.0,
        float(model.get("body_eyelet_gap_mm", _DEFAULT_SHOCK_MODEL["body_eyelet_gap_mm"])),
    )
    shaft_gap = max(
        0.0,
        float(model.get("shaft_eyelet_gap_mm", _DEFAULT_SHOCK_MODEL["shaft_eyelet_gap_mm"])),
    )
    eye_outer = max(18.0, 12.7 + 7.2)
    visual["body_end_gap_mm"] = body_gap
    visual["shaft_end_gap_mm"] = shaft_gap
    visual["body_eyelet"] = {
        "outer_diameter_mm": eye_outer,
        "bore_diameter_mm": 12.7,
    }
    visual["body_end_solid"] = {"length_mm": 0.0, "diameter_mm": chamber_d}
    visual["body_end_positive_chamber"] = {"length_mm": 0.0, "diameter_mm": chamber_d}
    visual["positive_annular_chamber"] = {
        "length_mm": 0.0,
        "inner_diameter_mm": chamber_d,
        "outer_diameter_mm": chamber_d,
    }
    visual["swept_air_chamber"] = {"length_mm": swept_len, "diameter_mm": chamber_d}
    visual["negative_chamber_extension"] = {
        "length_mm": neg_ext_len,
        "diameter_mm": chamber_d,
    }
    visual["negative_annular_chamber"] = {
        "length_mm": 0.0,
        "inner_diameter_mm": chamber_d,
        "outer_diameter_mm": chamber_d,
    }
    visual["damper_shaft"] = {"diameter_mm": damper_shaft_d}
    visual["positive_chamber_shaft"] = {"diameter_mm": pos_shaft_d}
    visual["piston"] = {"diameter_mm": chamber_d, "thickness_mm": piston_t}
    visual["shaft_eyelet"] = {
        "outer_diameter_mm": eye_outer,
        "bore_diameter_mm": 12.7,
    }
    return visual


def _normalize_shock_visual_model(raw_visual, model: dict[str, object]) -> dict[str, object]:
    visual = _default_shock_visual_model_for_model(model)
    if not isinstance(raw_visual, dict):
        return visual

    def _pair(name: str, diameter_default: float):
        current = visual.get(name)
        base = dict(current) if isinstance(current, dict) else {}
        raw = raw_visual.get(name)
        if isinstance(raw, dict):
            length = _parse_optional_finite(raw.get("length_mm"))
            diameter = _parse_optional_finite(raw.get("diameter_mm"))
            if length is not None:
                base["length_mm"] = max(0.0, length)
            if diameter is not None:
                base["diameter_mm"] = max(0.0, diameter)
        base.setdefault("length_mm", 0.0)
        base.setdefault("diameter_mm", diameter_default)
        visual[name] = base

    def _eyelet(name: str):
        current = visual.get(name)
        base = dict(current) if isinstance(current, dict) else {}
        raw = raw_visual.get(name)
        if isinstance(raw, dict):
            outer = _parse_optional_finite(raw.get("outer_diameter_mm"))
            bore = _parse_optional_finite(raw.get("bore_diameter_mm"))
            if outer is not None:
                base["outer_diameter_mm"] = max(1.0, outer)
            if bore is not None:
                base["bore_diameter_mm"] = max(0.0, bore)
        if base.get("outer_diameter_mm", 0.0) < base.get("bore_diameter_mm", 0.0):
            base["outer_diameter_mm"] = base["bore_diameter_mm"]
        visual[name] = base

    chamber_d = max(
        1.0,
        float(model.get("air_chamber_diameter_mm", _DEFAULT_SHOCK_MODEL["air_chamber_diameter_mm"])),
    )
    _eyelet("body_eyelet")
    _pair("body_end_solid", chamber_d)
    _pair("body_end_positive_chamber", chamber_d)
    _pair("swept_air_chamber", chamber_d)
    _pair("negative_chamber_extension", chamber_d)
    _eyelet("shaft_eyelet")

    def _annulus(name: str):
        annulus = visual.get(name)
        annulus_base = dict(annulus) if isinstance(annulus, dict) else {}
        annulus_raw = raw_visual.get(name)
        if isinstance(annulus_raw, dict):
            length = _parse_optional_finite(annulus_raw.get("length_mm"))
            inner_d = _parse_optional_finite(annulus_raw.get("inner_diameter_mm"))
            outer_d = _parse_optional_finite(annulus_raw.get("outer_diameter_mm"))
            if length is not None:
                annulus_base["length_mm"] = max(0.0, length)
            if inner_d is not None:
                annulus_base["inner_diameter_mm"] = max(0.0, inner_d)
            if outer_d is not None:
                annulus_base["outer_diameter_mm"] = max(0.0, outer_d)
        annulus_base.setdefault("length_mm", 0.0)
        annulus_base.setdefault("inner_diameter_mm", chamber_d)
        annulus_base.setdefault("outer_diameter_mm", chamber_d)
        annulus_base["outer_diameter_mm"] = max(
            annulus_base["inner_diameter_mm"], annulus_base["outer_diameter_mm"]
        )
        visual[name] = annulus_base

    _annulus("positive_annular_chamber")
    _annulus("negative_annular_chamber")

    piston = visual.get("piston")
    piston_base = dict(piston) if isinstance(piston, dict) else {}
    piston_raw = raw_visual.get("piston")
    if isinstance(piston_raw, dict):
        diameter = _parse_optional_finite(piston_raw.get("diameter_mm"))
        thickness = _parse_optional_finite(piston_raw.get("thickness_mm"))
        if diameter is not None:
            piston_base["diameter_mm"] = max(1.0, diameter)
        if thickness is not None:
            piston_base["thickness_mm"] = max(0.0, thickness)
    piston_base.setdefault("diameter_mm", chamber_d)
    piston_base.setdefault("thickness_mm", 0.0)
    visual["piston"] = piston_base

    for name, key in [
        ("damper_shaft", "air_shaft_diameter_mm"),
        ("positive_chamber_shaft", "air_positive_shaft_diameter_mm"),
    ]:
        current = visual.get(name)
        base = dict(current) if isinstance(current, dict) else {}
        raw = raw_visual.get(name)
        if isinstance(raw, dict):
            diameter = _parse_optional_finite(raw.get("diameter_mm"))
            if diameter is not None:
                base["diameter_mm"] = max(0.0, diameter)
        base.setdefault(
            "diameter_mm",
            max(0.0, float(model.get(key, _DEFAULT_SHOCK_MODEL[key]))),
        )
        visual[name] = base

    for key, legacy_key in [
        ("body_end_gap_mm", "body_eyelet_gap_mm"),
        ("shaft_end_gap_mm", "shaft_eyelet_gap_mm"),
    ]:
        value = _parse_optional_finite(raw_visual.get(key))
        if value is None:
            value = _parse_optional_finite(model.get(legacy_key))
        if value is None:
            value = float(visual.get(key) or chamber_d)
        visual[key] = max(0.0, value)

    return visual


def _build_full_default_shock_model(overrides: Optional[dict] = None) -> dict[str, object]:
    model: dict[str, object] = dict(_DEFAULT_SHOCK_MODEL)
    if isinstance(overrides, dict):
        for key in _DEFAULT_SHOCK_MODEL.keys():
            value = _parse_optional_finite(overrides.get(key))
            if value is not None:
                model[key] = float(value)
    model["visual_model"] = _normalize_shock_visual_model(
        overrides.get("visual_model") if isinstance(overrides, dict) else None,
        model,
    )
    return model


def _coerce_full_shock_visual_model(raw_visual) -> dict[str, object]:
    if not isinstance(raw_visual, dict):
        raise ValueError("Shock preset visual_model must be a dict")

    out: dict[str, object] = {}
    for key, default_value in _DEFAULT_SHOCK_VISUAL_MODEL.items():
        raw_value = raw_visual.get(key)
        if isinstance(default_value, dict):
            if not isinstance(raw_value, dict):
                raise ValueError(f"Shock preset visual_model.{key} must be a dict")
            nested: dict[str, float] = {}
            for nested_key in default_value.keys():
                parsed = _parse_optional_finite(raw_value.get(nested_key))
                if parsed is None:
                    raise ValueError(
                        f"Shock preset visual_model.{key}.{nested_key} is required"
                    )
                nested[nested_key] = float(parsed)
            out[key] = nested
            continue

        parsed = _parse_optional_finite(raw_value)
        if parsed is None:
            raise ValueError(f"Shock preset visual_model.{key} is required")
        out[key] = float(parsed)

    return out


def _coerce_full_shock_model_doc(raw_model) -> dict[str, object]:
    if not isinstance(raw_model, dict):
        raise ValueError("Shock preset shock_model must be a dict")

    model: dict[str, object] = {}
    for key in _DEFAULT_SHOCK_MODEL.keys():
        parsed = _parse_optional_finite(raw_model.get(key))
        if parsed is None:
            raise ValueError(f"Shock preset shock_model.{key} is required")
        model[key] = float(parsed)

    model["visual_model"] = _coerce_full_shock_visual_model(raw_model.get("visual_model"))
    return model


async def _ensure_default_shock_presets():
    presets = shock_presets_col()
    now = datetime.utcnow()
    for preset in _DEFAULT_SHOCK_PRESETS:
        preset_id = str(preset.get("preset_id") or "").strip()
        if not preset_id:
            continue
        doc = {
            "preset_id": preset_id,
            "name": preset.get("name"),
            "brand": preset.get("brand"),
            "category": preset.get("category"),
            "shock_type": preset.get("shock_type") or "air",
            "sort_order": int(preset.get("sort_order") or 100),
            "shock_model": _build_full_default_shock_model(
                preset.get("shock_model") if isinstance(preset.get("shock_model"), dict) else None
            ),
            "updated_at": now,
        }
        try:
            await presets.update_one(
                {"preset_id": preset_id},
                {
                    "$set": doc,
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
        except DuplicateKeyError:
            continue


def _shock_preset_doc_to_out(doc: dict) -> ShockPresetOut:
    shock_type = str(doc.get("shock_type") or "air").strip().lower()
    if shock_type not in {"air", "coil"}:
        shock_type = "air"
    model = _coerce_full_shock_model_doc(doc.get("shock_model"))
    return ShockPresetOut(
        id=str(doc.get("_id")),
        preset_id=str(doc.get("preset_id") or ""),
        name=str(doc.get("name") or "Shock Preset"),
        brand=(str(doc.get("brand")) if doc.get("brand") is not None else None),
        category=(str(doc.get("category")) if doc.get("category") is not None else None),
        shock_type=shock_type,  # type: ignore[arg-type]
        shock_model=ShockModel(**model),
    )


def _parse_positive_float(value) -> Optional[float]:
    parsed = _parse_optional_finite(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _parse_positive_int(value) -> Optional[int]:
    parsed = _parse_optional_finite(value)
    if parsed is None:
        return None
    rounded = int(round(parsed))
    if rounded <= 0:
        return None
    if abs(parsed - float(rounded)) > 1e-6:
        return None
    return rounded


def _get_wheel_outer_radius_mm(size_id: Optional[str]) -> Optional[float]:
    if not size_id:
        return None
    bsd = _WHEEL_BSD_MM.get(str(size_id))
    if bsd is None:
        return None
    return bsd * 0.5 + _TYRE_THICKNESS_MM


def _variant_requires_rest_pose(
    base_settings: dict,
    effective_settings: dict,
    overrides: Optional[dict] = None,
    points: Optional[List[BikePoint]] = None,
    bodies: Optional[List[RigidBody]] = None,
) -> bool:
    if (
        str(base_settings.get("front_wheel_size", "29")) != str(effective_settings.get("front_wheel_size", "29"))
        or str(base_settings.get("rear_wheel_size", "29")) != str(effective_settings.get("rear_wheel_size", "29"))
    ):
        return True

    if not isinstance(overrides, dict) or not overrides:
        return False

    if _parse_optional_finite(overrides.get("shock_eye_mm")) is not None:
        return True

    flip_chip_selected_state_ids = overrides.get("flip_chip_selected_state_ids")
    if isinstance(flip_chip_selected_state_ids, dict) and flip_chip_selected_state_ids:
        return True

    point_overrides = overrides.get("point_overrides")
    if not isinstance(point_overrides, dict) or not point_overrides:
        return False

    return any(
        str(point_id).strip() and isinstance(payload, dict)
        for point_id, payload in point_overrides.items()
    )


def _serialize_solver_step(step: SolverStep) -> dict[str, object]:
    points_payload: dict[str, list[float]] = {}
    raw_points = step.points if isinstance(step.points, dict) else {}
    for point_id, coords in raw_points.items():
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue
        try:
            x_value = float(coords[0])
            y_value = float(coords[1])
        except (TypeError, ValueError):
            continue
        points_payload[str(point_id)] = [x_value, y_value]

    return {
        "step_index": int(step.step_index),
        "shock_stroke": float(step.shock_stroke) if step.shock_stroke is not None else None,
        "shock_length": float(step.shock_length) if step.shock_length is not None else None,
        "rear_travel": float(step.rear_travel) if step.rear_travel is not None else None,
        "leverage_ratio": float(step.leverage_ratio) if step.leverage_ratio is not None else None,
        "anti_squat": float(step.anti_squat) if step.anti_squat is not None else None,
        "anti_rise": float(step.anti_rise) if step.anti_rise is not None else None,
        "shock_spring_rate": float(step.shock_spring_rate) if step.shock_spring_rate is not None else None,
        "rear_wheel_force": float(step.rear_wheel_force) if step.rear_wheel_force is not None else None,
        "points": points_payload,
    }

def _scale_optional_float(value, scale: float) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed * scale


def _scale_float_series(values, scale: float) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        scaled = _scale_optional_float(value, scale)
        if scaled is None:
            continue
        out.append(scaled)
    return out


def _scale_convergence_trace(
    raw_trace,
    scale_mm_per_px: float,
) -> dict[str, object]:
    return {
        "metric": "max_point_displacement_mm",
        "value_by_iteration_mm": _scale_float_series(
            raw_trace.get("value_by_iteration")
            if isinstance(raw_trace, dict)
            else None,
            scale_mm_per_px,
        ),
        "final_value_mm": (
            _scale_optional_float(
                raw_trace.get("final_value")
                if isinstance(raw_trace, dict)
                else None,
                scale_mm_per_px,
            )
            or 0.0
        ),
        "iterations_used": int(raw_trace.get("iterations_used", 0) or 0)
        if isinstance(raw_trace, dict)
        else 0,
    }


def _summarize_convergence_steps(convergence_steps: list[dict[str, object]]) -> dict[str, object]:
    if not convergence_steps:
        return {
            "metric": "max_point_displacement_mm",
            "step_count": 0,
            "max_iterations_used": 0,
            "worst_step_index": None,
            "worst_final_value_mm": 0.0,
        }

    worst_step = max(
        convergence_steps,
        key=lambda step: float(step.get("final_value_mm", 0.0) or 0.0),
    )
    return {
        "metric": "max_point_displacement_mm",
        "step_count": len(convergence_steps),
        "max_iterations_used": max(
            int(step.get("iterations_used", 0) or 0)
            for step in convergence_steps
        ),
        "worst_step_index": int(worst_step.get("step_index", 0) or 0),
        "worst_final_value_mm": max(
            float(step.get("final_value_mm", 0.0) or 0.0)
            for step in convergence_steps
        ),
    }


def _build_scaled_linkage_convergence_payload(
    raw_convergence,
    scale_mm_per_px: float,
    *,
    trim_pre_steps: int = 0,
) -> dict[str, object]:
    if not isinstance(raw_convergence, dict):
        return {}

    raw_steps = raw_convergence.get("steps")
    if not isinstance(raw_steps, list):
        raw_steps = []
    trim_count = max(0, int(trim_pre_steps or 0))
    trimmed_raw_steps = raw_steps[trim_count:] if trim_count > 0 else raw_steps

    scaled_steps: list[dict[str, object]] = []

    for seq_index, raw_step in enumerate(trimmed_raw_steps):
        if not isinstance(raw_step, dict):
            continue
        scaled_trace = _scale_convergence_trace(
            raw_step,
            scale_mm_per_px,
        )
        scaled_steps.append(
            {
                "step_index": seq_index,
                "shock_stroke_mm": _scale_optional_float(
                    raw_step.get("shock_stroke"),
                    scale_mm_per_px,
                ),
                "iterations_used": int(raw_step.get("iterations_used", 0) or 0),
                **scaled_trace,
            }
        )

    return {
        "metric": "max_point_displacement_mm",
        "steps": scaled_steps,
        "step_summaries": [
            {
                "step_index": int(step.get("step_index", 0) or 0),
                "shock_stroke_mm": step.get("shock_stroke_mm"),
                "iterations_used": int(step.get("iterations_used", 0) or 0),
                "final_value_mm": float(step.get("final_value_mm", 0.0) or 0.0),
            }
            for step in scaled_steps
        ],
        "summary": _summarize_convergence_steps(scaled_steps),
        "final_value_per_step_mm": [
            float(step.get("final_value_mm", 0.0) or 0.0)
            for step in scaled_steps
        ],
    }


def _derive_ground_reference_from_base_pose(
    *,
    points: List[BikePoint],
    scale_mm_per_px: float,
    base_settings: dict,
) -> tuple[Optional[dict[str, float | str]], Optional[str]]:
    if not (scale_mm_per_px > 0):
        return None, "invalid_scale"

    bb_point = next((p for p in points if p.type in ("bb", "bottom_bracket")), None)
    rear_axle_point = next((p for p in points if p.type == "rear_axle"), None)
    front_axle_point = next((p for p in points if p.type == "front_axle"), None)
    if not bb_point or not rear_axle_point or not front_axle_point:
        return None, "missing_anchor_points"

    base_rear_radius_mm = _get_wheel_outer_radius_mm(base_settings.get("rear_wheel_size", "29"))
    base_front_radius_mm = _get_wheel_outer_radius_mm(base_settings.get("front_wheel_size", "29"))
    if not base_rear_radius_mm or not base_front_radius_mm:
        return None, "invalid_base_wheel_sizes"

    base_rear_radius_px = base_rear_radius_mm / scale_mm_per_px
    base_front_radius_px = base_front_radius_mm / scale_mm_per_px

    return (
        {
            "source": "authored_base_pose",
            "bb_point_id": str(bb_point.id),
            "bb_x": float(bb_point.x),
            "rear_axle_point_id": str(rear_axle_point.id),
            "front_axle_point_id": str(front_axle_point.id),
            "base_rear_radius_px": float(base_rear_radius_px),
            "base_front_radius_px": float(base_front_radius_px),
            "rear_contact_y": float(rear_axle_point.y) + base_rear_radius_px,
            "front_contact_y": float(front_axle_point.y) + base_front_radius_px,
        },
        None,
    )


def _compute_variant_rest_pose(
    *,
    points: List[BikePoint],
    bodies: List[RigidBody],
    scale_mm_per_px: float,
    effective_settings: dict,
    ground_reference: Optional[dict[str, float | str]],
    iterations: int,
    override_point_constraints: Optional[dict[str, dict[str, float]]] = None,
) -> tuple[List[BikePoint], dict]:
    if not (scale_mm_per_px > 0):
        return points, {"mode": "rest_pose", "applied": False, "reason": "invalid_scale"}
    if not isinstance(ground_reference, dict):
        return points, {"mode": "rest_pose", "applied": False, "reason": "missing_ground_reference"}

    bb_point_id = str(ground_reference.get("bb_point_id") or "").strip()
    rear_axle_point_id = str(ground_reference.get("rear_axle_point_id") or "").strip()
    front_axle_point_id = str(ground_reference.get("front_axle_point_id") or "").strip()
    try:
        bb_x = float(ground_reference.get("bb_x"))
        rear_contact_y = float(ground_reference.get("rear_contact_y"))
        front_contact_y = float(ground_reference.get("front_contact_y"))
    except (TypeError, ValueError):
        return points, {"mode": "rest_pose", "applied": False, "reason": "invalid_ground_reference"}
    if not bb_point_id or not rear_axle_point_id or not front_axle_point_id:
        return points, {"mode": "rest_pose", "applied": False, "reason": "invalid_ground_reference"}

    new_rear_radius_mm = _get_wheel_outer_radius_mm(effective_settings.get("rear_wheel_size", "29"))
    new_front_radius_mm = _get_wheel_outer_radius_mm(effective_settings.get("front_wheel_size", "29"))
    if not new_rear_radius_mm or not new_front_radius_mm:
        return points, {"mode": "rest_pose", "applied": False, "reason": "invalid_effective_wheel_sizes"}

    new_rear_radius_px = new_rear_radius_mm / scale_mm_per_px
    new_front_radius_px = new_front_radius_mm / scale_mm_per_px

    point_constraints = {
        str(point_id): {
            key: float(value)
            for key, value in constraint.items()
            if key in {"x", "y"} and value is not None
        }
        for point_id, constraint in (override_point_constraints or {}).items()
        if isinstance(constraint, dict)
    }
    fixed_body_point_ids: set[str] = {
        str(point.id)
        for point in points
        if str(getattr(point, "type", "") or "").strip().lower() in {"fixed", "bb"}
    }
    for body in bodies or []:
        if str(getattr(body, "type", "") or "").strip().lower() != "fixed":
            continue
        for point_id in getattr(body, "point_ids", None) or []:
            point_id_str = str(point_id or "").strip()
            if point_id_str:
                fixed_body_point_ids.add(point_id_str)

    point_constraints.setdefault(bb_point_id, {})["x"] = bb_x
    point_constraints.setdefault(rear_axle_point_id, {})["y"] = (
        rear_contact_y - new_rear_radius_px
    )
    point_constraints.setdefault(front_axle_point_id, {})["y"] = (
        front_contact_y - new_front_radius_px
    )
    solved_points, debug = solve_bike_rest_pose(
        points,
        bodies,
        point_constraints=point_constraints,
        iterations=max(200, iterations * 3),
    )
    raw_pose_convergence = (
        debug.get("convergence")
        if isinstance(debug, dict)
        else None
    )
    if isinstance(raw_pose_convergence, dict):
        scaled_pose_convergence = _scale_convergence_trace(
            raw_pose_convergence,
            scale_mm_per_px,
        )
        debug["convergence"] = scaled_pose_convergence
    debug.update(
        {
            "applied": True,
            "ground_reference_source": str(ground_reference.get("source") or "authored_base_pose"),
            "base_front_contact_y": front_contact_y,
            "base_rear_contact_y": rear_contact_y,
            "target_front_axle_y": front_contact_y - new_front_radius_px,
            "target_rear_axle_y": rear_contact_y - new_rear_radius_px,
        }
    )
    return solved_points, debug


def _shock_length_mm_for_points(
    points: List[BikePoint],
    bodies: List[RigidBody],
    scale_mm_per_px: float,
) -> Optional[float]:
    if not (scale_mm_per_px > 0):
        return None
    point_by_id = {str(point.id): point for point in points}
    for body in bodies:
        if body.type != "shock":
            continue
        point_ids = [str(pid) for pid in (body.point_ids or []) if pid]
        if len(point_ids) < 2:
            continue
        a = point_by_id.get(point_ids[0])
        b = point_by_id.get(point_ids[1])
        if a is None or b is None:
            continue
        return math.hypot(float(b.x) - float(a.x), float(b.y) - float(a.y)) * scale_mm_per_px
    return None


def _rear_axle_debug_summary(
    steps: list[SolverStep],
    rear_axle_point_id: Optional[str],
    scale_mm_per_px: float,
) -> dict:
    summary: dict[str, object] = {
        "rear_axle_point_id": rear_axle_point_id,
        "step_count": len(steps or []),
        "zero_point_px": None,
        "final_point_px": None,
        "delta_px": None,
        "delta_mm": None,
        "max_abs_dy_mm": None,
        "max_abs_dx_mm": None,
        "max_rear_travel_mm_from_steps": None,
    }
    if not steps or not rear_axle_point_id or not (scale_mm_per_px > 0):
        return summary

    coords: list[tuple[float, float]] = []
    rear_travel_vals: list[float] = []
    for step in steps:
        if not isinstance(step, SolverStep):
            continue
        point_xy = step.points.get(rear_axle_point_id) if isinstance(step.points, dict) else None
        if isinstance(point_xy, (list, tuple)) and len(point_xy) >= 2:
            try:
                coords.append((float(point_xy[0]), float(point_xy[1])))
            except (TypeError, ValueError):
                pass
        if step.rear_travel is not None:
            try:
                value = float(step.rear_travel)
                if math.isfinite(value):
                    rear_travel_vals.append(value)
            except (TypeError, ValueError):
                pass

    if not coords:
        return summary

    x0, y0 = coords[0]
    x1, y1 = coords[-1]
    dx_px = x1 - x0
    dy_px = y1 - y0
    max_abs_dx_px = max(abs(x - x0) for x, _ in coords)
    max_abs_dy_px = max(abs(y - y0) for _, y in coords)
    summary.update(
        {
            "zero_point_px": {"x": x0, "y": y0},
            "final_point_px": {"x": x1, "y": y1},
            "delta_px": {"x": dx_px, "y": dy_px},
            "delta_mm": {
                "x": dx_px * scale_mm_per_px,
                "y": dy_px * scale_mm_per_px,
            },
            "max_abs_dx_mm": max_abs_dx_px * scale_mm_per_px,
            "max_abs_dy_mm": max_abs_dy_px * scale_mm_per_px,
            "max_rear_travel_mm_from_steps": (
                max(abs(v) for v in rear_travel_vals) if rear_travel_vals else None
            ),
        }
    )
    return summary


def _get_sprocket_pitch_radius_mm(teeth: Optional[int]) -> Optional[float]:
    if not teeth or teeth <= 0:
        return None
    denom = math.sin(math.pi / float(teeth))
    if not math.isfinite(denom) or denom <= 0:
        return None
    pitch_diameter = _CHAIN_PITCH_MM / denom
    if not math.isfinite(pitch_diameter) or pitch_diameter <= 0:
        return None
    return pitch_diameter * 0.5


def _normalize_shock_geometry_config(geometry: Optional[dict]) -> tuple[str, dict[str, object]]:
    geom = geometry if isinstance(geometry, dict) else {}
    raw_type = str(geom.get("shock_type") or "").strip().lower()
    shock_type = raw_type if raw_type in {"air", "coil"} else "air"

    raw_model = geom.get("shock_model")
    model: dict[str, float] = dict(_DEFAULT_SHOCK_MODEL)
    legacy_eyelet_gap_value = None
    body_eyelet_gap_value = None
    shaft_eyelet_gap_value = None
    if isinstance(raw_model, dict):
        for key in _DEFAULT_SHOCK_MODEL.keys():
            value = _parse_optional_finite(raw_model.get(key))
            if value is None:
                continue
            model[key] = value
        legacy_eyelet_gap_value = _parse_optional_finite(raw_model.get("eyelet_gap_mm"))
        body_eyelet_gap_value = _parse_optional_finite(raw_model.get("body_eyelet_gap_mm"))
        shaft_eyelet_gap_value = _parse_optional_finite(raw_model.get("shaft_eyelet_gap_mm"))

    if model["coil_rate_n_per_mm"] <= 0:
        model["coil_rate_n_per_mm"] = _DEFAULT_SHOCK_MODEL["coil_rate_n_per_mm"]
    if model["air_chamber_diameter_mm"] <= 0:
        model["air_chamber_diameter_mm"] = _DEFAULT_SHOCK_MODEL["air_chamber_diameter_mm"]
    fallback_eyelet_gap = (
        legacy_eyelet_gap_value
        if legacy_eyelet_gap_value is not None and legacy_eyelet_gap_value > 0
        else model["air_chamber_diameter_mm"]
    )
    if body_eyelet_gap_value is None or model["body_eyelet_gap_mm"] <= 0:
        model["body_eyelet_gap_mm"] = fallback_eyelet_gap
    if shaft_eyelet_gap_value is None or model["shaft_eyelet_gap_mm"] <= 0:
        model["shaft_eyelet_gap_mm"] = fallback_eyelet_gap
    if model["air_chamber_length_mm"] <= 0:
        model["air_chamber_length_mm"] = _DEFAULT_SHOCK_MODEL["air_chamber_length_mm"]
    if model["air_negative_chamber_length_mm"] <= 0:
        model["air_negative_chamber_length_mm"] = _DEFAULT_SHOCK_MODEL["air_negative_chamber_length_mm"]
    if model["air_piston_head_thickness_mm"] < 0:
        model["air_piston_head_thickness_mm"] = _DEFAULT_SHOCK_MODEL["air_piston_head_thickness_mm"]
    if model["air_initial_pressure_psi"] <= 0:
        model["air_initial_pressure_psi"] = _DEFAULT_SHOCK_MODEL["air_initial_pressure_psi"]

    min_shaft = 1.0
    max_shaft = max(1.0, model["air_chamber_diameter_mm"] - 0.5)
    if not math.isfinite(model["air_shaft_diameter_mm"]):
        model["air_shaft_diameter_mm"] = _DEFAULT_SHOCK_MODEL["air_shaft_diameter_mm"]
    model["air_shaft_diameter_mm"] = max(min_shaft, min(max_shaft, model["air_shaft_diameter_mm"]))
    if not math.isfinite(model["air_positive_shaft_diameter_mm"]):
        model["air_positive_shaft_diameter_mm"] = _DEFAULT_SHOCK_MODEL["air_positive_shaft_diameter_mm"]
    model["air_positive_shaft_diameter_mm"] = max(
        0.0,
        min(
            max_shaft,
            model["air_shaft_diameter_mm"],
            model["air_positive_shaft_diameter_mm"],
        ),
    )

    if model["air_reference_temp_c"] <= -273.0:
        model["air_reference_temp_c"] = _DEFAULT_SHOCK_MODEL["air_reference_temp_c"]
    if model["air_cold_temp_c"] <= -273.0:
        model["air_cold_temp_c"] = _DEFAULT_SHOCK_MODEL["air_cold_temp_c"]
    if model["air_hot_temp_c"] <= -273.0:
        model["air_hot_temp_c"] = _DEFAULT_SHOCK_MODEL["air_hot_temp_c"]

    model["visual_model"] = _normalize_shock_visual_model(
        raw_model.get("visual_model") if isinstance(raw_model, dict) else None,
        model,
    )

    return shock_type, model


def _compute_shock_force_and_rate_series(
    shock_stroke_series_mm: list[Optional[float]],
    shock_type: str,
    model: dict[str, float],
    temp_c: Optional[float] = None,
) -> tuple[list[Optional[float]], list[Optional[float]]]:
    force_series: list[Optional[float]] = []
    rate_series: list[Optional[float]] = []
    if not shock_stroke_series_mm:
        return force_series, rate_series

    if shock_type == "coil":
        rate = max(1e-9, float(model.get("coil_rate_n_per_mm", _DEFAULT_SHOCK_MODEL["coil_rate_n_per_mm"])))
        preload = float(model.get("coil_preload_n", _DEFAULT_SHOCK_MODEL["coil_preload_n"]))
        for stroke in shock_stroke_series_mm:
            s = _parse_optional_finite(stroke)
            if s is None:
                force_series.append(None)
                rate_series.append(None)
                continue
            s_clamped = max(0.0, s)
            force_series.append(preload + rate * s_clamped)
            rate_series.append(rate)
        return force_series, rate_series

    d_chamber_mm = max(
        1e-6,
        float(model.get("air_chamber_diameter_mm", _DEFAULT_SHOCK_MODEL["air_chamber_diameter_mm"])),
    )
    l_pos_mm = max(
        1e-6,
        float(model.get("air_chamber_length_mm", _DEFAULT_SHOCK_MODEL["air_chamber_length_mm"])),
    )
    l_neg_mm = max(
        1e-6,
        float(
            model.get(
                "air_negative_chamber_length_mm",
                _DEFAULT_SHOCK_MODEL["air_negative_chamber_length_mm"],
            )
        ),
    )
    piston_t_mm = max(
        0.0,
        float(
            model.get(
                "air_piston_head_thickness_mm",
                _DEFAULT_SHOCK_MODEL["air_piston_head_thickness_mm"],
            )
        ),
    )
    d_shaft_mm = max(
        1e-6,
        float(model.get("air_shaft_diameter_mm", _DEFAULT_SHOCK_MODEL["air_shaft_diameter_mm"])),
    )
    d_shaft_mm = min(d_shaft_mm, max(1e-6, d_chamber_mm - 0.5))
    d_pos_shaft_mm = max(
        0.0,
        float(
            model.get(
                "air_positive_shaft_diameter_mm",
                _DEFAULT_SHOCK_MODEL["air_positive_shaft_diameter_mm"],
            )
        ),
    )
    d_pos_shaft_mm = min(d_pos_shaft_mm, d_shaft_mm, max(0.0, d_chamber_mm - 0.5))
    p0_psi = max(
        1e-6,
        float(model.get("air_initial_pressure_psi", _DEFAULT_SHOCK_MODEL["air_initial_pressure_psi"])),
    )

    t_ref_c = float(model.get("air_reference_temp_c", _DEFAULT_SHOCK_MODEL["air_reference_temp_c"]))
    t_eval_c = float(temp_c if temp_c is not None else t_ref_c)

    t_ref_k = t_ref_c + 273.15
    t_eval_k = t_eval_c + 273.15
    if t_ref_k <= 0 or t_eval_k <= 0:
        return [None for _ in shock_stroke_series_mm], [None for _ in shock_stroke_series_mm]

    chamber_area_m2 = math.pi * ((d_chamber_mm * 1e-3) ** 2) * 0.25
    neg_shaft_area_m2 = math.pi * ((d_shaft_mm * 1e-3) ** 2) * 0.25
    pos_shaft_area_m2 = math.pi * ((d_pos_shaft_mm * 1e-3) ** 2) * 0.25
    pos_area_m2 = chamber_area_m2 - pos_shaft_area_m2
    neg_area_m2 = chamber_area_m2 - neg_shaft_area_m2
    if (
        not math.isfinite(pos_area_m2)
        or not math.isfinite(neg_area_m2)
        or pos_area_m2 <= 0
        or neg_area_m2 <= 0
    ):
        return [None for _ in shock_stroke_series_mm], [None for _ in shock_stroke_series_mm]

    pos_len_eff_mm = max(1e-3, l_pos_mm - piston_t_mm)
    neg_len_eff_mm = max(1e-3, l_neg_mm - piston_t_mm)
    v_pos0_m3 = pos_area_m2 * (pos_len_eff_mm * 1e-3)
    v_neg0_m3 = neg_area_m2 * (neg_len_eff_mm * 1e-3)
    if not math.isfinite(v_pos0_m3) or not math.isfinite(v_neg0_m3) or v_pos0_m3 <= 0 or v_neg0_m3 <= 0:
        return [None for _ in shock_stroke_series_mm], [None for _ in shock_stroke_series_mm]

    p0_abs_pa = (p0_psi + 14.6959) * _PSI_TO_PA
    temp_ratio = t_eval_k / t_ref_k
    c_pos0 = p0_abs_pa * v_pos0_m3 * temp_ratio
    c_neg0 = p0_abs_pa * v_neg0_m3 * temp_ratio
    min_pos_m3 = v_pos0_m3 * 0.02
    min_neg_m3 = v_neg0_m3 * 0.02

    def _volumes_at(stroke_mm: float) -> tuple[float, float]:
        stroke_m = max(0.0, float(stroke_mm)) * 1e-3
        return (
            v_pos0_m3 - pos_area_m2 * stroke_m,
            v_neg0_m3 + neg_area_m2 * stroke_m,
        )

    def _force_at(stroke_mm: float) -> Optional[float]:
        s = max(0.0, float(stroke_mm))
        v_pos_m3, v_neg_m3 = _volumes_at(s)
        if v_pos_m3 <= min_pos_m3 or v_neg_m3 <= min_neg_m3:
            return None
        p_pos = c_pos0 / v_pos_m3
        p_neg = c_neg0 / v_neg_m3

        if not (math.isfinite(p_pos) and math.isfinite(p_neg)):
            return None
        force_n = p_pos * pos_area_m2 - p_neg * neg_area_m2
        return force_n if math.isfinite(force_n) else None

    parsed_strokes: list[Optional[float]] = [
        (_parse_optional_finite(stroke) if stroke is not None else None)
        for stroke in shock_stroke_series_mm
    ]

    for stroke in parsed_strokes:
        if stroke is None:
            force_series.append(None)
            continue
        force_series.append(_force_at(stroke))

    max_stroke_mm = max((max(0.0, float(s)) for s in parsed_strokes if s is not None), default=0.0)
    delta_mm = 0.1
    for stroke in parsed_strokes:
        if stroke is None:
            rate_series.append(None)
            continue
        s0 = max(0.0, float(stroke))
        s_minus = max(0.0, s0 - delta_mm)
        s_plus = min(max_stroke_mm + delta_mm, s0 + delta_mm)
        if s_plus <= s_minus + 1e-9:
            rate_series.append(None)
            continue
        f_minus = _force_at(s_minus)
        f_plus = _force_at(s_plus)
        if f_minus is None or f_plus is None:
            rate_series.append(None)
            continue
        rate = (f_plus - f_minus) / (s_plus - s_minus)
        rate_series.append(rate if math.isfinite(rate) else None)

    return force_series, rate_series


def _compute_rear_wheel_force_series(
    rear_travel_series: list[Optional[float]],
    rear_wheel_force_n_series: list[Optional[float]],
) -> list[Optional[float]]:
    length = min(len(rear_travel_series), len(rear_wheel_force_n_series))
    if length <= 0:
        return []
    out: list[Optional[float]] = []
    for idx in range(length):
        x0 = _parse_optional_finite(rear_travel_series[idx])
        y0 = _parse_optional_finite(rear_wheel_force_n_series[idx])
        prev_idx: Optional[int] = None
        next_idx: Optional[int] = None

        for j in range(idx - 1, -1, -1):
            x_prev = _parse_optional_finite(rear_travel_series[j])
            y_prev = _parse_optional_finite(rear_wheel_force_n_series[j])
            if x_prev is None or y_prev is None:
                continue
            if x0 is not None and abs(float(x0) - float(x_prev)) <= 1e-9:
                continue
            prev_idx = j
            break

        for j in range(idx + 1, length):
            x_next = _parse_optional_finite(rear_travel_series[j])
            y_next = _parse_optional_finite(rear_wheel_force_n_series[j])
            if x_next is None or y_next is None:
                continue
            if x0 is not None and abs(float(x_next) - float(x0)) <= 1e-9:
                continue
            next_idx = j
            break

        if prev_idx is not None and next_idx is not None:
            x_prev = float(rear_travel_series[prev_idx])  # type: ignore[arg-type]
            y_prev = float(rear_wheel_force_n_series[prev_idx])  # type: ignore[arg-type]
            x_next = float(rear_travel_series[next_idx])  # type: ignore[arg-type]
            y_next = float(rear_wheel_force_n_series[next_idx])  # type: ignore[arg-type]
            dx = x_next - x_prev
            value = (y_next - y_prev) / dx if abs(dx) > 1e-9 else None
            out.append(value if value is not None and math.isfinite(value) else None)
            continue

        if x0 is not None and y0 is not None and next_idx is not None:
            x_next = float(rear_travel_series[next_idx])  # type: ignore[arg-type]
            y_next = float(rear_wheel_force_n_series[next_idx])  # type: ignore[arg-type]
            dx = x_next - float(x0)
            value = (y_next - float(y0)) / dx if abs(dx) > 1e-9 else None
            out.append(value if value is not None and math.isfinite(value) else None)
            continue

        if x0 is not None and y0 is not None and prev_idx is not None:
            x_prev = float(rear_travel_series[prev_idx])  # type: ignore[arg-type]
            y_prev = float(rear_wheel_force_n_series[prev_idx])  # type: ignore[arg-type]
            dx = float(x0) - x_prev
            value = (float(y0) - y_prev) / dx if abs(dx) > 1e-9 else None
            out.append(value if value is not None and math.isfinite(value) else None)
            continue

        out.append(None)
    return out


def _compute_rear_wheel_force_n_series(
    shock_force_series: list[Optional[float]],
    leverage_ratio_series: list[Optional[float]],
) -> list[Optional[float]]:
    length = min(len(shock_force_series), len(leverage_ratio_series))
    if length <= 0:
        return []
    out: list[Optional[float]] = []
    for idx in range(length):
        shock_force = _parse_optional_finite(shock_force_series[idx])
        leverage = _parse_optional_finite(leverage_ratio_series[idx])
        if shock_force is None or leverage is None:
            out.append(None)
            continue
        if not (abs(leverage) > 1e-9):
            out.append(None)
            continue
        value = shock_force / leverage
        out.append(value if math.isfinite(value) else None)
    return out


def _compute_external_tangents(
    c1: tuple[float, float],
    r1: float,
    c2: tuple[float, float],
    r2: float,
) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    if not (r1 > 0 and r2 > 0):
        return []
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    dist = math.hypot(dx, dy)
    if dist <= 1e-9:
        return []
    radius_delta = r2 - r1
    if dist <= abs(radius_delta) + 1e-6:
        return []

    a = radius_delta / dist
    b_sq = 1.0 - a * a
    if b_sq < 0:
        return []
    b = math.sqrt(max(0.0, b_sq))
    inv_dist = 1.0 / dist
    normals = [
        (
            (a * dx - b * dy) * inv_dist,
            (a * dy + b * dx) * inv_dist,
        ),
        (
            (a * dx + b * dy) * inv_dist,
            (a * dy - b * dx) * inv_dist,
        ),
    ]

    candidates: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for nx, ny in normals:
        p1 = (c1[0] + nx * r1, c1[1] + ny * r1)
        p2 = (c2[0] + nx * r2, c2[1] + ny * r2)
        score = p1[1] + p2[1]
        candidates.append((p1, p2, score))
    return candidates


def _compute_top_external_tangent(
    c1: tuple[float, float],
    r1: float,
    c2: tuple[float, float],
    r2: float,
) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    candidates = _compute_external_tangents(c1, r1, c2, r2)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[2])
    return candidates[0][0], candidates[0][1]


def _signed_distance_to_line(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> Optional[float]:
    x, y = point
    x1, y1 = a
    x2, y2 = b
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return None
    return ((x - x1) * dy - (y - y1) * dx) / length


def _pick_tangent_closest_to_reference_line(
    candidates: list[tuple[tuple[float, float], tuple[float, float], float]],
    reference_a: tuple[float, float],
    reference_b: tuple[float, float],
    endpoint: str,
) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    if not candidates:
        return None
    point_idx = 0 if endpoint == "p1" else 1
    ranked: list[tuple[float, float, tuple[tuple[float, float], tuple[float, float]]]] = []
    for p1, p2, score in candidates:
        target = p1 if point_idx == 0 else p2
        distance = _signed_distance_to_line(target, reference_a, reference_b)
        abs_distance = abs(distance) if distance is not None and math.isfinite(distance) else math.inf
        ranked.append((abs_distance, score, (p1, p2)))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2] if ranked else None


def _pick_top_tangent(
    candidates: list[tuple[tuple[float, float], tuple[float, float], float]],
    endpoint: str,
) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    if not candidates:
        return None
    point_idx = 0 if endpoint == "p1" else 1
    ranked: list[tuple[float, float, tuple[tuple[float, float], tuple[float, float]]]] = []
    for p1, p2, score in candidates:
        target = p1 if point_idx == 0 else p2
        y = target[1] if math.isfinite(target[1]) else math.inf
        ranked.append((y, score, (p1, p2)))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2] if ranked else None


def _build_drivetrain_force_segment(
    chainring_center: tuple[float, float],
    chainring_radius: float,
    cassette_center: tuple[float, float],
    cassette_radius: float,
    idler_center: Optional[tuple[float, float]] = None,
    idler_radius: Optional[float] = None,
) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    use_idler = (
        idler_center is not None
        and idler_radius is not None
        and math.isfinite(idler_center[0])
        and math.isfinite(idler_center[1])
        and idler_radius > 0
    )
    if use_idler:
        rear_candidates = _compute_external_tangents(
            idler_center, idler_radius, cassette_center, cassette_radius
        )
        rear_segment = _pick_top_tangent(rear_candidates, endpoint="p2")
        if rear_segment is not None:
            return rear_segment
        return None
    return _compute_top_external_tangent(
        chainring_center,
        chainring_radius,
        cassette_center,
        cassette_radius,
    )


def _intersect_infinite_lines(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> Optional[tuple[float, float]]:
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if not math.isfinite(denom) or abs(denom) < 1e-9:
        return None
    det_a = x1 * y2 - y1 * x2
    det_b = x3 * y4 - y3 * x4
    x = (det_a * (x3 - x4) - (x1 - x2) * det_b) / denom
    y = (det_a * (y3 - y4) - (y1 - y2) * det_b) / denom
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _intersect_line_with_vertical(
    a: tuple[float, float],
    b: tuple[float, float],
    x_vertical: float,
) -> Optional[tuple[float, float]]:
    x1, y1 = a
    x2, y2 = b
    dx = x2 - x1
    if abs(dx) < 1e-9:
        if abs(x_vertical - x1) < 1e-9:
            return x_vertical, y1
        return None
    t = (x_vertical - x1) / dx
    y = y1 + t * (y2 - y1)
    if not math.isfinite(y):
        return None
    return x_vertical, y


def _rotate_about_anchor(
    point: tuple[float, float],
    anchor: tuple[float, float],
    cos_t: float,
    sin_t: float,
) -> tuple[float, float]:
    dx = point[0] - anchor[0]
    dy = point[1] - anchor[1]
    return (
        anchor[0] + dx * cos_t - dy * sin_t,
        anchor[1] + dx * sin_t + dy * cos_t,
    )


def _compute_anti_squat_series(
    steps: list[SolverStep],
    instant_center_solver: list[dict[str, Optional[float]]],
    trim_index: int,
    rear_axle_point_id: Optional[str],
    front_axle_point_id: Optional[str],
    bb_point_id: Optional[str],
    idler_point_id: Optional[str],
    scale_mm_per_px: float,
    settings: dict,
) -> list[Optional[float]]:
    if not steps:
        return []
    if not rear_axle_point_id or not front_axle_point_id or not bb_point_id:
        return []
    if not (scale_mm_per_px > 0):
        return []

    chainring_teeth = _parse_positive_int(settings.get("drivetrain_chainring_teeth"))
    cassette_teeth = _parse_positive_int(settings.get("drivetrain_cassette_teeth"))
    if not chainring_teeth or not cassette_teeth:
        return []
    idler_teeth = _parse_positive_int(settings.get("drivetrain_idler_teeth"))
    chainring_radius_mm = _get_sprocket_pitch_radius_mm(chainring_teeth)
    cassette_radius_mm = _get_sprocket_pitch_radius_mm(cassette_teeth)
    if not chainring_radius_mm or not cassette_radius_mm:
        return []
    idler_radius_mm = _get_sprocket_pitch_radius_mm(idler_teeth) if idler_teeth else None

    rear_wheel_size = str(settings.get("rear_wheel_size", "29"))
    front_wheel_size = str(settings.get("front_wheel_size", "29"))
    rear_wheel_radius_mm = _get_wheel_outer_radius_mm(rear_wheel_size)
    front_wheel_radius_mm = _get_wheel_outer_radius_mm(front_wheel_size)
    if not rear_wheel_radius_mm or not front_wheel_radius_mm:
        return []

    frame_cg_x_mm = _parse_optional_finite(settings.get("frame_cg_x_mm"))
    frame_cg_y_mm = _parse_optional_finite(settings.get("frame_cg_y_mm"))
    frame_mass_kg = _parse_positive_float(settings.get("frame_mass_kg"))
    rider_cg_x_mm = _parse_optional_finite(settings.get("rider_cg_x_mm"))
    rider_cg_y_mm = _parse_optional_finite(settings.get("rider_cg_y_mm"))
    rider_mass_kg = _parse_positive_float(settings.get("rider_mass_kg"))
    if (
        frame_cg_x_mm is None
        or frame_cg_y_mm is None
        or frame_mass_kg is None
        or rider_cg_x_mm is None
        or rider_cg_y_mm is None
        or rider_mass_kg is None
    ):
        return []

    chainring_radius_px = chainring_radius_mm / scale_mm_per_px
    cassette_radius_px = cassette_radius_mm / scale_mm_per_px
    idler_radius_px = idler_radius_mm / scale_mm_per_px if idler_radius_mm else None
    rear_wheel_radius_px = rear_wheel_radius_mm / scale_mm_per_px
    front_wheel_radius_px = front_wheel_radius_mm / scale_mm_per_px

    series: list[Optional[float]] = []
    start_index = max(0, min(trim_index, len(steps)))
    for idx in range(start_index, len(steps)):
        step = steps[idx]
        rear = step.points.get(rear_axle_point_id)
        front = step.points.get(front_axle_point_id)
        bb = step.points.get(bb_point_id)
        idler = step.points.get(idler_point_id) if idler_point_id else None
        if not rear or not front or not bb:
            series.append(None)
            continue

        if idx >= len(instant_center_solver):
            series.append(None)
            continue
        ic = instant_center_solver[idx]
        x_ic = _parse_optional_finite(ic.get("x") if isinstance(ic, dict) else None)
        y_ic = _parse_optional_finite(ic.get("y") if isinstance(ic, dict) else None)
        if x_ic is None or y_ic is None:
            series.append(None)
            continue

        rear_xy = (float(rear[0]), float(rear[1]))
        front_xy = (float(front[0]), float(front[1]))
        bb_xy = (float(bb[0]), float(bb[1]))
        ic_xy = (x_ic, y_ic)

        frame_cg = (
            bb_xy[0] + frame_cg_x_mm / scale_mm_per_px,
            bb_xy[1] + frame_cg_y_mm / scale_mm_per_px,
        )
        rider_cg = (
            bb_xy[0] + rider_cg_x_mm / scale_mm_per_px,
            bb_xy[1] + rider_cg_y_mm / scale_mm_per_px,
        )
        total_mass = frame_mass_kg + rider_mass_kg
        if total_mass <= 0:
            series.append(None)
            continue
        combined_cg = (
            (frame_cg[0] * frame_mass_kg + rider_cg[0] * rider_mass_kg) / total_mass,
            (frame_cg[1] * frame_mass_kg + rider_cg[1] * rider_mass_kg) / total_mass,
        )

        rear_contact = (rear_xy[0], rear_xy[1] + rear_wheel_radius_px)
        front_contact = (front_xy[0], front_xy[1] + front_wheel_radius_px)

        # Level each frame to gravity by making the contact patch line horizontal.
        gdx = front_contact[0] - rear_contact[0]
        gdy = front_contact[1] - rear_contact[1]
        glen = math.hypot(gdx, gdy)
        if glen <= 1e-9:
            series.append(None)
            continue
        rotate_angle = -math.atan2(gdy, gdx)
        cos_t = math.cos(rotate_angle)
        sin_t = math.sin(rotate_angle)

        rear_lvl = _rotate_about_anchor(rear_xy, rear_contact, cos_t, sin_t)
        front_lvl = _rotate_about_anchor(front_xy, rear_contact, cos_t, sin_t)
        bb_lvl = _rotate_about_anchor(bb_xy, rear_contact, cos_t, sin_t)
        idler_lvl = (
            _rotate_about_anchor((float(idler[0]), float(idler[1])), rear_contact, cos_t, sin_t)
            if idler and idler_radius_px
            else None
        )
        ic_lvl = _rotate_about_anchor(ic_xy, rear_contact, cos_t, sin_t)
        cg_lvl = _rotate_about_anchor(combined_cg, rear_contact, cos_t, sin_t)
        rear_contact_lvl = _rotate_about_anchor(rear_contact, rear_contact, cos_t, sin_t)
        front_contact_lvl = _rotate_about_anchor(front_contact, rear_contact, cos_t, sin_t)

        tangent = _build_drivetrain_force_segment(
            bb_lvl,
            chainring_radius_px,
            rear_lvl,
            cassette_radius_px,
            idler_center=idler_lvl,
            idler_radius=idler_radius_px,
        )
        if tangent is None:
            series.append(None)
            continue
        ifc = _intersect_infinite_lines(tangent[0], tangent[1], rear_lvl, ic_lvl)
        if ifc is None:
            series.append(None)
            continue
        as_point = _intersect_line_with_vertical(
            rear_contact_lvl, ifc, front_contact_lvl[0]
        )
        if as_point is None:
            series.append(None)
            continue

        ground_y = rear_contact_lvl[1]
        cg_height = ground_y - cg_lvl[1]
        anti_squat_height = ground_y - as_point[1]
        if cg_height <= 1e-9 or not math.isfinite(anti_squat_height):
            series.append(None)
            continue
        anti_squat = (anti_squat_height / cg_height) * 100.0
        series.append(float(anti_squat) if math.isfinite(anti_squat) else None)

    return series


def _compute_anti_rise_series(
    steps: list[SolverStep],
    instant_center_solver: list[dict[str, Optional[float]]],
    trim_index: int,
    rear_axle_point_id: Optional[str],
    front_axle_point_id: Optional[str],
    bb_point_id: Optional[str],
    scale_mm_per_px: float,
    settings: dict,
) -> list[Optional[float]]:
    if not steps:
        return []
    if not rear_axle_point_id or not front_axle_point_id or not bb_point_id:
        return []
    if not (scale_mm_per_px > 0):
        return []

    rear_wheel_size = str(settings.get("rear_wheel_size", "29"))
    front_wheel_size = str(settings.get("front_wheel_size", "29"))
    rear_wheel_radius_mm = _get_wheel_outer_radius_mm(rear_wheel_size)
    front_wheel_radius_mm = _get_wheel_outer_radius_mm(front_wheel_size)
    if not rear_wheel_radius_mm or not front_wheel_radius_mm:
        return []

    frame_cg_x_mm = _parse_optional_finite(settings.get("frame_cg_x_mm"))
    frame_cg_y_mm = _parse_optional_finite(settings.get("frame_cg_y_mm"))
    frame_mass_kg = _parse_positive_float(settings.get("frame_mass_kg"))
    rider_cg_x_mm = _parse_optional_finite(settings.get("rider_cg_x_mm"))
    rider_cg_y_mm = _parse_optional_finite(settings.get("rider_cg_y_mm"))
    rider_mass_kg = _parse_positive_float(settings.get("rider_mass_kg"))
    if (
        frame_cg_x_mm is None
        or frame_cg_y_mm is None
        or frame_mass_kg is None
        or rider_cg_x_mm is None
        or rider_cg_y_mm is None
        or rider_mass_kg is None
    ):
        return []

    rear_wheel_radius_px = rear_wheel_radius_mm / scale_mm_per_px
    front_wheel_radius_px = front_wheel_radius_mm / scale_mm_per_px

    series: list[Optional[float]] = []
    start_index = max(0, min(trim_index, len(steps)))
    for idx in range(start_index, len(steps)):
        step = steps[idx]
        rear = step.points.get(rear_axle_point_id)
        front = step.points.get(front_axle_point_id)
        bb = step.points.get(bb_point_id)
        if not rear or not front or not bb:
            series.append(None)
            continue

        if idx >= len(instant_center_solver):
            series.append(None)
            continue
        ic = instant_center_solver[idx]
        x_ic = _parse_optional_finite(ic.get("x") if isinstance(ic, dict) else None)
        y_ic = _parse_optional_finite(ic.get("y") if isinstance(ic, dict) else None)
        if x_ic is None or y_ic is None:
            series.append(None)
            continue

        rear_xy = (float(rear[0]), float(rear[1]))
        front_xy = (float(front[0]), float(front[1]))
        bb_xy = (float(bb[0]), float(bb[1]))
        ic_xy = (x_ic, y_ic)

        frame_cg = (
            bb_xy[0] + frame_cg_x_mm / scale_mm_per_px,
            bb_xy[1] + frame_cg_y_mm / scale_mm_per_px,
        )
        rider_cg = (
            bb_xy[0] + rider_cg_x_mm / scale_mm_per_px,
            bb_xy[1] + rider_cg_y_mm / scale_mm_per_px,
        )
        total_mass = frame_mass_kg + rider_mass_kg
        if total_mass <= 0:
            series.append(None)
            continue
        combined_cg = (
            (frame_cg[0] * frame_mass_kg + rider_cg[0] * rider_mass_kg) / total_mass,
            (frame_cg[1] * frame_mass_kg + rider_cg[1] * rider_mass_kg) / total_mass,
        )

        rear_contact = (rear_xy[0], rear_xy[1] + rear_wheel_radius_px)
        front_contact = (front_xy[0], front_xy[1] + front_wheel_radius_px)

        # Level each frame to gravity by making the contact patch line horizontal.
        gdx = front_contact[0] - rear_contact[0]
        gdy = front_contact[1] - rear_contact[1]
        glen = math.hypot(gdx, gdy)
        if glen <= 1e-9:
            series.append(None)
            continue
        rotate_angle = -math.atan2(gdy, gdx)
        cos_t = math.cos(rotate_angle)
        sin_t = math.sin(rotate_angle)

        ic_lvl = _rotate_about_anchor(ic_xy, rear_contact, cos_t, sin_t)
        cg_lvl = _rotate_about_anchor(combined_cg, rear_contact, cos_t, sin_t)
        rear_contact_lvl = _rotate_about_anchor(rear_contact, rear_contact, cos_t, sin_t)
        front_contact_lvl = _rotate_about_anchor(front_contact, rear_contact, cos_t, sin_t)

        # Traditional anti-rise: rear contact line through selected brake-carrier IC.
        ar_point = _intersect_line_with_vertical(
            rear_contact_lvl, ic_lvl, front_contact_lvl[0]
        )
        if ar_point is None:
            series.append(None)
            continue

        ground_y = rear_contact_lvl[1]
        cg_height = ground_y - cg_lvl[1]
        anti_rise_height = ground_y - ar_point[1]
        if cg_height <= 1e-9 or not math.isfinite(anti_rise_height):
            series.append(None)
            continue
        anti_rise = (anti_rise_height / cg_height) * 100.0
        series.append(float(anti_rise) if math.isfinite(anti_rise) else None)

    return series


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
    bike = await bikes.find_one({"_id": oid, **_owner_filter(user_oid)})
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
        {"_id": oid, **_owner_filter(user_oid)},
        {"$set": {"geometry": geometry, **_stale_base_kinematics_cache_patch(), "updated_at": datetime.utcnow()}},
    )

    updated = await bikes.find_one({"_id": oid, **_owner_filter(user_oid)})
    hero_id = updated.get("hero_media_id")
    hero_url = await resolve_hero_url(hero_id)
    hero_thumb_url = await resolve_hero_variant_url(hero_id, "low")
    return bike_doc_to_out(updated, hero_url=hero_url, hero_thumb_url=hero_thumb_url, can_edit=True)


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
    variant_id: Optional[str] = None,
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
    is_admin = _is_admin_user(current_user)

    # Parse bike_id → ObjectId
    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bikes = bikes_col()
    doc = await bikes.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Bike not found")
    if not _can_view_bike(doc, user_oid, is_admin):
        raise HTTPException(status_code=403, detail="Not allowed to view this bike")
    is_owner = _is_bike_owner(doc, user_oid)

    variant_doc: Optional[dict] = None
    variant_overrides: dict = {}
    variant_fingerprint: Optional[str] = None
    if variant_id:
        variant_doc = await _load_variant_or_404(variant_id)
        if str(variant_doc.get("bike_id")) != str(oid):
            raise HTTPException(status_code=404, detail="Variant not found")
        variant_overrides = (
            dict(variant_doc.get("overrides") or {})
            if isinstance(variant_doc.get("overrides"), dict)
            else {}
        )

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
    variant_target_points = _apply_variant_point_overrides_to_points(points, variant_overrides)

    settings_doc = await bike_page_settings_col().find_one({"bike_id": oid, "user_id": user_oid})
    base_settings = _bike_shared_settings_from_doc(doc)
    base_settings.update(
        _normalize_page_settings(settings_doc.get("settings") if settings_doc else None)
    )
    settings = _merge_variant_overrides_into_settings(base_settings, variant_overrides)
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

    tx = None
    ty = None
    if H is not None:
        rectified_points: List[BikePoint] = []
        rectified_variant_target_points: List[BikePoint] = []
        scale = rectify.get("scale") if isinstance(rectify, dict) else None
        tx = rectify.get("tx") if isinstance(rectify, dict) else None
        ty = rectify.get("ty") if isinstance(rectify, dict) else None
        for source_points, target_points in (
            (points, rectified_points),
            (variant_target_points, rectified_variant_target_points),
        ):
            for p in source_points:
                mapped = apply_homography(H, p.x, p.y)
                if not mapped:
                    target_points.append(p)
                    continue
                x, y = mapped
                if scale and tx is not None and ty is not None:
                    x = x * scale + tx
                    y = y * scale + ty
                target_points.append(p.copy(update={"x": x, "y": y}))
        points = rectified_points
        variant_target_points = rectified_variant_target_points
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
    geom = _apply_variant_overrides_to_geometry(doc.get("geometry") or {}, variant_overrides)
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

    ground_reference, ground_reference_error = _derive_ground_reference_from_base_pose(
        points=points,
        scale_mm_per_px=scale_mm_per_px,
        base_settings=base_settings,
    )

    # ---- Apply overrides to authored bodies and prepare solver-only bodies ----
    authored_bodies = _apply_variant_overrides_to_bodies(bodies, variant_overrides, scale_mm_per_px)
    solver_bodies = _solver_rigid_bodies_only(authored_bodies)

    variant_point_constraints = _build_variant_point_constraints(
        variant_target_points,
        variant_overrides,
    )
    flip_chip_pose_constraints = _build_flip_chip_pose_constraints(variant_target_points, authored_bodies)
    rest_pose_constraints: dict[str, dict[str, float]] = {
        **flip_chip_pose_constraints,
        **variant_point_constraints,
    }

    pose_required = bool(rest_pose_constraints)
    if variant_doc is not None and _variant_requires_rest_pose(
        base_settings,
        settings,
        variant_overrides,
        points=points,
        bodies=solver_bodies,
    ):
        pose_required = True

    if pose_required:
        points, pose_debug = _compute_variant_rest_pose(
            points=points,
            bodies=solver_bodies,
            scale_mm_per_px=scale_mm_per_px,
            effective_settings=settings,
            ground_reference=ground_reference,
            iterations=iterations,
            override_point_constraints=rest_pose_constraints,
        )
    elif variant_doc is not None:
        pose_debug = {"mode": "rest_pose", "applied": False, "reason": "variant_no_pose_delta"}
    else:
        pose_debug = {"mode": "rest_pose", "applied": False, "reason": "base_bike"}

    if not pose_debug.get("applied") and ground_reference_error:
        pose_debug = {
            **pose_debug,
            "ground_reference_error": ground_reference_error,
        }

    shock_length0_px = _resolve_shock_length0_px(points, solver_bodies, geom, scale_mm_per_px)
    if shock_length0_px is not None:
        normalized_bodies_for_solver: List[RigidBody] = []
        for body in solver_bodies:
            if body.type == "shock":
                normalized_bodies_for_solver.append(body.copy(update={"length0": float(shock_length0_px)}))
            else:
                normalized_bodies_for_solver.append(body)
        solver_bodies = normalized_bodies_for_solver

    if variant_doc is not None:
        variant_fingerprint = _variant_fingerprint(
            doc,
            variant_doc,
            variant_overrides,
            settings,
            steps=steps,
            iterations=iterations,
        )

    # ---- Convert shock stroke from mm → px for the solver ----
    bodies_for_solver_px: List[RigidBody] = []
    for b in solver_bodies:
        if b.type == "shock" and b.stroke is not None:
            stroke_mm = float(b.stroke)
            stroke_px = stroke_mm / scale_mm_per_px
            b_px = b.copy(update={"stroke": stroke_px})
            bodies_for_solver_px.append(b_px)
        else:
            bodies_for_solver_px.append(b)

    # ---- Run solver (all geometry in px) ----
    pre_steps = 10
    if pose_debug.get("applied"):
        pre_steps = 0

    try:
        result = solve_bike_linkage(
            points=points,
            bodies=bodies_for_solver_px,  # NOTE: stroke now in px
            n_steps=steps,
            iterations=iterations,
            pre_steps=pre_steps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # --------------------------------------------------------
    # 1) Scale stroke / length / travel to mm, keep coords in px
    # --------------------------------------------------------
    solver_steps = sorted(result.steps, key=lambda s: s.step_index)
    full_steps = (
        sorted(result.full_steps, key=lambda s: s.step_index)
        if result.full_steps
        else []
    )
    scale_mm = scale_mm_per_px  # mm per px
    shared_step_objects = (
        bool(solver_steps)
        and len(solver_steps) == len(full_steps)
        and all(step is full_steps[idx] for idx, step in enumerate(solver_steps))
    )

    for s in solver_steps:
        # These came out of the solver in px
        s.shock_stroke = s.shock_stroke * scale_mm   # → mm
        s.shock_length = s.shock_length * scale_mm   # → mm
        if s.rear_travel is not None:
            s.rear_travel = s.rear_travel * scale_mm # → mm
    if not shared_step_objects:
        for s in full_steps:
            s.shock_stroke = s.shock_stroke * scale_mm
            s.shock_length = s.shock_length * scale_mm
            if s.rear_travel is not None:
                s.rear_travel = s.rear_travel * scale_mm
    if full_steps:
        result.full_steps = full_steps

    rectify_scale = None
    if isinstance(rectify, dict) and rectify.get("scale") is not None:
        try:
            rectify_scale = float(rectify.get("scale"))
        except Exception:
            rectify_scale = None

    def _map_solver_xy_to_image_xy(x: float, y: float) -> dict[str, Optional[float]]:
        if H_inv is not None and rectify_scale and tx is not None and ty is not None:
            rx = (float(x) - tx) / rectify_scale
            ry = (float(y) - ty) / rectify_scale
            mapped = apply_homography(H_inv, rx, ry)
            if mapped:
                return {"x": float(mapped[0]), "y": float(mapped[1])}
        return {"x": float(x), "y": float(y)}

    # --------------------------------------------------------
    # 2) Build coords per point_id: coords[i] = (x,y) at step i (STILL px)
    # --------------------------------------------------------
    coords_map: dict[str, list[dict]] = {}
    for step in solver_steps:
        for pid, (x, y) in step.points.items():
            mapped = _map_solver_xy_to_image_xy(float(x), float(y))
            coords_map.setdefault(pid, []).append(mapped)

    # Instant center coordinates (same coordinate space/shape as point coords lists).
    rear_body_point_ids = _pick_rear_body_point_ids(
        authored_bodies,
        result.rear_axle_point_id,
        points,
        preferred_body_id=str(settings.get("rear_brake_ic_body_id") or ""),
    )
    instant_center_solver = _compute_instant_center_series(solver_steps, rear_body_point_ids)
    instant_center_coords: list[dict[str, Optional[float]]] = []
    for ic in instant_center_solver:
        x_ic = ic.get("x")
        y_ic = ic.get("y")
        if x_ic is None or y_ic is None:
            instant_center_coords.append({"x": None, "y": None})
            continue
        instant_center_coords.append(_map_solver_xy_to_image_xy(float(x_ic), float(y_ic)))

    source_steps = full_steps or solver_steps
    instant_center_coords_full: list[dict[str, Optional[float]]] = []
    if source_steps:
        instant_center_solver_full = _compute_instant_center_series(source_steps, rear_body_point_ids)
        for ic in instant_center_solver_full:
            x_ic = ic.get("x")
            y_ic = ic.get("y")
            if x_ic is None or y_ic is None:
                instant_center_coords_full.append({"x": None, "y": None})
                continue
            instant_center_coords_full.append(_map_solver_xy_to_image_xy(float(x_ic), float(y_ic)))

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
    raw_linkage_convergence = (
        result.debug.get("convergence")
        if isinstance(result.debug, dict)
        else None
    )
    linkage_convergence = _build_scaled_linkage_convergence_payload(
        raw_linkage_convergence,
        scale_mm_per_px,
        trim_pre_steps=(
            int(result.debug.get("pre_steps", 0) or 0)
            if isinstance(result.debug, dict)
            else 0
        ),
    )
    linkage_convergence_summary = (
        linkage_convergence.get("summary", {})
        if isinstance(linkage_convergence, dict)
        else {}
    )
    linkage_convergence_worst_step = None
    if isinstance(linkage_convergence_summary, dict):
        worst_step_index = linkage_convergence_summary.get("worst_step_index")
        if isinstance(worst_step_index, int):
            linkage_steps = linkage_convergence.get("steps", [])
            if isinstance(linkage_steps, list) and 0 <= worst_step_index < len(linkage_steps):
                linkage_convergence_worst_step = linkage_steps[worst_step_index]
    result.debug = {
        "perspective_mode": perspective_mode,
        "settings_found": settings_found,
        "variant_id": str(variant_doc["_id"]) if variant_doc else None,
        "variant_is_base": bool(variant_doc.get("is_base")) if variant_doc else False,
        "variant_fingerprint": variant_fingerprint,
        "pose_debug": pose_debug,
        "rear_brake_ic_body_id_setting": str(settings.get("rear_brake_ic_body_id") or ""),
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
        "convergence": {
            "metric": linkage_convergence.get("metric", "max_point_displacement_mm"),
            "summary": linkage_convergence_summary,
            "worst_step": linkage_convergence_worst_step,
        },
        "rear_brake_caliper_point_id": _pick_rear_brake_caliper_point_id(
            bodies,
            result.rear_axle_point_id,
            points,
            preferred_body_id=str(settings.get("rear_brake_ic_body_id") or ""),
        ),
    }
    rear_axle_relative_mm: list[list[float]] = []
    leverage_ratio_series: list[Optional[float]] = []
    shock_stroke_mm_series: list[Optional[float]] = []
    rear_axle_relative_mm_full: list[list[float]] = []
    leverage_ratio_full: list[Optional[float]] = []
    shock_stroke_mm_full: list[Optional[float]] = []
    anti_squat_series: list[Optional[float]] = []
    anti_squat_full: list[Optional[float]] = []
    anti_rise_series: list[Optional[float]] = []
    anti_rise_full: list[Optional[float]] = []
    shock_force_series: list[Optional[float]] = []
    shock_spring_rate_series: list[Optional[float]] = []
    shock_spring_rate_cold_series: list[Optional[float]] = []
    shock_spring_rate_hot_series: list[Optional[float]] = []
    rear_wheel_force_series: list[Optional[float]] = []
    rear_wheel_force_n_series: list[Optional[float]] = []
    shock_force_full: list[Optional[float]] = []
    shock_spring_rate_full: list[Optional[float]] = []
    rear_wheel_force_full: list[Optional[float]] = []
    rear_wheel_force_n_full: list[Optional[float]] = []
    trim_index = 0
    front_axle_point_id = next((p.id for p in points if p.type == "front_axle"), None)
    bb_point_id = next((p.id for p in points if p.type in ("bb", "bottom_bracket")), None)
    idler_point_id = _find_idler_point_id(points)
    preferred_rear_brake_ic_body_id = str(settings.get("rear_brake_ic_body_id") or "")
    brake_caliper_point_id = _pick_rear_brake_caliper_point_id(
        bodies,
        result.rear_axle_point_id,
        points,
        preferred_body_id=preferred_rear_brake_ic_body_id,
    )
    shock_type, shock_model = _normalize_shock_geometry_config(geom)
    if result.rear_axle_point_id:
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

        # Full-series outputs (include negative travel) for debugging.
        if source_steps:
            origin_full = None
            for step in source_steps[trim_index:]:
                coords = step.points.get(result.rear_axle_point_id)
                if coords:
                    origin_full = (float(coords[0]), float(coords[1]))
                    break
            if origin_full:
                ox, oy = origin_full
                for step in source_steps:
                    coords = step.points.get(result.rear_axle_point_id)
                    if not coords:
                        rear_axle_relative_mm_full.append([0.0, 0.0])
                    else:
                        dx = (float(coords[0]) - ox) * scale_mm_per_px
                        dy = (float(coords[1]) - oy) * scale_mm_per_px
                        rear_axle_relative_mm_full.append([dx, dy])
                    shock_stroke_mm_full.append(step.shock_stroke)

        # Compute leverage ratio with one consistent numerical derivative across
        # the full pre-roll + travel series.
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
                leverage_ratio_full = [
                    (float(val) if np.isfinite(val) else None) for val in grad_full
                ]
                if trim_index > 0 and len(grad_full) > trim_index:
                    grad = grad_full[trim_index:]
                else:
                    grad = grad_full
                if len(grad) > len(shock_stroke_mm_series):
                    grad = grad[: len(shock_stroke_mm_series)]
                leverage_ratio_series = [
                    (float(val) if np.isfinite(val) else None) for val in grad
                ]
            except Exception:
                leverage_ratio_series = []

        reference_temp_c = shock_model.get("air_reference_temp_c")
        cold_temp_c = shock_model.get("air_cold_temp_c")
        hot_temp_c = shock_model.get("air_hot_temp_c")
        shock_force_series, shock_spring_rate_series = _compute_shock_force_and_rate_series(
            shock_stroke_mm_series,
            shock_type,
            shock_model,
            temp_c=reference_temp_c,
        )
        _, shock_spring_rate_cold_series = _compute_shock_force_and_rate_series(
            shock_stroke_mm_series,
            shock_type,
            shock_model,
            temp_c=cold_temp_c,
        )
        _, shock_spring_rate_hot_series = _compute_shock_force_and_rate_series(
            shock_stroke_mm_series,
            shock_type,
            shock_model,
            temp_c=hot_temp_c,
        )
        shock_force_full, shock_spring_rate_full = _compute_shock_force_and_rate_series(
            shock_stroke_mm_full,
            shock_type,
            shock_model,
            temp_c=reference_temp_c,
        )
        rear_wheel_force_n_series = _compute_rear_wheel_force_n_series(
            shock_force_series,
            leverage_ratio_series,
        )
        rear_wheel_force_n_full = _compute_rear_wheel_force_n_series(
            shock_force_full,
            leverage_ratio_full,
        )
        rear_travel_series = [
            (float(s.rear_travel) if s.rear_travel is not None else None)
            for s in source_steps[trim_index:]
        ]
        rear_travel_full = [
            (float(s.rear_travel) if s.rear_travel is not None else None)
            for s in source_steps
        ]
        rear_wheel_force_series = _compute_rear_wheel_force_series(
            rear_travel_series,
            rear_wheel_force_n_series,
        )
        rear_wheel_force_full = _compute_rear_wheel_force_series(
            rear_travel_full,
            rear_wheel_force_n_full,
        )

        anti_squat_series = _compute_anti_squat_series(
            steps=source_steps,
            instant_center_solver=instant_center_solver_full if source_steps else [],
            trim_index=trim_index,
            rear_axle_point_id=result.rear_axle_point_id,
            front_axle_point_id=front_axle_point_id,
            bb_point_id=bb_point_id,
            idler_point_id=idler_point_id,
            scale_mm_per_px=scale_mm_per_px,
            settings=settings,
        )
        anti_squat_full = _compute_anti_squat_series(
            steps=source_steps,
            instant_center_solver=instant_center_solver_full if source_steps else [],
            trim_index=0,
            rear_axle_point_id=result.rear_axle_point_id,
            front_axle_point_id=front_axle_point_id,
            bb_point_id=bb_point_id,
            idler_point_id=idler_point_id,
            scale_mm_per_px=scale_mm_per_px,
            settings=settings,
        )
        try:
            anti_rise_series = _compute_anti_rise_series(
                steps=source_steps,
                instant_center_solver=instant_center_solver_full if source_steps else [],
                trim_index=trim_index,
                rear_axle_point_id=result.rear_axle_point_id,
                front_axle_point_id=front_axle_point_id,
                bb_point_id=bb_point_id,
                scale_mm_per_px=scale_mm_per_px,
                settings=settings,
            )
            anti_rise_full = _compute_anti_rise_series(
                steps=source_steps,
                instant_center_solver=instant_center_solver_full if source_steps else [],
                trim_index=0,
                rear_axle_point_id=result.rear_axle_point_id,
                front_axle_point_id=front_axle_point_id,
                bb_point_id=bb_point_id,
                scale_mm_per_px=scale_mm_per_px,
                settings=settings,
            )
        except Exception as exc:
            logging.exception(
                "anti-rise compute failed for bike %s (rear_axle=%s caliper=%s): %s",
                bike_id,
                result.rear_axle_point_id,
                brake_caliper_point_id,
                exc,
            )
            anti_rise_series = []
            anti_rise_full = []

    for idx, s in enumerate(solver_steps):
        anti_squat_val = anti_squat_series[idx] if idx < len(anti_squat_series) else None
        anti_rise_val = anti_rise_series[idx] if idx < len(anti_rise_series) else None
        shock_rate_val = (
            shock_spring_rate_series[idx] if idx < len(shock_spring_rate_series) else None
        )
        rear_wheel_force_val = (
            rear_wheel_force_series[idx] if idx < len(rear_wheel_force_series) else None
        )
        rear_wheel_force_n_val = (
            rear_wheel_force_n_series[idx] if idx < len(rear_wheel_force_n_series) else None
        )
        s.anti_squat = anti_squat_val
        s.anti_rise = anti_rise_val
        s.shock_spring_rate = shock_rate_val
        s.rear_wheel_force = rear_wheel_force_val
        kin_steps.append(
            {
                "step_index": s.step_index,
                "shock_stroke": s.shock_stroke,      # mm
                "shock_length": s.shock_length,      # mm
                "rear_travel": s.rear_travel,        # mm
                "leverage_ratio": s.leverage_ratio,  # dimensionless
                "anti_squat": anti_squat_val,        # [%]
                "anti_rise": anti_rise_val,          # [%]
                "shock_spring_rate": shock_rate_val,  # [N/mm]
                "rear_wheel_force": rear_wheel_force_val,  # [N/mm]
                "rear_wheel_force_n": rear_wheel_force_n_val,  # [N]
            }
        )

    scaled_outputs = {
        "rear_axle_relative_mm": rear_axle_relative_mm,
        "leverage_ratio": leverage_ratio_series,
        "shock_stroke_mm": shock_stroke_mm_series,
        "anti_squat_series": anti_squat_series,
        "anti_rise_series": anti_rise_series,
        "shock_force_n": shock_force_series,
        "shock_spring_rate_n_per_mm": shock_spring_rate_series,
        "shock_spring_rate_cold_n_per_mm": shock_spring_rate_cold_series,
        "shock_spring_rate_hot_n_per_mm": shock_spring_rate_hot_series,
        "rear_wheel_force_n_per_mm": rear_wheel_force_series,
        "rear_wheel_force_n": rear_wheel_force_n_series,
        "shock_type": shock_type,
        "shock_model": shock_model,
        "instant_center_coords": instant_center_coords,
        "instant_center_rear_body_point_ids": rear_body_point_ids,
        "rear_axle_relative_mm_full": rear_axle_relative_mm_full,
        "leverage_ratio_full": leverage_ratio_full,
        "shock_stroke_mm_full": shock_stroke_mm_full,
        "anti_squat_full": anti_squat_full,
        "anti_rise_full": anti_rise_full,
        "shock_force_n_full": shock_force_full,
        "shock_spring_rate_n_per_mm_full": shock_spring_rate_full,
        "rear_wheel_force_n_per_mm_full": rear_wheel_force_full,
        "rear_wheel_force_n_full": rear_wheel_force_n_full,
        "instant_center_coords_full": instant_center_coords_full,
        "rear_brake_caliper_point_id": brake_caliper_point_id,
        "convergence_metric": linkage_convergence.get("metric", "max_point_displacement_mm"),
        "convergence_per_step": linkage_convergence.get("step_summaries", []),
    }

    max_travel_trim = _extract_max_travel_from_relative_series(rear_axle_relative_mm)
    max_travel_full = _extract_max_travel_from_relative_series(rear_axle_relative_mm_full)
    max_travel_candidates = [v for v in (max_travel_trim, max_travel_full) if v is not None]
    max_rear_travel_mm = (
        _round_to_nearest_10_mm(max(max_travel_candidates))
        if max_travel_candidates
        else None
    )
    rear_axle_debug = _rear_axle_debug_summary(
        solver_steps,
        result.rear_axle_point_id,
        scale_mm_per_px,
    )
    result.debug["travel_debug"] = {
        "front_wheel_size": str(settings.get("front_wheel_size", "29")),
        "rear_wheel_size": str(settings.get("rear_wheel_size", "29")),
        "scale_mm_per_px": scale_mm_per_px,
        "trim_index": trim_index,
        "max_travel_trim_mm": max_travel_trim,
        "max_travel_full_mm": max_travel_full,
        "max_rear_travel_mm": max_rear_travel_mm,
        "rear_axle": rear_axle_debug,
    }
    print(
        "[BikeVariantTravelDebug] "
        f"bike={bike_id} variant={str(variant_doc.get('_id')) if variant_doc else 'base'} "
        f"rear_wheel={str(settings.get('rear_wheel_size', '29'))} "
        f"front_wheel={str(settings.get('front_wheel_size', '29'))} "
        f"scale_mm_per_px={scale_mm_per_px:.6f} "
        f"trim_mm={max_travel_trim} full_mm={max_travel_full} "
        f"rear_axle={rear_axle_debug}"
    )

    serialized_solver_steps = [_serialize_solver_step(step) for step in solver_steps]

    persisted_debug = dict(result.debug or {})
    if isinstance(persisted_debug, dict):
        persisted_debug.pop("convergence", None)

    plot_cache_doc = {
        "rear_axle_point_id": result.rear_axle_point_id,
        "n_steps": len(solver_steps),
        "driver_stroke": solver_steps[-1].shock_stroke if solver_steps else None,
        "scaled_outputs": scaled_outputs,
        "debug": persisted_debug,
    }
    animation_cache_doc = {
        "rear_axle_point_id": result.rear_axle_point_id,
        "n_steps": len(solver_steps),
        "steps": serialized_solver_steps,
    }

    # --------------------------------------------------------
    # 4) Persist into Mongo
    # --------------------------------------------------------
    if is_owner and variant_doc is None:
        await bikes.update_one(
            {"_id": oid, **_owner_filter(user_oid)},
            {
                "$set": {
                    "points": new_points,
                    "kinematics_plot_cache": plot_cache_doc,
                    "kinematics_animation_cache": animation_cache_doc,
                    "kinematics_cache_fingerprint": variant_fingerprint,
                    "kinematics_cache_status": "ready",
                    "max_rear_travel_mm": max_rear_travel_mm,
                    "updated_at": datetime.utcnow(),
                }
            },
        )
    elif is_owner and variant_doc is not None:
        await bike_variants_col().update_one(
            {"_id": variant_doc["_id"], "bike_id": oid},
            {
                "$set": {
                    "kinematics_plot_cache": plot_cache_doc,
                    "kinematics_animation_cache": animation_cache_doc,
                    "kinematics_cache_fingerprint": variant_fingerprint,
                    "status": "ready",
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
    doc = await bikes.find_one({"_id": oid, **_owner_filter(user_oid)})
    if not doc:
        raise HTTPException(status_code=404, detail="Bike not found")

    hero_id = doc.get("hero_media_id")
    media_doc = None
    if hero_id:
        media_doc = await media_items_col().find_one({"_id": hero_id, "bike_id": oid})

    settings_doc = await bike_page_settings_col().find_one({"bike_id": oid, "user_id": user_oid})
    settings_payload = (
        _normalize_page_settings(settings_doc.get("settings")) if settings_doc else None
    )

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


@router.get("/{bike_id}/kinematics_animation_cache", response_model=KinematicsAnimationCache)
async def get_cached_bike_kinematics_animation_cache(
    bike_id: str,
    variant_id: Optional[str] = None,
    current_user=Depends(get_current_user),
):
    user_oid = _extract_user_oid(current_user)
    is_admin = _is_admin_user(current_user)

    try:
        oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    doc = await bikes_col().find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Bike not found")
    if not _can_view_bike(doc, user_oid, is_admin):
        raise HTTPException(status_code=403, detail="Not allowed to view this bike")

    cache_doc: Optional[dict] = None
    if variant_id:
        variant_doc = await _load_variant_or_404(variant_id)
        if str(variant_doc.get("bike_id")) != str(oid):
            raise HTTPException(status_code=404, detail="Variant not found")
        cache_doc = (
            variant_doc.get("kinematics_animation_cache")
            if isinstance(variant_doc.get("kinematics_animation_cache"), dict)
            else None
        )
    else:
        cache_doc = (
            doc.get("kinematics_animation_cache")
            if isinstance(doc.get("kinematics_animation_cache"), dict)
            else None
        )

    if not cache_doc:
        return KinematicsAnimationCache(
            rear_axle_point_id=None,
            n_steps=0,
            steps=[],
        )

    return KinematicsAnimationCache(
        rear_axle_point_id=cache_doc.get("rear_axle_point_id"),
        n_steps=cache_doc.get("n_steps") or (len(cache_doc.get("steps") or [])),
        steps=cache_doc.get("steps") or [],
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
    variants = bike_variants_col()

    doc = await bikes.find_one({"_id": oid, **_owner_filter(user_oid)})
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

    await bikes.delete_one({"_id": oid, **_owner_filter(user_oid)})
    await variants.delete_many({"bike_id": oid})
    return Response(status_code=status.HTTP_204_NO_CONTENT)
