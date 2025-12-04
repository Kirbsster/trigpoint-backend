# app/routers/bikes.py
from datetime import datetime
from typing import Optional, List
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from bson import ObjectId
from app.schemas import (
    BikePoint,
    BikePointsUpdate,
    RigidBody,
    BikeBodiesOut,
    BikeBodiesUpdate,
)

from app.db import bikes_col#, media_items_col
# from app.storage import generate_signed_url
from .auth import get_current_user
from app.utils_media import resolve_hero_url

router = APIRouter(prefix="/bikes", tags=["bikes"])


class BikeCreate(BaseModel):
    name: str
    brand: str
    model_year: Optional[int] = None


class BikeOut(BaseModel):
    id: str
    name: str
    brand: str
    model_year: Optional[int] = None
    user_id: str
    created_at: datetime
    updated_at: datetime
    hero_media_id: Optional[str] = None
    hero_url: Optional[str] = None
    points: Optional[List[BikePoint]] = None
    bodies: Optional[List[RigidBody]] = None


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

@router.put("/{bike_id}/points", response_model=BikeOut)
async def update_bike_points(
    bike_id: str,
    payload: BikePointsUpdate,
    current_user=Depends(get_current_user),
):
    """Update the annotated geometry points for a bike."""
    user_oid = _extract_user_oid(current_user)
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
