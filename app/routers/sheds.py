# app/routers/sheds.py
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Response
from pydantic import BaseModel, Field
from bson import ObjectId

from app.db import sheds_col, bikes_col
from app.routers.auth import get_current_user
from app.routers.bikes import BikeOut, bike_doc_to_out  # reuse existing models


router = APIRouter(prefix="/sheds", tags=["sheds"])


# ---------- Pydantic models ----------

class ShedCreate(BaseModel):
    name: str = Field(..., max_length=200)
    description: Optional[str] = None
    visibility: str = Field("private", pattern="^(private|unlisted|public)$")


class ShedOut(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    visibility: str
    owner_type: str
    owner_id: Optional[str] = None
    is_featured: bool
    bike_ids: List[str]
    bike_count: int          # <-- added for frontend
    created_at: datetime
    updated_at: datetime


def shed_doc_to_out(doc) -> ShedOut:
    bike_ids = doc.get("bike_ids", []) or []
    return ShedOut(
        id=str(doc["_id"]),
        name=doc["name"],
        description=doc.get("description"),
        visibility=doc.get("visibility", "private"),
        owner_type=doc.get("owner_type", "user"),
        owner_id=str(doc["owner_id"]) if doc.get("owner_id") is not None else None,
        is_featured=bool(doc.get("is_featured", False)),
        bike_ids=[str(bid) for bid in bike_ids],
        bike_count=len(bike_ids),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


def _extract_user_oid(current_user) -> ObjectId:
    """Same pattern you used in bikes.py."""
    raw_id = None
    if isinstance(current_user, dict):
        raw_id = current_user.get("id") or current_user.get("_id")
    else:
        raw_id = getattr(current_user, "id", None) or getattr(current_user, "_id", None)

    if raw_id is None:
        raise HTTPException(status_code=500, detail="Could not determine current user id")

    if isinstance(raw_id, str):
        try:
            return ObjectId(raw_id)
        except Exception:
            raise HTTPException(status_code=500, detail="Invalid current user id format")

    return raw_id


# ---------- Endpoints ----------

@router.post("", response_model=ShedOut, status_code=status.HTTP_201_CREATED)
async def create_shed(
    shed_in: ShedCreate,
    current_user=Depends(get_current_user),
):
    """Create a new user-owned shed."""
    now = datetime.utcnow()
    owner_oid = _extract_user_oid(current_user)

    doc = {
        "name": shed_in.name,
        "description": shed_in.description,
        "visibility": shed_in.visibility,
        "owner_type": "user",
        "owner_id": owner_oid,
        "is_featured": False,
        "bike_ids": [],
        "created_at": now,
        "updated_at": now,
    }

    sheds = sheds_col()
    result = await sheds.insert_one(doc)
    doc["_id"] = result.inserted_id
    return shed_doc_to_out(doc)


@router.get("", response_model=List[ShedOut])
async def list_my_sheds(current_user=Depends(get_current_user)):
    """List all sheds owned by the current user."""
    owner_oid = _extract_user_oid(current_user)
    sheds = sheds_col()
    cursor = sheds.find({"owner_id": owner_oid}).sort("created_at", 1)
    docs = await cursor.to_list(length=1000)
    return [shed_doc_to_out(d) for d in docs]


@router.get("/{shed_id}", response_model=ShedOut)
async def get_shed(
    shed_id: str,
    current_user=Depends(get_current_user),
):
    """Get a single shed (only if you own it for now)."""
    sheds = sheds_col()
    try:
        shed_oid = ObjectId(shed_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid shed_id")

    doc = await sheds.find_one({"_id": shed_oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Shed not found")

    owner_oid = _extract_user_oid(current_user)
    if doc.get("owner_id") != owner_oid:
        raise HTTPException(status_code=403, detail="Not your shed")

    return shed_doc_to_out(doc)


@router.post("/{shed_id}/bikes/{bike_id}", response_model=ShedOut)
async def add_bike_to_shed(
    shed_id: str,
    bike_id: str,
    current_user=Depends(get_current_user),
):
    """Add a bike to a shed (no duplicates)."""
    sheds = sheds_col()
    bikes = bikes_col()

    # Parse IDs
    try:
        shed_oid = ObjectId(shed_id)
        bike_oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid shed_id or bike_id")

    # Load shed and check ownership
    shed = await sheds.find_one({"_id": shed_oid})
    if not shed:
        raise HTTPException(status_code=404, detail="Shed not found")

    owner_oid = _extract_user_oid(current_user)
    if shed.get("owner_id") != owner_oid:
        raise HTTPException(status_code=403, detail="Not your shed")

    # Ensure bike exists and belongs to the same user
    bike = await bikes.find_one({"_id": bike_oid})
    if not bike:
        raise HTTPException(status_code=404, detail="Bike not found")
    if bike.get("user_id") != owner_oid:
        raise HTTPException(status_code=403, detail="You do not own this bike")

    # Add bike if not already present
    await sheds.update_one(
        {"_id": shed_oid},
        {
            "$addToSet": {"bike_ids": bike_oid},
            "$set": {"updated_at": datetime.utcnow()},
        },
    )
    updated = await sheds.find_one({"_id": shed_oid})
    return shed_doc_to_out(updated)


@router.delete("/{shed_id}/bikes/{bike_id}", response_model=ShedOut)
async def remove_bike_from_shed(
    shed_id: str,
    bike_id: str,
    current_user=Depends(get_current_user),
):
    """Remove a bike from a shed."""
    sheds = sheds_col()
    try:
        shed_oid = ObjectId(shed_id)
        bike_oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid shed_id or bike_id")

    shed = await sheds.find_one({"_id": shed_oid})
    if not shed:
        raise HTTPException(status_code=404, detail="Shed not found")

    owner_oid = _extract_user_oid(current_user)
    if shed.get("owner_id") != owner_oid:
        raise HTTPException(status_code=403, detail="Not your shed")

    await sheds.update_one(
        {"_id": shed_oid},
        {
            "$pull": {"bike_ids": bike_oid},
            "$set": {"updated_at": datetime.utcnow()},
        },
    )
    updated = await sheds.find_one({"_id": shed_oid})
    return shed_doc_to_out(updated)


@router.get("/{shed_id}/bikes", response_model=List[BikeOut])
async def list_bikes_in_shed(
    shed_id: str,
    current_user=Depends(get_current_user),
):
    """Return the bikes belonging to a shed, as BikeOut models."""
    sheds = sheds_col()
    bikes = bikes_col()

    try:
        shed_oid = ObjectId(shed_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid shed_id")

    shed = await sheds.find_one({"_id": shed_oid})
    if not shed:
        raise HTTPException(status_code=404, detail="Shed not found")

    owner_oid = _extract_user_oid(current_user)
    if shed.get("owner_id") != owner_oid:
        raise HTTPException(status_code=403, detail="Not your shed")

    bike_ids = shed.get("bike_ids", [])
    if not bike_ids:
        return []

    cursor = bikes.find({"_id": {"$in": bike_ids}})
    docs = await cursor.to_list(length=1000)

    # Reuse your existing BikeOut converter
    return [bike_doc_to_out(d) for d in docs]


# ---------- NEW: delete shed ----------

@router.delete("/{shed_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_shed(
    shed_id: str,
    current_user=Depends(get_current_user),
):
    """Delete a shed if it belongs to the current user."""
    sheds = sheds_col()

    try:
        shed_oid = ObjectId(shed_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid shed_id")

    doc = await sheds.find_one({"_id": shed_oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Shed not found")

    owner_oid = _extract_user_oid(current_user)
    if doc.get("owner_id") != owner_oid:
        raise HTTPException(status_code=403, detail="Not your shed")

    await sheds.delete_one({"_id": shed_oid})
    return Response(status_code=status.HTTP_204_NO_CONTENT)