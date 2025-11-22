# app/routers/bikes.py
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from bson import ObjectId

from app.db import bikes_col
from app.storage import generate_signed_url
from .auth import get_current_user

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


def bike_doc_to_out(doc) -> BikeOut:
    return BikeOut(
        id=str(doc["_id"]),
        name=doc["name"],
        brand=doc["brand"],
        model_year=doc.get("model_year"),
        user_id=str(doc["user_id"]),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        hero_media_id=(str(doc["hero_media_id"]) if doc.get("hero_media_id") else None),
        hero_url=doc.get("hero_url"),
    )


@router.post("", response_model=BikeOut, status_code=status.HTTP_201_CREATED)
async def create_bike(
    bike_in: BikeCreate,
    current_user=Depends(get_current_user),
):
    now = datetime.utcnow()

    user_id = getattr(current_user, "id", None) or getattr(current_user, "_id", None)
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)

    doc = {
        "user_id": user_id,
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
    user_id = getattr(current_user, "id", None) or getattr(current_user, "_id", None)
    if isinstance(user_id, str):
        user_oid = ObjectId(user_id)
    else:
        user_oid = user_id

    bikes = bikes_col()
    media_items = media_items_col()

    cursor = bikes.find({"user_id": user_oid})
    docs = await cursor.to_list(length=1000)

    out: list[BikeOut] = []
    for d in docs:
        hero_url: Optional[str] = None
        hero_id = d.get("hero_media_id")
        if hero_id:
            media_doc = await media_items.find_one({"_id": hero_id})
            if media_doc:
                key = media_doc["storage_key"]
                hero_url = generate_signed_url(key, expires_in=3600)
        out.append(bike_doc_to_out(d, hero_url=hero_url))

    return out