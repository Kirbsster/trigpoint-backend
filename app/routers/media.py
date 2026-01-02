# app/routers/media.py
from datetime import datetime
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    UploadFile,
    HTTPException,
    status,
    Response,
)
from pydantic import BaseModel
from bson import ObjectId
import os

from app.db import bikes_col, media_items_col
from app.routers.auth import get_current_user
from app.storage import upload_bike_image, download_media, delete_media, GCS_BUCKET_NAME


router = APIRouter(prefix="/bikes", tags=["media"])
DEFAULT_BUCKET = os.getenv("GCS_MEDIA_BUCKET", GCS_BUCKET_NAME)


class MediaOut(BaseModel):
    id: str
    bike_id: str
    user_id: str
    bucket: str
    storage_key: str
    content_type: Optional[str] = None
    size_bytes: int
    role: str
    created_at: datetime


def media_doc_to_out(doc) -> MediaOut:
    return MediaOut(
        id=str(doc["_id"]),
        bike_id=str(doc["bike_id"]),
        user_id=str(doc["user_id"]),
        bucket=doc["bucket"],
        storage_key=doc["storage_key"],
        content_type=doc.get("content_type"),
        size_bytes=doc["size_bytes"],
        role=doc.get("role", "hero"),
        created_at=doc["created_at"],
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
        print("DEBUG media._extract_user_oid: current_user has no id:", repr(current_user))
        raise HTTPException(status_code=500, detail="Could not determine current user id")

    if isinstance(raw_id, str):
        try:
            return ObjectId(raw_id)
        except Exception:
            print("DEBUG media._extract_user_oid: failed to convert raw_id to ObjectId:", raw_id)
            raise HTTPException(status_code=500, detail="Invalid current user id format")

    # raw_id is already an ObjectId (or at least something usable)
    return raw_id


async def _delete_media_doc(media_doc, media_items) -> None:
    bucket_name = media_doc.get("bucket", DEFAULT_BUCKET)
    key = media_doc.get("storage_key")
    if key:
        try:
            delete_media(bucket_name, key)
        except Exception as exc:
            print("WARN media.delete_media failed:", exc)
    await media_items.delete_one({"_id": media_doc["_id"]})


@router.post("/{bike_id}/media/hero", response_model=MediaOut, status_code=status.HTTP_201_CREATED)
async def upload_hero_image(
    bike_id: str,
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    """Upload a hero image for a bike and link it in Mongo + GCS."""
    bikes = bikes_col()
    media_items = media_items_col()

    # Validate bike_id
    try:
        bike_oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bike = await bikes.find_one({"_id": bike_oid})
    if not bike:
        raise HTTPException(status_code=404, detail="Bike not found")

    # Extract current user id robustly
    user_oid = _extract_user_oid(current_user)

    # Ownership check
    if bike.get("user_id") != user_oid:
        raise HTTPException(status_code=403, detail="Not your bike")

    old_hero_id = bike.get("hero_media_id")
    old_media_doc = None
    if old_hero_id:
        old_media_doc = await media_items.find_one({"_id": old_hero_id, "bike_id": bike_oid})

    # Upload to GCS via storage
    # NOTE: this assumes upload_bike_image returns (storage_key, size)
    storage_key, size = await upload_bike_image(str(user_oid), bike_id, file)

    now = datetime.utcnow()
    media_doc = {
        "user_id": user_oid,
        "bike_id": bike_oid,
        "bucket": os.getenv("GCS_MEDIA_BUCKET", GCS_BUCKET_NAME),
        "storage_key": storage_key,
        "content_type": file.content_type,
        "size_bytes": size,
        "role": "hero",
        "created_at": now,
    }
    result = await media_items.insert_one(media_doc)
    media_doc["_id"] = result.inserted_id

    # Link to bike as hero
    await bikes.update_one(
        {"_id": bike_oid},
        {"$set": {"hero_media_id": media_doc["_id"]}},
    )

    if old_media_doc and old_media_doc.get("_id") != media_doc["_id"]:
        await _delete_media_doc(old_media_doc, media_items)

    return media_doc_to_out(media_doc)


@router.delete("/{bike_id}/media/hero", status_code=status.HTTP_204_NO_CONTENT)
async def delete_hero_image(
    bike_id: str,
    current_user=Depends(get_current_user),
):
    bikes = bikes_col()
    media_items = media_items_col()

    try:
        bike_oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bike = await bikes.find_one({"_id": bike_oid})
    if not bike:
        raise HTTPException(status_code=404, detail="Bike not found")

    user_oid = _extract_user_oid(current_user)
    if bike.get("user_id") != user_oid:
        raise HTTPException(status_code=403, detail="Not your bike")

    hero_id = bike.get("hero_media_id")
    if not hero_id:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    media_doc = await media_items.find_one({"_id": hero_id, "bike_id": bike_oid})
    if media_doc:
        await _delete_media_doc(media_doc, media_items)

    await bikes.update_one(
        {"_id": bike_oid},
        {"$unset": {"hero_media_id": ""}},
    )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


# second router for serving media bytes via Cloud Run
media_router = APIRouter(prefix="/media", tags=["media"])


@media_router.get("/{media_id}")
async def get_media(
    media_id: str,
    current_user=Depends(get_current_user),
):
    """Stream a media file from GCS via Cloud Run, enforcing ownership."""
    media_items = media_items_col()

    try:
        media_oid = ObjectId(media_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid media_id")

    doc = await media_items.find_one({"_id": media_oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Media not found")

    # Extract current user id robustly
    user_oid = _extract_user_oid(current_user)

    # Ownership check
    if doc.get("user_id") != user_oid:
        raise HTTPException(status_code=403, detail="Not your media")

    bucket_name = doc.get("bucket", DEFAULT_BUCKET)
    key = doc["storage_key"]
    content_type = doc.get("content_type") or "application/octet-stream"

    data = download_media(bucket_name, key)
    return Response(content=data, media_type=content_type)
