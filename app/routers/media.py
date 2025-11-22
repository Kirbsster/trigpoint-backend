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
from app.storage import upload_bike_image, download_media, GCS_BUCKET_NAME

router = APIRouter(prefix="/bikes", tags=["media"])


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


@router.post("/{bike_id}/media/hero", response_model=MediaOut, status_code=status.HTTP_201_CREATED)
async def upload_hero_image(
    bike_id: str,
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    """Upload a hero image for a bike and link it in Mongo + GCS."""
    bikes = bikes_col()
    media_items = media_items_col()

    # Validate bike_id and ownership
    try:
        bike_oid = ObjectId(bike_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bike_id")

    bike = await bikes.find_one({"_id": bike_oid})
    if not bike:
        raise HTTPException(status_code=404, detail="Bike not found")

    user_id = getattr(current_user, "id", None) or getattr(current_user, "_id", None)
    if isinstance(user_id, str):
        user_oid = ObjectId(user_id)
    else:
        user_oid = user_id

    if bike.get("user_id") != user_oid:
        raise HTTPException(status_code=403, detail="Not your bike")

    # Upload to GCS via storage_gcs
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

    return media_doc_to_out(media_doc)


# second router for serving media bytes via Cloud Run
media_router = APIRouter(prefix="/media", tags=["media"])

DEFAULT_BUCKET = os.getenv("GCS_MEDIA_BUCKET", GCS_BUCKET_NAME)


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

    # Check ownership
    user_id = getattr(current_user, "id", None) or getattr(current_user, "_id", None)
    if isinstance(user_id, str):
        user_oid = ObjectId(user_id)
    else:
        user_oid = user_id

    if doc.get("user_id") != user_oid:
        raise HTTPException(status_code=403, detail="Not your media")

    bucket_name = doc.get("bucket", DEFAULT_BUCKET)
    key = doc["storage_key"]
    content_type = doc.get("content_type") or "application/octet-stream"

    data = download_media(bucket_name, key)

    return Response(content=data, media_type=content_type)