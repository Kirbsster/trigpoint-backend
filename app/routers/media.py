# app/routers/media.py
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from bson import ObjectId

from app.db import bikes_col, media_items_col
from app.routers.auth import get_current_user
from app.storage import upload_bike_image

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

    # Upload to GCS
    storage_key = await upload_bike_image(str(user_oid), bike_id, file)

    # Build media document
    content = await file.read()  # NOTE: file was already read; better to capture len before
    size = len(content)

    now = datetime.utcnow()
    media_doc = {
        "user_id": user_oid,
        "bike_id": bike_oid,
        "bucket": os.getenv("GCS_MEDIA_BUCKET", "trigpoint-media-testing"),
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