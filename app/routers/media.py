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
from app.storage import (
    upload_bytes_to_key,
    download_media,
    delete_media_prefix_except,
    GCS_BUCKET_NAME,
)
from app.image_processing import (
    detect_single_bike_bbox,
    crop_and_resize_webp,
    open_image_from_bytes,
)


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
    warning: Optional[str] = None


def media_doc_to_out(doc, warning: Optional[str] = None) -> MediaOut:
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
        warning=warning,
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

    original_content = await file.read()
    if not original_content:
        raise HTTPException(status_code=400, detail="Empty upload")

    filename = file.filename or "image"
    parts = filename.rsplit(".", 1)
    ext = parts[1].lower() if len(parts) == 2 else "bin"

    bucket_name = os.getenv("GCS_MEDIA_BUCKET", GCS_BUCKET_NAME)
    base_prefix = f"users/{user_oid}/bikes/{bike_id}/images"
    hero_prefix = f"{base_prefix}/hero_"

    warning: Optional[str] = None
    processed = {}

    try:
        image = open_image_from_bytes(original_content)
        bbox, warning = detect_single_bike_bbox(image)
        if bbox:
            processed["low"] = crop_and_resize_webp(image, bbox, long_edge_px=150)
            processed["med"] = crop_and_resize_webp(image, bbox, long_edge_px=450)
            processed["high"] = crop_and_resize_webp(image, bbox, long_edge_px=None)
    except Exception as exc:
        warning = f"Image processing failed; saved original image. ({exc})"

    original_key = f"{base_prefix}/hero_original.{ext}"
    upload_bytes_to_key(
        bucket_name=bucket_name,
        key=original_key,
        content=original_content,
        content_type=file.content_type or "application/octet-stream",
    )

    created_at = datetime.utcnow()
    media_docs = []
    media_docs.append({
        "user_id": user_oid,
        "bike_id": bike_oid,
        "bucket": bucket_name,
        "storage_key": original_key,
        "content_type": file.content_type,
        "size_bytes": len(original_content),
        "role": "hero_original",
        "created_at": created_at,
    })

    if processed and not warning:
        key_low = f"{base_prefix}/hero_low.webp"
        key_med = f"{base_prefix}/hero_med.webp"
        key_high = f"{base_prefix}/hero_high.webp"

        upload_bytes_to_key(bucket_name, key_low, processed["low"], "image/webp")
        upload_bytes_to_key(bucket_name, key_med, processed["med"], "image/webp")
        upload_bytes_to_key(bucket_name, key_high, processed["high"], "image/webp")

        media_docs.extend([
            {
                "user_id": user_oid,
                "bike_id": bike_oid,
                "bucket": bucket_name,
                "storage_key": key_low,
                "content_type": "image/webp",
                "size_bytes": len(processed["low"]),
                "role": "hero_low",
                "created_at": created_at,
            },
            {
                "user_id": user_oid,
                "bike_id": bike_oid,
                "bucket": bucket_name,
                "storage_key": key_med,
                "content_type": "image/webp",
                "size_bytes": len(processed["med"]),
                "role": "hero_med",
                "created_at": created_at,
            },
            {
                "user_id": user_oid,
                "bike_id": bike_oid,
                "bucket": bucket_name,
                "storage_key": key_high,
                "content_type": "image/webp",
                "size_bytes": len(processed["high"]),
                "role": "hero_high",
                "created_at": created_at,
            },
        ])

    result = await media_items.insert_many(media_docs)
    inserted_ids = result.inserted_ids
    for doc, inserted_id in zip(media_docs, inserted_ids):
        doc["_id"] = inserted_id

    hero_doc = None
    for doc in media_docs:
        if doc.get("role") == "hero_high":
            hero_doc = doc
            break
    if hero_doc is None:
        hero_doc = media_docs[0]

    await bikes.update_one(
        {"_id": bike_oid},
        {"$set": {"hero_media_id": hero_doc["_id"]}},
    )

    keep_keys = {doc["storage_key"] for doc in media_docs}
    delete_media_prefix_except(bucket_name, hero_prefix, keep_keys)

    await media_items.delete_many({
        "bike_id": bike_oid,
        "_id": {"$nin": inserted_ids},
        "role": {"$regex": "^hero_"},
    })

    return media_doc_to_out(hero_doc, warning=warning)


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

    base_prefix = f"users/{user_oid}/bikes/{bike_id}/images"
    hero_prefix = f"{base_prefix}/hero_"

    hero_docs = []
    async for doc in media_items.find({"bike_id": bike_oid, "role": {"$regex": "^hero_"}}):
        hero_docs.append(doc)

    bucket_names = {os.getenv("GCS_MEDIA_BUCKET", GCS_BUCKET_NAME)}
    for doc in hero_docs:
        bucket_names.add(doc.get("bucket", os.getenv("GCS_MEDIA_BUCKET", GCS_BUCKET_NAME)))

    for bucket_name in bucket_names:
        delete_media_prefix_except(bucket_name, hero_prefix, keep_keys=set())

    await media_items.delete_many({
        "bike_id": bike_oid,
        "role": {"$regex": "^hero_"},
    })

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
