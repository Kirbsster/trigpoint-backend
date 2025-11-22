# app/storage_gcs.py
import os
import uuid
from typing import Tuple

from google.cloud import storage
from fastapi import UploadFile

GCS_BUCKET_NAME = os.getenv("GCS_MEDIA_BUCKET", "trigpoint-media-testing")

_client: storage.Client | None = None
_bucket: storage.Bucket | None = None


def get_bucket() -> storage.Bucket:
    global _client, _bucket
    if _bucket is None:
        _client = storage.Client()
        _bucket = _client.bucket(GCS_BUCKET_NAME)
    return _bucket


async def upload_bike_image(user_id: str, bike_id: str, file: UploadFile) -> str:
    """Upload an image for a bike and return the GCS storage key."""
    content = await file.read()

    # derive extension
    filename = file.filename or "image"
    parts = filename.rsplit(".", 1)
    ext = parts[1].lower() if len(parts) == 2 else "bin"

    key = f"users/{user_id}/bikes/{bike_id}/images/{uuid.uuid4()}.{ext}"

    bucket = get_bucket()
    blob = bucket.blob(key)
    blob.upload_from_string(
        content,
        content_type=file.content_type or "application/octet-stream",
    )

    return key