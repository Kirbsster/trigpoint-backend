# app/storage.py
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import UploadFile
from google.cloud import storage
import google.auth
from google.auth import impersonated_credentials

# Bucket name for media
GCS_BUCKET_NAME = os.getenv("GCS_MEDIA_BUCKET", "trigpoint-media-testing")

_client: Optional[storage.Client] = None
_bucket: Optional[storage.Bucket] = None


def get_bucket() -> storage.Bucket:
    """Get (and cache) the GCS bucket."""
    global _client, _bucket
    if _bucket is None:
        _client = storage.Client()
        _bucket = _client.bucket(GCS_BUCKET_NAME)
    return _bucket


async def upload_bike_image(user_id: str, bike_id: str, file: UploadFile) -> str:
    """Upload an image for a bike and return the GCS storage key."""
    content = await file.read()

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


def download_media(bucket_name: str, key: str) -> bytes:
    """Download a media object from GCS and return its bytes."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)
    return blob.download_as_bytes()


def generate_signed_url(key: str, expires_in: int = 3600) -> str:
    """Generate a v4 signed URL for a GCS object.

    Works on Cloud Run without a local key by using IAM SignBlob via
    impersonated credentials.
    """
    bucket = get_bucket()
    blob = bucket.blob(key)

    # Get default application credentials (Cloud Run service account).
    credentials, project_id = google.auth.default()

    # Wrap them in impersonated credentials that can sign.
    signing_credentials = impersonated_credentials.Credentials(
        source_credentials=credentials,
        target_principal=credentials.service_account_email,
        target_scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
        lifetime=300,
    )

    expiration = timedelta(seconds=expires_in)

    return blob.generate_signed_url(
        version="v4",
        expiration=expiration,
        method="GET",
        credentials=signing_credentials,
    )