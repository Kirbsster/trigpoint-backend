# # app/storage_gcs.py
# import os
# import uuid
# from typing import Tuple
# from datetime import datetime, timedelta

# import google.auth
# from google.auth import impersonated_credentials

# from google.cloud import storage
# from fastapi import UploadFile

# GCS_BUCKET_NAME = os.getenv("GCS_MEDIA_BUCKET", "trigpoint-media-testing")

# _client: storage.Client | None = None
# _bucket: storage.Bucket | None = None


# def get_bucket(bucket_name: str | None = None) -> storage.Bucket:
#     """Return the default bucket or a named bucket."""
#     global _client, _bucket
#     if bucket_name is None or bucket_name == GCS_BUCKET_NAME:
#         if _bucket is None:
#             _client = storage.Client()
#             _bucket = _client.bucket(GCS_BUCKET_NAME)
#         return _bucket
#     # Allow override if you ever store in other buckets
#     client = storage.Client()
#     return client.bucket(bucket_name)


# async def upload_bike_image(user_id: str, bike_id: str, file: UploadFile) -> Tuple[str, int]:
#     """Upload an image for a bike and return (GCS storage key, size_bytes)."""
#     content = await file.read()
#     size = len(content)

#     # derive extension
#     filename = file.filename or "image"
#     parts = filename.rsplit(".", 1)
#     ext = parts[1].lower() if len(parts) == 2 else "bin"

#     key = f"users/{user_id}/bikes/{bike_id}/images/{uuid.uuid4()}.{ext}"

#     bucket = get_bucket()
#     blob = bucket.blob(key)
#     blob.upload_from_string(
#         content,
#         content_type=file.content_type or "application/octet-stream",
#     )
#     return key, size


# def download_media(bucket_name: str, key: str) -> bytes:
#     """Download raw bytes for a media object."""
#     bucket = get_bucket(bucket_name)
#     blob = bucket.blob(key)
#     return blob.download_as_bytes()


# # def generate_signed_url(key: str, expires_in: int = 3600) -> str:
# #     """Generate a signed URL for a GCS object in the default bucket."""
# #     bucket = get_bucket()
# #     blob = bucket.blob(key)
# #     expiration = datetime.utcnow() + timedelta(seconds=expires_in)
# #     return blob.generate_signed_url(expiration=expiration, method="GET")


# def generate_signed_url(key: str, expires_in: int = 3600) -> str:
#     """Generate a v4 signed URL for a GCS object in the default bucket.

#     Works on Cloud Run without a local key, by using IAM SignBlob via
#     impersonated credentials.
#     """
#     bucket = get_bucket()
#     blob = bucket.blob(key)

#     # Get the default credentials and service account email (from Cloud Run)
#     credentials, project_id = google.auth.default()

#     # Create impersonated credentials that can SIGN on behalf of this service account.
#     # target_principal is the same service account that Cloud Run is using.
#     signing_credentials = impersonated_credentials.Credentials(
#         source_credentials=credentials,
#         target_principal=credentials.service_account_email,
#         target_scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
#         lifetime=300,  # seconds; short-lived is fine
#     )

#     expiration = timedelta(seconds=expires_in)

#     # Ask GCS client library to generate a v4 signed URL using those signing creds.
#     return blob.generate_signed_url(
#         version="v4",
#         expiration=expiration,
#         method="GET",
#         credentials=signing_credentials,
#     )






# app/storage.py (or storage_gcs.py)
import os
import uuid
from datetime import timedelta
from typing import Optional

from fastapi import UploadFile
from google.cloud import storage
import google.auth
from google.auth import impersonated_credentials

GCS_BUCKET_NAME = os.getenv("GCS_MEDIA_BUCKET", "trigpoint-media-testing")

_client: Optional[storage.Client] = None
_bucket: Optional[storage.Bucket] = None


def get_bucket(bucket_name: str | None = None) -> storage.Bucket:
    global _client, _bucket
    if bucket_name is None or bucket_name == GCS_BUCKET_NAME:
        if _bucket is None:
            _client = storage.Client()
            _bucket = _client.bucket(GCS_BUCKET_NAME)
        return _bucket

    client = storage.Client()
    return client.bucket(bucket_name)


async def upload_bike_image(user_id: str, bike_id: str, file: UploadFile) -> tuple[str, int]:
    """Upload an image for a bike and return (storage_key, size_bytes)."""
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
    return key, len(content)


def upload_bytes_to_key(
    bucket_name: str,
    key: str,
    content: bytes,
    content_type: str,
) -> int:
    """Upload raw bytes to a specific key and return size."""
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(key)
    blob.upload_from_string(content, content_type=content_type)
    return len(content)


def download_media(bucket_name: str, key: str) -> bytes:
    """Download raw bytes for a media object."""
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(key)
    return blob.download_as_bytes()


def delete_media(bucket_name: str, key: str) -> None:
    """Delete a media object from storage."""
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(key)
    blob.delete()


def delete_media_prefix(bucket_name: str, prefix: str) -> None:
    """Delete all media objects under a key prefix."""
    bucket = get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        blob.delete()


def delete_media_prefix_except(bucket_name: str, prefix: str, keep_keys: set[str]) -> None:
    """Delete all media objects under prefix except the keep_keys."""
    bucket = get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if blob.name in keep_keys:
            continue
        blob.delete()


def generate_signed_url(key: str, expires_in: int = 3600) -> str:
    """Generate a v4 signed URL for a GCS object using IAM SignBlob.

    Works on Cloud Run without a local private key, by using the Cloud Run
    service account plus roles/iam.serviceAccountTokenCreator.
    """
    bucket = get_bucket()
    blob = bucket.blob(key)

    # Get the default credentials (Cloud Run service account)
    credentials, project_id = google.auth.default()

    # Wrap them in impersonated credentials that can sign
    signing_credentials = impersonated_credentials.Credentials(
        source_credentials=credentials,
        target_principal = os.getenv("CLOUD_RUN_SERVICE_ACCOUNT"),
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
