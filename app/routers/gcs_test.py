# app/routers/gcs_test.py
from fastapi import APIRouter, UploadFile, File
from google.cloud import storage
import uuid
import os

router = APIRouter(prefix="/test", tags=["test"])

BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "trigpoint-media")

@router.post("/upload")
async def upload_test_file(file: UploadFile = File(...)):
    """Simple sanity check: upload to GCS using Cloud Run service account."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # create a random key
    ext = (file.filename or "bin").split(".")[-1]
    key = f"test/{uuid.uuid4()}.{ext}"

    blob = bucket.blob(key)
    content = await file.read()
    blob.upload_from_string(content, content_type=file.content_type)

    return {
        "message": "Upload successful",
        "bucket": BUCKET_NAME,
        "key": key,
        "gs_path": f"gs://{BUCKET_NAME}/{key}",
    }