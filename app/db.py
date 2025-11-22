# app/db.py
from typing import Optional

from pymongo import AsyncMongoClient
from pymongo.errors import ServerSelectionTimeoutError

from .settings import settings

_client: Optional[AsyncMongoClient] = None
_db = None


def get_client() -> AsyncMongoClient:
    """Return a singleton async MongoDB client."""
    global _client
    if _client is None:
        _client = AsyncMongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=3000,
            appname=settings.app_name,
        )
    return _client


def get_db():
    """Return the connected async database object."""
    global _db
    if _db is None:
        _db = get_client()[settings.mongodb_db_name]
    return _db


# ---- Collection helpers ----

def users_col():
    return get_db()["users"]

def bikes_col():
    return get_db()["bikes"]

def media_items_col():
    return get_db()["media_items"]


# ---- Health / indexes ----

async def ping() -> bool:
    try:
        await get_db().command("ping")
        return True
    except ServerSelectionTimeoutError:
        return False


async def ensure_indexes():
    db = get_db()

    users = db["users"]
    await users.create_index(
        [("email_norm", 1)],
        unique=True,
        name="uniq_email_norm",
        partialFilterExpression={"email_norm": {"$type": "string"}},
    )

    bikes = db["bikes"]
    await bikes.create_index(
        [("user_id", 1)],
        name="idx_bikes_user_id",
    )

    media = db["media_items"]
    await media.create_index(
        [("bike_id", 1)],
        name="idx_media_bike_id",
    )
    await media.create_index(
        [("user_id", 1)],
        name="idx_media_user_id",
    )