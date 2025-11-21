from typing import Optional
from pymongo import AsyncMongoClient
from pymongo.errors import ServerSelectionTimeoutError
from .settings import settings

_client: Optional[AsyncMongoClient] = None
_db = None

def get_client() -> AsyncMongoClient:
    global _client
    if _client is None:
        _client = AsyncMongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=3000,
            appname=settings.app_name,
        )
    return _client

def get_db():
    global _db
    if _db is None:
        _db = get_client()[settings.mongodb_db_name]
    return _db

async def ping() -> bool:
    try:
        await get_db().command("ping")
        return True
    except ServerSelectionTimeoutError:
        return False

async def ensure_indexes():
    users = get_db()["users"]
    await users.create_index(
        [("email_norm", 1)],
        unique=True,
        name="uniq_email_norm",
        partialFilterExpression={"email_norm": {"$type": "string"}},
    )