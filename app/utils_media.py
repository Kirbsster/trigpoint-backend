from typing import Optional
import logging
from bson import ObjectId

from app.db import media_items_col
from app.storage import generate_signed_url

logger = logging.getLogger(__name__)


async def resolve_hero_url(hero_media_id: Optional[ObjectId]) -> Optional[str]:
    """Look up a media doc and return a signed hero URL (or None)."""
    if not hero_media_id:
        return None

    media_items = media_items_col()
    media_doc = await media_items.find_one({"_id": hero_media_id})
    if not media_doc:
        logger.warning("No media doc found for hero_media_id=%s", hero_media_id)
        return None

    variants = media_doc.get("variants") or {}
    key = (variants.get("high") or {}).get("storage_key") or media_doc.get("storage_key")
    if not key:
        return None

    try:
        return generate_signed_url(key, expires_in=3600)
    except Exception as e:
        logger.warning(
            "Failed to generate signed URL for media %s (key=%s): %s",
            hero_media_id,
            key,
            e,
        )
        return None


async def resolve_hero_variant_url(
    hero_media_id: Optional[ObjectId],
    variant: str,
) -> Optional[str]:
    """Look up a media doc and return a signed URL for a named variant."""
    if not hero_media_id:
        return None

    media_items = media_items_col()
    media_doc = await media_items.find_one({"_id": hero_media_id})
    if not media_doc:
        return None

    variants = media_doc.get("variants") or {}
    key = (variants.get(variant) or {}).get("storage_key")
    if not key:
        return None

    try:
        return generate_signed_url(key, expires_in=3600)
    except Exception as e:
        logger.warning(
            "Failed to generate signed URL for media variant=%s media=%s (key=%s): %s",
            variant,
            hero_media_id,
            key,
            e,
        )
        return None
