import asyncio
import hashlib
import json
from typing import Any

import redis.asyncio as redis

from e_customer_service.api_gateway.config import settings

redis_client = redis.from_url(settings.redis_url, decode_responses=True)


def normalize_customer_query(query: str) -> str:
    return "".join(query.strip().lower().split())


def customer_cache_key(query: str) -> str:
    digest = hashlib.sha256(normalize_customer_query(query).encode("utf-8")).hexdigest()
    return f"customer_service:qa:{digest}"


async def get_customer_cache(query: str) -> dict[str, Any] | None:
    try:
        raw = await asyncio.wait_for(
            redis_client.get(customer_cache_key(query)),
            timeout=settings.redis_timeout_seconds,
        )
    except Exception:
        return None

    if not raw:
        return None
    return json.loads(raw)


async def set_customer_cache(query: str, answer: str) -> None:
    payload = json.dumps({"answer": answer}, ensure_ascii=False)
    try:
        await asyncio.wait_for(
            redis_client.setex(
                customer_cache_key(query),
                settings.customer_cache_ttl_seconds,
                payload,
            ),
            timeout=settings.redis_timeout_seconds,
        )
    except Exception:
        return

