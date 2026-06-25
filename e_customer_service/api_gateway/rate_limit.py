import time
from collections import defaultdict, deque

from fastapi import HTTPException

from e_customer_service.api_gateway.config import settings

WINDOW_SECONDS = 60
_requests: dict[str, deque[float]] = defaultdict(deque)


def check_rate_limit(identity: str) -> None:
    now = time.monotonic()
    bucket = _requests[identity]

    while bucket and now - bucket[0] > WINDOW_SECONDS:
        bucket.popleft()

    if len(bucket) >= settings.per_user_rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Too many requests")

    bucket.append(now)

