import asyncio
from typing import Any

import httpx

from e_customer_service.api_gateway.config import settings
from e_customer_service.api_gateway.schemas import SourceDocument


async def retrieve_report_context(query: str, top_k: int = 5) -> list[SourceDocument]:
    async with httpx.AsyncClient(timeout=settings.rag_timeout_seconds) as client:
        response = await asyncio.wait_for(
            client.post(settings.rag_url, json={"query": query, "top_k": top_k}),
            timeout=settings.rag_timeout_seconds,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

    documents = data.get("documents", [])
    return [SourceDocument(**document) for document in documents]

