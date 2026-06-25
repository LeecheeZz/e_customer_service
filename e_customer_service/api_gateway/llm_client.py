from typing import Any

import httpx

from e_customer_service.api_gateway.config import settings


async def call_openai_compatible_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            url,
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": 0.9,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data["choices"][0]["message"]["content"]


async def call_customer_model(messages: list[dict[str, str]]) -> str:
    return await call_openai_compatible_chat(
        base_url=settings.customer_vllm_base_url,
        model=settings.customer_model,
        messages=messages,
        timeout_seconds=settings.llm_timeout_seconds,
        temperature=0.2,
        max_tokens=512,
    )


async def call_report_model(messages: list[dict[str, str]]) -> str:
    return await call_openai_compatible_chat(
        base_url=settings.report_vllm_base_url,
        model=settings.report_model,
        messages=messages,
        timeout_seconds=settings.llm_timeout_seconds,
        temperature=0.1,
        max_tokens=1200,
    )

