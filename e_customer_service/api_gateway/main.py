import asyncio

from fastapi import FastAPI, Request

from e_customer_service.api_gateway.cache import get_customer_cache, set_customer_cache
from e_customer_service.api_gateway.intent import classify_intent
from e_customer_service.api_gateway.llm_client import call_customer_model, call_report_model
from e_customer_service.api_gateway.prompts import build_customer_messages, build_report_messages
from e_customer_service.api_gateway.rag_client import retrieve_report_context
from e_customer_service.api_gateway.rate_limit import check_rate_limit
from e_customer_service.api_gateway.schemas import ChatRequest, ChatResponse

app = FastAPI(title="Customer Service + Research Report Gateway")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    client_host = request.client.host if request.client else "anonymous"
    identity = payload.user_id or payload.session_id or client_host
    check_rate_limit(identity)

    query = payload.query.strip()
    intent = classify_intent(query)

    if intent == "research_report":
        return await handle_report_query(query)
    return await handle_customer_query(payload)


async def handle_customer_query(payload: ChatRequest) -> ChatResponse:
    cached = await get_customer_cache(payload.query)
    if cached:
        return ChatResponse(
            answer=cached["answer"],
            intent="customer_service",
            source="cache",
            citations=[],
        )

    messages = build_customer_messages(
        query=payload.query,
        user_profile=payload.user_profile,
        order_info=payload.order_info,
    )

    try:
        answer = await call_customer_model(messages)
    except asyncio.TimeoutError:
        answer = "The customer-service model timed out. Please try again later."
    except Exception:
        answer = "The customer-service model is temporarily unavailable."

    await set_customer_cache(payload.query, answer)
    return ChatResponse(
        answer=answer,
        intent="customer_service",
        source="llm",
        citations=[],
    )


async def handle_report_query(query: str) -> ChatResponse:
    try:
        documents = await retrieve_report_context(query)
    except Exception:
        documents = []

    messages = build_report_messages(query, documents)

    try:
        answer = await call_report_model(messages)
    except asyncio.TimeoutError:
        answer = "The research-report model timed out. Please try again later."
    except Exception:
        answer = "The research-report service is temporarily unavailable."

    return ChatResponse(
        answer=answer,
        intent="research_report",
        source="rag",
        citations=documents,
    )
