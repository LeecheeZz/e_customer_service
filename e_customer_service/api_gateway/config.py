import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GatewaySettings:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    customer_vllm_base_url: str = os.getenv("CUSTOMER_VLLM_BASE_URL", "http://localhost:8000")
    customer_model: str = os.getenv("CUSTOMER_MODEL", "customer-service")
    report_vllm_base_url: str = os.getenv("REPORT_VLLM_BASE_URL", "http://localhost:8000")
    report_model: str = os.getenv("REPORT_MODEL", "customer-service")
    rag_url: str = os.getenv("RAG_URL", "http://localhost:8010/retrieve")
    redis_timeout_seconds: float = float(os.getenv("REDIS_TIMEOUT_SECONDS", "0.2"))
    rag_timeout_seconds: float = float(os.getenv("RAG_TIMEOUT_SECONDS", "3.0"))
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "8.0"))
    customer_cache_ttl_seconds: int = int(os.getenv("CUSTOMER_CACHE_TTL_SECONDS", "1800"))
    per_user_rate_limit_per_minute: int = int(os.getenv("PER_USER_RATE_LIMIT_PER_MINUTE", "60"))


settings = GatewaySettings()

