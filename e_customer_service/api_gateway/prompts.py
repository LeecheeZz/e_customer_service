from typing import Any

from e_customer_service.api_gateway.schemas import SourceDocument


def build_customer_messages(
    *,
    query: str,
    user_profile: dict[str, Any] | None,
    order_info: dict[str, Any] | None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是电商平台智能客服。回复要礼貌、简洁、可执行。"
                "如涉及订单、退款、物流、优惠券、售后，请优先给出下一步处理方式。"
                "信息不足时说明需要补充的信息或建议转人工客服。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：{query}\n"
                f"用户信息：{user_profile or {}}\n"
                f"订单信息：{order_info or {}}\n"
                "请直接给出客服回复。"
            ),
        },
    ]


def build_report_messages(query: str, documents: list[SourceDocument]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{index}] 标题：{document.title or '未命名'}\n"
        f"来源：{document.source or '未知'}\n"
        f"内容：{document.content}"
        for index, document in enumerate(documents, start=1)
    )

    return [
        {
            "role": "system",
            "content": (
                "你是金融研报问答助手。只能基于检索资料回答，不要编造。"
                "结论后需要用 [1]、[2] 这样的编号标注引用来源。"
                "资料不足时要明确说明无法从现有资料判断。"
            ),
        },
        {
            "role": "user",
            "content": f"用户问题：{query}\n\n检索资料：\n{context or '无'}",
        },
    ]

