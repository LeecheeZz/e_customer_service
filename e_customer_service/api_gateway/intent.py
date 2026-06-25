from e_customer_service.api_gateway.schemas import Intent


REPORT_KEYWORDS = {
    "研报",
    "财报",
    "估值",
    "评级",
    "目标价",
    "券商",
    "宏观",
    "行业分析",
    "公司分析",
    "利润率",
    "收入增速",
    "eps",
    "pe",
    "pb",
    "roe",
}

CUSTOMER_KEYWORDS = {
    "退款",
    "退货",
    "换货",
    "优惠券",
    "满减",
    "订单",
    "物流",
    "快递",
    "发票",
    "售后",
    "客服",
    "双11",
    "双十二",
    "保价",
}


def classify_intent(query: str) -> Intent:
    normalized = query.strip().lower()
    report_hits = sum(1 for keyword in REPORT_KEYWORDS if keyword in normalized)
    customer_hits = sum(1 for keyword in CUSTOMER_KEYWORDS if keyword in normalized)

    if report_hits > customer_hits:
        return "research_report"
    return "customer_service"

