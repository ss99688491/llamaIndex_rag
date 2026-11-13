"""Ten verified unanswerable questions for abstention evaluation."""

from __future__ import annotations

from typing import Any


NO_ANSWER_CASES = [
    {
        "question_id": "NA-01",
        "category": "无答案",
        "question": "小米 RD03 路由器的 USB 3.0 接口最大传输速度是多少？",
        "question_language": "zh",
        "target_document": "小米路由器.pdf",
        "absence_terms": ["USB 3.0", "USB 3", "SuperSpeed USB"],
        "why_unanswerable": "该路由器说明书只介绍网络端口和配网，没有 USB 接口或 USB 传输规格。",
        "expected_behavior": "说明当前手册未提供 USB 3.0 接口或速度。",
    },
    {
        "question_id": "NA-02",
        "category": "无答案",
        "question": "Logitech C920 内置电池充满后可以连续工作多久？",
        "question_language": "zh",
        "target_document": "高清摄像头.pdf",
        "absence_terms": ["built-in battery", "battery life", "rechargeable battery"],
        "why_unanswerable": "C920 说明书没有列出内置电池或电池续航参数。",
        "expected_behavior": "说明当前手册未提供内置电池续航。",
    },
    {
        "question_id": "NA-03",
        "category": "无答案",
        "question": "飞利浦 HD925X 空气炸锅的水箱容量是多少升？",
        "question_language": "zh",
        "target_document": "空气炸锅.pdf",
        "absence_terms": ["water tank", "water reservoir", "水箱", "水位", "水箱容量"],
        "why_unanswerable": "该空气炸锅为热空气煎炸产品，说明书没有水箱或水箱容量。",
        "expected_behavior": "说明当前手册未提供水箱容量。",
    },
    {
        "question_id": "NA-04",
        "category": "无答案",
        "question": "Osmo Mobile 6 的防水等级和最大水下使用深度是多少？",
        "question_language": "zh",
        "target_document": "云台.pdf",
        "absence_terms": ["waterproof", "underwater", "防水", "水下", "IPX"],
        "why_unanswerable": "云台说明书没有提供防水等级或水下使用深度。",
        "expected_behavior": "说明当前手册未提供防水和水下使用能力。",
    },
    {
        "question_id": "NA-05",
        "category": "无答案",
        "question": "OBD LA9039 的 Wi-Fi 默认 SSID 和密码是什么？",
        "question_language": "zh",
        "target_document": "OBD.pdf",
        "absence_terms": ["wifi", "wi-fi", "SSID", "Wi-Fi 密码"],
        "why_unanswerable": "OBD 说明书描述蜂窝网络上传和平台登录，没有 Wi-Fi 热点或默认凭据。",
        "expected_behavior": "说明当前手册未提供 Wi-Fi 默认 SSID 或密码。",
    },
    {
        "question_id": "NA-06",
        "category": "无答案",
        "question": "小米 RD03 路由器支持的最大 USB 存储共享容量是多少？",
        "question_language": "zh",
        "target_document": "小米路由器.pdf",
        "absence_terms": ["USB storage", "USB drive", "storage capacity"],
        "why_unanswerable": "说明书没有 USB 存储共享功能，也没有相关容量参数。",
        "expected_behavior": "说明当前手册未提供 USB 存储共享容量。",
    },
    {
        "question_id": "NA-07",
        "category": "无答案",
        "question": "飞利浦 HD925X 空气炸锅的蓝牙遥控最远距离是多少米？",
        "question_language": "zh",
        "target_document": "空气炸锅.pdf",
        "absence_terms": ["bluetooth", "蓝牙"],
        "why_unanswerable": "空气炸锅说明书没有蓝牙遥控功能或遥控距离。",
        "expected_behavior": "说明当前手册未提供蓝牙遥控距离。",
    },
    {
        "question_id": "NA-08",
        "category": "无答案",
        "question": "Logitech C920 的红外夜视功能最远可以识别多少米？",
        "question_language": "zh",
        "target_document": "高清摄像头.pdf",
        "absence_terms": ["infrared", "night vision", "红外", "夜视"],
        "why_unanswerable": "C920 说明书没有红外或夜视能力及距离参数。",
        "expected_behavior": "说明当前手册未提供红外夜视距离。",
    },
    {
        "question_id": "NA-09",
        "category": "无答案",
        "question": "Osmo Mobile 6 的太阳能充电功率是多少瓦？",
        "question_language": "zh",
        "target_document": "云台.pdf",
        "absence_terms": ["solar charging", "solar power", "太阳能"],
        "why_unanswerable": "云台说明书只说明 USB-C 充电，没有太阳能充电或功率。",
        "expected_behavior": "说明当前手册未提供太阳能充电功率。",
    },
    {
        "question_id": "NA-10",
        "category": "无答案",
        "question": "OBD LA9039 是否支持 Apple CarPlay，屏幕尺寸是多少英寸？",
        "question_language": "zh",
        "target_document": "OBD.pdf",
        "absence_terms": ["CarPlay", "screen size", "屏幕尺寸"],
        "why_unanswerable": "OBD 说明书没有 Apple CarPlay 或设备屏幕尺寸信息。",
        "expected_behavior": "说明当前手册未提供 CarPlay 支持和屏幕尺寸。",
    },
]


def verify_no_answer_cases(
    pdf_texts: dict[str, str], node_texts: dict[str, str]
) -> list[dict[str, Any]]:
    import re

    results = []
    for case in NO_ANSWER_CASES:
        document = case["target_document"]
        pdf_text = pdf_texts[document].lower()
        node_text = node_texts[document].lower()
        compact_pdf = re.sub(r"\s+", "", pdf_text)
        compact_nodes = re.sub(r"\s+", "", node_text)
        checks = []
        for term in case["absence_terms"]:
            term_lower = term.lower()
            compact_term = re.sub(r"\s+", "", term_lower)
            has_cjk = any("\u4e00" <= char <= "\u9fff" for char in term)
            pdf_present = (
                compact_term in compact_pdf if has_cjk else term_lower in pdf_text
            )
            nodes_present = (
                compact_term in compact_nodes
                if has_cjk
                else term_lower in node_text
            )
            checks.append(
                {
                    "term": term,
                    "present_in_pdf": pdf_present,
                    "present_in_nodes": nodes_present,
                }
            )
        if any(check["present_in_pdf"] or check["present_in_nodes"] for check in checks):
            raise ValueError(
                f"{case['question_id']} has a potentially answerable term: {checks}"
            )
        results.append(
            {
                **case,
                "verification_scope": "target PDF page text and target document Chroma nodes",
                "absence_checks": checks,
                "verified_unanswerable": True,
            }
        )
    return results
