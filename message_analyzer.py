import os
import re
import uuid
import torch
import torch.nn.functional as F
from datetime import datetime, timezone
from urllib.parse import urlparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from qr_url_analyzer import evaluate_url_risk

ANALYSIS_VERSION = "msg-kcelectra-hybrid-v2.0.0"

# 1. KcELECTRA 모델 로드 (같은 폴더의 kcelectra_bigdata_model 폴더 사용)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "kcelectra_bigdata_model")

msg_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
msg_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
msg_model.eval()

# 2. DB 설계서 7대 유형 기반 규칙 (하이라이트 matched_spans + 명확한 근거 제공용)
MESSAGE_RULES = [
    (
        "MSG_APK_INSTALL",
        "critical",
        60,
        "악성 앱(APK) 설치나 출처 불명의 앱 다운로드를 유도하는 표현이 감지되었습니다.",
        r"(?:\.apk|앱\s*설치|어플\s*설치|원격\s*제어|팀뷰어|애니데스크)"
    ),
    (
        "MSG_PUBLIC_AGENCY",
        "warning",
        30,
        "검찰·경찰·국세청·법원 등 공공기관을 사칭하는 표현이 포함되어 있습니다.",
        r"(?:검찰청?|경찰청?|국세청|법원|과태료|범칙금|사건\s*조회|출석\s*요구|민원24|정부24|건강보험)"
    ),
    (
        "MSG_FINANCE_URGENT",
        "warning",
        35,
        "결제 승인·계좌 정지·대출 등 금융 불안감을 조성하는 표현이 감지되었습니다.",
        r"(?:해외\s*결제|승인\s*완료|결제\s*완료|카드\s*발급|계좌\s*정지|저금리\s*대출|정부\s*지원금|본인\s*아닐\s*시)"
    ),
    (
        "MSG_DELIVERY_SCAM",
        "warning",
        25,
        "택배 배송 불가·주소지 불일치 등 택배 사칭 패턴이 감지되었습니다.",
        r"(?:주소지?\s*불일치|배송\s*불가|반송\s*예정|운송장\s*확인|택배\s*보관|미수령)"
    ),
    (
        "MSG_INVITATION_OBITUARY",
        "warning",
        25,
        "모바일 청첩장·부고장·돌잔치 등 지인 알림을 사칭하는 표현이 포함되어 있습니다.",
        r"(?:모바일\s*청첩장|부고\s*안내|부고장|장례식장|결혼식\s*초대|돌잔치\s*초대)"
    ),
    (
        "MSG_IMPERSONATION_FAMILY",
        "warning",
        35,
        "가족·지인을 사칭해 휴대폰 고장이나 금전·인증을 요구하는 패턴이 감지되었습니다.",
        r"(?:액정\s*파손|폰이?\s*고장|수리\s*맡겨|대신\s*결제|문화상품권|기프트카드|신분증\s*사진|비밀번호\s*알려)"
    ),
    (
        "MSG_INVESTMENT_SCAM",
        "warning",
        25,
        "고수익·급등주·리딩방 등 투자 사기 유도 표현이 포함되어 있습니다.",
        r"(?:고수익|원금\s*보장|급등주|무료\s*리딩|VIP\s*방|종목\s*추천|수익률\s*\d+%)"
    ),
    (
        "MSG_URGENCY_PRESSURE",
        "info",
        15,
        "즉시 확인·기한 임박 등 심리적 압박을 주는 표현이 포함되어 있습니다.",
        r"(?:즉시\s*확인|당일\s*마감|긴급\s*안내|미확인\s*시|자동\s*결제|법적\s*조치)"
    ),
]

URL_EXTRACT_REGEX = re.compile(r"https?://[^\s<>\"'\)\]]+", re.IGNORECASE)


def _normalize_url(raw_url: str) -> str:
    cleaned = raw_url.rstrip(".,;:!?")
    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        raise ValueError("INVALID_URL")
    normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized


def _predict_kcelectra_prob(text: str) -> float:
    inputs = msg_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(DEVICE)
    with torch.no_grad():
        logits = msg_model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]
    return float(probs[1])


def analyze_message_api(raw_message: str) -> dict:
    if not isinstance(raw_message, str) or not raw_message.strip():
        raise ValueError("VALIDATION_ERROR")
    if len(raw_message) > 5000:
        raise ValueError("VALIDATION_ERROR")

    char_count = len(raw_message)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. 규칙 기반 패턴 분석 + matched_spans 추출
    rule_score = 0
    matched_spans = []
    findings = []
    critical_rule_ids = []

    for rule_id, severity, weight, description, pattern in MESSAGE_RULES:
        matches = list(re.finditer(pattern, raw_message, re.IGNORECASE))
        if matches:
            rule_score += weight
            if severity == "critical":
                critical_rule_ids.append(rule_id)
            findings.append({
                "rule_id": rule_id,
                "severity": severity,
                "description": description
            })
            for m in matches:
                matched_spans.append({
                    "start": m.start(),
                    "end": m.end(),
                    "rule_id": rule_id
                })

    rule_score = min(rule_score, 100)
    matched_spans.sort(key=lambda x: (x["start"], x["end"]))

    # 2. KcELECTRA 딥러닝 문맥 추론
    ai_msg_prob = _predict_kcelectra_prob(raw_message)
    ai_msg_score = int(round(ai_msg_prob * 100))

    if ai_msg_score >= 60:
        findings.append({
            "rule_id": "AI_MSG_SMISHING_PATTERN",
            "severity": "critical",
            "description": f"AI 딥러닝(KcELECTRA) 문맥 분석 결과 스미싱 가능성이 매우 높습니다({ai_msg_score}%)."
        })
    elif ai_msg_score >= 30:
        findings.append({
            "rule_id": "AI_MSG_SMISHING_SUSPICIOUS",
            "severity": "warning",
            "description": f"AI 딥러닝(KcELECTRA) 분석 결과 스미싱 의심 문맥이 감지되었습니다({ai_msg_score}%)."
        })

    # 문자 자체 점수는 규칙 점수와 KcELECTRA 점수 중 최댓값 사용
    msg_score = max(rule_score, ai_msg_score)

    # 3. URL 추출, 정규화, 중복 제거 (최대 5개)
    raw_urls = URL_EXTRACT_REGEX.findall(raw_message)
    unique_candidates = []
    seen = set()

    for raw_u in raw_urls:
        try:
            norm_u = _normalize_url(raw_u)
            key = norm_u
        except ValueError:
            norm_u = None
            key = raw_u

        if key not in seen:
            seen.add(key)
            unique_candidates.append((raw_u, norm_u))

    url_candidate_count = len(unique_candidates)
    max_urls = 5
    omitted_url_count = max(0, url_candidate_count - max_urls)
    target_candidates = unique_candidates[:max_urls]

    warnings = []
    overall_status = "completed"

    if omitted_url_count > 0:
        overall_status = "partial"
        warnings.append({
            "code": "URL_LIMIT_EXCEEDED",
            "message": f"URL은 최대 {max_urls}개까지만 분석하며, 초과된 {omitted_url_count}개는 생략되었습니다."
        })

    # 4. 추출된 URL 분석 (LightGBM 모델 연계)
    url_results = []
    analyzed_url_count = 0
    successful_url_scores = []
    has_url_failure = False

    for idx, (raw_u, norm_u) in enumerate(target_candidates):
        if norm_u is None:
            has_url_failure = True
            url_results.append({
                "index": idx,
                "url": raw_u,
                "status": "failed",
                "score": None,
                "level": None,
                "error_code": "INVALID_URL"
            })
            continue

        try:
            u_risk = evaluate_url_risk(norm_u)
            analyzed_url_count += 1
            successful_url_scores.append(u_risk["score"])

            for cid in u_risk["critical_rule_ids"]:
                if cid not in critical_rule_ids:
                    critical_rule_ids.append(cid)
            for f in u_risk["findings"]:
                if not any(existing["rule_id"] == f["rule_id"] for existing in findings):
                    findings.append(f)

            url_results.append({
                "index": idx,
                "url": norm_u,
                "status": "completed",
                "score": u_risk["score"],
                "level": u_risk["level"],
                "error_code": None
            })
        except Exception:
            has_url_failure = True
            url_results.append({
                "index": idx,
                "url": norm_u,
                "status": "failed",
                "score": None,
                "level": None,
                "error_code": "INTERNAL_ERROR"
            })

    if has_url_failure:
        overall_status = "partial"
        warnings.append({
            "code": "URL_ANALYSIS_FAILED",
            "message": "문자 내 일부 URL 분석에 실패하여 가능한 결과만 반영했습니다."
        })

    # 5. 종합 점수 및 등급 계산
    final_score = max([msg_score] + successful_url_scores)
    final_score = min(max(final_score, 0), 100)

    if final_score < 30:
        base_level = "safe"
    elif final_score < 60:
        base_level = "caution"
    else:
        base_level = "danger"

    overridden = False
    level = base_level
    if critical_rule_ids and base_level != "danger":
        level = "danger"
        overridden = True

    if level == "danger":
        summary = "스미싱 위험이 매우 높습니다. 링크를 누르거나 앱을 설치하지 마세요."
        recommendations = [
            {"code": "DO_NOT_VISIT", "message": "문자 속 링크를 클릭하거나 첨부 파일을 실행하지 마세요."},
            {"code": "VERIFY_OFFICIAL_CHANNEL", "message": "공식 앱이나 기관 대표번호로 사실 여부를 확인하세요."}
        ]
    elif level == "caution":
        summary = "스미싱에 자주 쓰이는 의심 표현이나 문맥이 발견되었습니다. 출처를 꼭 확인하세요."
        recommendations = [
            {"code": "VERIFY_OFFICIAL_CHANNEL", "message": "발신 번호가 아닌 공식 앱이나 공식 연락처로 확인하세요."}
        ]
    else:
        summary = "문자 패턴 및 AI 분석에서 뚜렷한 위험 신호가 발견되지 않았습니다. 발신자가 불분명하면 주의하세요."
        recommendations = [
            {"code": "VERIFY_OFFICIAL_CHANNEL", "message": "공식 앱이나 공식 연락처로 확인하세요."}
        ]

    return {
        "request_id": str(uuid.uuid4()),
        "data": {
            "type": "message",
            "status": overall_status,
            "analyzed_at": now_iso,
            "analysis_version": ANALYSIS_VERSION,
            "risk": {
                "score": final_score,
                "base_level": base_level,
                "level": level,
                "overridden": overridden,
                "critical_rule_ids": critical_rule_ids,
                "summary": summary,
                "findings": findings,
                "recommendations": recommendations
            },
            "reputation_status": "not_applicable" if url_candidate_count == 0 else "disabled",
            "warnings": warnings,
            "details": {
                "character_count": char_count,
                "url_candidate_count": url_candidate_count,
                "analyzed_url_count": analyzed_url_count,
                "omitted_url_count": omitted_url_count,
                "matched_spans": matched_spans,
                "url_results": url_results
            }
        }
    }
