import math
import os
import re
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import cv2
import joblib
import numpy as np
import requests

ANALYSIS_VERSION = "qr-url-hybrid-v2.0.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "qr_url_model.pkl")

_bundle = joblib.load(MODEL_PATH)
url_model = _bundle["model"]

PROTECTED_BRANDS = [
    "naver", "kakao", "kakaobank", "kakaopay", "daum", "google", "apple",
    "microsoft", "coupang", "baemin", "toss", "kbstar", "shinhan",
    "wooribank", "hanabank", "nonghyup", "epost", "cjlogistics", "hanjin",
    "hometax", "police", "korail", "youtube", "instagram", "facebook", "samsung"
]

WHITELIST_DOMAINS = {
    "naver.com", "navercorp.com", "kakao.com", "kakaocorp.com", "kakaobank.com",
    "kakaopay.com", "daum.net", "google.com", "youtube.com", "apple.com",
    "microsoft.com", "coupang.com", "baemin.com", "woowahan.com", "toss.im",
    "kbstar.com", "shinhan.com", "wooribank.com", "hanabank.com", "nonghyup.com",
    "ibk.co.kr", "cjlogistics.com", "hanjin.com", "lotteglogis.com",
    "musinsa.com", "samsung.com", "kt.com", "sktelecom.com", "uplus.co.kr",
    "github.com", "instagram.com", "facebook.com"
}

SHORTENER_DOMAINS = {
    "bit.ly", "han.gl", "me2.do", "tinyurl.com", "t.co", "goo.gl",
    "is.gd", "v.gd", "url.kr", "vo.la", "c11.kr", "zrr.kr",
    "shorturl.at", "rebrand.ly", "ow.ly", "cutt.ly"
}

SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "pw", "cc",
    "buzz", "click", "loan", "work", "zip", "mov", "icu", "monster"
}

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "secure", "update", "account", "banking",
    "confirm", "wallet", "support", "service", "admin", "free", "bonus",
    "apk", "download", "pay", "auth", "cert", "event", "refund", "parcel"
]


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def is_whitelisted_host(host: str) -> bool:
    if not host:
        return False
    if host.endswith((".go.kr", ".ac.kr", ".or.kr", ".mil.kr")):
        return True
    return any(host == w or host.endswith("." + w) for w in WHITELIST_DOMAINS)


def check_brand_spoofing(host: str) -> tuple[bool, str]:
    if not host or is_whitelisted_host(host):
        return False, ""
    parts = host.split(".")
    sld = parts[-2] if len(parts) >= 2 else host
    tokens = set(re.split(r"[-.]", host))

    for brand in PROTECTED_BRANDS:
        if brand in host:
            return True, brand
        if len(brand) >= 5:
            for tok in [sld] + list(tokens):
                if abs(len(tok) - len(brand)) <= 2 and 1 <= _levenshtein(tok, brand) <= 2:
                    return True, brand
    return False, ""


def extract_url_features(url: str) -> list[float]:
    u = url.strip()
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    parsed = urlparse(u)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    full_lower = u.lower()

    url_len = len(u)
    host_len = len(host)
    path_len = len(path)
    query_len = len(query)

    dot_count = u.count(".")
    hyphen_count = u.count("-")
    underscore_count = u.count("_")
    slash_count = u.count("/")
    at_count = u.count("@")
    question_count = u.count("?")
    equal_count = u.count("=")
    percent_count = u.count("%")
    amp_count = u.count("&")

    digit_count = sum(ch.isdigit() for ch in u)
    digit_ratio = digit_count / max(url_len, 1)

    url_entropy = _shannon_entropy(u)
    host_entropy = _shannon_entropy(host)

    is_ip = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host) else 0
    is_https = 1 if parsed.scheme.lower() == "https" else 0

    host_parts = [p for p in host.split(".") if p]
    subdomain_depth = max(0, len(host_parts) - 2)
    tld = host_parts[-1] if host_parts else ""
    has_suspicious_tld = 1 if tld in SUSPICIOUS_TLDS else 0

    suspicious_kw_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in full_lower)
    has_punycode = 1 if "xn--" in host else 0
    spoofed, _ = check_brand_spoofing(host)
    brand_spoof_flag = 1 if spoofed else 0

    return [
        url_len, host_len, path_len, query_len,
        dot_count, hyphen_count, underscore_count, slash_count,
        at_count, question_count, equal_count, percent_count, amp_count,
        digit_count, digit_ratio, url_entropy, host_entropy,
        is_ip, is_https, subdomain_depth, has_suspicious_tld,
        suspicious_kw_count, has_punycode, brand_spoof_flag
    ]


def _normalize_url(raw_url: str) -> str:
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ValueError("VALIDATION_ERROR")
    cleaned = raw_url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", cleaned):
        cleaned = "http://" + cleaned
    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise ValueError("INVALID_URL")
    normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized


def _resolve_short_url(url: str) -> tuple[str, bool, list[str], bool]:
    """단축 URL인 경우 본문 다운로드 없이 리다이렉트 최종 목적지만 안전하게 추적"""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    is_short = any(host == s or host.endswith("." + s) for s in SHORTENER_DOMAINS)
    if not is_short:
        return url, False, [url], False

    chain = [url]
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; KNUT-Security-Scanner/2.0)"}
        resp = requests.head(url, allow_redirects=True, timeout=3.0, headers=headers)
        if resp.status_code in (403, 405) or not resp.history:
            with requests.get(url, allow_redirects=True, timeout=3.0, stream=True, headers=headers) as r_get:
                for h in r_get.history:
                    if h.url not in chain:
                        chain.append(h.url)
                if r_get.url not in chain:
                    chain.append(r_get.url)
                return r_get.url, True, chain, False
        else:
            for h in resp.history:
                if h.url not in chain:
                    chain.append(h.url)
            if resp.url not in chain:
                chain.append(resp.url)
            return resp.url, True, chain, False
    except Exception:
        return url, True, chain, True


def evaluate_url_risk(url: str) -> dict:
    """단일 URL 위험도 종합 평가 (message_analyzer.py와 qr_url_analyzer.py 공용)"""
    norm_url = _normalize_url(url)
    final_url, is_shortened, redirect_chain, resolve_failed = _resolve_short_url(norm_url)

    target_url = final_url if not resolve_failed else norm_url
    parsed = urlparse(target_url)
    host = (parsed.hostname or "").lower()
    full_lower = target_url.lower()

    findings = []
    critical_rule_ids = []
    rule_score = 0

    # 1. 단축 URL 사용 탐지
    if is_shortened:
        if resolve_failed:
            rule_score += 35
            findings.append({
                "rule_id": "URL_SHORTENER_UNRESOLVED",
                "severity": "warning",
                "description": "단축 URL이 사용되었으나 실제 최종 목적지 주소를 확인할 수 없습니다."
            })
        elif final_url != norm_url:
            rule_score += 15
            findings.append({
                "rule_id": "URL_SHORTENER_REDIRECTED",
                "severity": "info",
                "description": f"단축 URL이 실제 목적지({final_url})로 연결되는 것을 확인했습니다."
            })

    # 2. 공식 화이트리스트 확인
    whitelisted = is_whitelisted_host(host)
    if whitelisted:
        ai_prob = 0.02
    else:
        feats = np.array([extract_url_features(target_url)], dtype=np.float32)
        ai_prob = float(url_model.predict_proba(feats)[0][1])

    # 3. 정밀 보안 규칙 검사
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
        rule_score += 55
        critical_rule_ids.append("URL_RAW_IP_HOST")
        findings.append({
            "rule_id": "URL_RAW_IP_HOST",
            "severity": "critical",
            "description": "도메인 이름 대신 IP 주소를 직접 사용하는 비정상적인 주소입니다."
        })

    if full_lower.endswith(".apk") or ".apk?" in full_lower:
        rule_score += 65
        critical_rule_ids.append("URL_DIRECT_APK_DOWNLOAD")
        findings.append({
            "rule_id": "URL_DIRECT_APK_DOWNLOAD",
            "severity": "critical",
            "description": "악성 앱(APK) 파일을 직접 다운로드하도록 유도하는 링크입니다."
        })

    if "xn--" in host:
        rule_score += 45
        critical_rule_ids.append("URL_PUNYCODE_HOMOGRAPH")
        findings.append({
            "rule_id": "URL_PUNYCODE_HOMOGRAPH",
            "severity": "critical",
            "description": "유사 문자로 정상 사이트를 위장하는 국제화 도메인(Punycode) 공격 패턴이 감지되었습니다."
        })

    spoofed, brand_name = check_brand_spoofing(host)
    if spoofed:
        rule_score += 55
        critical_rule_ids.append("URL_BRAND_IMPERSONATION")
        findings.append({
            "rule_id": "URL_BRAND_IMPERSONATION",
            "severity": "critical",
            "description": f"공식 '{brand_name}' 도메인을 교묘하게 사칭하거나 오타를 유도한 유사 도메인입니다."
        })

    if "@" in target_url:
        rule_score += 40
        findings.append({
            "rule_id": "URL_AT_SYMBOL_OBFUSCATION",
            "severity": "warning",
            "description": "URL에 '@' 기호를 넣어 실제 접속 주소를 숨기는 기법이 감지되었습니다."
        })

    host_parts = [p for p in host.split(".") if p]
    tld = host_parts[-1] if host_parts else ""
    if tld in SUSPICIOUS_TLDS and not whitelisted:
        rule_score += 25
        findings.append({
            "rule_id": "URL_SUSPICIOUS_TLD",
            "severity": "warning",
            "description": f"피싱 사이트에 자주 악용되는 최상위 도메인(.{tld})이 사용되었습니다."
        })

    if parsed.scheme.lower() == "http" and not whitelisted:
        rule_score += 10
        findings.append({
            "rule_id": "URL_INSECURE_HTTP",
            "severity": "info",
            "description": "보안 인증서(HTTPS)가 적용되지 않은 암호화 미지원 주소입니다."
        })

    ai_score = int(round(ai_prob * 100))
    if not whitelisted:
        if ai_score >= 65:
            findings.append({
                "rule_id": "AI_URL_PHISHING_HIGH",
                "severity": "critical",
                "description": f"AI 구조·엔트로피 분석 결과 피싱 URL 가능성이 매우 높습니다({ai_score}%)."
            })
        elif ai_score >= 35:
            findings.append({
                "rule_id": "AI_URL_PHISHING_SUSPICIOUS",
                "severity": "warning",
                "description": f"AI 구조 분석 결과 피싱 의심 특징이 감지되었습니다({ai_score}%)."
            })

    final_score = min(max(max(rule_score, ai_score), 0), 100)
    if whitelisted and not critical_rule_ids:
        final_score = min(final_score, 15)

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

    return {
        "original_url": norm_url,
        "final_url": final_url,
        "is_shortened": is_shortened,
        "redirect_chain": redirect_chain,
        "is_whitelisted": whitelisted,
        "score": final_score,
        "base_level": base_level,
        "level": level,
        "overridden": overridden,
        "critical_rule_ids": critical_rule_ids,
        "findings": findings
    }


def _build_risk_summary_and_recs(level: str) -> tuple[str, list[dict]]:
    if level == "danger":
        return (
            "피싱 또는 악성 사이트 위험이 매우 높습니다. 절대 접속하거나 개인정보를 입력하지 마세요.",
            [
                {"code": "DO_NOT_VISIT", "message": "링크 접속 및 파일 다운로드를 즉시 중단하세요."},
                {"code": "VERIFY_OFFICIAL_CHANNEL", "message": "포털 검색이나 공식 앱을 통해 직접 접속하세요."}
            ]
        )
    elif level == "caution":
        return (
            "의심스러운 도메인 구조나 단축 링크가 감지되었습니다. 접속 전 출처를 확인하세요.",
            [
                {"code": "VERIFY_OFFICIAL_CHANNEL", "message": "도메인 철자가 공식 사이트와 일치하는지 확인하세요."}
            ]
        )
    else:
        return (
            "URL 구조 및 AI 분석에서 뚜렷한 피싱 위험 신호가 발견되지 않았습니다.",
            [
                {"code": "VERIFY_OFFICIAL_CHANNEL", "message": "로그인이나 결제 전 주소창의 도메인을 한 번 더 확인하세요."}
            ]
        )


def analyze_url_api(raw_url: str) -> dict:
    """URL 단건 분석 API 엔드포인트용 함수"""
    u_risk = evaluate_url_risk(raw_url)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary, recommendations = _build_risk_summary_and_recs(u_risk["level"])

    return {
        "request_id": str(uuid.uuid4()),
        "data": {
            "type": "url",
            "status": "completed",
            "analyzed_at": now_iso,
            "analysis_version": ANALYSIS_VERSION,
            "risk": {
                "score": u_risk["score"],
                "base_level": u_risk["base_level"],
                "level": u_risk["level"],
                "overridden": u_risk["overridden"],
                "critical_rule_ids": u_risk["critical_rule_ids"],
                "summary": summary,
                "findings": u_risk["findings"],
                "recommendations": recommendations
            },
            "reputation_status": "disabled",
            "warnings": [],
            "details": {
                "original_url": u_risk["original_url"],
                "final_url": u_risk["final_url"],
                "is_shortened": u_risk["is_shortened"],
                "redirect_chain": u_risk["redirect_chain"],
                "is_whitelisted": u_risk["is_whitelisted"]
            }
        }
    }


def _decode_qr_from_bytes(image_bytes: bytes) -> str:
    """메모리 바이트에서 다단계 전처리(원본 -> 그레이 -> CLAHE -> 이진화 -> 확대)로 QR 디코딩"""
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)) or len(image_bytes) == 0:
        raise ValueError("VALIDATION_ERROR")

    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("INVALID_IMAGE")

    detector = cv2.QRCodeDetector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 대비 강화 (저조도/그림자 촬영 보정)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Otsu 이진화
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    candidates = [img, gray, enhanced, thresh]

    # 이미지가 작을 경우 2배 확대 후보 추가
    h, w = gray.shape[:2]
    if max(h, w) < 500:
        upscaled = cv2.resize(enhanced, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        candidates.append(upscaled)

    for candidate in candidates:
        data, _, _ = detector.detectAndDecode(candidate)
        if data and data.strip():
            return data.strip()

    raise ValueError("QR_NOT_FOUND")


def analyze_qr_api(image_bytes: bytes) -> dict:
    """QR 이미지(메모리 바이트) 큐싱 탐지 API 엔드포인트용 함수"""
    payload = _decode_qr_from_bytes(image_bytes)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lower_payload = payload.lower()

    # Case 1: 웹 URL(http/https 또는 도메인 형태)인 경우 -> URL 엔진으로 정밀 분석
    if lower_payload.startswith(("http://", "https://")) or re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$", payload):
        u_risk = evaluate_url_risk(payload)
        summary, recommendations = _build_risk_summary_and_recs(u_risk["level"])
        return {
            "request_id": str(uuid.uuid4()),
            "data": {
                "type": "qr",
                "status": "completed",
                "analyzed_at": now_iso,
                "analysis_version": ANALYSIS_VERSION,
                "risk": {
                    "score": u_risk["score"],
                    "base_level": u_risk["base_level"],
                    "level": u_risk["level"],
                    "overridden": u_risk["overridden"],
                    "critical_rule_ids": u_risk["critical_rule_ids"],
                    "summary": summary,
                    "findings": u_risk["findings"],
                    "recommendations": recommendations
                },
                "reputation_status": "disabled",
                "warnings": [],
                "details": {
                    "qr_payload_type": "url",
                    "raw_payload": payload,
                    "original_url": u_risk["original_url"],
                    "final_url": u_risk["final_url"],
                    "is_shortened": u_risk["is_shortened"],
                    "redirect_chain": u_risk["redirect_chain"]
                }
            }
        }

    # Case 2: 비(非)웹사이트 QR 스킴 분석 (WIFI, SMSTO, TEL, APK 마켓 등)
    findings = []
    critical_rule_ids = []
    score = 10
    payload_type = "text"

    if lower_payload.startswith("wifi:"):
        payload_type = "wifi"
        score = 45
        findings.append({
            "rule_id": "QR_WIFI_AUTO_CONNECT",
            "severity": "warning",
            "description": "공용·악성 Wi-Fi에 자동 연결을 유도하는 QR 코드입니다. 트래픽 가로채기 위험이 있으니 신뢰할 수 있는 장소인지 확인하세요."
        })
    elif lower_payload.startswith(("smsto:", "sms:", "tel:")):
        payload_type = "action_scheme"
        score = 50
        findings.append({
            "rule_id": "QR_AUTO_SMS_CALL",
            "severity": "warning",
            "description": "스캔 시 문자 발송이나 전화 연결을 자동으로 실행하는 QR 코드입니다. 소액결제나 유료 국제전화 연결에 악용될 수 있습니다."
        })
    elif lower_payload.startswith(("market://", "intent://")) or ".apk" in lower_payload:
        payload_type = "app_install"
        score = 75
        critical_rule_ids.append("QR_APP_INSTALL_SCHEME")
        findings.append({
            "rule_id": "QR_APP_INSTALL_SCHEME",
            "severity": "critical",
            "description": "앱 설치 화면이나 실행 인텐트를 직접 호출하는 QR 코드입니다. 출처가 불분명하면 절대 설치하지 마세요."
        })
    else:
        findings.append({
            "rule_id": "QR_PLAIN_TEXT",
            "severity": "info",
            "description": "외부 링크가 없는 일반 텍스트 형식의 QR 코드입니다."
        })

    base_level = "safe" if score < 30 else ("caution" if score < 60 else "danger")
    level = "danger" if critical_rule_ids else base_level
    overridden = bool(critical_rule_ids and base_level != "danger")
    summary, recommendations = _build_risk_summary_and_recs(level)

    return {
        "request_id": str(uuid.uuid4()),
        "data": {
            "type": "qr",
            "status": "completed",
            "analyzed_at": now_iso,
            "analysis_version": ANALYSIS_VERSION,
            "risk": {
                "score": score,
                "base_level": base_level,
                "level": level,
                "overridden": overridden,
                "critical_rule_ids": critical_rule_ids,
                "summary": summary,
                "findings": findings,
                "recommendations": recommendations
            },
            "reputation_status": "not_applicable",
            "warnings": [],
            "details": {
                "qr_payload_type": payload_type,
                "raw_payload": payload
            }
        }
    }
