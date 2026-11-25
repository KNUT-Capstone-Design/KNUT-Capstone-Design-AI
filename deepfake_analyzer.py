import io
import os
import uuid
import torch
import timm
import torch.nn.functional as F
from datetime import datetime, timezone
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# 1. 모델 로드 (같은 폴더의 best_detector_model.pth 사용)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_detector_model.pth")
ANALYSIS_VERSION = "deepfake-effb4-v1.0.0"

model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 2. POST /api/v1/analysis/deepfake 용 핵심 함수 (메모리 바이트 직접 처리)
def analyze_deepfake_bytes_api(image_bytes: bytes) -> dict:
    # 디스크에 임시 저장하지 않고 메모리에서 바로 이미지 로드
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise ValueError("INVALID_IMAGE")

    width, height = image.size
    if width > 4096 or height > 4096:
        raise ValueError("IMAGE_DIMENSIONS_EXCEEDED")

    # AI 추론
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    # [0]: 진짜(Real) 확률, [1]: 가짜(Fake) 확률
    real_prob = round(float(probs[0]) * 100, 2)
    fake_prob = round(float(probs[1]) * 100, 2)

    # 명세서 4.1절 규격에 맞춰 0~100점 및 등급 변환
    score = int(round(fake_prob))
    if score < 30:
        level = "safe"
    elif score < 60:
        level = "caution"
    else:
        level = "danger"

    findings = []
    critical_rule_ids = []

    if level == "danger":
        critical_rule_ids.append("AI_DEEPFAKE_DETECTED")
        findings.append({
            "rule_id": "AI_DEEPFAKE_DETECTED",
            "severity": "critical",
            "description": f"AI 이미지 분석 결과 딥페이크(조작·합성) 가능성이 매우 높습니다({fake_prob}%)."
        })
        summary = "딥페이크로 조작되었을 가능성이 높은 이미지입니다. 사칭 피해에 주의하세요."
        recommendations = [
            {"code": "DO_NOT_TRUST_MEDIA", "message": "해당 이미지를 근거로 금전 거래나 개인정보 제공을 하지 마세요."},
            {"code": "VERIFY_OFFICIAL_CHANNEL", "message": "영상통화나 공식 연락처 등 다른 수단으로 본인 여부를 확인하세요."}
        ]
    elif level == "caution":
        findings.append({
            "rule_id": "AI_DEEPFAKE_SUSPICIOUS",
            "severity": "warning",
            "description": f"일부 합성·편집 흔적이 감지되었습니다(가짜 확률 {fake_prob}%)."
        })
        summary = "이미지에 일부 조작 또는 과도한 보정 흔적이 의심됩니다. 출처를 확인하세요."
        recommendations = [
            {"code": "VERIFY_OFFICIAL_CHANNEL", "message": "원본 출처를 확인하고 필요시 다른 경로로 본인 여부를 확인하세요."}
        ]
    else:
        summary = "AI 이미지 분석에서 뚜렷한 딥페이크 조작 신호가 발견되지 않았습니다."
        recommendations = [
            {"code": "VERIFY_OFFICIAL_CHANNEL", "message": "AI 판별은 100% 완벽하지 않으므로 의심 시 직접 확인하세요."}
        ]

    return {
        "request_id": str(uuid.uuid4()),
        "data": {
            "type": "deepfake",
            "status": "completed",
            "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "analysis_version": ANALYSIS_VERSION,
            "risk": {
                "score": score,
                "base_level": level,
                "level": level,
                "overridden": False,
                "critical_rule_ids": critical_rule_ids,
                "summary": summary,
                "findings": findings,
                "recommendations": recommendations
            },
            "reputation_status": "not_applicable",
            "warnings": [],
            "details": {
                "real_prob": real_prob,
                "fake_prob": fake_prob,
                "image_width": width,
                "image_height": height
            }
        }
    }
