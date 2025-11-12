from fastapi import APIRouter
from app.core.config import settings
from datetime import datetime

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""

    # Phase 2에서 모델 로드 상태 체크 추가 예정
    # from app.core.yolo_model import YOLOModelManager
    # model_manager = YOLOModelManager()

    return {
        "status": "healthy",
        "model_loaded": False,  # Phase 2에서 구현
        "model_name": "unknown",
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }
