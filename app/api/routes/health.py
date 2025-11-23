from fastapi import APIRouter
from app.core.config import settings
from datetime import datetime

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    from app.core.yolo_model import YOLOModelManager

    model_manager = YOLOModelManager()

    return {
        "status": "healthy" if model_manager.is_loaded() else "unhealthy",
        "model_loaded": model_manager.is_loaded(),
        "model_name": model_manager.get_model_name(),
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }
