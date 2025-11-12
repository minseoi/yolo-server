from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import traceback
import uuid

from app.core.config import settings
from app.utils.logger import get_logger
from app.api.routes import health

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # 모델 로드는 Phase 2에서 구현
    # from app.core.yolo_model import YOLOModelManager
    # model_manager = YOLOModelManager()
    # model_manager.load_model(settings.MODEL_PATH)

    yield

    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# 라우터 등록
app.include_router(health.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Global exception: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "traceback": traceback.format_exc()
        }
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "request_id": str(uuid.uuid4())
        }
    )


@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }
