from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.schemas.detection import DetectionResponse
from app.core.yolo_model import YOLOModelManager
from app.core.config import settings
from app.utils.image_processing import ImageProcessor
from app.utils.logger import get_logger
import time
import uuid
from pathlib import Path

router = APIRouter(prefix="/api/v1", tags=["detection"])
logger = get_logger(__name__)


@router.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(..., description="이미지 파일 (jpg, png, bmp)"),
    confidence: float = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="탐지 신뢰도 임계값 (0.0-1.0), None이면 Config 기본값 사용"
    )
):
    """
    객체 탐지 엔드포인트

    **하이브리드 파라미터 전략**:
    - confidence: Config 기본값 + 요청에서 선택적 override
    - debug: Config만 사용 (DEBUG_MODE 환경변수, 서버 전역)

    Returns:
        DetectionResponse: 탐지된 객체 목록 및 메타데이터
    """

    start_time = time.time()
    image_id = str(uuid.uuid4())

    # confidence 기본값 처리
    conf_threshold = confidence if confidence is not None else settings.CONFIDENCE_THRESHOLD

    try:
        # 이미지 로드 및 검증
        image, (width, height) = await ImageProcessor.load_image_from_upload(file)
        logger.info(
            f"Processing image: {image_id}",
            extra={
                "image_id": image_id,
                "uploaded_file": file.filename,
                "size": f"{width}x{height}",
                "confidence": conf_threshold
            }
        )

        # YOLO 추론
        model_manager = YOLOModelManager()
        detections = model_manager.predict(image, confidence=conf_threshold)
        logger.info(f"Detection complete: {len(detections)} objects found")

        # 디버그 모드: Config의 DEBUG_MODE에 따라 이미지 저장
        debug_path = None
        if settings.DEBUG_MODE and detections:
            output_dir = Path(settings.OUTPUT_DIR)
            output_dir.mkdir(exist_ok=True)
            debug_path = f"{settings.OUTPUT_DIR}/debug_{image_id[:8]}.jpg"
            ImageProcessor.draw_detections(image, detections, debug_path)

        # 처리 시간 계산
        processing_time = round(time.time() - start_time, 3)

        # 응답 생성
        response = DetectionResponse(
            success=True,
            image_id=image_id,
            detections=detections,
            count=len(detections),
            processing_time=processing_time,
            image_size={"width": width, "height": height},
            debug_image_path=debug_path
        )

        logger.info(
            f"Request completed successfully",
            extra={
                "image_id": image_id,
                "detections": len(detections),
                "processing_time": processing_time
            }
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(
            f"Detection error: {str(e)}",
            exc_info=True,
            extra={"image_id": image_id}
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error during detection"
        )
