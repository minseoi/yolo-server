# YOLO Detection API - 시스템 설계 문서

## 개요

학습 및 실제 배포를 위해 설계된 프로덕션 준비가 완료된 FastAPI 기반 YOLO 객체 탐지 서비스입니다. 이 시스템은 선택적 데이터베이스 로깅, 포괄적인 디버깅 기능, Docker 기반 배포 전략을 갖춘 동기식 이미지 객체 탐지를 제공합니다.

---

## 1. 시스템 아키텍처

### 1.1 고수준 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     클라이언트 레이어                         │
│  (웹 브라우저 / 모바일 앱 / API 클라이언트 / CLI 도구)       │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/HTTPS
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   API 게이트웨이 레이어                       │
│              (Nginx 리버스 프록시 - 선택사항)                 │
│              - SSL 종단 처리                                 │
│              - 속도 제한                                     │
│              - 요청 로깅                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  FastAPI 애플리케이션                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            API 라우트 레이어                          │   │
│  │  • /health          - 헬스 체크 엔드포인트           │   │
│  │  • /api/v1/detect   - 객체 탐지 엔드포인트          │   │
│  │  • /docs            - ReDoc 문서                    │   │
│  └──────────────┬───────────────────────────────────────┘   │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────────┐   │
│  │         비즈니스 로직 레이어                          │   │
│  │  • 요청 검증 (Pydantic)                             │   │
│  │  • 이미지 처리 파이프라인                            │   │
│  │  • YOLO 모델 추론                                   │   │
│  │  • 응답 직렬화                                      │   │
│  └──────────────┬───────────────────────────────────────┘   │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────────┐   │
│  │            핵심 서비스                                │   │
│  │  • YOLO 모델 매니저 (싱글톤)                        │   │
│  │  • 이미지 프로세서 (OpenCV)                         │   │
│  │  • 로거 서비스                                      │   │
│  │  • 설정 매니저                                      │   │
│  └──────────────┬───────────────────────────────────────┘   │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────────┐   │
│  │         데이터 액세스 레이어 (선택사항)               │   │
│  │  • SQLAlchemy ORM                                   │   │
│  │  • 비동기 DB 세션                                   │   │
│  │  • 리포지토리 패턴                                   │   │
│  └──────────────┬───────────────────────────────────────┘   │
└─────────────────┼────────────────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────────────────┐
│                  외부 리소스                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   MySQL DB   │  │ 파일 시스템   │  │ YOLO 모델    │       │
│  │   (선택사항)  │  │  (uploads/   │  │   (.pt 파일) │       │
│  │              │  │   outputs/)  │  │              │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 컴포넌트 상호작용 흐름

```
┌──────────┐
│ 클라이언트 │
└─────┬────┘
      │ 1. POST /api/v1/detect (이미지 파일)
      │
┌─────▼──────────────────────────────────────────────────┐
│  FastAPI 라우터 (detection.py)                         │
│  • multipart/form-data 수락                           │
│  • file, confidence, debug 파라미터 추출              │
└─────┬──────────────────────────────────────────────────┘
      │ 2. 요청 검증
      │
┌─────▼──────────────────────────────────────────────────┐
│  Pydantic 스키마 검증                                  │
│  • 파일 타입 체크 (jpg, png)                          │
│  • confidence 범위 검증 (0.0-1.0)                     │
│  • 입력값 정제                                        │
└─────┬──────────────────────────────────────────────────┘
      │ 3. 이미지 로드
      │
┌─────▼──────────────────────────────────────────────────┐
│  이미지 프로세서 (utils/image_processing.py)           │
│  • 파일 바이트 읽기 → NumPy 배열                       │
│  • OpenCV 디코드                                      │
│  • 이미지 크기 검증                                    │
└─────┬──────────────────────────────────────────────────┘
      │ 4. 추론 실행
      │
┌─────▼──────────────────────────────────────────────────┐
│  YOLO 모델 매니저 (core/yolo_model.py)                 │
│  • 싱글톤 인스턴스 (모델 미리 로드됨)                   │
│  • confidence threshold로 예측 실행                    │
│  • bbox, 클래스, 점수 추출                            │
└─────┬──────────────────────────────────────────────────┘
      │ 5. 결과 후처리
      │
┌─────▼──────────────────────────────────────────────────┐
│  탐지 결과 처리                                        │
│  • DetectionResponse로 결과 포맷                       │
│  • 선택적으로 bbox 그리기 (debug=true인 경우)          │
│  • outputs/에 디버그 이미지 저장                       │
└─────┬──────────────────────────────────────────────────┘
      │ 6. DB에 로그 기록 (선택사항)
      │
┌─────▼──────────────────────────────────────────────────┐
│  데이터베이스 로거 (선택사항)                           │
│  • detection_logs 테이블에 비동기 저장                 │
│  • 기록: timestamp, image_id, 결과                    │
└─────┬──────────────────────────────────────────────────┘
      │ 7. JSON 응답 반환
      │
┌─────▼──────────────────────────────────────────────────┐
│  FastAPI 응답                                          │
│  {                                                     │
│    "success": true,                                    │
│    "detections": [...],                                │
│    "processing_time": 0.123                            │
│  }                                                     │
└────────────────────────────────────────────────────────┘
```

---

## 2. API 설계

### 2.1 API 명세

#### 헬스 체크 엔드포인트
```
GET /health
```

**설명**: API 서버 및 모델 상태 확인

**응답** (200 OK):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "yolov8n",
  "version": "1.0.0",
  "timestamp": "2025-11-11T10:30:00Z"
}
```

**에러 응답** (503 Service Unavailable):
```json
{
  "status": "unhealthy",
  "model_loaded": false,
  "error": "Model file not found"
}
```

---

#### 객체 탐지 엔드포인트
```
POST /api/v1/detect
Content-Type: multipart/form-data
```

**파라미터**:
| 파라미터     | 타입     | 필수 여부 | 기본값  | 설명                                    |
|-------------|----------|----------|---------|----------------------------------------|
| `file`      | File     | 예       | -       | 이미지 파일 (jpg, jpeg, png, bmp)      |
| `confidence`| float    | 아니오    | Config 기본값 | 탐지 신뢰도 임계값 (0.0-1.0), Config 기본값 override 가능 |

**설정 기반 동작**:
- `debug`: 환경변수 `DEBUG_MODE`로 설정 (서버 시작 시 설정, 요청마다 변경 불가)
- DEBUG_MODE=true일 경우 모든 탐지 결과에 대해 bbox 이미지가 outputs/에 저장됨

**성공 응답** (200 OK):
```json
{
  "success": true,
  "image_id": "uuid-v4-string",
  "detections": [
    {
      "class": "person",
      "class_id": 0,
      "confidence": 0.95,
      "bbox": {
        "x1": 100,
        "y1": 150,
        "x2": 300,
        "y2": 450
      }
    },
    {
      "class": "dog",
      "class_id": 16,
      "confidence": 0.87,
      "bbox": {
        "x1": 350,
        "y1": 200,
        "x2": 500,
        "y2": 400
      }
    }
  ],
  "count": 2,
  "processing_time": 0.123,
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "debug_image_path": "/outputs/debug_20251111_103045.jpg"
}
```

**에러 응답**:

*400 Bad Request* - 잘못된 입력:
```json
{
  "success": false,
  "error": "Invalid file type. Supported: jpg, jpeg, png, bmp",
  "details": {
    "file_type": "gif"
  }
}
```

*413 Payload Too Large* - 파일 크기 초과:
```json
{
  "success": false,
  "error": "File size exceeds 10MB limit",
  "details": {
    "file_size": 15728640,
    "max_size": 10485760
  }
}
```

*500 Internal Server Error* - 처리 실패:
```json
{
  "success": false,
  "error": "YOLO inference failed",
  "details": {
    "message": "CUDA out of memory"
  }
}
```

---

### 2.2 요청/응답 흐름

```
클라이언트 요청 → FastAPI 라우터 → 검증 → 처리 → 응답

1. 클라이언트 전송:
   POST /api/v1/detect
   Content-Type: multipart/form-data

   file=@image.jpg
   confidence=0.7  (선택사항, Config 기본값 override)

2. FASTAPI 수신:
   - multipart form 파싱
   - UploadFile 객체 추출
   - confidence 파라미터 추출 (없으면 Config 기본값)

3. 검증:
   - 파일 타입 체크
   - 파일 크기 체크 (< 10MB)
   - Confidence 범위 (0.0-1.0)
   - 이미지 디코드 체크

4. 처리:
   - 이미지 로드 → NumPy 배열
   - YOLO 추론 실행 (confidence 값 사용)
   - 탐지 결과 추출
   - 결과 포맷팅
   - Config DEBUG_MODE=true면 디버그 이미지 저장

5. 응답:
   - JSON 직렬화
   - HTTP 200 + DetectionResponse
```

---

## 3. 데이터 모델

### 3.1 Pydantic 스키마

#### 탐지 요청 스키마
```python
# app/schemas/detection.py

from pydantic import BaseModel, Field, field_validator
from typing import Optional

class DetectionRequest(BaseModel):
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold"
    )
    debug: bool = Field(
        default=False,
        description="Save annotated image for debugging"
    )

    @field_validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('confidence must be between 0.0 and 1.0')
        return v
```

#### BBox 스키마
```python
from pydantic import BaseModel

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    class Config:
        json_schema_extra = {
            "example": {
                "x1": 100,
                "y1": 150,
                "x2": 300,
                "y2": 450
            }
        }
```

#### 탐지 객체 스키마
```python
from pydantic import BaseModel

class DetectionObject(BaseModel):
    class_name: str = Field(..., alias="class")
    class_id: int
    confidence: float
    bbox: BBox

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "class": "person",
                "class_id": 0,
                "confidence": 0.95,
                "bbox": {
                    "x1": 100,
                    "y1": 150,
                    "x2": 300,
                    "y2": 450
                }
            }
        }
```

#### 탐지 응답 스키마
```python
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ImageSize(BaseModel):
    width: int
    height: int

class DetectionResponse(BaseModel):
    success: bool = True
    image_id: str
    detections: List[DetectionObject]
    count: int
    processing_time: float
    image_size: ImageSize
    debug_image_path: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "image_id": "550e8400-e29b-41d4-a716-446655440000",
                "detections": [
                    {
                        "class": "person",
                        "class_id": 0,
                        "confidence": 0.95,
                        "bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 450}
                    }
                ],
                "count": 1,
                "processing_time": 0.123,
                "image_size": {"width": 1920, "height": 1080},
                "timestamp": "2025-11-11T10:30:00Z"
            }
        }
```

---

### 3.2 SQLAlchemy 모델 (선택사항)

#### 탐지 로그 모델
```python
# app/models/detection_log.py

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(String(36), unique=True, index=True)
    image_filename = Column(String(255))
    image_size_width = Column(Integer)
    image_size_height = Column(Integer)
    confidence_threshold = Column(Float)
    detection_count = Column(Integer)
    detections_json = Column(JSON)  # 전체 탐지 결과 저장
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<DetectionLog(id={self.id}, image_id={self.image_id}, count={self.detection_count})>"
```

#### 데이터베이스 스키마
```sql
CREATE TABLE detection_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_id VARCHAR(36) UNIQUE NOT NULL,
    image_filename VARCHAR(255),
    image_size_width INT,
    image_size_height INT,
    confidence_threshold FLOAT,
    detection_count INT,
    detections_json JSON,
    processing_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_image_id (image_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

---

## 4. 컴포넌트 설계

### 4.1 핵심 컴포넌트

#### YOLO 모델 매니저 (싱글톤 패턴)
```python
# app/core/yolo_model.py

from ultralytics import YOLO
from typing import Optional, List, Dict
import numpy as np
from pathlib import Path

class YOLOModelManager:
    """효율적인 추론을 위한 싱글톤 YOLO 모델 매니저"""

    _instance: Optional['YOLOModelManager'] = None
    _model: Optional[YOLO] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_path: str) -> None:
        """시작 시 YOLO 모델을 한 번만 로드"""
        if self._model is None:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self._model = YOLO(model_path)

    def predict(
        self,
        image: np.ndarray,
        confidence: float = 0.5
    ) -> List[Dict]:
        """이미지에 대한 추론 실행"""
        if self._model is None:
            raise RuntimeError("Model not loaded")

        results = self._model.predict(
            source=image,
            conf=confidence,
            verbose=False
        )

        return self._format_results(results[0])

    def _format_results(self, result) -> List[Dict]:
        """YOLO 결과를 표준 포맷으로 변환"""
        detections = []

        for box in result.boxes:
            detection = {
                "class_name": result.names[int(box.cls)],
                "class_id": int(box.cls),
                "confidence": float(box.conf),
                "bbox": {
                    "x1": int(box.xyxy[0][0]),
                    "y1": int(box.xyxy[0][1]),
                    "x2": int(box.xyxy[0][2]),
                    "y2": int(box.xyxy[0][3])
                }
            }
            detections.append(detection)

        return detections

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model_name(self) -> str:
        if self._model:
            return self._model.model_name
        return "unknown"
```

**설계 결정사항**:
- **싱글톤 패턴**: 여러 번 모델 로드 방지 (무거운 작업)
- **지연 로딩**: 요청마다가 아닌 시작 시 한 번만 모델 로드
- **스레드 안전**: Python GIL이 싱글톤 안전성 보장
- **에러 처리**: 누락된 모델에 대한 명확한 예외

---

#### 이미지 프로세서
```python
# app/utils/image_processing.py

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from fastapi import UploadFile

class ImageProcessor:
    """YOLO 파이프라인을 위한 이미지 처리 유틸리티"""

    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    @staticmethod
    async def load_image_from_upload(
        file: UploadFile
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """FastAPI UploadFile에서 이미지 로드"""

        # 파일 타입 검증
        if not ImageProcessor._is_valid_format(file.filename):
            raise ValueError(
                f"Unsupported format. Allowed: {ImageProcessor.SUPPORTED_FORMATS}"
            )

        # 파일 바이트 읽기
        contents = await file.read()

        # 파일 크기 검증
        if len(contents) > ImageProcessor.MAX_FILE_SIZE:
            raise ValueError("File size exceeds 10MB limit")

        # 이미지 디코드
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        height, width = image.shape[:2]
        return image, (width, height)

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: List[Dict],
        output_path: str
    ) -> None:
        """디버깅을 위해 이미지에 바운딩 박스 그리기"""

        annotated = image.copy()

        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

            # 사각형 그리기
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # 라벨 그리기
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        cv2.imwrite(output_path, annotated)

    @staticmethod
    def _is_valid_format(filename: Optional[str]) -> bool:
        if not filename:
            return False
        return any(filename.lower().endswith(fmt) for fmt in ImageProcessor.SUPPORTED_FORMATS)
```

**설계 결정사항**:
- **비동기 I/O**: 논블로킹 파일 읽기를 위해 `await file.read()` 사용
- **검증**: 처리 전 파일 타입 및 크기 체크
- **OpenCV 통합**: YOLO 호환성을 위한 NumPy 배열 포맷
- **디버그 지원**: 개발을 위한 선택적 bbox 렌더링

---

### 4.2 API 라우트

#### 탐지 라우트 핸들러
```python
# app/api/routes/detection.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.schemas.detection import DetectionResponse, DetectionRequest
from app.core.yolo_model import YOLOModelManager
from app.utils.image_processing import ImageProcessor
from app.utils.logger import get_logger
import time
import uuid
from pathlib import Path

router = APIRouter(prefix="/api/v1", tags=["detection"])
logger = get_logger(__name__)

@router.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = Query(default=None, ge=0.0, le=1.0)
):
    """
    객체 탐지 엔드포인트

    Args:
        file: 이미지 파일 (jpg, png, bmp)
        confidence: 탐지 신뢰도 임계값 (0.0-1.0), None이면 Config 기본값 사용

    Returns:
        탐지된 객체가 포함된 DetectionResponse

    Notes:
        - DEBUG_MODE는 환경변수로 설정 (서버 전역)
        - confidence는 요청마다 override 가능
    """

    from app.core.config import settings

    start_time = time.time()
    image_id = str(uuid.uuid4())

    # confidence 기본값 처리 (None이면 Config 기본값 사용)
    conf_threshold = confidence if confidence is not None else settings.CONFIDENCE_THRESHOLD

    try:
        # 이미지 로드 및 검증
        image, (width, height) = await ImageProcessor.load_image_from_upload(file)
        logger.info(f"Image loaded: {image_id}, size={width}x{height}, confidence={conf_threshold}")

        # YOLO 추론 실행
        model_manager = YOLOModelManager()
        detections = model_manager.predict(image, confidence=conf_threshold)
        logger.info(f"Detection complete: {len(detections)} objects found")

        # 디버그 모드: Config의 DEBUG_MODE에 따라 이미지 저장
        debug_path = None
        if settings.DEBUG_MODE and detections:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            debug_path = f"outputs/debug_{image_id[:8]}.jpg"
            ImageProcessor.draw_detections(image, detections, debug_path)
            logger.info(f"Debug image saved: {debug_path}")

        # 처리 시간 계산
        processing_time = round(time.time() - start_time, 3)

        # 응답 빌드
        response = DetectionResponse(
            success=True,
            image_id=image_id,
            detections=detections,
            count=len(detections),
            processing_time=processing_time,
            image_size={"width": width, "height": height},
            debug_image_path=debug_path
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
```

**설계 결정사항**:
- **하이브리드 파라미터 전략**:
  - `confidence`: Config 기본값 + 요청에서 선택적 override (이미지마다 다른 임계값 테스트 가능)
  - `debug`: Config만 사용 (전역 개발/프로덕션 모드)
- **비동기 핸들러**: 더 나은 동시성을 위한 논블로킹 파일 I/O
- **UUID 생성**: 추적 및 디버깅을 위한 고유 image_id
- **에러 처리**: 검증(400)과 처리(500) 에러 분리
- **로깅**: 디버깅 및 모니터링을 위한 구조화된 로깅
- **성능 추적**: 종단간 처리 시간 측정

---

## 5. 개발 워크플로우 설계

### 5.1 Dev Container 구성

#### .devcontainer/devcontainer.json
```json
{
  "name": "YOLO Detection API",
  "dockerComposeFile": "../docker/docker-compose.dev.yml",
  "service": "app",
  "workspaceFolder": "/workspace",

  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.black-formatter",
        "ms-python.isort",
        "charliermarsh.ruff",
        "ms-azuretools.vscode-docker"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.linting.enabled": true,
        "python.linting.pylintEnabled": false,
        "python.formatting.provider": "black",
        "python.testing.pytestEnabled": true,
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
          "source.organizeImports": true
        }
      }
    }
  },

  "forwardPorts": [8000],
  "postCreateCommand": "pip install -r requirements-dev.txt",
  "remoteUser": "vscode"
}
```

#### docker/docker-compose.dev.yml
```yaml
version: '3.8'

services:
  app:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dev
    volumes:
      - ..:/workspace:cached
      - venv:/workspace/.venv
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - MODEL_PATH=/workspace/models/yolov8n.pt
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    depends_on:
      - db

  db:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: yolo_db
      MYSQL_USER: yolo_user
      MYSQL_PASSWORD: yolo_pass
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql

volumes:
  venv:
  mysql_data:
```

### 5.2 Dockerfile 전략

#### 멀티 스테이지 프로덕션 Dockerfile
```dockerfile
# docker/Dockerfile

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# requirements 복사
COPY requirements.txt .

# wheels 빌드
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# 런타임 의존성만 설치
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# builder에서 wheels 복사
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt .

# wheels로부터 설치
RUN pip install --no-cache /wheels/*

# 애플리케이션 코드 복사
COPY ./app /app/app

# 디렉토리 생성
RUN mkdir -p /app/models /app/uploads /app/outputs /app/logs

# 비루트 사용자
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# 헬스 체크
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**설계 결정사항**:
- **멀티 스테이지 빌드**: 이미지 크기를 40-50% 감소
- **Slim 베이스 이미지**: 최소 풋프린트를 위한 python:3.11-slim
- **비루트 사용자**: 보안 모범 사례
- **헬스 체크**: 컨테이너 오케스트레이션 지원
- **레이어 캐싱**: 재빌드 시간 최적화

---

## 6. 배포 아키텍처

### 6.1 로컬 개발
```
개발자 머신
├── VSCode
│   └── Dev Container 확장
│       └── Docker Desktop
│           ├── FastAPI 컨테이너 (port 8000)
│           └── MySQL 컨테이너 (port 3306)
└── 볼륨 마운트
    ├── 소스 코드 (라이브 리로드)
    └── 모델 디렉토리
```

### 6.2 프로덕션 배포 (Docker + Nginx)
```
                    인터넷
                       │
                       ▼
               ┌───────────────┐
               │  Nginx Proxy  │
               │  (Port 80/443)│
               └───────┬───────┘
                       │
               ┌───────▼────────┐
               │   FastAPI      │
               │  (Port 8000)   │
               └───────┬────────┘
                       │
           ┌───────────┴──────────┐
           │                      │
    ┌──────▼──────┐      ┌───────▼──────┐
    │   MySQL DB  │      │ 파일 시스템   │
    │ (Port 3306) │      │  (Volumes)   │
    └─────────────┘      └──────────────┘
```

#### Nginx 구성
```nginx
# /etc/nginx/sites-available/yolo-api

upstream fastapi_backend {
    server localhost:8000;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    client_max_body_size 10M;

    location / {
        proxy_pass http://fastapi_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 긴 처리를 위한 타임아웃
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /docs {
        proxy_pass http://fastapi_backend/docs;
    }

    location /health {
        proxy_pass http://fastapi_backend/health;
        access_log off;
    }
}
```

---

## 7. 에러 처리 전략

### 7.1 에러 카테고리

| 카테고리 | HTTP 코드 | 예시 | 처리 방법 |
|----------|-----------|----------|----------|
| 검증 | 400 | 잘못된 파일 타입, 잘못된 confidence | 클라이언트가 수정 후 재시도 |
| 인증 | 401 | 잘못된 API 키 | 클라이언트 인증 |
| 권한 | 403 | 속도 제한 초과 | 클라이언트 백오프 |
| 미발견 | 404 | 엔드포인트를 찾을 수 없음 | 클라이언트가 URL 수정 |
| 페이로드 | 413 | 파일이 너무 큼 | 클라이언트가 크기 축소 |
| 서버 에러 | 500 | YOLO 크래시, DB 에러 | 서버 조사 |
| 서비스 불가 | 503 | 모델 미로드 | 서버 재시작 |

### 7.2 에러 응답 포맷
```python
from pydantic import BaseModel
from typing import Optional, Dict

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[Dict] = None
    request_id: Optional[str] = None
```

### 7.3 전역 예외 핸들러
```python
# app/main.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import traceback

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
```

---

## 8. 테스트 전략

### 8.1 테스트 피라미드

```
                    ▲
                   / \
                  /   \
                 / E2E \
                /───────\
               /         \
              / 통합 테스트 \
             /─────────────\
            /               \
           /   단위 테스트    \
          /___________________\
```

### 8.2 테스트 커버리지 목표

| 레이어 | 커버리지 목표 | 도구 |
|-------|--------------|-------|
| 단위 테스트 | >80% | pytest, pytest-cov |
| 통합 테스트 | >60% | pytest, httpx |
| E2E 테스트 | 핵심 경로 | pytest, 실제 이미지 |

### 8.3 테스트 구조

```python
# tests/test_detection.py

import pytest
from fastapi.testclient import TestClient
from app.main import app
from pathlib import Path

client = TestClient(app)

@pytest.fixture
def sample_image():
    """테스트 이미지 로드"""
    return Path("tests/fixtures/sample.jpg")

def test_health_check():
    """헬스 엔드포인트 테스트"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_detect_valid_image(sample_image):
    """유효한 이미지로 탐지 테스트"""
    with open(sample_image, "rb") as f:
        response = client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", f, "image/jpeg")},
            params={"confidence": 0.5}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "detections" in data
    assert "processing_time" in data

def test_detect_invalid_file_type():
    """잘못된 파일 타입 거부 테스트"""
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )

    assert response.status_code == 400
    assert "Unsupported format" in response.json()["error"]

def test_detect_with_debug_mode(sample_image):
    """디버그 모드로 출력 이미지 생성 테스트"""
    with open(sample_image, "rb") as f:
        response = client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", f, "image/jpeg")},
            params={"debug": True}
        )

    data = response.json()
    if data["count"] > 0:
        assert data["debug_image_path"] is not None
        assert Path(data["debug_image_path"]).exists()
```

---

## 9. 모니터링 및 관찰성

### 9.1 로깅 전략

#### 구조화된 로깅
```python
# app/utils/logger.py

import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.FileHandler("logs/app.log")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
```

### 9.2 추적할 메트릭

| 메트릭 | 설명 | 알림 임계값 |
|--------|-------------|-----------------|
| 요청 비율 | 분당 요청 수 | > 100/분 |
| 평균 응답 시간 | 탐지 엔드포인트 지연시간 | > 2초 |
| 에러 비율 | 5xx 에러 비율 | > 5% |
| 모델 로드 시간 | 시작 시 모델 로딩 | > 30초 |
| 메모리 사용량 | 컨테이너 메모리 | > 80% |
| 디스크 사용량 | uploads/ 및 outputs/ | > 90% |

### 9.3 헬스 체크 구현
```python
# app/api/routes/health.py

from fastapi import APIRouter
from app.core.yolo_model import YOLOModelManager
from app.core.config import settings
from datetime import datetime

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    model_manager = YOLOModelManager()

    return {
        "status": "healthy" if model_manager.is_loaded() else "unhealthy",
        "model_loaded": model_manager.is_loaded(),
        "model_name": model_manager.get_model_name(),
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }
```

---

## 10. 보안 고려사항

### 10.1 보안 체크리스트

- [ ] **입력 검증**: 파일 타입, 크기 및 내용 검증
- [ ] **속도 제한**: 남용 방지 (Nginx 레벨 또는 FastAPI 미들웨어)
- [ ] **파일 크기 제한**: 이미지당 최대 10MB
- [ ] **경로 순회**: 파일명 정제, UUID 사용
- [ ] **CORS**: 허용된 출처 구성
- [ ] **HTTPS**: 프로덕션에서 SSL 인증서 사용
- [ ] **시크릿 관리**: 하드코딩이 아닌 환경 변수 사용
- [ ] **비루트 컨테이너**: 권한 없는 사용자로 실행
- [ ] **의존성 스캔**: 정기적인 취약점 체크
- [ ] **API 키**: 선택적 인증 미들웨어

### 10.2 속도 제한 미들웨어
```python
# app/middleware/rate_limit.py

from fastapi import Request, HTTPException
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests: int = 100, window: int = 60):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window)
        self.requests = defaultdict(list)

    async def __call__(self, request: Request, call_next):
        client_ip = request.client.host
        now = datetime.utcnow()

        # 오래된 요청 정리
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < self.window
        ]

        # 속도 제한 확인
        if len(self.requests[client_ip]) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        self.requests[client_ip].append(now)
        response = await call_next(request)
        return response
```

---

## 11. 성능 최적화

### 11.1 최적화 전략

| 영역 | 전략 | 예상 효과 |
|------|----------|---------------|
| 모델 로딩 | 싱글톤 패턴, 시작 시 사전 로드 | 요청당 지연시간 -90% |
| 이미지 처리 | opencv-python-headless 사용 | 메모리 -30% |
| 응답 | 파일 작업에 비동기 I/O | 처리량 +50% |
| 데이터베이스 | 커넥션 풀링, 비동기 쿼리 | DB 지연시간 -40% |
| 캐싱 | 모델 예측 캐싱 (선택사항) | 중복에 대해 -95% |

### 11.2 성능 목표

| 메트릭 | 목표 | 측정 방법 |
|--------|--------|-------------|
| 콜드 스타트 | < 30초 | 모델 로드 시간 |
| 추론 시간 | < 500ms | YOLO 예측 |
| 종단간 | < 1초 | API 응답 시간 |
| 처리량 | > 10 req/sec | 단일 인스턴스 |
| 메모리 | < 2GB | 컨테이너 제한 |

---

## 12. 구성 관리

### 12.1 환경 변수
```python
# app/core/config.py

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # 앱
    APP_NAME: str = "YOLO Detection API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # 모델
    MODEL_PATH: str = "models/yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5  # API 요청에서 override 가능

    # 디버그 모드 (서버 전역 설정, 요청마다 변경 불가)
    DEBUG_MODE: bool = False  # true면 모든 탐지 결과에 대해 bbox 이미지 저장

    # 경로
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    LOG_FILE: str = "logs/app.log"

    # 데이터베이스 (선택사항)
    DATABASE_URL: Optional[str] = None

    # 로깅
    LOG_LEVEL: str = "INFO"

    # 보안
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".bmp"]

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

---

## 13. 다음 구현 단계

### Phase 1: 기초 (1주차)
1. ✅ 프로젝트 구조 생성
2. ✅ Dev Container 구성 설정
3. ✅ 핵심 config 관리 구현
4. ✅ 로깅 시스템 설정
5. ✅ 헬스 체크가 포함된 기본 FastAPI 앱 생성

### Phase 2: 핵심 기능 (2주차)
1. ✅ YOLOModelManager 싱글톤 구현
2. ✅ ImageProcessor 유틸리티 생성
3. ✅ 탐지 엔드포인트 구현
4. ✅ Pydantic 스키마 및 검증 추가
5. ✅ 샘플 이미지로 테스트

### Phase 3: 개선 (3주차)
1. ✅ 데이터베이스 통합 추가 (선택사항)
2. ✅ bbox 렌더링이 포함된 디버그 모드 구현
3. ✅ 포괄적인 에러 처리 추가
4. ✅ 단위 및 통합 테스트 작성
5. ✅ API 문서 생성

### Phase 4: 배포 (4주차)
1. ✅ 프로덕션 Dockerfile 생성
2. ✅ 멀티 서비스용 Docker Compose 설정
3. ✅ Nginx 리버스 프록시 구성
4. ✅ 모니터링 및 로깅 추가
5. ✅ 서버에 배포 및 테스트

---

## 14. 성공 메트릭

| 메트릭 | 목표 | 측정 방법 |
|--------|--------|--------------------|
| API 가용성 | >99% | 업타임 모니터링 |
| 응답 시간 | <1s (p95) | 애플리케이션 로그 |
| 에러 비율 | <1% | HTTP 상태 추적 |
| 테스트 커버리지 | >80% | pytest-cov 리포트 |
| 문서화 | 100% 엔드포인트 | ReDoc 완성도 |
| 코드 품질 | A 등급 | 정적 분석 |

---

## 부록 A: 기술 결정 근거

### FastAPI vs Flask
- ✅ FastAPI: 비동기, 자동 문서화, 타입 안전성, 더 나은 성능
- ❌ Flask: 동기만 가능, 수동 검증, 네이티브 비동기 없음

### YOLOv8 vs YOLOv5
- ✅ YOLOv8: 최신, 더 나은 정확도, 공식 ultralytics
- ❌ YOLOv5: 구버전, 커뮤니티 포크

### MySQL vs PostgreSQL
- ✅ MySQL: 더 간단한 설정, 좋은 JSON 지원, 널리 채택됨
- ❌ PostgreSQL: 더 많은 기능이지만 이 사용 사례에는 과함

### Docker vs Native
- ✅ Docker: 환경 일관성, 쉬운 배포
- ❌ Native: 플랫폼 종속적 설정 문제

### 파라미터 설계: Confidence (하이브리드) vs Debug (Config만)

**Confidence - 하이브리드 접근 (Config 기본값 + API Override)**

선택 이유:
- ✅ 이미지마다 다른 임계값으로 테스트 필요성 (예: 저조도 이미지 vs 밝은 이미지)
- ✅ 실험과 튜닝을 위한 유연성 (개발/테스트 단계에서 중요)
- ✅ 단일 클라이언트이지만 다양한 시나리오 테스트 가능
- ⚠️ Config 기본값으로 일관성 유지, 필요시만 override

구현 방식:
```python
# .env
CONFIDENCE_THRESHOLD=0.5  # 기본값

# API 요청
POST /api/v1/detect  # 0.5 사용 (기본값)
POST /api/v1/detect?confidence=0.7  # 0.7 사용 (override)
```

**Debug - Config만 접근 (환경변수로만 제어)**

선택 이유:
- ✅ 전역 개발/프로덕션 모드 설정 (서버 전체 동작 방식)
- ✅ 요청마다 바꿀 필요 없음 (디버깅은 세션 레벨)
- ✅ 단순한 on/off 스위치 (복잡한 제어 불필요)
- ✅ 실수로 프로덕션에서 디버그 모드 켜는 것 방지
- ✅ 학습용 프로젝트에 적합한 단순성

구현 방식:
```python
# .env
DEBUG_MODE=false  # 프로덕션
DEBUG_MODE=true   # 개발

# 서버 재시작 시 적용됨
# API 요청에서는 제어 불가
```

대안 고려 및 거부 이유:
- ❌ Debug도 API 파라미터로: 너무 복잡, 학습용 프로젝트에 과함
- ❌ Confidence도 Config만: 유연성 부족, 실험 어려움
- ❌ 둘 다 API 파라미터: API가 복잡해지고 설정 관리 어려움
- ❌ 둘 다 Config만: Confidence 테스트를 위해 매번 서버 재시작 필요

---

## 부록 B: 트러블슈팅 가이드

### 일반적인 문제

**문제**: 모델 로드 실패
```
해결책:
1. .env의 MODEL_PATH 확인
2. .pt 파일 존재 확인
3. 파일 권한 확인
4. ultralytics 버전 호환성 확인
```

**문제**: 메모리 부족 에러
```
해결책:
1. 더 작은 YOLO 모델 사용 (yolov8n.pt)
2. Docker 메모리 제한 증가
3. 배치 크기를 1로 축소
4. uploads/ 및 outputs/ 정기적으로 정리
```

**문제**: 느린 추론
```
해결책:
1. CPU vs GPU 확인 (필요시 CUDA 지원 추가)
2. 추론 전 이미지 해상도 축소
3. confidence threshold를 낮춰 후처리 감소
4. 더 작은 YOLO 모델 변형 사용
```

---

**문서 버전**: 1.0
**최종 업데이트**: 2025-11-11
**작성자**: 시스템 설계 팀
**상태**: 구현 준비 완료
