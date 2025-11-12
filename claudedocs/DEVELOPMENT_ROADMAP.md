# YOLO Detection API - Development Roadmap

**프로젝트**: FastAPI 기반 YOLO 이미지 객체 탐지 API 서버
**목적**: 학습 및 실제 배포를 위한 프로덕션 준비 완료 시스템
**로드맵 생성일**: 2025-11-11

---

## 📋 프로젝트 개요

### 핵심 목표
- FastAPI + YOLO(v8/v10) 기반 실시간 객체 탐지 API
- VSCode Dev Container 기반 개발 환경 (개발 = 배포 환경)
- Docker 기반 배포 전략 (로컬 + 실제 서버)
- 선택적 MySQL 데이터베이스 로깅
- ReDoc 자동 API 문서화

### 기술 스택 요약
- **Backend**: Python 3.11, FastAPI 0.104+, Uvicorn
- **YOLO**: Ultralytics (YOLOv8/v10), OpenCV-headless, NumPy
- **Database**: MySQL 8.0 + SQLAlchemy 2.0 (선택사항)
- **Dev Environment**: VSCode Dev Containers, Docker Compose
- **Testing**: Pytest, httpx, pytest-cov

---

## 🗓️ 개발 단계 (4주 계획)

---

## Phase 1: 프로젝트 기초 설정 (1주차)

### 목표
✅ 개발 환경 완전 자동화
✅ 기본 FastAPI 앱 + 헬스체크 구현
✅ 구조화된 로깅 시스템 설정
✅ 핵심 설정 관리 구현

### 세부 작업

#### 1.1 프로젝트 구조 생성
```bash
server_study/
├── .devcontainer/
│   └── devcontainer.json          # VSCode Dev Container 설정
├── .vscode/
│   └── launch.json                # 디버거 설정
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 엔트리포인트
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── detection.py       # YOLO 탐지 엔드포인트
│   │       └── health.py          # 헬스체크
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic Settings
│   │   └── yolo_model.py          # YOLO 싱글톤 매니저
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── detection.py           # 탐지 요청/응답 스키마
│   │   └── common.py
│   ├── models/                    # SQLAlchemy 모델 (선택)
│   │   ├── __init__.py
│   │   └── detection_log.py
│   ├── db/
│   │   ├── __init__.py
│   │   └── database.py
│   └── utils/
│       ├── __init__.py
│       ├── image_processing.py    # OpenCV 이미지 유틸
│       └── logger.py              # JSON 구조화 로깅
├── models/                        # YOLO .pt 파일 (.gitignore)
├── uploads/                       # 임시 업로드 (.gitignore)
├── outputs/                       # 디버그 bbox 이미지 (.gitignore)
├── tests/
│   ├── __init__.py
│   ├── test_detection.py
│   └── test_health.py
├── docker/
│   ├── Dockerfile                 # 프로덕션 멀티스테이지
│   ├── Dockerfile.dev             # 개발용
│   └── docker-compose.yml
├── claudedocs/                    # 프로젝트 문서
│   ├── tech-stack.md
│   └── system-design.md
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── README.md
```

**체크리스트**:
- [x] 모든 디렉토리 생성 (`mkdir -p`)
- [x] 각 Python 패키지에 `__init__.py` 생성
- [x] `.gitkeep` 파일로 빈 디렉토리 유지 (models/, uploads/, outputs/)

---

#### 1.2 Dev Container 설정

**파일**: `.devcontainer/devcontainer.json`
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
        "charliermarsh.ruff",
        "ms-azuretools.vscode-docker"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.testing.pytestEnabled": true,
        "editor.formatOnSave": true
      }
    }
  },

  "forwardPorts": [8000],
  "postCreateCommand": "pip install -r requirements-dev.txt",
  "remoteUser": "vscode"
}
```

**체크리스트**:
- [x] `devcontainer.json` 생성
- [ ] VSCode Dev Containers 확장 설치
- [ ] Docker Desktop 설치 및 실행 확인

---

#### 1.3 Docker 개발 환경 설정

**파일**: `docker/docker-compose.dev.yml`
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
      - DEBUG_MODE=true
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

**파일**: `docker/Dockerfile.dev`
```dockerfile
FROM python:3.11-slim

WORKDIR /workspace

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치는 postCreateCommand에서 처리
COPY requirements-dev.txt .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

**체크리스트**:
- [x] `docker-compose.dev.yml` 생성
- [x] `Dockerfile.dev` 생성
- [x] MySQL 서비스 선택사항 확인

---

#### 1.4 의존성 관리

**파일**: `requirements.txt`
```
# Core
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0

# YOLO & Vision
ultralytics==8.0.220
opencv-python-headless==4.8.1.78
numpy==1.24.3

# Database (optional)
sqlalchemy==2.0.23
asyncmy==0.2.9
aiomysql==0.2.0
```

**파일**: `requirements-dev.txt`
```
-r requirements.txt

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
httpx==0.25.2

# Debugging
debugpy==1.8.0

# Code Quality
black==23.12.1
ruff==0.1.9
mypy==1.8.0
```

**체크리스트**:
- [x] `requirements.txt` 생성
- [x] `requirements-dev.txt` 생성

---

#### 1.5 핵심 설정 관리 구현

**파일**: `app/core/config.py`
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # App
    APP_NAME: str = "YOLO Detection API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Model
    MODEL_PATH: str = "models/yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5  # API에서 override 가능

    # Debug Mode (서버 전역 설정)
    DEBUG_MODE: bool = False  # true면 모든 탐지에 bbox 이미지 저장

    # Paths
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    LOG_FILE: str = "logs/app.log"

    # Database (optional)
    DATABASE_URL: Optional[str] = None

    # Logging
    LOG_LEVEL: str = "INFO"

    # Security
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".bmp"]

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

**파일**: `.env.example`
```bash
APP_NAME=YOLO Detection API
APP_VERSION=1.0.0
DEBUG=false

# Model
MODEL_PATH=models/yolov8n.pt
CONFIDENCE_THRESHOLD=0.5

# Debug Mode (전역 설정, 서버 재시작 필요)
DEBUG_MODE=false

# Database (선택사항)
# DATABASE_URL=mysql+asyncmy://yolo_user:yolo_pass@db:3306/yolo_db

# Paths
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
LOG_FILE=logs/app.log

# Logging
LOG_LEVEL=INFO
```

**체크리스트**:
- [x] `app/core/config.py` 구현
- [x] `.env.example` 생성
- [x] `.env` 파일 생성 (`.gitignore`에 추가)

---

#### 1.6 구조화된 로깅 시스템

**파일**: `app/utils/logger.py`
```python
import logging
import json
from datetime import datetime
from pathlib import Path

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
        # 로그 디렉토리 생성
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # 파일 핸들러
        file_handler = logging.FileHandler("logs/app.log")
        file_handler.setFormatter(JSONFormatter())

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

    return logger
```

**체크리스트**:
- [x] `app/utils/logger.py` 구현
- [x] `logs/` 디렉토리 `.gitignore`에 추가
- [ ] JSON 로그 형식 테스트

---

#### 1.7 기본 FastAPI 앱 + 헬스체크

**파일**: `app/main.py`
```python
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
```

**파일**: `app/api/routes/health.py`
```python
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
```

**체크리스트**:
- [x] `app/main.py` 구현
- [x] `app/api/routes/__init__.py` 생성
- [x] `app/api/routes/health.py` 구현
- [ ] `/health` 엔드포인트 테스트

---

#### 1.8 VSCode 디버거 설정

**파일**: `.vscode/launch.json`
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
      ],
      "jinja": true,
      "justMyCode": false
    }
  ]
}
```

**체크리스트**:
- [x] `.vscode/launch.json` 생성
- [ ] F5로 디버거 실행 테스트
- [ ] 브레이크포인트 동작 확인

---

#### 1.9 Git 설정

**파일**: `.gitignore`
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
*.egg-info/

# IDE
.vscode/
.idea/

# Environment
.env

# Models
models/*.pt

# Data
uploads/
outputs/
logs/

# Database
*.db
*.sqlite

# Docker
.docker/

# OS
.DS_Store
Thumbs.db
```

**체크리스트**:
- [x] `.gitignore` 생성
- [ ] Git 초기화 (`git init`)
- [ ] 첫 커밋 생성

---

### Phase 1 검증 체크리스트

- [ ] VSCode에서 "Reopen in Container" 성공
- [ ] 컨테이너 내부에서 의존성 자동 설치 완료
- [ ] `http://localhost:8000` 접근 가능
- [ ] `http://localhost:8000/health` 응답 정상
- [ ] `http://localhost:8000/docs` ReDoc 문서 확인
- [ ] F5 디버거 실행 및 브레이크포인트 동작
- [ ] Hot reload 동작 확인 (파일 수정 → 자동 재시작)
- [ ] MySQL 컨테이너 정상 실행 (선택사항)
- [ ] 구조화된 로그 `logs/app.log`에 기록

**예상 소요 시간**: 3-4일

---

## Phase 2: YOLO 핵심 기능 구현 (2주차)

### 목표
✅ YOLO 모델 매니저 싱글톤 구현
✅ 이미지 처리 파이프라인 구축
✅ 객체 탐지 API 엔드포인트 완성
✅ Pydantic 스키마 검증 추가

### 세부 작업

#### 2.1 YOLO 모델 매니저 싱글톤 구현

**파일**: `app/core/yolo_model.py`
```python
from ultralytics import YOLO
from typing import Optional, List, Dict
import numpy as np
from pathlib import Path
from app.utils.logger import get_logger

logger = get_logger(__name__)

class YOLOModelManager:
    """싱글톤 YOLO 모델 매니저 - 시작 시 한 번만 로드"""

    _instance: Optional['YOLOModelManager'] = None
    _model: Optional[YOLO] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_path: str) -> None:
        """시작 시 YOLO 모델 로드"""
        if self._model is None:
            model_file = Path(model_path)
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            logger.info(f"Loading YOLO model: {model_path}")
            self._model = YOLO(model_path)
            logger.info(f"Model loaded successfully: {self.get_model_name()}")

    def predict(
        self,
        image: np.ndarray,
        confidence: float = 0.5
    ) -> List[Dict]:
        """이미지에 대한 YOLO 추론 실행"""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

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
            # ultralytics YOLO 객체는 model_name 속성이 없을 수 있음
            # 대신 파일명이나 task로 식별
            return str(self._model.model_name if hasattr(self._model, 'model_name') else "yolo")
        return "unknown"
```

**체크리스트**:
- [ ] `app/core/yolo_model.py` 구현
- [ ] `models/yolov8n.pt` 모델 파일 다운로드 (ultralytics에서 자동 다운로드 가능)
- [ ] 싱글톤 패턴 테스트
- [ ] 모델 로드 시간 측정

**모델 다운로드 방법**:
```python
# Python shell에서 자동 다운로드
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # 자동으로 ~/.cache/에 다운로드
```

---

#### 2.2 이미지 처리 유틸리티 구현

**파일**: `app/utils/image_processing.py`
```python
import cv2
import numpy as np
from typing import Tuple, List, Dict
from fastapi import UploadFile
from pathlib import Path
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ImageProcessor:
    """YOLO 파이프라인을 위한 이미지 처리"""

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

        # 파일 읽기
        contents = await file.read()

        # 파일 크기 검증
        if len(contents) > ImageProcessor.MAX_FILE_SIZE:
            raise ValueError(
                f"File size exceeds {ImageProcessor.MAX_FILE_SIZE / 1024 / 1024}MB limit"
            )

        # 이미지 디코드
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        height, width = image.shape[:2]
        logger.info(f"Image loaded: {width}x{height}")

        return image, (width, height)

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: List[Dict],
        output_path: str
    ) -> None:
        """디버깅을 위해 bbox 그리기"""

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

        # 디렉토리 생성
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(output_path, annotated)
        logger.info(f"Debug image saved: {output_path}")

    @staticmethod
    def _is_valid_format(filename: str) -> bool:
        if not filename:
            return False
        return any(filename.lower().endswith(fmt) for fmt in ImageProcessor.SUPPORTED_FORMATS)
```

**체크리스트**:
- [ ] `app/utils/image_processing.py` 구현
- [ ] 비동기 파일 읽기 테스트
- [ ] 이미지 검증 로직 테스트
- [ ] bbox 렌더링 테스트

---

#### 2.3 Pydantic 스키마 정의

**파일**: `app/schemas/detection.py`
```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "x1": 100,
                "y1": 150,
                "x2": 300,
                "y2": 450
            }
        }
    }

class DetectionObject(BaseModel):
    class_name: str = Field(..., alias="class")
    class_id: int
    confidence: float
    bbox: BBox

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
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
    }

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

    model_config = {
        "json_schema_extra": {
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
    }

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[dict] = None
    request_id: Optional[str] = None
```

**체크리스트**:
- [ ] `app/schemas/detection.py` 구현
- [ ] Pydantic v2 문법 확인
- [ ] 스키마 예제 검증

---

#### 2.4 객체 탐지 엔드포인트 구현

**파일**: `app/api/routes/detection.py`
```python
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
                "filename": file.filename,
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
```

**파일**: `app/main.py` 업데이트 (모델 로드 추가)
```python
from contextlib import asynccontextmanager
from app.core.yolo_model import YOLOModelManager
from app.api.routes import detection  # 추가

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # YOLO 모델 로드 (시작 시 한 번만)
    try:
        model_manager = YOLOModelManager()
        model_manager.load_model(settings.MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise

    yield

    logger.info(f"Shutting down {settings.APP_NAME}")

# 라우터 등록
app.include_router(health.router)
app.include_router(detection.router)  # 추가
```

**파일**: `app/api/routes/health.py` 업데이트
```python
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
```

**체크리스트**:
- [ ] `app/api/routes/detection.py` 구현
- [ ] `app/main.py` 모델 로드 추가
- [ ] `app/api/routes/health.py` 업데이트
- [ ] 라우터 등록 확인

---

#### 2.5 수동 테스트

**cURL 예제**:
```bash
# 1. 헬스체크
curl http://localhost:8000/health

# 2. 객체 탐지 (기본 confidence)
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@test_image.jpg"

# 3. 객체 탐지 (confidence override)
curl -X POST "http://localhost:8000/api/v1/detect?confidence=0.7" \
  -F "file=@test_image.jpg"

# 4. ReDoc 문서 확인
open http://localhost:8000/docs
```

**Postman/Insomnia 테스트**:
1. POST `http://localhost:8000/api/v1/detect`
2. Body → form-data
3. Key: `file`, Type: File, Value: (이미지 파일 선택)
4. Query Params: `confidence=0.6` (선택사항)

**체크리스트**:
- [ ] 샘플 이미지 준비 (person, car 등 포함)
- [ ] 기본 confidence로 탐지 테스트
- [ ] confidence override 테스트 (0.3, 0.5, 0.8)
- [ ] DEBUG_MODE=true로 bbox 이미지 생성 확인
- [ ] 잘못된 파일 타입 에러 테스트
- [ ] 처리 시간 측정 (목표: < 1초)

---

### Phase 2 검증 체크리스트

- [ ] YOLO 모델 로드 성공 (시작 시 로그 확인)
- [ ] `/health` 엔드포인트에서 `model_loaded: true`
- [ ] `/api/v1/detect` 엔드포인트 정상 동작
- [ ] 탐지 결과 JSON 응답 정확성
- [ ] confidence 파라미터 override 동작
- [ ] DEBUG_MODE=true일 때 bbox 이미지 `outputs/`에 저장
- [ ] 파일 타입 검증 동작 (gif, txt 거부)
- [ ] 파일 크기 제한 동작 (> 10MB 거부)
- [ ] ReDoc 문서 자동 생성 확인
- [ ] 에러 응답 형식 일관성
- [ ] 로그 파일 `logs/app.log`에 구조화된 로그 기록

**예상 소요 시간**: 4-5일

---

## Phase 3: 기능 개선 및 테스트 (3주차)

### 목표
✅ 데이터베이스 로깅 추가 (선택사항)
✅ 포괄적인 에러 처리 및 검증
✅ 단위/통합 테스트 작성
✅ API 문서 개선

### 세부 작업

#### 3.1 데이터베이스 통합 (선택사항)

**파일**: `app/db/database.py`
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

Base = declarative_base()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

**파일**: `app/models/detection_log.py`
```python
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime
from app.db.database import Base

class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(String(36), unique=True, index=True)
    image_filename = Column(String(255))
    image_size_width = Column(Integer)
    image_size_height = Column(Integer)
    confidence_threshold = Column(Float)
    detection_count = Column(Integer)
    detections_json = Column(JSON)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<DetectionLog(id={self.id}, count={self.detection_count})>"
```

**DB 초기화 스크립트**: `scripts/init_db.py`
```python
import asyncio
from app.db.database import engine, Base
from app.models.detection_log import DetectionLog

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created successfully")

if __name__ == "__main__":
    asyncio.run(init_db())
```

**체크리스트**:
- [ ] `app/db/database.py` 구현
- [ ] `app/models/detection_log.py` 구현
- [ ] `scripts/init_db.py` 생성
- [ ] DB 초기화 실행
- [ ] `detection.py`에 로깅 로직 추가 (선택)

---

#### 3.2 에러 처리 개선

**파일**: `app/middleware/error_handler.py`
```python
from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.utils.logger import get_logger
import uuid

logger = get_logger(__name__)

class ErrorHandlerMiddleware:
    """전역 에러 핸들러 미들웨어"""

    async def __call__(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            request_id = str(uuid.uuid4())
            logger.error(
                f"Unhandled exception",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method
                }
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "Internal server error",
                    "request_id": request_id
                }
            )
```

**파일**: `app/main.py` 업데이트
```python
from app.middleware.error_handler import ErrorHandlerMiddleware

# 미들웨어 추가
app.middleware("http")(ErrorHandlerMiddleware())
```

**체크리스트**:
- [ ] `app/middleware/error_handler.py` 구현
- [ ] 미들웨어 등록
- [ ] 에러 시 request_id 추적 확인

---

#### 3.3 테스트 작성

**파일**: `pytest.ini`
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

**파일**: `tests/conftest.py`
```python
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_image():
    """테스트 이미지 fixture"""
    fixture_path = Path(__file__).parent / "fixtures" / "sample.jpg"
    return fixture_path
```

**파일**: `tests/test_health.py`
```python
def test_health_check(client):
    """헬스체크 엔드포인트 테스트"""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data
    assert "timestamp" in data
```

**파일**: `tests/test_detection.py`
```python
import pytest

def test_detect_valid_image(client, sample_image):
    """유효한 이미지로 탐지 테스트"""
    if not sample_image.exists():
        pytest.skip("Sample image not found")

    with open(sample_image, "rb") as f:
        response = client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "image_id" in data
    assert "detections" in data
    assert "count" in data
    assert "processing_time" in data
    assert isinstance(data["detections"], list)

def test_detect_with_confidence_override(client, sample_image):
    """confidence override 테스트"""
    if not sample_image.exists():
        pytest.skip("Sample image not found")

    with open(sample_image, "rb") as f:
        response = client.post(
            "/api/v1/detect",
            params={"confidence": 0.7},
            files={"file": ("test.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

def test_detect_invalid_file_type(client):
    """잘못된 파일 타입 거부 테스트"""
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )

    assert response.status_code == 400
    data = response.json()
    assert "Unsupported format" in data["detail"]

def test_detect_missing_file(client):
    """파일 없이 요청 테스트"""
    response = client.post("/api/v1/detect")

    assert response.status_code == 422  # FastAPI validation error
```

**체크리스트**:
- [ ] `pytest.ini` 생성
- [ ] `tests/conftest.py` 생성
- [ ] `tests/test_health.py` 작성
- [ ] `tests/test_detection.py` 작성
- [ ] `tests/fixtures/sample.jpg` 준비
- [ ] 테스트 실행: `pytest tests/ -v`
- [ ] 커버리지 측정: `pytest tests/ --cov=app --cov-report=html`

---

#### 3.4 API 문서 개선

**파일**: `app/main.py` 업데이트
```python
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## YOLO Detection API

학습 및 실제 배포를 위한 FastAPI 기반 YOLO 객체 탐지 서비스입니다.

### 주요 기능
- 실시간 이미지 객체 탐지
- 신뢰도 임계값 조정 가능
- 디버그 모드 (bbox 시각화)
- 구조화된 JSON 응답

### 사용 예제
```bash
curl -X POST http://localhost:8000/api/v1/detect \\
  -F "file=@image.jpg" \\
  -F "confidence=0.5"
```
    """,
    lifespan=lifespan,
    docs_url="/docs",  # ReDoc
    openapi_url="/openapi.json"
)
```

**체크리스트**:
- [ ] API 설명 추가
- [ ] 엔드포인트 docstring 개선
- [ ] 예제 요청/응답 추가
- [ ] ReDoc 문서 시각적 확인

---

### Phase 3 검증 체크리스트

- [ ] 데이터베이스 테이블 생성 성공 (선택사항)
- [ ] 모든 단위 테스트 통과
- [ ] 테스트 커버리지 > 80%
- [ ] 에러 핸들링 일관성 확인
- [ ] API 문서 완성도 (ReDoc)
- [ ] 로그 품질 검증

**예상 소요 시간**: 5-6일

---

## Phase 4: 프로덕션 배포 준비 (4주차)

### 목표
✅ 프로덕션 Dockerfile 최적화
✅ Docker Compose 배포 설정
✅ Nginx 리버스 프록시 구성
✅ 모니터링 및 로깅 개선

### 세부 작업

#### 4.1 프로덕션 Dockerfile (멀티스테이지)

**파일**: `docker/Dockerfile`
```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# 시스템 의존성 (빌드용)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Requirements 복사
COPY requirements.txt .

# Wheels 빌드
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

# Builder에서 wheels 복사
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt .

# Wheels로부터 설치
RUN pip install --no-cache /wheels/*

# 애플리케이션 코드 복사
COPY ./app /app/app

# 디렉토리 생성
RUN mkdir -p /app/models /app/uploads /app/outputs /app/logs

# 비루트 사용자
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# 헬스체크
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**체크리스트**:
- [ ] `docker/Dockerfile` 생성
- [ ] 멀티스테이지 빌드 테스트
- [ ] 이미지 크기 확인 (목표: < 800MB)
- [ ] 헬스체크 동작 확인

---

#### 4.2 프로덕션 Docker Compose

**파일**: `docker/docker-compose.prod.yml`
```yaml
version: '3.8'

services:
  app:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - APP_NAME=YOLO Detection API
      - APP_VERSION=1.0.0
      - DEBUG=false
      - DEBUG_MODE=false
      - MODEL_PATH=/app/models/yolov8n.pt
      - CONFIDENCE_THRESHOLD=0.5
      - DATABASE_URL=mysql+asyncmy://yolo_user:yolo_pass@db:3306/yolo_db
      - LOG_LEVEL=INFO
    volumes:
      - ../models:/app/models:ro
      - app_uploads:/app/uploads
      - app_outputs:/app/outputs
      - app_logs:/app/logs
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  db:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: yolo_db
      MYSQL_USER: yolo_user
      MYSQL_PASSWORD: ${MYSQL_PASSWORD}
    volumes:
      - mysql_data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    restart: unless-stopped

volumes:
  app_uploads:
  app_outputs:
  app_logs:
  mysql_data:
```

**체크리스트**:
- [ ] `docker-compose.prod.yml` 생성
- [ ] 환경변수 파일 `.env.prod` 생성
- [ ] 볼륨 마운트 전략 확인
- [ ] 헬스체크 의존성 설정

---

#### 4.3 Nginx 리버스 프록시 설정

**파일**: `docker/nginx.conf`
```nginx
events {
    worker_connections 1024;
}

http {
    upstream fastapi_backend {
        server app:8000;
    }

    # 로그 형식
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    server {
        listen 80;
        server_name localhost;

        client_max_body_size 10M;

        location / {
            proxy_pass http://fastapi_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # 타임아웃
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        location /health {
            proxy_pass http://fastapi_backend/health;
            access_log off;
        }

        location /docs {
            proxy_pass http://fastapi_backend/docs;
        }
    }
}
```

**체크리스트**:
- [ ] `docker/nginx.conf` 생성
- [ ] client_max_body_size 설정 확인
- [ ] 타임아웃 설정 확인
- [ ] 로그 형식 확인

---

#### 4.4 배포 스크립트

**파일**: `scripts/deploy.sh`
```bash
#!/bin/bash
set -e

echo "🚀 Starting deployment..."

# 1. 환경 확인
if [ ! -f .env.prod ]; then
    echo "❌ .env.prod not found"
    exit 1
fi

# 2. 모델 파일 확인
if [ ! -f models/yolov8n.pt ]; then
    echo "⚠️  Model file not found, downloading..."
    mkdir -p models
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    cp ~/.cache/ultralytics/yolov8n.pt models/
fi

# 3. 이미지 빌드
echo "🔨 Building Docker images..."
docker-compose -f docker/docker-compose.prod.yml build

# 4. 컨테이너 시작
echo "🐳 Starting containers..."
docker-compose -f docker/docker-compose.prod.yml up -d

# 5. 헬스체크
echo "🏥 Waiting for health check..."
sleep 10
curl -f http://localhost/health || exit 1

echo "✅ Deployment complete!"
echo "📚 API Docs: http://localhost/docs"
```

**체크리스트**:
- [ ] `scripts/deploy.sh` 생성
- [ ] 실행 권한 부여: `chmod +x scripts/deploy.sh`
- [ ] 배포 스크립트 테스트

---

#### 4.5 모니터링 및 로깅 개선

**파일**: `docker/docker-compose.prod.yml` 업데이트 (로그 드라이버)
```yaml
services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**로그 확인 명령**:
```bash
# 앱 로그 확인
docker-compose -f docker/docker-compose.prod.yml logs -f app

# 실시간 로그 모니터링
tail -f logs/app.log

# Nginx 로그
docker-compose -f docker/docker-compose.prod.yml logs nginx
```

**체크리스트**:
- [ ] 로그 로테이션 설정
- [ ] 로그 볼륨 마운트 확인
- [ ] 구조화된 JSON 로그 검증

---

#### 4.6 프로덕션 README 작성

**파일**: `README.md`
```markdown
# YOLO Detection API

FastAPI 기반 YOLO 이미지 객체 탐지 API 서버

## 빠른 시작

### 개발 환경 (VSCode Dev Container)
1. VSCode 열기
2. "Reopen in Container" 클릭
3. 자동 실행: http://localhost:8000

### 프로덕션 배포
```bash
# 환경 설정
cp .env.example .env.prod
# .env.prod 편집

# 배포 실행
bash scripts/deploy.sh

# 서비스 확인
curl http://localhost/health
```

## API 사용법

### 객체 탐지
```bash
curl -X POST http://localhost/api/v1/detect \
  -F "file=@image.jpg" \
  -F "confidence=0.5"
```

## 개발 가이드
- [기술 스택](claudedocs/tech-stack.md)
- [시스템 설계](claudedocs/system-design.md)
- [개발 로드맵](claudedocs/DEVELOPMENT_ROADMAP.md)

## 라이선스
MIT
```

**체크리스트**:
- [ ] `README.md` 작성
- [ ] 빠른 시작 가이드 검증
- [ ] API 사용 예제 테스트

---

### Phase 4 검증 체크리스트

- [ ] 프로덕션 이미지 빌드 성공
- [ ] Docker Compose 전체 스택 실행 성공
- [ ] Nginx 리버스 프록시 동작 확인
- [ ] 헬스체크 동작 (Docker + Nginx)
- [ ] 로그 수집 및 로테이션 동작
- [ ] 배포 스크립트 실행 성공
- [ ] API 문서 접근 가능 (http://localhost/docs)
- [ ] 종단간 탐지 테스트 성공
- [ ] 프로덕션 환경 성능 측정 (< 1초 응답)

**예상 소요 시간**: 5-6일

---

## 🎯 성공 메트릭

### 기술 메트릭
| 메트릭 | 목표 | 측정 방법 |
|--------|--------|--------------------|
| API 가용성 | >99% | 업타임 모니터링 |
| 응답 시간 | <1s (p95) | 로그 분석 |
| 에러 비율 | <1% | HTTP 상태 추적 |
| 테스트 커버리지 | >80% | pytest-cov |
| 문서화 | 100% | ReDoc 완성도 |

### 개발 메트릭
| 메트릭 | 목표 | 현황 |
|--------|------|------|
| 프로젝트 구조 | 완성 | ⏳ Phase 1 |
| YOLO 통합 | 완성 | ⏳ Phase 2 |
| 테스트 작성 | >80% 커버리지 | ⏳ Phase 3 |
| 배포 준비 | 프로덕션 가능 | ⏳ Phase 4 |

---

## 📚 참고 자료

### 프로젝트 문서
- [기술 스택 상세](tech-stack.md)
- [시스템 설계 문서](system-design.md)

### 외부 문서
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Docker 베스트 프랙티스](https://docs.docker.com/develop/dev-best-practices/)
- [VSCode Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)

---

## 🔧 트러블슈팅

### 일반적인 문제

**문제**: 모델 로드 실패
```bash
# 해결책
1. .env의 MODEL_PATH 확인
2. 모델 파일 다운로드: python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
3. models/ 디렉토리 권한 확인
```

**문제**: 메모리 부족
```bash
# 해결책
1. 더 작은 모델 사용 (yolov8n.pt)
2. Docker 메모리 제한 증가
3. uploads/ 및 outputs/ 정기 정리
```

**문제**: 느린 추론
```bash
# 해결책
1. confidence threshold 상향 (불필요한 탐지 감소)
2. 이미지 해상도 축소
3. CPU 최적화 확인
```

---

## 📋 체크리스트 요약

### Phase 1: 기초 (1주차)
- [ ] 프로젝트 구조 생성
- [ ] Dev Container 설정
- [ ] Docker 개발 환경
- [ ] 핵심 설정 관리
- [ ] 구조화된 로깅
- [ ] 기본 FastAPI + 헬스체크
- [ ] VSCode 디버거 설정
- [ ] Git 설정

### Phase 2: YOLO 통합 (2주차)
- [ ] YOLO 모델 매니저 싱글톤
- [ ] 이미지 처리 유틸리티
- [ ] Pydantic 스키마
- [ ] 객체 탐지 엔드포인트
- [ ] 수동 테스트

### Phase 3: 개선 (3주차)
- [ ] 데이터베이스 통합 (선택)
- [ ] 에러 처리 개선
- [ ] 단위/통합 테스트
- [ ] API 문서 개선

### Phase 4: 배포 (4주차)
- [ ] 프로덕션 Dockerfile
- [ ] Docker Compose 배포
- [ ] Nginx 리버스 프록시
- [ ] 배포 스크립트
- [ ] 모니터링 및 로깅
- [ ] README 작성

---

**로드맵 버전**: 1.0
**최종 업데이트**: 2025-11-11
**다음 업데이트 예정**: Phase 1 완료 후
