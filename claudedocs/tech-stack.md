# YOLO Detection API - Technology Stack

## Project Overview
FastAPI 기반 YOLO 이미지 객체 탐지 API 서버 (학습용)

**목적**: 이미지를 받아서 YOLO 모델로 객체 탐지 후 결과 반환
**규모**: 단일 클라이언트 요청 처리 (스터디 용도)
**배포**: Docker 기반 로컬 + 실제 서버

---

## Core Technology Stack

### Backend Framework
- **Python 3.11**
  - 안정성과 성능의 균형
  - YOLO 라이브러리 호환성 우수
  - Type hints 완전 지원

- **FastAPI 0.104+**
  - 비동기 처리 지원
  - 자동 API 문서화 (ReDoc)
  - Pydantic 기반 타입 검증
  - 빠른 성능 (Starlette + Uvicorn)

- **Uvicorn**
  - ASGI 서버
  - 비동기 요청 처리

### YOLO & Image Processing
- **Ultralytics (YOLOv8/v10)**
  - 공식 YOLO 구현체
  - Pre-trained 모델 지원
  - 커스텀 모델 로드 가능

- **OpenCV (opencv-python-headless)**
  - 서버용 headless 버전 (GUI 없음)
  - 이미지 전처리/후처리
  - BBox 렌더링 (디버그 모드)
  - Ultralytics와 완벽한 호환

- **NumPy**
  - 배열 연산
  - OpenCV-YOLO 데이터 파이프라인

### Database (Optional)
- **MySQL 8.0**
  - 요청 로그 저장
  - 탐지 결과 히스토리 (선택)

- **SQLAlchemy 2.0**
  - ORM (권장)
  - 비동기 DB 작업 지원
  - Type-safe 쿼리

### API Features
- **이미지 입력 방식**: `multipart/form-data` 파일 업로드
  - 직관적이고 테스트 용이
  - FastAPI 기본 지원
  - 추가 옵션: URL 기반 입력도 가능

- **처리 방식**: 동기 즉시 응답
  - YOLO는 빠르므로 실시간 응답 가능
  - 단일 이미지 처리

- **디버그 옵션**: `debug=true` 플래그
  - BBox 그려진 이미지 로컬 저장
  - 개발/디버깅 용도

### Development & Deployment
- **VSCode Dev Containers**
  - 개발 환경 = 배포 환경 (완전 동일)
  - 디버거 통합 지원
  - Hot reload 지원 (볼륨 마운트)
  - 필요한 확장 자동 설치

- **Docker**
  - CPU 최적화 이미지
  - Multi-stage build (경량화)
  - python:3.11-slim 베이스

- **Docker Compose**
  - 개발: FastAPI + MySQL (선택)
  - 멀티 서비스 통합
  - 네트워크 자동 설정

- **배포 전략**
  - 로컬 개발: VSCode Dev Container
  - 서버: Docker + Nginx reverse proxy
  - 대안: Railway, Render, AWS ECS

### API Documentation
- **ReDoc**: `/docs` - 깔끔한 자동 생성 API 문서

### Testing & Quality
- **Pytest**
  - 단위 테스트
  - 통합 테스트

- **httpx**
  - 비동기 HTTP 클라이언트
  - FastAPI 테스트

### Additional Tools
- **Pydantic v2**
  - 요청/응답 스키마 검증
  - 자동 직렬화/역직렬화

- **python-dotenv**
  - 환경변수 관리
  - `.env` 파일 지원

- **debugpy**
  - VSCode 디버거 연동
  - 브레이크포인트, 변수 검사

- **Python logging**
  - 구조화된 로깅
  - 파일 rotation

---

## Project Structure

```
server_study/
├── .devcontainer/
│   └── devcontainer.json          # VSCode Dev Container 설정
│
├── .vscode/
│   └── launch.json                # 디버거 설정
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 애플리케이션 엔트리포인트
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── detection.py       # YOLO 탐지 엔드포인트
│   │       └── health.py          # 헬스체크 엔드포인트
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # 앱 설정 (환경변수)
│   │   └── yolo_model.py          # YOLO 모델 래퍼 클래스
│   ├── schemas/                   # Pydantic 스키마
│   │   ├── __init__.py
│   │   ├── detection.py           # 탐지 요청/응답 스키마
│   │   └── common.py
│   ├── models/                    # SQLAlchemy 모델 (선택)
│   │   ├── __init__.py
│   │   └── detection_log.py
│   ├── db/
│   │   ├── __init__.py
│   │   └── database.py            # DB 연결 설정
│   └── utils/
│       ├── __init__.py
│       ├── image_processing.py    # 이미지 유틸리티
│       └── logger.py              # 로깅 설정
│
├── models/                        # YOLO 모델 파일 (.gitignore)
│   ├── yolov8n.pt                # Pre-trained 또는 커스텀 모델
│   └── checkpoints/
│
├── uploads/                       # 임시 업로드 (.gitignore)
│
├── outputs/                       # 디버그용 bbox 이미지 (.gitignore)
│
├── tests/
│   ├── __init__.py
│   ├── test_detection.py
│   └── test_health.py
│
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.dev             # 개발용 이미지
│   └── docker-compose.yml
│
├── claudedocs/                    # 프로젝트 문서
│   └── tech-stack.md
│
├── .env.example                   # 환경변수 템플릿
├── .gitignore
├── requirements.txt               # 운영 의존성
├── requirements-dev.txt           # 개발 의존성
├── pytest.ini                     # Pytest 설정
└── README.md
```

---

## Model Storage Strategy

### 로컬 개발
```
models/
├── yolov8n.pt          # Pre-trained 모델
├── custom_v1.pt        # 커스텀 학습 모델
└── .gitkeep
```

### Git 관리
- `.gitignore`에 `models/*.pt` 추가
- 모델 파일은 Git에서 제외 (용량 문제)
- `.gitkeep`으로 디렉토리 구조 유지

### 배포 시
- 모델 파일을 별도로 다운로드/복사
- Docker 빌드 시 COPY 또는 볼륨 마운트
- 옵션: S3/GCS에 저장 후 시작 시 다운로드

---

## API Endpoints (예상)

### Detection
```
POST /api/v1/detect
Content-Type: multipart/form-data

Parameters:
- file: 이미지 파일 (required)
- confidence: float (default: 0.5)
- debug: bool (default: false)

Response:
{
  "success": true,
  "detections": [
    {
      "class": "person",
      "confidence": 0.95,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "processing_time": 0.123,
  "debug_image_path": "/outputs/debug_001.jpg"  # if debug=true
}
```

### Health Check
```
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

## Development Workflow (Dev Container 기반)

### 🚀 초기 설정 (최초 1회)

```bash
# 1. VSCode 설치
# 2. Dev Containers 확장 설치
# 3. Docker Desktop 설치 및 실행
```

### 💻 개발 시작

```bash
# 1. 프로젝트 열기
code /path/to/server_study

# 2. VSCode 팝업에서 "Reopen in Container" 클릭
#    또는 Cmd+Shift+P → "Dev Containers: Reopen in Container"

# 3. 자동 진행:
#    - Docker 이미지 빌드
#    - 컨테이너 시작 (FastAPI + MySQL)
#    - VSCode 컨테이너 내부 연결
#    - Python 확장 자동 설치
#    - 의존성 자동 설치

# 4. 서버 자동 시작
#    - uvicorn이 --reload 모드로 실행
#    - http://localhost:8000 접근 가능
```

### 🐛 디버깅

```bash
# F5 또는 Run > Start Debugging
# - 브레이크포인트 설정
# - 변수 검사
# - 스텝 실행
# - YOLO 모델 내부까지 디버깅 가능
```

### 🔄 Hot Reload 개발

```bash
# 로컬에서 파일 수정 → 자동 저장
# → 볼륨 마운트로 컨테이너 내부 반영
# → uvicorn --reload로 자동 재시작
# → 브라우저 새로고침으로 즉시 확인
```

### ✅ 테스트 실행

```bash
# 컨테이너 내부 터미널에서 (VSCode 터미널 사용)
pytest tests/ -v

# 커버리지 포함
pytest tests/ --cov=app --cov-report=html
```

### 📚 API 문서 확인

```bash
# ReDoc
http://localhost:8000/docs
```

### 🔍 로그 확인

```bash
# VSCode 터미널에서 (컨테이너 내부)
tail -f logs/app.log

# Docker 로그 (호스트에서)
docker-compose logs -f app
```

### 🛑 종료

```bash
# VSCode 닫기 → 컨테이너 자동 정리
# 또는 Cmd+Shift+P → "Dev Containers: Reopen Locally"
```

### 🔧 컨테이너 내부 접속 (필요시)

```bash
# 호스트 터미널에서
docker-compose exec app bash

# 컨테이너 내부에서 명령 실행
python -m pytest
pip list
```

---

## Environment Variables

```bash
# .env
APP_NAME=YOLO Detection API
APP_VERSION=1.0.0
DEBUG=false

# Model
MODEL_PATH=models/yolov8n.pt
CONFIDENCE_THRESHOLD=0.5

# Database (optional)
DATABASE_URL=mysql+asyncmy://user:pass@localhost:3306/yolo_db

# Paths
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

---

## Dependencies

### requirements.txt (운영 환경)
```
# Core
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
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

### requirements-dev.txt (개발 환경)
```
# Include production dependencies
-r requirements.txt

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
httpx==0.25.2

# Debugging
debugpy==1.8.0

# Code Quality (optional)
black==23.12.1
flake8==7.0.0
mypy==1.8.0
isort==5.13.2
```

---

## Dev Container vs venv 비교

| 측면 | venv | Dev Container |
|------|------|---------------|
| 환경 일관성 | ⚠️ OS별 차이 | ✅ 완전 동일 |
| 설정 복잡도 | 🟡 중간 | ✅ 간단 |
| OpenCV 설치 | ⚠️ 시스템 의존성 문제 | ✅ 이미지에 포함 |
| 멀티 서비스 | ❌ 별도 설치 | ✅ 자동 통합 |
| 배포 환경 일치 | ❌ 차이 있음 | ✅ 100% 동일 |
| VSCode 디버거 | ✅ 쉬움 | ✅ 완벽 통합 |
| Hot Reload | ✅ 빠름 | ✅ 볼륨 마운트로 가능 |
| 온보딩 시간 | 🟡 10-30분 | ✅ 5분 |

**결론**: VSCode + 디버거 사용 환경에서는 **Dev Container 강력 권장**

---

## Next Steps
1. ✅ 기술 스택 확정
2. Dev Container 설정 파일 생성
3. 프로젝트 구조 생성
4. 기본 FastAPI 앱 설정
5. YOLO 모델 통합
6. API 엔드포인트 구현
7. 테스트 작성
8. 배포 준비
