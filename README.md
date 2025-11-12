# YOLO Detection API

FastAPI 기반 YOLO 이미지 객체 탐지 API 서버

## 프로젝트 개요

학습 및 실제 배포를 위한 프로덕션 준비 완료 시스템

### 핵심 기능
- FastAPI + YOLO(v8/v10) 기반 실시간 객체 탐지 API
- VSCode Dev Container 기반 개발 환경 (개발 = 배포 환경)
- Docker 기반 배포 전략 (로컬 + 실제 서버)
- 선택적 MySQL 데이터베이스 로깅
- ReDoc 자동 API 문서화

### 기술 스택
- **Backend**: Python 3.11, FastAPI 0.104+, Uvicorn
- **YOLO**: Ultralytics (YOLOv8/v10), OpenCV-headless, NumPy
- **Database**: MySQL 8.0 + SQLAlchemy 2.0 (선택사항)
- **Dev Environment**: VSCode Dev Containers, Docker Compose
- **Testing**: Pytest, httpx, pytest-cov

## 빠른 시작

### 개발 환경 (VSCode Dev Container)

1. 사전 요구사항
   - VSCode 설치
   - Docker Desktop 설치 및 실행
   - VSCode Dev Containers 확장 설치

2. 컨테이너에서 프로젝트 열기
   ```bash
   # VSCode에서 프로젝트 폴더 열기
   # Command Palette (Cmd/Ctrl + Shift + P)
   # "Dev Containers: Reopen in Container" 선택
   ```

3. 컨테이너가 빌드되고 의존성이 자동 설치됩니다

4. 서버 실행 확인
   ```bash
   # 자동 실행됨: http://localhost:8000
   # 헬스체크: http://localhost:8000/health
   # API 문서: http://localhost:8000/docs
   ```

### 로컬 개발 (Docker 없이)

```bash
# Python 3.11 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements-dev.txt

# 환경 변수 설정
cp .env.example .env

# 서버 실행
uvicorn app.main:app --reload

# 접속: http://localhost:8000
```

## 프로젝트 구조

```
yolo-server/
├── .devcontainer/          # VSCode Dev Container 설정
├── .vscode/                # VSCode 디버거 설정
├── app/
│   ├── main.py            # FastAPI 엔트리포인트
│   ├── api/routes/        # API 라우트
│   ├── core/              # 핵심 설정 및 모델
│   ├── schemas/           # Pydantic 스키마
│   ├── models/            # SQLAlchemy 모델
│   ├── db/                # 데이터베이스 설정
│   └── utils/             # 유틸리티 함수
├── models/                # YOLO .pt 파일
├── uploads/               # 임시 업로드
├── outputs/               # 디버그 bbox 이미지
├── tests/                 # 테스트 코드
├── docker/                # Docker 설정
├── claudedocs/            # 프로젝트 문서
└── scripts/               # 유틸리티 스크립트
```

## API 엔드포인트

### 헬스체크
```bash
GET /health
```

### 객체 탐지 (Phase 2에서 구현 예정)
```bash
POST /api/v1/detect
```

## 개발 상태

✅ **Phase 1: 프로젝트 기초 설정** (완료)
- 프로젝트 구조 생성
- Dev Container 설정
- Docker 개발 환경
- 핵심 설정 관리
- 구조화된 로깅 시스템
- 기본 FastAPI + 헬스체크
- VSCode 디버거 설정
- Git 설정

⏳ **Phase 2: YOLO 핵심 기능 구현** (대기 중)
⏳ **Phase 3: 기능 개선 및 테스트** (대기 중)
⏳ **Phase 4: 프로덕션 배포 준비** (대기 중)

## 문서

- [기술 스택](claudedocs/tech-stack.md)
- [시스템 설계](claudedocs/system-design.md)
- [개발 로드맵](claudedocs/DEVELOPMENT_ROADMAP.md)

## 라이선스

MIT