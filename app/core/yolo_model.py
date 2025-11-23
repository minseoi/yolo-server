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
                logger.warning(f"Model file not found at {model_path}, downloading YOLOv8-Nano...")
                # Ultralytics will auto-download if not found
                self._model = YOLO('yolov8n.pt')
                logger.info("YOLOv8-Nano downloaded successfully")
            else:
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
            # YOLOv8-Nano 기본 모델 식별
            return "YOLOv8-Nano"
        return "unknown"
