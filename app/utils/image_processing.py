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
