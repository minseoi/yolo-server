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
                "timestamp": "2025-11-23T10:30:00Z"
            }
        }
    }


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[dict] = None
    request_id: Optional[str] = None
