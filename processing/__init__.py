# processing/__init__.py

"""
Processing module for license plate detection and recognition
"""

from .detect import (
    load_models,             # Tải mô hình YOLO và CNN (Cần thiết khi khởi động)
    process_image_and_recognize # Hàm xử lý chính (YOLO -> Cắt -> CNN)
)

__all__ = [
    'load_models',
    'process_image_and_recognize'
]