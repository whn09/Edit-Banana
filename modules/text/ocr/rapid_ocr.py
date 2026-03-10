"""
RapidOCR adapter — wraps RapidOCR to return OCRResult compatible with Edit-Banana.

RapidOCR uses PaddleOCR models via ONNX runtime, giving much better
detection and recognition than Tesseract, especially for:
  - Text line detection (proper grouping, not word-level fragmentation)
  - Accurate bounding boxes (4-point polygons)
  - Multi-language support
"""

from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from .base import TextBlock, OCRResult


class RapidOCRAdapter:
    """OCR adapter using RapidOCR (PaddleOCR models via ONNX)."""

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self._engine = RapidOCR()

    def analyze_image(self, image_path: str) -> OCRResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path)
        width, height = img.size

        result, elapse = self._engine(str(image_path))

        if not result:
            print("[RapidOCR] No text detected")
            return OCRResult(image_width=width, image_height=height)

        text_blocks = []
        for item in result:
            bbox_points, text, confidence = item

            if not text or not text.strip():
                continue
            if confidence < 0.3:
                continue

            # bbox_points is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            polygon = [(p[0], p[1]) for p in bbox_points]

            # Calculate height for font_size_px
            ys = [p[1] for p in polygon]
            h = max(ys) - min(ys)

            block = TextBlock(
                text=text.strip(),
                polygon=polygon,
                confidence=confidence,
                font_size_px=float(h),
                spans=[],
            )
            text_blocks.append(block)

        print(f"[RapidOCR] Detected {len(text_blocks)} text blocks ({elapse[0]:.2f}s det, {elapse[2]:.2f}s rec)")
        return OCRResult(
            image_width=width,
            image_height=height,
            text_blocks=text_blocks,
        )
