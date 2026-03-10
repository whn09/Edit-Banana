"""
OCR 数据源模块

包含：
    - LocalOCR: 本地 Tesseract OCR（默认，易安装）
    - VlmOCR: VLM 视觉语言模型 OCR（需要部署 VLM 服务）
    - Pix2TextOCR: Pix2Text 公式识别（可选）
    - TextBlock, OCRResult: 通用数据结构
"""

from .base import TextBlock, OCRResult
from .local_ocr import LocalOCR
from .vlm_ocr import VlmOCR

try:
    from .pix2text import Pix2TextOCR, Pix2TextBlock, Pix2TextResult
except ImportError:
    Pix2TextOCR = None
    Pix2TextBlock = None
    Pix2TextResult = None

__all__ = [
    "TextBlock",
    "OCRResult",
    "LocalOCR",
    "VlmOCR",
    "Pix2TextOCR",
    "Pix2TextBlock",
    "Pix2TextResult",
]
