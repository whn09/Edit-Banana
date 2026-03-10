"""
VLM OCR 模块 -- 使用视觉语言模型进行文字识别

功能：
    调用 OpenAI 兼容 API（如本地部署的 Qwen3.5-9B）对图片做 OCR。
    支持两种工作模式：
      - enhance: 用 Tesseract 获取 bbox，VLM 矫正文字内容（推荐）
      - full:    完全由 VLM 识别文字和位置（bbox 精度较低，适合 Tesseract 效果差的场景）

使用示例：
    ocr = VlmOCR(base_url="http://localhost:11434/v1", model="qwen3.5-9b")
    result = ocr.analyze_image("input.png")
"""

import base64
import io
import json
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from PIL import Image

from .base import TextBlock, OCRResult


# -- Prompts ----------------------------------------------------------------

_PROMPT_FULL_TEMPLATE = (
    "This image is {w}x{h} pixels. "
    "List ALL text visible in this image as a JSON array. "
    "Each element: {{\"text\":\"the text\",\"bbox_2d\":[x1,y1,x2,y2],"
    "\"font_size\":12,\"is_bold\":false,\"is_italic\":false,\"font_color\":\"#000000\","
    "\"font_family\":\"Arial\"}}. "
    "bbox_2d MUST be in pixel coordinates where x ranges 0-{w} and y ranges 0-{h}. "
    "font_size: estimate the font size in points (e.g. 8, 10, 12, 14, 16, 20, 24). "
    "font_color: detect actual text color (e.g. #1d1d1d, not always #000000). "
    "font_family: detect the font (Arial, Times New Roman, Georgia, etc). "
    "Include EVERY piece of text, even small labels. Preserve original language. "
    "Return ONLY the JSON array."
)

_PROMPT_ENHANCE_TEMPLATE = (
    "You are an expert OCR corrector. I give you an image and text blocks "
    "detected by a basic OCR engine. Some texts may be wrong or garbled.\n\n"
    "For each block, output the corrected text and detect style.\n"
    "Return a JSON array in the SAME order as input:\n"
    '[{{"index":0,"text":"corrected","is_bold":false,"is_italic":false,"font_color":"#000000"}}]\n\n'
    "OCR-detected blocks:\n{ocr_blocks}\n\n"
    "Return ONLY the JSON array. No explanation, no markdown."
)


class VlmOCR:
    """
    VLM OCR 客户端（OpenAI 兼容 API）
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen3.5-9b",
        api_key: str = "not-needed",
        mode: str = "enhance",
        max_tokens: int = 8000,
        timeout: int = 300,
        fallback_to_tesseract: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.mode = mode
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.fallback_to_tesseract = fallback_to_tesseract
        self._tesseract_ocr = None

    @property
    def tesseract_ocr(self):
        if self._tesseract_ocr is None:
            from .local_ocr import LocalOCR
            self._tesseract_ocr = LocalOCR()
        return self._tesseract_ocr

    # -- Main entry ---------------------------------------------------------

    def analyze_image(self, image_path: str) -> OCRResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        width, height = img.size

        try:
            if self.mode == "enhance":
                return self._analyze_enhance(image_path, img, width, height)
            else:
                return self._analyze_full(image_path, img, width, height)
        except Exception as e:
            print(f"[VlmOCR] VLM call failed: {e}")
            if self.fallback_to_tesseract:
                print("[VlmOCR] Falling back to Tesseract")
                return self.tesseract_ocr.analyze_image(str(image_path))
            raise

    # -- enhance mode -------------------------------------------------------

    ENHANCE_BATCH_SIZE = 50  # max blocks per VLM call to avoid timeout

    def _analyze_enhance(self, image_path, img, width, height):
        tess_result = self.tesseract_ocr.analyze_image(str(image_path))
        if not tess_result.text_blocks:
            return tess_result

        image_b64 = self._encode_image(img)
        blocks = tess_result.text_blocks
        total_corrected = 0

        # Process in batches to avoid prompt-too-long / timeout
        for batch_start in range(0, len(blocks), self.ENHANCE_BATCH_SIZE):
            batch_end = min(batch_start + self.ENHANCE_BATCH_SIZE, len(blocks))
            batch_blocks = blocks[batch_start:batch_end]

            ocr_blocks_desc = []
            for i, block in enumerate(batch_blocks):
                ocr_blocks_desc.append({
                    "index": i,
                    "text": block.text,
                    "bbox": self._polygon_to_bbox(block.polygon),
                })

            prompt_text = _PROMPT_ENHANCE_TEMPLATE.format(
                ocr_blocks=json.dumps(ocr_blocks_desc, ensure_ascii=False)
            )

            try:
                vlm_response = self._call_vlm(image_b64, prompt_text)
                corrections = self._parse_json_response(vlm_response)
            except Exception as e:
                print(f"[VlmOCR] enhance batch {batch_start}-{batch_end} failed: {e}")
                continue

            if not corrections:
                continue

            correction_map = {}
            for item in corrections:
                idx = item.get("index")
                if idx is not None:
                    correction_map[idx] = item

            for i, block in enumerate(batch_blocks):
                corr = correction_map.get(i)
                if not corr:
                    continue
                corrected_text = corr.get("text", "").strip()
                if corrected_text:
                    block.text = corrected_text
                if corr.get("is_bold"):
                    block.is_bold = True
                    block.font_weight = "bold"
                if corr.get("is_italic"):
                    block.is_italic = True
                    block.font_style = "italic"
                font_color = corr.get("font_color")
                if font_color and font_color != "null" and font_color != "#000000":
                    block.font_color = font_color
                total_corrected += 1

        print(f"[VlmOCR] enhance done: corrected {total_corrected}/{len(blocks)} blocks")
        return tess_result

    # -- full mode ----------------------------------------------------------

    def _analyze_full(self, image_path, img, width, height):
        # Send image at original resolution with size hint in prompt
        image_b64 = self._encode_image(img, max_dim=max(width, height))
        prompt = _PROMPT_FULL_TEMPLATE.format(w=width, h=height)
        vlm_response = self._call_vlm(image_b64, prompt)
        blocks_data = self._parse_json_response(vlm_response)

        if not blocks_data:
            print("[VlmOCR] VLM returned no valid result")
            if self.fallback_to_tesseract:
                print("[VlmOCR] Falling back to Tesseract")
                return self.tesseract_ocr.analyze_image(str(image_path))
            return OCRResult(image_width=width, image_height=height)

        # Detect if VLM coords exceed image bounds and compute normalization
        all_coords = [b.get("bbox_2d", [0, 0, 0, 0]) for b in blocks_data if b.get("bbox_2d")]
        if all_coords:
            vlm_x_max = max(c[2] for c in all_coords)
            vlm_y_max = max(c[3] for c in all_coords)
            # If VLM coords exceed image by >10%, apply proportional scaling
            norm_x = width / vlm_x_max if vlm_x_max > width * 1.1 else 1.0
            norm_y = height / vlm_y_max if vlm_y_max > height * 1.1 else 1.0
            if norm_x != 1.0 or norm_y != 1.0:
                print(f"[VlmOCR] Normalizing coords: x*{norm_x:.3f}, y*{norm_y:.3f}")
        else:
            norm_x, norm_y = 1.0, 1.0

        text_blocks = []
        for item in blocks_data:
            text = item.get("text", "").strip()
            if not text:
                continue

            # Support both bbox_2d [x1,y1,x2,y2] and bbox [x,y,w,h]
            bbox = item.get("bbox_2d") or item.get("bbox", [0, 0, 100, 20])
            if len(bbox) == 4:
                if item.get("bbox_2d"):
                    # [x1, y1, x2, y2] format — normalize then clamp
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    x1, x2 = x1 * norm_x, x2 * norm_x
                    y1, y2 = y1 * norm_y, y2 * norm_y
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                else:
                    x1, y1 = float(bbox[0]), float(bbox[1])
                    x2, y2 = x1 + float(bbox[2]), y1 + float(bbox[3])
            else:
                x1, y1, x2, y2 = 0, 0, 100, 20

            polygon = [
                (x1, y1), (x2, y1), (x2, y2), (x1, y2),
            ]
            h = y2 - y1

            font_size = item.get("font_size") or max(h * 0.5, 6)
            font_color = item.get("font_color")
            if font_color in ("null", None, "#000000"):
                font_color = None

            font_name = item.get("font_family")
            if font_name in ("null", None, ""):
                font_name = None

            block = TextBlock(
                text=text,
                polygon=polygon,
                confidence=0.9,
                font_size_px=float(font_size),
                spans=[],
                is_bold=bool(item.get("is_bold", False)),
                is_italic=bool(item.get("is_italic", False)),
                font_weight="bold" if item.get("is_bold") else None,
                font_style="italic" if item.get("is_italic") else None,
                font_color=font_color,
                font_name=font_name,
            )
            text_blocks.append(block)

        print(f"[VlmOCR] full mode: detected {len(text_blocks)} text blocks")
        return OCRResult(
            image_width=width,
            image_height=height,
            text_blocks=text_blocks,
            styles=[],
        )

    # -- VLM API call -------------------------------------------------------

    def _call_vlm(self, image_b64: str, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        if resp.status_code != 200:
            raise RuntimeError(f"VLM API error: {resp.status_code} - {resp.text[:500]}")

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content

    # -- Utilities ----------------------------------------------------------

    @staticmethod
    def _encode_image(img: Image.Image, max_dim: int = 768) -> str:
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    @staticmethod
    def _encode_image_with_size(img: Image.Image, max_dim: int = 768) -> tuple:
        """Encode image and return (base64_str, resized_width, resized_height)."""
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            w, h = int(w * scale), int(h * scale)
            img = img.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii"), w, h

    @staticmethod
    def _polygon_to_bbox(polygon) -> List[int]:
        if not polygon:
            return [0, 0, 100, 20]
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - min(xs)), int(max(ys) - min(ys))
        return [x, y, max(w, 1), max(h, 1)]

    @staticmethod
    def _parse_json_response(text: str) -> Optional[list]:
        text = text.strip()

        # Remove <think>...</think> reasoning blocks (Qwen3.5 thinking mode)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Remove markdown code fences
        fenced = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", text)
        if fenced:
            text = fenced.group(1).strip()

        # Find outermost JSON array with bracket matching
        bracket_start = text.find("[")
        if bracket_start != -1:
            depth = 0
            for i in range(bracket_start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = text[bracket_start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            # Try fixing common issues: trailing commas
                            cleaned = re.sub(r",\s*]", "]", candidate)
                            cleaned = re.sub(r",\s*}", "}", cleaned)
                            try:
                                return json.loads(cleaned)
                            except json.JSONDecodeError:
                                break

        # Try direct parse
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Last resort: try wrapping in array brackets if we see JSON objects
        if text.startswith("{"):
            try:
                result = json.loads("[" + text + "]")
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # Try fixing truncated JSON: find last complete object
        if bracket_start is not None and bracket_start != -1:
            # Find the last complete JSON object by looking for "},"  or "}\n"
            # then close the array
            for end_pattern in ["},", "}\n", "}\t", "}"]:
                last_obj_end = text.rfind(end_pattern)
                if last_obj_end != -1 and last_obj_end > bracket_start:
                    attempt = text[bracket_start:last_obj_end + 1] + "]"
                    attempt = re.sub(r",\s*]", "]", attempt)
                    attempt = re.sub(r",\s*$", "", attempt) + "]" if not attempt.endswith("]") else attempt
                    try:
                        result = json.loads(attempt)
                        if isinstance(result, list) and len(result) > 0:
                            print(f"[VlmOCR] Recovered {len(result)} items from truncated response")
                            return result
                    except json.JSONDecodeError:
                        continue

        print(f"[VlmOCR] Cannot parse VLM response ({len(text)} chars): {text[:300]}...")
        return None
