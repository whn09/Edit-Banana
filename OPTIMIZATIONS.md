# Edit-Banana Optimization Notes

This fork contains optimizations to the Edit-Banana pipeline for improved text recognition, shape detection, and overall output quality when running locally without the Azure OCR backend.

## Overview

The original Edit-Banana relies on Azure AI OCR for text extraction, which was removed in the open-source release. This fork adds local OCR alternatives and improves several pipeline components.

### Pipeline Flow

```
Input Image
    │
    ├─[1] Text OCR (RapidOCR / VLM / Tesseract)
    │       ├─ Text detection & recognition
    │       ├─ Position snapping (VLM → RapidOCR)
    │       ├─ Font size estimation (spatial hierarchy)
    │       └─ Style detection (bold, color, font family)
    │
    ├─[2] SAM3 Segmentation
    │       └─ Grounding-based element detection
    │
    ├─[3] Shape Processing
    │       ├─ SAM3 shapes (rectangle, ellipse, etc.)
    │       └─ CV fallback detection (background panels, title bars)
    │
    ├─[4] Arrow Processing
    │       └─ Vector arrow detection & conversion
    │
    └─[5] XML Merge
            ├─ Layer-based ordering (Background → Shape → Image → Arrow → Text)
            └─ Final .drawio.xml output
```

## Key Optimizations

### 1. RapidOCR Integration (`modules/text/ocr/rapid_ocr.py`)

Replaced Tesseract with [RapidOCR](https://github.com/RapidAI/RapidOCR) (PaddleOCR models via ONNX Runtime) as the primary OCR engine.

**Why:**
- **Accuracy**: Much better text detection and recognition than Tesseract, especially for mixed-language content and complex layouts
- **Speed**: ~1 second per image (vs 150s for VLM full mode)
- **Position precision**: Pixel-accurate 4-point polygon bounding boxes
- **No GPU required**: Runs on CPU via ONNX Runtime

**Configuration** (`config/config.yaml`):
```yaml
multimodal:
  ocr_engine: "rapidocr"   # Options: rapidocr, vlm, tesseract
```

### 2. VLM OCR Module (`modules/text/ocr/vlm_ocr.py`)

New module for using Vision Language Models (e.g., Qwen-VL) for OCR via OpenAI-compatible API.

**Two modes:**
- `full`: VLM detects both text content and positions. Better text quality but positions need correction.
- `enhance`: Tesseract provides positions, VLM corrects text content and detects styles.

**Features:**
- Coordinate normalization: auto-detects and corrects VLM coordinate system mismatch
- Robust JSON parser: handles markdown fences, trailing commas, truncated responses, `<think>` blocks
- Font style detection: bold, italic, color, font family
- Batch processing in enhance mode (50 blocks per VLM call)

**Configuration:**
```yaml
multimodal:
  ocr_engine: "vlm"
  local_base_url: "http://localhost:11434/v1"
  local_model: "qwen3.5-9b"
  vlm_ocr_mode: "full"     # or "enhance"
  max_tokens: 8000
  timeout: 300
```

### 3. Hybrid Position Snapping (`modules/text/restorer.py`)

When using VLM OCR (which has imprecise positions), the pipeline can snap VLM text blocks to RapidOCR-detected positions:

1. Run VLM for clean text content
2. Run RapidOCR for accurate bounding boxes
3. Match VLM blocks to RapidOCR blocks by text similarity + spatial proximity
4. Use RapidOCR position for matched blocks, keep VLM position for unmatched

This gives the best of both worlds: VLM text quality + pixel-accurate positions.

### 4. Spatial Font Size Estimation (`modules/text/processors/font_size.py`)

The original `height - 1.0` formula doesn't work well for local OCR. New approach uses **spatial hierarchy** to assign font sizes:

| Signal | How it's used |
|--------|--------------|
| Character density (w/ch) | Large w/ch (>13) + long text → section headers (20pt) |
| Bbox height | Height >32px → large labels (18pt) |
| Y position | Bottom 12% → small labels (8pt) |
| Y position | Bottom 25% → medium labels (10pt) |
| Default | Regular body text (10-12pt) |

This creates a visual hierarchy matching the original diagram structure.

### 5. Improved Shape Detection (`modules/basic_shape_processor.py`)

Relaxed CV detection parameters to catch shapes SAM3 misses:

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| `aspect` filter | >4 | >30 | Allow title bars (aspect ~24.5) |
| `max_area_ratio` | 0.5 | 0.95 | Allow large background panels |
| `border_contrast` | 15 | 5 | Detect low-contrast borders |
| `min_area_ratio` | 0.07 | 0.02 | Detect smaller shapes |
| `min_rectangularity` | 0.7 | 0.5 | More lenient shape matching |
| `enabled_methods` | contour only | +region | Additional detection method |

**Result:** Detects background panels, colored quadrant backgrounds, and yellow title bars that were previously missed.

### 6. Text Block Merging (`modules/text/restorer.py`)

Optional step to merge Tesseract's word-level fragments into line-level blocks using union-find:
- Same-line detection: Y-center within 0.6× avg height
- Height ratio check: within 2× ratio
- Horizontal gap: within 1.5× avg height

Enabled automatically when using Tesseract/RapidOCR as standalone OCR.

## Layer System

Edit-Banana uses a 5-level Z-ordering system defined in `modules/data_types.py`:

```python
class LayerLevel(Enum):
    BACKGROUND = 0      # Background panels, title bars
    BASIC_SHAPE = 1     # Rectangles, ellipses
    IMAGE = 2           # Icons, pictures
    ARROW = 3           # Arrows, connectors
    TEXT = 4            # Text (topmost)
```

The XML merger sorts fragments by `layer_level` ascending, then by area descending within each layer. This ensures proper stacking: backgrounds at bottom, text on top.

## Installation

```bash
# Base dependencies (same as original)
pip install -r requirements.txt

# RapidOCR (recommended OCR engine)
pip install rapidocr_onnxruntime

# Optional: VLM OCR (requires a running VLM server)
# Start Qwen-VL with vLLM:
# python -m vllm.entrypoints.openai.api_server \
#   --model Qwen/Qwen2.5-VL-7B-Instruct --port 11434 \
#   --enforce-eager --max-model-len 16384
```

## Usage

```bash
# Default: RapidOCR (fast, accurate positions)
python main.py -i input/test.png

# VLM mode (better text quality, slower)
# Set ocr_engine in config.yaml, or it auto-selects VLM when configured
python main.py -i input/test.png
```

## Comparison

| Aspect | Original (Azure) | RapidOCR | VLM (Qwen) |
|--------|-------------------|----------|-------------|
| Speed | ~5s | ~1s | ~150s |
| Text accuracy | Excellent | Good | Good |
| Position accuracy | Excellent | Excellent | Fair (needs snap) |
| Font size | From API | Spatial heuristic | Spatial heuristic |
| Bold/style | From API | Not detected | Detected by VLM |
| Languages | All | CJK + Latin | All |
| Cost | Azure API fees | Free (local) | Free (local GPU) |
| GPU required | No | No | Yes (~12GB VRAM) |

## Modified Files

- `main.py` — Added RapidOCR/VLM engine selection
- `config/config.yaml` — Added `ocr_engine` option
- `modules/text/ocr/rapid_ocr.py` — **New**: RapidOCR adapter
- `modules/text/ocr/vlm_ocr.py` — **New**: VLM OCR module
- `modules/text/restorer.py` — Hybrid position snapping, text merging
- `modules/text/processors/font_size.py` — Spatial font size estimation
- `modules/basic_shape_processor.py` — Relaxed CV detection parameters
