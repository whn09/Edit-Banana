#!/usr/bin/env python3
"""
FastAPI Backend Server — web service entry for Edit Banana.

Provides:
  - Static frontend at /
  - Upload & conversion API at /convert
  - File download at /download
  - Health check at /health

Run with: python server_pa.py
Server runs at http://0.0.0.0:8000
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="Edit Banana API",
    description="Universal Content Re-Editor — image/PDF to editable DrawIO or PPTX",
    version="1.0.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the frontend."""
    index_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Edit Banana</h1><p>Frontend not found. Place index.html in frontend/</p>")


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    """Upload image or PDF and return editable output (DrawIO XML)."""
    name = file.filename or ""
    ext = Path(name).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".pdf", ".bmp", ".tiff", ".webp"}:
        raise HTTPException(400, "Unsupported format. Use image or PDF.")

    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    if not os.path.exists(config_path):
        raise HTTPException(503, "Server not configured (missing config/config.yaml)")

    try:
        from main import load_config, Pipeline
        import tempfile
        import shutil

        config = load_config()
        output_dir = config.get("paths", {}).get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=os.path.join(PROJECT_ROOT, "input")) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            pipeline = Pipeline(config)
            result_path = pipeline.process_image(
                tmp_path,
                output_dir=output_dir,
                with_refinement=False,
                with_text=True,
            )
            if not result_path or not os.path.exists(result_path):
                raise HTTPException(500, "Conversion failed — no output generated")
            return {"success": True, "output_path": result_path, "filename": os.path.basename(result_path)}
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/download")
def download(path: str = Query(...)):
    """Download a result file."""
    # Security: only allow files under output directory
    abs_path = os.path.abspath(path)
    output_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "output"))
    if not abs_path.startswith(output_dir):
        raise HTTPException(403, "Access denied")
    if not os.path.exists(abs_path):
        raise HTTPException(404, "File not found")
    return FileResponse(
        abs_path,
        media_type="application/xml",
        filename=os.path.basename(abs_path),
    )


# Serve static files (logo, demo images, etc.)
static_dir = os.path.join(PROJECT_ROOT, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


def main():
    print("🍌 Edit Banana Server starting at http://0.0.0.0:8000")
    print("   Frontend: http://localhost:8000/")
    print("   API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
