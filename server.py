import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware


API_TOKEN = os.getenv("API_TOKEN", "logicgo@123")
GARMENT_INPUT_DIR = Path("garment_input")
GARMENT_OUTPUT_DIR = Path("garment_output")
HALLOWEEN_OUTPUT_DIR = Path("halloween_output")

GARMENT_INPUT_DIR.mkdir(exist_ok=True)
GARMENT_OUTPUT_DIR.mkdir(exist_ok=True)
HALLOWEEN_OUTPUT_DIR.mkdir(exist_ok=True)


def _auth(authorization: Optional[str]):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        scheme, token = authorization.split(" ", 1)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    if scheme.lower() != "bearer" or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


app = FastAPI(title="Halloween + Virtual Try-On API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "healthy", "services": {"halloween": True, "garment": True}}


@app.get("/garment/list")
def garment_list(authorization: Optional[str] = None):
    _auth(authorization)
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    items = []
    for p in sorted(GARMENT_INPUT_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            items.append({"filename": p.name, "url": f"/garment_input/{p.name}"})
    return {"garments": items}


@app.get("/preview/garment/{filename}")
def preview_garment(filename: str, authorization: Optional[str] = None):
    _auth(authorization)
    file_path = GARMENT_OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="image/webp")


@app.get("/download/garment/{filename}")
def download_garment(filename: str, authorization: Optional[str] = None):
    _auth(authorization)
    file_path = GARMENT_OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)


@app.post("/garment/transform")
async def garment_transform(
    authorization: Optional[str] = None,
    source_file: UploadFile = File(...),
    garment_filename: Optional[str] = Form(None),
    garment_file: Optional[UploadFile] = File(None),
):
    _auth(authorization)

    # If custom garment file provided, save it and use as the chosen garment
    if garment_file is not None:
        # Save custom garment to input dir
        dst = GARMENT_INPUT_DIR / garment_file.filename
        with dst.open("wb") as f:
            while chunk := await garment_file.read(1024 * 1024):
                f.write(chunk)
        garment_filename = garment_file.filename

    if not garment_filename:
        raise HTTPException(status_code=400, detail="Provide garment_filename or upload garment_file")

    # Save source image (for demonstration; real pipeline would process it)
    src_path = GARMENT_OUTPUT_DIR / ("preview_" + os.path.splitext(source_file.filename or "src")[0] + ".webp")
    with src_path.open("wb") as f:
        while chunk := await source_file.read(1024 * 1024):
            f.write(chunk)

    # In a real implementation, run try-on pipeline here using source_file + garment_filename
    # For now, echo a fake filename to match the documented contract
    filename = src_path.name
    return JSONResponse({"status": "success", "preview_url": f"/garment_output/{filename}", "filename": filename})


@app.post("/halloween/transform")
async def halloween_transform(
    authorization: Optional[str] = None,
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    _auth(authorization)
    # Just save uploaded file as a stub response
    dst = HALLOWEEN_OUTPUT_DIR / (os.path.splitext(file.filename or "out")[0] + ".webp")
    with dst.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    return {"status": "success", "url": f"/halloween_output/{dst.name}"}


if __name__ == "__main__":
    # Run with: python3 server.py
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


