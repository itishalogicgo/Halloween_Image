import os
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient


API_TOKEN = os.getenv("API_TOKEN", "logicgo@123")
MONGODB_URI = "mongodb+srv://harilogicgo_db_user:g6Zz4M2xWpr3B2VM@cluster0.bnzjt7f.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# MongoDB connection with error handling
try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client.halloween_db
    garments_collection = db.garments
    # Test the connection
    client.admin.command('ping')
    print("MongoDB connection successful")
except Exception as e:
    print(f"MongoDB connection failed: {e}")
    client = None
    db = None
    garments_collection = None

GARMENT_INPUT_DIR = Path("garment_input")
GARMENT_TEMPLATES_DIR = Path("Halloween Dress")  # local recommendations folder
GARMENT_OUTPUT_DIR = Path("garment_output")
HALLOWEEN_OUTPUT_DIR = Path("halloween_output")

GARMENT_INPUT_DIR.mkdir(exist_ok=True)
GARMENT_OUTPUT_DIR.mkdir(exist_ok=True)
HALLOWEEN_OUTPUT_DIR.mkdir(exist_ok=True)
GARMENT_TEMPLATES_DIR.mkdir(exist_ok=True)


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

# Serve local recommendations from /garment_templates/<filename>
app.mount("/garment_templates", StaticFiles(directory=str(GARMENT_TEMPLATES_DIR)), name="garment_templates")

THEMES = [
    {"name": "Witch costume", "prompt": "transform the person into a witch wearing a classic black hat and robe, add a moonlit night background"},
    {"name": "Vampire", "prompt": "make the person look like a vampire with pale skin, red eyes, and a dark cape, gothic castle background"},
    {"name": "Ghost", "prompt": "turn the person into a translucent ghost with a soft glow and misty background"},
    {"name": "Zombie", "prompt": "turn the person into a zombie with subtle decayed features and eerie lighting"},
    {"name": "Skeleton", "prompt": "stylize the person as a glowing skeleton with neon bones, dark background"},
    {"name": "Pumpkin spirit", "prompt": "add a pumpkin-head mask and autumn leaves background with warm cinematic lighting"},
    {"name": "Monster", "prompt": "turn the person into a friendly halloween monster with vibrant colors"},
    {"name": "Pirate", "prompt": "transform the person into a pirate with hat and eyepatch, wooden ship deck background"},
    {"name": "Fairy", "prompt": "make the person a halloween fairy with glowing wings and sparkly particles"},
    {"name": "Cyberpunk", "prompt": "apply a neon cyberpunk halloween theme with glowing accents and city night backdrop"},
]


@app.get("/health")
def health():
    return {"status": "healthy", "services": {"halloween": True, "garment": True}, "version": "2.0"}


@app.get("/themes")
def themes(authorization: Optional[str] = None):
    # Temporarily disable auth for testing
    # _auth(authorization)
    return {"themes": THEMES}


@app.get("/garment/list")
def garment_list(authorization: Optional[str] = None):
    # Temporarily disable auth for testing
    # _auth(authorization)
    
    items = []
    
    # Try to get garments from MongoDB if available
    if garments_collection is not None:
        try:
            for doc in garments_collection.find({}, {"filename": 1, "url": 1}):
                items.append({
                    "filename": doc["filename"],
                    "url": doc["url"]
                })
        except Exception as e:
            print(f"Error reading from MongoDB: {e}")
    
    # If no items in MongoDB, fallback to local files
    if not items:
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        # Prefer local recommendations from 'Halloween Dress' and expose via /garment_templates
        if GARMENT_TEMPLATES_DIR.exists():
            for p in sorted(GARMENT_TEMPLATES_DIR.iterdir()):
                if p.is_file() and p.suffix.lower() in exts:
                    items.append({"filename": p.name, "url": f"/garment_templates/{p.name}"})
        # Fallback: also include any files from garment_input for compatibility
        if GARMENT_INPUT_DIR.exists():
            for p in sorted(GARMENT_INPUT_DIR.iterdir()):
                if p.is_file() and p.suffix.lower() in exts:
                    items.append({"filename": p.name, "url": f"/garment_input/{p.name}"})
    
    return {"garments": items}


@app.get("/preview/garment/{filename}")
def preview_garment(filename: str, authorization: Optional[str] = None):
    # _auth(authorization)
    
    # Try to get from MongoDB first if available
    if garments_collection is not None:
        try:
            doc = garments_collection.find_one({"filename": filename})
            if doc and "image_data" in doc:
                image_data = base64.b64decode(doc["image_data"])
                content_type = doc.get("content_type", "image/webp")
                return Response(content=image_data, media_type=content_type)
        except Exception as e:
            print(f"Error reading from MongoDB: {e}")
    
    # Fallback to file system - check multiple locations
    possible_paths = [
        GARMENT_OUTPUT_DIR / filename,
        GARMENT_TEMPLATES_DIR / filename,
        GARMENT_INPUT_DIR / filename
    ]
    
    for file_path in possible_paths:
        if file_path.exists():
            return FileResponse(file_path, media_type="image/webp")
    
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/download/garment/{filename}")
def download_garment(filename: str, authorization: Optional[str] = None):
    # _auth(authorization)
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
    # _auth(authorization)

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


@app.post("/upload-template")
async def upload_template(
    authorization: Optional[str] = None,
    file: UploadFile = File(...),
    filename: Optional[str] = Form(None),
):
    # _auth(authorization)
    # Save uploaded file to templates directory
    target_filename = filename or file.filename
    if not target_filename:
        raise HTTPException(status_code=400, detail="filename required")
    
    dst = GARMENT_TEMPLATES_DIR / target_filename
    with dst.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    return {"status": "success", "filename": target_filename, "url": f"/garment_templates/{target_filename}"}


@app.post("/halloween/transform")
async def halloween_transform(
    authorization: Optional[str] = None,
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    # _auth(authorization)
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


