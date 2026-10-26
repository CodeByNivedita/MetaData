import sys
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from pipeline import predict 

app = FastAPI(title="Rental Agreement Extractor")

ALLOWED = [".docx", ".png", ".jpg", ".jpeg"]


@app.get("/")
def home():
    return {"message": "POST a file to /extract"}


@app.post("/extract")
async def extract(file: UploadFile):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"sirf ye files chalti hain: {ALLOWED}")

        # asli naam se save karte hain, taaki dataset images ka OCR cache mil jaye
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = predict(tmp_path)
    except ImportError:
        raise HTTPException(503, "OCR for new images needs a GPU with torch and transformers. Try a .docx or a dataset image.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"file": file.filename, **result}