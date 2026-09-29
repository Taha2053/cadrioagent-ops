from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="CardioAgent API Gateway")

WEB_DIR = Path(__file__).parent / "web"

app.mount(
    "/static",
    StaticFiles(directory=WEB_DIR),
    name="static",
)


@app.get("/")
def home():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze_pcg")
async def analyze_pcg(file: UploadFile = File(...)):
    filename = file.filename or ""

    if Path(filename).suffix.lower() != ".wav":
        raise HTTPException(
            status_code=400,
            detail="Please upload a .wav file.",
        )

    try:
        content = await file.read()
    finally:
        await file.close()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    return {
        "status": "received",
        "filename": filename,
        "size_bytes": len(content),
    }
