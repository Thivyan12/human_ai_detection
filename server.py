# server.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from inference import run_inference

app = FastAPI(title="AI Voice Detection API")

API_KEY = "sk_test_123456789"

# ---------------- REQUEST SCHEMA ----------------
class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.on_event("startup")
def startup_event():
    print("✅ AI Voice Detection API is live")
# server.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from inference import run_inference

app = FastAPI(title="AI Voice Detection API")

API_KEY = "sk_test_123456789"

# ---------------- REQUEST SCHEMA ----------------
class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.on_event("startup")
def startup_event():
    print("✅ AI Voice Detection API is live")


# ---------------- API HANDLER ----------------
@app.post("/api/voice-detection")
def voice_detection(
    request: VoiceDetectionRequest,
    x_api_key: str = Header(None)
):
    # API key validation
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key or malformed request"
        )

    # Input validation
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail="Only MP3 audio format is supported"
        )

    if not request.audioBase64:
        raise HTTPException(
            status_code=400,
            detail="audioBase64 field is required"
        )

    try:
        result = run_inference(request.audioBase64)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # FINAL RESPONSE (MATCHES SPEC EXACTLY)
# ---------------- API HANDLER ----------------
@app.post("/api/voice-detection")
def voice_detection(
    request: VoiceDetectionRequest,
    x_api_key: str = Header(None)
):
    # API key validation
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key or malformed request"
        )

    # Input validation
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail="Only MP3 audio format is supported"
        )

    if not request.audioBase64:
        raise HTTPException(
            status_code=400,
            detail="audioBase64 field is required"
        )

    try:
        result = run_inference(request.audioBase64)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # FINAL RESPONSE (MATCHES SPEC EXACTLY)
    return {
        "status": "success",
        "language": request.language,
        "classification": result["classification"],
        "confidenceScore": result["confidenceScore"],
        "explanation": result["explanation"]
        "status": "success",
        "language": request.language,
        "classification": result["classification"],
        "confidenceScore": result["confidenceScore"],
        "explanation": result["explanation"]
    }
