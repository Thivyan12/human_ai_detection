from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from inference import run_inference

# Initialize FastAPI app
app = FastAPI(title="AI Voice Detection API")

# Static API key for request authentication
API_KEY = "sk_test_123456789"


# Request body schema definition
class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


# Runs once when server starts
@app.on_event("startup")
def startup_event():
    print("✅ AI Voice Detection API is live")


# Main API endpoint
@app.post("/api/voice-detection")
def voice_detection(
    request: VoiceDetectionRequest,
    x_api_key: str = Header(None)
):

    # Validate API key
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key or malformed request"
        )

    # Only MP3 format supported
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail="Only MP3 audio format is supported"
        )

    # Ensure audio data is present
    if not request.audioBase64:
        raise HTTPException(
            status_code=400,
            detail="audioBase64 field is required"
        )

    # Run ML inference pipeline
    try:
        result = run_inference(request.audioBase64)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # Structured API response
    return {
        "status": "success",
        "language": request.language,
        "classification": result["classification"],
        "confidenceScore": result["confidenceScore"],
        "explanation": result["explanation"]
    }
