# inference.py
import base64
import io
import pickle
import numpy as np
import librosa

# ---------------- LOAD MODEL ----------------
with open("artifacts/best_model_baseline.pkl", "rb") as f:
    model = pickle.load(f)


# ---------------- FEATURE EXTRACTION ----------------
def extract_features_from_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Decode MP3 bytes → waveform → extract 4 features
    """
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

    if len(y) == 0:
        raise ValueError("Empty or corrupted audio")

    # 1. MFCC variability
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_var = np.std(np.std(mfccs, axis=1))

    # 2. ZCR variability
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_var = np.std(zcr)

    # 3. Energy CV
    rms = librosa.feature.rms(y=y)[0]
    energy_cv = np.std(rms) / (np.mean(rms) + 1e-8)

    # 4. Voiced-frame ratio
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    voiced_ratio = np.sum(flatness < 0.1) / len(flatness)

    return np.array([mfcc_var, zcr_var, energy_cv, voiced_ratio])


# ---------------- EXPLANATION LOGIC ----------------
def generate_explanation(raw_features: np.ndarray, confidence: float, classification: str) -> str:
    """
    Confidence-aware, human-readable explanation
    (no feature leakage, no external stats)
    """

    # Low-confidence case
    if confidence < 0.6:
        return (
            "The audio exhibits mixed acoustic characteristics and does not "
            "strongly align with typical human or AI-generated speech patterns"
        )

    # Higher confidence explanations
    if classification == "AI_GENERATED":
        return (
            "Unnatural pitch consistency and robotic speech patterns detected"
        )
    else:
        return (
            "Natural variations in pitch, energy, and speech flow detected"
        )


# ---------------- MAIN INFERENCE ----------------
def run_inference(audio_base64: str) -> dict:
    """
    Base64 MP3 → prediction result
    """
    # Decode Base64
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise ValueError("Invalid Base64 audio input")

    # Feature extraction
    raw_features = extract_features_from_audio_bytes(audio_bytes)

    # Model prediction
    probabilities = model.predict_proba(raw_features.reshape(1, -1))[0]

    # ⚠️ Ensure label mapping matches your training
    # Index 1 → AI, Index 0 → HUMAN
    if probabilities[1] > probabilities[0]:
        classification = "AI_GENERATED"
        confidence = float(probabilities[1])
    else:
        classification = "HUMAN"
        confidence = float(probabilities[0])

    explanation = generate_explanation(
        raw_features=raw_features,
        confidence=confidence,
        classification=classification
    )

    return {
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
