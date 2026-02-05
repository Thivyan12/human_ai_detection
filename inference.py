# inference.py
import base64
import io
import pickle
import numpy as np
import librosa

# ---------------- LOAD MODEL ----------------
with open("artifacts/final_updated_model.pkl", "rb") as f:
    model = pickle.load(f)


# ---------------- FEATURE EXTRACTION ----------------
def extract_features_from_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Decode audio bytes → waveform → extract 6 features
    """
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

    if len(y) == 0:
        raise ValueError("Empty or corrupted audio")

    # Loudness normalize
    y = librosa.util.normalize(y)

    # 1. MFCC variability
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_variability = np.mean(np.std(mfccs, axis=1))

    # 2. ZCR variability
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_variability = np.std(zcr)

    # 3. Energy CV
    rms = librosa.feature.rms(y=y)[0]
    energy_cv = np.std(rms) / (np.mean(rms) + 1e-6)

    # 4. Voiced ratio (pitch presence)
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    voiced_frames = np.sum(pitches > 0)
    total_frames = pitches.size
    voiced_ratio = voiced_frames / (total_frames + 1e-6)

    # 5. Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast = np.mean(np.std(contrast, axis=1))

    # 6. Pitch variability
    pitch_values = pitches[pitches > 0]
    if len(pitch_values) > 10:
        pitch_variability = np.std(pitch_values)
    else:
        pitch_variability = 0.0

    return np.array([mfcc_variability, zcr_variability, energy_cv, voiced_ratio, spectral_contrast, pitch_variability])


def chunk_audio(y: np.ndarray, sr: int, chunk_duration: float = 5.0) -> list:
    """
    Split audio into chunks of specified duration (default 5 seconds)
    If audio is shorter than chunk_duration, return it as a single chunk
    """
    chunk_samples = int(chunk_duration * sr)
    chunks = []
    
    # If audio is shorter than chunk duration, treat entire audio as one chunk
    if len(y) <= chunk_samples:
        chunks.append(y)
    else:
        # Split into multiple chunks
        for i in range(0, len(y), chunk_samples):
            chunk = y[i:i + chunk_samples]
            # Only include chunks that are at least 1 second long
            if len(chunk) >= sr:
                chunks.append(chunk)
    
    return chunks


def extract_features_from_chunks(audio_bytes: bytes) -> list:
    """
    Decode audio → split into 5-sec chunks → extract features from each chunk
    Returns list of feature arrays (one per chunk)
    """
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

    if len(y) == 0:
        raise ValueError("Empty or corrupted audio")
    
    # Minimum 0.5 seconds required for reliable feature extraction
    if len(y) < sr * 0.5:
        raise ValueError("Audio too short (minimum 0.5 seconds required)")

    # Split into 5-second chunks
    chunks = chunk_audio(y, sr, chunk_duration=5.0)
    
    if len(chunks) == 0:
        raise ValueError("Audio processing error")

    all_features = []
    
    for chunk in chunks:
        # Loudness normalize
        chunk = librosa.util.normalize(chunk)

        # 1. MFCC variability
        mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
        mfcc_variability = np.mean(np.std(mfccs, axis=1))

        # 2. ZCR variability
        zcr = librosa.feature.zero_crossing_rate(chunk)[0]
        zcr_variability = np.std(zcr)

        # 3. Energy CV
        rms = librosa.feature.rms(y=chunk)[0]
        energy_cv = np.std(rms) / (np.mean(rms) + 1e-6)

        # 4. Voiced ratio (pitch presence)
        pitches, mags = librosa.piptrack(y=chunk, sr=sr)
        voiced_frames = np.sum(pitches > 0)
        total_frames = pitches.size
        voiced_ratio = voiced_frames / (total_frames + 1e-6)

        # 5. Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=chunk, sr=sr)
        spectral_contrast = np.mean(np.std(contrast, axis=1))

        # 6. Pitch variability
        pitch_values = pitches[pitches > 0]
        if len(pitch_values) > 10:
            pitch_variability = np.std(pitch_values)
        else:
            pitch_variability = 0.0

        all_features.append([mfcc_variability, zcr_variability, energy_cv, voiced_ratio, spectral_contrast, pitch_variability])
    
    return all_features


# ---------------- EXPLANATION LOGIC ----------------
def generate_explanation(raw_features: np.ndarray, confidence: float, classification: str) -> str:
    """
    Feature-based, human-readable explanation
    """
    mfcc_var, zcr_var, energy_cv, voiced_ratio, spectral_contrast, pitch_var = raw_features

    # Low-confidence case
    if confidence < 0.6:
        return (
            "The audio exhibits mixed acoustic characteristics that do not strongly align with typical human or AI-generated speech patterns."
        )

    # Feature-based explanations
    if classification == "AI_GENERATED":
        # Determine primary indicators
        indicators = []
        if mfcc_var < 2.0:
            indicators.append("unusually consistent pitch patterns")
        if zcr_var < 0.02:
            indicators.append("uniform voice texture")
        if energy_cv < 0.3:
            indicators.append("steady energy levels")
        if voiced_ratio > 0.85:
            indicators.append("highly regular vocal framing")
        if spectral_contrast < 15.0:
            indicators.append("flat frequency distribution")
        if pitch_var < 50.0:
            indicators.append("robotic pitch consistency")
        
        if indicators:
            return f"AI-generated speech detected due to {' and '.join(indicators[:2])} suggesting synthetic production."
        else:
            return "AI-generated speech detected with robotic consistency in voice characteristics."
    
    else:  # HUMAN
        # Determine primary indicators
        indicators = []
        if mfcc_var > 3.0:
            indicators.append("natural pitch variations")
        if zcr_var > 0.04:
            indicators.append("organic voice texture changes")
        if energy_cv > 0.5:
            indicators.append("dynamic energy fluctuations")
        if voiced_ratio < 0.75:
            indicators.append("natural speech rhythm")
        if spectral_contrast > 20.0:
            indicators.append("rich frequency dynamics")
        if pitch_var > 100.0:
            indicators.append("natural pitch modulation")
        
        if indicators:
            return f"Human speech detected with {' and '.join(indicators[:2])} typical of natural voice production."
        else:
            return "Human speech detected with natural variations in voice characteristics."


def run_inference(audio_base64: str) -> dict:
    """
    Base64 audio → prediction result (probability-based aggregation over chunks)
    """
    # Decode Base64 (robust to newlines/spaces)
    try:
        audio_base64 = audio_base64.strip().replace("\n", "").replace("\r", "")
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        raise ValueError(f"Invalid Base64 audio input: {str(e)}")

    # Feature extraction with chunking
    chunk_features = extract_features_from_chunks(audio_bytes)

    ai_probabilities = []
    human_probabilities = []
    
    for features in chunk_features:
        features_array = np.array(features).reshape(1, -1)

        # Predict probabilities: [P(HUMAN), P(AI)]
        probs = model.predict_proba(features_array)[0]

        human_probabilities.append(float(probs[0]))
        ai_probabilities.append(float(probs[1]))

    # Average probabilities across all chunks
    avg_human_prob = float(np.mean(human_probabilities))
    avg_ai_prob = float(np.mean(ai_probabilities))

    # Calculate variance/uncertainty across chunks
    human_std = np.std(human_probabilities)
    ai_std = np.std(ai_probabilities)

    # Final classification based on higher average probability
    if avg_ai_prob > avg_human_prob:
        classification = "AI_GENERATED"
        raw_confidence = avg_ai_prob
        uncertainty = float(ai_std)
    else:
        classification = "HUMAN"
        raw_confidence = avg_human_prob
        uncertainty = float(human_std)

    # Apply calibration to avoid extreme confidence scores
    # Calibration formula: adjusted = 0.5 + (raw - 0.5) * scaling_factor - uncertainty_penalty
    scaling_factor = 0.8  # Reduces extreme predictions
    uncertainty_penalty = uncertainty * 0.3  # Penalize inconsistent predictions

    confidence = 0.5 + (raw_confidence - 0.5) * scaling_factor - uncertainty_penalty

    # Ensure confidence stays in reasonable bounds [0.51, 0.95]
    confidence = max(0.51, min(0.95, confidence))

    # Average features for explanation
    avg_features = np.mean(chunk_features, axis=0)

    explanation = generate_explanation(
        raw_features=avg_features,
        confidence=confidence,
        classification=classification
    )

    return {
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
