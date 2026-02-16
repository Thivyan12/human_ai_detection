import base64
import io
import pickle
import numpy as np
import librosa

# Load trained ML model
with open("artifacts/final_voice_detection_model.pkl", "rb") as f:
    model = pickle.load(f)


# --------------------------------------------------
# Speech Duration Detection (for chunk filtering)
# --------------------------------------------------
def calculate_speech_duration(chunk, sr):
    frame_length = int(0.025 * sr)   # 25 ms frame
    hop_length = int(0.010 * sr)     # 10 ms hop

    # Normalize chunk before measuring energy
    chunk = librosa.util.normalize(chunk)

    # RMS energy per frame
    rms = librosa.feature.rms(
        y=chunk,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # Fixed speech threshold
    SPEECH_THRESHOLD = 0.02

    # Count frames above threshold
    speech_frames = rms > SPEECH_THRESHOLD

    # Convert frame count → seconds
    speech_duration = np.sum(speech_frames) * (hop_length / sr)

    return speech_duration


# --------------------------------------------------
# Chunk Acceptance + Weight Calculation
# --------------------------------------------------
def should_accept_chunk(chunk, sr, min_speech_duration=2.0):
    speech_duration = calculate_speech_duration(chunk, sr)

    # Accept only if enough speech present
    accept = speech_duration >= min_speech_duration

    # Calculate speech ratio for weighting
    chunk_duration = len(chunk) / sr
    speech_ratio = speech_duration / chunk_duration

    # Weight increases with speech content
    weight = min(1.0, speech_ratio / 0.5)

    return accept, speech_duration, weight


# --------------------------------------------------
# Feature Extraction (6 acoustic features)
# --------------------------------------------------
def extract_features(chunk, sr):
    chunk = librosa.util.normalize(chunk)

    # 1. MFCC variability
    mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
    mfcc_var = np.mean(np.std(mfccs, axis=1))

    # 2. ZCR variability
    zcr = librosa.feature.zero_crossing_rate(chunk)[0]
    zcr_var = np.std(zcr)

    # 3. Energy coefficient of variation
    rms = librosa.feature.rms(y=chunk)[0]
    energy_cv = np.std(rms) / (np.mean(rms) + 1e-6)

    # 4. Voiced ratio
    pitches, _ = librosa.piptrack(y=chunk, sr=sr)
    voiced_ratio = np.sum(pitches > 0) / (pitches.size + 1e-6)

    # 5. Spectral contrast variability
    contrast = librosa.feature.spectral_contrast(y=chunk, sr=sr)
    spectral_contrast = np.mean(np.std(contrast, axis=1))

    # 6. Pitch variability
    pitch_values = pitches[pitches > 0]
    pitch_var = np.std(pitch_values) if len(pitch_values) > 10 else 0.0

    return np.array([
        mfcc_var,
        zcr_var,
        energy_cv,
        voiced_ratio,
        spectral_contrast,
        pitch_var
    ])


# --------------------------------------------------
# Split audio into fixed-length chunks
# --------------------------------------------------
def split_chunks(y, sr, duration=5.0):
    size = int(sr * duration)
    return [y[i:i+size] for i in range(0, len(y), size) if len(y[i:i+size]) >= sr]


# --------------------------------------------------
# Aggregation: Normalized Weighted Log-Odds Fusion
# --------------------------------------------------
def normalized_weighted_log_odds(ai_probs, weights):
    ai_probs = np.array(ai_probs)
    weights = np.array(weights)

    # Avoid log(0) and log(1)
    ai_probs = np.clip(ai_probs, 1e-6, 1 - 1e-6)

    # Convert probability → log-odds
    logits = np.log(ai_probs / (1 - ai_probs))

    # Weighted average in logit space
    final_logit = np.sum(weights * logits) / np.sum(weights)

    # Convert back to probability
    final_prob = 1 / (1 + np.exp(-final_logit))

    return float(final_prob)


# --------------------------------------------------
# Final Binary Decision Logic (3-stage)
# --------------------------------------------------
def final_binary_decision(final_prob, chunk_probs):

    # Stage 1 — primary decision
    if final_prob > 0.5:
        return "AI_GENERATED"
    if final_prob < 0.5:
        return "HUMAN"

    # Stage 2 — tie breaker using mean probability
    mean_prob = np.mean(chunk_probs)

    if mean_prob > 0.5:
        return "AI_GENERATED"
    if mean_prob < 0.5:
        return "HUMAN"

    # Stage 3 — absolute tie fallback
    return "HUMAN"


# --------------------------------------------------
# Explanation Generator
# --------------------------------------------------
def generate_explanation(prob):

    if prob > 0.85:
        return "Strong synthetic speech patterns detected."
    if prob > 0.65:
        return "Speech characteristics indicate likely AI generation."
    if prob > 0.55:
        return "Subtle synthetic patterns detected."
    if prob < 0.15:
        return "Strong natural speech characteristics detected."
    if prob < 0.35:
        return "Speech patterns consistent with human voice."

    return "Mixed acoustic signals detected."


# --------------------------------------------------
# Main Inference Pipeline
# --------------------------------------------------
def run_inference(audio_base64: str):

    # Decode base64 audio
    try:
        audio_bytes = base64.b64decode(audio_base64.strip())
    except Exception:
        raise ValueError("Invalid base64 audio")

    # Load waveform at 16kHz mono
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

    # Minimum duration check
    if len(y) < sr * 0.5:
        raise ValueError("Audio too short (minimum 0.5 seconds required)")

    # Split into 5-second chunks
    chunks = split_chunks(y, sr)

    ai_probs = []
    weights = []

    # Process each chunk
    for chunk in chunks:

        # Speech filtering
        accept, speech_dur, weight = should_accept_chunk(chunk, sr)
        if not accept:
            continue

        # Extract features
        features = extract_features(chunk, sr).reshape(1, -1)

        # Predict probability
        probs = model.predict_proba(features)[0]
        ai_prob = float(probs[1])

        ai_probs.append(ai_prob)
        weights.append(weight)

    # If no valid chunks detected
    if len(ai_probs) == 0:
        raise ValueError("No valid speech chunks detected")

    # Aggregate chunk predictions
    final_ai_prob = normalized_weighted_log_odds(ai_probs, weights)

    # Final classification
    classification = final_binary_decision(final_ai_prob, ai_probs)

    # Confidence score = probability of predicted class
    confidence = final_ai_prob if classification == "AI_GENERATED" else 1 - final_ai_prob

    # Explanation text
    explanation = generate_explanation(final_ai_prob)

    # Final response
    return {
        "classification": classification,
        "confidenceScore": round(float(confidence), 4),
        "explanation": explanation
    }
