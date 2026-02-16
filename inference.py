# inference.py
import base64
import io
import pickle
import numpy as np
import librosa


# ==================== LOAD MODEL ====================
with open("artifacts/final_voice_detection_model.pkl", "rb") as f:
    model = pickle.load(f)


# =========================================================
# =============== CHUNK SELECTION (FIXED THRESHOLD) =======
# =========================================================

def calculate_speech_duration(chunk, sr):
    """
    PRODUCTION-READY speech duration calculator
    Uses fixed threshold for consistency across chunks
    
    Args:
        chunk: Audio chunk (numpy array)
        sr: Sample rate
    
    Returns:
        speech_duration: Duration of speech content in seconds
    """
    frame_length = int(0.025 * sr)  # 25 ms
    hop_length = int(0.010 * sr)    # 10 ms
    
    # Normalize chunk first (CRITICAL!)
    chunk = librosa.util.normalize(chunk)
    
    # Calculate RMS energy per frame
    rms = librosa.feature.rms(
        y=chunk,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    # Fixed threshold (empirically validated for normalized audio)
    SPEECH_THRESHOLD = 0.02
    
    # Count speech frames
    speech_frames = rms > SPEECH_THRESHOLD
    speech_duration = np.sum(speech_frames) * (hop_length / sr)
    
    return speech_duration


def should_accept_chunk(chunk, sr, min_speech_duration=2.0):
    """
    Accept chunk if it has >= 2 seconds of speech
    
    Args:
        chunk: Audio chunk
        sr: Sample rate
        min_speech_duration: Minimum speech required (default 2.0 sec)
    
    Returns:
        accept: Whether to accept this chunk
        speech_duration: Actual speech duration
        weight: Quality weight for aggregation
    """
    speech_duration = calculate_speech_duration(chunk, sr)
    
    # Decision rule
    accept = speech_duration >= min_speech_duration
    
    # Calculate weight based on speech content
    chunk_duration = len(chunk) / sr
    speech_ratio = speech_duration / chunk_duration
    weight = min(1.0, speech_ratio / 0.5)  # Full weight at 50% speech
    
    return accept, speech_duration, weight


# =========================================================
# ================= FEATURE EXTRACTION ====================
# =========================================================

def extract_features(chunk, sr):
    """
    Extract 6 acoustic features from a single chunk
    
    Features:
    1. MFCC variability
    2. ZCR variability
    3. Energy CV (Coefficient of Variation)
    4. Voiced ratio
    5. Spectral contrast
    6. Pitch variability
    
    Args:
        chunk: Audio chunk
        sr: Sample rate
    
    Returns:
        features: 6-element feature vector
    """
    # Normalize chunk
    chunk = librosa.util.normalize(chunk)
    
    # 1. MFCC variability
    mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
    mfcc_var = np.mean(np.std(mfccs, axis=1))
    
    # 2. ZCR variability
    zcr = librosa.feature.zero_crossing_rate(chunk)[0]
    zcr_var = np.std(zcr)
    
    # 3. Energy CV (Coefficient of Variation)
    rms = librosa.feature.rms(y=chunk)[0]
    energy_cv = np.std(rms) / (np.mean(rms) + 1e-6)
    
    # 4. Voiced ratio (pitch presence)
    pitches, _ = librosa.piptrack(y=chunk, sr=sr)
    voiced_ratio = np.sum(pitches > 0) / (pitches.size + 1e-6)
    
    # 5. Spectral contrast
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


# =========================================================
# ==================== CHUNKING ===========================
# =========================================================

def split_chunks(y, sr, duration=5.0):
    """
    Split audio into chunks of specified duration (default 5 seconds)
    
    Args:
        y: Audio waveform
        sr: Sample rate
        duration: Duration of each chunk in seconds
    
    Returns:
        chunks: List of audio chunks
    """
    size = int(sr * duration)
    return [y[i:i+size] for i in range(0, len(y), size) if len(y[i:i+size]) >= sr]


# =========================================================
# =================== AGGREGATION =========================
# =========================================================

def normalized_weighted_log_odds(ai_probs, weights):
    """
    Normalized Weighted Log-Odds Fusion
    
    Mathematically principled approach that:
    1. Respects nonlinear probability scale
    2. Weights evidence by chunk quality
    3. Normalizes to prevent length bias
    4. Avoids correlation inflation
    
    Args:
        ai_probs: List of AI probabilities per chunk
        weights: Corresponding chunk weights
    
    Returns:
        final_prob: Aggregated AI probability
    """
    ai_probs = np.array(ai_probs)
    weights = np.array(weights)
    
    # Clip probabilities to avoid log(0) or log(1)
    ai_probs = np.clip(ai_probs, 1e-6, 1 - 1e-6)
    
    # Convert to log-odds (evidence space)
    logits = np.log(ai_probs / (1 - ai_probs))
    
    # Normalized weighted aggregation
    final_logit = np.sum(weights * logits) / np.sum(weights)
    
    # Convert back to probability space
    final_prob = 1 / (1 + np.exp(-final_logit))
    
    return float(final_prob)


# =========================================================
# ================= DECISION LOGIC ========================
# =========================================================

def final_binary_decision(final_prob, chunk_probs):
    """
    Deterministic binary classification with three-stage tie-breaking
    
    Stage 1: Use aggregated probability (log-odds fusion)
    Stage 2: Use mean probability (simple average)
    Stage 3: Default to HUMAN (safer bias)
    
    Args:
        final_prob: Aggregated AI probability from log-odds fusion
        chunk_probs: Original chunk-level AI probabilities
    
    Returns:
        classification: "AI_GENERATED" or "HUMAN"
    """
    # Stage 1: Primary Decision
    if final_prob > 0.5:
        return "AI_GENERATED"
    
    if final_prob < 0.5:
        return "HUMAN"
    
    # Stage 2: Tie Resolver (final_prob == 0.5)
    # Use simple mean as secondary evidence
    mean_prob = np.mean(chunk_probs)
    
    if mean_prob > 0.5:
        return "AI_GENERATED"
    
    if mean_prob < 0.5:
        return "HUMAN"
    
    # Stage 3: Absolute Tie Fallback
    # Default to HUMAN (safer classification bias)
    return "HUMAN"


# =========================================================
# ================== EXPLANATION ENGINE ===================
# =========================================================

def generate_explanation(prob):
    """
    Generate human-readable explanation based on final probability
    
    Args:
        prob: Final AI probability
    
    Returns:
        explanation: Human-readable explanation string
    """
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


# =========================================================
# ===================== MAIN PIPELINE =====================
# =========================================================

def run_inference(audio_base64: str):
    """
    Complete inference pipeline: Base64 → Classification
    
    Pipeline:
    1. Decode Base64 → audio bytes
    2. Load audio → waveform
    3. Chunk audio (5-sec chunks)
    4. Filter chunks (≥2 sec speech)
    5. Extract features (6 features per chunk)
    6. Get model predictions
    7. Aggregate using normalized weighted log-odds
    8. Apply three-stage decision logic
    9. Calculate confidence (probability of predicted class)
    10. Generate explanation
    
    Args:
        audio_base64: Base64-encoded MP3 audio
    
    Returns:
        result: Dictionary with classification, confidence, explanation
    """
    # ===== STEP 1: Decode Base64 =====
    try:
        audio_bytes = base64.b64decode(audio_base64.strip())
    except Exception:
        raise ValueError("Invalid base64 audio")
    
    # ===== STEP 2: Load Waveform =====
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    
    if len(y) < sr * 0.5:
        raise ValueError("Audio too short (minimum 0.5 seconds required)")
    
    # ===== STEP 3: Split into Chunks =====
    chunks = split_chunks(y, sr)
    
    ai_probs = []
    weights = []
    
    # ===== STEP 4-6: Process Each Chunk =====
    for chunk in chunks:
        
        # Check if chunk has sufficient speech
        accept, speech_dur, weight = should_accept_chunk(chunk, sr)
        
        if not accept:
            continue  # Skip chunks with insufficient speech
        
        # Extract 6 features
        features = extract_features(chunk, sr).reshape(1, -1)
        
        # Get model prediction
        probs = model.predict_proba(features)[0]
        ai_prob = float(probs[1])
        
        # Store results
        ai_probs.append(ai_prob)
        weights.append(weight)
    
    # Validate that we have at least one valid chunk
    if len(ai_probs) == 0:
        raise ValueError("No valid speech chunks detected")
    
    # ===== STEP 7: Aggregate Using Normalized Weighted Log-Odds =====
    final_ai_prob = normalized_weighted_log_odds(ai_probs, weights)
    
    # ===== STEP 8: Classification (Three-Stage Decision) =====
    classification = final_binary_decision(final_ai_prob, ai_probs)
    
    # ===== STEP 9: Confidence (Probability of Predicted Class) =====
    confidence = final_ai_prob if classification == "AI_GENERATED" else 1 - final_ai_prob
    
    # ===== STEP 10: Generate Explanation =====
    explanation = generate_explanation(final_ai_prob)
    
    
    return {
        "classification": classification,
        "confidenceScore": round(float(confidence), 4),
        "explanation": explanation
    }

