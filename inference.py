
import base64
import io
import pickle
import numpy as np
import librosa

with open("artifacts/final_voice_detection_model.pkl", "rb") as f:
    model = pickle.load(f)

def calculate_speech_duration(chunk, sr):
    frame_length = int(0.025 * sr)  # 25 ms
    hop_length = int(0.010 * sr)    # 10 ms
    
    chunk = librosa.util.normalize(chunk)
    
    rms = librosa.feature.rms(
        y=chunk,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    SPEECH_THRESHOLD = 0.02
    
    speech_frames = rms > SPEECH_THRESHOLD
    speech_duration = np.sum(speech_frames) * (hop_length / sr)
    
    return speech_duration

def should_accept_chunk(chunk, sr, min_speech_duration=2.0):

    speech_duration = calculate_speech_duration(chunk, sr)
    accept = speech_duration >= min_speech_duration
    chunk_duration = len(chunk) / sr
    speech_ratio = speech_duration / chunk_duration
    weight = min(1.0, speech_ratio / 0.5)  # Full weight at 50% speech
    return accept, speech_duration, weight


def extract_features(chunk, sr):

    chunk = librosa.util.normalize(chunk)    
    mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
    mfcc_var = np.mean(np.std(mfccs, axis=1))
    zcr = librosa.feature.zero_crossing_rate(chunk)[0]
    zcr_var = np.std(zcr)
    rms = librosa.feature.rms(y=chunk)[0]
    energy_cv = np.std(rms) / (np.mean(rms) + 1e-6)
    pitches, _ = librosa.piptrack(y=chunk, sr=sr)
    voiced_ratio = np.sum(pitches > 0) / (pitches.size + 1e-6)
    
    contrast = librosa.feature.spectral_contrast(y=chunk, sr=sr)
    spectral_contrast = np.mean(np.std(contrast, axis=1))
    
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

def split_chunks(y, sr, duration=5.0):
    size = int(sr * duration)
    return [y[i:i+size] for i in range(0, len(y), size) if len(y[i:i+size]) >= sr]

def normalized_weighted_log_odds(ai_probs, weights):
    ai_probs = np.array(ai_probs)
    weights = np.array(weights)
    ai_probs = np.clip(ai_probs, 1e-6, 1 - 1e-6)
    logits = np.log(ai_probs / (1 - ai_probs))
    final_logit = np.sum(weights * logits) / np.sum(weights)
    final_prob = 1 / (1 + np.exp(-final_logit))
    
    return float(final_prob)


def final_binary_decision(final_prob, chunk_probs):
    if final_prob > 0.5:
        return "AI_GENERATED"
    
    if final_prob < 0.5:
        return "HUMAN"

    mean_prob = np.mean(chunk_probs)
    
    if mean_prob > 0.5:
        return "AI_GENERATED"
    
    if mean_prob < 0.5:
        return "HUMAN"
        return "HUMAN"


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



def run_inference(audio_base64: str):

    try:
        audio_bytes = base64.b64decode(audio_base64.strip())
    except Exception:
        raise ValueError("Invalid base64 audio")
    
    
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    
    if len(y) < sr * 0.5:
        raise ValueError("Audio too short (minimum 0.5 seconds required)")
    
    
    chunks = split_chunks(y, sr)
    
    ai_probs = []
    weights = []
    
    
    for chunk in chunks:
        
        # Check if chunk has sufficient speech
        accept, speech_dur, weight = should_accept_chunk(chunk, sr)
        
        if not accept:
            continue  # Skip chunks with insufficient speech
        
        
        features = extract_features(chunk, sr).reshape(1, -1)
        
        
        probs = model.predict_proba(features)[0]
        ai_prob = float(probs[1])
        
        
        ai_probs.append(ai_prob)
        weights.append(weight)
    
    
    if len(ai_probs) == 0:
        raise ValueError("No valid speech chunks detected")
    
    
    final_ai_prob = normalized_weighted_log_odds(ai_probs, weights)
    
    
    classification = final_binary_decision(final_ai_prob, ai_probs)
    
    
    confidence = final_ai_prob if classification == "AI_GENERATED" else 1 - final_ai_prob
    
    
    explanation = generate_explanation(final_ai_prob)
    
    
    return {
        "classification": classification,
        "confidenceScore": round(float(confidence), 4),
        "explanation": explanation
    }

