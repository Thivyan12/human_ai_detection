# AI Voice Detection System

## Overview

This project is an audio classification system that detects whether a voice sample is **AI-generated** or **human speech**. It works by analyzing short segments of audio, extracting acoustic features, and using a machine learning model to make a final prediction with a confidence score.

---

## How It Works

The system follows a structured pipeline:

1. **Input Audio**

   * Accepts Base64-encoded audio from an API request.
   * Decodes and converts it into a waveform sampled at **16 kHz mono**.

2. **Chunk Processing**

   * Splits audio into **5-second segments**.
   * Each chunk is evaluated individually.

3. **Chunk Selection**

   * Chunks must contain at least **2 seconds of speech**.
   * Low-speech or mostly silent chunks are ignored.

4. **Feature Extraction**
   For each valid chunk, six acoustic features are computed:

   * MFCC variability
   * Zero-Crossing Rate variability
   * Energy coefficient of variation
   * Voiced ratio
   * Spectral contrast
   * Pitch variability

5. **Prediction**

   * A trained **Random Forest model** predicts AI vs Human probability for each chunk.

6. **Aggregation**

   * Probabilities from all chunks are combined using **Normalized Weighted Log-Odds Fusion**.
   * Chunks with more speech content receive higher weight.

7. **Final Decision**

   * Final probability determines classification:

     * `> 0.5 → AI_GENERATED`
     * `< 0.5 → HUMAN`
   * Tie cases are resolved using mean probability, then default to HUMAN.

8. **Confidence Score**

   * Confidence equals the probability of the predicted class.

---

## Project Structure

```
project/
│
├── server.py          # FastAPI API endpoint
├── inference.py       # Full prediction pipeline
├── artifacts/
│   └── final_updated_model.pkl   # Trained model
└── dataset_builder.py # Training dataset feature extractor
```

---

## API Usage

**Endpoint**

```
POST /api/voice-detection
```

**Headers**

```
x-api-key: your_api_key
```

**Request Body**

```json
{
  "language": "en",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_STRING"
}
```

**Response**

```json
{
  "status": "success",
  "language": "en",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.87,
  "explanation": "Strong synthetic speech patterns detected."
}
```

---

## Training Methodology

Training data is processed using the same logic as inference:

* Audio resampled to 16 kHz mono
* Split into 5-second chunks
* Features extracted per chunk
* Each chunk treated as one training sample

This ensures **training–inference consistency**, improving real-world performance.

---

## Key Design Principles

* Chunk-based analysis for robustness
* Fixed speech threshold for stability
* Weighted aggregation for reliability
* Probabilistic confidence scoring
* Deterministic classification logic

---

## Requirements

Install dependencies:

```
pip install numpy pandas librosa fastapi uvicorn tqdm
```

FFmpeg must also be installed and available in system PATH.

---

## Running the API

```
uvicorn server:app --reload
```

---

## Summary

This system is designed to be:

* robust to noisy audio
* stable across varying speech quality
* reliable on unseen data

It achieves this by combining signal processing, feature engineering, and ensemble machine learning.

---
