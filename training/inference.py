import os
import librosa
import numpy as np
import joblib

# === CONFIG ===
MODEL_PATH = "../node2-processor/models/random_forest_audio_classifier.pkl"
ENCODER_PATH = "../node2-processor/models/label_encoder.pkl"
SAMPLE_RATE = 16000
N_MFCC = 13

# === Load Model + Encoder ===
clf = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# === Feature Extraction (must match training!) ===
def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)  # same averaging as training

# === Inference ===
def classify(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    feat = extract_features(filepath).reshape(1, -1)
    pred = clf.predict(feat)[0]
    label = le.inverse_transform([pred])[0]

    # Optional: get class probability (RandomForest supports predict_proba)
    proba = clf.predict_proba(feat)[0]
    confidence = np.max(proba)

    print("==== Classification Result ====")
    print(f"File: {filepath}")
    print(f"Predicted Label: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")

    return label, confidence

# === CLI usage example ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python classify_audio.py <path_to_wav>")
        exit()

    audio_file = sys.argv[1]
    classify(audio_file)
