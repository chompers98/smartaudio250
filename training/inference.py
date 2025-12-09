# Inference script for classification testing
import os
import librosa
import numpy as np
import joblib

# Set local constants
MODEL_PATH = "../node2-processor/models/random_forest_audio_classifier.pkl"
ENCODER_PATH = "../node2-processor/models/label_encoder.pkl"
SAMPLE_RATE = 16000
N_MFCC = 13

# Load model
clf = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# Extract features to match training
def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)  

# Get prediction
def classify(filepath):
    feat = extract_features(filepath).reshape(1, -1)
    pred = clf.predict(feat)[0]
    label = le.inverse_transform([pred])[0]

    proba = clf.predict_proba(feat)[0]
    confidence = np.max(proba)

    print(f"Classification Result for {filepath}:")
    print(f"Predicted Label: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    return label, confidence

# Call classification function (input: .wav path)
if __name__ == "__main__":
    import sys

    audio_file = sys.argv[1]
    classify(audio_file)
