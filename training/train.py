import os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# === CONFIG ===
AUDIO_DIR = "../data"      # Folder where all .wav files are stored
LABELS_CSV = "../data/labels.csv"      # Path to CSV with filename,label
SAMPLE_RATE = 16000
N_MFCC = 13

# === Load Labels ===
df = pd.read_csv(LABELS_CSV)
print(f"Loaded {len(df)} labeled samples.")

# === Extract MFCC Features ===
def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)

features = []
labels = []

for idx, row in df.iterrows():
    fname = row["filename"]
    label = row["label"]
    path = os.path.join(AUDIO_DIR, fname)
    
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {path}")
        continue
    try:
        feat = extract_features(path)
        features.append(feat)
        labels.append(label)
    except Exception as e:
        print(f"❌ Failed to extract {fname}: {e}")

# === Prepare Data ===
X = np.array(features)
le = LabelEncoder()
y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
