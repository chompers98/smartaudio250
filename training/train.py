# Random forest classifier training
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

# Set local variables
AUDIO_DIR = "../data"      
LABELS_CSV = "../data/labels.csv"     
SAMPLE_RATE = 16000
N_MFCC = 13

# Load classes
df = pd.read_csv(LABELS_CSV)
print(f"Loaded {len(df)} labeled samples.")

# Extract (Mel-Frequency Cepstral Coefficients) MFCC features
def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)

features = []
labels = []

# Extract features for every audio file in data
for idx, row in df.iterrows():
    fname = row["filename"]
    label = row["label"]
    path = os.path.join(AUDIO_DIR, fname)
    feat = extract_features(path)
    features.append(feat)
    labels.append(label)

# Prepare dataset for training
X = np.array(features)
le = LabelEncoder()
y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and define random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Training completed!")

# Evaluate model on test set
y_pred = clf.predict(X_test)
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Visualizations
import matplotlib.pyplot as plt

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("../output/confusion_matrix.png")
plt.close()

# Create feature importance graph
importances = clf.feature_importances_
plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances)
plt.title("Feature Importances")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.savefig("../output/feature_importances.png")
plt.close()

# Save model to node 2 processor
joblib.dump(clf, "../node2-processor/models/random_forest_audio_classifier.pkl")
joblib.dump(le, "../node2-processor/models/label_encoder.pkl")  
print("Model saved!")