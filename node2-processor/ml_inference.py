import joblib
import librosa
import numpy as np
import logging
from config import SAMPLE_RATE, N_MFCC

logger = logging.getLogger(__name__)

# load and run inference with custom Random Forest model
class SoundClassifier:
    
    # load pretrained model
    def __init__(self, model_path='models/random_forest_audio_classifier.pkl',
                 encoder_path='models/label_encoder.pkl'):
        try:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            
            self.classes = self.label_encoder.classes_
            
            logger.info(f"Model loaded successfully.")
            logger.info(f"Classes: {list(self.classes)}")
        
        except Exception as e:
            logger.error(f"Model loading failure: {e}")
            raise
    
    # extract MFCC features from audio data
    def extract_features_from_audio(self, audio_data):
        try:
            # convert to float if needed
            if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # extract MFCC as is done in training, so it is ready for inference
            mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
            features = np.mean(mfcc.T, axis=0)
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    # run inference on feature vector
    def predict(self, features):
        try:
            # if features look like raw audio, extract MFCC
            if features.shape[0] > 100:
                features = self.extract_features_from_audio(features)
            
            # reshape for sklearn
            if features.ndim == 1:
                features = features.reshape(1, -1)
                        
            # get predictions
            pred_class_idx = self.model.predict(features)[0]
            pred_class = self.label_encoder.inverse_transform([pred_class_idx])[0]
            
            # get probabilities
            probabilities = self.model.predict_proba(features)[0]
            prob_dict = {
                cls: float(prob)
                for cls, prob in zip(self.classes, probabilities)
            }
            confidence = float(probabilities[pred_class_idx])
            
            logger.info(f"Prediction: {pred_class} ({confidence:.2%})")
            logger.info(f"Probabilities: {prob_dict}")
            
            return {
                'class': pred_class,
                'confidence': confidence,
                'probabilities': prob_dict
            }
        
        except Exception as e:
            logger.error(f"Error in prediction: {e}", exc_info=True)
            # fallback
            return {
                'class': 'silence',
                'confidence': 0.5,
                'probabilities': {cls: 1.0/len(self.classes) for cls in self.classes}
            }
