import numpy as np
import logging
import librosa
import joblib
import tensorflow_hub as hub

logger = logging.getLogger(__name__)

# Our custom trained random forest classifier
class SoundClassifier:
    """
    Sound classifier using a trained Random Forest model.
    """
    
    def __init__(self, rf_model_path="random_forest_audio_classifier.pkl", label_encoder_path="label_encoder.pkl"):
        self.sample_rate = 16000
        
        logger.info("Loading Random Forest model...")
        self.model = joblib.load(rf_model_path)
        self.le = joblib.load(label_encoder_path)
        logger.info("✓ Random Forest model loaded successfully.")

    def _extract_mfcc(self, audio_data):
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        return np.mean(mfcc.T, axis=0).reshape(1, -1)

    def predict(self, audio_data):
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        try:
            mfcc_features = self._extract_mfcc(audio_data)
            prediction = self.model.predict(mfcc_features)[0]
            label = self.le.inverse_transform([prediction])[0]
            probs = self.model.predict_proba(mfcc_features)[0]
            prob_dict = dict(zip(self.le.classes_, probs))
            return {
                "class": label,
                "confidence": float(probs[prediction]),
                "probabilities": prob_dict
            }
        except Exception as e:
            logger.error(f"Random Forest prediction failed: {e}")
            return {"class": "unknown", "confidence": 0.0, "probabilities": {}}

# Google's YAMNet model for audio classification
class YAMNetClassifier:
    """
    Use Google's YAMNet pre-trained audio classification model.
    Recognizes 521 sound classes including household sounds.
    """
    
    def __init__(self, model_path=None):
        logger.info("Loading YAMNet model from TensorFlow Hub...")
        
        # Load YAMNet model
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Load class names
        self.class_names = self._load_class_names()
        
        # Map YAMNet classes to our 6 classes
        self.class_mapping = {
            'knock': ['knock', 'door', 'tap', 'percussion', 'beatboxing', 'clap', 'applause', 'bang', 'hit', 'strike', 'wood', 'boing'],
            'speaking': ['speech', 'voice', 'conversation', 'talk', 'speaking', 'human voice', 'man speaking', 'woman speaking', 'child speech'],
            'baby_crying': ['crying', 'baby cry', 'infant cry', 'wail', 'sob'],
            'music': ['music', 'song', 'instrument', 'piano', 'guitar', 'violin', 'singing', 'melody', 'pop music'],
            'pet_noises': ['dog', 'cat', 'bark', 'meow', 'animal', 'pet', 'animal cry', 'zoo animal'],
            'silence': ['silence', 'quiet', 'inside small room', 'office']
        }
        
        logger.info("✓ YAMNet model loaded successfully")
    
    def _load_class_names(self):
        """Load YAMNet class names"""
        try:
            import csv
            import urllib.request
            
            # Download class names from YAMNet repo
            url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audio_set/yamnet/yamnet_classes.csv'
            
            class_names = []
            response = urllib.request.urlopen(url)
            for row in csv.reader(response.read().decode('utf-8').splitlines()):
                class_names.append(row[2])  # Get the display name
            
            return class_names
        except Exception as e:
            logger.warning(f"Could not load class names from URL: {e}")
            # Fallback: return generic names
            return [f"class_{i}" for i in range(521)]
    
    def predict(self, audio_data):
        """Predict sound class using YAMNet."""
        try:
            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            logger.info(f"Audio shape: {audio_data.shape}, normalized min: {np.min(audio_data):.4f}, max: {np.max(audio_data):.4f}")
            
            # Run inference
            scores, embeddings, spectrogram = self.model(audio_data)
            
            # Get top predictions
            scores_np = scores.numpy()
            mean_scores = np.mean(scores_np, axis=0)
            top_class_idx = np.argmax(mean_scores)
            top_confidence = float(mean_scores[top_class_idx])
            
            logger.info(f"Top 3 predictions:")
            for idx in np.argsort(mean_scores)[-3:][::-1]:
                logger.info(f"  {self.class_names[idx]}: {mean_scores[idx]:.4f}")
            
            yamnet_class = self.class_names[top_class_idx]
            mapped_class = self._map_to_our_classes(yamnet_class)
            probabilities = self._create_probabilities(scores_np, yamnet_class)
            
            logger.info(f"Detected: {yamnet_class} → mapped to {mapped_class} ({top_confidence:.2%})")
            
            return {
                'class': mapped_class,
                'confidence': top_confidence,
                'probabilities': probabilities
            }
        
        except Exception as e:
            logger.error(f"Error in YAMNet prediction: {e}", exc_info=True)
            return {
                'class': 'silence',
                'confidence': 0.5,
                'probabilities': {cls: 0.0 for cls in ['knock', 'speaking', 'baby_crying', 'music', 'pet_noises', 'silence']}
            }
    
    def _map_to_our_classes(self, yamnet_class):
        """Map YAMNet class name to one of our 6 classes"""
        yamnet_class_lower = yamnet_class.lower()
        
        for our_class, keywords in self.class_mapping.items():
            for keyword in keywords:
                if keyword.lower() in yamnet_class_lower:
                    return our_class
        
        # Default mapping
        return 'silence'
    
    def _create_probabilities(self, scores, top_yamnet_class):
        """Create probability distribution for our 6 classes"""
        probabilities = {
            'knock': 0.0,
            'speaking': 0.0,
            'baby_crying': 0.0,
            'music': 0.0,
            'pet_noises': 0.0,
            'silence': 0.0
        }
        
        # Calculate average score for each YAMNet class
        mean_scores = np.mean(scores, axis=0)
        
        # Map YAMNet probabilities to our classes
        for yamnet_idx, score in enumerate(mean_scores):
            yamnet_class = self.class_names[yamnet_idx]
            our_class = self._map_to_our_classes(yamnet_class)
            probabilities[our_class] += float(score)
        
        # Normalize
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}
        
        return probabilities

#import torch
#import numpy as np
#from config import MODEL_PATH, MODEL_CLASSES, MODEL_INPUT_DIM
#import logging
#
#
#logger = logging.getLogger(__name__)
#
#class SoundClassifier:
#    """
#    Load and run inference with the trained sound classification model.
#    Assumes your teammate provides a PyTorch model.
#    """
#    
#    def __init__(self, model_path=MODEL_PATH):
#        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#        self.model = self._load_model(model_path)
#        self.classes = MODEL_CLASSES
#        logger.info(f"Model loaded on device: {self.device}")
#    
#    def _load_model(self, model_path):
#        """
#        Load PyTorch model from file.
#        Expects the model file to be a .pt file (PyTorch state dict or full model)
#        """
#        try:
#            # Option 1: If saved as full model
#            model = torch.load(model_path, map_location=self.device)
#            return model
#        except Exception as e:
#            logger.error(f"Failed to load model: {e}")
##            raise
#    
#    def predict(self, features):
#        """
#        Run inference on feature vector.
#        
#        Args:
#            features: numpy array of shape (MODEL_INPUT_DIM,)
#        
#        Returns:
#            dict with keys: 'class', 'confidence', 'probabilities'
#        """
#        # Convert to tensor and batch
#        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
#        features_tensor = features_tensor.to(self.device)
#        
#        # Forward pass
#        with torch.no_grad():
#            logits = self.model(features_tensor)
#            probabilities = torch.softmax(logits, dim=1)
#        
#        # Get predictions
#        probs_np = probabilities.cpu().numpy()[0]  # Shape: (num_classes,)
#        pred_idx = np.argmax(probs_np)
#        pred_class = self.classes[pred_idx]
#        pred_confidence = float(probs_np[pred_idx])
#        
#        # Create probability dict
#        prob_dict = {cls: float(prob) for cls, prob in zip(self.classes, probs_np)}
#        
#        return {
#            'class': pred_class,
#            'confidence': pred_confidence,
#            'probabilities': prob_dict
#        }
