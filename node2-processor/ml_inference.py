import torch
import numpy as np
from config import MODEL_PATH, MODEL_CLASSES, MODEL_INPUT_DIM
import logging

logger = logging.getLogger(__name__)

class SoundClassifier:
    """
    Load and run inference with the trained sound classification model.
    Assumes your teammate provides a PyTorch model.
    """
    
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.classes = MODEL_CLASSES
        logger.info(f"Model loaded on device: {self.device}")
    
    def _load_model(self, model_path):
        """
        Load PyTorch model from file.
        Expects the model file to be a .pt file (PyTorch state dict or full model)
        """
        try:
            # Option 1: If saved as full model
            model = torch.load(model_path, map_location=self.device)
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, features):
        """
        Run inference on feature vector.
        
        Args:
            features: numpy array of shape (MODEL_INPUT_DIM,)
        
        Returns:
            dict with keys: 'class', 'confidence', 'probabilities'
        """
        # Convert to tensor and batch
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
        features_tensor = features_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(features_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        # Get predictions
        probs_np = probabilities.cpu().numpy()[0]  # Shape: (num_classes,)
        pred_idx = np.argmax(probs_np)
        pred_class = self.classes[pred_idx]
        pred_confidence = float(probs_np[pred_idx])
        
        # Create probability dict
        prob_dict = {cls: float(prob) for cls, prob in zip(self.classes, probs_np)}
        
        return {
            'class': pred_class,
            'confidence': pred_confidence,
            'probabilities': prob_dict
        }
