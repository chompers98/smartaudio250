import numpy as np
import random

class DummyClassifier:
    def __init__(self):
        self.classes = ['knock', 'speaking', 'baby_crying', 'music', 'pet_noises', 'silence']
    
    def predict(self, audio_data):
        """Return random classification for testing"""
        sound_class = random.choice(self.classes)
        confidence = random.uniform(0.7, 0.99)
        probabilities = {c: random.uniform(0, 1) for c in self.classes}
        
        return {
            'sound_class': sound_class,
            'confidence': confidence,
            'probabilities': probabilities
        }

classifier = DummyClassifier()
