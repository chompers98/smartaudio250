# Create a rolling buffer to track audio snippet before event was detected
import numpy as np
from collections import deque
from threading import Lock

# Maintain a rolling buffer, includes data before threshold is triggered to ensure data isn't lost
class RollingAudioBuffer:
    def __init__(self, max_frames):
        self.buffer = deque(maxlen=max_frames)
        self.lock = Lock()
    
    # Add an audio frame
    def add_frame(self, frame):
        with self.lock:
            self.buffer.append(frame.copy())
    
    # Retrieve frames from the buffer 
    def get_all(self):
        with self.lock:
            if not self.buffer:
                return np.array([])
            return np.concatenate(list(self.buffer))

# Compute statistics on the audio frames to determine event trigger (RMS, dB level)
class AudioProcessor:

    @staticmethod
    # Calculate root mean square (RMS) of audio frame
    def calculate_rms(audio_frame):
        frame_float = audio_frame.astype(np.float32) / 32768.0 # LLM Tip: Convert to float for overflow error
        rms = np.sqrt(np.mean(frame_float ** 2))
        return rms
    
    # Calculate the decibel level from an audio frame
    @staticmethod
    def calculate_decibel(audio_frame, reference_level=0.005):
        rms = AudioProcessor.calculate_rms(audio_frame)
        if rms == 0:
            return -np.inf
        db = 20 * np.log10(rms / reference_level) # Convert RMS to dB
        return db
