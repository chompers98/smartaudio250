import numpy as np
from collections import deque
from threading import Lock

class RollingAudioBuffer:
    """
    Maintains a rolling buffer of audio frames for pre-capture.
    When threshold is triggered, we can include data from before the event.
    """
    def __init__(self, max_frames):
        self.buffer = deque(maxlen=max_frames)
        self.lock = Lock()
    
    def add_frame(self, frame):
        """Add a single audio frame (numpy array)"""
        with self.lock:
            self.buffer.append(frame.copy())
    
    def get_all(self):
        """Get all buffered frames as single array"""
        with self.lock:
            if not self.buffer:
                return np.array([])
            return np.concatenate(list(self.buffer))

class AudioProcessor:
    """Compute audio statistics (RMS, dB level)"""
    
    @staticmethod
    def calculate_rms(audio_frame):
        """
        Calculate RMS (root mean square) of audio frame.
        Args:
            audio_frame: numpy array of int16 samples
        Returns:
            float: RMS value
        """
        # Convert to float for computation
        frame_float = audio_frame.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(frame_float ** 2))
        return rms
    
    @staticmethod
    def calculate_decibel(audio_frame, reference_level=0.005):
        """
        Convert RMS to decibel scale.
        Args:
            audio_frame: numpy array of int16 samples
            reference_level: reference RMS for dB calculation
        Returns:
            float: Decibel level
        """
        rms = AudioProcessor.calculate_rms(audio_frame)
        
        if rms == 0:
            return -np.inf
        
        db = 20 * np.log10(rms / reference_level)
        return db
