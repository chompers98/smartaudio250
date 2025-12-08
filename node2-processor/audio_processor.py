import numpy as np
import librosa
from config import SAMPLE_RATE, N_MFCC, N_MELS, FMAX

class AudioFeatureExtractor:
    """Extract MFCC features from audio"""
    
    @staticmethod
    def extract_mfcc_features(audio_data, sample_rate=SAMPLE_RATE):
        """
        Extract MFCC coefficients + deltas from audio.
        
        Args:
            audio_data: numpy array of audio samples (int16 or float)
            sample_rate: sample rate in Hz
        
        Returns:
            numpy array of shape (n_features,) where n_features = n_mfcc * 2
            (includes MFCC + delta features)
        """
        # Normalize audio to [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=sample_rate,
            n_mfcc=N_MFCC,
            n_mels=N_MELS,
            fmax=FMAX
        )  # Shape: (n_mfcc, time_steps)
        
        # Compute delta (first derivative)
        mfcc_delta = librosa.feature.delta(mfcc)  # Shape: (n_mfcc, time_steps)
        
        # Aggregate over time: compute mean for each feature
        mfcc_mean = np.mean(mfcc, axis=1)         # Shape: (n_mfcc,)
        delta_mean = np.mean(mfcc_delta, axis=1)  # Shape: (n_mfcc,)
        
        # Concatenate to form final feature vector
        features = np.concatenate([mfcc_mean, delta_mean])  # Shape: (n_mfcc * 2,)
        
        return features
    
    @staticmethod
    def extract_fft_spectrum(audio_data, sample_rate=SAMPLE_RATE):
        """
        Compute FFT magnitude spectrum (useful for visualization/debugging).
        
        Args:
            audio_data: numpy array of audio samples
            sample_rate: sample rate in Hz
        
        Returns:
            frequencies, magnitudes
        """
        # Normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Apply Hann window
        windowed = audio_data * np.hanning(len(audio_data))
        
        # Compute FFT
        fft_result = np.fft.fft(windowed)
        magnitude = np.abs(fft_result[:len(fft_result)//2])
        
        # Frequency axis
        frequencies = np.fft.fftfreq(len(windowed), 1/sample_rate)[:len(fft_result)//2]
        
        return frequencies, magnitude
