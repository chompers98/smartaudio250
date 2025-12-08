import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import requests
import json
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv
import logging
import time
import uuid
import socket

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
load_dotenv()
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512
CAPTURE_DURATION = 4
NODE2_SERVER_URL = os.getenv('NODE2_SERVER_URL', 'http://localhost:8000')

class AudioRecorder:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.session_id = str(uuid.uuid4())
        self.noise_floor_db = 50  # Default fallback
        self.dynamic_threshold = 60  # Default fallback
        Path("./audio_clips").mkdir(exist_ok=True)
        logger.info("AudioRecorder initialized")
        
    def test_connection(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('192.0.0.2', 8000))
        if result == 0:
            print("✓ Port is open")
        else:
            print(f"✗ Port is closed or unreachable. Error code: {result}")
        sock.close()
    
    def record_chunk(self, duration=0.03):
        """Record a chunk of audio"""
        samples = int(self.sample_rate * duration)
        audio = sd.rec(samples, samplerate=self.sample_rate, channels=CHANNELS, dtype='int16', blocking=True)
        return audio.flatten()
    
    def calculate_db(self, audio_data):
        """Calculate dB level from audio data"""
        # Handle edge case of silent audio
        if audio_data is None or len(audio_data) == 0:
            return -np.inf
        
        # Convert to float and normalize
        audio_float = audio_data.astype(np.float32) / 32768.0  # Normalize int16 to [-1, 1]
        
        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms == 0 or rms < 1e-10:
            return -np.inf
        
        # Use proper dB calculation with standard reference
        # For audio: dB = 20 * log10(RMS / reference)
        # Common reference for audio is 20e-6 Pa (0 dBFS)
        db = 20 * np.log10(rms)
        
        return db
        
    def measure_noise_floor(self, duration=5):
        """
        Measure ambient noise level for the first few seconds.
        This helps set an adaptive threshold.
        
        Args:
            duration: How long to measure noise (in seconds)
        
        Returns:
            tuple: (average_db, max_db)
        """
        logger.info(f"📊 Measuring ambient noise for {duration}s (please be quiet)...")
        print("🔇 Measuring ambient noise... keep quiet for a few seconds")
        
        db_levels = []
        samples_per_second = 10
        total_samples = int(duration * samples_per_second)
        
        try:
            for i in range(total_samples):
                chunk = self.record_chunk(0.1)  # 100ms chunks
                db = self.calculate_db(chunk)
                db_levels.append(db)
                
                # Progress indicator
                progress = int((i + 1) / total_samples * 100)
                print(f"  Progress: {progress}% [{db:.1f} dB]", end='\r')
                time.sleep(0.05)
            
            avg_db = np.mean(db_levels)
            max_db = np.max(db_levels)
            min_db = np.min(db_levels)
            
            logger.info(f"Noise floor measured: avg={avg_db:.1f} dB, max={max_db:.1f} dB, min={min_db:.1f} dB")
            print(f"\n✓ Noise floor: {avg_db:.1f} dB (range: {min_db:.1f} - {max_db:.1f} dB)")
            
            return avg_db, max_db
        
        except Exception as e:
            logger.error(f"Error measuring noise floor: {e}")
            logger.warning(f"Using default noise floor: {self.noise_floor_db} dB")
            return self.noise_floor_db, self.noise_floor_db + 5
    
    def record_audio(self, duration=CAPTURE_DURATION):
        """Record audio for specified duration"""
        samples = int(self.sample_rate * duration)
        logger.info(f"Recording {duration}s of audio...")
        audio = sd.rec(samples, samplerate=self.sample_rate, channels=CHANNELS, dtype='int16', blocking=True)
        return audio.flatten()
    
    def save_wav(self, audio_data, filename=None):
        """Save audio to WAV file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"audio_clip_{timestamp}.wav"
        
        filepath = Path("./audio_clips") / filename
        wavfile.write(str(filepath), self.sample_rate, audio_data.astype(np.int16))
        logger.info(f"Saved: {filepath}")
        return str(filepath)
    
    def send_to_node2(self, filepath, db_level):
        """Send audio to Node 2 for processing"""
        try:
            with open(filepath, 'rb') as f:
                files = {'audio': f}
                data = {
                    'decibel_level': db_level,
                    'timestamp': datetime.now().isoformat(),
                    'session_id': 'smartaudio'
                }
                response = requests.post(
                    f"{NODE2_SERVER_URL}/classify",
                    files=files,
                    data=data,
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    # Fix: Use 'classification' not 'sound_class'
                    logger.info(f"✓ Classification: {result['classification']} ({result['confidence']:.2%})")
                    return result
                else:
                    logger.error(f"Node 2 returned status {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Error sending to Node 2: {e}")
        return None
    
    def run(self):
        """Main monitoring loop with adaptive threshold"""
        logger.info("Starting audio monitor with adaptive noise detection")
        print("\n" + "="*60)
        print("🎤 HOUSEHOLD SOUND DETECTION - NODE 1")
        print("="*60)
        
        # Step 1: Measure noise floor
        print("\n📊 Step 1: Calibrating to your environment...")
        avg_noise, max_noise = self.measure_noise_floor(duration=5)
        
        # Step 2: Set dynamic threshold
        # Use max noise + 10 dB cushion to avoid false positives
        self.noise_floor_db = avg_noise
        self.dynamic_threshold = max_noise + 5
        
        logger.info(f"Dynamic threshold set to {self.dynamic_threshold:.1f} dB")
        print(f"\n🎯 Dynamic Threshold: {self.dynamic_threshold:.1f} dB")
        print(f"   (Will trigger when sound is 10 dB above ambient noise)")
        print(f"\n✓ Ready to detect sounds!")
        print("="*60)
        print("🎤 Listening for sounds...\n")
        
        try:
            while True:
                # Record short chunk and calculate dB
                chunk = self.record_chunk(0.05)
                db = self.calculate_db(chunk)
                
                # Check if sound exceeds dynamic threshold
                if db > self.dynamic_threshold:
                    logger.info(f"🔊 Sound detected! dB: {db:.1f} (threshold: {self.dynamic_threshold:.1f})")
                    print(f"\n🔊 SOUND DETECTED: {db:.1f} dB")
                    
                    # Record full audio clip
                    audio = self.record_audio(CAPTURE_DURATION)
                    
                    # Save and send
                    filepath = self.save_wav(audio)
                    if filepath:
                        result = self.send_to_node2(filepath, db)
                        if result:
                            print(f"   ✓ Classification: {result['classification']} ({result['confidence']:.2%})")
                        else:
                            print(f"   ✗ Could not reach Node 2 for classification")
                    print()
        
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            print("\n\n👋 Stopped listening")

if __name__ == '__main__':
    recorder = AudioRecorder()
    recorder.test_connection()
    #recorder.run()
