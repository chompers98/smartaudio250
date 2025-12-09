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

# Setup config variables
load_dotenv()
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512
CAPTURE_DURATION = 4
NODE2_SERVER_URL = os.getenv('NODE2_SERVER_URL')

# Audio recording class 
class AudioRecorder:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.session_id = str(uuid.uuid4())
        self.noise_floor_db = 50  
        self.dynamic_threshold = 60  
        Path("./audio_clips").mkdir(exist_ok=True)
        logger.info("AudioRecorder initialized")

    # Test connection function for debugging   
    def test_connection(self):
        ngrok_url = NODE2_SERVER_URL
        try:
            requests.get(f"{ngrok_url}/health", timeout=5)
            print("Node 2 Connected!")
        except Exception as e:
            print(f"Node 2 Connection Failed: {e}")

    # Record audio in small segments 
    def record_chunk(self, duration=0.03):
        samples = int(self.sample_rate * duration)
        audio = sd.rec(samples, samplerate=self.sample_rate, channels=CHANNELS, dtype='int16', blocking=True)
        return audio.flatten()
    
    # Calcilate dB level of audio chunk
    def calculate_db(self, audio_data):
        # Return lowest reading for silence
        if audio_data is None or len(audio_data) == 0:
            return -np.inf
        
        # Convert to float and normalize
        audio_float = audio_data.astype(np.float32) / 32768.0 
        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms == 0 or rms < 1e-10:
            return -np.inf
        # Equation is UdB = 20 * log10(RMS / reference)
        db = 20 * np.log10(rms)
        return db

    # Calibrate the threshold based on environmental noise
    def measure_noise_floor(self, duration=5):
        logger.info(f"Measuring ambient noise for {duration}s (please be quiet)...")
        print(f"Measuring ambient noise for {duration}s (please be quiet)...")
        
        db_levels = []
        samples_per_second = 10
        total_samples = int(duration * samples_per_second)
    
        for i in range(total_samples):
            chunk = self.record_chunk(0.1) # 100ms
            db = self.calculate_db(chunk)
            db_levels.append(db)
            
            # LLM Generated: Show user progress of calibration
            progress = int((i + 1) / total_samples * 100)
            print(f"Progress: {progress}% [{db:.1f} dB]", end='\r')
            time.sleep(0.05)
        
        avg_db = np.mean(db_levels)
        max_db = np.max(db_levels)
        min_db = np.min(db_levels)
        
        logger.info(f"Noise floor measured: avg={avg_db:.1f} dB, max={max_db:.1f} dB, min={min_db:.1f} dB")
        print(f"\nNoise floor: {avg_db:.1f} dB (range: {min_db:.1f} - {max_db:.1f} dB)")
        
        return avg_db, max_db
    
    # Record audio for exact duration
    def record_audio(self, duration=CAPTURE_DURATION):
        samples = int(self.sample_rate * duration)
        logger.info(f"Recording {duration}s of audio")
        audio = sd.rec(samples, samplerate=self.sample_rate, channels=CHANNELS, dtype='int16', blocking=True)
        return audio.flatten()
    
    # Save audio to .wav file to transmit
    def save_wav(self, audio_data, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"audio_clip_{timestamp}.wav"
        
        filepath = Path("./audio_clips") / filename
        wavfile.write(str(filepath), self.sample_rate, audio_data.astype(np.int16))
        logger.info(f"Saved: {filepath}")
        return str(filepath)
    
    # Transmit audio to processor node
    def send_to_node2(self, filepath, db_level):
        try:
            with open(filepath, 'rb') as f:
                files = {'audio': f}
                data = {
                    'decibel_level': db_level,
                    'timestamp': datetime.now().isoformat(),
                    'session_id': 'smartaudio'
                }
                # Send POST request to Node 2
                response = requests.post(
                    f"{NODE2_SERVER_URL}/classify",
                    files=files,
                    data=data,
                    timeout=10
                )
                # Once response is received, print in Node 1
                if response.status_code == 200:
                    result = response.json() 
                    logger.info(f"Classification: {result['classification']} ({result['confidence']:.2%})")
                    return result
                else:
                    logger.error(f"Node 2 failed with {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Error sending to Node 2: {e}")
        return None
    
    # Initiate constant listening loop
    def run(self):
        logger.info("Starting audio monitor with adaptive noise detection")

        # Measure baseline ambient noise
        print("\nCalibrating to your environment...")
        avg_noise, max_noise = self.measure_noise_floor(duration=5)
        
        # Set threshold to current average and 5 dB 
        self.noise_floor_db = avg_noise
        self.dynamic_threshold = max_noise + 5
        
        logger.info(f"Dynamic threshold set to {self.dynamic_threshold:.1f} dB")
        print("Detecting noise...\n")
        
        try:
            # Loop continuously
            while True:
                # Record short chunk and calculate dB
                chunk = self.record_chunk(0.05)
                db = self.calculate_db(chunk)
                
                # Check if sound exceeds dynamic threshold
                if db > self.dynamic_threshold:
                    logger.info(f"Sound detected! dB: {db:.1f} (threshold: {self.dynamic_threshold:.1f})")
                    print(f"\nSound detected: {db:.1f} dB")
                    
                    # Record full audio clip
                    audio = self.record_audio(CAPTURE_DURATION)
                    
                    # Save and send
                    filepath = self.save_wav(audio)
                    if filepath:
                        result = self.send_to_node2(filepath, db)
                        if result:
                            print(f"Classification: {result['classification']} ({result['confidence']:.2%})")
                        else:
                            print("Error reaching node 2 for classification")
                    print()
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            print("\nStopped listening")

if __name__ == '__main__':
    recorder = AudioRecorder()
    recorder.run()
