import pyaudio
import numpy as np
import wave
import logging
import requests
import json
from datetime import datetime
from pathlib import Path
import uuid
from config import *
from audio_buffer import RollingAudioBuffer, AudioProcessor

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioRecorder:
    """
    Main recorder class. Monitors audio continuously and captures clips
    when threshold is exceeded.
    """
    
    def __init__(self, threshold_db=DECIBEL_THRESHOLD):
        self.threshold_db = threshold_db
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.chunk_size = CHUNK_SIZE
        
        # Audio device
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        # Buffers
        self.pre_buffer = RollingAudioBuffer(PRE_BUFFER_SIZE)
        
        # State
        self.is_monitoring = False
        self.session_id = str(uuid.uuid4())
        
        logger.info(f"AudioRecorder initialized. Threshold: {threshold_db} dB")
    
    def open_stream(self):
        """Open audio input stream"""
        try:
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                exceptions=False
            )
            logger.info("Audio stream opened")
        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")
            raise
    
    def close_stream(self):
        """Close audio input stream"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            logger.info("Audio stream closed")
    
    def read_chunk(self):
        """
        Read a single chunk of audio from the stream.
        Returns:
            numpy array of int16 samples
        """
        try:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_frame = np.frombuffer(data, dtype=np.int16)
            return audio_frame
        except Exception as e:
            logger.error(f"Error reading audio chunk: {e}")
            return None
    
    def capture_audio_clip(self, duration_seconds=CAPTURE_DURATION):
        """
        Capture audio for specified duration.
        Args:
            duration_seconds: how long to capture
        Returns:
            numpy array of audio samples
        """
        frames = []
        frames_to_capture = int(self.sample_rate * duration_seconds / self.chunk_size)
        
        logger.info(f"Capturing {duration_seconds}s audio clip ({frames_to_capture} chunks)")
        
        for i in range(frames_to_capture):
            chunk = self.read_chunk()
            if chunk is not None:
                frames.append(chunk)
        
        if frames:
            audio_clip = np.concatenate(frames)
            return audio_clip
        return np.array([])
    
    def save_wav(self, audio_data, filename=None):
        """
        Save audio data to WAV file.
        Args:
            audio_data: numpy array of int16 samples
            filename: output filename (auto-generated if None)
        Returns:
            str: path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"audio_clip_{timestamp}.wav"
        
        filepath = Path("./audio_clips") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        try:
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.astype(np.int16).tobytes())
            
            logger.info(f"Saved audio clip: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save WAV file: {e}")
            return None
    
    def send_to_node2(self, audio_filepath, decibel_level):
        """
        Send audio clip to Node 2 for classification.
        Args:
            audio_filepath: path to WAV file
            decibel_level: detected decibel level (float)
        Returns:
            dict: classification result from Node 2, or None if failed
        """
        try:
            with open(audio_filepath, 'rb') as audio_file:
                files = {'audio': audio_file}
                data = {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'decibel_level': str(decibel_level),
                    'session_id': self.session_id
                }
                
                logger.info(f"Sending audio to {CLASSIFY_ENDPOINT}")
                response = requests.post(
                    CLASSIFY_ENDPOINT,
                    files=files,
                    data=data,
                    timeout=REQUEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Classification result: {result['classification']} "
                              f"(confidence: {result['confidence']:.2f})")
                    return result
                else:
                    logger.error(f"Node 2 returned status {response.status_code}: {response.text}")
                    return None
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send audio to Node 2: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error sending to Node 2: {e}")
            return None
    
    def run(self):
        """Main monitoring loop"""
        self.open_stream()
        self.is_monitoring = True
        
        logger.info(f"Starting audio monitoring (threshold: {self.threshold_db} dB)")
        print(f"Monitoring audio... (Threshold: {self.threshold_db} dB)")
        
        try:
            while self.is_monitoring:
                # Read frame
                chunk = self.read_chunk()
                if chunk is None:
                    continue
                
                # Add to pre-buffer
                self.pre_buffer.add_frame(chunk)
                
                # Calculate dB
                db = AudioProcessor.calculate_decibel(chunk, REFERENCE_LEVEL)
                
                # Check threshold
                if db > self.threshold_db:
                    logger.info(f"Threshold exceeded! dB: {db:.1f} (threshold: {self.threshold_db})")
                    
                    # Include pre-buffer data + new capture
                    pre_buffer_data = self.pre_buffer.get_all()
                    new_capture = self.capture_audio_clip(CAPTURE_DURATION)
                    
                    if len(pre_buffer_data) > 0:
                        audio_clip = np.concatenate([pre_buffer_data, new_capture])
                    else:
                        audio_clip = new_capture
                    
                    # Save and send
                    filepath = self.save_wav(audio_clip)
                    if filepath:
                        self.send_to_node2(filepath, db)
        
        except KeyboardInterrupt:
            logger.info("Recording stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}", exc_info=True)
        finally:
            self.close_stream()
            self.pyaudio.terminate()
            logger.info("Recorder shutdown complete")

def main():
    recorder = AudioRecorder(threshold_db=DECIBEL_THRESHOLD)
    recorder.run()

if __name__ == '__main__':
    main()
