import os
from dotenv import load_dotenv

load_dotenv()

# audio config variables
SAMPLE_RATE = 16000 # Hz
CHUNK_SIZE = 512 # Samples per frame (~32ms at 16kHz)
CHANNELS = 1 # Mono

# detection threshold variables
DECIBEL_THRESHOLD = 55 # dB
REFERENCE_LEVEL = 0.005 # Reference for dB calculation
CAPTURE_DURATION = 4 # Seconds to capture after detection

# server variables
NODE2_SERVER_URL = os.getenv('NODE2_SERVER_URL', 'http://localhost:8000')
