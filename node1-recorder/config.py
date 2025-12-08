import os
from dotenv import load_dotenv

load_dotenv()

# Audio Configuration
SAMPLE_RATE = 16000          # Hz (sufficient for speech/household sounds)
CHUNK_SIZE = 512             # Samples per frame (~32ms at 16kHz)
CHANNELS = 1                 # Mono
AUDIO_FORMAT = 16            # 16-bit PCM

# Detection Configuration
DECIBEL_THRESHOLD = 55       # dB - tune based on environment
REFERENCE_LEVEL = 0.005      # Reference for dB calculation
CAPTURE_DURATION = 4         # Seconds to capture after detection
PRE_BUFFER_SIZE = 10         # Chunks to keep from before threshold (pre-roll)

# Server Configuration
NODE2_SERVER_URL = os.getenv('NODE2_SERVER_URL', 'http://localhost:8000')
CLASSIFY_ENDPOINT = f'{NODE2_SERVER_URL}/classify'
REQUEST_TIMEOUT = 30         # Seconds

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'node1.log'
