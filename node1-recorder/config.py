import os
from dotenv import load_dotenv

load_dotenv()

# audio config variables
SAMPLE_RATE = 16000
CHUNK_SIZE = 512 # samples per frame (~32ms @ 16kHz)
CHANNELS = 1
CAPTURE_DURATION = 4

# server variables
NODE2_SERVER_URL = os.getenv('NODE2_SERVER_URL', 'http://localhost:8000')
