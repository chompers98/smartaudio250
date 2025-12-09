# Define global variables for audio processor
import os
from dotenv import load_dotenv

load_dotenv()

# Audio Processing variables
SAMPLE_RATE = 16000
N_MFCC = 13                 # Number of MFCC coefficients
N_MELS = 128                # Number of mel bands
FMAX = 8000                 # Max frequency (Hz)
CHUNK_DURATION = 1          # Process in 1-second chunks

# Model variables
MODEL_PATH = './models/sound_classifier.pt'
MODEL_INPUT_DIM = 26        # 13 MFCC + 13 deltas
MODEL_CLASSES = ['knock', 'speaking', 'baby_crying', 'music', 'pet_noises', 'silence']
CONFIDENCE_THRESHOLD = 0.6  # Only notify if confidence > this

# Database (PostgreSQL) variables
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'sound_detection')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')

# Notification variables
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_FROM = os.getenv('TWILIO_PHONE_FROM')
NOTIFY_PHONE_TO = os.getenv('NOTIFY_PHONE_TO')

# Server variables
HOST = '0.0.0.0'
PORT = 8000
