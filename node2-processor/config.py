import os
from dotenv import load_dotenv

load_dotenv()

# audio processing variables
SAMPLE_RATE = 16000
N_MFCC = 13                 # number of MFCC coefficients
N_MELS = 128                # number of mel bands
FMAX = 8000                 # max freq (Hz)

# server variables
HOST = '0.0.0.0'
PORT = 8000
