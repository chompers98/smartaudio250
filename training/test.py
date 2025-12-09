# Sandbox to test intermediate steps
import os
import wave
import contextlib
from scipy.io import wavfile

# Check whether .wav files were standardized correctly
path = '../data/talking/talking1.wav'
        
try:
    with contextlib.closing(wave.open(path, 'r')) as f:
        channels = f.getnchannels()
        framerate = f.getframerate()
        sampwidth = f.getsampwidth()
        nframes = f.getnframes()
        duration = nframes / float(framerate)
        
        if channels != 1:
            print("Not mono}")
        if framerate != 16000:
            print("Not 16 kHz")
        if sampwidth != 2:
            print("Not 16-bit")
        if duration < 4.5 or duration > 5.5:
            print("Incorrect duration")
except Exception as e:
    print("Error reading file")
