import os
import wave
import contextlib
from scipy.io import wavfile

path = '../data/talking/talking1.wav'
        
try:
    with contextlib.closing(wave.open(path, 'r')) as f:
        channels = f.getnchannels()
        framerate = f.getframerate()
        sampwidth = f.getsampwidth()
        nframes = f.getnframes()
        duration = nframes / float(framerate)
        
        issues = []
        if channels != 1:
            issues.append("not mono")
        if framerate != 16000:
            issues.append("not 16kHz")
        if sampwidth != 2:
            issues.append("not 16-bit")
        if duration < 4.5 or duration > 5.5:
            issues.append(f"duration {duration:.2f}s")

        if issues:
            print(f"{path}: ❌ {'; '.join(issues)}")
        else:
            print(f"{path}: ✅ OK")
except Exception as e:
    print(f"{path}: ⚠️ Error reading file - {e}")
