# Debugging File: Created with the help of Copilot to test record and send functionality
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
CHANNELS = 1

print("🤫 Recording 10 seconds of audio to diagnose noise...")
print("🔇 (Be completely silent, turn off fans, close apps)\n")

audio = sd.rec(int(SAMPLE_RATE * 10), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', blocking=True)
audio = audio.flatten()

# Calculate dB at different points
for i, name in enumerate(['0-2s', '2-4s', '4-6s', '6-8s', '8-10s']):
    chunk = audio[i*SAMPLE_RATE*2:(i+1)*SAMPLE_RATE*2]
    rms = np.sqrt(np.mean(chunk ** 2))
    if rms > 0:
        db = 20 * np.log10(rms / 0.005)
    else:
        db = -np.inf
    print(f"{name}: {db:.1f} dB (RMS: {rms:.1f})")

# Save for inspection
import scipy.io.wavfile as wavfile
wavfile.write('test_audio.wav', SAMPLE_RATE, audio.astype(np.int16))
print("\nSaved to test_audio.wav ✅")
