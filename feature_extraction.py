import librosa
import numpy as np
import os
import pandas as pd

# 1. Loading the audio file
#audio_path = 'Datasets/Bhairava/bhairav1.wav'
audio_path = 'Datasets/Bhoopali/bhoop1.wav'
y, sr = librosa.load(audio_path)

#print(f"Audio loaded with sampling rate: {sr} Hz")
#print(f"Number of audio samples: {len(y)}")

# 2. Extracting the Fundamental Frequency
f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

# 'f0' is an array of estimated fundamental frequencies over time
# We'll remove the parts where no pitch was detected (-1)
f0_non_zero = f0[f0 > 0]

if len(f0_non_zero) > 0:
    average_pitch_hz = np.mean(f0_non_zero)
    print(f"\nAverage fundamental frequency (pitch): {average_pitch_hz:.2f} Hz")

    # Converting to a musical note name
    closest_note = librosa.hz_to_note(average_pitch_hz)
    print(f"Closest musical note: {closest_note}")
else:
    print("\nCould not detect a clear fundamental frequency.")

# 3. Extracting Tempo
onset_env = librosa.onset.onset_detect(y=y, sr=sr)
if len(onset_env) > 1:
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    print(f"\nEstimated tempo: {tempo:.2f} BPM (beats per minute)")
else:
    print("\nCould not reliably estimate the tempo.")

# 4. Energy
rms = librosa.feature.rms(y=y)[0]
average_rms = np.mean(rms)
print(f"\nAverage RMS (Root Mean Square) energy: {average_rms:.4f}")

# 7. Chroma Features
#chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
#average_chroma = np.mean(chromagram, axis=1)
#print("\nAverage Chroma Features (for each of the 12 notes):")
#note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
#for i, chroma_value in enumerate(average_chroma):
#    print(f"{note_names[i]}: {chroma_value:.4f}")

#Feature extractor using mfcc
mfccs = librosa.feature.mfcc(y=y, sr=sr,n_mfcc= 40);
mfccs_scaled_features = np.mean(mfccs.T, axis=0)
zero_crossings = librosa.feature.zero_crossing_rate(y)
rolloff = librosa.feature.spectral_rolloff(y)
spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, feature='spectral')

extracted_features = []
extracted_features.append(mfccs_scaled_features)
    
