import librosa
import numpy as np
import os
import pandas as pd

ragas = []
features = []
#audio_path = 'Datasets/Bhairava/bhairav1.wav'
#y, sr = librosa.load(audio_path)
#print(f"Audio loaded with sampling rate: {sr} Hz")
#print(f"Number of audio samples: {len(y)}")

# Iterate through the folders and the files
for ragas_folder in os.listdir("Datasets/"):
    ragas_path = os.path.join("Datasets/", ragas_folder)
    # print(ragas_folder)
    if os.path.isdir(ragas_path):
        for filename in os.listdir(ragas_path):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                audio_path = os.path.join(ragas_path, filename)
                try:
                    #Load the audio file
                    y, sr = librosa.load(audio_path, duration=30)
                    #Feature extractor using mfcc
                    mfccs = librosa.feature.mfcc(y=y, sr=sr,n_mfcc= 40)
                    mfccs_scaled_features = np.mean(mfccs.T, axis=0)
                    #zero_crossings = librosa.feature.zero_crossing_rate(y)
                    #rolloff = librosa.feature.spectral_rolloff()
                    #spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, feature='spectral')
                    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    mfccs_mean = np.mean(mfccs, axis=1)
                    feature_vector = np.concatenate(mfccs_mean, mfccs_scaled_features)
                    features.append(feature_vector)
                    ragas.append(ragas_folder)
                    print(f"Processed: {filename} in {ragas_folder}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

#f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr) #fundamental frequency
#f0_non_zero = f0[f0 > 0]

#if len(f0_non_zero) > 0:
    #average_pitch_hz = np.mean(f0_non_zero)
    #print(f"\nAverage fundamental frequency (pitch): {average_pitch_hz:.2f} Hz")

# Converting to a musical note name
#     closest_note = librosa.hz_to_note(average_pitch_hz)
#     print(f"Closest musical note: {closest_note}")
# else:
#     print("\nCould not detect a clear fundamental frequency.")

# Extracting Tempo
# onset_env = librosa.onset.onset_detect(y=y, sr=sr)
# if len(onset_env) > 1:
#     tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
#     print(f"\nEstimated tempo: {tempo:.2f} BPM (beats per minute)")
# else:
#     print("\nCould not reliably estimate the tempo.")

# # Energy
# rms = librosa.feature.rms(y=y)[0]
# average_rms = np.mean(rms)
# print(f"\nAverage RMS (Root Mean Square) energy: {average_rms:.4f}")

# Convert the lists to a Pandas DataFrame
feature_df = pd.DataFrame(features)
ragas_df = pd.DataFrame({'genre': ragas})
dataset = pd.concat([feature_df, ragas_df], axis=1)

print("\nDataset created successfully!")
print(dataset.head())

# Convert dataset to csv
dataset.to_csv("audio_features_dataset.csv", index=False)
    
