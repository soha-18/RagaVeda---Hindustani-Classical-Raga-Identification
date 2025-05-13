import librosa
import numpy as np
import os
import pandas as pd
import random
import matplotlib.pyplot as plt

ragas = []
features = []

def audio_augmentation(audio_file, sr, augmentation_type='time_stretch', factor=None):
    if augmentation_type == 'time_stretch':
        if factor is None:
            factor = random.uniform(0.8, 1.2)
        y_stretched = librosa.effects.time_stretch(audio_file, rate=factor)
        return y_stretched
    elif augmentation_type == 'pitch_shift':
        if factor is None:
            factor = random.uniform(-2, 2)
        y_shifted = librosa.effects.pitch_shift(audio_file, sr=sr, n_steps=factor)
        return y_shifted
    elif augmentation_type == 'add_noise':
        noise = 0.005 * np.random.randn(len(audio_file))
        y_noisy = audio_file + noise
        return y_noisy
    else:
        return audio_file
  
def create_spectrogram(file, n_fft, hop):
    try:
        stft_output = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        spectrogram = np.abs(stft_output)
        
        return spectrogram
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None

def plot_spectrogram(spectrogram, sr, hop_length):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max),
                             sr=sr,
                             hop_length=hop_length,
                             x_axis='time',
                             y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram")
        plt.tight_layout()
        plt.show()

#Feature extractor using mfcc
def extract_mfcc_feature_vector(audio_file, sr):
    #Load the audio file
    #y, sr = librosa.load(audio_path, duration=30)
    mfccs = librosa.feature.mfcc(y=audio_file, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
    #mfccs_scaled_features = np.mean(mfccs.T, axis=0)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    feature_matrix = np.concatenate([mfccs, delta_mfccs, delta2_mfccs], axis=0)
    feature_vector = np.mean(feature_matrix, axis=1)
    return feature_vector
    
# Iterate through the folders and the files
for ragas_folder in os.listdir("Datasets/"):
    ragas_path = os.path.join("Datasets/", ragas_folder)
    if os.path.isdir(ragas_path):
        for filename in os.listdir(ragas_path):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                audio_path = os.path.join(ragas_path, filename)
                try:
                    y, sr = librosa.load(audio_path, duration=30)
                    y_stretched = audio_augmentation(y, sr, augmentation_type='time_stretch')
                    y_shifted = audio_augmentation(y_stretched, sr, augmentation_type='pitch_shift')
                    aug_audio = audio_augmentation(y_shifted, sr, augmentation_type='add_noise')
                    feature_vector = extract_mfcc_feature_vector(aug_audio, sr)
                    features.append(feature_vector)
                    ragas.append(ragas_folder)
                    #print(f"Processed: {filename} in {ragas_folder}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

# Convert the lists to a Pandas DataFrame
feature_df = pd.DataFrame(features)
ragas_df = pd.DataFrame({'Ragas': ragas})
dataset = pd.concat([feature_df, ragas_df], axis=1)
prefix = "mfcc_"
new_columns = [prefix + str(col) for col in dataset.columns[:-1]]
dataset.columns = new_columns + [dataset.columns[-1]]

print("\nDataset created successfully!")
#print(dataset.head())

# Convert dataset to csv
dataset.to_csv("audio_features_dataset.csv", index=False)
