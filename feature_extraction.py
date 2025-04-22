import librosa
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

ragas = []
features = []

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
    
# Iterate through the folders and the files
for ragas_folder in os.listdir("Datasets/"):
    ragas_path = os.path.join("Datasets/", ragas_folder)
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
                    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    mfccs_mean = np.mean(mfccs, axis=1)
                    feature_vector = np.concatenate((mfccs_mean, mfccs_scaled_features))
                    features.append(feature_vector)
                    ragas.append(ragas_folder)
                    print(f"Processed: {filename} in {ragas_folder}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

# Convert the lists to a Pandas DataFrame
feature_df = pd.DataFrame(features)
ragas_df = pd.DataFrame({'genre': ragas})
dataset = pd.concat([feature_df, ragas_df], axis=1)
prefix = "mfcc_"
new_columns = [prefix + str(col) for col in dataset.columns[:-1]]
dataset.columns = new_columns + [dataset.columns[-1]]

print("\nDataset created successfully!")
print(dataset.head())

# Convert dataset to csv
dataset.to_csv("audio_features_dataset.csv", index=False)
    
