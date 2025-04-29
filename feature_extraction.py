import librosa
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

ragas = []
features = []

##Feature extractor using mfcc
def extract_mfcc_feature_vector(audio_file, n_mfcc=40, n_fft=2048, hop_length=512):
    y, sr = librosa.load(audio_file, duration=30)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    # mfccs_scaled_features = np.mean(mfccs.T, axis=0)
    #feature_vector = np.mean(mfccs.T, axis=0)
    feature_vector = np.mean(mfccs, axis=1)
    # feature_vector = np.concatenate((mfccs_mean, mfccs_scaled_features))
    return feature_vector


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


# audio_path = 'Datasets/Bhairava/bhairav1.wav'
# y, sr = librosa.load(audio_path, duration=30)
# # fft_length = 2048
# # hop_length = 512
# print(f"Audio loaded with sampling rate: {sr} Hz")
# print(f"Number of audio samples: {len(y)}")
# time_step = 512/sr
# print(time_step)
# zero_crossings = librosa.feature.zero_crossing_rate(y)
#print(zero_crossings)
# spectrogram_signal = create_spectrogram(y, fft_length, hop_length)
# rolloff = librosa.feature.spectral_rolloff(S=spectrogram_signal, sr=sr)
# plot_spectrogram(spectrogram_signal, sr, hop_length)
#print(rolloff)
#spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, feature='spectral')
#print(spectral_flux)

# # Iterate through the folders and the files
for ragas_folder in os.listdir("Datasets"):
    ragas_path = os.path.join("Datasets", ragas_folder)
    # print(ragas_folder)
    if os.path.isdir(ragas_path):
        for filename in os.listdir(ragas_path):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                #print(filename)
                audio_path = os.path.join(ragas_path, filename)
                #print(audio_path)
                try:
                    ##Load the audio file
                    #y, sr = librosa.load(audio_path, duration=30)
                    feature_vector = extract_mfcc_feature_vector(audio_path)
                    # print(feature_vector.shape)
                    features.append(feature_vector)
                    ragas.append(ragas_folder)
                    #print(f"Processed: {filename} in {ragas_folder}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

# #f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr) #fundamental frequency
# #f0_non_zero = f0[f0 > 0]

# #if len(f0_non_zero) > 0:
#     #average_pitch_hz = np.mean(f0_non_zero)
#     #print(f"\nAverage fundamental frequency (pitch): {average_pitch_hz:.2f} Hz")

# # Converting to a musical note name
# #     closest_note = librosa.hz_to_note(average_pitch_hz)
# #     print(f"Closest musical note: {closest_note}")
# # else:
# #     print("\nCould not detect a clear fundamental frequency.")

# # Extracting Tempo
# # onset_env = librosa.onset.onset_detect(y=y, sr=sr)
# # if len(onset_env) > 1:
# #     tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
# #     print(f"\nEstimated tempo: {tempo:.2f} BPM (beats per minute)")
# # else:
# #     print("\nCould not reliably estimate the tempo.")

# # # Energy
# # rms = librosa.feature.rms(y=y)[0]
# # average_rms = np.mean(rms)
# # print(f"\nAverage RMS (Root Mean Square) energy: {average_rms:.4f}")

## Convert the lists to a Pandas DataFrame
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
    
