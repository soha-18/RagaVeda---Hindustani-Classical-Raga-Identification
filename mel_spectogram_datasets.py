import os
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

mel_features = []
#ragas = []
mel_dataset = []

def extract_features_mel(file, sr, n_mels=128, max_pad_len=174):
    mel_spec = librosa.feature.melspectrogram(y=file, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = mel_spec_db.T
    mel_spec_db_padded = pad_sequences([mel_spec_db], padding='post', maxlen=max_pad_len, dtype='float32')[0]

    return mel_spec_db_padded

for ragas_folder in os.listdir("Datasets/"):
    ragas_path = os.path.join("Datasets/", ragas_folder)
    if os.path.isdir(ragas_path):
        for filename in os.listdir(ragas_path):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                audio_path = os.path.join(ragas_path, filename)
                try:
                    y, sr = librosa.load(audio_path, duration=30)
                    feature_vector = extract_features_mel(y, sr)
                    #mel_features.append(feature_vector)
                    #ragas.append(ragas_folder)
                    mel_features.append([feature_vector, ragas_folder])

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

#feature_df = pd.DataFrame({'Mel_Features':mel_features})
#ragas_df = pd.DataFrame({'Ragas': ragas})
#mel_dataset = pd.concat([feature_df, ragas_df], axis=1)
mel_dataset = pd.DataFrame(mel_features)

mel_dataset.to_csv("mel_features_dataset.csv", index=False)
print("\nDataset created successfully!")
