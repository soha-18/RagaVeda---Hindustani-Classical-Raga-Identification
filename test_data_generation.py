import librosa
import numpy as np
import os
import pandas as pd
from feature_extraction import create_mfcc_dataset, extract_mfcc_feature_vector, create_melSpectogram_dataset, audio_augmentation

mfcc_test_features = []
mfcc_test_audio_aug_features = []
mel_test_features = []
test_ragas = []

def extract_label(file_path):
    label = os.path.splitext(os.path.basename(file_path))[0]
    return label

audio_folder = 'Test'

for root, dirs, files in os.walk(audio_folder):
        for file in files:
            if file.endswith(".wav") or file.endswith(".mp3"):
                file_path = os.path.join(root, file)
                try:
                    labels = extract_label(file_path)
                    y, sr = librosa.load(file_path, duration=30)
                    ad = audio_augmentation
                    y_stretched = ad.time_stretch(y)
                    y_shifted = ad.pitch_shift(y_stretched, sr)
                    aug_audio = ad.noise_addition(y_shifted)
                    mfcc_feature_vector = extract_mfcc_feature_vector(y, sr)
                    feature_vector_audio_aug = extract_mfcc_feature_vector(aug_audio, sr)
                    feature_vector_mel = create_melSpectogram_dataset(y, sr)
                    mfcc_test_features.append(mfcc_feature_vector)
                    mfcc_test_audio_aug_features.append(feature_vector_audio_aug)
                    test_ragas.append(labels)
                    mel_test_features.append([feature_vector_mel, labels])


                except Exception as e:
                    print(f"Error processing {file}: {e}")

## Convert the lists to a Pandas DataFrame
test_mel_dataset = pd.DataFrame(mel_test_features, columns = ("Mel_Features", "Ragas"))
test_mel_dataset['Ragas'] = test_mel_dataset['Ragas'].str.replace('\d+', '', regex=True)
test_feature_df = pd.DataFrame(mfcc_test_features)
test_ragas_df = pd.DataFrame({'Ragas': test_ragas})
test_dataset = pd.concat([test_feature_df, test_ragas_df], axis=1)
test_audio_aug_feature_df = pd.DataFrame(mfcc_test_audio_aug_features)
test_audio_aug_feature_dataset = pd.concat([test_audio_aug_feature_df, test_ragas_df], axis=1)
prefix = "mfcc_"
new_columns = [prefix + str(col) for col in test_dataset.columns[:-1]]
test_dataset.columns = new_columns + [test_dataset.columns[-1]]
test_audio_aug_feature_dataset.columns = new_columns + [test_audio_aug_feature_dataset.columns[-1]]
test_dataset['Ragas'] = test_dataset['Ragas'].str.replace('\d+', '', regex=True)
test_dataset['Ragas'] = test_dataset['Ragas'].replace(['bhoop', 'bhoopali'], 'Bhoopali')
test_dataset['Ragas'] = test_dataset['Ragas'].replace(['DKanada', 'darbari'], 'Darbari')
test_dataset['Ragas'] = test_dataset['Ragas'].str.capitalize()
test_audio_aug_feature_dataset['Ragas'] = test_audio_aug_feature_dataset['Ragas'].str.replace('\d+', '', regex=True)
test_audio_aug_feature_dataset['Ragas'] = test_audio_aug_feature_dataset['Ragas'].replace(['bhoop', 'bhoopali'], 'Bhoopali')
test_audio_aug_feature_dataset['Ragas'] = test_audio_aug_feature_dataset['Ragas'].replace(['DKanada', 'darbari'], 'Darbari')
test_audio_aug_feature_dataset['Ragas'] = test_audio_aug_feature_dataset['Ragas'].str.capitalize()
test_mel_dataset['Ragas'] = test_mel_dataset['Ragas'].str.capitalize()
test_mel_dataset['Ragas'] = test_mel_dataset['Ragas'].replace('Bhoop', 'Bhoopali')
test_mel_dataset['Ragas'] = test_mel_dataset['Ragas'].replace('Dkanada', 'Darbari')
print("\nDataset created successfully!")


test_dataset.to_csv("mfcc_test_dataset.csv", index=False)
test_mel_dataset.to_csv("mel_test_dataset.csv", index=False)
test_audio_aug_feature_dataset.to_csv("mfcc_test_dataset.csv", index=False)


