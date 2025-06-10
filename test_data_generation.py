import librosa
import numpy as np
import os
import pandas as pd
from feature_extraction import extract_mfcc_feature_vector
from feature_extraction import audio_augmentation
from mel_spectogram_datasets import extract_features_mel

test_features = []
mel_test_features = []
test_ragas = []

def extract_label(file_path):
    label = os.path.splitext(os.path.basename(file_path))[0]
    return label

audio_folder = 'Test'

for root, dirs, files in os.walk(audio_folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    labels = extract_label(file_path)
                    y, sr = librosa.load(file_path, duration=30)
                    y_stretched = audio_augmentation(y, sr, augmentation_type='time_stretch')
                    y_shifted = audio_augmentation(y_stretched, sr, augmentation_type='pitch_shift')
                    aug_audio = audio_augmentation(y_shifted, sr, augmentation_type='add_noise')
                    feature_vector = extract_mfcc_feature_vector(aug_audio, sr)
                    feature_vector_mel = extract_features_mel(y, sr)
                    test_features.append(feature_vector)
                    test_ragas.append(labels)
                    mel_test_features.append([feature_vector_mel, labels])
                    #print(f"File Processed")

                except Exception as e:
                    print(f"Error processing {file}: {e}")

## Convert the lists to a Pandas DataFrame
test_mel_dataset = pd.DataFrame(mel_test_features, columns = ("Mel_Features", "Ragas"))
test_mel_dataset['Ragas'] = test_mel_dataset['Ragas'].str.replace('\d+', '', regex=True)
# test_feature_df = pd.DataFrame(test_features)
# test_ragas_df = pd.DataFrame({'Ragas': test_ragas})
# test_dataset = pd.concat([test_feature_df, test_ragas_df], axis=1)
# prefix = "mfcc_"
# new_columns = [prefix + str(col) for col in test_dataset.columns[:-1]]
# test_dataset.columns = new_columns + [test_dataset.columns[-1]]
# test_dataset['Ragas'] = test_dataset['Ragas'].str.replace('\d+', '', regex=True)
# test_dataset['Ragas'] = test_dataset['Ragas'].replace(['bhoop', 'bhoopali'], 'Bhoopali')
# test_dataset['Ragas'] = test_dataset['Ragas'].replace(['DKanada', 'darbari'], 'Darbari')
# test_dataset['Ragas'] = test_dataset['Ragas'].str.capitalize()
test_mel_dataset['Ragas'] = test_mel_dataset['Ragas'].str.capitalize()
test_mel_dataset['Ragas'] = test_mel_dataset['Ragas'].replace('Bhoop', 'Bhoopali')
test_mel_dataset['Ragas'] = test_mel_dataset['Ragas'].replace('Dkanada', 'Darbari')
print("\nDataset created successfully!")
#print(dataset.head())

# Convert dataset to csv
test_mel_dataset.to_csv("test_mel_dataset.csv", index=False)
# test_dataset.to_csv("test_dataset.csv", index=False)
