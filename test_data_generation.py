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
                    #print(f"File Processed")

                except Exception as e:
                    print(f"Error processing {file}: {e}")

## Convert the lists to a Pandas DataFrame
test_mel_dataset = pd.DataFrame(mel_test_features, columns = ("Mel_Features", "Ragas"))
test_mel_dataset['Ragas'] = test_mel_dataset['Ragas'].str.replace('\d+', '', regex=True)
test_feature_df = pd.DataFrame(mfcc_test_features)
test_ragas_df = pd.DataFrame({'Ragas': test_ragas})
test_dataset = pd.concat([test_feature_df, test_ragas_df], axis=1)
prefix = "mfcc_"
new_columns = [prefix + str(col) for col in test_dataset.columns[:-1]]
test_dataset.columns = new_columns + [test_dataset.columns[-1]]
test_dataset['Ragas'] = test_dataset['Ragas'].str.replace('\d+', '', regex=True)
test_dataset['Ragas'] = test_dataset['Ragas'].replace(['bhoop', 'bhoopali'], 'Bhoopali')
test_dataset['Ragas'] = test_dataset['Ragas'].replace(['DKanada', 'darbari'], 'Darbari')
test_dataset['Ragas'] = test_dataset['Ragas'].str.capitalize()
test_mel_dataset['Ragas'] = test_mel_dataset['Ragas'].str.capitalize()
test_mel_dataset['Ragas'] = test_mel_dataset['Ragas'].replace('Bhoop', 'Bhoopali')
test_mel_dataset['Ragas'] = test_mel_dataset['Ragas'].replace('Dkanada', 'Darbari')
print("\nDataset created successfully!")
#print(dataset.head())

if __name__ == "__main__":
    while True:
        print("\n--- Select a test dataset type to create ---")
        print("1. MFCC Feature")
        print("2. MFCC features with Audio Augmentation")
        print("3. Mel Spectogram")
        print("4. Exit")

        feature_selection = input("Enter your choice (1-4): ")

        if feature_selection == '1':
            dataset = create_mfcc_dataset()
            file_name = "mfcc_test_dataset.csv"
        elif feature_selection == '2':
            dataset = create_mfcc_dataset_with_audio_aug()
            file_name = "audio_aug_mfcc_test_dataset.csv"
        elif feature_selection == '3':
            dataset = create_melSpectogram_dataset()
            file_name = "mel_test_dataset.csv"
        elif feature_selection == '4':
            print("Exiting from feature extraction. Goodbye!")
            break
        else:
            print("Invalid selection. Please enter a number between 1 and 4.")
            dataset = None

        if dataset is not None:
            print("\nDataset created successfully!")
            save_option = input("\nDo you want to save this dataset to a CSV file? (yes/no): ").lower()
            if save_option == 'yes':
                # file_name = input("Enter filename: ")
                try:
                    dataset.to_csv(file_name, index=False)
                    print(f"Dataset saved to {file_name}")
                except Exception as e:
                    print(f"Error saving file: {e}")

