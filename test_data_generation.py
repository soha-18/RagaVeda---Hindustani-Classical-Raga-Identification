import librosa
import numpy as np
import os
import pandas as pd
from feature_extraction import extract_mfcc_feature_vector
from feature_extraction import audio_augmentation

test_features = []
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
                    feature_vector = extract_mfcc_feature_vector(file_path)
                    test_features.append(feature_vector)
                    test_ragas.append(labels)
                    #print(f"File Processed")

                except Exception as e:
                    print(f"Error processing {file}: {e}")

## Convert the lists to a Pandas DataFrame
test_feature_df = pd.DataFrame(test_features)
test_ragas_df = pd.DataFrame({'Ragas': test_ragas})
test_dataset = pd.concat([test_feature_df, test_ragas_df], axis=1)
prefix = "mfcc_"
new_columns = [prefix + str(col) for col in test_dataset.columns[:-1]]
test_dataset.columns = new_columns + [test_dataset.columns[-1]]
test_dataset['Ragas'] = test_dataset['Ragas'].str.replace('\d+', '', regex=True)
test_dataset_extracted = test_dataset[(test_dataset['Ragas'] == 'bhoop') | (test_dataset['Ragas'] == 'bhoopali') | (test_dataset['Ragas'] == 'yaman')]
print("\nDataset created successfully!")
#print(dataset.head())

# Convert dataset to csv
test_dataset_extracted.to_csv("test_dataset.csv", index=False)
