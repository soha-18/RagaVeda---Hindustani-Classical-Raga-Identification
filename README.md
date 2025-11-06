# Hindustani-Classical-Raga-Identification

This project focuses on identification and classification of hindustani classical ragas using Machine Learning techniques. 
The Mel-Frequency Cepstral Coefficients (MFCCs) will be extracted from various audio recordings, which will then be used to train and evaluate CNN to accurately identify and distinguish between different ragas.
MFCCs are highly effective in representing the timbral and spectral characteristics of sound. 
The goal is to develop a robust system for raga recognition, which has potential applications in music information retrieval, music education, and cultural preservation.

In the initial phase we are taregting on the below Ragas to develop our CNN model.
- Asavari
- Bageshree
- Bhairava
- Bhairavi
- Bhoopali
- Darbari
- Malkauns
- Sarang
- Yaman

In the later phase we will incorporate more hindustani ragas from hugging face to increase its flexibility and usability. It also helps to assert the accuracy of the model to be able to detect the ragas.

## Model Description
This project comprises model structure based on two specific audio characteristics.
- MFCC coefficient
- Mel Spectogram 

We preprocess our audio data using MFCC coefficients and split into training and validation dataset for our 2D CNN model.
In addition to that, this project consists of a third structure with some audio augmented features using the MFCC coefficient to compare the performance with the MFCC model.

## Project Workflow
```mermaid
graph TD
    A[Raw Audio Files (.wav)] --> B(Pre-processing: Segment & Standardize);
    B --> C(Feature Extraction: MFCCs);
    C --> D{Dataset (MFCCs + Labels)};
    D --> E[Train/Test Split];
    E --> F(Train CNN Model);
    F --> G(Evaluate Model);
    G --> H[Classified Raga];
```

1. Data Collection:Collect the audio recordings (.wav, .mp3) of the Hindustani classical ragas for classification.
2. Audio Preprocessing:Read the audio files using librosa and converts a list of variable-length feature sequences into a uniform, zero-padded Pandas DataFrame, making it suitable for machine learning inputs.
3. Feature Extraction:For each audio segment, extracts MFCCs along with their first (delta) and second (delta-delta) derivatives from an audio file, then computes their mean over time to produce a single, representative feature vector.This converts the complex audio signal into a 2D representation.
4. Dataset Preparation:Saving the extracted MFCCs (the features) and their corresponding raga labels (the targets) as csv file for easy loading during training.
5. Data Splitting:The complete dataset has been divided into Training Set (e.g., 80%) for training the CNN model and Validation Set (e.g., 10%) to monitor training progress, tune hyperparameters, and prevent overfitting.
6. Model Architecture (CNN):Define the Convolutional Neural Network (CNN) architecture using a framework like TensorFlow/Keras.Conv2D layers (to detect patterns in the MFCCs).MaxPooling2D layers (to reduce dimensionality).Dropout layers (to prevent overfitting).A Flatten layer (to transition from 2D feature maps to a 1D vector).
Dense (fully connected) layers for classification.A final Dense output layer with a softmax activation function to output a probability for each raga class.

7. Model Training:
Compile the model, specifying an optimizer (e.g., Adam), a loss function (e.g., sparse_categorical_crossentropy or categorical_crossentropy), and evaluation metrics (e.g., accuracy).Train the model by "fitting" it to the training dataset, using the validation set to check for improvement after each epoch.
