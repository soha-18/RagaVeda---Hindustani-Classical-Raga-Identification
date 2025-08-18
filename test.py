import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re

# data = []
# with open('mel_dataset.csv', 'r') as f:
#     for line in f:
#         # Split the line by the comma
#         parts = line.strip().split(',')
#         features_str = parts[0].strip('[]')
#         #label = parts[1].strip()

#         try:
#             numbers = re.findall(r'-?\d+\.?\d*', features_str)
#             features = [float(num) for num in numbers]
#             features = np.array(features)
#             # Clean up the string to make it a valid list representation
#             #features_str = features_str.replace('...', '').replace('[', '').replace(']', '')
#             #features = np.fromstring(features_str, sep=' ')
#             data.append({'features': features})  #, 'label': label})
#         except ValueError:
#             print(f"Skipping malformed line: {line}")

# Load the .npz file
data = np.load('mel_features.npz', allow_pickle=True)

# Access the arrays by their key names
X_loaded = data['arr1']
y_loaded = data['arr2']

# You can now use X_loaded and y_loaded for further preprocessing or model training
print("Shape of loaded X:", X_loaded.shape)
print("Shape of loaded y:", y_loaded.shape)

data.close()
print(X_loaded.dtype)
X_loaded = np.vstack(X_loaded)
X_loaded = X_loaded.astype(float)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_loaded)
# print("Shape of scaled X:", X_scaled.shape)
# df = pd.DataFrame(data)
# #print(df)
# X = df['features'].values
# print(X.shape)
# print(X[23])
# #print(type(X[1]))
