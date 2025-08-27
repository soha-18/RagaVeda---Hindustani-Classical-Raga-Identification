import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re

data = []
feature_parts = []

with open('mel_dataset.csv', 'r') as f:
    for line in f:
        # Split the line by the comma
        parts = line.strip().split(',')
        features_str = parts[0].strip('[]')
        feature_parts.append(features_str)

X = np.array(feature_parts)
print(X)

        # try:
        #     numbers = re.findall(r'-?\d+\.?\d*', features_str)
        #     features = [float(num) for num in numbers]
        #     features = np.array(features)
        #     #print(features)
        #     # Clean up the string to make it a valid list representation
        #     features_str = features_str.replace('...', '').replace('[', '').replace(']', '')
        #     features = np.fromstring(features_str, sep=' ')
        #     data.append({'features': features})
        # except ValueError:
        #     print(f"Skipping malformed line: {line}")


