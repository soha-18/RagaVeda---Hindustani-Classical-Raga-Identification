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


