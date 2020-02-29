import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('test.csv')
data = data.round(3)
data.fillna(0)
data = data.drop(data.index[1152])
scale = MinMaxScaler()
data_scaled = scale.fit_transform(data)
data = np.genfromtxt('test.csv', delimiter=',')

print('means', data_scaled.mean(axis=1))
print('std', data_scaled.std(axis=1))
print(data_scaled.min(axis=1))
print(data_scaled.max(axis=1))

