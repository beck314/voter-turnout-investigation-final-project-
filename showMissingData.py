import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r"C:\Users\becke\Downloads\useful databases\secondTestRegionSort.csv"
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

print(df.shape)
df2test = df.loc[:,'year ':].astype(float)

for i in range(42):
    print(df2test.columns.values[i])
    print((np.count_nonzero(np.isnan(df2test.iloc[:,i]))/np.count_nonzero(df2test.iloc[:,i]))*100)

missingArr = np.array([1.34,1.34,17.72,19.91,19.91,19.91,3.42,19.24,17.66,68.09,18.94,72.84,73.93,19.55,84.29,17.36,40.26,33.86,79.54,79.54,35.38,35.38,35.38,35.38,83.68,83.98,82.75,84.24,83.98,83.98,83.98,83.98,6.46,17.36,6.46,43.91,43.91,82.95,82.95,45.25,1.64])

plt.hist(missingArr, bins=10, range=(0,100))
plt.show()