import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
import matplotlib.pyplot as plt

file_path = r"C:\Users\becke\Downloads\useful databases\secondTest.csv" ##first test set had commas so was being treated as strings
file_path2 = r"C:\Users\becke\Downloads\useful databases\secondTestRegionSort.csv"
df2 = pd.read_csv(file_path2)


df2test = df2.loc[:,'year ':].astype(float)

##local standard dev for each region and predictor 

sum=np.zeros(42)
total = 0
loc = 0


#print(df2test.info())
##global standard deviation of the data
print((df2test.std(axis=0, skipna=True)/df2test.mean(axis=0, skipna=True))*100)

##scaled standard deviation as a percentage of the mean
while(loc<1641):
    #print(loc)
    counter = 1
    while(df2test.loc[loc+counter, 'year '] >= df2test.loc[loc+counter-1, 'year ']):
        counter+=1
       # if counter>6: print(df2test.loc[loc+counter, 'year '],  df2test.loc[loc, 'year '])
    #print(df2test.loc[loc:loc+counter-1, 'year '])
    #print(df2test.loc[loc:loc+counter-1,'year ':].std(axis=0, skipna=True))
    fsum = np.nansum(np.stack((df2test.loc[loc:loc+counter-1,'year ':].std(axis=0, skipna=True), sum), axis=0), axis=0)
    #print(counter)
    loc = loc+counter
    total+=1

dfi = pd.DataFrame(np.round(np.divide(fsum, total), 3))
print(dfi)



