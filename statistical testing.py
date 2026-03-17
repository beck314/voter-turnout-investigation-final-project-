import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
import matplotlib.pyplot as plt

file_path = r"C:\Users\becke\Downloads\useful databases\secondTest.csv" ##first test set had commas so was being treated as strings
file_path2 = r"C:\Users\becke\Downloads\useful databases\secondTestRegionSort.csv"
df2 = pd.read_csv(file_path2)


df2test = df2.loc[:,'year ':].astype(float)
'''
##local standard dev for each region and predictor 

sum=np.zeros(42)
total = 0
loc = 0

while(loc<1641):
    #print(loc)
    counter = 1
    while(df2test.loc[loc+counter, 'year '] >= df2test.loc[loc+counter-1, 'year ']):
        counter+=1
       # if counter>6: print(df2test.loc[loc+counter, 'year '],  df2test.loc[loc, 'year '])
    #print(df2test.loc[loc:loc+counter-1, 'year '])
    #print(df2test.loc[loc:loc+counter-1,'year ':].std(axis=0, skipna=True))
    sum = np.nansum(np.stack((df2test.loc[loc:loc+counter-1,'year ':].std(axis=0, skipna=True), sum), axis=0), axis=0)
    #print(counter)
    loc = loc+counter
    total+=1

dfi = pd.DataFrame(np.round(np.divide(sum, total), 3))
print(dfi)

#res = s.NormalDist(mu=3.016, sigma=0.13)
#print(res.samples(100,seed=42))

missingArr = np.array([1.34,1.34,17.72,19.91,19.91,19.91,3.42,19.24,17.66,68.09,18.94,72.84,73.93,19.55,84.29,17.36,40.26,33.86,79.54,79.54,35.38,35.38,35.38,35.38,83.68,83.98,82.75,84.24,83.98,83.98,83.98,83.98,6.46,17.36,6.46,43.91,43.91,82.95,82.95,45.25,1.64])

plt.hist(missingArr, bins=10, range=(0,100))
plt.show()

for i in range(42):
    print(df2test.columns.values[i])
    print((np.count_nonzero(np.isnan(df2test.iloc[:,i]))/np.count_nonzero(df2test.iloc[:,i]))*100)

'''


##KNN imputing

imputer = KNNImputer(n_neighbors=2, weights='uniform')
df_imputed = imputer.fit_transform(df2test)
#print("Data after KNN Imputation:\n", df_imputed)

print(np.array(df2test.loc[35,:]))
print(df_imputed[35])

print(df_imputed[:,0])

pd_imp = pd.DataFrame(df_imputed, columns=df2test.columns.values)
print(pd_imp.columns.values)

#for i in range(42):
    #print(df_imputed.columns.values[i])
    #print((np.count_nonzero(np.isnan(df_imputed[:,i]))/np.count_nonzero(df_imputed[:,i]))*100)


'''
##replacing NaN with mean of column

meanz = np.nanmean(df2test, axis=0, keepdims=True)
stdz = np.array(df2test.std(axis=0))

print(meanz)
print(stdz)

meanzNoNAN = np.nan_to_num(df2test, nan=meanz)


print(np.array(df2test.loc[35,:]))
print(np.array(meanzNoNAN[35,:]))
'''
