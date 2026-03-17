import math
from asyncio.windows_events import NULL
from email.policy import default

import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

file_path2 = r"C:\Users\becke\Downloads\useful databases\secondTestRegionSort.csv"
df2 = pd.read_csv(file_path2)

df2test = df2.loc[:,'year ':].astype(float)

imputer = KNNImputer(n_neighbors=2, weights='uniform')
df_imputed = imputer.fit_transform(df2test)
pd_imp = pd.DataFrame(df_imputed, columns=df2test.columns.values)



"""
##average of local standard deviation
df2test = pd_imp.loc[:,'year ':].astype(float)
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
"""
#dataToPlot = pd_imp.loc[:, 'region_name':'year '] +  pd_imp.loc[:, 'unemployment %']

fig, ax = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)


ax[0,0].hist(pd_imp.loc[:, 'unemployment %'], bins=50, color='blue', edgecolor='black')#.xlabel("unemployment %")
ax[0,0].set_title("unemployment %")
ax[0,0].set_xlabel("%")
ax[0,1].hist(pd_imp.loc[:, 'population % 16-64'], bins=50, color='red', edgecolor='black')#.xlabel("population %")
ax[0,1].set_title("population % 16-64")
ax[0,1].set_xlabel('%')
ax[1,0].hist(pd_imp.loc[:, 'population-density (persons per km^2)'], bins=50, color='green', edgecolor='black')#.xlabel("persons per km^2")
ax[1,0].set_title("population-density (persons/km^2)")
ax[1,0].set_xlabel("persons per km^2")
ax[1,1].hist(pd_imp.loc[:, 'GDP per head £'], bins=50, color='yellow', edgecolor='black')#.xlabel("GDP per head £")
ax[1,1].set_title("GDP per head £")
ax[1,1].set_xlabel("£")

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(8, 6),constrained_layout=True)
ax[0,0].set_xlabel("£")
ax[0,1].set_xlabel("persons per 100000")
ax[1,0].set_xlabel("additions per 1000")
ax[1,1].set_xlabel("%")

ax[0,0].hist(pd_imp.loc[:, 'GDP per head current value £'], bins=50, color='blue', edgecolor='black')
#ax[0,0].xlabel("GDP current value £")
ax[0,0].set_title("GDP per head current value £")
ax[0,1].hist(pd_imp.loc[:, 'further education/skills participation per 100000'], bins=50, color='red', edgecolor='black')#.xlabel("participation per 100000")
ax[0,1].set_title("further education/skills participation /100000")
ax[1,0].hist(pd_imp.loc[:, 'net additions to housing stock per 1000 dwellings '], bins=50, color='green', edgecolor='black')#.xlabel("additions per 1000 dwellings")
ax[1,0].set_title("additions to housing stock/1000 dwellings")
ax[1,1].hist(pd_imp.loc[:, 'children in relative poverty %'], bins=50, color='yellow', edgecolor='black')#.xlabel("% of children")
ax[1,1].set_title("children in relative poverty %")

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(8, 6),constrained_layout=True)
ax[0,0].set_xlabel("%")
ax[0,1].set_xlabel("%")
ax[1,0].set_xlabel("%")
ax[1,1].set_xlabel("%")

ax[0,0].hist(pd_imp.loc[:, '% of people achieving maths and english gcses by 19 '], bins=50, color='blue', edgecolor='black')#.xlabel("% of people")
ax[0,0].set_title("% achieving maths and english gcses by 19")
ax[0,1].hist(pd_imp.loc[:, '% of people with level 3 qualifications'], bins=50, color='red', edgecolor='black')#.xlabel("% of people"))
ax[0,1].set_title("% people with level 3 qualifications")
ax[1,0].hist(pd_imp.loc[:, '% children with expected reading writing and maths level at ks2'], bins=50, color='green', edgecolor='black')#.xlabel("% of children")
ax[1,0].set_title("% at expected reading, writing and maths level ks2")
ax[1,1].hist(pd_imp.loc[:, '% with no qualifications'], bins=50, color='yellow', edgecolor='black')#.xlabel("% with no qualifications")
ax[1,1].set_title("% with no qualifications")

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(8, 6),constrained_layout=True)
ax[0,0].set_xlabel("%")
ax[0,1].set_xlabel("achievements per 100000")
ax[1,0].set_xlabel("%")
ax[1,1].set_xlabel("%")

ax[0,0].hist(pd_imp.loc[:, '% of obesity in adults '], bins=50, color='blue', edgecolor='black')#.xlabel("% of obesity in adults")
ax[0,0].set_title("% of obesity in adults ")
ax[0,1].hist(pd_imp.loc[:, 'apprenticeship achievements per 100000'], bins=50, color='red', edgecolor='black')#.xlabel("achievements per 100000")
ax[0,1].set_title("apprenticeship achievements per 100,000")
ax[1,0].hist(pd_imp.loc[:, '% children with persistent absence from schools '], bins=50, color='green', edgecolor='black')#.xlabel("% of people smoking")
ax[1,0].set_title("% children with persistent absence from schools ")
ax[1,1].hist(pd_imp.loc[:, '% of people smoking  '], bins=50, color='yellow', edgecolor='black')#.xlabel("% of children with persistent absence")
ax[1,1].set_title('% of people smoking ')

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(8, 6),constrained_layout=True)
ax[0,0].set_xlabel("deaths per 100000")
ax[0,1].set_xlabel("%")
ax[1,0].set_xlabel("years")
ax[1,1].set_xlabel("years")

ax[0,0].hist(pd_imp.loc[:, 'preventable cardiovascular deaths per 100000'], bins=50, color='blue', edgecolor='black')
ax[0,0].set_title("preventable cardiovascular deaths per 100000")
ax[0,1].hist(pd_imp.loc[:, '% of cancer diagnoses in stage 1 or 2'], bins=50, color='red', edgecolor='black')
ax[0,1].set_title("% of cancer diagnoses in stage 1 or 2")
ax[1,0].hist(pd_imp.loc[:, 'years in good health (male)'], bins=50, color='green', edgecolor='black')
ax[1,0].set_title("years in good health (male)")
ax[1,1].hist(pd_imp.loc[:, 'years in good health (female)'], bins=50, color='yellow', edgecolor='black')
ax[1,1].set_title("years in good health (female)")

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(8, 6),constrained_layout=True)
ax[0,0].set_xlabel("1-10")
ax[0,1].set_xlabel("1-10")
ax[1,0].set_xlabel("1-10")
ax[1,1].set_xlabel("1-10")

ax[0,0].hist(pd_imp.loc[:, 'Average Anxiety out of 10'], bins=50, color='blue', edgecolor='black')
ax[0,0].set_title("Average Anxiety out of 10")
ax[0,1].hist(pd_imp.loc[:, 'Average feeling life is worthwhile out of 10'], bins=50, color='red', edgecolor='black')
ax[0,1].set_title("Average feeling life is worthwhile out of 10")
ax[1,0].hist(pd_imp.loc[:, 'Average happiness out of 10'], bins=50, color='green', edgecolor='black')
ax[1,0].set_title("Average happiness out of 10")
ax[1,1].hist(pd_imp.loc[:, 'Average life satisfaction'], bins=50, color='yellow', edgecolor='black')
ax[1,1].set_title("Average life satisfaction")

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(10, 8),constrained_layout=True)
ax[0,0].set_xlabel("minutes")
ax[0,1].set_xlabel("facilities per 10000")
ax[1,0].set_xlabel("museums per 100000")
ax[1,1].set_xlabel("%")

ax[0,0].hist(pd_imp.loc[:, 'time to employment centre by walking/public transport'], bins=50, color='blue', edgecolor='black')
ax[0,0].set_title("time to employment centre by walking/public transport")
ax[0,1].hist(pd_imp.loc[:, 'sports facilities per 10000'], bins=50, color='red', edgecolor='black')
ax[0,1].set_title("sports facilities per 10000")
ax[1,0].hist(pd_imp.loc[:, 'Museums per 100000'], bins=50, color='green', edgecolor='black')
ax[1,0].set_title("Museums per 100000")
ax[1,1].hist(pd_imp.loc[:, '% of people within 30 minute walk of train station'], bins=50, color='yellow', edgecolor='black')
ax[1,1].set_title("% of people within 30 minute walk of train station")

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(8, 6),constrained_layout=True)
ax[0,0].set_xlabel("%")
ax[0,1].set_xlabel("%")
ax[1,0].set_xlabel("%")
ax[1,1].set_xlabel("%")

ax[0,0].hist(pd_imp.loc[:, '% of people within 30 minute walk to library'], bins=50, color='blue', edgecolor='black')
ax[0,0].set_title("% of people within 30 minute walk to library")
ax[0,1].hist(pd_imp.loc[:, '% of people engaged in the arts '], bins=50, color='red', edgecolor='black')
ax[0,1].set_title("% of people engaged in the arts ")
ax[1,0].hist(pd_imp.loc[:, '% of people who visited a heritage site '], bins=50, color='green', edgecolor='black')
ax[1,0].set_title("% of people who visited a heritage site ")
ax[1,1].hist(pd_imp.loc[:, '% people who visited a library '], bins=50, color='yellow', edgecolor='black')
ax[1,1].set_title("% people who visited a library ")

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(8, 6),constrained_layout=True)
ax[0,0].set_xlabel("count")
ax[0,1].set_xlabel("tCO2e")
ax[1,0].set_xlabel("1e6")
ax[1,1].set_xlabel("count")

ax[0,0].hist(pd_imp.loc[:, 'average number of parks and playing fields within 1000m'], bins=50, color='blue', edgecolor='black')
ax[0,0].set_title("average number of parks/fields in 1000m")
ax[0,1].hist(pd_imp.loc[:, 'greenhouse gas emmisions tCO2e'], bins=50, color='red', edgecolor='black')
ax[0,1].set_title("greenhouse gas emmisions tCO2e")
ax[1,0].hist(pd_imp.loc[:, 'average combined size of parks playing fields and public gardens'], bins=50, color='green', edgecolor='black')
ax[1,0].set_title("average combined size of public green space")
ax[1,1].hist(pd_imp.loc[:, 'total crimes '], bins=50, color='yellow', edgecolor='black')
ax[1,1].set_title("total crimes")

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(8, 6),constrained_layout=True)
ax[0,0].set_xlabel("count")
ax[0,1].set_xlabel("%")
ax[1,0].set_xlabel("%")
ax[1,1].set_xlabel("")

ax[0,0].hist(pd_imp.loc[:, 'crimes against people'], bins=50, color='blue', edgecolor='black')
ax[0,0].set_title("crimes against people")
ax[0,1].hist(pd_imp.loc[:, '% religious'], bins=50, color='red', edgecolor='black')
ax[0,1].set_title("% people who are religious ")
ax[1,0].hist(pd_imp.loc[:, '% no religion'], bins=50, color='green', edgecolor='black')
ax[1,0].set_title("% people not religious")

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(8, 6),constrained_layout=True)
ax[0,0].set_xlabel("million £ 2025/26 rates unweighted")
ax[0,1].set_xlabel("")
ax[1,0].set_xlabel("")
ax[1,1].set_xlabel("count")

ax[0,0].hist(pd_imp.loc[:, 'yearly_spending(million £ 2025/26 rates unweighted)'], bins=50, color='blue', edgecolor='black')
ax[0,0].set_title("yearly_spending")
ax[1,1].hist(pd_imp.loc[:, 'available_seats'], bins=50, color='yellow', edgecolor='black')
ax[1,1].set_title("available_seats")

plt.show()
