import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

file_path = r"C:\Users\becke\Downloads\useful databases\gtTurnout.csv"
file_path2 = r"C:\Users\becke\Downloads\useful databases\secondTest.csv"

predictors = pd.read_csv(file_path2)

predictors.rename(columns={"region_name": "Region Name"}, inplace=True)
predictors.rename(columns={"area_code": "Region Code"}, inplace=True)
predictors.rename(columns={"year ": "Year"}, inplace=True)
predictors_allnum = predictors.loc[:,'available_seats':].astype(float)

turnoutTable = pd.read_csv(file_path)
turnoutTable.rename(columns={"Region Code": "Region Code"}, inplace=True)
turnoutTable.rename(columns={"Year": "Year"}, inplace=True)
turnouts = turnoutTable.loc[:,'Turnout'].astype(float)



def myImpute(dataf):
    imputer = KNNImputer(n_neighbors=2, weights='uniform')
    df_imputed = imputer.fit_transform(dataf)
    return pd.DataFrame(df_imputed, columns=dataf.columns.values)

pd_imp = myImpute(predictors_allnum)
predictors = predictors.loc[:, :'Year']
predictors = pd.concat([predictors, pd_imp], axis=1)

trueGT = turnoutTable.loc[:,"Turnout"]

#print(df2.shape)
#print(turnoutTable.shape)
#print(pd_imp.shape)

merged = pd.merge(predictors, turnoutTable, on=["Year", "Region Code"], how="inner")
noNan = merged.dropna()

pTargets = pd.DataFrame(columns=["region_type", "Turnout"])
pTargets['region_type'] = noNan.loc[:,'region_type']
pTargets['Turnout'] = noNan.loc[:,'Turnout']

pPreds = noNan.loc[:,:'% religious']

metroPred = pPreds.loc[pPreds["region_type"] == "metropolitan city council"]
metroPred = metroPred.loc[:, "available_seats":]
metroTargets = pTargets.loc[pTargets["region_type"] == "metropolitan city council"]

unitary_districtPred = pd.concat([pPreds.loc[pPreds["region_type"] == "district/unitary council"], pPreds.loc[pPreds["region_type"] == "district/unitary authority"]])
unitary_districtPred = unitary_districtPred.loc[:, "available_seats":]
unitary_districtTargets = pd.concat([pTargets.loc[pTargets["region_type"] == "district/unitary council"], pTargets.loc[pTargets["region_type"] == "district/unitary authority"]])

countiesPred = pPreds.loc[pPreds["region_type"] == "county council"]
countiesPred = countiesPred.loc[:, "available_seats":]
countiesTargets = pTargets.loc[pTargets["region_type"] == "county council"]

def mySplit(datap, gt, ratio):
    split = int(len(datap) * ratio)
    print(split)
    return datap[:split], datap[split:], gt[:split], gt[split:]

'''
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3)),
    ('regression', SVR(kernel="rbf", C= 10, epsilon= 0.01, gamma= 0.01)),
    ])

metTrainP, metTestP, metTrainT, metTestT = mySplit(metroPred, metroTargets.loc[:,"Turnout"], ratio=0.7)
'''

scaler = StandardScaler()
metroScaled = scaler.fit_transform(metroPred)
districtScaled = scaler.fit_transform(unitary_districtPred)
countiesScaled = scaler.fit_transform(countiesPred)

pca = PCA(n_components=3)
metroReduced = pca.fit_transform(metroScaled)
districtReduced = pca.fit_transform(districtScaled)
countiesReduced = pca.fit_transform(countiesScaled)

metTrainP, metTestP, metTrainT, metTestT = mySplit(metroReduced, metroTargets.loc[:,"Turnout"], ratio=0.7)
districtTrainP, districtTestP, districtTrainT, districtTestT = mySplit(districtReduced, unitary_districtTargets.loc[:,"Turnout"], ratio=0.7)
countiesTrainP, countiesTestP, countiesTrainT, countiesTestT = mySplit(countiesReduced, countiesTargets.loc[:,"Turnout"], ratio=0.7)

print(countiesTrainP.shape)
print(countiesTestP.shape)
print(countiesTrainT.shape)
print(countiesTestT.shape)

'''
pipeline.fit(metTrainP, metTrainT)
print(pipeline.score(metTestP, metTestT))
'''

svr = SVR(kernel="rbf", C= 10, epsilon= 0.01, gamma= 0.01)
svr.fit(metTrainP, metTrainT)
metPreds = svr.predict(metTestP)
countiesPredsL = svr.predict(countiesTestP)

svr.fit(districtTrainP, districtTrainT)
districtPreds = svr.predict(districtTestP)

svr.fit(countiesTrainP, countiesTrainT)
countiesPreds = svr.predict(countiesTestP)

print("metropolitan rmse: ")
print(root_mean_squared_error(metPreds, metTestT))
print("unitary/district rmse: ")
print(root_mean_squared_error(districtPreds, districtTestT))
print("counties rmse: ")
print(root_mean_squared_error(countiesPreds, countiesTestT))
print("countiesP rmse: ")
print(root_mean_squared_error(countiesPredsL, countiesTestT))


fig, ax = plt.subplots(2, 2, figsize=(8, 6))

intLinex = np.arange(0.25,0.45,0.01)
intLiney = intLinex

ax[0,0].scatter(metPreds, metTestT, color='blue', marker='o')
ax[0,0].scatter(intLinex, intLiney, color='red', marker='x')
ax[0,0].set_title("Metropolitan")
ax[0,0].set_xlabel("prediction")
ax[0,0].set_ylabel("target")

ax[0,1].scatter(districtPreds, districtTestT, color='blue', marker='o')
ax[0,1].scatter(intLinex, intLiney, color='red', marker='x')
ax[0,1].set_title("Unitary/District")
ax[0,1].set_xlabel("prediction")
ax[0,1].set_ylabel("target")

ax[1,0].scatter(countiesPredsL, countiesTestT, color='blue', marker='o')
ax[1,0].scatter(intLinex, intLiney, color='red', marker='x')
ax[1,0].set_title("Counties")
ax[1,0].set_xlabel("prediction")
ax[1,0].set_ylabel("target")


plt.show()

