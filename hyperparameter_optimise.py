import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

file_path = r"C:\Users\becke\Downloads\useful databases\gtTurnout.csv"
file_path2 = r"C:\Users\becke\Downloads\useful databases\secondTest.csv"

df2 = pd.read_csv(file_path2)

df2.rename(columns={"area_code": "Region Code"}, inplace=True)
df2.rename(columns={"year ": "Year"}, inplace=True)
df2test = df2.loc[:,'available_seats':].astype(float)

turnoutTable = pd.read_csv(file_path)
turnoutTable.rename(columns={"Region Code": "Region Code"}, inplace=True)
turnoutTable.rename(columns={"Year": "Year"}, inplace=True)
turnouts = turnoutTable.loc[:,'Turnout'].astype(float)

##takes a dataframe an returns one with all values filled in with the KNN average
def myImpute(dataf):
    imputer = KNNImputer(n_neighbors=2, weights='uniform')
    df_imputed = imputer.fit_transform(dataf)
    return pd.DataFrame(df_imputed, columns=dataf.columns.values)

pd_imp = myImpute(df2test)

trueGT = turnoutTable.loc[:,"Turnout"]

merged = pd.merge(df2, turnoutTable, on=["Year", "Region Code"], how="inner")

pd_imp["Turnout"] = merged.loc[:,'Turnout'].astype(float)
fin_data = pd_imp.dropna()
print(fin_data.shape)
preds = fin_data.loc[:,:'% religious']
gts = fin_data.loc[:,'Turnout']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(preds)

def mySplit(datap, gt, ratio):
    split = int(len(datap) * ratio)
    print(split)
    return datap[:split,:], datap[split:, :], gt[:split], gt[split:]

pca = PCA(n_components=3)
pcs = pca.fit_transform(scaled_data)
print(pcs.shape)

trainPred, testPred, trainGT, testGT = mySplit(pcs, gts, 0.7)

grid_svr = {
    "kernel": ["linear", "rbf", "sigmoid"],
    "C": [0.1,1,10,100],
    "gamma": [0.01, 0.1, 1, 'scale'],
    "epsilon": [0.01, 0.1, 0.5]
}

grid_ridge = {
    "alpha": [0.1,1,10,100]
}

grid_rfr = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}



grid_search = GridSearchCV(SVR(), grid_svr, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(trainPred, trainGT)
print("best params: ")
print(grid_search.best_params_)
##{'C': 1, 'epsilon': 0.01, 'gamma': 0.01}  linear
##{'C': 10, 'epsilon': 0.01, 'gamma': 0.01}  rbf
#{'C': 0.1, 'epsilon': 0.1, 'gamma': 0.01}  sigmoid


'''
grid_search = GridSearchCV(Ridge(), grid_ridge, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(trainPred, trainGT)
print("best params: ")
print(grid_search.best_params_)
##{'alpha': 10}
'''
'''
grid_search = GridSearchCV(RandomForestRegressor(), grid_rfr, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(trainPred, trainGT)
print("best params: ")
print(grid_search.best_params_)
##{'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
'''