import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn.externals.array_api_compat import numpy
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error
from scipy.stats import spearmanr

file_path = r"C:\Users\becke\Downloads\useful databases\gtTurnout.csv"
file_path2 = r"C:\Users\becke\Downloads\useful databases\secondTest.csv"

df2 = pd.read_csv(file_path2)

df2.rename(columns={"area_code": "Region Code"}, inplace=True)
df2.rename(columns={"year ": "Year"}, inplace=True)
df2test = df2.loc[:,'available_seats':].astype(float)


turnoutTable = pd.read_csv(file_path)
print(turnoutTable.columns.values)
turnoutTable.rename(columns={"Region Code": "Region Code"}, inplace=True)
turnoutTable.rename(columns={"Year": "Year"}, inplace=True)
turnouts = turnoutTable.loc[:,'Turnout'].astype(float)
'''
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.hist(turnouts, bins=25, color='blue', edgecolor='black')
ax.set_xlabel("Turnout")
ax.set_ylabel("Count")
ax.set_title("Histogram to show Regional Voter Turnout")

plt.show()
'''


##takes a dataframe an returns one with all values filled in with the KNN average
def myImpute(dataf):
    imputer = KNNImputer(n_neighbors=2, weights='uniform')
    df_imputed = imputer.fit_transform(dataf)
    return pd.DataFrame(df_imputed, columns=dataf.columns.values)

pd_imp = myImpute(df2test)

trueGT = turnoutTable.loc[:,"Turnout"]

#print(df2.shape)
#print(turnoutTable.shape)
#print(pd_imp.shape)

merged = pd.merge(df2, turnoutTable, on=["Year", "Region Code"], how="inner")
#print(merged.columns[45:47])


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

#trainPred = trainPred[:,:20:2]
#testPred = testPred[:,:20:2]

pca = PCA(n_components=3)
pcs = pca.fit_transform(scaled_data)

'''
trainPred, testPred, trainGT, testGT = mySplit(pcs, gts, 0.7)
print(trainPred.shape, trainGT.shape)
print(testPred.shape, testGT.shape)
'''

models = {
    "RandomForest":RandomForestRegressor(bootstrap= True, max_depth= 10, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 100),
    "SVR": SVR(kernel="rbf", C= 10, epsilon= 0.01, gamma= 0.01),
    "Ridge": Ridge(alpha= 10)
}

for name, model in models.items():
    scores = cross_val_score(
        model, scaled_data, gts,
        cv=5,
        scoring='neg_mean_squared_error'
    )

    mse_scores = -scores  # convert back to positive
    print(f"{name} MSE: {mse_scores.mean():.5f} ± {mse_scores.std():.5f}")

def spearman_corr(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

spearman_scorer = make_scorer(spearman_corr)

for name, model in models.items():
    scores = cross_val_score(
        model, scaled_data, gts,
        cv=5,
        scoring=spearman_scorer
    )

    print(f"{name} Spearman rho: {scores.mean():.3f} ± {scores.std():.3f}")