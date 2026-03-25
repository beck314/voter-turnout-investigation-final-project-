import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

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
print(pcs.shape)

#print(trainPred.shape)
'''
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()
'''
trainPred, testPred, trainGT, testGT = mySplit(pcs, gts, 0.7)
print(trainPred.shape, trainGT.shape)
print(testPred.shape, testGT.shape)

##feature selection with SVR (can be changed to any other model) should be compared to limit overfitting
##will be trained on the ?whole set? so models which do not predict well for large dimensionality may be discarded
##random forest performs quite well but so does ridge
#svr_linear = SVR(kernel='linear', max_iter=1000)
#svr_linear.fit(trainPred, trainGT)

'''

rfe = RFE(estimator=Ridge(), n_features_to_select=10)
rfe.fit(trainPred, trainGT)
y_pred = rfe.predict(testPred)
print("Selected features:", rfe.support_)
#print("Test Accuracy:", accuracy_score(testGT, y_pred))

reduced_train_set = pd.DataFrame()
reduced_test_set = pd.DataFrame()

for i in range(len(rfe.support_)):
    if rfe.support_[i]:
        print(pd_imp.columns.values[i])
        reduced_train_set[pd_imp.columns.values[i]] = trainPred[:,i]
        reduced_test_set[pd_imp.columns.values[i]] = testPred[:,i]

trainPred = reduced_train_set
testPred = reduced_test_set
'''


svr_linear = SVR(kernel="rbf", C= 10, epsilon= 0.01, gamma= 0.01)
svr_linear.fit(trainPred, trainGT)
y_pred_svr = svr_linear.predict(testPred)

linReg = Ridge(alpha= 10)
linReg.fit(trainPred, trainGT)
y_pred_linearRidge = linReg.predict(testPred)

rfr = RandomForestRegressor(bootstrap= True, max_depth= 10, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 100)
rfr.fit(trainPred, trainGT)
y_pred_forest = rfr.predict(testPred)

rmse_svr = root_mean_squared_error(testGT, y_pred_svr)
rmse_rf = root_mean_squared_error(testGT, y_pred_forest)
rmse_ridge = root_mean_squared_error(testGT, y_pred_linearRidge)

print("SVR MSE:", rmse_svr)
print("Random Forest MSE:", rmse_rf)
print("Ridge MSE:", rmse_ridge)

trueX = np.arange(len(trainGT))
predX = np.arange(len(testGT))

fig, ax = plt.subplots(2, 2, figsize=(8, 6))

intLinex = np.arange(0.25,0.45,0.01)
intLiney = intLinex


ax[0,0].scatter(y_pred_svr, testGT, color='blue', marker='o')
ax[0,0].scatter(intLinex, intLiney, color='red', marker='x')
ax[0,0].set_title("SVR")

ax[0,1].scatter(y_pred_linearRidge, testGT, color='blue', marker='o')
ax[0,1].scatter(intLinex, intLiney, color='red', marker='x')
ax[0,1].set_title("Linear Ridge")

ax[1,0].scatter(y_pred_forest, testGT, color='blue', marker='o')
ax[1,0].scatter(intLinex, intLiney, color='red', marker='x')
ax[1,0].set_title("Random Forest")
plt.show()
