import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

file_path2 = r"C:\Users\becke\Downloads\useful databases\secondTest.csv"

df2 = pd.read_csv(file_path2)

df2.rename(columns={"area_code": "Region Code"}, inplace=True)
df2.rename(columns={"year ": "Year"}, inplace=True)
df2test = df2.loc[:,'available_seats':].astype(float)

df2test = df2test.dropna(subset=["unemployment %"])

target = "unemployment %"

features = [
    "population-density (persons per km^2)",
    "population % 16-64",
    "GDP per head £",
    "children in relative poverty %",
    "% with no qualifications",
    "% of people with level 3 qualifications",
    "time to employment centre by walking/public transport",
    "poverty_x_no_qual",
]

df2test["poverty_x_no_qual"] = (
    df2test["children in relative poverty %"] *
    df2test["% with no qualifications"]
)


X = df2test[features]
y = df2test[target]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Preprocessing pipeline
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, features)
    ]
)


model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
])

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

cv_scores = cross_val_score(
    model, X, y, cv=5, scoring="r2"
)

print(f"CV R² mean: {cv_scores.mean():.3f}")
print(f"CV R² std: {cv_scores.std():.3f}")