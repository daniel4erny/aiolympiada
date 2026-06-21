from sklearn.ensemble import RandomForestRegressor
import pandas as pd

data = pd.read_csv("train.csv")
print(data.info())

X = data.drop(axis=0, columns=["PitNextLap"])
y = data["PitNextLap"]

X = pd.get_dummies(X)
print(X.info())

model = RandomForestRegressor(n_estimators=20, n_jobs=-1, verbose=2)
model.fit(X, y)
