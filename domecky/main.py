import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
print(train.info())

X = train.drop(columns=["SalePrice", "Id"])
y = train["SalePrice"]

catHelper = X.select_dtypes("str").columns.to_list()
X[catHelper] = X[catHelper].fillna("unkown")

model = CatBoostRegressor(n_estimators=1000, cat_features=catHelper)
model.fit(X, y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8)
#
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# print(mean_absolute_percentage_error(predictions, y_test) * 100)

test = pd.read_csv("test.csv")
ids = test["Id"]
X_test = test.drop(columns=["Id"])
X_test[catHelper] = X_test[catHelper].fillna("unkown")
predictions = model.predict(X_test)

submission = pd.DataFrame({"Id": ids, "SalePrice": predictions})

submission.to_csv("submission.csv", index=False)
