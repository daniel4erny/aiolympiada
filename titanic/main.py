from catboost import CatBoostClassifier
import pandas as pd

data = pd.read_csv("train.csv")

X = data.drop(columns=["PassengerId", "Survived"])
y = data["Survived"]

helper = X.select_dtypes(include="str").columns.to_list()
print(helper)
X[helper] = X[helper].fillna("unkown")


model = CatBoostClassifier(cat_features=helper, iterations=300)
model.fit(X, y)

test_data = pd.read_csv("test.csv")
ids = test_data["PassengerId"]
X_test = test_data.drop(columns="PassengerId")
X_test[helper] = X_test[helper].fillna("unkown")

predictions_percent = model.predict_proba(X_test)
print(predictions_percent)
predictions = model.predict(X_test)

predict_csv = pd.DataFrame({"PassengerId": ids, "Survived": predictions})


predict_csv.to_csv("predictions.csv", index=False)
