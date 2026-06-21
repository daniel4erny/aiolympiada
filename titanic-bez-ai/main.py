import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


def loadData():
    # Column       Non-Null Count  Dtype
    # ---  ------       --------------  -----
    #  0   PassengerId  891 non-null    int64
    #  1   Survived     891 non-null    int64
    #  2   Pclass       891 non-null    int64
    #  3   Name         891 non-null    str
    #  4   Sex          891 non-null    str
    #  5   Age          714 non-null    float64
    #  6   SibSp        891 non-null    int64
    #  7   Parch        891 non-null    int64
    #  8   Ticket       891 non-null    str
    #  9   Fare         891 non-null    float64
    #  10  Cabin        204 non-null    str
    #  11  Embarked     889 non-null    str
    global X, y
    data = pd.read_csv("train.csv")
    X = data.drop(columns=["PassengerId", "Survived"])
    y = data["Survived"]


loadData()
print(X.info())
print(y.info())
catHelpers = X.select_dtypes("str").columns.to_list()
model = CatBoostClassifier(cat_features=catHelpers, n_estimators=1000)
X[catHelpers] = X[catHelpers].fillna("unkown")
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, random_state=42, test_size=0.2
# )
#
# model.fit(X_train, y_train)
#
# predictions = model.predict(X_test)
# print((predictions == y_test).mean().mean() * 100)
model.fit(X, y)
test_data = pd.read_csv("test.csv")
ids = test_data["PassengerId"]
X_test = test_data.drop(columns=["PassengerId"])
X_test[catHelpers] = X_test[catHelpers].fillna("unkown")

predictions = model.predict(X_test)

submission = pd.DataFrame({"PassengerId": ids, "Survived": predictions})

submission.to_csv("submission.csv", index=False)
