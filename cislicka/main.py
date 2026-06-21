from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

train = pd.read_csv("train.csv")
X = train.drop(columns=["label"])
y = train["label"]

model = RandomForestClassifier(n_estimators=500, n_jobs=-1, verbose=4)

# X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
#
# print(accuracy_score(predictions, y_test))
# print(classification_report(predictions, y_test))

model.fit(X, y)

test = pd.read_csv("test.csv")

predictions = model.predict(test)

submission = pd.DataFrame({"label": predictions})
submission.index = submission.index + 1

submission.to_csv("submission.csv", index_label="ImageId")
