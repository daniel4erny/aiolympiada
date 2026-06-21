from catboost import CatBoostRegressor  # OPRAVENO: Regressor místo Classifier
import pandas as pd
import numpy as np

# --- 1. NAČTENÍ DAT ---
train = pd.read_csv("train.csv")
oil = pd.read_csv("oil.csv")
stores = pd.read_csv("stores.csv")
holidays = pd.read_csv("holidays_events.csv")

# OPRAVENO: Každá tabulka upravuje své vlastní datum!
train["date"] = train["date"].astype(str)
oil["date"] = oil["date"].astype(str)
holidays["date"] = holidays["date"].astype(str)

# --- 2. MERGE A FEATURE ENGINEERING (TRÉNOVACÍ DATA) ---
df_train = pd.merge(train, stores, on="store_nbr", how="left")
df_train = pd.merge(df_train, oil, on="date", how="left")
df_train = pd.merge(df_train, holidays, on="date", how="left")

df_train["date"] = pd.to_datetime(df_train["date"])
df_train["year"] = df_train["date"].dt.year
df_train["month"] = df_train["date"].dt.month
df_train["day"] = df_train["date"].dt.day
df_train["dayofweek"] = df_train["date"].dt.dayofweek
df_train = df_train.drop(columns=["date"])

# Příprava kategorií
cat_features = df_train.select_dtypes(include=["object"]).columns.tolist()
cat_features.append("store_nbr")
df_train[cat_features] = df_train[cat_features].astype(str).fillna("Unknown")

# Rozdělení na X a y
X_train = df_train.drop(columns=["id", "sales"])
y_train = df_train["sales"]

# --- 3. TRÉNOVÁNÍ MODELU ---
model = CatBoostRegressor(
    cat_features=cat_features, iterations=200
)  # Přidány iterace, ať to netrvá hodiny
model.fit(X_train, y_train)


# --- 4. PŘÍPRAVA TESTOVACÍCH DAT ---
test = pd.read_csv("test(1).csv")  # Přejmenováno z 'train' na 'test' pro přehlednost
test["date"] = test["date"].astype(str)

# Spojení tabulek pro test (oil a holidays už máme načtené z minula)
df_test = pd.merge(test, stores, on="store_nbr", how="left")
df_test = pd.merge(df_test, oil, on="date", how="left")
df_test = pd.merge(df_test, holidays, on="date", how="left")

df_test["date"] = pd.to_datetime(df_test["date"])
df_test["year"] = df_test["date"].dt.year
df_test["month"] = df_test["date"].dt.month
df_test["day"] = df_test["date"].dt.day
df_test["dayofweek"] = df_test["date"].dt.dayofweek
df_test = df_test.drop(columns=["date"])

# Použijeme stejné kategorie jako u trénovacích dat
df_test[cat_features] = df_test[cat_features].astype(str).fillna("Unknown")

ids = df_test["id"]
X_test = df_test.drop(columns=["id"])

# POJISTKA: Seřadíme sloupce v testu přesně tak, jak byly v trénovacím X
X_test = X_test[X_train.columns]

# --- 5. PREDIKCE A ULOŽENÍ ---
predictions = model.predict(X_test)

# POJISTKA: Tržby nemůžou být záporné (přepíše případná záporná čísla na nulu)
predictions = np.clip(predictions, 0, None)

submissions_csv = pd.DataFrame({"id": ids, "sales": predictions})
submissions_csv.to_csv("submission.csv", index=False)
