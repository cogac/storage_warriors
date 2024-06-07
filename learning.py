import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Daten laden
data = pd.read_csv('wanted_data.csv')

# Erste Inspektion der Daten
print(data.head())
input('Press Enter if continue')
# Feature Engineering: Datum in Jahr, Monat und Tag aufteilen
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

# Relevante Features auswählen
features = ['product_id', 'weight', 'year', 'month', 'day']
X = data[features]
y = data['sales']

# Kategorische Daten in numerische umwandeln
X = pd.get_dummies(X, columns=['product_id'], drop_first=True)

# Datensatz in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell initialisieren und trainieren
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Vorhersagen auf den Testdaten machen
y_pred = model.predict(X_test)

# Modellbewertung
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Beispielvorhersage für neue Daten
new_data = {
    'product_id': '9021',
    'weight': 32.0,
    'year': 2024,
    'month': 6,
    'day': 15
}

# Konvertierung der neuen Daten in das benötigte Format
new_data_df = pd.DataFrame([new_data])
new_data_df = pd.get_dummies(new_data_df, columns=['product_id'], drop_first=True)
new_data_df = new_data_df.reindex(columns=X.columns, fill_value=0)

# Vorhersage machen
predicted_sales = model.predict(new_data_df)
print(f'Predicted Sales: {predicted_sales[0]}')
