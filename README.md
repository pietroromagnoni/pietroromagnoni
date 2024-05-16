import pandas as pd

wine_data = pd.read_csv(r"C:\Users\pirom\OneDrive\Documenti\html\wine_dataset.csv")

print(wine_data.head())

print(wine_data.describe())

import matplotlib.pyplot as plt
import seaborn as sns

plt.hist(wine_data['alcohol'])
plt.xlabel('Alcohol')
plt.ylabel('Frequenza')
plt.title('Distribuzione dell\'alcol nel vino')
plt.show()

print(wine_data.isnull().sum())

wine_data.dropna(inplace=True) 

from scipy import stats

numeric_columns = wine_data.select_dtypes(include=['float64', 'int64']).columns
z_scores = stats.zscore(wine_data[numeric_columns])
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
wine_data = wine_data[filtered_entries]

numeric_columns = wine_data.select_dtypes(include=['float64', 'int64']).columns
Q1 = wine_data[numeric_columns].quantile(0.25)
Q3 = wine_data[numeric_columns].quantile(0.75)
wine_data, _ = wine_data[numeric_columns].align(Q1, axis=1)
IQR = Q3 - Q1
wine_data = wine_data[~((wine_data < (Q1 - 1.5 * IQR)) | (wine_data > (Q3 + 1.5 * IQR))).any(axis=1)] 

correlation_matrix = wine_data.corr()
print(correlation_matrix)

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlazione tra variabili nel dataset')
plt.show()

from sklearn.preprocessing import StandardScaler, LabelEncoder

scaler = StandardScaler()
numeric_features = wine_data.select_dtypes(include=['float64', 'int64'])
wine_data[numeric_features.columns] = scaler.fit_transform(numeric_features)

label_encoder = LabelEncoder()
wine_data['categoria_encoded'] = label_encoder.fit_transform(wine_data['ph'])

from sklearn.model_selection import train_test_split

X = wine_data.drop('sugar', axis=1)
y = wine_data['sugar']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

linear_model = LinearRegression()
tree_model = DecisionTreeRegressor()
forest_model = RandomForestRegressor()

linear_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
import numpy as np

models = [linear_model, tree_model, forest_model]
model_names = ['Regressione Lineare', 'Albero di Decisione', 'Random Forest']
mse_scores = []

for model, name in zip(models, model_names):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    print(f"MSE per {name}: {mse}")

plt.bar(model_names, mse_scores)
plt.xlabel('Modello')
plt.ylabel('Mean Squared Error')
plt.title('Confronto prestazioni dei modelli')
plt.show()

best_model_index = np.argmin(mse_scores)
best_model = models[best_model_index]
best_model_name = model_names[best_model_index]
print(f"Il miglior modello è {best_model_name}")
