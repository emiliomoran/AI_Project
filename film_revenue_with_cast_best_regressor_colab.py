# Importando librerías

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargando los csv
dataset_X_reimported = pd.read_csv('Encoded_X2.csv')
dataset_y_reimported = pd.read_csv('Encoded_y-revenue2.csv')
dataset_reimported = pd.concat([dataset_X_reimported,dataset_y_reimported],axis=1)
dataset_reimported = dataset_reimported.replace([np.inf, -np.inf], np.nan)
dataset_reimported = dataset_reimported.dropna() # Se eliminan los valores nulos

X = dataset_reimported.iloc[:, 1:-2].values
y = dataset_reimported.iloc[:, -1].values

# Se divide los datos leídos, entre datos de entrenamiento y datos de prueba
from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Importando la librería de xgboots
from xgboost import XGBRegressor

# Conjunto de hiper-parámetros necesarios para xgboots y entrenar nuestro modelo
params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4], 'objective':['reg:squarederror']}

# Combinación de parámetros obtenidos
# colsample_bytree= 0.6,
# gamma= 0.7,
# max_depth= 4,
# min_child_weight= 5,
# subsample = 0.8, objective='reg:squarederror'

xgb = XGBRegressor(nthread=-1)

# El tiempo para encontrar los hiperparámetros y entrenar nuestro modelo es de aproximado 12 horas
regressor = GridSearchCV(xgb, params)
regressor.fit(X, y)

# Usando el modelo con el dato de entrenamiento, para obtener predicción
y_pred = regressor.predict(X_test)

# Calculando métricas de evaluación de nuestro modelo
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
score = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("----EVALUATION METRICS----")
print("R-square:", score)
print("MAE:", mae)
print("MSE:", mse)
print("--------------------------")

# Generando gráfico de Measured Revenue vs Predicted Revenue
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured revenue')
ax.set_ylabel('Predicted revenue')
plt.title('Measured versus predicted revenue')
plt.ylim((50000000, 300000000))   # set the ylim to bottom, top
plt.xlim(50000000, 300000000)     # set the ylim to bottom, top
plt.show()