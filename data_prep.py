# Importando librerías
import numpy as np
import pandas as pd
from encode_json_column import encode_json_column
from datetime import datetime

# Leyendo los archivos de datos
dataset = pd.read_csv('tmdb_5000_movies.csv')
dataset_credits = pd.read_csv('tmdb_5000_credits.csv')
dataset = pd.concat([dataset, dataset_credits], axis=1)

# Se reemplaza los valores de presupuesto de 0 con el valor promedio de presupuesto de las películas
dataset['budget']=dataset['budget'].replace(0,dataset['budget'].mean())

X = dataset.iloc[:, :].values
y_revenue = dataset.iloc[:, 12].values
y_rating = dataset.iloc[:, 18].values

X = X[:,[0,1,4,9,11,13,14,15,22,23]]

# Luego, posteriormente se remueven los valores de taquilla y presupuesto con valor 0
y_revenue_removed = []
y_rating_removed = []
X_removed = []
for l in range(0,len(y_revenue)):
    if y_revenue[l] !=0:
        y_revenue_removed.append(y_revenue[l])
        y_rating_removed.append(y_rating[l])
        X_removed.append(X[l])
y_revenue = np.array(y_revenue_removed)
y_rating = np.array(y_rating_removed)
X = np.array(X_removed)

# Se procedió a separar la fecha entre año y día
for l in range(0,len(y_revenue)):
    film_date = X[l,4]
    try:
        datetime_object = datetime.strptime(film_date, '%Y-%m-%d')
        X[l,4] = datetime_object.timetuple().tm_yday
    except:
        X[l,4] = 0

dataset =  pd.DataFrame(X)

dataset = encode_json_column(dataset, 1,"name")

# Codificando keywords, se trabaja solo con el top 100 de las keywords
dataset = encode_json_column(dataset, 1, "name", 500, 1)

# Codificando compañías productoras, se trabaja solo con el top 100 de las compañías productoras
dataset = encode_json_column(dataset, 1,"name", 500, 1)

# Se codifica todos los lenguajes de las películas
dataset = encode_json_column(dataset, 3,"iso_639_1")

# Codificando elenco, se toman el top 100 del elenco
dataset = encode_json_column(dataset, 4,"name", 500, 1) #was 500

# Codificando personal, se toman el top 100 del personal
dataset = encode_json_column(dataset, 4,"name", 500, 1) #was 500

# Se guardan los archivos para usarlos en la creación del modelo
dataset.to_csv(r'Encoded_X2.csv')
dataset_y_revenue = pd.DataFrame(y_revenue)
dataset_y_revenue.to_csv(r'Encoded_y - revenue2.csv')
dataset_y_rating = pd.DataFrame(y_rating)
dataset_y_rating.to_csv(r'Encoded_y - rating2.csv')