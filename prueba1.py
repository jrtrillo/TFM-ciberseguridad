import numpy as np
import pandas as pd
import csv
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler




dtypes = {
    # 'attack_time': 'datetime64[ns]',
    "watcher_country": "category",
    "watcher_as_num": "float64",
    "watcher_as_name": "category",
    "attacker_country": "category",
    "attacker_as_num": "float64",
    "attacker_as_name": "category",
    "attack_type": "category",
    "watcher_uuid_enum": "int64",
    "attacker_ip_enum": "int64",
    "label": "int64",
}
#df = pl.read_parquet("train.parq")
df = pd.read_csv("train.csv", dtype=dtypes, parse_dates=["attack_time"])
df.dtypes
df.shape


dtypes2 = {
    # 'attack_time': 'datetime64[ns]',
    "watcher_country": "category",
    "watcher_as_num": "float64",
    "watcher_as_name": "category",
    "attacker_country": "category",
    "attacker_as_num": "float64",
    "attacker_as_name": "category",
    "attack_type": "category",
    "watcher_uuid_enum": "int64",
    "attacker_ip_enum": "int64",
}

df2 = pd.read_csv("test.csv", dtype=dtypes2, parse_dates=["attack_time"])
#df2['label'] = None
df2.dtypes
df2.shape


etiqueta = df['label']

df = df.drop(['attack_time'], axis=1)
df = df.drop(['label'], axis=1)
df2 = df2.drop(['attack_time'], axis=1)
#de aqui para arriba intocable

df = df.drop(['watcher_country'], axis=1)
df2 = df2.drop(['watcher_country'], axis=1)

df = df.drop(['watcher_as_name'], axis=1)
df2 = df2.drop(['watcher_as_name'], axis=1)

df = df.drop(['attacker_country'], axis=1)
df2 = df2.drop(['attacker_country'], axis=1)

df = df.drop(['attacker_as_name'], axis=1)
df2 = df2.drop(['attacker_as_name'], axis=1)

#numerica = df.drop(['watcher_country','watcher_as_name','attacker_country','attacker_as_name','attack_type'], axis=1)
#numerica2 = df2.drop(['watcher_country','watcher_as_name','attacker_country','attacker_as_name','attack_type'], axis=1)

#categorica = df.filter(['watcher_country','watcher_as_name','attacker_country','attacker_as_name','attack_type'])
#categorica2 = df2.filter(['watcher_country','watcher_as_name','attacker_country','attacker_as_name','attack_type'])


numerica = df.drop(['attack_type'], axis=1)
numerica2 = df2.drop(['attack_type'], axis=1)

categorica = df.filter(['attack_type'])

categorica2 = df2.filter(['attack_type'])

#se puede modificar esto de arriba


cat_categorica = pd.get_dummies(categorica, drop_first=False)
cat_categorica2 = pd.get_dummies(categorica2, drop_first=False)


train = pd.concat([cat_categorica, numerica], axis=1)
test = pd.concat([cat_categorica2, numerica2], axis=1)





dtypes3 = {
    # 'attack_time': 'datetime64[ns]',
    "attacker_ip_enum": "int64",
    "label": "int64",

}

df3 = pd.read_csv("samples_submissions.csv", dtype=dtypes3)
#df2['label'] = None
df3.dtypes
df3.shape
df3['label'] = 0



df3 = df3.values.tolist()


train= train.values.tolist()
test= test.values.tolist()

columna = int(len(test[0])-1)

# Reemplazar NaN e infinitos con la media de la columna
for i in range(len(train[0])):
    train_column = [row[i] for row in train]
    mean_value = np.mean([val for val in train_column if val not in [np.inf, -np.inf]])
    for j in range(len(train)):
        if train[j][i] in [np.inf, -np.inf]:
            train[j][i] = mean_value

for i in range(len(test[0])):
    test_column = [row[i] for row in test]
    mean_value = np.mean([val for val in test_column if val not in [np.inf, -np.inf]])
    for j in range(len(test)):
        if test[j][i] in [np.inf, -np.inf]:
            test[j][i] = mean_value






#tree = DecisionTreeClassifier(random_state=0).fit(train, etiqueta)
	
#neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(train, etiqueta)
#y_pred = neigh.predict(test)

#clf = svm.SVC()
#clf.fit(train, etiqueta)
#y_pred = clf.predict(test)

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.fit_transform(test)

# Verificar y manejar NaN o infinitos en los conjuntos de datos escalados
train_scaled = np.nan_to_num(train_scaled, nan=np.nanmean(train_scaled))
test_scaled = np.nan_to_num(test_scaled, nan=np.nanmean(test_scaled))

# Ajustar el modelo de árbol de decisión
tree = DecisionTreeClassifier(random_state=0).fit(train_scaled, etiqueta.astype('int64'))
y_pred = tree.predict(test_scaled)


for i in range(len(y_pred)):
    for j in range(len(df3)):
        if(df3[j][0] == test[i][columna]):
            if(y_pred[i] == 1):
                df3[j][1] = df3[j][1] + 1
            else:
                df3[j][1] = df3[j][1] - 1

for i in range(len(df3)):
    if(df3[i][1] <= 0.0):
        df3[i][1] = 0
    else:
        df3[i][1] = 1
    print (df3[i])


print(y_pred)
columnas = ["attacker_ip_enum", "label"]

# Crear un objeto de archivo CSV
with open("prueba1.csv", "w", newline="") as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow(columnas)
    escritor.writerows(df3)


