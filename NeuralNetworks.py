import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# Load the data
df = pd.read_csv('DataMining/DW.csv', sep='|', lineterminator='\n', header=1)
df.columns=['Job_Duration','Data_Volume', 'Nbr_Components']
print(df)

# Calculer l'IQR pour chaque colonne
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Identifier les valeurs aberrantes pour chaque colonne
outliers_count = (df['Data_Volume'] < (Q1['Data_Volume'] - 1.5 * IQR['Data_Volume'])) | (df['Data_Volume'] > (Q3['Data_Volume'] + 1.5 * IQR['Data_Volume']))
outliers_duration = (df['Job_Duration'] < (Q1['Job_Duration'] - 1.5 * IQR['Job_Duration'])) | (df['Job_Duration'] > (Q3['Job_Duration'] + 1.5 * IQR['Job_Duration']))

# Supprimer les valeurs aberrantes du dataframe
df = df[~(outliers_count | outliers_duration)]

# normalize the data
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print(df)

# Séparer les données en variables d'entrée (X) et variable de sortie (y)
X = df[['Data_Volume', 'Nbr_Components']]
y = df['Job_Duration']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle de réseau de neurones
model = Sequential()
model.add(Dense(32, input_dim=2, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compiler le modèle
model.compile(loss='mse', optimizer='adam')

# Entraîner le modèle
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

print(model.summary())

# Predict the values of y for X_test using the trained model
y_pred = model.predict(X_test)

#just a test
res = pd.DataFrame([[71600, 9]], columns=['Data_Volume', 'Nbr_Components'])
print(model.predict(res)[0])

def predict(dataVolume, nbrComponent):
    #load data into a DataFrame object:
    res = pd.DataFrame([[dataVolume, nbrComponent]], columns=['Data_Volume', 'Nbr_Components'])
    #res = pd.DataFrame(scaler.transform(res), columns=res.columns)

    return model.predict(res)[0]