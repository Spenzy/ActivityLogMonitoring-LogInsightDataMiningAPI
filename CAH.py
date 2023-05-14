import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

# Import Data
df = pd.read_csv('DataMining/DW.csv', sep='|', lineterminator='\n', header=0)
df.columns=['Job_Duration','Data_Volume', 'Nbr_Components']
print(df)

# Normalize the data
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Segmentation
y_pred = AgglomerativeClustering(n_clusters= 3, affinity='euclidean', linkage='ward').fit_predict(df)

# Assign jobs to clusters
df['Cluster'] = y_pred

# Results
print(df.head())

def clusters():
    return df.reset_index().to_json(orient='records')


# Predicting clusters ( Decision Tree )
# Pour éviter Overfitting
df=df.sample(n=4000,random_state=42)

# Diviser le dataframe en un ensemble d'entraînement et un ensemble de test
X_train, X_test, y_train, y_test = train_test_split(df[['Job_Duration','Data_Volume','Nbr_Components']], df['Cluster'], test_size=0.2, random_state=42)

# Entraîner un modèle d'arbre de décision sur l'ensemble d'entraînement
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = dt.predict(X_test)

""""
test = pd.DataFrame([[30000, 98000, 11]], columns=['Job_Duration','Data_Volume', 'Nbr_Components'])
print("Testo:",test.head())
test = pd.DataFrame(scaler.transform(test), columns=test.columns)

cluster = dt.predict(test)
print("Predicted cluster for new job:", cluster[0])
"""

def predict(jobDuration, dataVolume, nbrComponent):
    #load data into a DataFrame object:
    res = pd.DataFrame([[jobDuration, dataVolume, nbrComponent]], columns=['Job_Duration','Data_Volume', 'Nbr_Components'])
    res = pd.DataFrame(scaler.transform(res), columns=res.columns)

    return dt.predict(res)[0]