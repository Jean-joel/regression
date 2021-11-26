import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, classification_report, precision_score

data = pd.read_excel('C:/Users/yakri/Downloads/Coeur.xlsx')
df = data.copy()

for col in df.drop('CŒUR', axis =1).select_dtypes(np.number).columns:
    df[col] = df[col]/df[col].max()

for col in df.select_dtypes('object').columns:
    df[col] = df[col].astype('category').cat.codes

#Séparer la variable cible (coeur) et les variables explicatives

y = df['CŒUR']
x = df.drop('CŒUR', axis = 1)

#Subdivision du jeu de données en apprentissage et en test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2, random_state =1)
#train_set, test_set = train_test_split(df,test_size =0.2, random_state =1)

#Création d'un objet lr de la classe LogisticRegression
lr = LogisticRegression(solver ='newton-cg', random_state =1)

#Apprentissage du modèle
model = lr.fit(x_train, y_train)
#model = lr.fit(train_set.drop('CŒUR', axis = 1), train_set['CŒUR'])

#Probabilité d'appartenance à l'une des classes
predict_proba = model.predict_proba(x_test)
predict_proba[:5, :]

#Application du modèle au données de test
y_pred = model.predict(x_test)
y_pred[:5]

#matrice de confusion
mc = confusion_matrix(y_test, y_pred)
