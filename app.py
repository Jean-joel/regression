from flask import Flask, request
from flask import render_template
from flask import url_for
from werkzeug.security import generate_password_hash
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, classification_report, precision_score

data = pd.read_excel('C:/Users/yakri/Downloads/Copie_Coeur.xlsx')
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

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

app = Flask(__name__)


@app.route('/')
def index():
    return 'hi hi'


@app.route('/prediction', methods=['POST', 'GET'])

def vue_form(name=None):
    error = None
    valeur=0
    name = request.form.get("name", "")
    age = request.form.get("age", "")
    response = "Hey there {}! You said you are {} years old.".format(name, age)

    if request.method == 'POST':
        if (request.form['username']):
            valeur=1
        else:
            error = 'Invalid username/password'

    return render_template("forlmulaire.html", error=error)

@app.route('/predit', methods=['POST', 'GET'])

def traitement():
    if request.method == 'POST':
        nom=request.form['nom']
        prenom=request.form['prenom']
        sexe=float(request.form['sexe'])
        TDT=float(request.form['TDT'])
        age=float(request.form['age'])
        par=float(request.form['par'])
        CHOL=float(request.form['CHOL'])
        GAJ=float(request.form['GAJ'])
        ECG=float(request.form['ECG'])
        FCMAX=float(request.form['FCMAX'])
        ANGINE=float(request.form['ANGINE'])
        DEPRESSION=float(request.form['DEPRESSION'])
        PENTE=float(request.form['PENTE'])
        donne_explicatif=pd.DataFrame({"sexe":[sexe],
                          "TDT" :[TDT],
                          "age":[age],
                          "par":[par],
                          "CHOL":[CHOL],
                          "GAJ":[GAJ],
                          "ECG":[ECG],
                          "FCMAX":[FCMAX],
                          "ANGINE":[ANGINE],
                          "DEPRESSION":[DEPRESSION],
                          "PENTE":[PENTE]})

        return render_template("prediction.html",
                               nom=nom, prenom=prenom, aprediction=y_pred[:5], pred=prediction(donne_explicatif)
                               )
    else:
        return render_template("forlmulaire.html")

def prediction(elemt_explicatif):

    y=model.predict(elemt_explicatif)
    return y
if __name__ == "__main__":
    app.run()
