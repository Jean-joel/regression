from flask import Flask, request
from flask import render_template
from flask import url_for
from werkzeug.security import generate_password_hash
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, classification_report, precision_score

with open('mod.pkl', 'rb') as f1:
    model = pickle.load(f1)
print (model)
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

app = Flask(__name__)


@app.route('/')
def index():
    return 'hi hi'


@app.route('/prediction', methods=['POST', 'GET'])

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

        return render_template("forlmulaire.html",
                               nom=nom, prenom=prenom, pred=prediction(donne_explicatif)
                               )
    else:
        return render_template("forlmulaire.html")






def prediction(elemt_explicatif):

    y=model.predict(elemt_explicatif)
    return y
if __name__ == "__main__":
    app.run()
