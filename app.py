from flask import Flask, request,flash, redirect
from flask import render_template
from flask_sqlalchemy import SQLAlchemy
from flask import url_for
from werkzeug.utils import secure_filename,send_file,send_from_directory



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
print(model)
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///db.sqlite3'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"]=False
app.secret_key="prediction_coeur"
upload_folder="upload/"
if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

app.config['UPLOAD_FOLDER']=upload_folder


#app.config['SECRET_KEY'] = "prediction_coeur"
db=SQLAlchemy(app)

class prediction_cardiaque(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True)
    username=db.Column(db.String(100))
    age = db.Column(db.Integer)
    par = db.Column(db.Integer)
    chol = db.Column(db.Integer)
    fcmax = db.Column(db.Integer)
    gaj = db.Column(db.Integer)
    sexe = db.Column(db.String(30))
    tdt = db.Column(db.String(20))
    ecg = db.Column(db.String(30))
    angine = db.Column(db.String(30))
    depres = db.Column(db.Integer)
    pente = db.Column(db.String(30))

    def __init__(self, username,age, par,chol, fcmax, gaj, sexe, tdt, ecg, angine, depres, pente):
        self.username = username
        self.age = age
        self.par = par
        self.chol = chol
        self.fcmax = fcmax
        self.gaj =gaj
        self.sexe =sexe
        self.tdt =tdt
        self.ecg = ecg
        self.angine = angine
        self.depres = depres
        self.pente = pente

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template("index.html")

@app.route('/predicti', methods=['POST', 'GET'])
def regression():
    if request.method == 'POST':
        regression=request.form['regression']
        if regression== 'simple':
            return render_template("forlmulaire.html")
        else:
            return render_template('view_db.html', individus=prediction_cardiaque.query.all())






@app.route('/prediction', methods=['POST', 'GET'])

def traitement():
    if request.method == 'POST':
        nom=request.form['nom']
        prenom=request.form['prenom']
        sexe=request.form['sexe']
        TDT=request.form['TDT']
        age=float(request.form['age'])
        par=float(request.form['par'])
        CHOL=float(request.form['CHOL'])
        GAJ=float(request.form['GAJ'])
        ECG=request.form['ECG']
        FCMAX=float(request.form['FCMAX'])
        ANGINE=request.form['ANGINE']
        DEPRESSION=float(request.form['DEPRESSION'])
        PENTE=request.form['PENTE']
        donne_explicatif=pd.DataFrame({"SEXE":[sexe],
                          "TDT" :[TDT],
                          "AGE":[age],
                          "PAR":[par],
                          "CHOL":[CHOL],
                          "GAJ":[GAJ],
                          "ECG":[ECG],
                          "FCMAX":[FCMAX],
                          "ANGINE":[ANGINE],
                          "DEPRESSION":[DEPRESSION],
                          "PENTE":[PENTE]})
        r_donne_explicatif=encdage(donne_explicatif)
        print(r_donne_explicatif)

        return render_template("index.html",
                               nom=nom, prenom=prenom, pred=prediction(r_donne_explicatif)
                               )
    else:
        return render_template("forlmulaire.html")


@app.route('/db')
def show_db():
    return render_template('view_db.html', individus=prediction_cardiaque.query.all())

@app.route('/add')
def add_indiv():
    return render_template('ajouter_indiv.html')

@app.route('/add_table')
def add_table():
    return "hi"

@app.route('/add_dbb' , methods=['POST', 'GET'])
def upload_db():
    if request.method == "POST":
        if request.files:
            tableur=request.files['tableur']
            s_tableur=tableur.save(secure_filename((tableur.filename)))
            tableur.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename((tableur.filename))))
            data = pd.read_excel(tableur.filename)
            print(data)
    return "hi"

@app.route('/add_db' , methods=['POST', 'GET'])
def add_db():
    if request.method == 'POST':
        if not request.form['nom'] or not request.form['prenom']\
                or not request.form['sexe'] or not request.form['TDT']\
                or not request.form['age'] or not request.form['par']\
                or not request.form['CHOL']or not request.form['GAJ']\
                or not request.form['ECG'] or not request.form['FCMAX']\
                or not request.form['ANGINE'] or not request.form['DEPRESSION']\
                or not request.form['PENTE']:
               flash('Please enter all the fields', 'error')
        else:
            nom = request.form['nom']
            prenom = request.form['prenom']
            sexe = request.form['sexe']
            TDT = request.form['TDT']
            age = float(request.form['age'])
            par = float(request.form['par'])
            CHOL = float(request.form['CHOL'])
            GAJ = float(request.form['GAJ'])
            ECG = request.form['ECG']
            FCMAX = float(request.form['FCMAX'])
            ANGINE = request.form['ANGINE']
            DEPRESSION = float(request.form['DEPRESSION'])
            PENTE = request.form['PENTE']
            individu=prediction_cardiaque(username=nom+" "+prenom ,age=age ,par=par ,chol=CHOL,fcmax=FCMAX,gaj=GAJ ,sexe=sexe,tdt=TDT,ecg=ECG,angine=ANGINE,depres=DEPRESSION,pente=PENTE)
            db.session.add(individu)
            db.session.commit()
            flash('Record was successfully added')
            return redirect(url_for('show_db'))

    return render_template('view_db.html')

#@app.route('/edit', methods=['GET', 'POST'])
#def new():
 #   if request.method == 'POST':
 #
 #   return render_template('add_item.html')



def prediction(elemt_explicatif):
    text=""
    y=model.predict(elemt_explicatif)
    if y==1:
        text="Vous avez probablement un probleme cardiaque"
    else:
        text="Vous ??tes bien portant"
    return text

def encdage(data):
    dic_sexe={"homme":1 ,"femme":0}
    data['SEXE'].replace(dic_sexe, inplace=True)
    dic_tdt={"AA":0 ,"DNA":3 , "ASY":1 ,"AT":2}
    data['TDT'].replace(dic_tdt, inplace=True)
    dic_ecg={"Normal":1 , "ST":2 , "LVH":0}
    data['ECG'].replace(dic_ecg, inplace=True)
    dic_ang={"Oui":1 , "Non":0}
    data['ANGINE'].replace(dic_ang, inplace=True)
    dic_pen={"Ascendant":2 , "Plat":0 ,"Descendant":1}
    data['PENTE'].replace(dic_pen, inplace=True)
    return data


if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)
