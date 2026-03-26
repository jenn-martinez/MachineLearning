from flask import Flask, render_template, request
import LinearRegression
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

app = Flask(__name__)
BASE_DIR = Path(__file__).parent

@app.route("/")
def home():
    return "hello flask"

@app.route('/casodeuso')
def firstPage():
    return render_template('index.html')

@app.route("/LinearRegression", methods=["GET", "POST"])
def calculateGrade():
    calculateResult = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        calculateResult = LinearRegression.calculateGrade(hours)
    return render_template('LinearRegressionGrades.html', result=calculateResult)

@app.route("/RegresionLogistica", methods=["GET", "POST"])
def regresion_logistica():
    # Entrenar el modelo
    df = pd.read_csv(r'C:\Users\sergi\Documents\universidad\Machine Learning\dataset_regresion_logistica.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    prediccion = None
    if request.method == "POST":
        edad = float(request.form["edad"])
        ingreso = float(request.form["ingreso_mensual"])
        visitas = float(request.form["visitas_web_mes"])
        tiempo = float(request.form["tiempo_sitio_min"])
        compras = float(request.form["compras_previas"])
        descuento = float(request.form["descuento_usado"])

        resultado = model.predict([[edad, ingreso, visitas, tiempo, compras, descuento]])
        prediccion = "si compra" if resultado[0] == 1 else " NO compra"

    return render_template('RegresionLogistica.html', prediccion=prediccion)

