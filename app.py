from flask import Flask, render_template, request
import pandas as pd
from models.linearRegression import train_predict
from models.logisticRegression import train_predict_logistic
from models.perceptron_model import train_perceptron, prediction_perceptron

app = Flask(__name__)

# ================== DATASET ==================
df = pd.read_csv('medals.csv')
df_logistic = pd.read_csv('fifa24.csv')
df_perceptron = pd.read_csv('breastCancerWisconsin.csv')

# ================== ROUTES ==================

@app.route('/')
def home():
    return render_template('mainMenu.html')

# ---------- USE CASE ----------

@app.route('/BankFraud')
def bank_page():
    return render_template('caseUse/BankFraud.html')

@app.route('/BBVA')
def bbva_page():
    return render_template('caseUse/BBVAPipeline.html')

@app.route('/FacialRecognition')
def facial_page():
    try:
        return render_template('caseUse/FacialRecognized.html')
    except Exception as e:
        return f"<h1>Error en /casodeuso</h1><p>{str(e)}</p>"

@app.route('/customerChurn')
def customerChurn_page():
    return render_template('caseUse/customerChurn.html')

# ---------- LINEAR REGRESSION ----------

@app.route('/linearRegression/concepts')
def linealConcept():
    return render_template('linearRegression/linealRConcepts.html')

@app.route('/linearRegression/application')
def linealApplication():
    columns = list(df.select_dtypes(include='number').columns)
    columns = [col for col in columns if col != 'Total']
    return render_template('linearRegression/linealRApplication.html', columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    x_col = request.form['x_variable']
    input_value = float(request.form['input_value'])

    prediction = train_predict(df, x_col, input_value)
    columns = list(df.select_dtypes(include='number').columns)
    columns = [col for col in columns if col != 'Total']

    return render_template('linearRegression/linealRApplication.html',
                           columns=columns,
                           prediction=prediction,
                           x_col=x_col,
                           input_value=input_value)

# ---------- LOGISTIC REGRESSION ----------

@app.route('/logisticRegression/concepts')
def logisticConcept():
    return render_template('logisticRegression/logisticConcepts.html')

@app.route('/logisticRegression/application')
def logisticApplication():
    return render_template('logisticRegression/logisticApplication.html')

@app.route('/predict_logistic', methods=['POST'])
def predict_logistic():
    pace_input = float(request.form['pace'])
    shooting_input = float(request.form['shooting'])
    passing_input = float(request.form['passing'])
    dribbling_input = float(request.form['dribbling'])

    result_data = train_predict_logistic(
        df_logistic,
        pace_input,
        shooting_input,
        passing_input,
        dribbling_input
    )
    return render_template('logisticRegression/logisticApplication.html', **result_data)

# ---------- PERCEPTRON ----------

@app.route('/perceptron/concepts')
def perceptronConcepts():
    return render_template('classificationPerceptron/perceptronConcepts.html')

@app.route('/perceptron/application', methods=['GET', 'POST'])
def perceptronApp():
    data = train_perceptron(df_perceptron)
    prediction = None

    if request.method == 'POST':
        prediction = prediction_perceptron(
            data["model"],
            data["scaler"],
            data["X"],
            request.form
        )

    return render_template('classificationPerceptron/perceptronApplication.html',
                           metrics=data["metrics"],
                           plot_url_cm=data["plot_url_cm"],
                           plot_url_roc=data["plot_url_roc"],
                           plot_url_errors=data["plot_url_errors"],
                           prediction=prediction)

# ---------- K-MEANS ----------

@app.route('/kmeans/concepts')
def kmeans_concepts():
    return render_template('kMeans/kmeansConcepts.html')

# ================== RUN ==================
if __name__ == '__main__':
    app.run(debug=True)