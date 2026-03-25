from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

app = Flask(__name__)

# ================== DATASET ==================
df = pd.read_csv('medals.csv')

# VARIABLES CORRECTAS
X = df[['Gold Medal', 'Silver Medal', 'Bronze Medal']]
y = df['Total']

# MODELO
model = LinearRegression()
model.fit(X, y)

# ================== GRÁFICA ==================
plt.figure()

x = df['Gold Medal']
y_real = df['Total']
y_pred = model.predict(X)

plt.scatter(x, y_real)
plt.plot(x, y_pred)

plt.xlabel('Gold Medals')
plt.ylabel('Total Medals')
plt.title('Linear Regression')

plt.savefig('static/graph.png')
plt.close()


# ================== RUTAS ==================

@app.route('/')
def home():
    return render_template('mainMenu.html')


@app.route('/BankFraud')
def bank_page():
    return render_template('BankFraud.html')


@app.route('/BBVA')
def bbva_page():
    return render_template('BBVAPipeline.html',
                           titulo="Case One",
                           des="Machine Learning Class")


@app.route('/FacialRecognition')
def facial_page():
    try:
        return render_template('FacialReconogized.html')
    except Exception as e:
        return f"<h1>Error en /casodeuso</h1><p>{str(e)}</p>"


@app.route('/SB')
def customerChurn_page():
    return render_template('customerChurn.html')

@app.route('/linearRegression/concepts')
def linealConcept():
    return render_template('linealRConcepts.html')


@app.route('/linearRegression/application')
def linealApplication():
    return render_template('linealRApplication.html')


@app.route('/predict', methods=['POST'])
def predict():
    gold = float(request.form['gold'])
    silver = float(request.form['silver'])
    bronze = float(request.form['bronze'])

    prediction = model.predict([[gold, silver, bronze]])[0]

    return render_template('linealRApplication.html',
                           prediction=round(prediction, 2))


# ================== RUN ==================
if __name__ == '__main__':
    app.run(debug=True)