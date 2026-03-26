from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# ================== DATASET ==================
df = pd.read_csv('medals.csv')

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

# Pasa las columnas del CSV al HTML para los dropdowns
@app.route('/linearRegression/application')
def linealApplication():
    columns = list(df.select_dtypes(include='number').columns)
    return render_template('linealRApplication.html', columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    x_col = request.form['x_variable']
    input_value = float(request.form['input_value'])

    # Regresion simple: una sola variable X predice Y fijo (Total)
    X = df[[x_col]]
    y = df['Total']

    model = LinearRegression()
    model.fit(X, y)

    plt.figure()
    plt.scatter(df[x_col], y, color='blue', label='Real data')
    plt.plot(df[x_col], model.predict(X), color='red', label='Regression line')
    plt.xlabel(x_col)
    plt.ylabel('Total')
    plt.title(f'Simple Linear Regression: {x_col} vs Total')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/graph.png')
    plt.close()

    prediction = model.predict([[input_value]])[0]

    columns = list(df.select_dtypes(include='number').columns)
    columns = [col for col in columns if col != 'Total']

    return render_template('linealRApplication.html',
                           columns=columns,
                           prediction=round(prediction, 2),
                           x_col=x_col,
                           input_value=input_value)


# ================== RUN ==================
if __name__ == '__main__':
    app.run(debug=True)
