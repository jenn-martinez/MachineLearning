from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# ================== DATASET ==================
df = pd.read_csv('medals.csv')
df_logistic = pd.read_csv('fifa24.csv')
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

# ================== LOGISTIC REGRESSION ==================
@app.route('/logisticRegression/concepts')
def logisticConcept():
    return render_template('logisticConcepts.html')


@app.route('/logisticRegression/application')
def logisticApplication():
    return render_template('logisticApplication.html')


@app.route('/predict_logistic', methods=['POST'])
def predict_logistic():
    # 1. Preparacion de datos
    df_log = df_logistic.copy()
    df_log['Elite'] = (df_log['Overall'] >= 80).astype(int)

    X = df_log[['Pace', 'Shooting', 'Passing', 'Dribbling']].fillna(0)
    y = df_log['Elite']

    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Entrenamiento del modelo
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # 5. Evaluacion del modelo
    train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
    test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

    # 6. Prediccion con inputs del usuario
    pace_input = float(request.form['pace'])
    shooting_input = float(request.form['shooting'])
    passing_input = float(request.form['passing'])
    dribbling_input = float(request.form['dribbling'])

    user_input = scaler.transform([[pace_input, shooting_input, passing_input, dribbling_input]])
    pred_class = model.predict(user_input)[0]
    pred_prob = model.predict_proba(user_input)[0][1]

    # 7. Grafica
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X['Pace'], X['Shooting'], c=y, cmap='RdYlBu_r', alpha=0.6)
    plt.colorbar(scatter, label='Elite (1) / Average (0)')
    plt.scatter([pace_input], [shooting_input], color='black', s=200, marker='*', label='Your Input')
    plt.xlabel('Pace')
    plt.ylabel('Shooting')
    plt.title('Logistic Regression: Elite Player?')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.text(0.1, 0.75, f'Train Accuracy:  {train_accuracy:.1%}', fontsize=13)
    plt.text(0.1, 0.55, f'Test Accuracy:   {test_accuracy:.1%}', fontsize=13)
    plt.text(0.1, 0.35, f'Prediction:      {"ELITE" if pred_class else "AVERAGE"}', fontsize=13)
    plt.text(0.1, 0.15, f'Probability:     {pred_prob:.1%}', fontsize=13)
    plt.axis('off')
    plt.title('Model Performance')

    plt.tight_layout()
    plt.savefig('static/logistic_graph.png', dpi=150)
    plt.close()

    result = "ELITE PLAYER" if pred_class == 1 else "AVERAGE PLAYER"
    return render_template('logisticApplication.html',
                           result=result,
                           probability=f"{pred_prob:.1%}",
                           pace=pace_input,
                           shooting=shooting_input,
                           passing=passing_input,
                           dribbling=dribbling_input,
                           train_acc=f"{train_accuracy:.1%}",
                           test_acc=f"{test_accuracy:.1%}")

# ================== RUN ==================
if __name__ == '__main__':
    app.run(debug=True)
