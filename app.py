from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import seaborn as sns
import io
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from perceptron import Perceptron


app = Flask(__name__)

# ================== DATASET ==================

df = pd.read_csv('medals.csv')
df_logistic = pd.read_csv('fifa24.csv')
df_perceptron = pd.read_csv('breastCancerWisconsin.csv')

#---------- Graphs base64 ----------
def get_plot_url():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

# ================== ROUTE ==================

@app.route('/')
def home():
    return render_template('mainMenu.html')

#---------- USE CASE ----------

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

#---------- LINEAR REGRESSION ----------

@app.route('/linearRegression/concepts')
def linealConcept():
    return render_template('linearRegression/linealRConcepts.html')

@app.route('/linearRegression/application')
def linealApplication():
    columns = list(df.select_dtypes(include='number').columns)
    return render_template('linearRegression/linealRApplication.html', columns=columns)

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

    return render_template('linearRegression/linealRApplication.html',
                           columns=columns,
                           prediction=round(prediction, 2),
                           x_col=x_col,
                           input_value=input_value)

#---------- LOGISTIC REGRESSION ----------

@app.route('/logisticRegression/concepts')
def logisticConcept():
    return render_template('logisticRegression/logisticConcepts.html')


@app.route('/logisticRegression/application')
def logisticApplication():
    return render_template('logisticRegression/logisticApplication.html')


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
    return render_template('logisticRegression/logisticApplication.html',
                           result=result,
                           probability=f"{pred_prob:.1%}",
                           pace=pace_input,
                           shooting=shooting_input,
                           passing=passing_input,
                           dribbling=dribbling_input,
                           train_acc=f"{train_accuracy:.1%}",
                           test_acc=f"{test_accuracy:.1%}")

#---------- CLASSIFICATION PERPECTRON ----------

@app.route ('/perceptron/metrics')
def perceptron_metrics():
    if 'id' in df_perceptron.columns: df_perceptron.drop(columns=['id'])
    if 'Unnamed: 32' in df_perceptron.columns: df_perceptron.drop(columns=['Unnamed: 32'])

    le = LabelEncoder()
    df_perceptron['diagnosis'] = le.fit_transform(df_perceptron['diagnosis'])

    x = df_perceptron.iloc[:,1:11].values
    y = df_perceptron['diagnosis'].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #---------- Trainning ----------
    p_or = Perceptron(learning_rate=0.1, epochs=100)
    p_or.fit(X_train_scaled, y_train)

    #---------- 
    y_pred = p_or.predict(X_test_scaled)

    res_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4), 
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4) 
    }

    plt.figure(figsize=(5,4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title('Matrix of Confusion - Breast Cancer')
    plot_url_cm = get_plot_url
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:2f})')
    plt.plot([0,1], [0,1], color='navy', linestyle='--')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.legend(loc="lower right")
    plot_url_roc = get_plot_url()
    plt.close()

    return render_template('classificationPerceptron/perceptronEvaluationMetrics.html',
                           metrics=res_metrics,
                           plot_url_cm=plot_url_cm,
                           plot_url_roc=plot_url_roc)

# ================== RUN ==================
if __name__ == '__main__':
    app.run(debug=True)
