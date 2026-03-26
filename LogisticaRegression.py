import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import threading
from flask import Flask, render_template, request
from pathlib import Path

# Datos y modelo
BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / 'dataset_regresion_logistica.csv')
print(df.head())
print(df.info())
print(df.describe())

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Exactitud:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.switch_backend('TkAgg')
plt.show()  # cuando cierras esta ventana → arranca Flask

# Flask arranca después de cerrar la figura
app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'))

@app.route("/RegresionLogistica", methods=["GET", "POST"])
def regresion_logistica():
    prediccion = None
    if request.method == "POST":
        edad = float(request.form["edad"])
        ingreso = float(request.form["ingreso_mensual"])
        visitas = float(request.form["visitas_web_mes"])
        tiempo = float(request.form["tiempo_sitio_min"])
        compras = float(request.form["compras_previas"])
        descuento = float(request.form["descuento_usado"])
        resultado = model.predict([[edad, ingreso, visitas, tiempo, compras, descuento]])
        prediccion = "sÍ compra" if resultado[0] == 1 else " NO compra"
    return render_template('RegresionLogistica.html', prediccion=prediccion)

if __name__ == "__main__":
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000/RegresionLogistica')
    app.run(debug=False)
