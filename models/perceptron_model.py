import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import seaborn as sns
import io
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from perceptron import Perceptron

#---------- Graphs base64 ----------
def get_plot_url():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf-8')

#--------- Preparing data ---------------
def prepare_perceptron_data(df_perceptron):
    df_p = df_perceptron.copy()

    if 'id' in df_p.columns: 
        df_p = df_p.drop(columns=['id'])

    if 'Unnamed: 32' in df_p.columns: 
        df_p = df_p.drop(columns=['Unnamed: 32'])

    le = LabelEncoder()
    df_p['diagnosis'] = le.fit_transform(df_p['diagnosis'])

    X = df_p.iloc[:, 1:11].values
    y = df_p['diagnosis'].values
    return X, y

#---------- Training and graphs ----------
def train_perceptron(df_perceptron):

    scaler_p = StandardScaler()
    p_or = Perceptron(learning_rate=0.1, epochs=100)

    #---------- Data ----------
    X, y = prepare_perceptron_data(df_perceptron)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_scaled = scaler_p.fit_transform(X_train)
    X_test_scaled = scaler_p.transform(X_test)

    #---------- Trainning Data ----------
    p_or.fit(X_train_scaled, y_train)

    #---------- Calculation of metrics ----------
    y_pred = p_or.predict(X_test_scaled)
    res_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4), 
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4) 
    }

    #---------- Graphics ----------
    #---------- Matrix of Confusion ----------
    plt.figure(figsize=(4,3))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title('Matrix of Confusion')
    plot_url_cm = get_plot_url()

    #---------- Curve ROC ----------
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure(figsize=(4,3))
    plt.plot(fpr, tpr, label=f'ROC curve = {auc(fpr, tpr):.2f}')
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.legend()
    plot_url_roc = get_plot_url()

    #---------- ERRORS ----------
    plt.figure(figsize=(5, 4))
    plt.plot(range(1, len(p_or.errors_per_epoch) + 1), p_or.errors_per_epoch, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.title('Training Error')
    plot_url_errors = get_plot_url()

    return {
        "metrics": res_metrics,
        "plot_url_cm": plot_url_cm,
        "plot_url_roc": plot_url_roc,
        "plot_url_errors": plot_url_errors,
        "model": p_or,
        "scaler": scaler_p,
        "X": X
    }

    #---------- Prediction ----------
def prediction_perceptron(model, scaler, X, form_data):

    avg_values = np.mean(X, axis=0)

    inputs = [
        float(form_data['radius']),
        float(form_data['texture']),
        float(form_data['perimeter']),
        float(form_data['area']),
        avg_values[4], avg_values[5], avg_values[6],
        avg_values[7], avg_values[8], avg_values[9]
    ]

    input_scaled = scaler.transform([inputs])
    prediction = int(model.predict(input_scaled)[0])

    return prediction