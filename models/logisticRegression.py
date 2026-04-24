import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def train_predict_logistic(df_logistic, pace_input, shooting_input, passing_input, dribbling_input):
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
    return {
        "result": result,
        "probability": f"{pred_prob:.1%}",
        "pace": pace_input,
        "shooting": shooting_input,
        "passing": passing_input,
        "dribbling": dribbling_input,
        "train_acc": f"{train_accuracy:.1%}",
        "test_acc": f"{test_accuracy:.1%}"
    }