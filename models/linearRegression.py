from flask import request
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def train_predict(df, x_col, input_values):
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

    return round(prediction, 2)
