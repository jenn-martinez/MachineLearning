from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

app = Flask(__name__)


df = pd.read_csv('medals.csv')


print(df.columns)


X = df[['Gold Medal', 'Silver Medal', 'Bronze Medal']]
y = df['Total']


model = LinearRegression()
model.fit(X, y)


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



@app.route('/')
def home():
    return render_template('mainMenu.html')


@app.route('/SG')
def sg_page():
    return render_template('Sergio.html')


@app.route('/BBVA')
def jm_page():
    return render_template('BBVAPipeline.html', titulo="Case One",
                           des="Machine Learning Class")


@app.route('/MD')
def md_page():
    return render_template('medina.html')


@app.route('/SB')
def sb_page():
    return render_template('santiago.html')


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



if __name__ == '__main__':
    app.run(debug=True)