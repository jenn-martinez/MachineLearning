from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', titulo = "First Page",
                           des = "Machine Learning Class")

@app.route ('/FirstPage')
def firstPage():
    return render_template('index.html', titulo2 = "Second Page",
                           descri = "Monster High")