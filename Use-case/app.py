from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return"hello flask"

@app.route('/casodeuso')
def firstPage():
    return render_template('index.html')