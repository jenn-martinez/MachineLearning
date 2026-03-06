from flask import Flask, render_template

app = Flask(__name__)

@app.route('/Jennyfer')
def home():
    return render_template('index.html', titulo = "First Page",
                           des = "Machine Learning Class")
