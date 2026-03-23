from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/SG')
def sg_page():
    return render_template('Sergio.html')

@app.route('/JM')
def jm_page():
    return render_template('jennifer.html', titulo = "First Page",
                           des = "Machine Learning Class")

if __name__ == '__main__':
    app.run(debug=True)

