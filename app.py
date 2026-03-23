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

@app.route('/MD')
def md_page():
    try:
        return render_template('medina.html')
    except Exception as e:
        return f"<h1>Error en /casodeuso</h1><p>{str(e)}</p>"

@app.route('/SB')
def firstPage():
    return render_template('santiago.html')

if __name__ == '__main__':
    app.run(debug=True)
