from flask import Flask, render_template
import os

app = Flask(__name__)

# Mensajes de depuración en terminal
print("="*50)
print("✅ SERVIDOR INICIANDO...")
print(f"📁 Directorio actual: {os.getcwd()}")
print(f"📁 Carpeta templates: {app.template_folder}")
print(f"📄 Archivos en templates: {os.listdir(app.template_folder)}")
print("="*50)

@app.route('/')
def home():
    return "hola flask" 

@app.route('/JM')
def jennifer():
    try:
        return render_template('jennifer.html')
    except Exception as e:
        return f"<h1>Error en /JM</h1><p>{str(e)}</p>"

@app.route('/casodeuso')
def medina():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"<h1>Error en /casodeuso</h1><p>{str(e)}</p>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)