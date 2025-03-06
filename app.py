from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model scikit-learn
model = joblib.load('model_final_3.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        hours = float(request.form['hours'])
        hours_array = np.array([[hours]])
        prediction = model.predict(hours_array)[0]
        prediction_scalar = prediction.item()  # Ambil nilai skalar
        return render_template('index.html', prediction=round(prediction_scalar, 2))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)