from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ✅ Load both model and scaler
model, scaler = joblib.load(open('flood.save', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/intro')
def intro():
    return render_template('intro.html')

@app.route('/predict')
def predict():
    return render_template('index.html')

@app.route('/predict_flood', methods=['POST'])
def predict_flood():
    # Extract input values
    temp = float(request.form['Temp'])
    humidity = float(request.form['Humidity'])
    cloud_cover = float(request.form['cloud_cover'])
    annual_rainfall = float(request.form['ANNUAL'])
    jan_feb = float(request.form['Jan-Feb'])
    mar_may = float(request.form['Mar-May'])
    jun_sep = float(request.form['Jun-Sep'])
    oct_dec = float(request.form['Oct-Dec'])
    avg_june = float(request.form['avgjune'])
    sub = float(request.form['sub'])

    # Prepare input DataFrame
    input_data = pd.DataFrame([[temp, humidity, cloud_cover, annual_rainfall, jan_feb,
                                mar_may, jun_sep, oct_dec, avg_june, sub]],
                              columns=['Temp', 'Humidity', 'Cloud Cover', 'ANNUAL', 'Jan-Feb',
                                       'Mar-May', 'Jun-Sep', 'Oct-Dec', 'avgjune', 'sub'])

    # ✅ Scale the input
    input_scaled = scaler.transform(input_data)

    # ✅ Predict
    prediction = model.predict(input_scaled)
    print("Prediction:", prediction[0])

    return render_template('imageprediction.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
