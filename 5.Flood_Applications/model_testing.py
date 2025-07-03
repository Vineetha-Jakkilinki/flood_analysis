import joblib
import pandas as pd

# ✅ Load both model and scaler from flood.save
model, scaler = joblib.load(open('flood.save', 'rb'))

# ✅ Test input (replace with real values if needed)
sample_input = pd.DataFrame([[30, 65, 60, 1200, 100, 200, 300, 400, 35, 1]],
    columns=['Temp', 'Humidity', 'Cloud Cover', 'ANNUAL', 'Jan-Feb',
             'Mar-May', 'Jun-Sep', 'Oct-Dec', 'avgjune', 'sub'])

# ✅ Scale the input
scaled_input = scaler.transform(sample_input)

# ✅ Predict
prediction = model.predict(scaled_input)

# ✅ Output
result = "🌊 Flood Likely" if prediction[0] == 1 else "✅ No Flood"
print("🧪 Model Test Result:", result)
