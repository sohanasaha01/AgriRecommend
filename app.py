from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoder
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""

    if request.method == "POST":
        try:
            N = float(request.form["nitrogen"])
            P = float(request.form["phosphorus"])
            K = float(request.form["potassium"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = model.predict(input_data)
            crop = le.inverse_transform(prediction)

            prediction_text = f"Recommended Crop: {crop[0]}"

        except:
            prediction_text = "Invalid Input. Please enter valid numbers."

    return render_template("index.html", prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)