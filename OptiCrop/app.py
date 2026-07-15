from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load saved files
model = pickle.load(open("crop_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    # Create input array
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Scale the input
    data = scaler.transform(data)

    # Predict
    prediction = model.predict(data)

    # Convert numeric label back to crop name
    crop = encoder.inverse_transform(prediction)

    return render_template("result.html", prediction=crop[0])

if __name__ == "__main__":
    app.run(debug=True)