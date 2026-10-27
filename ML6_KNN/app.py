from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
data = joblib.load("KNNDiabetesModel.joblib")

model = data["model"]
scaler = data["scaler"]
features = data["feature_names"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from HTML form
        input_data = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"])
        ]

        # Convert to numpy array
        input_array = np.array(input_data).reshape(1, -1)

        # Scale input using the same scaler used during training
        input_scaled = scaler.transform(input_array)

        # Prediction
        prediction = model.predict(input_scaled)[0]

        # Probability
        probability = model.predict_proba(input_scaled)[0][1] * 100

        if prediction == 1:
            result = "Diabetes Detected"
        else:
            result = "No Diabetes Detected"

        return render_template(
            "index.html",
            prediction=result,
            probability=round(probability, 2)
        )

    except Exception as e:
        return render_template(
            "index.html",
            error="Please enter valid values."
        )


if __name__ == "__main__":
    app.run(debug=True)