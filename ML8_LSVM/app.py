
from flask import Flask, render_template, request
import joblib
import math

app = Flask(__name__)

# Load the trained Linear SVM model
model = joblib.load("banknote_svm.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    values = {
        "variance": "",
        "skewness": "",
        "curtosis": "",
        "entropy": ""
    }

    if request.method == "POST":
        try:
            # Get input values from the HTML form
            for feature in values:
                values[feature] = request.form.get(
                    feature, ""
                ).strip()

                if not values[feature]:
                    raise ValueError(
                        "Please enter all four feature values."
                    )

            variance = float(values["variance"])
            skewness = float(values["skewness"])
            curtosis = float(values["curtosis"])
            entropy = float(values["entropy"])

            features = [
                variance,
                skewness,
                curtosis,
                entropy
            ]

            if not all(math.isfinite(x) for x in features):
                raise ValueError(
                    "Please enter valid finite numbers."
                )

            # Predict using the trained SVM pipeline
            result = int(model.predict([features])[0])

            if result == 0:
                prediction = "Genuine Banknote"
            else:
                prediction = "Counterfeit Banknote"

        except ValueError as e:
            error = str(e) or "Please enter valid numbers."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)
