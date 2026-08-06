from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("main.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        age = int(request.form["age"])

        features = np.array([[area, bedrooms, age]])

        prediction = model.predict(features)[0]

        # Display in Indian Rupees
        formatted_price = f"₹ {prediction:,.2f}"

        return render_template(
            "main.html",
            prediction_text=f"Estimated House Price: {formatted_price}"
        )

    except Exception as e:
        return render_template(
            "main.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)