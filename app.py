
import numpy as np
import pandas as pd
import pickle
import sys

from flask import Flask, request, render_template

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Student_Performance.csv")

# Separate independent variables
X = df.iloc[:, :-1]

# Apply same encoding as main.py
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Load trained model
with open("Std.Pred.pkl", "rb") as file:
    model = pickle.load(file)


# HOME PAGE
@app.route("/")
def home():
    return render_template("index.html")


# PREDICTION
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get input values from HTML
        hours = float(request.form.get("hours"))
        scores = float(request.form.get("scores"))
        activities = request.form.get("activities")
        sleep = float(request.form.get("sleep"))
        papers = float(request.form.get("papers"))

        # Create input DataFrame
        input_data = pd.DataFrame([{
            'Hours Studied': hours,
            'Previous Scores': scores,
            'Extracurricular Activities': activities,
            'Sleep Hours': sleep,
            'Sample Question Papers Practiced': papers
        }])

        # Apply same encoding
        input_data = pd.get_dummies(
            input_data,
            drop_first=True,
            dtype=int
        )

        # Match training columns
        input_data = input_data.reindex(
            columns=X.columns,
            fill_value=0
        )

        # Predict
        result = model.predict(input_data)

        prediction = round(float(result[0]), 2)

        return render_template(
            "index.html",
            prediction=prediction
        )

    except Exception as e:

        er_ty, er_msg, er_line = sys.exc_info()

        print(
            f"Error in line no: {er_line.tb_lineno} : "
            f"{er_msg} : {er_ty}"
        )

        return render_template(
            "index.html",
            error="Please enter valid details."
        )


# MAIN EXECUTION
if __name__ == "__main__":

    app.run(debug=True)
