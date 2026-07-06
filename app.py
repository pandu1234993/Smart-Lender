from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    gender = int(request.form["Gender"])
    married = int(request.form["Married"])
    income = float(request.form["ApplicantIncome"])
    loan = float(request.form["LoanAmount"])
    credit = float(request.form["Credit_History"])

    data = np.array([[gender, married, income, loan, credit]])

    prediction = model.predict(data)

    print("Prediction:", prediction)

    if int(prediction[0]) == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)