from flask import Flask,request,render_template
import joblib
import numpy as np
app=Flask(__name__)

model=joblib.load("model.pkl")
@app.route("/")
def home():
    return render_template("main.html")

@app.route("/predict",methods=["POST"])
def predict():
    area=float(request.form["area"])
    bedrooms=int(request.form["bedrooms"])
    age=int(request.form["age"])
    
    features=np.array([[area,bedrooms,age]])
    prediction=model.predict(features)[0]
    return render_template("main.html",prediction_text=f"Estimated price:₹{prediction:,.0f}")
if __name__=="__main__":
    app.run(debug=True)
    