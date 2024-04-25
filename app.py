# from flask import Flask

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return "Hello, World"

# if __name__ == "__main__":
#     app.run(debug = True)



import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd


app = Flask(__name__)
## load the model
regmodel  = pickle.load(open("regmodel.pkl", "rb"))
scaler  = pickle.load(open("scaling.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html") 

@app.route("/predict_api", methods = ["POST"]) # install postman app
def predict_api():
    data = request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route("/predict", methods = ["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    input = scaler.transform(np.array(data).reshape(1, -1))
    print(input)
    output = regmodel.predict(input)[0]
    return render_template("home.html", prediction_text = "Predicted house price is {}".format(output))






if __name__ == "__main__":
    app.run(debug = True)

