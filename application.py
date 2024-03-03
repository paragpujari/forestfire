import pickle
from flask import Flask,request, jsonify, render_template
import numpy as numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler

application  = Flask(__name__)
app = application

# import ridge regressor and standard scaler pickle

# Here the Web Application will interact with these two models and process the data and will get the results.
ridge_model = pickle.load(open('models/Ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/sc.pkl','rb'))

@app.route("/", methods = ['GET', 'POST'])
def predict_datapoint():
    if(request.method == 'POST'):
        # enter all the input values
        Temperature    = float(request.form.get('Temperature'))
        RH             = float(request.form.get('RH'))
        Ws             = float(request.form.get('Ws'))
        Rain           = float(request.form.get('Rain'))
        FFMC           = float(request.form.get('FFMC'))
        DMC            = float(request.form.get('DMC'))
        ISI            = float(request.form.get('ISI'))
        Classes        = float(request.form.get('Classes'))
        Region         = float(request.form.get('Region'))

        # transform all the inputs and store the tranformed data into a new value
        new_scaled_data = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        result = ridge_model.predict(new_scaled_data)

        return render_template('home.html', results = result[0])
    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
