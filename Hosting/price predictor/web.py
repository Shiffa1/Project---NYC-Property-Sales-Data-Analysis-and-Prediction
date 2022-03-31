import numpy as np
from flask import Flask, request, render_template
import pickle
import requests


app=Flask(__name__)
model=pickle.load(open('rf_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])  
def predict():

    borough = str(request.values['borough']).lower()
    if borough == 'bronx':
        borough = 3
    
    elif borough == 'brooklyn':
        borough = 4

    elif borough == 'manhattan':
        borough = 0

    elif borough == 'queens':
        borough = 2

    elif borough == 'staten island':
        borough = 1

    r_units = int(request.values['r_units'])
    c_units = int(request.values['c_units'])
    tot_units = r_units + c_units
    land_area = float(request.values['land_area'])
    gross_area = float(request.values['gross_area'])
    building_age = float(request.values['building_age'])

    arr = np.array([borough, r_units, c_units, tot_units,land_area,gross_area,building_age])

    prediction = model.predict([arr])

    return render_template('result.html', score_prediction = "Your property's estimated price is {} USD".format(float(prediction)))

if __name__=='__main__':
    app.run(port=8000)


