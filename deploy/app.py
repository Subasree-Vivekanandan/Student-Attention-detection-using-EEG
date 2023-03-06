# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 19:11:15 2023

@author: HP
"""


import numpy as np
from flask import Flask, request,render_template
import pickle as pkl


app = Flask(__name__)
model = pkl.load(open('knn_eeg_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    if(prediction==0.0):
        output = 'Not Confused'
    else:
        output = 'Confused'
    
    
    return render_template('index.html',prediction_text='Student is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')    
        
    