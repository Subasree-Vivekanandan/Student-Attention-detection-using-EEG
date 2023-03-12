# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 19:11:15 2023

@author: HP
"""


import numpy as np
from flask import Flask, request,render_template
import pickle as pkl


app = Flask(__name__)
knn_model = pkl.load(open('knn_eeg_model.pkl','rb'))
svm_model = pkl.load(open('svm_eeg_model.pkl','rb'))
lr_model = pkl.load(open('log_reg_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/svm')
def svm_page():
    return render_template('svm.html')

@app.route('/predictsvm',methods=['POST'])
def predict_svm():
    svmf = [float(x) for x in request.form.values()]
    svm_final = [np.array(svmf)]
    svm_prediction = svm_model.predict(svm_final)
    
    if (svm_prediction==0.0):
        svm_output = 'Not Confused'
    else:
        svm_output = 'Confused'
    
    return  render_template('predict.html',prediction_text='Student is {}'.format(svm_output))

@app.route('/logreg')
def logreg_page():
    return render_template('logreg.html')

@app.route('/lrpredict',methods=['POST'])
def predict_lr():
    lr_features =  [float(x) for x in request.form.values()]
    lr_final = [np.array(lr_features)]
    lr_prediction = lr_model.predict(lr_final)
    
    if(lr_prediction==0.0):
        lr_output = 'Not Confused'
    else:
        lr_output = 'Confused'
    
    return render_template('predict.html',prediction_text='Student is {}'.format(lr_output))
    
@app.route('/knn')
def knn_page():
    return render_template('knn.html')

@app.route('/knnpredict',methods=['POST'])
def predict_knn():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = knn_model.predict(final_features)
    
    if(prediction==0.0):
        output = 'Not Confused'
    else:
        output = 'Confused'
    
    
    return render_template('predict.html',prediction_text='Student is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)    
        
    
    
    