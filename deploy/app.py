# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 19:11:15 2023

@author: HP
"""

import pandas as pd
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
    return render_template('svm_upload.html')

@app.route('/predictsvm',methods=['POST'])
def predict_svm():
    f = request.files.get('fileupload')
    df = pd.read_csv(f, encoding='latin-1')
    numpy_f = df.to_numpy()
    y_preds = []
    output = []
    
    student_id = list(df.SubjectID)
    attention = list(df["Attention"])
    for i in range(len(numpy_f)):
        features = [numpy_f[i]]
        y_pr = svm_model.predict(features)
        y_p = y_pr.item()
        y_preds.append(y_p)
        
    for i in range(len(y_preds)):
        if(y_preds[i]==1.0):
            output.append("Confused")
        else:
            output.append("Not Confused")
    
        
    return render_template('check_p.html', Student=student_id,Attention=attention,Output=output)

@app.route('/logreg')
def logreg_page():
    return render_template('lr_upload.html')

@app.route('/lrpredict',methods=['POST'])
def predict_lr():
    f = request.files.get('fileupload')
    df = pd.read_csv(f, encoding='latin-1')
    numpy_f = df.to_numpy()
    y_preds = []
    output = []
    
    student_id = list(df.SubjectID)
    attention = list(df["Attention"])
    for i in range(len(numpy_f)):
        features = [numpy_f[i]]
        y_pr = lr_model.predict(features)
        y_p = y_pr.item()
        y_preds.append(y_p)
        
    for i in range(len(y_preds)):
        if(y_preds[i]==1.0):
            output.append("Confused")
        else:
            output.append("Not Confused")
    
        
    return render_template('check_p.html', Student=student_id,Attention=attention,Output=output)
    
@app.route('/knn')
def knn_page():
    return render_template('knn_upload.html')

@app.route('/knnpredict',methods=['POST'])
def predict_knn():
    f = request.files.get('fileupload')
    df = pd.read_csv(f, encoding='latin-1')
    numpy_f = df.to_numpy()
    y_preds = []
    output = []
    
    student_id = list(df.SubjectID)
    attention = list(df["Attention"])
    for i in range(len(numpy_f)):
        features = [numpy_f[i]]
        y_pr = knn_model.predict(features)
        y_p = y_pr.item()
        y_preds.append(y_p)
        
    for i in range(len(y_preds)):
        if(y_preds[i]==1.0):
            output.append("Confused")
        else:
            output.append("Not Confused")
    
        
    return render_template('check_p.html', Student=student_id,Attention=attention,Output=output)

#@app.route('/upload')
#def upload_route_summary():
#    return render_template('upload.html')

#@app.route('/file',methods=['POST'])
#def file_upload():
#       f = request.files.get('fileupload')
#       df = pd.read_csv(f, encoding='latin-1')
#       output = df.to_numpy()
#       output_arr = np.array(output)
       
#       return render_template('display.html', dict_output=output_arr)
    
if __name__ == "__main__":
    app.run(debug=True)    
        
    
    
    