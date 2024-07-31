#import clf as clf
import joblib as joblib
from flask import Flask, render_template, redirect, request, session,jsonify
#from flask_session import Session
from flask import Flask, render_template, request
import pickle
import pandas as pd
import json
import os
import numpy as np
import os
app=Flask(__name__)

model = pickle.load(open('best_rf_model.pkl', 'rb'))
@app.route('/')
def home():
    return "Depression Prediction Model API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_features = np.array(data['input']).reshape(1, -1)
    prediction = model.predict(input_features)
    return jsonify({'prediction': prediction.tolist()})
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5001,debug=True)
