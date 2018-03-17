import flask
import keras
import pandas as pd
import numpy as np
from keras.models import load_model




def feature_scaling(matrix):                #this is the normalization step
    X_max = np.max(matrix, axis=0)
    X_min = np.min(matrix, axis=0)
    return (matrix - X_min)/(X_max - X_min)

def standardardize(matrix):
    mu = np.mean(matrix, axis=0)
    sigma = np.std(matrix, axis=0)
    return (matrix - mu)/sigma

def inverse_feature_scaling(matrix, scaled_matrix):
    return (scaled_matrix*(np.max(matrix['Crude Oil']) - np.min(matrix['Crude Oil'])))+np.min(matrix['Crude Oil'])



df = pd.read_csv('eia_cleaned.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
data_frame = df.sort_index(axis=1 ,ascending=True)
data_frame = df.iloc[::-1]
split_date = pd.Timestamp('01-01-2015')
train = data_frame.loc[:split_date]
test = data_frame.loc[split_date:]


def findMaxMin(matrix):
    X_max = np.max(matrix, axis=0)
    X_min = np.min(matrix, axis=0)
    #print([X_max , X_min])
    return [X_max , X_min]

def findNormalizedValues(inputValues):
    #inputValues = inputValues.set_index('Date')
    print(inputValues)
    minMaxValues = findMaxMin(test)
    print(minMaxValues[0])
    print(minMaxValues[1])
    return (inputValues - minMaxValues[1])/(minMaxValues[0] - minMaxValues[1])

def findInverseTransform(value):
    return inverse_feature_scaling(test , value)

from flask import request , Flask , json

app = Flask(__name__)
@app.route('/', methods=['POST' , 'OPTIONS'])
def predict():
    print('loading model')
    model = load_model('./model_1L_5N_20180125.h5')
    #print(request.json())
    data = [float(request.form['data1']),float(request.form['data2']),float(request.form['data3']),float(request.form['data4']), float(request.form['data5'])]
    print(findNormalizedValues(data))
    X_test = findNormalizedValues(data)
    y_pred = model.predict(np.asarray([X_test]))
    print(y_pred)
    #return flask.jsonify(result=findInverseTransform(y_pred)[0][0])
    return '{}'.format(findInverseTransform(y_pred)[0][0])
