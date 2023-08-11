# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 22:22:33 2023

@author: Bernardo A V de Souza

"""

import pandas as pd
from flask import Flask, request, Response
from classes.pipeline import CrossSell



app = Flask(__name__)

@app.route('/cross_sell/predict', methods = ['POST'])
def predict():
    test_json = request.get_json()

    if test_json:
        if isinstance( test_json, dict ): # unique example
            df = pd.DataFrame( test_json, index=[0] )
   
        else: # multiple example
            df = pd.DataFrame( test_json, columns=test_json[0].keys() )
        pipeline = CrossSell()   
        df_out = pipeline.make_prediction(df)
    
        return df_out.to_dict(orient = 'records')

    else:    
        return Response( '{}', status=200, mimetype='application/json' )
    
    
    
    
    
if __name__ == '__main__':
    app.run('127.0.0.1')