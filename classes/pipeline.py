# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 22:25:20 2023

@author: Bernardo A V de Souza
"""

import pickle
import sklearn
import xgboost


class CrossSell():
    
    
    def __init__(self):
        self.fe_dict_rc = pickle.load(open('parameter/fe_dict_rc_encoding.pkl', 'rb'))
        self.le_va = pickle.load(open('parameter/va_encoding.pkl', 'rb'))
        self.mms_age = pickle.load(open('parameter/mms_age_scaler.pkl', 'rb'))
        self.model = pickle.load(open('models/final_model.pkl', 'rb'))
        
        
    
    
    def data_cleaning(self, df1):
        ## 1.1. Rename Columns

        new_cols_name = ['ID', 'Gender', 'Age', 'HaveDrivingLicense', 'RegionCode',
                         'HaveInsurance', 'VehicleAge', 'HaveDamagedVehicle', 'AnnualCost',
                         'CommunicationChannel', 'CustomerSinceDays']
   
        df1.columns = new_cols_name
        
        ## 1.6. Changing Data Types
        
        # RegionCode
        df1['RegionCode'] = df1['RegionCode'].astype('object')
        
        # HaveDamagedVehicle
        df1['HaveDamagedVehicle'] = df1['HaveDamagedVehicle'].map({'Yes': 1, 'No': 0 })
        
        # CommunicationChannel
        df1['CommunicationChannel'] = df1['CommunicationChannel'].astype('object')
        
        
        # Data Selection
        selected_columns = ['Gender', 'Age', 'HaveDrivingLicense', 'RegionCode',
       'HaveInsurance', 'VehicleAge', 'HaveDamagedVehicle']
        
        df1 = df1.loc[:, selected_columns]
    
        return df1
    
    
    
    def data_preparation(self, df):
        df = self.label_encoding_apply(df)
        df = self.frequency_encoding_apply(df)
        df = self.min_max_apply(df)
        
        return df
        
    
    
    def make_prediction(self, df_raw):
        df = df_raw.copy()
        df = self.data_cleaning(df)
        df = self.data_preparation(df)
        df_out = self.final_df(df, df_raw)
        
        return df_out
        
        
    
    def frequency_encoding_apply(self, df):
        df['RegionCode'] = df['RegionCode'].map(self.fe_dict_rc)
        
        return df



    def label_encoding_apply(self, df):
        # Gender
        df['Gender'] = df['Gender'].map({'Female': -1, "Male": 1})
    
        # VehicleAge
        df['VehicleAge'] = self.le_va.transform(df['VehicleAge'])
        
        return df
    
     

    def min_max_apply(self, df):
        # Age
        df['Age'] = self.mms_age.transform(df[['Age']])
        
        return df
    
    
    
    def final_df(self, df_in, df_raw):
        propensity = self.model.predict_proba(df_in)[:, 1]
        df_raw['Propensity'] = propensity
        
        df = df_raw.sort_values('Propensity', ascending = False)
        df['Propensity_Order'] = range(1, df.shape[0] + 1)
        df = df.drop('Propensity', axis = 1)
        
        return df
    
    
    

        
        
        
        