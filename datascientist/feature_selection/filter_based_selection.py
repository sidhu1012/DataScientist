# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:28:56 2020

@author: parva
"""


import pandas as pd
import numpy as np

'''Note:- Only numerical or categorcial values encoded to numerical can be passed as features or target. Anything else data type will throw error if passed. '''

class PearsonCorrelation:
    
    '''https://en.wikipedia.org/wiki/Pearson_correlation_coefficient'''
    
    '''Params:
        features:Features are individual independent variables that act as the input in your system.
        Prediction models use features to make predictions.
        
        target : The target is whatever the output of the input variables.
        It could be the individual classes that the input variables maybe mapped to in case of a classification problem or the output value range in a regression problem.     
        '''
    
    def __init__(self,features,target):
        
        self.X = features
        self.y = target
        
    
    
    def corr_score(self,sort = False,reset_index = False):
        
        '''Evaluate the Pearson Correlation Coefficient of each feature column with the target column.
        
        Parameters
        ----------
        - sort: boolean value(default = False),if set True, then arranges the features in descending order of coefficients value.
        - reset_index: boolean value(default = False),if set True, then resets the index of output dataframe.
        
        Returns
        -------
        -cor_score: DataFrame with one column as the features name and the other as Coefficient value.'''
                       
        cor_list = []
        feature_name = self.X.columns.tolist()
        cor_score = pd.DataFrame(columns = ['feature','cor_score'])
        for i in feature_name:
            cor = np.corrcoef(self.X[i], self.y)[0,1]
            cor_list.append(cor)
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        cor_score['feature'] = feature_name
        cor_score['cor_score'] = cor_list
        if sort == True:
            cor_score = cor_score.sort_values(by = ['cor_score'],ascending = False)
        if reset_index == True:
            cor_score = cor_score.reset_index(drop = True)
        
        return cor_score
    
    
    def top_corr_featurenames(self,feat_num = 1,ascending = True):
        
        '''Parameters
        ----------
        - feat_num: integer value(default = 1),can take value from 1 to total no. of features, to decide the top feat_num of features according to their coefficient value. 
        - ascending: boolean value(default = True),if set False, then the feature with higher coefficient value is first element folloewed by lower ones. 
        
        Returns
        -------
        -cor_feature: List containing top feat_num features name. '''
        
        cor_list = []
        feature_name = self.X.columns.tolist()
        # cor_score = pd.DataFrame(columns = ['feature','cor_score'])
        for i in feature_name:
            cor = np.corrcoef(self.X[i], self.y)[0,1]
            cor_list.append(cor)
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        cor_feature = self.X.iloc[:,np.argsort(np.abs(cor_list))[-(feat_num):]].columns.tolist()
        if ascending == False:
            cor_feature = cor_feature[::-1]
        return cor_feature
    
    
    def top_corr_features(self,feat_num = 1,ascending = True):
        
        '''Parameters
        ----------
        - feat_num: integer value(default = 1),can take value from 1 to total no. of features, to decide the top feat_num of features according to their coefficient value. 
        - ascending: boolean value(default = True),if set False, then the feature with higher coefficient value is first column folloewed by lower ones. 
        
        Returns
        -------
        -out_feature: DataFrame containing top feat_num features with all the data(rows). '''
        
        cor_list = []
        feature_name = self.X.columns.tolist()
        # cor_score = pd.DataFrame(columns = ['feature','cor_score'])
        for i in feature_name:
            cor = np.corrcoef(self.X[i], self.y)[0,1]
            cor_list.append(cor)        
        cor_feature = self.X.iloc[:,np.argsort(np.abs(cor_list))[-(feat_num):]].columns.tolist()
        if ascending == False:
            cor_feature = cor_feature[::-1]
        out_features = self.X[cor_feature]
        return out_features
    
    
    