#Reading the test file having the data of footballers and target being their
#overall skill score ( 1-100 ).
import pandas as pd
import numpy as np

player_df = pd.read_csv("datascientist/feature_selection/test/CSV/data.csv")

#Taking only those columns which have numerical or categorical values since 
#feature selection with Pearson Correlation can be performed on numerical data.
numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling',
           'LongPassing', 'BallControl', 'Acceleration','SprintSpeed',
           'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance',
           'ShotPower','Strength','LongShots','Aggression','Interceptions']
catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']
player_df = player_df[numcols+catcols]

#encoding categorical values with one-hot encoding.
traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

#dropping rows with Nan values
traindf = traindf.dropna()
traindf = pd.DataFrame(traindf,columns=features)

#Separating features(X) and target(y).
y = traindf['Overall']
X = traindf.copy()
X = X.drop(['Overall'],axis = 1)

#Generating the expected results for assert statements
from scipy import stats
columns = X.columns.tolist()
coeff = []
for column in columns:
       coeff.append(round(stats.pearsonr(X[column],y)[0],6))
coeff = [0 if np.isnan(i) else i for i in coeff]

#for corr_score method with different parameter values
a1 = pd.DataFrame(columns = ['feature','cor_score'])
a1['feature'] = columns
a1['cor_score'] = coeff
a2 = a1.sort_values(by = ['cor_score'],ascending = False)
a3 = a1.reset_index(drop = True)
a4 = a2.reset_index(drop = True)

#for top_corr_featurenames method with different parameter values/
b1 = X.iloc[:,np.argsort(np.abs(coeff))[-(1):]].columns.tolist()
b2 = X.iloc[:,np.argsort(np.abs(coeff))[-(15):]].columns.tolist()
b3 = b2[::-1]
b4 = X.iloc[:,np.argsort(np.abs(coeff))[-(30):]].columns.tolist()[::-1]

#for top_corr_features method with different parameter values.
C1 = X[b1]
C2 = X[b2]
C3 = X[b3]
C4 = X[b4]

from datascientist.feature_selection.filter_based_selection import PearsonCorrelation
Col_sel = PearsonCorrelation(X, y)

#using corr_score method with different parameter values.
score1 = Col_sel.corr_score()
assert score1.equals(a1)

score2 = Col_sel.corr_score(sort = True)
assert score2.equals(a2)

score3 = Col_sel.corr_score(reset_index = True)
assert score3.equals(a3)

score4 = Col_sel.corr_score(sort = True,reset_index = True)
assert score4.equals(a4)

#using top_corr_featurenames method with different parameter values.
topfeatname1 = Col_sel.top_corr_featurenames()
assert topfeatname1 == b1

topfeatname2 = Col_sel.top_corr_featurenames(feat_num = 15)
assert topfeatname2 == b2

topfeatname3 = Col_sel.top_corr_featurenames(feat_num = 15,ascending = False)
assert topfeatname3 == b3

topfeatname4 = Col_sel.top_corr_featurenames(feat_num = 30,ascending = False)
assert topfeatname4 == b4

#using top_corr_features method with different parameter values.
X_mod1 = Col_sel.top_corr_features()
assert X_mod1.equals(C1)

X_mod2 = Col_sel.top_corr_features(feat_num = 15)
assert X_mod2.equals(C2)

X_mod3 = Col_sel.top_corr_features(feat_num = 15,ascending = False)
assert X_mod3.equals(C3)

X_mod4 = Col_sel.top_corr_features(feat_num = 30,ascending = False)
assert X_mod4.equals(C4)