#Reading the test file having the data of footballers and target being their
#overall skill score ( 1-100 ).
import pandas as pd
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


from datascientist.feature_selection.filter_based_selection import PearsonCorrelation
Col_sel = PearsonCorrelation(X, y)

#using corr_score method with different parameter values.
score1 = Col_sel.corr_score()
score2 = Col_sel.corr_score(sort = True)
score3 = Col_sel.corr_score(reset_index = True)
score4 = Col_sel.corr_score(sort = True,reset_index = True)

#using top_corr_featurenames method with different parameter values.
topfeatname1 = Col_sel.top_corr_featurenames()
topfeatname2 = Col_sel.top_corr_featurenames(feat_num = 15)
topfeatname3 = Col_sel.top_corr_featurenames(feat_num = 15,ascending = False)
topfeatname4 = Col_sel.top_corr_featurenames(feat_num = 30,ascending = False)

#using top_corr_features method with different parameter values.
X_mod1 = Col_sel.top_corr_features()
X_mod2 = Col_sel.top_corr_features(feat_num = 15)
X_mod3 = Col_sel.top_corr_features(feat_num = 15,ascending = False)
X_mod4 = Col_sel.top_corr_features(feat_num = 30,ascending = False)
