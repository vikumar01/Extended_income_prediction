import numpy as np
#import matplotlib.pyplot as plt'
import pandas as pd
#from sklearn import metrics
from sklearn import preprocessing

#importing dataset
X = pd.read_csv('tcd-ml-1920-group-income-train.csv')
X=X.drop_duplicates()
X1 = pd.read_csv('tcd-ml-1920-group-income-test.csv')


#Rename columns to not contain spaces
newnames = {"Year of Record" : "Year",
            "Housing Situation" : "House",
          "Crime Level in the City of Employement" : "Crime",
           "Work Experience in Current Job [years]" : "WorkExp",
           "Satisfation with employer" : "Satisfaction",
           "Size of City" : "Size",
           "University Degree" : "Degree",
           "Wears Glasses" : "Glasses",
           "Hair Color" : "Hair",
           "Body Height [cm]" : "Height",
           "Yearly Income in addition to Salary (e.g. Rental Income)" : "Additional_income",
           "Total Yearly Income [EUR]" : "Income"
          }


X.rename(columns=newnames, inplace=True)
X1.rename(columns=newnames, inplace=True)


def preprocess(dataset):
    #dataset = dataset[newnames]
    p_gender(dataset)
    p_age(dataset)
    p_year(dataset)
    p_profession(dataset)
    p_degree(dataset)
    p_hair(dataset)
    p_house(dataset)
    p_workexp(dataset)
    p_satisfaction(dataset)
    p_addIncome(dataset)
#    p_encoding(dataset)
    return dataset

    
#merging Gender
def p_gender(X):
    X["Gender"] = X["Gender"].astype('category')
    X["Gender_cat"] = X["Gender"].cat.codes
    X.replace(X["Gender"],X["Gender_cat"])
    del X["Gender"]
    X.Gender = X.Gender_cat.replace( 'other' ,'missing_gender')
    X.Gender = X.Gender_cat.replace( 'f' ,'female')
    X.Gender = X.Gender_cat.replace( np.NaN ,'missing_gender') 
    X.Gender = X.Gender_cat.replace( 'unknown' ,'missing_gender')
    X.Gender = X.Gender_cat.replace( '0' ,'missing_gender')

#removing age more than 100
#X = X[X['Age'] <= 100]
def p_age(X):
    age_median = X['Age'].median()
    X['Age'].replace(np.nan, age_median, inplace=True)
    #X['Age'] = (X['Age'] * X['Age']) ** (0.5)
    
def p_year(X):
    #Replacing missing_year year with median
    #p=X["Year"].mean()
    X.Year = X.Year.replace( np.NaN ,X.Year.median())
    
def p_profession(X):
    # Transform profession data into categories codes
    X["Profession"] = X["Profession"].astype('category')
    X["profession_cat"] = X["Profession"].cat.codes
    X.replace(X["Profession"],X["profession_cat"])
    del X["Profession"]
    X.profession_cat = X.profession_cat.replace( '0' ,np.NaN)
    X.profession_cat = X.profession_cat.replace( np.NaN ,"missing_prof")
    
    
def p_degree(X):
    #merging University Degree
    X["Degree"] = X["Degree"].astype('category')
    X["Degree_cat"] = X["Degree"].cat.codes
    X.replace(X["Degree"],X["Degree_cat"])
    del X["Degree"]
    X.Degree_cat = X.Degree_cat.replace( '0' ,np.NaN)
    X.Degree_cat = X.Degree_cat.replace( np.NaN ,"missing_degree")
    
    
def p_hair(X):
    #merging Hair Colour
    X["Hair"] = X["Hair"].astype('category')
    X["Hair_cat"] = X["Hair"].cat.codes
    X.replace(X["Hair"],X["Hair_cat"])
    del X["Hair"]
    X.Hair_cat = X.Hair_cat.replace( '0' ,np.NaN)
    X.Hair_cat = X.Hair_cat.replace( np.NaN ,"missing_hair")

def p_house(X):
    #merging Housing Situation
    X.House = X.House.replace( 'Medium Apartment','Medium House')
    X.House = X.House.replace( 'Small Apartment' ,'Small House')
    X.House = X.House.replace( 'Large Apartment' ,'Large House')
    X["House"] = X["House"].astype('category')
    X["House_cat"] = X["House"].cat.codes
    X.replace(X["House"],X["House_cat"])
    del X["House"] 
    X.House_cat = X.House_cat.replace( 'nA' ,np.NaN)
    X.House_cat = X.House_cat.replace( '0' ,np.NaN)
    X.House_cat = X.House_cat.replace( np.NaN ,"missing_house")
    
def p_workexp(X):
    #merging work experience
    X.WorkExp = X.WorkExp.replace( '#NUM!', np.NaN)
    X.WorkExp = X.WorkExp.replace( np.NaN ,X.WorkExp.median())
    #the datatype was object so converted to float
    X['WorkExp'].astype(float)
    X.WorkExp.dtype                                 
       
def p_satisfaction(X):                                                
    #merging satis 
    X["Satisfaction"] = X["Satisfaction"].astype('category')
    X["Satisfaction_cat"] = X["Satisfaction"].cat.codes
    X.replace(X["Satisfaction"],X["Satisfaction_cat"])
    del X["Satisfaction"]
    X.Satisfaction_cat.replace( np.NaN ,'missing_Satis')
     
def p_addIncome(X):   
    #Extra income to be changed to int from string
    X.Additional_income = X.Additional_income.astype(str).str.rstrip(' EUR')
    X.Additional_income.dtype
    #Now converting this from string to int
    X['Additional_income'] = X['Additional_income'].astype(float)


X = preprocess(X)
X1 = preprocess(X1)


from category_encoders import TargetEncoder
y = X.Income
y = y - X['Additional_income']
X = X.drop('Income', 1)
X = X.drop('Instance', 1)
X = X.drop('Additional_income',1)

y1 = X1.Income
y1 = y1 - X1['Additional_income']
X1 = X1.drop('Income', 1)
X1 = X1.drop('Instance', 1)
temp = X1['Additional_income']
X1 = X1.drop('Additional_income',1)

t1 = TargetEncoder()
t1.fit(X,y)
X = t1.transform(X)
X1 = t1.transform(X1)

mm_scaler = preprocessing.MinMaxScaler()
X = mm_scaler.fit_transform(X)
X1 = mm_scaler.transform(X1)

from sklearn.model_selection import train_test_split 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.10, random_state=0)

#from sklearn.linear_model import BayesianRidge
#regressor = BayesianRidge()
#reg = regressor.fit(X, y)
##fitResult = regressor.fit(Xtrain, Ytrain)
#YPred = regressor.predict(X1)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 250 , random_state= 0)
#from catboost import CatBoostRegressor
##categorical_features_indices = np.where(X.dtypes != np.fl5at)[0]
#CB = CatBoostRegressor(iterations=60000, depth=3, learning_rate=0.001, loss_function='MAE', 
#                              early_stopping_rounds = 300)

import lightgbm as lgb
X_0 = lgb.Dataset(Xtrain, label = Ytrain)
X_test1 = lgb.Dataset(Xtest, label = Ytest)

params = {}
params['learning_rate'] = 0.001
params['boosting_type'] = 'gbdt'
params['metric'] = 'mae'
params['verbosity'] = -1
params['bagging_seed'] = 11 
params['max_depth'] = 20

RF = regressor.fit(Xtrain, Ytrain)
#cat = CB.fit(Xtrain,Ytrain)
LGB1 = lgb.train(params, X_0, 100000, valid_sets = [X_0,X_test1], early_stopping_rounds=400 )


YPred_RF = RF.predict(Xtest)
YPred_lgb1 = LGB1.predict(Xtest)

YPred1_RF = RF.predict(X1)
YPred1_lgb1 = LGB1.predict(X1)


data1 = pd.read_csv('tcd-ml-1920-group-income-submission.csv')
data1['Total Yearly Income [EUR]'] = YPred1_lgb1
data1.to_csv('LGB.csv', index = False)


stacked_pred = np.column_stack((YPred_RF,YPred_lgb1))
stacked_test_pred = np.column_stack((YPred1_RF,YPred1_lgb1))

import xgboost
meta_model = xgboost.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 1,
                max_depth = 8, alpha = 0.5, n_estimators = 250)
meta_model.fit(stacked_pred,Ytest)
final_pred = meta_model.predict(stacked_test_pred)


final_pred = final_pred + temp

#print('Mean Absolute Error:', metrics.mean_absolute_error(Ytest, YPred1))
#print('Mean Squared Error:', metrics.mean_squared_error(Ytest, YPred1))
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Ytest, YPred1)))

data = pd.read_csv('tcd-ml-1920-group-income-submission.csv')
data['Total Yearly Income [EUR]'] = final_pred
data.to_csv('Both.csv', index = False)
