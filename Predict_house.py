
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
dataset = pd.read_csv('22_workshop_data.csv')
#aaa = dataset.iloc[:,26:27]


#remove column if  50% data is null
for col in dataset.columns:
    null_sum = dataset[col].isnull().sum()
    if((null_sum/1460*100)>50):
      dataset = dataset.drop(col, axis=1)
#remove column if frequency gt 99%      
for col in dataset.columns:
    tmp = dataset[col].value_counts()
    total = dataset[col].count()
    for i in tmp:
        if((i/total*100)>99):
          dataset = dataset.drop(col, axis=1)          
# remove row if 50% data  is null
for index, row in dataset.iterrows():
    null_sum = row.isnull().sum()
    if((null_sum/80*100)>50):
         dataset = dataset.drop(index, axis=0)

#Takimg of missing data
#replace with mean
repalce_mean =  dataset[['LotFrontage','GarageYrBlt']]
simple_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
simple_imputer = simple_imputer.fit(repalce_mean)
repalce_mean = simple_imputer.transform(repalce_mean)
dataset[['LotFrontage','GarageYrBlt']] = repalce_mean

#replace NA to 0
replace_0 = dataset[['MasVnrArea']]
simple_imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant',fill_value = 0)
simple_imputer = simple_imputer.fit(replace_0)
replace_0 = simple_imputer.transform(replace_0)
dataset[['MasVnrArea']] = replace_0

#replace NA to Other
replace_Other = dataset[['MasVnrType','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','TotalBsmtSF','Electrical','BsmtFullBath','BsmtHalfBath','GarageFinish','GarageCars','GarageQual','GarageCond','EnclosedPorch','FireplaceQu','GarageType']]
simple_imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant',fill_value = 'Other')
simple_imputer = simple_imputer.fit(replace_Other)
replace_Other = simple_imputer.transform(replace_Other)
dataset[['MasVnrType','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','TotalBsmtSF','Electrical','BsmtFullBath','BsmtHalfBath','GarageFinish','GarageCars','GarageQual','GarageCond','EnclosedPorch','FireplaceQu','GarageType']] = replace_Other

#set data
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1].values
#y = np.log(y)

#X = X[['OverallQual','GrLivArea','TotalBsmtSF','GarageArea','1stFlrSF','YearBuilt','GarageCars','ExterQual','TotRmsAbvGrd','BsmtFinSF1']]

X_dummies = pd.get_dummies(X,drop_first=True)

X = X_dummies.values

# Feature Scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape((-1, 1))).ravel()

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)

#feature_importances
feature_importances = pd.DataFrame(regressor.feature_importances_,index = X_dummies.columns,columns=['importance']).sort_values('importance',ascending=False)

#Applying k-fold cross validation
accuracies = cross_val_score(estimator = regressor,X = X_train,y = y_train,cv = 10, n_jobs = -1)
print(accuracies.mean(),accuracies.std())


#y_pred = regressor.predict(X_test)
##
#fr_pred = pd.DataFrame(y_pred)
#fr_test = pd.DataFrame(y_test)
#
param_grid = {
    'n_estimators': [155,255],
    'criterion':['mse','mae'],
    'max_features': ['sqrt','log2','auto'],
    'max_depth' : [10,11,12]
}

from sklearn.model_selection import GridSearchCV
CV_rfc = GridSearchCV(estimator=regressor, param_grid=param_grid, cv= 10, n_jobs = -1)
CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_
#
regressor.score(X_test, y_test)


#
