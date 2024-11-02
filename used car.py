import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# loading the car details csv file

df=pd.read_csv('C:/Users/Welcome/Desktop/projects.py/Car details v3.csv')

# cleaning process

print(df.shape)
print(df.head(15))
print(df.name.nunique())
print(df.isnull().sum())

#plotting selling price with km driven

sns.displot(df['selling_price'],kind='kde')
plt.show()

print(df['selling_price'].describe())
info=df.loc[(df['selling_price'] < 20_00_000) & (df['km_driven'] < 5_00_000)]
sns.regplot(data=info,x='km_driven',y='selling_price')
plt.show()

table=pd.DataFrame()
table['Mileage']=df.loc[(df['selling_price']<10_00_000)].mileage.str.split(expand=True)[0].astype('float64')
table['sp']=df.loc[(df['selling_price']<10_00_000)].selling_price
sns.regplot(data=table,x='Mileage',y='sp')
plt.show()
sns.catplot(data=df,x='owner',y='selling_price')
plt.figure(figsize=(10,10))
plt.show()
df=df[~(df.owner=='Test Drive Car')]
sns.catplot(data=df,x='seats',y='selling_price')
plt.figure(figsize=(10,10))
plt.show()
sns.catplot(data=df,x='year',y='selling_price')
plt.figure(figsize=(10,10))
plt.show()
sns.catplot(data=df,x='transmission',y='selling_price')
plt.figure(figsize=(10,10))
plt.show()
plt.figure(figsize=(7,7))
sns.boxplot(data=df,x='fuel',y='selling_price')
plt.show()

#  splitting Brand name from brand
df['Brand']=df.name.str.split(expand=True)[0]
df.drop(columns=['name','torque'],inplace=True)
brand=pd.DataFrame(df.groupby(['Brand']).year.count()).sort_values(ascending=False,by='year')
print(brand)

# removing last ten records of brands

rem=list(brand.index[-10:])
print(rem)
df=df[~df.Brand.isin(rem)]
plt.figure(figsize=(20,10))
sns.countplot(data=df,x='brand')
plt.show()

#changing  the datatype of columns
df.loc[:,'engine']=df['engine'].astype('str').str.split(expand=True)[0]
df.loc[:,'mileage']=df['mileage'].astype('str').str.split(expand=True)[0]
df.loc[:,'max_power']=df['max_power'].astype('str').str.split(expand=True)[0]

df['engine']=df['engine'].astype('float64')
df=df.loc[~df['max_power'].isin(['bhp'])]
df['mileage']=df['mileage'].astype('float64')
df['max_power']=df['max_power'].astype('float64')

# heatmap

corrmatrix=df.corr()
sns.heatmap(corrmatrix,square=True,annot=True)

print(df.info())

sns.set()
col=['selling_price','year','transmission','engine','max_power','owner','fuel']
sns.pairplot(df[col], size=3)
plt.show()

print(df.info())

df.dropna(inplace=True)

ownerohc=pd.get_dummies(df['owner'])

fuelohc=pd.get_dummies(df['fuel'])
df['transmission']=df.transmission.map({'Manual':1,'Automatic':0})

df.drop(columns=['fuel','owner'],inplace=True)
df=pd.concat([df,ownerohc,fuelohc],axis=1)

print(df)

cols=['year',
 'selling_price',
 'km_driven',
 'mileage',
 'transmission',
 
 'engine',
 'max_power',
 
 'Brand',
 'First Owner',
 'Fourth & Above Owner',
 'Second Owner',
 
 'Third Owner',
 'CNG',
 'Diesel',
 'LPG',
 'Petrol']

dat=df[cols]

brandohc=pd.get_dummies(dat['Brand'])
dat.drop(columns=['Brand'],inplace=True)
dat=pd.concat([dat,brandohc],axis=1)

#modeling

X=dat.drop(columns=['selling_price'])
y=dat['selling_price']

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

# creating Linear Regressin Model

model1=LinearRegression()
model1.fit(X_train,y_train)
preds=model1.predict(X_test)
mean_absolute_error(y_test,preds)

y.mean()

# creating Decision Tree Regression Model

from sklearn.tree import DecisionTreeRegressor
model2=DecisionTreeRegressor(random_state=0)
model2.fit(X_train,y_train)
preds2=model2.predict(X_test)
mean_absolute_error(y_test,preds2)

model2


def mae(max_leaf, X_train, X_test, y_train, y_test):
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf, random_state=0)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    meanabse=mean_absolute_error(y_test,pred)
    return(meanabse)

for m in [50,100,250,500,750,800,825,850,875,900,1000,2000,5000,7500,10000]:
    error=mae(m,X_train, X_test, y_train, y_test)
    print('Max leaf nodes = {}, MAE = {}'.format(m,error))

    # XGBoost Regressor Models

from xgboost import XGBRegressor
model3=XGBRegressor()
model3.fit(X_train,y_train)
preds3=model3.predict(X_test)
mean_absolute_error(y_test,preds3)

from xgboost import XGBRegressor
model3=XGBRegressor(early_stopping_rounds=5,n_estimators=500)
model3.fit(X_train,y_train)
preds3=model3.predict(X_test)
mean_absolute_error(y_test,preds3)

sns.scatterplot(x=preds3,y=y_test-preds3)

from sklearn.metrics import r2_score
r2_score(y_test,preds3)