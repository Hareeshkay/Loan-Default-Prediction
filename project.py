# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:56:51 2020

@author: U6067583
"""
#We will start by importing the packages that will be used throughout the analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pickle

#Reading the dataset
project=pd.read_csv("C:\\Users\\u6067583\\Downloads\\New folder\\bank_final.csv")

#Let us now see how the values in the MIS_Status column are distributed. 
#We will plot an histogram of values against count of times the status appears on the dataframe
m =project['MIS_Status'].value_counts()
m = m.to_frame()
m.reset_index(inplace=True)
m.columns = ['MIS_Status','Count']
plt.subplots(figsize=(7,5))
sns.barplot(y='Count', x='MIS_Status', data=m)
plt.xlabel("Length")
plt.ylabel("Count")
plt.title("Distribution of Loan Status in our Dataset")
plt.show()

#converting the dataset to dataframe
df = pd.DataFrame(project) 

#Data Cleaning, #EDA
df.columns
df.shape
df.isnull().sum() 
df.isna().sum() 
# we need to remove columns with large number of null or na values which are not necessory
#removing ChgOffDate which has more null values and columns related to personal details are removed since they are not required for loan prediction
#'Name','City','State','Zip','Bank','BankState','CCSC','ApprovalDate','DisbursementDate' removed
df.drop(['Name','City','State','Zip','Bank','BankState',
         'CCSC','ApprovalDate','ChgOffDate','DisbursementDate'], inplace=True, axis=1)

#finding duplicate records and dropping them, around 16408 rows are duplicates
df.duplicated().sum()
df.shape
df.drop_duplicates(inplace=True) 
df.shape

#making FranchiseCodes are categorical like ISFRANCHISE=0 and 1 and dropping the main column FranchiseCode

row_indexes=df[df['FranchiseCode']>1].index
df.loc[row_indexes,'isfranchise']='1'
row_indexes=df[df['FranchiseCode']<=1].index
df.loc[row_indexes,'isfranchise']='0'
df.drop(['FranchiseCode'],axis=1,inplace=True)
df.isfranchise=df.isfranchise.astype(int)


#RevLineCr has some special charectors with minimum count and will drop those rows(6) replace other values with mode
df.RevLineCr.value_counts()
df.RevLineCr.value_counts().index[0]
df.index[df['RevLineCr'] == True].tolist()
print(df[df['RevLineCr']== ','].index.values)
print(df[df['RevLineCr']== '1'].index.values)
print(df[df['RevLineCr']== '`'].index.values)
df.reset_index(drop=True, inplace=True)
df.drop(df.index[[6056,19246,26574,123170,5571,79684]],axis=0,inplace=True)
df['RevLineCr'] = df['RevLineCr'].map({'Y':1,'N':0})
df.RevLineCr = df.RevLineCr.fillna(df.RevLineCr.value_counts().index[0])
df.RevLineCr=df.RevLineCr.astype(int)

#lowdoc 
df.LowDoc.value_counts()
print(df[df['LowDoc']== '1'].index.values)
df.drop(df.index[121293],axis=0,inplace=True)
df['LowDoc'] = df['LowDoc'].map({'Y':0,'N':1,'C':2})
df = df[df.LowDoc != 2]


#urban rural has values 0,1,2. replacing the 2 with mode
df['UrbanRural'].nunique()
df['UrbanRural'].value_counts()
UrbanRural_Mode = df['UrbanRural'].mode()
df['UrbanRural'] = df['UrbanRural'].replace(2,1)
df['UrbanRural'].unique()
df['UrbanRural'].value_counts()

# choosing columns which has currency values in it and removing special charectors 
Currency_Columns = ['DisbursementGross','BalanceGross','ChgOffPrinGr',
                    'GrAppv','SBA_Appv']
df[Currency_Columns] = df[Currency_Columns].replace('[\$,]',"",
                        regex=True).astype(float)

#MIS STATUS is the dependant here so we hare making it as PIF as 1 and CHGOFF as 0,filling na values with mode
df['MIS_Status'] = df['MIS_Status'].map({'P I F':1,'CHGOFF':0})
df.MIS_Status.value_counts()
df.MIS_Status.value_counts().index[0]
df.MIS_Status = df.MIS_Status.fillna(df.MIS_Status.value_counts().index[0])
df.dtypes
df.describe() 

#checking the correlation matrix and skipping independant variables highly correlated
correlation = df.corr(method='pearson')
df.info() 
df.drop(['LowDoc','DisbursementGross', 'BalanceGross', 'ChgOffPrinGr',
         'GrAppv','SBA_Appv','NoEmp','CreateJob','RetainedJob'], inplace = True, axis=1) 


#repositioning the MIS STATUS column
df = df[['ApprovalFY', 'Term','NewExist','UrbanRural','RevLineCr','isfranchise','MIS_Status']]

df.describe() 



#Now splitting the data using scikitlearn's train_test_split and using 60% data for training and 40% for testing.       
traindata, testdata = train_test_split(df, stratify=df['MIS_Status'],test_size=0.4, random_state=17)
testdata.reset_index(drop=True, inplace=True)
traindata.reset_index(drop=True, inplace=True)

#We'll now scale the data so that each column has a mean of zero and unit standard deviation
sc = StandardScaler()
Xte = testdata.drop('MIS_Status', axis=1)
yte = testdata['MIS_Status']
numerical = Xte.columns[(Xte.dtypes == 'float64') | (Xte.dtypes == 'int64')].tolist()
Xte[numerical] = sc.fit_transform(Xte[numerical])

#Now we will try to use a balanced dataset with equal amount of zeroes and 1's. The following part does the same.
y_default = traindata[traindata['MIS_Status'] == 0]
n_paid = traindata[traindata['MIS_Status'] == 1].sample(n=len(y_default), random_state=17) ##chosing equal amount of 1's

##creating a new dataframe for balanced set
data = y_default.append(n_paid) 
data.MIS_Status.value_counts()

##creating the independent and dependent array
Xbal = data.drop('MIS_Status', axis=1)
ybal = data['MIS_Status']

#scaling it again
numerical = Xbal.columns[(Xbal.dtypes == 'float64') | (Xbal.dtypes == 'int64')].tolist()
Xbal[numerical] = sc.fit_transform(Xbal[numerical])

models = {'RF': RandomForestClassifier(n_estimators=100),
          }

balset = {}
for i in models.keys():
    scores = cross_val_score(models[i], Xbal - np.min(Xbal) + 1,
                                    ybal, scoring='roc_auc', cv=3)
    balset[i] = scores
    print(i, scores, np.mean(scores))


#RF [0.95980535 0.96133742 0.9617388 ] 0.9609605224559034

#we are going to select Random Forst method and will try to find the optimal number of 
 #trees using the gridsearchcv and try to make the predition based on this and lets see if there is 
  #any improvements in predicting 0's
    
model = RandomForestClassifier(n_estimators=100)
model.fit(Xbal, ybal)
predict = model.predict(Xte)

predict = model.predict(Xte)
fig, axes = plt.subplots(figsize=(8,6))
cm = confusion_matrix(yte, predict).T
cm = cm.astype('float')/cm.sum(axis=0)
ax = sns.heatmap(cm, annot=True, cmap='Blues');
ax.set_xlabel('True Label')
ax.set_ylabel('Predicted Label')
ax.axis('equal')


#Let's find the optimum number of estimators for this model and use that for prediction. 
#This time we are going to use 5 fold cross validation.

params = {'n_estimators': [50, 100, 200, 400, 600, 800]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid=params,
                                   scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(Xbal, ybal)
print(grid_search.best_params_)
print(grid_search.best_score_)
#{'n_estimators': 600}
0.9207914438502673

grid_search.best_estimator_.fit(Xbal, ybal)

predict = model.predict(Xte)
fig, axes = plt.subplots(figsize=(15,9))
cm = confusion_matrix(yte, predict).T
cm = cm.astype('float')/cm.sum(axis=0)
ax = sns.heatmap(cm, annot=True, cmap='Blues');
ax.set_xlabel('True Label')
ax.set_ylabel('Predicted Label')
ax.axis('equal')

#Since random forest is based on decision trees, we can also plot the variable importance.
 # Variable importance tells us which variable had highest importance when predicting an 
   #outcome.
r = pd.DataFrame(columns=['Feature','Importance'])
ncomp = 15
r['Feature'] = feat_labels = Xbal.columns
r['Importance'] = model.feature_importances_
r.set_index(r['Feature'], inplace=True)
ax = r.sort_values('Importance', ascending=False)[:ncomp].plot.bar(width=0.9, legend=False, figsize=(15,8))
ax.set_ylabel('Relative Importance')

#deployment

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

# predict the output
y_pred = model.predict(Xte)



