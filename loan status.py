# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import classification_report as cr
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns

path="D:\python\project\Python_Module_Day_15.2_Credit_Risk_Train_data.csv"
cread_train=pd.read_csv(path) 


cread_train.columns
cread_train.head(6)
cread_train.tail(6)
cread_train.dtypes
cread_train.shape

#convert fact to num for y variable
cread_train['Loan_Status1']=0
cread_train['Loan_Status1'][cread_train.Loan_Status=="Y"]=1
cread_train.describe
cread_train=cread_train.drop(['Loan_Status'],axis=1) 


#check for y distribution (equivalent to r table fuction)
cread_train.Loan_Status1.value_counts()
 
#drop loanid
cread_train=cread_train.drop(['Loan_ID'],axis=1)

#split the data types 
cread_train.dtypes
num_cols=cread_train.select_dtypes(include=['int32','int64','float32','float64']).columns
print(num_cols)

fact_cols=cread_train.select_dtypes(include=['object']).columns
print(fact_cols)

###EDA for numeric data
cread_train[num_cols].isnull().sum() 
##nulls in LoanAmount ,Loan_Amount_Term  ,Credit_History    

#update Loan_Amount_Term wid mean
cread_train['Loan_Amount_Term'].value_counts() 
mean_loan_amt_term=cread_train.Loan_Amount_Term[cread_train.Loan_Amount_Term>0].mean()
cread_train['Loan_Amount_Term']=cread_train.Loan_Amount_Term.replace(np.NaN,mean_loan_amt_term)

#update LoanAmount wid mean
cread_train['LoanAmount'].value_counts()
mean_loanamount=cread_train.LoanAmount[cread_train.LoanAmount>0].mean()
cread_train['LoanAmount']=cread_train.LoanAmount.replace(np.NaN,mean_loanamount)

#update creadit_history with 0
cread_train['Credit_History']=cread_train.Credit_History.replace(np.NaN,0.0) 

'''
cread_train['Credit_History'].value_counts() 
cread_train['Loan_Amount_Term'].index[cread_train['Loan_Amount_Term'].apply(np.isnan)]
cread_train['LoanAmount'].isnull().sum()
cread_train['Credit_History'].isnull().sum()
#replace na wid value 
cread_train['Loan_Amount_Term']=cread_train.Loan_Amount_Term.replace(np.NaN,123)
'''

#check for 0
cread_train[num_cols][cread_train[num_cols]==0].count()
###Credit_History ,Loan_Status1 ,CoapplicantIncome 
#update coapplicantinclome wid mean
m_coapinc=cread_train.CoapplicantIncome[cread_train.CoapplicantIncome>0].mean()
cread_train.CoapplicantIncome[cread_train.CoapplicantIncome==0]=m_coapinc


#corelation
num_cols
cor=cread_train[num_cols].iloc[:,0:5].corr() 
cor=np.tril(cor,k=1)
sns.heatmap(cor,
            xticklabels=num_cols,
            yticklabels=num_cols,
            annot=True,vmin=0,vmax=1,square=True)

###EDA for factor data
#null check
cread_train[fact_cols].isnull().sum() 

cread_train['Gender'].value_counts() 
cread_train['Gender']=cread_train.Gender.replace(np.NaN,'Male') 

cread_train['Self_Employed']=cread_train.Self_Employed.replace(np.NaN,'not specified') 

cread_train['Dependents']=cread_train.Dependents.replace(np.NaN,'0') 


cread_train['Married'].value_counts()
cread_train['Married']=cread_train.Married.replace(np.NaN,'Yes')

#zero check
cread_train[fact_cols][cread_train[fact_cols]==0].count() 


cread_train['Gender'].unique() 
cread_train['Married'].unique()
cread_train['Dependents'].unique()
cread_train['Education'].unique()
cread_train['Self_Employed'].unique()
cread_train['Property_Area'].unique()


#EDA
#distribution of class
sns.countplot(x="Loan_Status1",data=cread_train,palette="hls")

# avg valuesof features for 0/1
means=pd.DataFrame(cread_train.groupby('Loan_Status1').mean())
means.columns
means[['ApplicantIncome']]

#cross-tab
pd.crosstab(cread_train.Gender,cread_train.Loan_Status1).plot(kind="bar")

pd.crosstab(cread_train.Married,cread_train.Loan_Status1).plot(kind="bar")



#dummy variable creation into the dataset

newdf=cread_train.copy(deep=True)
for e in fact_cols:
    dummy=pd.get_dummies(newdf[e],drop_first=True,prefix=e)
    newdf=newdf.join(dummy)

print(newdf.columns)
fact_cols
newdfcol=newdf.columns 
newdfcol=list(set(newdfcol).difference(set(fact_cols)))
newdfcol

#rearange
newdf=newdf[newdfcol]


newdf=pd.concat([newdf.drop(['Loan_Status1'],axis=1),newdf['Loan_Status1']],axis=1)
newdf
newdf.columns 
newdf.dtypes
newdf.shape

totalcols=len(newdf.columns)

trainx=newdf.iloc[:,0:totalcols-1]
trainy=newdf.iloc[:,totalcols-1]
'''
testx=test.iloc[:,0:totalcols-1]
testy=test.iloc[:,totalcols-1]
'''
print("trains={},trainy={}".format(trainx.shape,trainy.shape))


#build the logistic reg model
m1=sm.Logit(trainy,trainx).fit() 
#summary2 is name of fuction 4 summary
m1.summary2()




#prediction
path="D:\python\project\Python_Module_Day_15.4_Credit_Risk_Validate_data.csv"
cread_test=pd.read_csv(path)
cread_test.columns
cread_test.head(6)
cread_test.tail(6)
cread_test.dtypes
cread_test.shape

cread_test['Loan_Status1']=0 
cread_test['Loan_Status1'][cread_test.outcome=="Y"]=1
cread_test.describe
cread_test=cread_test.drop(['outcome'],axis=1)


#check for y distribution (equivalent to r table fuction)
cread_test.Loan_Status1.value_counts()

#drop loanid
cread_test=cread_test.drop(['Loan_ID'],axis=1)

#split the data types
num_cols=cread_test.select_dtypes(include=['int32','int64','float32','float64']).columns
print(num_cols)

fact_cols=cread_test.select_dtypes(include=['object']).columns
print(fact_cols)

cread_test[num_cols].isnull().sum()

#update Loan_Amount_Term wid mean
cread_test['Loan_Amount_Term'].value_counts()
mean_loan_amt_term=cread_test.Loan_Amount_Term[cread_test.Loan_Amount_Term>0].mean()
cread_test['Loan_Amount_Term']=cread_test.Loan_Amount_Term.replace(np.NaN,mean_loan_amt_term)

#update LoanAmount wid mean
cread_test['LoanAmount'].value_counts()
mean_loanamount=cread_test.LoanAmount[cread_test.LoanAmount>0].mean()
cread_test['LoanAmount']=cread_test.LoanAmount.replace(np.NaN,mean_loanamount)

#update creadit_history with 0
cread_test['Credit_History']=cread_test.Credit_History.replace(np.NaN,0.0)

#check for 0
cread_test[num_cols][cread_test[num_cols]==0].count()
###Credit_History ,Loan_Status1 ,CoapplicantIncome 
#update coapplicantinclome wid mean
m_coapinc=cread_test.CoapplicantIncome[cread_test.CoapplicantIncome>0].mean()
cread_test.CoapplicantIncome[cread_test.CoapplicantIncome==0]=m_coapinc


###EDA for factor data
#null check
cread_test[fact_cols].isnull().sum()

cread_test['Gender'].value_counts()
cread_test['Gender']=cread_test.Gender.replace(np.NaN,'Male')

cread_test['Self_Employed']=cread_test.Self_Employed.replace(np.NaN,'not specified')

cread_test['Dependents']=cread_test.Dependents.replace(np.NaN,'0')


cread_test['Married'].value_counts()
cread_test['Married']=cread_test.Married.replace(np.NaN,'Yes')

#zero check
cread_test[fact_cols][cread_test[fact_cols]==0].count()

##
newdf2=cread_test.copy(deep=True)
for e in fact_cols:
    dummy=pd.get_dummies(newdf2[e],drop_first=True,prefix=e)
    newdf2=newdf2.join(dummy)

print(newdf2.columns)
fact_cols
newdfcol2=newdf2.columns
newdfcol2=list(set(newdfcol2).difference(set(fact_cols)))
newdfcol2

#rearange
newdf2=newdf2[newdfcol2]


newdf2=pd.concat([newdf2.drop(['Loan_Status1'],axis=1),newdf2['Loan_Status1']],axis=1)
newdf2

newdf2.columns
newdf2.dtypes
newdf2.shape

totalcols2=len(newdf2.columns)

testx=newdf2.iloc[:,0:totalcols2-1]
testy=newdf2.iloc[:,totalcols2-1]
'''
testx=test.iloc[:,0:totalcols-1]
testy=test.iloc[:,totalcols-1]
'''
print("testx={},testy={}".format(testx.shape,testy.shape))

p1=m1.predict(testx)
p1

c1=len(p1[p1<=0.5])
c1
c2=len(p1[p1>0.5])
c2
print("<=0.5  {} , >0.5   {}  ".format(c1,c2))


predy=p1.copy(deep=True)

predy[predy<=0.5]=0
predy[predy>0.5]=1
predy.value_counts()

#confusion matrix
ConfusionMatrix(testy,predy)
print(cr(testy,predy))


#roc
from sklearn import metrics
fpr,tpr,threshold= metrics.roc_curve(testy,predy)
#auc
roc_auc=metrics.auc(fpr,tpr)
#plot
plt.title('Receiver Operating Characterstics')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% roc_auc)
plt.legend(loc='Lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Feature selection
from sklearn.feature_selection import f_classif as fs
features=trainx.columns
f_score,pval=fs(trainx,trainy)
df1=pd.DataFrame({'feature': features,'f_score':f_score,'pval':pval})
print(df1)
