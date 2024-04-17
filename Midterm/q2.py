
## Q2-a Code

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')
df.head(10)


# In[2]:


df['Sex_Code'] = df['Sex'].map({'female':1, 'male':0}).astype('int') ##將資料文字轉換成數字
df['Sex'] = df['Sex_Code']
df['Age'] = df['Age'].fillna(df['Age'].min())
df


# In[3]:


X = df[["Pclass", "Sex", "Age" , "SibSp" , "Parch"]]
y = df["Survived"]


# In[4]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X,y)
Jack = clf.predict([[3, 0, 20 ,0,0]])
Rose = clf.predict([[1, 1, 17, 1, 1]])
print ("Logistic Regression: ")
print("Jack", Jack)
print("Rose", Rose)


# In[5]:


# K-NN
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier()
clf.fit(X,y)
Jack = clf.predict([[3, 0, 20 ,0,0]])
Rose = clf.predict([[1, 1, 17, 1, 1]])
print("K-NN: ")
print("Jack", Jack)
print("Rose", Rose)

# In[6]:


# SVC
from sklearn import svm
clf = svm.SVC()
clf.fit(X,y)
Jack = clf.predict([[3, 0, 20 ,0,0]])
Rose = clf.predict([[1, 1, 17, 1, 1]])
print("SVC: ")
print("Jack", Jack)
print("Rose", Rose)


# In[7]:


# Gaussian Naive bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X,y)
Jack = clf.predict([[3, 0, 20 ,0,0]])
Rose = clf.predict([[1, 1, 17, 1, 1]])
print("Gaussian Naive bayes: ")
print("Jack", Jack)
print("Rose", Rose)


# In[8]:


# Multinomail Naive bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X,y)
Jack = clf.predict([[3, 0, 20 ,0,0]])
Rose = clf.predict([[1, 1, 17, 1, 1]])
print("Multinomail Naive bayes: ")
print("Jack", Jack)
print("Rose", Rose)


# In[9]:


# Decision Tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X,y)
Jack = clf.predict([[3, 0, 20 ,0,0]])
Rose = clf.predict([[1, 1, 17, 1, 1]])
print("Decision Tree: ")
print("Jack", Jack)
print("Rose", Rose)

# In[10]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X,y)
Jack = clf.predict([[3, 0, 20 ,0,0]])
Rose = clf.predict([[1, 1, 17, 1, 1]])
print("Random Forest: ")
print("Jack", Jack)
print("Rose", Rose)


# In[11]:


# XGBoost
from xgboost.sklearn import XGBClassifier
clf = XGBClassifier()
clf.fit(X,y)
two = {"Pclass":[3,1], "Sex":[0,1], "Age":[23.0,20.0] ,"SibSp":[0,1] ,"Parch":[0,1]}
df1 = pd.DataFrame(two)
print("XGBoost: ")
Jack = clf.predict(df1.iloc[0:1])
print("Jack", Jack)
Rose = clf.predict(df1.iloc[1:2])
print("Rose", Rose)


print("Practice")

"""

Q2-a Summary:
According to the eight models, the results are as follows:
Logistic Regression: Jack [0] Rose [1]
K-NN: Jack [0] Rose [1]
SVC: Jack [0] Rose [0]
Gaussian Naive bayes: Jack [0] Rose [1]
Multinomail Naive bayes: Jack [0] Rose [1]
Decision Tree: Jack [0] Rose [1]
Random Forest: Jack [0] Rose [1]
XGBoost: Jack [0] Rose [1]

0 means dead, 1 means survived.

"""

"""
Q2-b:
The result from SVC is different from others.

"""

"""
Q2-c:
The reasons might be as follow:
1. There are missing data in the data and can cause the SVC process differently.
2. SVC is sensitve to the scale of features since it is based on the distance between the data points.
3. SVC needs proper tuning of hyperparameters to get better results. Since we just simply import the model
    and fit the data, the result might not be the best.
4. The data might not be linearly separable and SVC is not good at handling non-linear data.

"""


