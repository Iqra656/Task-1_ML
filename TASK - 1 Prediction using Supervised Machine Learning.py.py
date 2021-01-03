#!/usr/bin/env python
# coding: utf-8

# # DATA SCIENCE AND BUSINESS ANALYTICS AT SPARKS FOUNDATION

# ## TASK 1- PREDICTION USING SUPERVISED MACHINE LEARNING

# ## BY : IQRA MALIK
# 

# ## Problem Statement : Predict the percentage of a student based on the no. of study hours
# ## Tools used : Pandas, Numpy, Matplotlib,Seaborn, Sklearn,Jupyter Notebook

# In[52]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[53]:


#Reading data
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data= pd.read_csv(url)
print("Data imported Successfully")
data.head(10)


# In[54]:


data.shape


# In[55]:


data.info()
data.describe()


# In[65]:


#Plot the data
data.plot(x="Hours",y="Scores",style="o")
plt.title("Hours Vs Percentage")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Scored")
plt.grid()
plt.show()


# In[66]:


#Dividing the data into dependent and independent variables
#Data Preparation
x= data.iloc[:,:-1].values
y= data.iloc[:,1].values


# In[67]:


#Splitting the data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[68]:


#Training the Algorithm
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[69]:


#Visualizing the model
line = model.coef_*x + model.intercept_
#Plotting the training data
plt.scatter(x_train, y_train, color= "red")
plt.plot(x,line);
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Scored")
plt.grid()
plt.show()


# In[70]:


#Plotting the testing data
#Plotting the testing data
plt.scatter(x_test, y_test, color= "green")
plt.plot(x,line);
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Scored")
plt.grid()
plt.show()


# In[71]:


#Making Predictions
y_predicted = model.predict(x_test)


# In[72]:


#Comparing actual Vs Predicted
df = pd.DataFrame({'Actual Score': y_test, 'Predicted Score': y_predicted})
df


# In[73]:


#Now Testing what will be the predicted score if the student studies 9.25hrs/day
hours = 9.25
our_pred= model.predict([[hours]])
print("Predicted Score if the student studies {}hrs/day is {}".format(hours,our_pred[0]))


# In[74]:


#Evaluating the model
#Now atlast we will calculate the mean absolute error 
from sklearn import metrics
print("Mean absolute error of our model is:",metrics.mean_absolute_error(y_test,y_predicted))


# # Thank you
