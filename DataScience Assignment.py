#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


#1.Import the data set in Python.
Mhousing=pd.read_csv(r"C:\Users\Shreyas s\Desktop\wipro ML\Melbourne_housing_FULL.csv")


# In[8]:


#2.View the dataset
Mhousing


# In[9]:


#3.See the structure and the summary of the dataset to understand the data.
Mhousing.info()


# In[10]:


Mhousing.shape


# In[11]:


Mhousing.columns


# In[12]:


Mhousing.describe()


# In[13]:


#4.	Find out the number of: a.) Numeric attributes   b.) Categorical attributes:
a=0;b=0
for col in Mhousing:
    if Mhousing[col].dtype!='object':
        a=a+1
    if Mhousing[col].dtype=='object':
        b=b+1
print ("Number of Numeric attributes:",a)
print ("Number of Categorical attributes:",b)
        


# In[14]:


#Duplicate values: Identify if the datasets have duplicate values or not and remove the duplicate values. 
#Find out the number of rows present in the dataset: a)Before removing duplicate values  b)After removing duplicate values

Mhousing_dup = Mhousing.drop_duplicates()
if len(Mhousing)!=len(Mhousing_dup):
    print ("Duplicates detected in the dataset")
    print ("Number of rows before removing duplicate values",len(Mhousing))
    print ("Number of rows after removing duplicate values",len(Mhousing_dup))
else:
    print ("Duplicates not detected")


# In[15]:


#Missing value treatment: Check which variables have missing values and use appropriate treatments. 
#For each of the variables, find the number of missing values and provide the value that they have been imputed with.

#SOLUTION: Finding variables with missing values & the number of missing values for each variable
for col in Mhousing:   
    if (Mhousing[col].isnull().values.any()):       
        print('\n',col,"has missing values")
        print("Number of missing values:",Mhousing[col].isnull().sum())


# In[16]:


#SOLUTION: Filling above missing values (mean value for numerics, 0 for Postcode and YearBuilt & 'Random' for string)
Mhousing.Price=Mhousing.Price.fillna(1.050173e+06)
Mhousing.Distance=Mhousing.Distance.fillna(11)
Mhousing.Postcode=Mhousing.Postcode.fillna(0)
Mhousing.Bedroom2=Mhousing.Bedroom2.fillna(3)
Mhousing.Bathroom=Mhousing.Bathroom.fillna(2)
Mhousing.Car=Mhousing.Car.fillna(2)
Mhousing.Landsize=Mhousing.Landsize.fillna(593)
Mhousing.BuildingArea=Mhousing.BuildingArea.fillna(160)
Mhousing.YearBuilt=Mhousing.YearBuilt.fillna(0)
Mhousing.CouncilArea=Mhousing.CouncilArea.fillna('Random')
Mhousing.Lattitude=Mhousing.Lattitude.fillna(0)
Mhousing.Longtitude=Mhousing.Longtitude.fillna(0)
Mhousing.Regionname=Mhousing.Regionname.fillna('Random')
Mhousing.Propertycount=Mhousing.Propertycount.fillna(0)

#DATA CHECK for anymore missing values
for col in Mhousing:   
    if (Mhousing[col].isnull().values.any()):       
        print('\n',col,"has missing values")
        print("Number of missing values:",Mhousing[col].isnull().sum())
    else:
        print ("No missing values for column", col)


# In[17]:


#Variable type: Check if all the variables have the correct variable type, based on the data dictionary. If not, then change them.
#For how many attributes did you need to change the data type?

#SOLUTION: Data type of each variable
Mhousing.dtypes


# In[18]:


#As per data dictionary, 
##  $ Suburb       : Factor 
##  $ Rooms        : int  
##  $ Type         : Factor
##  $ Price        : int  
##  $ Method       : Factor
##  $ SellerG      : Factor 
##  $ Date         : Factor 
##  $ Distance     : num  
##  $ Postcode     : Factor 
##  $ Bedroom2     : int  
##  $ Bathroom     : int  
##  $ Car          : int  
##  $ Landsize     : int  
##  $ YearBuilt    : int  
##  $ Regionname   : Factor 
##  $ Propertycount: Factor 
# where Factor in R is equivalent to Category in pandas


#SOLUTION: collecting all columns with dtype as object & converting to category (except Address & Council Area)
obj_columns = Mhousing.select_dtypes(['object']).columns
cnt=0
for col in obj_columns:
    if (col!='Address' and col!='CouncilArea' ):
        Mhousing[col] = Mhousing[col].astype('category')
        cnt=cnt+1

#SOLUTION: converting dtype of Postcode & Propertycount to category
Mhousing.Postcode = Mhousing.Postcode.astype('category')
cnt=cnt+1
Mhousing.Propertycount = Mhousing.Propertycount.astype('category')
cnt=cnt+1
Mhousing.dtypes


# In[19]:


#SOLUTION: collecting all columns with dtype as float64 & converting to int64 
fl_columns = Mhousing.select_dtypes(['float64']).columns
for col in fl_columns:
    Mhousing[col] = Mhousing[col].astype('int64')
    cnt=cnt+1

#Count 
print("Number of changes in datatype",cnt)
Mhousing.dtypes


# In[20]:


#Outlier Treatment: 
#Identify the varibales : Make a subset of the dataset with all the numeric variables. 
#Outliers : For each variable of this subset, carry out the outlier detection.
#           Find out the percentile distribution of each variable and carry out capping and flooring for outlier values.  

#SOLUTION :a subset of the dataset with data tyep as int64. 
Mhousing_num=Mhousing.select_dtypes(['int64'])
Mhousing_num


# In[21]:


#SOLUTION:sample percentile distribution
print(Mhousing_num.quantile([.01,.02,.03,.04,.05,.1,.2,.4,.5,.95,.96,.99] ))

#SOLUTION:sample BoxPlot for variable "Rooms"
import seaborn as sns
sns.boxplot(x=Mhousing_num['Rooms'])


# In[22]:


sns.boxplot(x=Mhousing_num['Distance'])


# In[23]:


sns.boxplot(x=Mhousing_num['Price'])


# In[24]:


sns.boxplot(x=Mhousing_num['Bedroom2'])


# In[25]:


sns.boxplot(x=Mhousing_num['Bathroom'])


# In[26]:


sns.boxplot(x=Mhousing_num['Car'])


# In[27]:


sns.boxplot(x=Mhousing_num['Landsize'])


# In[28]:


sns.boxplot(x=Mhousing_num['BuildingArea'])


# In[29]:


#SOLUTION: Outlier Detection & Capping and Flooring of Outlier Values using IQR method.
 

for col in Mhousing_num:
    #As Lattitude & Longitude will not affect the Price, not considering them for Outlier treatment
    if col!='Lattitude' and col!='Longtitude':
        print("\n\nOutlier Treatment for ",col)
       
        #Outlier Detection using IQR method where Cap=IQ3+1.5*IQR & Floor=1.5*IQR-IQ1
        Q1 = Mhousing_num[col].quantile(0.25)
        Q3 = Mhousing_num[col].quantile(0.75)
        IQR = Q3 - Q1
        print("IQR for ",col,"=",IQR)
        limit=1.5*IQR
        floor = limit - Q1
        print("Floor for ",col,"=",floor)
        Cap = Q3 + limit
        print("Cap for ",col,"=",Cap)
        
        #Removal of Outlier Values. i.e values less than (1.5*IQR-Q1) and more than (1.5*IQR+Q3) are removed.
        #Large number of Outliers are detected for Price. So Removing Outlier values for Price.
        if col=='Price':
            Mhousing_num=Mhousing_num[(Mhousing_num[col]>=floor)&(Mhousing_num[col]<=Cap)]
            print(Mhousing_num.shape)


# In[30]:


#Identify variables that have non-linear trends. How many variables have non-linear trends? Transform them (as required)
plt.style.use('seaborn-whitegrid')
x=Mhousing_num['Rooms']
y=Mhousing_num['Price']
plt.plot(x,y,'o',color='black')


# In[31]:


x=Mhousing_num['Bedroom2']
plt.plot(x,y,'o',color='black')


# In[32]:


x=Mhousing_num['Bathroom']
plt.plot(x,y,'o',color='black')


# In[33]:


x=Mhousing_num['Car']
plt.plot(x,y,'o',color='black')


# In[34]:


x=Mhousing_num['Landsize']
plt.plot(x,y,'o',color='black')


# In[35]:


x=Mhousing_num['BuildingArea']
plt.plot(x,y,'o',color='black')


# In[36]:


Mhousing_num.corr()


# In[37]:


#Based on the above Scatter Plots & correlation matrix, it can be concluded that 
#    Rooms, Bedroom2, Bathroom & Car have weak positive correlation with Price.
#    Distance have weak weak negative correlation with Price
#    Landsize & Building Area do not have much correlation with Price. i.e, it is having non-linear trend & need to be transformed

Mhousing_P=Mhousing_num.loc[:,['Price','Landsize','BuildingArea']]
Mhousing_P['logLS']=np.log(Mhousing_P['Landsize'])
Mhousing_P['LS^2']=Mhousing_P['Landsize']**2
Mhousing_P['LS^3']=Mhousing_P['Landsize']**3
Mhousing_P['LSsqrt']=np.sqrt(Mhousing_P['Landsize'])

Mhousing_P['logBA']=np.log(Mhousing_P['BuildingArea'])
Mhousing_P['BA^2']=Mhousing_P['BuildingArea']**2
Mhousing_P['BA^3']=Mhousing_P['BuildingArea']**3
Mhousing_P['BAsqrt']=np.sqrt(Mhousing_P['BuildingArea'])
Mhousing_P.corr()


# In[39]:


#Standardization:Name the variables to be standardised before using a distance-based algorithm


#SOLUTION: All numeric variables should be standardised before applying distance based algorithm such as KNN method

         #using z-score method
int_columns=['Rooms','Price','Distance','Bedroom2','Bathroom','Car','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude']

from scipy.stats import zscore
Mhousing_num_z=Mhousing_num[int_columns].apply(zscore)
print(Mhousing_num_z)


# In[42]:


#Dummy encoding :Identify the number of dummy variables to be created for the variable steel.

#SOLUTION: There is no variable called as 'steel' in the data set. Hence selecting variable 'Suburb' to work on.

print("Dummy variables to be created for variable Suburb :")
Mhousing['Suburb'].value_counts()

#dummies=pd.get_dummies(Mhousing['Suburb'])
#dummies.head()

