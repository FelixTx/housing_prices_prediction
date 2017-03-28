import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# LOAD DATA

train=pd.read_csv('/home/felix/Desktop/housing_prices/train.csv')
test=pd.read_csv('/home/felix/Desktop/housing_prices/test.csv')
train['Type']='Train' #Create a flag for Train and Test Data set
test['Type']='Test'
fullData = pd.concat([train,test],axis=0) #Combined both Train and Test Data set

fullData.columns # This will show all the column names
fullData.head(10) # Show first 10 records of dataframe
fullData.describe() #You can look at summary of numerical fields by using describe() function