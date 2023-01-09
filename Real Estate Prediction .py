#!/usr/bin/env python
# coding: utf-8

# ## Real Estate price predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


#for printing the first five rows of the data frame just for checking is it working properly or not:
housing.head()


# In[4]:


housing.info()


# In[5]:


housing["CHAS"]


# In[6]:


housing["CHAS"].value_counts()


# In[7]:


#about housing (data frame)
housing.describe()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


#for ploting graphs:
import matplotlib as plt
housing.hist(bins=50, figsize=(20, 50))


# In[ ]:





# ## Train-Test Splitting

# In[10]:


#eigther we can use this or we can use sklearn which we discuss further:
import numpy as np
def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[11]:


#train_set, test_set = split_train_test(housing,0.2)


# In[12]:


#print("Rows in:",train_set)


# In[13]:


#print("Rows in:",test_set)


# In[14]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing,test_size = 0.2, random_state=42)
print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[15]:


# here we're going to use stratified shuffel point which will devide the data into train and test data in equal porttion
#coz we don't want ki ek kisi pointer ka data training me to hai but test me hai hi nhi or we can say vice versa....
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[16]:


strat_test_set.describe()


# In[17]:


strat_test_set['CHAS'].value_counts()


# In[18]:


strat_train_set['CHAS'].value_counts()


# In[19]:


95/7  #excatly same ration of 0 and 1 which will describe that statrified shuffle point is working by dividing the data in same ratio


# In[20]:


housing = strat_train_set.copy()


# 365/28

# ## Correlations

# In[21]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[22]:


# in upper output 1.0000 vslue is said 'strong positive correlation'
# -0.74737373 is said to be stron negative corretion
#  value of RM is just stron correlation


# In[23]:


from pandas.plotting import scatter_matrix
attributes = ['MEDV','RM','ZN','LSTAT']
scatter_matrix(housing[attributes], figsize = (12,8))


# In[24]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.7)


# In[25]:


#  why we use correlations?
#in the above data set we can see that there are alot out-liers which we can 
# remove from our data for the better functioning of the model
# fro the removal of the out-liers we use this correlation olny...


# ## Attribute Combinations

# In[26]:


housing.head()


# In[27]:


housing['TAXRM'] = housing['TAX']/housing['RM']


# In[28]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[29]:


housing.plot(kind="scatter",x='TAXRM',y='MEDV',alpha=0.8)


# In[30]:


housing = strat_train_set.drop("MEDV",axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## Missing Attributes

# In[31]:


# to take care of the missing attiributes:
#  1> get rid of the missing data points
#  2> get rid of the whole attribute
#  3> set the values to some value(0,mean or median)


# In[32]:


a=housing.dropna(subset=['RM'])  # option one if we remove some data points from the data frame
a.shape
# note that the original housing data will remain unchanged


# In[33]:


housing.drop("RM",axis=1).shape  # this is the option 2 where we have deleted the whole column.


# In[34]:


# note that there is no RM column and also note that original data from of the data frame is remain unchanged


# In[35]:


median = housing["RM"].median()  # this is the option 3


# In[36]:


housing["RM"].fillna(median)
# note that the original housing dataframe will remain unchanged..


# In[37]:


housing.shape


# In[38]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[39]:


imputer.statistics_


# In[40]:


# if we use option three then jo hamne taining set se median nikala hai usee test set me bhi fit
# karna pade ga kyuki maybe possible test set me bhi value missing ho sktko hai\
# to is cheej ko aiutomate krne k liye sklearn me impute naam k function ka use kr k 
# ham ye kar skte hai..... as we did in above code..


# In[41]:


x = imputer.transform(housing)


# In[42]:


housing_tr = pd.DataFrame(x, columns=housing.columns)


# In[43]:


housing_tr.describe()


# In[44]:


imputer.statistics_.shape


# In[45]:


x = imputer.transform(housing)


# In[46]:


housing_tr = pd.DataFrame(x, columns=housing.columns)


# In[47]:


housing_tr.describe()


# ## Scikit-Learn Design
# 
# basically contain of three type of objects
# 
# 
# 1> Estimators -  it estimates some parameters based on the data set
#        for example. Imputer it has a fit method and transform method. Fit method --> fits the dataset and calculate internal parameters
# 
# 
# 2> Transformers - transform method function takes input anf returns output based upon the learning from the fit(). it also has a convineance function called fir_transform() which fits and then transforms
# 
# 
# 
# 3> pridictors - linear regression model is an example of the predictors
# fit() and predict are the two common functions. it also gives score() function which evaluatethe predictions.

# ## Feature Scalling
# Basically two types of feature scalling are there
# 
# 
# 1> Normalization(min-max scalling):
#      (value-min)/(max-min) is the formula and for doing this sklearn provides us the class..MinMaxscaller class
# 
# 
# 2> Standardization:
# 
# (value-mean)/(standard deviation)
# 
# fro doing this sklearn provides a class called standard scaller
# 

# ##  Making a Pipe-Line

# In[48]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler())
])


# In[49]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[50]:


housing_num_tr


# In[51]:


housing_num_tr.shape


#  ## Model Selection

# In[52]:


from sklearn.linear_model import LinearRegression 
model = LinearRegression()
model.fit(housing_num_tr, housing_labels)


# In[53]:


some_data = housing.iloc[:5]


# In[54]:


some_labels = housing_labels.iloc[:5]


# In[55]:


prepared_data = my_pipeline.transform(some_data)


# In[56]:


model.predict(prepared_data)


# In[57]:


list(some_labels)


# ## Evaluating Model

# In[58]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
lin_mse = np.sqrt(lin_mse) 


# In[59]:


lin_mse  #mean square error


# In[60]:


# second model:
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(housing_num_tr,housing_labels)


# In[61]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
lin_mse = np.sqrt(lin_mse) 


# In[62]:


lin_mse #mean squared error got decresed..  but model is not good coz overfitting has happend.


# # Using better evaluation technique which is cross validation

# In[63]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = "neg_mean_squared_error",cv =10)
rmse_scores = np.sqrt(-scores)


# In[64]:


rmse_scores


# In[ ]:





# In[65]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
lin_rse = mean_squared_error(housing_labels,housing_predictions)
lin_mse = np.sqrt(lin_mse) 


# ## Using cross validation with linear regression

# In[66]:


from sklearn.linear_model import LinearRegression 
model = LinearRegression()
model.fit(housing_num_tr, housing_labels)


# In[67]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = "neg_mean_squared_error",cv =10)
rmse_scores = np.sqrt(-scores)


# In[78]:


def print_scores(scores):
    print("Scores are:",scores)
    print("mean:",scores.mean())
    print("Standard deviation:",scores.std())


# In[80]:


rmse_scores


# In[81]:


print_scores(rmse_scores)


# ## choosing decision tree with cross validation

# In[69]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(housing_num_tr,housing_labels)


# In[70]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = "neg_mean_squared_error",cv =10)
rmse_scores = np.sqrt(-scores)


# In[71]:


rmse_scores


# In[72]:


def print_scores(scores):
    print("Scores are:",scores)
    print("mean:",scores.mean())
    print("Standard deviation:",scores.std())


# In[73]:


print_scores(rmse_scores)


# ## Now using random forest method:

# In[74]:


#from sklearn.liner_model import LinearRegression
#from sklearn.tree import DecisionTreeRegression
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[75]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = "neg_mean_squared_error",cv =10)
rmse_scores = np.sqrt(-scores)


# In[76]:


def print_scores(scores):
    print("Scores are:",scores)
    print("mean:",scores.mean())
    print("Standard deviation:",scores.std())


# In[77]:


print_scores(rmse_scores)


# ## CONCLUSION:-
# 
# 
# 1. Linear Regression :
#       
#        Mean : 2.83248545
#        Standard Deviation : 0.7054270117355301
#        
# 2. Decision Tree: 
# 
#         Mean : 4.2236105723567245
#         Standard Deviation :  0.9649492979999748
#         
# 3. Random Forest: 
#         
#         
#         Mean :3.305936186216151 
#         Standard Deviation : 0.7054270117355301
#         

# ## Result:
# 
# Hence we can say that Random Forest model works better as compare to the Linear Regression
# and Decision Tree Model.

# ## Saving The Model 

# In[84]:


from joblib import dump,load
dump(model, "Dragon.joblib")


# ## Testing Model

# In[94]:


X_test = strat_test_set.drop("MEDV",axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[93]:


final_rmse


# In[87]:


# we can see that the error get decresed..in random forest.


# In[95]:


X_test = strat_test_set.drop("MEDV",axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions , list(Y_test))


# In[ ]:




