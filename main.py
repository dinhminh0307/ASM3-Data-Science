#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'online_shoppers_intention.csv'
df = pd.read_csv(file_path)


# In[26]:


df.head()


# In[27]:


df.info()


# In[28]:


bool_columns = ['Weekend', 'Revenue']
df[bool_columns] = df[bool_columns].astype(int)

# Verify the conversion
df.info()


# In[29]:


df.isnull().sum()


# In[30]:


df["Month"] = df["Month"].replace("June", "Jun") # 'Jun' is spelt as 'June' in raw data


#  Sort by Month

# In[31]:


months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

df["Month"] = pd.Categorical(df["Month"], categories=months, ordered=True)
df = df.sort_values("Month")


# ## Visualize data distribution

# In[32]:


numerical_features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                      'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


# ### Distribution of Visit by month

# In[33]:


plt.figure(figsize=(10, 6))
sns.countplot(x='Month', data=df, order=df['Month'].value_counts().index, palette='viridis')
plt.title('Distribution of Visits by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.show()


# ### Distribution of the Revenue

# In[34]:


plt.figure(figsize=(8, 4))
sns.countplot(x='Revenue', data=df, palette='Set2')
plt.title('Distribution of Revenue (Target)')
plt.xlabel('Revenue (False=No, True=Yes)')
plt.ylabel('Count')
plt.show()


# ## Distribution of Visitor Types

# In[35]:


plt.figure(figsize=(8, 4))
sns.countplot(x='VisitorType', data=df, palette='coolwarm')
plt.title('Distribution of Visitor Types')
plt.xlabel('Visitor Type')
plt.ylabel('Count')
plt.show()


# ## Feature Engineering

# In[36]:


# Select only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
corrMatrix = numeric_df.corr()

# Style the correlation matrix for visualization
corrMatrix.style.background_gradient(cmap='Blues')


# In[37]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import RFE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


# ## Encoding the categorical column

# In[38]:


# Perform one-hot encoding for categorical features: 'Month' and 'VisitorType'
df = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)

# Display the first few rows of the transformed dataset
df.head()


# In[39]:


X = df.drop(columns=['Revenue'])
Y = df['Revenue'].astype(int)


# ### Check the data imbalancing

# In[40]:


value_counts = Y.value_counts()
print(value_counts)


# ### Oversample the imbalance data
# 

# In[41]:


ros = RandomOverSampler()
X, Y = ros.fit_resample(X, Y) # take more from the less class to increase its size


# In[42]:


value_counts = Y.value_counts()
print(value_counts)


# ## Spliting the data

# In[43]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


# In[44]:


X = df.drop('Revenue', axis=1)    
y = df['Revenue']


# In[45]:


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[46]:


# Initializing the RandomForestClassifier
random_forest = RandomForestClassifier(random_state=42)


# In[47]:


# Setting up the GridSearch to find the best parameters
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [10, 20, 30, None],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required at each leaf node
}

grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and model
best_rf = grid_search.best_estimator_

# Making predictions with the best model
y_pred = best_rf.predict(X_test)

# Generating the classification report
report = classification_report(y_test, y_pred)
print(report)


# In[ ]:




