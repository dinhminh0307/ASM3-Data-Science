#!/usr/bin/env python
# coding: utf-8

# # Assessment 3 - Online Shoppers Purchasing Intention
# #### Objective: 

# In[1]:


# data manipulation
import pandas as pd
import numpy as np


# In[2]:


# data visualization
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


# machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef)


# In[4]:


# dimensionality reduction
from sklearn.decomposition import PCA


# In[5]:


# clustering
from sklearn.cluster import KMeans, DBSCAN


# In[6]:


# data preprocessing
from sklearn.preprocessing import MinMaxScaler


# In[7]:


# handling imbalanced data
from imblearn.over_sampling import SMOTE


# In[8]:


# displaying outputs in jupyter
from IPython.display import display


# In[9]:


# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# utility functions

# In[10]:


def plotbox_and_hist(df, columns, figsize=(30, 80)):
    fig, axes = plt.subplots(len(columns), 2, figsize=figsize)

    for i, column in enumerate(columns):
        sns.boxplot(x=df[column], ax=axes[i, 0])
        sns.histplot(x=df[column], ax=axes[i, 1])

    plt.tight_layout()
    plt.show()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


def countplot_value(df, columns, figsize=(20, 30)):
    # Compute the number of rows and columns for subplots
    num_columns = 2
    num_rows = (len(columns) + 1) // num_columns  # Ensures proper grid size for odd numbers
    
    fig, axes = plt.subplots(num_rows, num_columns, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, column in enumerate(columns):
        ax = axes[i]
        sns.countplot(x=df[column], ax=ax)
        
        # Annotate bar heights
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), textcoords='offset points')
        ax.set_title(f'{column} Counts')

    # Remove any unused subplot axes
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ## 1. Retrieving and Preparing the Data

# ### 1.1. Data Loading

# In[13]:


# load the dataset
file_path = 'online_shoppers_intention.csv'
df = pd.read_csv(file_path)


# ### 1.2. Dataset Observation

# In[14]:


# display the first 5 rows of the dataset
display(df.head())


# In[15]:


# summary of the dataset
df.info()


# In[16]:


df.describe()


# ##### Analysis
# The dataset contains 12,330 rows and 18 columns. There are no missing values across any of the columns, as all 18 attributes have 12,330 non-null entries.
# 
# The data types are primarily integers and floats, with a few categorical variables (object types) like **Month** and **VisitorType**, as well as Boolean variables like **Weekend** and **Revenue**. The attributes can be categorized as follows:
# 
# - Categorical values: **Month**, **VisitorType**.
# - Numerical values:
#   - Discrete: **Administrative**, **Informational**, **ProductRelated**, **OperatingSystems**, **Browser**, **Region**, **TrafficType**.
#   - Continuous: **Administrative_Duration**, **Informational_Duration**, **ProductRelated_Duration**, **BounceRates**, **ExitRates**, **PageValues**, **SpecialDay**.
#   
# The **Revenue** column is a binary variable indicating whether a transaction resulted in revenue. The **Weekend** column is also Boolean, representing whether the visit occurred on the weekend.
# Further analysis can be conducted to explore relationships between these variables, especially the conversion-related columns like **Revenue**.

# ### 1.3. Detailed Analysis and Cleaning

# In[17]:


# create a copy of the dataframe to store the cleaned data
df_clean = df.copy()


# In[18]:


# display the unique values of the 'Month' and 'VisitorType' columns
print(df_clean['Month'].unique())
print(df_clean['VisitorType'].unique())


# In[19]:


# fix typos in the 'Month' column
df_clean['Month'] = df_clean['Month'].replace({'June': 'Jun'})
# convert the 'Month' column to numerical values
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
df_clean['Month'] = df_clean['Month'].map(month_map)
df_clean = df_clean.sort_values('Month')
# verify the changes
print(df_clean['Month'].unique())


# In[20]:


# convert the 'Month' and 'Revenue' columns to numerical
bool_columns = ['Weekend', 'Revenue']
df_clean[bool_columns] = df_clean[bool_columns].astype(int)


# In[21]:


# verify the conversion
df_clean.info()


# #### 1.3.1. Univariate Analysis of Numerical values

# In[22]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_clean['VisitorType'] = encoder.fit_transform(df_clean['VisitorType'])
print(df_clean['VisitorType'])


# In[23]:


columns = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
           "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
           "SpecialDay"]


# In[24]:


plotbox_and_hist(df_clean, columns)


# ##### Analysis
# - Histograms: Most features are right-skewed, with values concentrated on the lower end and long tails extending to higher values. Features like **Administrative** and **Administrative_Duration** show that most sessions involve minimal administrative activities, with outliers at higher values. **Informational** and **Informational_Duration** follow a similar pattern, suggesting that informational activities are not common in most sessions. **ProductRelated** and **ProductRelated_Duration** have a broader spread compared to administrative and informational features. This indicates more varied user engagement with product-related pages, although there are still some significant outliers. **BounceRates** and **ExitRates** are mostly near zero, showing that users usually stay on the site instead of leaving immediately. **PageValues** is skewed with most values being zero, while a few sessions have high values, showing only a small number of sessions generate significant revenue. **SpecialDay** has specific peaks that correspond to certain predefined special days in the data.
# 
# - Boxplots: Many features have visible outliers, represented by points beyond the whiskers of the boxplot. **Administrative_Duration** and **Informational_Duration** show tight interquartile ranges (IQRs) with outliers at higher values, confirming that most sessions have minimal durations for these activities. **ProductRelated_Duration** has a wider IQR, reflecting more variability in user interactions with product pages, but also includes a few extreme outliers. **BounceRates** and **ExitRates** have very narrow IQRs, with most values near zero, but occasional outliers suggest some sessions have high rates. **PageValues** has a compact IQR but includes extreme outliers, reflecting rare but impactful sessions with high page values. **SpecialDay** does not show significant outliers since its values are predefined and specific.
# 
# Further analysis will help us decide whether to clean datapoints, apply data imputation, or drop the problematic columns.

# #### 1.3.2. Univariate Analysis of Categorical values

# In[25]:


columns = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend", "Revenue"]


# In[26]:


countplot_value(df_clean, columns)


# ##### Analysis
# There are heavy imbalances across all categorical variables, specifically:
# - **Month**: March and November have the highest counts, while other months have significantly lower counts.
# - **OperatingSystems**: Operating systems 1, 2, and 3 dominate the data, while other systems have lower counts.
# - **Browser**: Browsers 1 and 2 have high counts, while other browsers show much lower counts.
# - **Region**: Regions 1 and 2 represent most of the data, with other regions showing significantly lower counts.
# - **TrafficType**: Traffic types 1 and 2 have higher counts, while other types are underrepresented.
# - **VisitorType**: Returning visitors are the most common, while new visitors and the 'other' category are less frequent.
# - **Weekend**: Non-weekend visits have the highest count, while weekend visits are much less frequent.
# - **Revenue**: Non-revenue generating visits dominate, while revenue-generating visits are much fewer.
# 
# Overall, imbalances could adversely affect the accuracy of predictions or analyses. Data imputation techniques might be considered to address this issue.

# #### 1.3.3. Multivariate Analysis

# In[27]:


sns.pairplot(df_clean, hue="Revenue")


# #### 1.3.4. Cleaning Data

# In[28]:


# remove outliers with low frequencies
df_temp = df_clean.copy()
df_clean = df_clean[((df_clean['Administrative'] < 25) & 
                     (df_clean['Administrative_Duration'] < 2000) & 
                     (df_clean['Informational'] < 15) & 
                     (df_clean['Informational_Duration'] < 1750) & 
                     (df_clean['ProductRelated'] < 475) & 
                     (df_clean['ProductRelated_Duration'] < 30000) & 
                     (df_clean['BounceRates'] < 0.175) & 
                     (df_clean['ExitRates'] < 0.19) &
                     (df_clean['PageValues'] < 250))]


# In[29]:


# verify removal
rows_remove = len(df_temp) - len(df_clean)
print(f"The numbers of rows removed: {rows_remove}")


# ## 2. Feature Engineering

# ### 2.1. Preprocessing Features and Plotting Correlations

# In[30]:


# since we do not have enough context to extract meaning from the values of categorical variables such as 'OperatingSystems', 'Browser', 'Region', and 'TrafficType', 
# we will drop these columns.
df_features = df.copy()
df_features = df_clean.drop(columns=["OperatingSystems", "Browser", "Region", "TrafficType"])


# In[31]:


# apply one-hot encoding to 'VisitorType' and 'Month' and concatenate with the original DataFrame
df_features = pd.concat([df_features, pd.get_dummies(df_features["Month"], prefix='Month_')], axis=1)
df_features = pd.concat([df_features, pd.get_dummies(df_features["VisitorType"], prefix='VisitorType_')], axis=1)


# In[32]:


# drop the original 'VisitorType' column
df_features.drop("Month", axis=1, inplace=True)
df_features.drop("VisitorType", axis=1, inplace=True)


# In[33]:


# correlation analysis
target_df = df_features['Revenue']


# In[34]:


all_corr = df_features.corr(method = 'pearson')


# In[35]:


mask = np.zeros_like(all_corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True


# In[36]:


ax = plt.figure(figsize=(15, 10))
ax = sns.heatmap(all_corr, cmap='Blues', annot=True, fmt='.2f', mask = mask)
ax = plt.xticks(rotation=85)
ax = plt.title("Correlations between features")


# In[ ]:





# ## 3. Data Modelling

# ### 3.1. Regression

# ### 3.2. Clustering

# ### 3.3. Classification

# ### 2.2. Analysis and Hypothesis Proposal
# #### 2.2.1. Regression problem
#  
# #### 2.2.2. Clustering problem
#  
# #### 2.2.3. Classification problem
# ##### 1. Visitor Engagement
# - **Observation**: Scatterplots of `Administrative_Duration`, `Informational_Duration`, and `ProductRelated_Duration` against `Revenue` often show higher densities of sessions with greater durations linked to purchases (`Revenue = True`).
#   - `PageValues` seems to have a strong positive correlation with `Revenue`. Higher page values indicate a higher likelihood of purchases, aligning with the hypothesis.
# 
# - **Hypothesis**: Higher engagement leads to higher purchase probabilities.
#   - Higher values for `Administrative_Duration`, `Informational_Duration`, and `ProductRelated_Duration` might indicate greater visitor interest, resulting in purchases.
#   - A higher `PageValues` score is likely to correlate positively with purchases.
# 
# ##### 2. Bounce and Exit Rates
# - **Observation**: From scatterplots of `BounceRates` and `ExitRates` against `Revenue`, sessions with higher bounce rates (`BounceRates`) and exit rates (`ExitRates`) generally correspond to no purchases (`Revenue = False`), which supports the hypothesis of poor user experience reducing purchase likelihood.
# 
# - **Hypothesis**: Poor user experience decreases purchase likelihood.
#   - Higher `BounceRates` and `ExitRates` might indicate user dissatisfaction, leading to fewer purchases.
# 
# ##### 3. Time Factors
# - **Observation**: Visualizations of `SpecialDay` against `Revenue` show an increase in purchases as the proximity to a special day increases. Similarly, analysis of `Month` against `Revenue` highlights seasonal trends, with months like November showing higher purchase probabilities, likely due to shopping holidays like Black Friday.
# 
# - **Hypothesis**: Shopping behavior depends on timing.
#   - Visits closer to a `SpecialDay` (e.g., Black Friday or holidays) might have a higher likelihood of purchases.
#   - Certain months (`Month`) might reflect seasonal shopping trends, influencing purchase behavior.
# 
# ##### 4. User Types
# - **Observation**: From categorical plots of `VisitorType` and `Revenue`, returning visitors have a noticeably higher likelihood of generating revenue compared to new visitors. This observation supports the hypothesis that returning visitors are more likely to purchase.
# 
# - **Hypothesis**: Returning visitors are more likely to purchase.
#   - Returning visitors (`VisitorType = Returning_Visitor`) might have a higher likelihood of making a purchase compared to new visitors (`VisitorType = New_Visitor`).
# 
# ##### 5. Technical Features
# - **Observation**: Bar charts and heatmaps for `OperatingSystems`, `Browser`, and `TrafficType` reveal varying purchase probabilities across categories. For instance, some browsers or traffic sources have a stronger association with purchases, supporting the hypothesis that technical accessibility impacts purchases.
# 
# - **Hypothesis**: Technical accessibility impacts purchases.
#   - Different `OperatingSystems`, `Browser`, and `TrafficType` values might influence the likelihood of purchases based on usability or accessibility.
# 
# ##### 6. Weekend Influence
# - **Observation**: A categorical plot of `Weekend` against `Revenue` shows slight differences in purchase likelihood between weekend and weekday sessions. This observation suggests some behavioral differences in shopping patterns based on the day of the week.
# - **Hypothesis**: Shopping behavior differs on weekends.
#   - Visits during weekends (`Weekend = True`) might have different purchase rates compared to weekday visits.
# 
# ---

# In[37]:


X = df_clean.drop(columns=['Revenue'])
Y = df_clean['Revenue'].astype(int)


# ### Check the data imbalancing

# In[38]:


value_counts = Y.value_counts()
print(value_counts)


# ### Oversample the imbalance data
# 

# In[39]:


from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier


# In[40]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
ros = RandomOverSampler()
X_KNN, Y_KNN = ros.fit_resample(X, Y) # take more from the less class to increase its size


# In[41]:


value_counts = Y_KNN.value_counts()
print(value_counts)


# ## Spliting the data

# In[42]:


from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


# In[97]:


# Assuming X_KNN and Y_KNN are already defined
x_train_origin, x_test_origin, y_train_origin, y_test_origin = train_test_split(X_KNN, Y_KNN, test_size=0.3, random_state=0)


# In[98]:


# Permutation Importance for KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_origin, y_train_origin)


# In[99]:


# Compute permutation importance
perm_importance = permutation_importance(knn, x_test_origin, y_test_origin, n_repeats=10, random_state=42)


# In[100]:


# Get feature importance scores
perm_importance_df = pd.DataFrame({
    'Feature': X_KNN.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)


# In[101]:


# Save or print the results
print(perm_importance_df)  # To print the importance scores
perm_importance_df.to_csv('permutation_importance.csv', index=False)  # Save as CSV file


# # Select features to feed

# In[137]:


selected_features = [
    "PageValues",
    "ProductRelated_Duration",
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated"
]


# In[138]:


# Creating a new dataset with the selected features
X_KNN_selected = X_KNN[selected_features]


# In[187]:


# Assuming X_KNN and Y_KNN are already defined
X = df_clean[selected_features]   
y = df_clean['Revenue'].astype(int)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[188]:


print(y_train.value_counts(), y_test.value_counts())


# In[189]:


# Apply RandomOverSampler to balance the classes
# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)


# In[190]:


print(y_train.value_counts(), y_test.value_counts())


# In[191]:


# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[192]:


param_dist = {
    'n_neighbors': randint(3, 15),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}


# In[193]:


# Initialize the KNN classifier
knn = KNeighborsClassifier()


# In[194]:


# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=knn, param_distributions=param_dist, n_iter=20, cv=5, scoring='accuracy', verbose=1, n_jobs=-1, random_state=42)
random_search.fit(x_train, y_train)


# In[195]:


# Get the best parameters and best estimator
best_params = random_search.best_params_
best_knn = random_search.best_estimator_


# In[196]:


# Evaluate the best model
y_pred = best_knn.predict(x_test)


# In[197]:


cm = confusion_matrix(y_test, y_pred)
y_pred_knn = y_pred
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# In[199]:


# Print the best parameters and evaluation metrics
print("Best Parameters:", best_params)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))


# ## Random Forest

# In[200]:


X = df_clean[selected_features]   
y = df_clean['Revenue'].astype(int)


# In[186]:


# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
display(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[168]:


# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


# In[62]:


# Before oversampling
unique, counts = np.unique(y_train, return_counts = True)
print(np.asarray((unique, counts)).T)

# After oversampling
unique, counts = np.unique(y_train_smote, return_counts = True)
print(np.asarray((unique, counts)).T)


# In[122]:


# Setting up the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


# In[125]:


# Create a RandomForestClassifier with GridSearchCV
clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc', verbose=2)
clf.fit(X_train_smote, y_train_smote)


# In[126]:


# Display the best parameters
print("Best Parameters Found:\n", clf.best_params_)


# In[127]:


# Predict probabilities
y_pred_rf = clf.predict(X_test)
y_scores = clf.predict_proba(X_test)[:, 1]


# In[169]:


# Classification Report
report = classification_report(y_test, y_pred_rf)
print("Classification Report:\n", report)


# In[129]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
print('\nConfusion Matrix:\n', cm)


# In[130]:


from sklearn.metrics import roc_curve, auc


# In[131]:


group_names = ['True Neg','False Neg','False Pos','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

ax.set_title('Classification Model 1 Confusion Matrix\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[132]:


# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)


# In[133]:


# Plotting the ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[201]:


def calculate_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    p = precision_score(y_test, y_pred, pos_label=0)
    r = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred, pos_label=0)
    mmc = matthews_corrcoef(y_test, y_pred)
    return acc, p, r, f1, mmc

# Calculate metrics for each model
metrics_rf = calculate_metrics(y_test, y_pred_rf)
metrics_knn = calculate_metrics(y_test, y_pred_knn)

# Prepare for plotting
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MMC']
models = ['Random Forest', 'KNN']
metrics_values = np.array([metrics_rf, metrics_knn])

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.15
bar_positions = np.arange(len(models))

for i, metric_name in enumerate(metrics_names):
    ax.bar(bar_positions + i * bar_width,
           metrics_values[:, i], width=bar_width, label=metric_name)

for i, model in enumerate(models):
    for j, metric_value in enumerate(metrics_values[i]):
        ax.text(i + j * bar_width, metric_value + 0.01,
                f'{metric_value:.2f}', ha='center', va='bottom')

ax.set_xticks(bar_positions + (len(metrics_names) - 1) * bar_width / 2)
ax.set_xticklabels(models)
ax.set_ylabel('Metrics Value')
ax.set_title('Models Overall Comparison')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(metrics_names))

plt.show()


# ## XGBOOST

# In[172]:


file_path = 'online_shoppers_intention.csv'
bach_df = df_clean.copy()


# In[173]:


bach_df.info()


# In[ ]:





# In[174]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score


# In[ ]:





# In[175]:


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.columns_to_drop_ = []

    def fit(self, X, y):
        numeric_columns = X.select_dtypes(include='number').columns
        df_with_target = pd.concat([X, y.rename('target')], axis=1)
        corr_matrix = df_with_target.corr()

        self.columns_to_drop_ = [
            col for col in numeric_columns
            if abs(corr_matrix['target'][col]) < self.threshold
        ]
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')


# In[ ]:





# In[176]:


from sklearn.preprocessing import RobustScaler


# In[177]:


class MyRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = RobustScaler()
        self.numeric_features = None

    def fit(self, X, y=None):
        self.numeric_features = X.select_dtypes(include='number').columns.tolist()
        self.scaler.fit(X[self.numeric_features])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])
        return X


# In[178]:


class MyLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        # Identify object (categorical) columns
        self.categorical_columns = X.select_dtypes(include=['object']).columns
        # Fit LabelEncoders for each categorical column
        for col in self.categorical_columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self

    def transform(self, X):
        X = X.copy()
        # Transform categorical columns using the fitted encoders
        for col, encoder in self.encoders.items():
            X[col] = encoder.transform(X[col])
        return X


# In[179]:


from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# In[180]:


# Define pipeline
pipeline = Pipeline([
    ('correlation_filter', CorrelationFilter(threshold=0.1)),
    ('label_encode', MyLabelEncoder()),
    ('robust_scaling', MyRobustScaler()),
    ('xgboost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])


# In[181]:


X = bach_df.drop(columns=['Revenue'])
Y = bach_df['Revenue'].astype(int)


# In[182]:


value_counts = Y.value_counts()
print(value_counts)


# In[183]:


ros = RandomOverSampler()
X, Y = ros.fit_resample(X, Y)


# In[184]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[ ]:





# In[185]:


param_grid = {
    'xgboost__n_estimators': [50, 100, 150],
    'xgboost__max_depth': [3, 5, 7],
    'xgboost__learning_rate': [0.01, 0.1, 0.2]
}


# In[186]:


# GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(x_train, y_train)


# In[187]:


# Get the best parameters and evaluate the model
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)


# In[ ]:


print(f"Best Parameters: {grid_search.best_params_}")
print(f"Test Accuracy: {accuracy:.4f}")

