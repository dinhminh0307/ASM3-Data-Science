# # Assessment 3 - Online Shoppers Purchasing Intention
# #### Objective: 

# data manipulation
import pandas as pd
import numpy as np

# data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef)

# dimensionality reduction
from sklearn.decomposition import PCA

# clustering
from sklearn.cluster import KMeans, DBSCAN

# data preprocessing
from sklearn.preprocessing import MinMaxScaler

# handling imbalanced data
from imblearn.over_sampling import SMOTE

# displaying outputs in jupyter
from IPython.display import display

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# utility functions

def plotbox_and_hist(df, columns, figsize=(30, 80)):
    fig, axes = plt.subplots(len(columns), 2, figsize=figsize)

    for i, column in enumerate(columns):
        sns.boxplot(x=df[column], ax=axes[i, 0])
        sns.histplot(x=df[column], ax=axes[i, 1])

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

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

# load the dataset
file_path = 'online_shoppers_intention.csv'
df = pd.read_csv(file_path)


# ### 1.2. Dataset Observation

# display the first 5 rows of the dataset
display(df.head())

# summary of the dataset
df.info()

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

# create a copy of the dataframe to store the cleaned data
df_clean = df.copy()

# display the unique values of the 'Month' and 'VisitorType' columns
print(df_clean['Month'].unique())
print(df_clean['VisitorType'].unique())

# fix typos in the 'Month' column
df_clean['Month'] = df_clean['Month'].replace({'June': 'Jun'})
# convert the 'Month' column to numerical values
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
df_clean['Month'] = df_clean['Month'].map(month_map)
df_clean = df_clean.sort_values('Month')
# verify the changes
print(df_clean['Month'].unique())

# convert the 'Month' and 'Revenue' columns to numerical
bool_columns = ['Weekend', 'Revenue']
df_clean[bool_columns] = df_clean[bool_columns].astype(int)

# verify the conversion
df_clean.info()


# #### 1.3.1. Univariate Analysis of Numerical values

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_clean['VisitorType'] = encoder.fit_transform(df_clean['VisitorType'])
print(df_clean['VisitorType'])

columns = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
           "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
           "SpecialDay"]

plotbox_and_hist(df_clean, columns)


# ##### Analysis
# - Histograms: Most features are right-skewed, with values concentrated on the lower end and long tails extending to higher values. Features like **Administrative** and **Administrative_Duration** show that most sessions involve minimal administrative activities, with outliers at higher values. **Informational** and **Informational_Duration** follow a similar pattern, suggesting that informational activities are not common in most sessions. **ProductRelated** and **ProductRelated_Duration** have a broader spread compared to administrative and informational features. This indicates more varied user engagement with product-related pages, although there are still some significant outliers. **BounceRates** and **ExitRates** are mostly near zero, showing that users usually stay on the site instead of leaving immediately. **PageValues** is skewed with most values being zero, while a few sessions have high values, showing only a small number of sessions generate significant revenue. **SpecialDay** has specific peaks that correspond to certain predefined special days in the data.
# 
# - Boxplots: Many features have visible outliers, represented by points beyond the whiskers of the boxplot. **Administrative_Duration** and **Informational_Duration** show tight interquartile ranges (IQRs) with outliers at higher values, confirming that most sessions have minimal durations for these activities. **ProductRelated_Duration** has a wider IQR, reflecting more variability in user interactions with product pages, but also includes a few extreme outliers. **BounceRates** and **ExitRates** have very narrow IQRs, with most values near zero, but occasional outliers suggest some sessions have high rates. **PageValues** has a compact IQR but includes extreme outliers, reflecting rare but impactful sessions with high page values. **SpecialDay** does not show significant outliers since its values are predefined and specific.
# 
# Further analysis will help us decide whether to clean datapoints, apply data imputation, or drop the problematic columns.

# #### 1.3.2. Univariate Analysis of Categorical values

columns = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend", "Revenue"]

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

sns.pairplot(df_clean, hue="Revenue")


# #### 1.3.4. Cleaning Data

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

# verify removal
rows_remove = len(df_temp) - len(df_clean)
print(f"The numbers of rows removed: {rows_remove}")


# ## 2. Feature Engineering

# ### 2.1. Preprocessing Features and Plotting Correlations

# since we do not have enough context to extract meaning from the values of categorical variables such as 'OperatingSystems', 'Browser', 'Region', and 'TrafficType', 
# we will drop these columns.
df_features = df.copy()
df_features = df_clean.drop(columns=["OperatingSystems", "Browser", "Region", "TrafficType"])

# apply one-hot encoding to 'VisitorType' and 'Month' and concatenate with the original DataFrame
df_features = pd.concat([df_features, pd.get_dummies(df_features["Month"], prefix='Month_')], axis=1)
df_features = pd.concat([df_features, pd.get_dummies(df_features["VisitorType"], prefix='VisitorType_')], axis=1)


# drop the original 'VisitorType' column
df_features.drop("Month", axis=1, inplace=True)
df_features.drop("VisitorType", axis=1, inplace=True)

# correlation analysis
target_df = df_features['Revenue']

all_corr = df_features.corr(method = 'pearson')

mask = np.zeros_like(all_corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

ax = plt.figure(figsize=(15, 10))
ax = sns.heatmap(all_corr, cmap='Blues', annot=True, fmt='.2f', mask = mask)
ax = plt.xticks(rotation=85)
ax = plt.title("Correlations between features")



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

X = df_clean.drop(columns=['Revenue'])
Y = df_clean['Revenue'].astype(int)


# ### Check the data imbalancing

value_counts = Y.value_counts()
print(value_counts)


# ### Oversample the imbalance data
# 

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
ros = RandomOverSampler()
X_KNN, Y_KNN = ros.fit_resample(X, Y) # take more from the less class to increase its size

value_counts = Y_KNN.value_counts()
print(value_counts)


# ## Spliting the data

from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Assuming X_KNN and Y_KNN are already defined
x_train_origin, x_test_origin, y_train_origin, y_test_origin = train_test_split(X_KNN, Y_KNN, test_size=0.2, random_state=0)

# Permutation Importance for KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_origin, y_train_origin)

# Compute permutation importance
perm_importance = permutation_importance(knn, x_test_origin, y_test_origin, n_repeats=10, random_state=42)

# Get feature importance scores
perm_importance_df = pd.DataFrame({
    'Feature': X_KNN.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

# Save or print the results
print(perm_importance_df)  # To print the importance scores
perm_importance_df.to_csv('permutation_importance.csv', index=False)  # Save as CSV file

## Select features to feed

selected_features = [
    "PageValues",
    "ProductRelated_Duration",
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated"
]

# Creating a new dataset with the selected features
X_KNN_selected = X_KNN[selected_features]

# Assuming X_KNN and Y_KNN are already defined
x_train, x_test, y_train, y_test = train_test_split(X_KNN_selected, Y_KNN, test_size=0.2, random_state=0)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

param_dist = {
    'n_neighbors': randint(3, 15),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Initialize the KNN classifier
knn = KNeighborsClassifier()

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=knn, param_distributions=param_dist, n_iter=20, cv=5, scoring='accuracy', verbose=1, n_jobs=-1, random_state=42)
random_search.fit(x_train, y_train)

# Get the best parameters and best estimator
best_params = random_search.best_params_
best_knn = random_search.best_estimator_

# Evaluate the best model
y_pred = best_knn.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Print the best parameters and evaluation metrics
print("Best Parameters:", best_params)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

X = df_clean[selected_features]   
y = df_clean['Revenue'].astype(int)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Setting up the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create a RandomForestClassifier with GridSearchCV
clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc', verbose=2)
clf.fit(X_train_smote, y_train_smote)

# Display the best parameters
print("Best Parameters Found:\n", clf.best_params_)

# Predict probabilities
y_pred = clf.predict(X_test)
y_scores = clf.predict_proba(X_test)[:, 1]

# Classification Report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

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


### XGBOOST



file_path = 'online_shoppers_intention.csv'
bach_df = df_clean.copy()



bach_df.info()



import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score



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



from sklearn.preprocessing import RobustScaler


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



from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Define pipeline
pipeline = Pipeline([
    ('correlation_filter', CorrelationFilter(threshold=0.1)),
    ('label_encode', MyLabelEncoder()),
    ('robust_scaling', MyRobustScaler()),
    ('xgboost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])



X = bach_df.drop(columns=['Revenue'])
Y = bach_df['Revenue'].astype(int)



value_counts = Y.value_counts()
print(value_counts)



ros = RandomOverSampler()
X, Y = ros.fit_resample(X, Y)



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)



param_grid = {
    'xgboost__n_estimators': [50, 100, 150],
    'xgboost__max_depth': [3, 5, 7],
    'xgboost__learning_rate': [0.01, 0.1, 0.2]
}

# GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(x_train, y_train)

# Get the best parameters and evaluate the model
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Test Accuracy: {accuracy:.4f}")
