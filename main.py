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
                             f1_score, matthews_corrcoef, silhouette_score, davies_bouldin_score)

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

from sklearn.mixture import GaussianMixture

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from joblib import Parallel, delayed


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

# load the dataset
file_path = 'online_shoppers_intention.csv'
df = pd.read_csv(file_path)

# display the first 5 rows of the dataset
display(df.head())

# summary of the dataset
df.info()

df.describe()

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

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_clean['VisitorType'] = encoder.fit_transform(df_clean['VisitorType'])
print(df_clean['VisitorType'])

columns = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
           "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
           "SpecialDay"]

plotbox_and_hist(df_clean, columns)

columns = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend", "Revenue"]

countplot_value(df_clean, columns)

sns.pairplot(df_clean, hue="Revenue")

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

# copy the cleaned dataframe
df_cluster = df_clean.copy()

# define model parameters
model_target = 'Revenue'
model_features = ['Administrative', 'Administrative_Duration', 'Informational', 
                 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

# prepare the data
df_cluster = df_cluster[model_features]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(df_cluster[model_features])
df_cluster[model_features] = scaled_features

# determine optimal number of pca components using explained variance ratio
pca_full = PCA()
pca_full.fit(df_cluster[model_features])
explained_variance_ratio = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(explained_variance_ratio >= 0.95) + 1  # keep 95% of variance

# apply pca with optimal components
pca = PCA(n_components=n_components)
pca_features = pca.fit_transform(df_cluster[model_features])

def fit_gmm(n_components, covariance_type, init_params, max_iter, pca_features):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        init_params=init_params,
        max_iter=max_iter,
        random_state=42
    )
    gmm.fit(pca_features)
    bic = gmm.bic(pca_features)
    return bic, gmm

best_bic = float('inf')
best_gmm_bic = None
best_n_components = None
best_covariance_type = None
best_init_params = None
best_max_iter = None

# define the grid of hyperparameters to search
param_grid = [(n_components, covariance_type, init_params, max_iter) 
              for n_components in range(2, 21)
              for covariance_type in ['full', 'tied', 'diag', 'spherical']
              for init_params in ['kmeans', 'random']
              for max_iter in [100, 200, 300]]

# use parallel processing for hyperparameter search
results = Parallel(n_jobs=-1)(
    delayed(fit_gmm)(n_components, covariance_type, init_params, max_iter, pca_features)
    for (n_components, covariance_type, init_params, max_iter) in param_grid
)

# extract best gmm based on bic
for bic, gmm in results:
    if bic < best_bic:
        best_bic = bic
        best_gmm_bic = gmm

# predict with the best gmm model
gmm_clusters = best_gmm_bic.predict(pca_features)

# visualize the clusters
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=gmm_clusters, cmap='viridis')  # Use gmm_clusters
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clusters Visualization with Best GMM Model')
plt.show()

# print the best model parameters
print(f'Best BIC: {best_bic}')
print(f'Best number of components: {best_n_components}')
print(f'Best covariance type: {best_covariance_type}')
print(f'Best init params: {best_init_params}')
print(f'Best max iterations: {best_max_iter}')

# add cluster labels to the DataFrame
df_cluster['Cluster'] = gmm_clusters

# analyze cluster characteristics
cluster_summary = df_cluster.groupby('Cluster')[model_features].mean()
print(cluster_summary)

# add the target variable back to the DataFrame for analysis
df_cluster[model_target] = df_clean[model_target]

# link clusters to hypothesis by checking purchase rates
purchase_rate = df_cluster.groupby('Cluster')[model_target].mean()
print(purchase_rate)

from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

def fit_dbscan(eps, min_samples, pca_features):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(pca_features)

    # Skip if all points are noise
    if len(np.unique(clusters)) <= 1:
        return -1, None  # Return -1 for invalid score

    try:
        score = silhouette_score(pca_features, clusters)
        return score, dbscan
    except:
        return -1, None  # Return -1 in case of error

best_silhouette = -1
best_eps = None
best_min_samples = None
best_dbscan = None

# Define parameter ranges
eps_range = np.arange(0.1, 1.1, 0.1)
min_samples_range = range(5, 21, 5)

# Create the parameter grid
param_grid = [(eps, min_samples) for eps in eps_range for min_samples in min_samples_range]

# Use parallel processing for the grid search
results = Parallel(n_jobs=-1)(
    delayed(fit_dbscan)(eps, min_samples, pca_features) for eps, min_samples in param_grid
)

# Process results to find the best model based on silhouette score
for score, dbscan in results:
    if score > best_silhouette:
        best_silhouette = score
        best_dbscan = dbscan
        best_eps = dbscan.eps
        best_min_samples = dbscan.min_samples

# Train final model with best parameters
final_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_clusters = final_dbscan.fit_predict(pca_features)

# visualize results
plt.figure(figsize=(10, 6))
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('DBSCAN Clustering Results')
plt.colorbar(label='Cluster')
plt.show()

# print best parameters for DBSCAN model
print(f'Best eps: {best_eps}')
print(f'Best min_samples: {best_min_samples}')
print(f'Best silhouette score: {best_silhouette}')
print(f'Number of clusters found: {len(np.unique(dbscan_clusters[dbscan_clusters != -1]))}')  # Use dbscan_clusters
print(f'Number of noise points: {len(dbscan_clusters[dbscan_clusters == -1])}')  # Use dbscan_clusters

# add cluster labels to the dataframe
df_cluster['Cluster'] = dbscan_clusters

# analyze cluster characteristics
cluster_summary = df_cluster.groupby('Cluster')[model_features].mean()
print(cluster_summary)

# add the target variable back to the dataframe for analysis
df_cluster[model_target] = df_clean[model_target]

# link clusters to hypothesis by checking purchase rates
purchase_rate = df_cluster.groupby('Cluster')[model_target].mean()
print(purchase_rate)

# remove noise from DBSCAN results for evaluation
filtered_dbscan_clusters = dbscan_clusters[dbscan_clusters != -1]
filtered_dbscan_features = pca_features[dbscan_clusters != -1]

# DBSCAN Metrics
if len(np.unique(filtered_dbscan_clusters)) > 1:
    dbscan_silhouette = silhouette_score(filtered_dbscan_features, filtered_dbscan_clusters)
    dbscan_dbi = davies_bouldin_score(filtered_dbscan_features, filtered_dbscan_clusters)
else:
    dbscan_silhouette = None
    dbscan_dbi = None
dbscan_noise_percentage = np.sum(dbscan_clusters == -1) / len(dbscan_clusters)

# GMM Metrics
if len(np.unique(gmm_clusters)) > 1:
    gmm_silhouette = silhouette_score(pca_features, gmm_clusters)
    gmm_dbi = davies_bouldin_score(pca_features, gmm_clusters)
else:
    gmm_silhouette = None
    gmm_dbi = None

# add target variable to compare purity
df_cluster[model_target] = df_clean[model_target]

# purity function
def purity_score(true_labels, predicted_clusters):
    contingency_matrix = pd.crosstab(predicted_clusters, true_labels)
    return np.sum(np.amax(contingency_matrix.values, axis=1)) / len(true_labels)

# purity for DBSCAN (excluding noise)
dbscan_purity = purity_score(df_cluster[model_target][dbscan_clusters != -1], filtered_dbscan_clusters)

# purity for GMM
gmm_purity = purity_score(df_cluster[model_target], gmm_clusters)

# output results
print("DBSCAN Metrics:")
if dbscan_silhouette is not None:
    print(f" - Silhouette Score: {dbscan_silhouette}")
    print(f" - Davies-Bouldin Index: {dbscan_dbi}")
else:
    print(" - Silhouette Score: Not applicable (only one cluster)")
    print(" - Davies-Bouldin Index: Not applicable (only one cluster)")
print(f" - Noise Percentage: {dbscan_noise_percentage:.2%}")
print(f" - Purity Score: {dbscan_purity}")

print("\nGMM Metrics:")
if gmm_silhouette is not None:
    print(f" - Silhouette Score: {gmm_silhouette}")
    print(f" - Davies-Bouldin Index: {gmm_dbi}")
else:
    print(" - Silhouette Score: Not applicable (only one cluster)")
    print(" - Davies-Bouldin Index: Not applicable (only one cluster)")
print(f" - Purity Score: {gmm_purity}")

X = df_clean.drop(columns=['Revenue'])
Y = df_clean['Revenue'].astype(int)

value_counts = Y.value_counts()
print(value_counts)

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
