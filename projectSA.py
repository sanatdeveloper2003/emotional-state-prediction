# PART 0: Setup environment
# Basic environment setup using a virtual environment in the project.
# To activate: source .venv/bin/activate

# PART 1: Import libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For statistical data visualization

# For Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# For 3D plotting
from mpl_toolkits.mplot3d import Axes3D

# For decision trees
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# For Sequential Feature Selection (SFS)
from sklearn.feature_selection import SequentialFeatureSelector as SFS

# For logistic regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# For model development and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# PART 2: Load dataset
df = pd.read_csv('Dataset_Study3.csv')
print(df.head())  # Display first 5 rows
print(df.shape)  # Dimensions of the dataset
print(df.describe())  # Summary statistics

# PART 3: Data exploration and EDA (Exploratory Data Analysis)
# Checking for missing values, duplicates, and data types
print(df.dtypes)  # Data types of each column
print(df.isnull().sum())  # Check for missing values
df.info()  # Detailed information about the dataset
print(df.duplicated().sum())  # Check for duplicated rows

# PART 4: Data visualization
# Boxplot for each column
# A boxplot visualizes the distribution based on five-number summary (min, Q1, median, Q3, max).
df.boxplot(figsize=(20, 10))
plt.show()

# Histogram for each column
# A histogram represents the distribution of numerical data. It bins the range of values and counts how many fall into each bin.
df.hist(figsize=(20, 10))
plt.show()

# PART 5: Data preprocessing
# Function to detect outliers using Interquartile Range (IQR)
def find_outliers(df, column):
    q1 = df[column].quantile(0.25)  # First quartile
    q3 = df[column].quantile(0.75)  # Third quartile
    iqr = q3 - q1  # Interquartile range
    lower_bound = q1 - (1.5 * iqr)  # Lower bound for outliers
    upper_bound = q3 + (1.5 * iqr)  # Upper bound for outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]  # Identify outliers
    return outliers

# For HRV_VLF
plt.figure(figsize=(10, 8))
df.boxplot(column='HRV_VLF', grid=False, vert=False)
plt.show()
df['HRV_VLF'].hist()
plt.show()

# Remove outliers and replace them with the median value
outliers = find_outliers(df, 'HRV_VLF')
df.loc[outliers.index, 'HRV_VLF'] = df['HRV_VLF'].median()

# Replot the boxplot and histogram for HRV_VLF after outlier removal
plt.figure(figsize=(10, 8))
df.boxplot(column='HRV_VLF', grid=False, vert=False)
plt.show()
df['HRV_VLF'].hist()
plt.show()

# For HRV_LF
plt.figure(figsize=(10, 8))
df.boxplot(column='HRV_LF', grid=False, vert=False)
plt.show()
df['HRV_LF'].hist()
plt.show()

# Remove outliers and replace them with the median value
outliers = find_outliers(df, 'HRV_LF')
df.loc[outliers.index, 'HRV_LF'] = df['HRV_LF'].median()

# Replot the boxplot and histogram for HRV_LF after outlier removal
plt.figure(figsize=(10, 8))
df.boxplot(column='HRV_LF', grid=False, vert=False)
plt.show()
df['HRV_LF'].hist()
plt.show()

# For HRV_HF
plt.figure(figsize=(10, 8))
df.boxplot(column='HRV_HF', grid=False, vert=False)
plt.show()
df['HRV_HF'].hist()
plt.show()

# Remove outliers and replace them with the median value
outliers = find_outliers(df, 'HRV_HF')
df.loc[outliers.index, 'HRV_HF'] = df['HRV_HF'].median()

# Replot the boxplot and histogram for HRV_HF after outlier removal
plt.figure(figsize=(10, 8))
df.boxplot(column='HRV_HF', grid=False, vert=False)
plt.show()
df['HRV_HF'].hist()
plt.show()

# For HRV_TP
plt.figure(figsize=(10, 8))
df.boxplot(column='HRV_TP', grid=False, vert=False)
plt.show()
df['HRV_TP'].hist()
plt.show()

# Remove outliers and replace them with the median value
outliers = find_outliers(df, 'HRV_TP')
df.loc[outliers.index, 'HRV_TP'] = df['HRV_TP'].median()

# Replot the boxplot and histogram for HRV_TP after outlier removal
plt.figure(figsize=(10, 8))
df.boxplot(column='HRV_TP', grid=False, vert=False)
plt.show()
df['HRV_TP'].hist()
plt.show()

outliers = find_outliers(df, 'HRV_TP')
outliers.shape ## 15 rows, 20 columns
df['HRV_TP'].loc[outliers.index] = df['HRV_TP'].median() ## replace outliers with median value
df.boxplot(figsize=(20,12)) ## boxplot for all columns
plt.show()
df.hist(figsize = (12,15))
plt.show()



# PART 6: Data correlation and visual analysis

# Remove the 'Label' column as it is not needed for correlation analysis
df_clean = df.drop('Label', axis=1)

# Calculate the correlation matrix (showing correlations between features)
correlation_matrix = df_clean.corr(numeric_only=True)

# Plot a heatmap to visualize the correlation matrix
plt.figure(figsize=(17, 15))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, vmin=-1, vmax=1, linewidths=0.5, square=True)
plt.title('Correlation Matrix')
plt.show()

# Pairplot to visualize pairwise relationships between features in the dataset
sns.pairplot(df, height=3)
plt.show()

# Histograms for the 'HRV_TP' feature, grouped by conditions (baseline, gratitude, natural)
conditions = {'baseline': -1, 'gratitude': 310, 'natural': 109}
plt.figure(figsize=(10, 6))

for label, condition in conditions.items():
    plt.hist(df[df['Label'] == condition]['HRV_TP'], bins=30, alpha=0.5, label=label)

plt.xlabel('HRV_TP values')
plt.ylabel('Frequency')
plt.title('Histogram of HRV_TP by Condition')
plt.legend()
plt.show()

# Boxplot of HRV_TP values grouped by different conditions (baseline, gratitude, natural)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Label', y='HRV_TP', data=df[df['Label'].isin(conditions.values())])
plt.xlabel('Condition')
plt.ylabel('HRV_TP values')
plt.title('Boxplot of HRV_TP by Condition')
plt.show()

# Violin plot to visualize HRV_TP distribution under different conditions
plt.figure(figsize=(10, 6))
sns.violinplot(x='Label', y='HRV_TP', data=df)
plt.title('Violin Plot of HRV_TP by Condition')
plt.show()


# PART 7: Principal Component Analysis (PCA)
# PCA reduces the dimensionality of the data while retaining most of the information.

# Step 1: Separate features and labels
X = df.drop('Label', axis=1)  # Features
y = df['Label']  # Labels

# Step 2: Standardize the features (mean=0, variance=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Perform PCA and store the components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Step 4: Create a DataFrame for PCA components
pca_df = pd.DataFrame(X_pca, columns=[f'PCA{i}' for i in range(1, X.shape[1] + 1)])

# Step 5: Add labels back to the PCA DataFrame
pca_df['Label'] = y.values

# Step 6: Calculate explained variance ratio for each component
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio: {explained_variance}")

# Step 7: Plot the explained variance
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Bar plot of explained variance for each component
axes[0].bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
axes[0].set_xlabel('Principal Components')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Explained Variance Ratio by Principal Components')

# Line plot of explained variance ratio
axes[1].plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-', color='b')
axes[1].set_xlabel('Principal Components')
axes[1].set_ylabel('Explained Variance Ratio')
axes[1].set_title('Explained Variance Ratio (Line Plot)')

plt.tight_layout()
plt.show()

# PART 8: Feature Selection using various methods

# 1. Random Forest Classifier to determine feature importance
rf_model = RandomForestClassifier()
rf_model.fit(X, y)
importances = rf_model.feature_importances_

# Print feature rankings based on Random Forest importance
print("Feature Ranking based on Random Forest:")
rf_indices = np.argsort(importances)[::-1]  # Sort in descending order
for idx in rf_indices[:5]:  # Show top 5 features
    print(f"Feature: {X.columns[idx]} Importance: {importances[idx]:.6f}")

# Store top 5 features selected by Random Forest
selected_rf_features = X.columns[rf_indices[:5]]
print(f"\nTop 5 features selected by Random Forest: {list(selected_rf_features)}")

# 2. Decision Tree Classifier for feature importance
dt_model = DecisionTreeClassifier(random_state=0)
dt_model.fit(X, y)
dt_importances = dt_model.feature_importances_

# Print feature rankings for Decision Tree
print("\nFeature Ranking based on Decision Tree:")
dt_indices = np.argsort(dt_importances)[::-1]  # Sort in descending order
for idx in dt_indices[:5]:  # Show top 5 features
    print(f"Feature: {X.columns[idx]} Importance: {dt_importances[idx]:.6f}")

# Store top 5 features selected by Decision Tree
selected_dt_features = X.columns[dt_indices[:5]]
print(f"\nTop 5 features selected by Decision Tree: {list(selected_dt_features)}")

# 3. Sequential Feature Selection (SFS) using Random Forest
sfs_model = SFS(rf_model, n_features_to_select=5, direction='forward', scoring='accuracy', cv=5)
sfs_model.fit(X, y)

# Print and store features selected by SFS
selected_sfs_features = X.columns[sfs_model.get_support()]
print(f"\nFeatures selected by Sequential Feature Selection (SFS): {list(selected_sfs_features)}")

# 4. Recursive Feature Elimination (RFE) using Logistic Regression
lr_model = LogisticRegression(solver='lbfgs', max_iter=6000)
rfe_model = RFE(lr_model, n_features_to_select=5)
rfe_model.fit(X_scaled, y)

# Print and store features selected by RFE
selected_rfe_features = X.columns[rfe_model.get_support()]
print(f"\nFeatures selected by Recursive Feature Elimination (RFE): {list(selected_rfe_features)}")

# Display the final selected features from all methods
print("\nSummary of selected features:")
print(f"Random Forest: {list(selected_rf_features)}")
print(f"Decision Tree: {list(selected_dt_features)}")
print(f"SFS: {list(selected_sfs_features)}")
print(f"RFE: {list(selected_rfe_features)}")

# PART 9: Model Development and Evaluation
# Function to build and evaluate a model
def evaluate_model(X, y, test_size=0.3):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train Logistic Regression model
    lr_model = LogisticRegression(solver='lbfgs', max_iter=6000)
    lr_model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = lr_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    
    return accuracy, precision, recall

# Evaluate models using PCA components and RFE-selected features
print("\nEvaluation with PCA components:")
pca_accuracy, pca_precision, pca_recall = evaluate_model(X_pca, y)
print(f"Accuracy: {pca_accuracy:.4f}, Precision: {pca_precision:.4f}, Recall: {pca_recall:.4f}")

print("\nEvaluation with RFE-selected features:")
rfe_accuracy, rfe_precision, rfe_recall = evaluate_model(rfe_model.transform(X_scaled), y)
print(f"Accuracy: {rfe_accuracy:.4f}, Precision: {rfe_precision:.4f}, Recall: {rfe_recall:.4f}")
