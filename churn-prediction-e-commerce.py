
# Import Package
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer #for Imputing missing value
from sklearn.preprocessing import LabelEncoder # for encoding Gender
from sklearn.model_selection import train_test_split # for spliting data

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

"""# Understand Data

## Load Dataset
"""

df = pd.read_csv('/content/drive/MyDrive/UNIGAP/Machine Learning/Unsupervised Learning/churn_prediction.csv')
df.head(3)

"""## Check values and dtypes"""

numeric_cols = df.select_dtypes(include=['int64', 'float']).columns
categorical_cols = df.select_dtypes(exclude=['int64', 'float']).columns
print(numeric_cols)
print(categorical_cols)

df.info()

df.describe()

for col in categorical_cols:
    unique_vals = df[col].unique()
    print(f'Column: {col}, Unique Values: {unique_vals}')

df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace({'Phone': 'Mobile Phone'})
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({'COD': 'Cash on Delivery','CC': 'Credit Card'})
df['PreferedOrderCat'] = df['PreferedOrderCat'].replace({'Mobile': 'Mobile Phone'})
for col in categorical_cols:
    unique_vals = df[col].unique()
    print(f'Column: {col}, Unique Values: {unique_vals}')

"""### check outlier"""

data = df.copy()
for col in numeric_cols:
    data_col = data[col]
    #Tính Q1,Q3, IQR
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    #Tính lower bound và upper bound
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    #Tìm các giá trị ngoại lệ
    outliers = data_col[(data_col < lower_bound) | (data_col > upper_bound)]
    #Calculate the percentage of outliers
    outlier_pct = len(outliers) / len(data_col) * 100
    print(f'Column: {col}, lower_bound: {lower_bound}, upper_bound: {upper_bound}')

"""# Missing and Duplicate data

## Missing
"""

check_null = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
check_null['%missing'] = check_null[0] / len(df) * 100
check_null.columns = ['count', '%missing']
print(check_null)

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mean_cols = ['HourSpendOnApp']
median_cols = df.columns[df.isna().sum()>0].drop('HourSpendOnApp')

imp_mean = SimpleImputer(strategy='mean')
X_train[mean_cols] = imp_mean.fit_transform(X_train[mean_cols])
X_test[mean_cols] = imp_mean.transform(X_test[mean_cols])

imp_median = SimpleImputer(strategy='median')
X_train[median_cols] = imp_median.fit_transform(X_train[median_cols])
X_test[median_cols] = imp_median.transform(X_test[median_cols])
X_full_imputed = pd.concat([X_train, X_test]).sort_index()
df.update(X_full_imputed[mean_cols + list(median_cols)])

print(df.isnull().sum())

"""## Duplicate"""

duplicate = df.duplicated().sum()
print(duplicate)

"""# Data Processing

## Encoding
- Gender: 2 unique values --> Label Encoding
- Others: one-hot encoding
"""

unique_counts = df[categorical_cols].nunique()
unique_counts

df_encoded = df.copy()
le = LabelEncoder()
df_encoded['Gender'] = le.fit_transform(df_encoded['Gender'])

onehot_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus']
df_encoded = pd.get_dummies(df_encoded, columns=onehot_cols, drop_first=True)
df_encoded.head()

bool_cols = df_encoded.select_dtypes(include='bool').columns
for col in bool_cols:
    df_encoded[col] = df_encoded[col].astype(int)
df_encoded.head(5)

print(df_encoded.isnull().sum().sort_values(ascending=False))

"""## Drop columns that not in use
- Customer ID
"""

df_encoded.drop(columns='CustomerID', inplace=True)
df_encoded.head(3)

"""# Churn Prediction

## Model Training

### Split dataset
"""

X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(X_train.shape, X_test.shape, X_val.shape)

"""### Normalize"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

"""### Apply Model

#### Logistic Regression
"""

kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]
              , 'penalty': ['l1', 'l2']
              , 'solver': ['liblinear','saga']}

log_reg = LogisticRegression(max_iter=1000,)
grid_search = GridSearchCV(log_reg, param_grid, cv=kf, scoring='balanced_accuracy')
grid_search.fit(X_train_scaled, y_train)
# Print the best parameters
print("Best Parameters: ", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
log_y_pred_test = best_model.predict(X_test_scaled)
log_y_pre_val = best_model.predict(X_val_scaled)
log_test_score = accuracy_score(y_test, log_y_pred_test)
log_val_score = accuracy_score(y_val, log_y_pre_val)
print(f'Balanced Accuracy Test: {log_test_score}')
print(f'Balanced Accuracy Validation: {log_val_score}')

print(confusion_matrix(y_test, log_y_pred_test))
print(classification_report(y_test, log_y_pred_test))

"""#### KNN"""

train_accuracies = {}
test_accuracies = {}
val_accuracies = {}
neighbors = np.arange(1,26)
for neighbor in neighbors:
  knn = KNeighborsClassifier(n_neighbors = neighbor)
  knn.fit(X_train_scaled, y_train)
  train_accuracies[neighbor] = knn.score(X_train_scaled, y_train)
  test_accuracies[neighbor] = knn.score(X_test_scaled, y_test)
print(train_accuracies)
print(test_accuracies)

plt.figure(figsize=(8,6))
plt.title("KNN: Varying of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label = "Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label = "Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()

best_k = 2
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
knn_y_test_pred = knn.predict(X_test_scaled)
knn_y_val_pred = knn.predict(X_val_scaled)
knn_test_score = accuracy_score(y_test, knn_y_test_pred)
knn_val_score = accuracy_score(y_val, knn_y_val_pred)
print(f'Test Accuracy: {knn_test_score}')
print(f'Validation Accuracy: {knn_val_score}')

print(confusion_matrix(y_test, knn_y_test_pred))
print(classification_report(y_test, knn_y_test_pred))

"""#### Random Forest"""

kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'n_estimators': [100,150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}
rdf = RandomForestClassifier()
grid_search = GridSearchCV(rdf, param_grid, cv=kf, scoring='balanced_accuracy')

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters: ", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

best_rdf= grid_search.best_estimator_
rdf_y_pred_test = best_rdf.predict(X_test)
rdf_y_pre_val = best_rdf.predict(X_val)
print("Accuracy (Test):", accuracy_score(y_test, rdf_y_pred_test))
print("Accuracy (Validation):", accuracy_score(y_val, rdf_y_pre_val))

print(confusion_matrix(y_test, rdf_y_pred_test))
print(classification_report(y_test, rdf_y_pred_test))

print(confusion_matrix(y_val, rdf_y_pre_val))
print(classification_report(y_val, rdf_y_pre_val))

"""# Importance Feature"""

print(X.columns)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

feats = {}
for feature, importance in zip(X.columns, clf.feature_importances_):
    feats[feature] = importance

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances = importances.sort_values(by='Gini-importance', ascending=True).reset_index()

plt.figure(figsize=(8,8))
plt.barh(importances['index'].tail(20), importances['Gini-importance'].tail(20))
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Gini Importance')
plt.tight_layout()
plt.show()

"""# Churn Group

## Dimension Reduction
"""

df_churned = df_encoded[df_encoded['Churn'] == 1].copy()
df_churned.head(5)

df_churned = df_churned.drop(columns=['Churn'])
df_churned.info()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_churned)
scaled_df = pd.DataFrame(scaled_data, columns=df_churned.columns)
scaled_df.info()

pca = PCA(n_components=3)
X_pca = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
pca_df.head()

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

"""## Apply K-Means Model

## Which n_cluster?
"""

ks = range(1,11)
inertias = []
for k in ks:
  model = KMeans(n_clusters=k, init='k-means++', random_state=42)
  model.fit(pca_df)
  inertias.append(model.inertia_)
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

"""## Apply K-Means"""

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(pca_df[['PC1', 'PC2', 'PC3']])
pca_df['Cluster'] = clusters
df_churned['Cluster'] = clusters
df_churned

df_churned[['CashbackAmount','Tenure','WarehouseToHome','SatisfactionScore','CouponUsed']].groupby(df_churned['Cluster']).mean().round(2)

df_churned['Cluster'].value_counts().sort_index()

"""## Evaluating model"""

from sklearn.metrics import silhouette_score

sil_score = silhouette_score(pca_df, clusters)
print(sil_score)

plt.figure(figsize=(6,6))

sns.countplot(
    data=df,
    x="PreferedOrderCat",
    hue=df_churned["Cluster"],
    palette="Set2"
)

plt.title("Cluster's Profile Based On Preferred Order Category")
plt.xlabel("Preferred Order Category")
plt.ylabel("Customer Count")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x=df["DaySinceLastOrder"],
    y=df["WarehouseToHome"],
    hue=df_churned["Cluster"],
    palette="Set2"
)
plt.title("Cluster's Profile Based on Last Order and Delivery Distance")
plt.xlabel("Days Since Last Order")
plt.ylabel("Warehouse To Home Distance")
plt.legend(title="Cluster")
plt.show()
