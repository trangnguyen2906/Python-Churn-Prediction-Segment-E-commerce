# 📊 Predict & Segment E-commerce Churned Users Using Python & Machine Learning

## 📑 Table of Contents  
1. [📌 Background & Overview](#-background--overview)  
2. [📂 Dataset Description & Data Structure](#-dataset-description--data-structure)  
3. [🔎 Final Conclusion & Recommendations](#-final-conclusion--recommendations)

---

## 📌 Background & Overview  

### 🧩 Project Context:
A e-commerce company is facing a business challenge: user churn. Many customers stop interacting with the platform after a short period, resulting in lost revenue and reduced customer lifetime value. However, the company currently lacks a systematic, data-driven approach to understand: 

- Why users churn
- How to predict churn in advance
- How to re-engage churned users effectively

### Objective:

### 📖 What is this project about? What Business Question will it solve?

This project uses **Machine Learning** to help an e-commerce company proactively tackle customer churn and design better promotion strategies. 

It focuses on answering three key business questions:

- 📊 What are the **patterns and behaviors of churned users?**
→ Understand why users churn and suggest actionable retention strategies

- 🔍 Which **users** are **likely to churn**?
→ Build and fine-tune supervised ML models to predict churn risk

- 🧩 How can we **group churned users** for targeted promotions? 
→ Apply unsupervised ML (clustering) to segment churned users by behavior and value 
 
🎯 The goal is to support data-driven decision-making in **user retention, promotion targeting, and customer lifecycle management.**


### 👤 Who is this project for?  
- **Marketing Teams** – to target churned users with personalized campaigns
- **Data Analysts** – to uncover churn patterns and build models
- **Decision Makers** – to guide strategy and resource allocation

---

## 📂 Dataset Description & Data Structure  

### 📌 Data Source  
- Source: Internal company dataset provided
- Size: 5,630 rows × 20 columns 
- Format: `.csv` 
### 📊 Data Structure & Relationships  

#### 1️⃣ Tables Used:  
- The dataset contains one main table: `churn_prediction.csv`

#### 2️⃣ Table Schema 

| Column Name                | Description                                                   |
|----------------------------|---------------------------------------------------------------|
| `CustomerID`              | Unique customer ID                                            |
| `Churn`                   | Target variable: churn flag (1 = churned, 0 = retained)       |
| `Tenure`                  | Duration the customer has used the platform       |
| `PreferredLoginDevice`    | Most frequently used device to access the platform            |
| `CityTier`                | City classification (1, 2, 3) indicating customer location     |
| `WarehouseToHome`         | Distance between customer’s home and warehouse                |
| `PreferPaymentMethod`     | Customer’s preferred payment method                           |
| `Gender`                  | Gender of the customer                                        |
| `HourSpendOnApp`          | Total hours spent on the app or website                       |
| `NumberOfDeviceRegistered`| Number of devices registered to this customer                 |
| `PreferredOrderCat`       | Category most often ordered in the last month                 |
| `SatisfactionScore`       | Satisfaction rating from the customer                         |
| `MaritalStatus`           | Marital status of the customer                                |
| `NumberOfAddress`         | Number of saved addresses by customer                         |
| `Complain`                | Whether any complaint was raised in the last month            |
| `OrderAmountHikeFromLastYear` | % increase in order amount vs. last year                   |
| `CouponUsed`              | Number of coupons used in the last month                      |
| `OrderCount`              | Number of orders placed in the last month                     |
| `DaySinceLastOrder`       | Days since last order was placed                              |
| `CashbackAmount`          | Average cashback received in the last month  


---

## ⚒️ Main Process

## 1️⃣ Data Cleaning
- 🔹 **Load Dataset**  
  Imported dataset (`churn_prediction.csv`) and previewed the first few rows using `df.head()` to verify structure.

- 🔹 **Check Data Types & Values**  
  - Separated numerical and categorical columns using `df.select_dtypes()`
  - Inspected overall structure with `df.info()` and summary statistics using `df.describe()`
  - Checked unique values in each categorical column for understanding data distribution

- 🔹 **Standardize Categorical Labels**  
  - Cleaned inconsistent category labels (e.g., mapping `'phone'` → `'Mobile Phone'`, `'COD'` → `'Cash on Delivery'`)

```
numeric_cols = df.select_dtypes(include=['int64', 'float']).columns
categorical_cols = df.select_dtypes(exclude=['int64', 'float']).columns
print(numeric_cols)
print(categorical_cols)

for col in categorical_cols:
    unique_vals = df[col].unique()
    print(f'Column: {col}, Unique Values: {unique_vals}')

df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace({'Phone': 'Mobile Phone'})
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({'COD': 'Cash on Delivery','CC': 'Credit Card'})
df['PreferedOrderCat'] = df['PreferedOrderCat'].replace({'Mobile': 'Mobile Phone'})
for col in categorical_cols:
    unique_vals = df[col].unique()
    print(f'Column: {col}, Unique Values: {unique_vals}')
```
<img src="https://drive.google.com/uc?export=view&id=17Jd7X3lE7M3mWC9-czqchDk5uDjV0mfE"/>

- 🔹 **Check for Missing Values**  
  - Calculated missing counts and percentages  
  - Found several columns (e.g., `DaySinceLastOrder`, `Tenure`, `CouponUsed`) with missing data  
  - Handled missing values:
    - Used **mean** imputation for `HourSpendOnApp`  
    - Used **median** imputation for other numerical columns with missing values:  `Tenure`, `CouponUsed`, `DaySinceLastOrder` are often **skewed** or **contain outliers**, so median is more robust and reduces distortion in such cases.

- 🔹 **Check for Duplicates**  
  Verified that there are **no duplicated rows** in the dataset using `df.duplicated().sum()`

```
## Missing
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

## Duplicate
duplicate = df.duplicated().sum()
print(duplicate)
```
<img src="https://drive.google.com/uc?export=view&id=1U1wGPULAzrJ1h-VaTH58bm2R6HSuQta5"/>

## 2️⃣ Data Preprocessing

- 🔹 **Encoding Categorical Variables** : Categorical columns (`PreferredLoginDevice`, `PreferredPaymentMode`, etc.): used **One-Hot Encoding** with `drop_first=True` to avoid multicollinearity.

- 🔹 **Convert Boolean Columns**  
  - Converted `True/False` dummy variables to `0/1` integers for compatibility with ML algorithms.

- 🔹 **Check Post-Encoding**  
  - Verified that no missing values remained using `df_encoded.isnull().sum()`

- 🔹 **Drop Unused Columns**  
  - Removed `CustomerID` since it's only an identifier and not informative for modeling

```
df_encoded = df.copy()
onehot_cols = ['Gender','PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus']
df_encoded = pd.get_dummies(df_encoded, columns=onehot_cols, drop_first=True)
df_encoded.head()

bool_cols = df_encoded.select_dtypes(include='bool').columns
for col in bool_cols:
    df_encoded[col] = df_encoded[col].astype(int)
```
<img src="https://drive.google.com/uc?export=view&id=1Gkf6AojbrJXr4RDQ0fx6YAs4HUu6yQVq" />

## 3️⃣ Churn Prediction – Supervised Learning
This section focuses on building classification models to predict whether a user is likely to churn.

### 🔹 Split Dataset: Divide the data into training and test sets
Divided the dataset into training (70%), validation (15%), and test (15%) sets using `train_test_split`.

```
X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(X_train.shape, X_test.shape, X_val.shape)

(3941, 25) (844, 25) (845, 25)
```

### 🔹 Normalize Features: Scale numerical values for better model performance
Applied `StandardScaler` to standardize features for better convergence and model performance.

```
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)
```

### 🔸 Models Training:

#### **🤖 Logistic Regression**
 - Tuned using GridSearchCV with `penalty`, `C`, and `solver`
    - Best cross-validation accuracy: **~0.756**
    - Test accuracy: **0.88**, Validation accuracy: **0.886**
    - Precision (Class 1): **0.71**, Recall: **0.52**, F1-score: **0.60**

```
## Logistic Regression Model Training 
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
```

```
## Logistic Regression - Model Evaluation
best_model = grid_search.best_estimator_
log_y_pred_test = best_model.predict(X_test_scaled)
log_y_pre_val = best_model.predict(X_val_scaled)
log_test_score = accuracy_score(y_test, log_y_pred_test)
log_val_score = accuracy_score(y_val, log_y_pre_val)
print(f'Balanced Accuracy Test: {log_test_score}')
print(f'Balanced Accuracy Validation: {log_val_score}')
```

```
print(confusion_matrix(y_test, log_y_pred_test))
print(classification_report(y_test, log_y_pred_test))
```

#### **🧭 K-Nearest Neighbors (KNN)**
- Explored multiple `k` values and visualized accuracy trend
    - Best `k = 2` with test accuracy: **0.924**, validation accuracy: **0.936**
    - Balanced precision/recall across classes

```
## KNN Model Training - Choosing n_neighbors - plot the accuracies 
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
```
<img src="https://drive.google.com/uc?export=view&id=1vhXDThFMPJMT7hdtMtW8DRglWaW2WGun" width="700"/>

```
## KNN - Model Evaluation with k=2
best_k = 2
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
knn_y_test_pred = knn.predict(X_test_scaled)
knn_y_val_pred = knn.predict(X_val_scaled)
knn_test_score = accuracy_score(y_test, knn_y_test_pred)
knn_val_score = accuracy_score(y_val, knn_y_val_pred)
print(f'Test Accuracy: {knn_test_score}')
print(f'Validation Accuracy: {knn_val_score}')
```

```
print(confusion_matrix(y_test, knn_y_test_pred))
print(classification_report(y_test, knn_y_test_pred))
```

#### **🌲 Random Forest**
 - Tuned multiple hyperparameters: `n_estimators`, `max_depth`, `min_samples_leaf`, etc.
    - Best test accuracy: **0.953**, validation accuracy: **0.963**
    - F1-score (Class 1): **0.86**, Recall: **0.79**, Precision: **0.94**
    - Most balanced performance among all models

```
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
```

```
best_rdf= grid_search.best_estimator_
rdf_y_pred_test = best_rdf.predict(X_test)
rdf_y_pre_val = best_rdf.predict(X_val)
print("Accuracy (Test):", accuracy_score(y_test, rdf_y_pred_test))
print("Accuracy (Validation):", accuracy_score(y_val, rdf_y_pre_val))
```

```
print(confusion_matrix(y_test, rdf_y_pred_test))
print(classification_report(y_test, rdf_y_pred_test))
print(confusion_matrix(y_val, rdf_y_pre_val))
print(classification_report(y_val, rdf_y_pre_val))
```
<img src="https://drive.google.com/uc?export=view&id=1Ug3rhANnnGBo0ADoZeoCgzKrhdH4KlRE" width="700"/>


🔍 **Summary**

| Model              | Test Accuracy | Val Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|-------------------|---------------|--------------|-------------------|----------------|------------|
| Logistic Regression | 0.88          | 0.886        | 0.71              | 0.52           | 0.60       |
| KNN (k=2)          | 0.924         | 0.936        | 0.92              | 0.62           | 0.74       |
| Random Forest      | **0.953**     | **0.963**    | **0.94**          | **0.79**       | **0.86**   |

🌟 **Random Forest outperformed all other models** in both accuracy and class balance, and was chosen for churn prediction.

### 🔹Feature Importance (via Random Forest)
Used feature importances from the trained Random Forest model to understand which variables most influence churn predictions.

#### Finding the importance features 

```
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
plt.tight_layout()
plt.show()
```
<img src="https://drive.google.com/uc?export=view&id=1k4rKJmCddAc7V3hJLn5sleXOuyCmJ1Tw" width="700"/>

🔍 Observations:
From the chart, we can see that there are **5 variables** that most influence the churn predition: `Tenure`, `CashbackAmount`, `WarehouseToHome`, `Complain`, `DaySinceLastOrder`

#### Analyse features from initial Random Forest model
In this part, from the previous observations, I will analyze and examine how these features affect the churn

```
def count_percentage(df, column, target):
    ### This function to create the table calculate the percentage of churn/non-churn customer on total customer group by category values
    

    # Create 2 dataframes of churn and non-churn
    churn = df[df[target]==1].groupby(column).size().reset_index(name='count').sort_values(ascending=False, by = 'count')
    not_churn = df[df[target]==0].groupby(column).size().reset_index(name='count').sort_values(ascending=False, by = 'count')

    #Merge 2 dataframe into one:
    cate_df = churn.merge(not_churn, on = column , how = 'outer')
    cate_df = cate_df.fillna(0)
    # Rename columns to be more descriptive
    cate_df.rename(columns = {'count_x':'churn','count_y':'not_churn'}, inplace = True)

    #Caculate the percentage:
    cate_df['%'] = cate_df['churn']/(cate_df['not_churn']+cate_df['not_churn'])
    cate_df = cate_df.sort_values(by='%', ascending=False)

    return cate_df
    
```


## 4️⃣ Churn Segmentation – Unsupervised Learning

### 🔹 Dimension Reduction: Reduce features for efficient clustering

### 🔸 K-Means Clustering

### 🔹 Cluster Analysis:


---

## 🔎 Final Conclusion & Recommendations  

👉🏻 Based on the insights and findings above, we would recommend the [stakeholder team] to consider the following:  

📌 Key Takeaways:  
✔️ Recommendation 1  
✔️ Recommendation 2  
✔️ Recommendation 3
