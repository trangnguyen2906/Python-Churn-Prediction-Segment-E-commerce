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

### 🔹 Normalize Features: Scale numerical values for better model performance
Applied `StandardScaler` to standardize features for better convergence and model performance.

### 🔸 Models Training:

#### **🤖 Logistic Regression**
 - Tuned using GridSearchCV with `penalty`, `C`, and `solver`
    - Best cross-validation accuracy: **~0.756**
    - Test accuracy: **0.88**, Validation accuracy: **0.886**
    - Precision (Class 1): **0.71**, Recall: **0.52**, F1-score: **0.60**

#### **🧭 K-Nearest Neighbors (KNN)**
- Explored multiple `k` values and visualized accuracy trend
    - Best `k = 2` with test accuracy: **0.924**, validation accuracy: **0.936**
    - Balanced precision/recall across classes

#### **🌲 Random Forest**
 - Tuned multiple hyperparameters: `n_estimators`, `max_depth`, `min_samples_leaf`, etc.
    - Best test accuracy: **0.953**, validation accuracy: **0.963**
    - F1-score (Class 1): **0.86**, Recall: **0.79**, Precision: **0.94**
    - Most balanced performance among all models

🔍 **Summary**

| Model              | Test Accuracy | Val Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|-------------------|---------------|--------------|-------------------|----------------|------------|
| Logistic Regression | 0.88          | 0.886        | 0.71              | 0.52           | 0.60       |
| KNN (k=2)          | 0.924         | 0.936        | 0.92              | 0.62           | 0.74       |
| Random Forest      | **0.953**     | **0.963**    | **0.94**          | **0.79**       | **0.86**   |

🌟 **Random Forest outperformed all other models** in both accuracy and class balance, and was chosen for churn prediction.

### 🔹Feature Importance (via Random Forest)


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
