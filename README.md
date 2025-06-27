# 📊 Predict & Segment E-commerce Churned Users Using Python & Machine Learning

## 📑 Table of Contents  
1. [📌 Background & Overview](#-background--overview)  
2. [📂 Dataset Description & Data Structure](#-dataset-description--data-structure)  
3. [🔎 Final Conclusion & Recommendations](#-final-conclusion--recommendations)

---

## 📌 Background & Overview  

### Objective:

### 📖 What is this project about? What Business Question will it solve?

A e-commerce company is facing a business challenge: **User Churn**. Many customers stop interacting with the platform after a short period, resulting in lost revenue and reduced customer lifetime value. 
This project uses **Machine Learning** to help an e-commerce company proactively **tackle customer churn** and **design better promotion strategies.** 

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

<details>
<summary>📄 <strong>Table Schema: churn_prediction.csv</strong></summary>

| Column Name                   | Description                                                  |
|------------------------------|--------------------------------------------------------------|
| `CustomerID`                 | Unique customer ID                                           |
| `Churn`                      | Target variable: churn flag (1 = churned, 0 = retained)      |
| `Tenure`                     | Duration the customer has used the platform                 |
| `PreferredLoginDevice`       | Most frequently used device to access the platform           |
| `CityTier`                   | City classification (1, 2, 3) indicating customer location   |
| `WarehouseToHome`           | Distance between customer’s home and warehouse              |
| `PreferPaymentMethod`        | Customer’s preferred payment method                         |
| `Gender`                     | Gender of the customer                                       |
| `HourSpendOnApp`             | Total hours spent on the app or website                     |
| `NumberOfDeviceRegistered`   | Number of devices registered to this customer               |
| `PreferredOrderCat`          | Category most often ordered in the last month               |
| `SatisfactionScore`          | Satisfaction rating from the customer                       |
| `MaritalStatus`              | Marital status of the customer                              |
| `NumberOfAddress`            | Number of saved addresses by customer                       |
| `Complain`                   | Whether any complaint was raised in the last month          |
| `OrderAmountHikeFromLastYear`| % increase in order amount vs. last year                    |
| `CouponUsed`                 | Number of coupons used in the last month                    |
| `OrderCount`                 | Number of orders placed in the last month                   |
| `DaySinceLastOrder`          | Days since last order was placed                            |
| `CashbackAmount`             | Average cashback received in the last month                 |

</details>

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

<img src="https://drive.google.com/uc?export=view&id=17Jd7X3lE7M3mWC9-czqchDk5uDjV0mfE"/>

- 🔹 **Check for Missing Values**  
  - Calculated missing counts and percentages  
  - Found several columns (e.g., `DaySinceLastOrder`, `Tenure`, `CouponUsed`) with missing data  
  - Handled missing values:
    - Used **mean** imputation for `HourSpendOnApp`  
    - Used **median** imputation for other numerical columns with missing values:  `Tenure`, `CouponUsed`, `DaySinceLastOrder` are often **skewed** or **contain outliers**, so median is more robust and reduces distortion in such cases.

- 🔹 **Check for Duplicates**  
  Verified that there are **no duplicated rows** in the dataset using `df.duplicated().sum()`

<img src="https://drive.google.com/uc?export=view&id=1U1wGPULAzrJ1h-VaTH58bm2R6HSuQta5"/>

## 2️⃣ Data Preprocessing

- 🔹 **Encoding Categorical Variables** : Categorical columns (`PreferredLoginDevice`, `PreferredPaymentMode`, etc.): used **One-Hot Encoding** with `drop_first=True` to avoid multicollinearity.

- 🔹 **Convert Boolean Columns**  
  - Converted `True/False` dummy variables to `0/1` integers for compatibility with ML algorithms.

- 🔹 **Check Post-Encoding**  
  - Verified that no missing values remained using `df_encoded.isnull().sum()`

- 🔹 **Drop Unused Columns**  
  - Removed `CustomerID` since it's only an identifier and not informative for modeling

## 3️⃣ Churn Prediction – Supervised Learning
This section focuses on building classification models to predict whether a user is likely to churn.

### 🔹 Split Dataset: Divide the data into training and test sets
Divided the dataset into training (70%), validation (15%), and test (15%) sets using `train_test_split`.

```python
X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(X_train.shape, X_test.shape, X_val.shape)

(3941, 25) (844, 25) (845, 25)
```

### 🔹 Normalize Features: Scale numerical values for better model performance
Applied `StandardScaler` to standardize features for better convergence and model performance.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)
```

### 🔸 Models Training:

#### 🔍 **Summary**

| Model              | Test Accuracy | Val Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|-------------------|---------------|--------------|-------------------|----------------|------------|
| Logistic Regression | 0.88          | 0.886        | 0.71              | 0.52           | 0.60       |
| KNN (k=2)          | 0.924         | 0.936        | 0.92              | 0.62           | 0.74       |
| Random Forest      | **0.953**     | **0.963**    | **0.94**          | **0.79**       | **0.86**   |

🌟 **Random Forest outperformed all other models** in both accuracy and class balance, and was chosen for churn prediction.

#### **🤖 Logistic Regression**
 - Tuned using GridSearchCV with `penalty`, `C`, and `solver`
    - Best cross-validation accuracy: **~0.756**
    - Test accuracy: **0.88**, Validation accuracy: **0.886**
    - Precision (Class 1): **0.71**, Recall: **0.52**, F1-score: **0.60**

```python
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

```python
## Logistic Regression - Model Evaluation
best_model = grid_search.best_estimator_
log_y_pred_test = best_model.predict(X_test_scaled)
log_y_pre_val = best_model.predict(X_val_scaled)
log_test_score = accuracy_score(y_test, log_y_pred_test)
log_val_score = accuracy_score(y_val, log_y_pre_val)
print(f'Balanced Accuracy Test: {log_test_score}')
print(f'Balanced Accuracy Validation: {log_val_score}')
```

```python
print(confusion_matrix(y_test, log_y_pred_test))
print(classification_report(y_test, log_y_pred_test))
```

#### **🧭 K-Nearest Neighbors (KNN)**
- Explored multiple `k` values and visualized accuracy trend
    - Best `k = 2` with test accuracy: **0.924**, validation accuracy: **0.936**
    - Balanced precision/recall across classes

```python
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

```python
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

```python
print(confusion_matrix(y_test, knn_y_test_pred))
print(classification_report(y_test, knn_y_test_pred))
```

#### **🌲 Random Forest**
 - Tuned multiple hyperparameters: `n_estimators`, `max_depth`, `min_samples_leaf`, etc.
    - Best test accuracy: **0.953**, validation accuracy: **0.963**
    - F1-score (Class 1): **0.86**, Recall: **0.79**, Precision: **0.94**
    - Most balanced performance among all models

```python
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

```python
best_rdf= grid_search.best_estimator_
rdf_y_pred_test = best_rdf.predict(X_test)
rdf_y_pre_val = best_rdf.predict(X_val)
print("Accuracy (Test):", accuracy_score(y_test, rdf_y_pred_test))
print("Accuracy (Validation):", accuracy_score(y_val, rdf_y_pre_val))
```

```python
print(confusion_matrix(y_test, rdf_y_pred_test))
print(classification_report(y_test, rdf_y_pred_test))
print(confusion_matrix(y_val, rdf_y_pre_val))
print(classification_report(y_val, rdf_y_pre_val))
```
<img src="https://drive.google.com/uc?export=view&id=1Ug3rhANnnGBo0ADoZeoCgzKrhdH4KlRE" width="700"/>


### 🔹 Feature Importance (via Random Forest)
Used feature importances from the trained Random Forest model to understand which variables most influence churn predictions.

#### 📌 Finding the importance features 

```python
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

🔍 **Observations:**

From the chart, we can see that there are **5 variables** that most influence the churn predition: `Tenure`, `CashbackAmount`, `WarehouseToHome`, `Complain`, `DaySinceLastOrder`

#### 📊 Analyse features from initial Random Forest model
In this part, from the previous observations, I will analyze and examine how these features affect the churn

```python
def count_percentage(df, column, target):
    ### Function to create the table calculate the percentage
    ### of churn/non-churn customer on total customer group by category values
    

    # Create 2 dataframes of churn and non-churn
    churn = df[df[target]==1].groupby(column).size().reset_index(name='count').sort_values(ascending=False, by = 'count')
    not_churn = df[df[target]==0].groupby(column).size().reset_index(name='count').sort_values(ascending=False, by = 'count')

    #Merge 2 dataframe into one:
    cate_df = churn.merge(not_churn, on = column , how = 'outer')
    cate_df = cate_df.fillna(0)
    # Rename columns
    cate_df.rename(columns = {'count_x':'churn','count_y':'not_churn'}, inplace = True)

    #Caculate the percentage:
    cate_df['%'] = cate_df['churn']/(cate_df['not_churn']+cate_df['not_churn'])
    cate_df = cate_df.sort_values(by='%', ascending=False)

    return cate_df
    
```

📎 **Tenure:** Verify whether the **time/duration** that customer used the platform does affect the churn.

```python
plot_df = count_percentage(df, 'Tenure', 'Churn')

#Visualize the data:
fig, ax = plt.subplots(figsize=(14, 5))
sns.barplot(data=plot_df, x='Tenure',y='%', ax=ax)

plt.title("Tenure vs Churn Percentage")
plt.xlabel("Tenure")
plt.ylabel("Churn Percentage")
plt.show()
```
<img src="https://drive.google.com/uc?export=view&id=1b90HUuTcdjr0ui3q13V2CxixXPhdTS_F" width="700"/>


```python
count_df = df.groupby('Tenure').size().reset_index(name='count')

# Visualize the data:
fig, ax = plt.subplots(figsize=(14, 5))
sns.barplot(data=count_df, x='Tenure', y='count', ax=ax)

plt.title("Tenure vs Number of Users")
plt.xlabel("Tenure")
plt.ylabel("Number of Users")
plt.show()
```
<img src="https://drive.google.com/uc?export=view&id=1oTyfAVQ9YONap8lUixSHOYV2hKo9VsuF" width="700"/>

🔍 **Observations:**

- 💡 **New users (Tenure = 0 or 1)** make up the largest user group, but they also have the highest churn rates **(over 50%)**. This means that most users **leave the service very early**
- 📉 After the initial stage **(Tenure ≥ 2)**, churn rates **drop sharply** and remain consistently **low**. This suggests that once users stay beyond the early phase, they tend to be more loyal and get more value from the service.
- 🔢 The number of users declines as tenure increases, meaning it’s **challenging to retain users over the long term**. However, those who do **stay longer** show very **low churn rates**, indicating they are high-value users worth investing in.

📎 **Warehouse to home:** Verify whether the **distance** between customer’s home and warehouse does affect the churn.

```python
fig, ax = plt.subplots(figsize=(6, 6))
sns.boxplot(data=df, x='Churn',y='WarehouseToHome', showfliers = False)
```

<img src="https://drive.google.com/uc?export=view&id=1_iKzMn10U-txv7vVoPhQxUDaZJqzD_1e" width="400"/>

🔍 **Observations:**
- There're **no strong evidences** show that there different between churn and not churn for warehousetohome → We should **exclude this feature** when apply model for not being bias.

📎 **Days since last order:** Verify whether the **number of days since a customer’s last order** influences their likelihood to churn.

```python
fig, ax = plt.subplots(figsize=(6, 6))
sns.boxplot(data=df, x='Churn',y='DaySinceLastOrder',ax=ax, showfliers = False)
```

<img src="https://drive.google.com/uc?export=view&id=1WqRImb8tO-TJGXtv7FLPzn0KHb3GZDF9" width="400"/>


🔍 **Observations:**
From this chart, we see for churned users, they had orders recently (the day since last order less than not churned users) --> This quite strange, we should monitor more features for this insight (satisfaction_score, complain,..)

```python
churn_df = df[df['Churn']==1]
fig, ax = plt.subplots(figsize=(6, 6))
sns.boxplot(data=churn_df, x='Complain',y='DaySinceLastOrder',ax=ax, showfliers = False)
```

<img src="https://drive.google.com/uc?export=view&id=1ldrp_Xje_sOSrKl9DJJMUNvkZet8V3BV" width="400"/>

🔍 **Observations:**
For churned users with complain = 1, they had daysincelastorder higher than churn users with compain = 0

📎 **Complain** Verify whether having **raised a complain** increases a customer's likelihood to churn.


```python
plot_df = count_percentage(df, 'Complain', 'Churn')
#Visualize the data:
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=plot_df, x='Complain',y='%', ax=ax)
```

<img src="https://drive.google.com/uc?export=view&id=1pVqArOUfJ0K9l4C0wZHJfAVua3Abv3Lo" width="400"/>

🔍 **Observations:**
Users who raised complaints show a **much higher churn rate**, indicating complaint resolution is crucial for retention.

📎 **CashbackAmount:** Verify whether receiving a **lower cashback value** is associated with a higher likelihood of churn.

```python
fig, ax = plt.subplots(figsize=(6, 6))
sns.boxplot(data=df, x='Churn',y='CashbackAmount',ax=ax, showfliers = False)
```

<img src="https://drive.google.com/uc?export=view&id=1yKjuQ0_B8-xP3F9erZthpSh07PJtOSfX" width="400"/>

🔍 **Observations:**
Churned users tend to **receive less cashback**, suggesting that higher cashback may help improve retention.

## 4️⃣ Churn Segmentation – Unsupervised Learning

### 🔹 Dimension Reduction: Reduce features for efficient clustering
- **Excluded `Churn` and `WarehouseToHome`** based on prior analysis (warehouse distance showed minimal correlation to churn).
  
- Applied `MinMaxScaler` to normalize all numerical features.
  
- Used **PCA** to reduce dimensionality to 3 components.

-   **Explained variance ratio**: [0.1549, 0.1510, 0.1063] → ~41.2% of total variance explained.

```python
df_churned = df_encoded[df_encoded['Churn'] == 1].copy()

df_churned = df_churned.drop(columns=['Churn','WarehouseToHome'])
df_churned.info()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_churned)
scaled_df = pd.DataFrame(scaled_data, columns=df_churned.columns)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])

explained_variance = pca.explained_variance_ratio_
print(explained_variance)
```

### 🔸 K-Means Clustering + Evaluation

- Applied the **Elbow Method** to determine the optimal number of clusters.
- Selected **k = 4** clusters.
- Performed clustering on the top 3 PCA components.
- **Silhouette Score**: `0.4802` → indicates **moderate clustering structure**.

```python
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
```

```python
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(pca_df[['PC1', 'PC2', 'PC3']])
pca_df['Cluster'] = clusters
df_churned['Cluster'] = clusters
df_churned
```

<p align="center">
  <img src="https://drive.google.com/uc?id=1dc5UjRhUiowP0SMV9kL7UMfHy6KzubjM" width="500"/>
</p>

### 🔹 Cluster Analysis:

```python
df_churned[['CashbackAmount',
            'Tenure',
            'Complain',
            'SatisfactionScore',
            'PreferedOrderCat_Grocery',
            'PreferedOrderCat_Laptop & Accessory',
            'PreferedOrderCat_MobilePhone']].groupby(df_churned['Cluster']).mean().round(2)
```

**📈 Cluster Summary Table**

| Cluster | CashbackAmount | Tenure | Complain | SatisfactionScore | Grocery | Laptop & Accessory | Mobile Phone |
|--------:|----------------:|--------:|----------:|--------------------:|--------:|--------------------:|--------------:|
|   0     | 183.91          | 4.14    | 0.52      | 3.16                | 0.02    | 0.74                | 0.01          |
|   1     | 140.32          | 2.02    | 0.46      | 3.51                | 0.00    | 0.02                | 0.96          |
|   2     | 153.24          | 4.13    | 0.63      | 3.52                | 0.03    | 0.04                | 0.78          |
|   3     | 160.98          | 4.57    | 0.52      | 3.38                | 0.03    | 0.10                | 0.68          |

🔍 **Key Observations:**
- **Cluster 0**: Loyal users with longer tenure and high cashback, least likely to order via mobile; mostly interested in Laptop & Accessory products.
- **Cluster 1**: Newer users with lowest tenure and cashback, fewer complaints, but heavily mobile-oriented with preference for Mobile Phone category.
- **Cluster 2**: Moderate tenure and highest complaint rate, slightly more satisfied than others; strong mobile preference.
- **Cluster 3**: Most satisfied users with the highest tenure and broader product interest; prefer mobile but also show interest in laptops.

🔢 **Preferred Category**

```python
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

```

<p align="center">
  <img src="https://drive.google.com/uc?id=13s-I5OhQ5Rac9S2oHRUsCMvVlr2mDA0N" width="500"/>
</p>

- **Mobile Phone** is the dominant choice among churned users, especially in Clusters 1 and 2.

- **Laptop & Accessory** is highly preferred by **high-tenure users** in Cluster 0.

- **Grocery** remains a **minor segment** across all clusters.

---

## 🔎 Final Conclusion & Recommendations  

👉🏻 Based on the insights and findings above, we would recommend the stakeholder team to consider the following:  

📌 **Key Takeaways:**  
- **Early churn is critical**: Most users churn within the first 1–2 months. Onboarding and early engagement are key to retention.

- **Top churn drivers**: Short **tenure**, low **cashback**, and **complaints** are the strongest predictors of churn.

- **Complaints triple churn risk**: Fast and effective complaint handling is essential.

- **4 churn segments identified**:
  - **Cluster 0**: Loyal, tech-focused (prefer laptops), high cashback
  - **Cluster 1**: New, mobile-first, lowest cashback
  - **Cluster 2**: Moderate tenure, highest complaints
  - **Cluster 3**: Most satisfied, broad interests

- **Mobile Phone orders dominate churned users**, but long-tenure users prefer **Laptop & Accessory**.

✅ **Recommendations:**

✔️ **Enhance Early Engagement:** Improve onboarding, offer early incentives, and personalize initial experiences.

✔️ **Boost Cashback for Retention**: Strategically increase cashback offers for medium-tenure users to retain them.

✔️ **Address Complaints Promptly**: Prioritize fast, empathetic complaint resolution processes.

✔️ **Product Category Targeting**: Focus on high-churn segments who prefer Mobile Phones and Laptops. Deprioritize Grocery for churn prevention efforts.




