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

## 1️⃣ Data Cleaning & Preprocessing
- 🔹 **Load Dataset**  
  Imported dataset (`churn_prediction.csv`) and previewed the first few rows using `df.head()` to verify structure.

- 🔹 **Check Data Types & Values**  
  - Separated numerical and categorical columns using `df.select_dtypes()`
  - Inspected overall structure with `df.info()` and summary statistics using `df.describe()`
  - Checked unique values in each categorical column for understanding data distribution

- 🔹 **Standardize Categorical Labels**  
  - Cleaned inconsistent category labels (e.g., mapping `'phone'` → `'Mobile Phone'`, `'COD'` → `'Cash on Delivery'`)

- 🔹 **Check for Missing Values**  
  - Calculated missing counts and percentages  
  - Found several columns (e.g., `DaySinceLastOrder`, `Tenure`, `CouponUsed`) with missing data  
  - Handled missing values:
    - Used **mean** imputation for `HourSpendOnApp`  
    - Used **median** imputation for other numerical columns with missing values

- 🔹 **Check for Duplicates**  
  Verified that there are **no duplicated rows** in the dataset using `df.duplicated().sum()`

## 2️⃣ Churn Prediction – Supervised Learning

### 🔹 Split Dataset: Divide the data into training and test sets

### 🔹 Normalize Features: Scale numerical values for better model performance

### 🔸 Models Training:

#### **🤖 Logistic Regression**

#### **🧭 K-Nearest Neighbors (KNN)**

#### **🌲 Random Forest**

### 🔹Feature Importance (via Random Forest)

## 3️⃣ Churn Segmentation – Unsupervised Learning

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
