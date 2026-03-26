# 💧 Water Pump Status Prediction (Machine Learning Project)
#### Link to the Datab: Click [here]((https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/).) <br>

## 🚀 Overview
This project focuses on predicting the operational status of water pumps using machine learning.  
The goal is to classify pumps into:

- ✅ Functional  
- ⚠️ Functional Needs Repair  
- ❌ Non-Functional  

This is a real-world classification problem with strong impact on infrastructure, maintenance planning, and resource allocation.

---

## 🧠 Problem Statement
Access to clean water is critical. Many water pumps fail due to poor maintenance or environmental conditions.

👉 The challenge:
Build a model that predicts pump status based on features like location, installer, water quality, and construction details.

---

## 👥 Team
**HydroDominion** (3-member ML team)

---
## The columns in this dataset
* amount_tsh - Total static head (amount water available to waterpoint) <br>
* date_recorded - The date the row was entered<br>
* funder - Who funded the well<br>
* gps_height - Altitude of the well<br>
* installer - Organization that installed the well<br>
* longitude - GPS coordinate<br>
* latitude - GPS coordinate<br>
* wpt_name - Name of the waterpoint if there is one<br>
* num_private -<br>
* basin - Geographic water basin<br>
* subvillage - Geographic location<br>
* region - Geographic location<br>
* region_code - Geographic location (coded)<br>
* district_code - Geographic location (coded)<br>
* lga - Geographic location<br>
* ward - Geographic location<br>
* population - Population around the well<br>
* public_meeting - True/False<br>
* recorded_by - Group entering this row of data<br>
* scheme_management - Who operates the waterpoint<br>
* scheme_name - Who operates the waterpoint<br>
* permit - If the waterpoint is permitted<br>
* construction_year - Year the waterpoint was constructed<br>
* extraction_type - The kind of extraction the waterpoint uses<br>
* extraction_type_group - The kind of extraction the waterpoint uses<br>
* extraction_type_class - The kind of extraction the waterpoint uses<br>
* management - How the waterpoint is managed<br>
* management_group - How the waterpoint is managed<br>
* payment - What the water costs<br>
* payment_type - What the water costs<br>
* water_quality - The quality of the water<br>
* quality_group - The quality of the water<br>
* quantity - The quantity of water<br>
* quantity_group - The quantity of water<br>
* source - The source of the water<br>
* source_type - The source of the water<br>
* source_class - The source of the water<br>
* waterpoint_type - The kind of waterpoint<br>
* waterpoint_type_group - The kind of waterpoint<br>

## 📊 Dataset
The dataset consists of:
- **Training Values** (features)
- **Training Labels** (target: `status_group`)
- **Test Values** (for prediction)

Total features: 40+  
Total samples: ~59,000  

---

## 🧹 Data Preprocessing
Key steps:
- Handling missing values
- Converting `date_recorded` → year, month
- Fixing mixed data types (str, bool, float)
- Encoding categorical variables using `OrdinalEncoder`
- Dropping low-value columns (e.g., `scheme_name`, `recorded_by`)

---

## ⚙️ Models Used
We implemented and compared 6 models:
1. Logistic Regression  
2. Decision Tree  
3. Random Forest 🌳  
4. XGBoost ⚡  
5. LightGBM 🚀  
6. CatBoost 🐱  

---

## 🏆 Results (Validation Performance)
| Model               | Accuracy |
|--------------------|---------|
| Logistic Regression | ~57% |
| Decision Tree       | ~59% |
| Random Forest       | ~69% ✅ |
| XGBoost             | (improving) |
| LightGBM            | (in progress) |
| CatBoost            | (in progress) |

👉 **Random Forest performed best so far**

---

## 📉 Challenges Faced
- Mixed data types (bool + string)
- Missing values in categorical features
- Date format issues (`YYYY-MM-DD`)
- Encoding errors across train/test sets
- Model compatibility issues (XGBoost, LightGBM)

---

## 🧪 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

## 📁 Project Structure
water-pump-prediction-ml/
│
├── data/
├── notebooks/
├── src/
├── models/
├── submission.csv
├── README.md


water-pump-prediction-ml/
│
├── data/
├── notebooks/
├── src/
├── models/
├── submission.csv
├── README.m


📈 Future Improvements
Feature engineering (pump age, region grouping)
Hyperparameter tuning
Cross-validation
Model ensembling
Deployment (API or dashboard)

💡 Key Takeaway
The real power of machine learning is not just in models —
it’s in how well you understand and prepare your data.

📬 Contact
Feel free to connect or collaborate!
