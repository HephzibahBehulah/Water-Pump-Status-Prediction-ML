# Sprint 1: Exploratory Data Analysis
# 💧 Water Pump Status Prediction (Machine Learning Project)
#### Link to the Datab: (https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/)
### 👥 Team: **Hydro Dominion**

## 📌 Project Overview
Access to clean water is a critical global challenge. In many regions, water pumps fail due to poor maintenance, environmental conditions, or installation quality.
This project aims to **predict the functionality status of water pumps** using machine learning.

🔍 In **Sprint 1**, our focus is:
* Understanding the dataset
* Cleaning and preparing the data
* Extracting insights through Exploratory Data Analysis (EDA)

---

## 🧪 Dataset Description
We used three datasets:
* `Training Set Values.csv` → Features
* `Training Set Labels.csv` → Target (`status_group`)
* `Test Set Values.csv` → Final prediction dataset

After merging:
* 📊 Total rows: **59,400**
* 🎯 Target classes:

  * Functional
  * Functional needs repair
  * Non-functional

---

## 🧠 Hypotheses

We defined the following hypotheses to guide our analysis:

# H1: Pump type influences functionality  
# H2: Installer affects pump reliability  
# H3: Geography (region/basin) impacts pump performance  
# H4: Older pumps are more likely to fail  
# H5: Population served influences pump failure rate  
# H6: Water quality affects pump functionality
# H7: Maintenance frequency impacts pump reliability
# H8: Pump age and installer together influence failure rates
# H9: Certain pump types are more prone to failure in specific regions
# H10: Socioeconomic factors of the area affect pump maintenance and functionality

---

## ❓ Research Questions

* Do older pumps fail more often?
* Are pumps serving larger populations more likely to break?
* Are certain regions more prone to failures?
* Does installer quality affect reliability?
* Which pump types perform best?

---

## 🧹 Data Cleaning & Preparation

Key steps performed:

* ✅ Merged datasets on `id`
* ✅ Removed duplicates
* ✅ Handled missing values:

  * Categorical → `"missing"`
  * Numerical → `0`
* ✅ Converted `date_recorded` → extracted year & month
* ✅ Fixed mixed data types (converted to consistent formats)
* ✅ Dropped irrelevant columns:

  * `id`, `recorded_by`, `scheme_name`

---

## 📊 Exploratory Data Analysis (EDA)

We performed visual analysis to validate our hypotheses.

### 🔹 Target Distribution

* Majority of pumps are **functional**
* Minority class → *functional needs repair* ⚠️ (class imbalance)

---

### 🔹 Pump Age vs Functionality

* Older pumps show higher failure rates ✔️
* Strong support for **H4**

---

### 🔹 Geography (Region & Basin)

* Significant variation across regions
* Some regions have disproportionately high failures ✔️
* Supports **H3**

---

### 🔹 Installer Impact

* Clear differences between top installers
* Some installers consistently produce better outcomes ✔️
* Supports **H2**

---

### 🔹 Population Impact

* Higher population usage correlates with failures ✔️
* Supports **H5**

---

### 🔹 Pump Type Analysis

* Certain pump/extraction types perform better ✔️
* Supports **H1**

---

## 📈 Additional Analysis

* 📦 Outliers detected in `amount_tsh`
* 🔥 Correlation analysis performed for numeric features
* ⚠️ Dataset imbalance identified (important for modeling)

---

## 🧠 Key Insights

1. Older pumps are more likely to fail
2. Geography significantly impacts pump functionality
3. Installer quality plays a major role
4. Certain pump types are more reliable
5. High population usage increases failure risk
6. Dataset is imbalanced and requires handling in modeling

---

## 🛠️ Tools & Technologies

* Python (Pandas, NumPy)
* Visualization (Matplotlib, Seaborn)
* Power BI (for dashboard exploration)
* Git & GitHub (collaboration)

---

## 🎯 Sprint 1 Deliverables

✔ Hypotheses & research questions
✔ Cleaned and validated dataset
✔ Exploratory data analysis with visualizations
✔ Key insights for modeling
✔ Reproducible code

---

## 🚀 Next Steps (Sprint 2)

In the next sprint, we will:

* Build machine learning models:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * XGBoost
  * LightGBM
  * CatBoost
* Handle class imbalance
* Optimize performance
* Generate predictions for test dataset

---

## 🎤 Team Reflection

> “We didn’t just analyze data — we uncovered patterns that explain real-world system failures.
> These insights will directly guide our predictive models.”

## 💬 Final Note

Sprint 1 laid the **foundation**.

We now understand:

* what matters
* what affects pump failure
* and what features will drive prediction

Sprint 2 is where we **turn insight into intelligence** ⚡

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


---

