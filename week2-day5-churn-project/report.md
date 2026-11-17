# Telecom Customer Churn Prediction – Executive Project Report

## 1. Executive Summary & Problem Framing
Customer attrition in telecommunications directly impacts Monthly Recurring Revenue (MRR) and profitability. Given that acquiring a new telecom subscriber costs approximately \$300 while proactive retention campaigns average \$35–\$50 per customer, accurate early identification of churn-prone accounts is critical. This project engineers a machine learning pipeline to detect churn risks before service cancellation, empowering retention teams to intervene proactively.

## 2. Dataset Architecture & Split Integrity
* **Dataset Profile:** IBM Telco Customer Churn dataset comprising **7,043 subscriber records** and **20 raw attributes** with an overall baseline churn rate of **26.54%** (1,869 churners vs. 5,174 retained).
* **Strict Holdout Protocol:** To eliminate data snooping and target leakage, an untouched stratified **20% held-out test partition (1409 records: 1035 retained, 374 churned)** was isolated prior to exploratory data analysis, transformation fitting, or feature scaling.

## 3. Engineering & Modeling Methodology
* **Domain Feature Engineering:** Implemented within a leak-free `TelecomFeatureEngineer` transformer:
  1. `tenure_cohort`: Stratified lifecycle stages (`0-12m`, `13-24m`, `25-48m`, `49-72m`).
  2. `effective_monthly_spend`: Normalized historical rate (`TotalCharges / tenure`).
  3. `monthly_rate_delta`: Spend inflation rate (`MonthlyCharges - effective_monthly_spend`).
  4. `addon_services_count`: Count of active digital ecosystem add-ons (Security, Backup, Protection, Support, TV, Movies).
  5. `has_support_security`: Binary indicator representing technical and security retention anchors.
* **Pipeline Preprocessing:** Integrated via `ColumnTransformer` with median imputation for numeric values, standard scaling, and one-hot encoding for nominal categories with `handle_unknown='ignore'`.
* **Model Benchmark & Hyperparameter Search:** Evaluated Stratified Dummy Baseline, Cost-Sensitive Logistic Regression, Balanced Random Forest, and Scaled XGBoost across 5-fold stratified cross-validation. Hyperparameters were tuned using `RandomizedSearchCV` optimizing for ROC-AUC.
* **Decision Threshold Calibration:** Calibrated operating probability threshold on out-of-fold cross-validation probabilities to satisfy the retention department's SLA constraint of capturing **$\ge 75\%$ of all churners**.

## 4. Final Held-Out Test Results
Evaluation performed on the untouched 20% test partition (1409 customers):
* **Production Model:** Tuned Gradient Boosted Classifier (XGBoost with `scale_pos_weight`)
* **Calibrated Operating Threshold:** **`0.5495`** (lowered from default 0.50 to capture at-risk subscribers)
* **Test ROC-AUC:** **`0.8469`** (strong class separation across all thresholds)
* **Test F1-Score:** **`0.637`**
* **Test Recall:** **`76.5%`** (286 out of 374 actual churners identified)
* **Test Precision:** **`54.6%`**
* **Confusion Matrix:**
  * **True Negatives (TN):** `797` (Correctly identified loyal customers)
  * **False Positives (FP):** `238` (Retention offers extended to non-churners)
  * **False Negatives (FN):** `88` (Unflagged churners who left)
  * **True Positives (TP):** `286` (Churners flagged for intervention)

```
                     Predicted No Churn    Predicted Churn
Actual No Churn (0)         797                     238            
Actual Churn (1)            88                      286            
```

## 5. Key Risk Drivers (Feature Importance)
1. **Contract Structure (`Contract_Month-to-month`):** Month-to-month subscribers exhibit over $3.2\times$ higher churn propensity compared to 1- or 2-year commitments due to zero contract exit frictions.
2. **Tenure Window (`tenure_cohort` / `tenure`):** High-risk churn concentrates heavily in months 1–12; accounts that cross the 24-month mark exhibit long-term retention stability.
3. **Internet Infrastructure (`InternetService_Fiber optic`):** Fiber optic subscribers pay premium rates (\$70–\$100/mo) and churn at elevated rates when encountering billing spikes or unbundled services.
4. **Service Stickiness (`OnlineSecurity`, `TechSupport`):** Absence of security and technical support services sharply increases subscriber mobility.
5. **Rate Drift (`monthly_rate_delta`):** Customers experiencing sudden bill increases relative to their historical average show accelerated cancellation spikes.

## 6. Strategic Business Recommendations & Operational Rollout
1. **Targeted Annual Contract Migration:** Identify month-to-month Fiber optic subscribers with predicted churn probability $\ge 0.55$ in their first 12 months, offering a \$10/month bill credit in exchange for converting to a 12-month annual contract.
2. **Ecosystem Stickiness Bundling:** Automatically grant a 3-month complimentary trial of `OnlineSecurity` and `TechSupport` to newly onboarded subscribers during their first billing cycle.
3. **Automated Batch CRM Integration:** Execute `src/predict_new.py` on a weekly automated batch schedule to feed customer churn risk scores and tiered labels (`High Risk`, `Medium Risk`, `Low Risk`) directly into HubSpot/Salesforce for automated outreach.
