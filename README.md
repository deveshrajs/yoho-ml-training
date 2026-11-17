# Applied Machine Learning & Systems Architecture Portfolio

An end-to-end applied machine learning and data systems engineering repository covering non-linear tree ensembles, cost-sensitive learning for extreme class imbalance, leak-free MLOps production pipelines, automated data auditing, and enterprise AI solution architectures.

---

## Repository Overview & Curriculum Structure

```
yoho-ml-training/
├── week2-day2-trees-rf/                  # Tree Models & Ensemble Mechanics
│   ├── wine_trees_rf.ipynb               # Decision tree tuning, bagging, & ExtraTrees
│   └── wine_tree.png                     # Rendered decision tree flowchart
│
├── week2-day4-metrics-roc/               # Imbalanced Learning & Metric Optimization
│   ├── imbalanced_metrics_roc.ipynb      # Cost-sensitive classification & threshold search
│   └── roc_pr_curves.png                 # Comparative ROC & Precision-Recall curves
│
├── week2-day5-churn-project/             # End-to-End MLOps Customer Churn Pipeline
│   ├── data/                             # Raw customer subscription records
│   ├── models/                           # Serialized production pipeline artifacts (.joblib)
│   ├── notebooks/                        # EDA, tuning, explainability, & validation
│   │   └── churn_pipeline.ipynb
│   ├── src/                              # Modular production source code
│   │   ├── features.py                   # Custom scikit-learn feature transformer
│   │   └── predict_new.py                # Standalone batch inference CLI script
│   ├── eda_5_charts.png                  # Multi-variable exploratory data analysis
│   ├── feature_importance.png            # XGBoost MDI & feature attribution chart
│   ├── predictions.csv                   # Scored batch output with risk tiering
│   ├── report.md                         # Executive one-page business report
│   └── roc_comparison.png                # Multi-model CV ROC curve benchmark
│
├── week3-day1-fundamentals-audit/        # Data Quality Sanitization & Feasibility
│   ├── audit_and_decisions.md            # Client-ready data audit & feasibility table
│   ├── customers_raw.csv                 # Unsanitized input dataset with anomalies
│   ├── customers_clean.csv               # Sanitized dataset with quality indicator flags
│   ├── data_quality_audit.py             # Automated audit, cleaning, & learning curve script
│   └── sample_size_curve.png             # Generalization performance vs. sample volume
│
├── week3-day3-decision-proposals/        # Applied Architecture & Client Proposals
│   ├── decision_template.md              # 8-problem enterprise architectural decision matrix
│   ├── proposal_1_support_tickets.md     # NLP Support Ticket Triage (Linear/LightGBM)
│   ├── proposal_2_late_invoices.md       # B2B Invoice Late-Payment Early Warning (XGBoost)
│   └── proposal_3_hr_policy_rag.md       # Enterprise HR Policy Q&A Assistant (Hybrid RAG)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Module Breakdown

### 1. Decision Trees & Ensemble Methods (`week2-day2-trees-rf`)
* **Focus:** Non-linear decision rules, splitting metrics (Gini vs. Shannon Entropy), and variance reduction through Bootstrap Aggregation (Bagging).
* **Dataset:** Multiclass Wine Recognition Dataset (178 chemical samples, 13 features).
* **Key Implementations:**
  * 5-fold cross-validated grid search tuning `max_depth` and `min_samples_leaf`.
  * Visualized decision tree logic (`wine_tree.png`) demonstrating root splitting along `proline` thresholds.
  * Out-of-Bag (OOB) scoring in Random Forests, evaluating ensemble stabilization across `n_estimators` ($10$ to $300$).
  * Architectural comparison against Extremely Randomized Trees (`ExtraTreesClassifier`).

### 2. Imbalanced Classification & Performance Evaluation (`week2-day4-metrics-roc`)
* **Focus:** Addressing extreme 95:5 class skew without data leakage.
* **Key Implementations:**
  * Proof of the **Accuracy Paradox**: Majority-class baseline yields 95% accuracy while failing 100% of minority detections ($0.0$ Recall, $0.0$ F1).
  * Cost-sensitive loss weighting (`class_weight='balanced'` and `scale_pos_weight`) to penalize minority misclassifications $\sim 19\times$ higher.
  * Side-by-side analysis of **ROC Curves vs. Precision-Recall (PR) Curves** (`roc_pr_curves.png`), demonstrating why PR curves are essential when True Negatives dominate.
  * Custom threshold optimizer function satisfying a strict minimum Recall constraint ($\ge 80\%$) while maximizing precision.

### 3. Production Customer Churn Pipeline (`week2-day5-churn-project`)
* **Focus:** Industrial-grade end-to-end churn prediction on the IBM Telco dataset (7,043 subscribers).
* **Architecture:**
  * **Zero-Leakage Holdout:** 20% test partition (1,409 accounts) isolated *prior* to EDA, imputation, or scaling.
  * **Custom Scikit-Learn Transformer:** `TelecomFeatureEngineer` computing lifecycle cohorts (`tenure_cohort`), spend velocity (`effective_monthly_spend`), rate drift (`monthly_rate_delta`), and service bundle depth.
  * **Leak-Free Preprocessing:** `ColumnTransformer` with median imputation, standard scaling, and one-hot encoding with unknown handling.
  * **Hyperparameter Optimization:** `RandomizedSearchCV` on XGBoost and Logistic Regression across Stratified 5-Fold CV.
  * **Threshold Calibration:** Probability threshold calibrated at `0.5495` on out-of-fold predictions to capture $\ge 75\%$ of churners.
  * **Held-Out Test Results:** Test ROC-AUC of **0.8469**, Test F1 of **0.6370**, capturing **76.5% of churners** ($TN=797, FP=238, FN=88, TP=286$).
  * **Inference CLI:** Standalone `src/predict_new.py` scoring raw customer batches into tiered risk classifications (`High Risk`, `Medium Risk`, `Low Risk`).

### 4. Data Quality Audit & ML Feasibility (`week3-day1-fundamentals-audit`)
* **Focus:** Automated anomaly detection, data sanitization, and statistical feasibility analysis.
* **Key Implementations:**
  * Automated audit pipeline identifying whitespace, mixed casing, non-physical outlier ages (`-4`, `245`), and truncated phone records.
  * Value preservation via binary indicators (`email_missing_flag`, `valid_phone_flag`) rather than data deletion.
  * Decision framework evaluating when ML is appropriate vs. deterministic alternatives (SQL, Python functions, RAG).
  * Empirical sample size experiment (`sample_size_curve.png`) establishing model stability requirements ($\sim 500+$ clean samples per class).

### 5. Enterprise AI System Architecture Proposals (`week3-day3-decision-proposals`)
* **Focus:** Technical proposals and decision matrices for client engineering initiatives.
* **Key Deliverables:**
  * **Decision Matrix:** Categorization of 8 real-world enterprise engineering problems by input/output type, constraints, and architecture.
  * **Proposal 1:** NLP Support Ticket Triage Engine (TF-IDF + LinearSVC/LightGBM) with dual-threshold routing gates.
  * **Proposal 2:** Predictive B2B Invoice Late-Payment Early Warning System (XGBoost + SHAP) with tiered collections integration.
  * **Proposal 3:** Enterprise Policy Intelligence Q&A System (Hybrid RAG + RBAC Metadata Filtering) with token cost economics.

---

## Installation & Quickstart

### 1. Environment Setup
```bash
# Clone the repository
git clone <YOUR_REPOSITORY_URL>
cd yoho-ml-training

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Batch Churn Inference
To generate predictions for new customer records using the trained production model:
```bash
python week2-day5-churn-project/src/predict_new.py \
  --input week2-day5-churn-project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --output week2-day5-churn-project/predictions.csv
```

### 3. Running Data Quality Audit
```bash
cd week3-day1-fundamentals-audit
python data_quality_audit.py
```
