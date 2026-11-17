# Week 3 – Day 1: Data Quality Audit, Cleansing & ML Feasibility Assessment

## Task 1: Client-Readable Data Quality Report
1. **Initial Audit Findings:** Ingested 32 raw customer accounts and identified 3 redundant duplicate submissions, inconsistent city text casing and whitespace (`CHENNAI `, `chennai`, ` Coimbatore`), and a 15.6% missing email rate.
2. **Outlier & Schema Anomalies:** Age validation identified non-physical values (`-4` and `245` years), while phone field parsing flagged incomplete 4-digit numbers violating standard 10-digit telephony schemas.
3. **Deduplication & Normalization:** Purged identical duplicate records, trimmed surrounding whitespace, and normalized regional city names into standard Title Case across all entries.
4. **Data Preservation Strategy:** Rather than dropping customer profiles missing emails or containing truncated phone numbers, we introduced binary verification flags (`email_missing_flag`, `valid_phone_flag`) and imputed invalid ages with the median customer age (31.0 years).
5. **Production Readiness:** The resulting 29-row cleansed dataset (`customers_clean.csv`) conforms 100% to production schema standards and is safe for downstream CRM routing and predictive modeling.

---

## Task 2: Machine Learning Feasibility Decision Matrix

Evaluating when statistical machine learning is the correct tool versus deterministic programming or LLMs:

| Business Problem | Recommended Architecture | Architectural Justification |
| :--- | :--- | :--- |
| **(a) Calculate GST on an invoice** | **Python Rule / Function** | Tax calculations follow statutory deterministic percentage formulas (`subtotal * rate`). Using probabilistic ML introduces rounding errors and legal compliance risks. |
| **(b) Find the order status for order #1042** | **SQL Query** | Looking up an exact record by primary key (`WHERE order_id = 1042`) is a deterministic relational database operation requiring exact zero-latency retrieval. |
| **(c) Predict which customers will leave next month** | **Machine Learning (ML)** | Churn is governed by non-linear behavioral interactions across tenure, spend drift, and contract terms that cannot be captured by rigid static rules. |
| **(d) Answer questions from a 60-page policy PDF** | **RAG + LLM** | Requires semantic language understanding, document chunk retrieval, and synthesis across unstructured natural language text. |
| **(e) Flag invoices above 1 Lakh for approval** | **Python Rule Engine** | A fixed monetary constraint (`invoice_amount > 100,000`) is an explicit company business policy best enforced by a deterministic `if/else` condition. |

---

## Analysis: Sample Size Requirements & Model Generalization (3 Lines)
1. **Low Sample Regime (<100 rows):** At sample sizes below 100 observations, test generalization is low and volatile because models easily overfit to idiosyncratic noise rather than genuine signal.
2. **Knee of the Curve (250–500 rows):** Accuracy gains scale steeply between 100 and 500 samples, reaching an operational plateau where marginal sample returns begin to diminish.
3. **Production Readiness Rule:** Before architecting bespoke supervised ML pipelines for enterprise clients, teams must verify that at least ~500 to 1,000 clean, representative labeled examples per target class are accessible.

---

## Bonus Architectural Study: Business Process Better Solved with SQL & Rules
* **Employee Leave Balance & Entitlement Validation:**
  * Many organizations attempt to deploy conversational generative AI to answer employee questions like *"Can I take 3 days of sick leave next Thursday?"*
  * While the query interface is conversational, the business logic is entirely deterministic: validating remaining leave quotas in an HRMS database table, verifying non-overlapping company holidays, and checking manager approval policies.
  * Attempting to resolve leave eligibility using an LLM or statistical model introduces severe hallucination risks and latency. A structured **SQL query combined with Python business rules** delivers 100% mathematical precision, sub-10ms response latency, complete auditability, and zero API token overhead.
