# Proposal 1: Intelligent Support Ticket Triage & Routing System

## 1. Executive Summary & Problem Framing
Customer support operations at high growth enterprises face substantial operational friction during incoming ticket triage. Tier-1 support specialists currently dedicate an estimated 15 to 20 minutes of every working hour manually reviewing, categorizing, and assigning incoming customer queries across 8 target queues (Billing & Invoicing, Account Authentication, Technical Bugs, Feature Requests, Service Outages, Subscription Changes, Security & Compliance, General Inquiries). During peak transaction periods, manual routing creates severe backlog latency, increases First Response Time (FRT) by up to 45%, and introduces routing errors that frustrate enterprise customers.

## 2. Recommended Engineering Architecture
We recommend deploying an optimized **Supervised Natural Language Classification Pipeline (TF-IDF Vectorization with Linear Support Vector Classifier / LightGBM)** integrated into the support ticketing webhook:
* **Why Not Static Keyword Matching?** Keyword rules fail abruptly when handling linguistic nuances, synonyms, typos, or conversational queries (e.g., distinguishing between *"can't log into portal"* and *"need invoice for my accounting team"* when both contain identical company portal references). Maintaining exhaustive rule dictionaries becomes an engineering bottleneck.
* **Why Not a Large Language Model (LLM)?** Invoking a generative LLM for fixed 8-way classification introduces unnecessary latency (1,500ms to 3,000ms per request) and compounding API costs ($150–$300/month at scale). In contrast, a trained linear classifier executes inference in sub-10 milliseconds directly on existing server infrastructure with zero recurring token costs.

## 3. Data Infrastructure & Requirements
* **Historical Volume Requirement:** A clean export of 5,000 to 10,000 historical resolved customer tickets, ensuring a minimum of 600 to 1,200 verified examples per ticket category.
* **Feature Schema:** Raw text fields including `ticket_subject`, `ticket_body_text`, `submission_channel`, and the historical verified `resolved_category_id`.
* **Governance & Preprocessing:** Automated PII redaction pipeline removing credit card numbers, phone numbers, passwords, and sensitive client identifiers before vectorization.

## 4. Evaluation Criteria & SLAs
* **Primary Optimization Metric:** **Macro F1-Score $\ge 0.86$** (ensuring equitable classification performance across both high-volume categories like Billing and critical low-volume categories like Security Incidents).
* **Operational SLA:** Overall routing accuracy of $\ge 92\%$ on tickets meeting the automated dispatch confidence threshold, with inference latency $< 15\text{ms}$.

## 5. Implementation Roadmap (5-Day Delivery Plan)
* **Day 1: Ingestion & Governance:** Extract historical ticketing logs, execute PII scrubbers, perform exploratory category balance audits, and validate the 8-class taxonomy.
* **Day 2: NLP Pipeline Engineering:** Build text cleaning pipelines (tokenization, domain stop-word filtering, n-gram extraction) and establish baseline classification benchmarks (TF-IDF + Logistic Regression, Naive Bayes, LinearSVC).
* **Day 3: Hyperparameter Optimization:** Conduct 5-fold cross-validated grid search tuning vectorizer n-gram ranges, sublinear term frequencies, and regularization penalty parameters.
* **Day 4: Decision Boundary & Rejection Calibration:** Establish dual-threshold confidence gates. Tickets predicted with confidence $\ge 0.80$ route automatically; ambiguous tickets route to a human triage queue.
* **Day 5: Webhook API & Monitoring Rollout:** Package the serialized pipeline into a containerized FastAPI service with real-time confidence telemetry and drift logging.

## 6. Key System Trade-Offs
* **Coverage vs. Routing Precision:** By enforcing a confidence cutoff of $0.80$, the system autonomously routes approximately $78\%\text{--}83\%$ of all inbound tickets with $>93\%$ accuracy, while redirecting ambiguous or multi-topic queries to human agents. This completely prevents costly misdirection while eliminating over three-quarters of manual triage work.

## 7. Failure Modes & Production Guardrails
* **Multi-Intent Customer Queries:** Customers occasionally report two distinct issues in a single ticket (e.g., reporting a technical bug while inquiring about an overdue invoice). When the difference between the top-two class probabilities is $< 0.15$, the ticket is automatically tagged as `Multi-Intent` and routed to a human team lead.
* **Vocabulary & Concept Drift:** When new product features or external disruptions emerge, vocabulary distributions shift. We establish automated logging of human re-assignment actions, triggering automated monthly retraining whenever human reclassification exceeds $8\%$.
