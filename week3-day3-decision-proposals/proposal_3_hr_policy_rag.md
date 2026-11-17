# Proposal 3: Enterprise Policy Intelligence & Compliance Q&A System

## 1. Executive Summary & Problem Framing
Navigating complex internal organizational policies (e.g., Leave Entitlement, International Travel Per Diems, POSH Compliance, Parental Benefits, IT Security Handbooks, Annual Appraisal Schedules) creates substantial friction across growing enterprises. HR Business Partners routinely spend 25% of their daily bandwidth answering repetitive, basic policy questions that are technically documented across hundreds of pages of disparate PDF files. Traditional keyword intranet search engines frequently fail because employees use conversational colloquial phrasing (e.g., *"How many unused vacation days roll over to next year?"*) that does not match formal administrative terminology (*"Privilege Leave Encashment & Carry-Forward Caps"*).

## 2. Recommended Engineering Architecture
We recommend designing a secure **Retrieval-Augmented Generation (`RAG + LLM`) System** utilizing hybrid sparse-dense vector retrieval and strict citation guardrails:
* **Why Not Parameter Fine-Tuning?** Fine-tuning an LLM does not guarantee factual recall of exact numbers (e.g., daily allowance caps), cannot enforce role-based access control (RBAC), and requires expensive parameter re-training every time an HR policy is updated.
* **Why Hybrid Retrieval (Dense Embeddings + BM25)?** Pure dense vector semantic search occasionally misses exact clause references or specific policy numbers, while pure keyword search fails on conversational natural language questions. A reciprocal rank fusion (RRF) hybrid retriever guarantees both conceptual understanding and exact term precision.

## 3. Data Ingestion, Governance & RBAC Requirements
* **Document Source Corpus:** Master HR policy documents in native PDF, Word, and Markdown formats, partitioned with semantic hierarchical chunking (preserving chapter, section, and sub-clause hierarchies).
* **Metadata & Role-Based Access Control (RBAC):** Every document chunk is tagged with access control metadata (`allowed_grades`, `department`, `country_jurisdiction`, `effective_date`). At query time, vector retrieval enforces metadata filtering to ensure employees cannot access confidential executive bonus structures or location-inapplicable benefits.
* **Continuous Synchronization:** Automated ingestion webhook triggered upon document revision approvals in Google Drive/SharePoint, immediately updating the vector index and archiving deprecated clauses.

## 4. Evaluation Framework & Operational Metrics
* **Retrieval Precision@3 $\ge 88\%$:** The authoritative policy clause must appear within the top 3 retrieved context passages.
* **Faithfulness Score $\ge 96\%$:** (Measured via automated RAGAS evaluation framework) Zero generated statements may exist outside the retrieved document passages; any claim lacking explicit source backing is treated as a critical defect.
* **Answer Relevancy $\ge 92\%$:** The generated response directly addresses the employee's intent with exact policy page and clause citations.

## 5. Implementation Roadmap (7-Day Delivery Plan)
* **Days 1–2: Ingestion & RBAC Preprocessing:** Build document parsers preserving table structures, create clean semantic chunks (512 tokens with 64 token overlap), and tag RBAC metadata.
* **Days 3–4: Hybrid Retrieval Engine:** Deploy embedded vector storage (pgvector / Qdrant) alongside BM25 indexing; calibrate Reciprocal Rank Fusion (RRF) weights.
* **Day 5: Guardrails & Citation Formatting:** Engineer system prompt guardrails mandating exact document/clause citation hyperlinks and an explicit refusal prompt (*"I cannot find this information in official HR policies—please contact your HRBP directly"*).
* **Days 6–7: Benchmark Evaluation & Chatbot Deployment:** Validate system performance against a golden evaluation dataset of 60 verified HR Q&A pairs using RAGAS, followed by Slack/Microsoft Teams app integration.

## 6. System Trade-Offs & Human Safety Escalations
* **Hallucination Prevention vs. Answer Speed:** Implementing double-pass faithfulness verification adds approximately 800ms of latency (bringing total response time to ~2.2 seconds). In HR compliance, factual accuracy is strictly non-negotiable compared to sub-second conversational latency.
* **Mandatory Human Escalation:** Inquiries regarding disciplinary proceedings, formal grievances, termination procedures, medical leaves, or instances where retrieval similarity falls below confidence threshold ($< 0.70$) automatically bypass the AI and generate an expedited confidential HRBP support ticket.

## 7. Monthly Operating Cost Model (Token Economics)
* **Assumptions:** 250 Enterprise Employees $\times$ 6 queries/month = **1,500 queries/month**.
* **Token Usage per Query:** ~2,000 input context tokens (prompt + retrieved chunks) + ~250 output response tokens.
  * Monthly Input Tokens: $1,500 \times 2,000 = 3,000,000$ tokens ($\approx \$0.45$ on Gemini Flash / GPT-4o-mini).
  * Monthly Output Tokens: $1,500 \times 250 = 375,000$ tokens ($\approx \$0.25$).
* **Vector Indexing & Storage:** 1,500 query embeddings + 600 indexed policy chunks $\approx \$0.05/\text{month}$.
* **Total Estimated Running Cost:** **$\approx \$2.00\text{--}\$4.50\text{ per month}$**, delivering near-zero marginal operational cost compared to dozens of saved HR specialist hours.
