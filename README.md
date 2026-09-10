# Uber Support Copilot: Evidence-Grounded AI Support Agent

An evidence-grounded, production-grade AI support copilot built for the **Hiver SDE Intern Take-Home Assignment**. 

Built on **41,185 real reconstructed customer support conversations** from the public Twitter Customer Support dataset (`@Uber_Support`), the system classifies incoming customer inquiries, retrieves similar historical resolutions, drafts policy-grounded replies, and makes explicit, transparent escalation decisions with actionable reasons and risk indicators.

---

## 🏗️ System Architecture

The copilot operates as a multi-stage decision pipeline with a shared evidence and safety layer:

```
                  Incoming Customer Message & Thread Context
                                      │
                                      ▼
                        Turn Isolation & Preprocessing
                  (Filters #myfirstTweet, handles, setup noise)
                                      │
                                      ▼
                   TF-IDF Knowledge Index (41,185 cases)
                   ┌──────────────────┴──────────────────┐
                   │  (Strict Runtime Golden Exclusion)  │
                   ▼                                     ▼
      Top-3 Historical Cases                 Weighted Intent Classifier
      (Resolutions & Evidence)               (Domain-Stopword Suppressed)
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      │
                                      ▼
                        Risk & Escalation Engine
             ┌─────────────────────────────────────────────────┐
             │ • Safety / Conduct Triggers                     │
             │ • Payment Discrepancy & Dispute Triggers        │
             │ • Property Damage & Account Access Triggers     │
             │ • Intent Confidence & Retrieval Agreement       │
             │ • "Sticky" Escalation Invariants                │
             └────────────────────────┬────────────────────────┘
                                      │
                  ┌───────────────────┴───────────────────┐
                  ▼                                       ▼
          [DECISION: ESCALATE]                  [DECISION: AUTO-HANDLE]
        • Stated human-review reason          • Empathetic draft response
        • Tagged risk indicators              • Grounded in historical evidence
        • Specialist queue routing            • Zero hallucinated refunds/promises
```

---

## ⚡ Quickstart: Reproduce Results in Under 15 Minutes

The entire evaluation pipeline is fully automated, deterministic, and executes in **~2.5 minutes** on a standard laptop.

### 1. Environment Setup
```powershell
# Clone the repository and navigate to root
cd assignment-hiver

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

*(Optional)* Configure LLM generation in `.env`:
```ini
GROQ_API_KEY=your_groq_api_key_here
# or
NVIDIA_API_KEY=your_nvidia_api_key_here
```
> **Offline Reproducibility:** If no API key is provided, the system seamlessly uses its deterministic rule engine, vector retriever, and heuristic rubric judge. Everything runs 100% offline without third-party dependencies.

### 2. Run the Full Pipeline
```powershell
python scripts/run_final_pipeline.py
```
This executes all 11 phases in sequence:
1. `profile_data.py`: Corpus volume and taxonomy verification (41,185 cases).
2. `create_golden_set.py` & `golden_profile.py`: Stratified sampling of 200 holdout cases.
3. `leakage_audit.py`: Confirms 0 golden cases exist in the runtime retrieval corpus.
4. `tfidf_baseline.py`: Conventional Logistic Regression training (Macro-F1: 0.732).
5. `run_evaluation.py`: Evaluates proposed agent vs. trivial and simple baselines.
6. `retrieval_eval.py`: Information retrieval metrics (Recall@1, 3, 5, MRR).
7. `calibration.py`: Probability calibration & Expected Calibration Error (ECE: 0.074).
8. `run_judge.py`: Evaluates draft responses using the 5-dimension judge rubric.
9. `failure_analysis.py`: Extracts and categorizes top failure modes.
10. `requirements_matrix.py`: Traceability matrix against assignment brief.

### 3. Launch the Support Workspace Dashboard
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** to interact with the system:
* **Analyze Case:** Test incoming messages, select real edge cases from dropdown, and view real-time intent, confidence, decision badges, risk flags, and draft replies.
* **Evaluation Overview:** Inspect live metrics, baseline comparisons, and judge scores.
* **Golden Set Review:** Review the 200-ticket golden set, edit intent/decision labels, score responses, and save reviewed ground truth.

---

## 📊 Headline Benchmark Results

Evaluated on the **200-conversation Golden Set** with holdout conversation IDs strictly excluded from retrieval:

| System Architecture | Intent Macro-F1 | Overall Accuracy | Automation Coverage | Unsafe Auto-Handling Rate | Calibration Error (ECE) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Proposed Copilot (Retrieval + Policy)** | **0.708** | **72.5%** | **24.5%** | **3.2%** | **0.074** (7.4%) |
| **Simple Baseline (Nearest-Case TF-IDF)** | 0.421 | 40.0% | 0.0% | N/A (Manual) | 0.285 |
| **Trivial Baseline (Majority Intent + Escalate)** | 0.028 | 12.5% | 0.0% | 0.0% | N/A |

### Key Observations:
* **Overcoming the Trivial Baseline:** A majority-class predictor yields `0.028` Macro-F1 across 8 classes and offers 0% automation, sending 100% of tickets to human agents.
* **Overcoming the Simple Baseline:** 1-nearest-neighbor matching without domain weighting scored only `0.421` Macro-F1 because generic terms (`"driver"`, `"ride"`) dominate Uber conversations, pulling safety and payment issues into generic trip categories.
* **Calibrated Confidence:** Expected Calibration Error (ECE) dropped from 0.292 to **0.074**. The high-confidence bucket (0.8–1.0) contains **114 cases with 85.3% accuracy and 85.3% mean confidence**, demonstrating genuine alignment between predicted probabilities and real-world accuracy.
* **Retrieval Ranking:** Recall@1 = `0.400`, Recall@3 = `0.595`, Recall@5 = `0.660`, MRR = `0.503`.
* **LLM-as-Judge Response Quality:** Average **4.64 / 5.0** across Groundedness, Correctness, Completeness, Tone, and Safety (`reports/judge_scores.csv`).
* **Human-Judge Agreement Evidence:** Blinded human audit on 50 golden cases verified **100% agreement within ±1 point** (MAE = 0.284, exact agreement = 71.6%, safety correlation $r = 0.742$). See [REPORT.md](file:///c:/Users/Kiran%20Teja/Downloads/projects/assignment-hiver/REPORT.md#23-evaluation-harness--human-judge-agreement-evidence).

---

## 🎯 The Three Core Capabilities (Hiver Specifications)

### 1. Intent Classification
Derived from clustering and frequency analysis across the 41,185 Uber support conversations, the system uses a **compact 8-class taxonomy**:
1. `safety_or_driver_conduct`: Assault, harassment, dangerous driving, accidents, intoxication, physical threats.
2. `payment_problem`: Double charges, Paytm / card discrepancies, wallet issues, fare disputes.
3. `refund_request`: Cancellation fee refunds, post-cancellation charges, fee adjustments.
4. `lost_item`: Items left in vehicle (phone, wallet, keys, bag).
5. `account_or_login`: Deactivated accounts, locked profiles, password reset, auth/OTP errors.
6. `promo_or_coupon`: Promo codes not applying, discount vouchers, referral credit.
7. `trip_or_pickup`: Route discrepancies, driver no-shows, ETA delays, pickup location issues.
8. `other_or_unclear`: Vague queries, social media greetings, out-of-domain messages.

> **Non-Obvious Engineering:** Generic ride-hailing words like `"driver"`, `"ride"`, `"trip"`, and `"car"` appear in over 92% of all messages. The classifier uses weighted domain patterns where specific problem verbs (`charge double`, `paytm`, `unsafe`, `left my`) override generic vehicle nouns.

### 2. Evidence-Grounded Reply Drafting
* **Historical Retrieval:** Top-3 nearest historical resolutions from the 41k corpus are retrieved to ground response drafting.
* **Strict Non-Fabrication:** The model is forbidden from hallucinating refunds, specific payout amounts, or policy exceptions. For payment disputes, it drafts empathy and requests verifiable identifiers (Paytm transaction IDs, trip dates) while routing the ticket to an agent queue.
* **Deterministic Safety Invariants:** Output validation scans draft replies for compliance, prompt injections, and unsupported promises before rendering.

### 3. Escalation as a First-Class Citizen
Escalation is not a fallback—it is a primary output driven by explicit risk detection:
* **"Sticky" Escalation Invariant:** If deterministic safety checks detect safety incidents, property damage, or financial disputes, the ticket is **permanently locked to `ESCALATE`** (red warning badge). The generative LLM cannot override it to auto-handle.
* **Transparent Stated Reasons:** Every escalation details the exact trigger:
  - `Human review recommended because: account-specific payment dispute or review`
  - `Human review recommended because: safety or driver conduct incident`
  - `Human review recommended because: weak historical evidence`
* **Risk Indicators:** Transparently displayed as tags (e.g. `⚠️ account-specific payment dispute or review`, `⚠️ property damage claim`).

---

## 📁 Deliverables & Repository Map

| Deliverable | Location | Description |
| :--- | :--- | :--- |
| **Runnable Pipeline** | [`scripts/run_final_pipeline.py`](file:///c:/Users/Kiran%20Teja/Downloads/projects/assignment-hiver/scripts/run_final_pipeline.py) | Master pipeline reproducing all numbers in < 3 minutes. |
| **Golden Evaluation Set** | [`data/golden/uber_golden_set.csv`](file:///c:/Users/Kiran%20Teja/Downloads/projects/assignment-hiver/data/golden/uber_golden_set.csv) | 200 stratified real conversations (25/class) with review tracking. |
| **Evaluation Harness & Rubric** | [`reports/evaluation.json`](file:///c:/Users/Kiran%20Teja/Downloads/projects/assignment-hiver/reports/evaluation.json)<br>[`reports/judge_rubric.json`](file:///c:/Users/Kiran%20Teja/Downloads/projects/assignment-hiver/reports/judge_rubric.json) | Macro-F1, coverage, unsafe rates, and 5-dimension judge rubric. |
| **Comprehensive Report** | [`REPORT.md`](file:///c:/Users/Kiran%20Teja/Downloads/projects/assignment-hiver/REPORT.md) | Full report with problem framing, baselines, failure analysis, headline critique, and roadmap. |
| **Decision Log** | [`DECISION_LOG.md`](file:///c:/Users/Kiran%20Teja/Downloads/projects/assignment-hiver/DECISION_LOG.md) | 13 non-obvious engineering decisions and trade-offs. |
| **Interview Talking Points** | [`docs/interview_notes.md`](file:///c:/Users/Kiran%20Teja/Downloads/projects/assignment-hiver/docs/interview_notes.md) | Architecture rationale and live defense preparation. |

---

## 🔍 Golden Evaluation Set Sampling Protocol

* **Volume:** 200 multi-turn customer conversations extracted from the 41,185 Uber dataset.
* **Sampling Strategy:** Stratified uniformly with **25 examples per category** across all 8 canonical intents.
* **Why Stratified over Random?** Natural Twitter support volume follows a power-law distribution where routine queries account for >70% of messages. A random sample of 200 tickets would contain only 1–2 safety incidents. Stratification ensures rigorous statistical validation on rare, high-consequence edge cases (assault, overcharges, property damage).
* **Leakage Auditing:** Audited via `scripts/leakage_audit.py` to ensure holdout conversation IDs are strictly excluded from the search index at evaluation time.

---

## ⚠️ "What is Misleading About My Headline Number?" (Mandatory Section)

Support leadership must understand the nuances behind headline scores before deployment:

1. **Stratified vs. Production Class Distribution:** Our evaluation uses a balanced 25-ticket-per-intent distribution to guarantee safety rigor. In raw production traffic, >70% of messages are routine status queries. Consequently, real-world accuracy will appear higher, but safety recall is much more challenging to preserve.
2. **Offline Retrieval vs. Production Latency:** Offline evaluation over an in-memory index runs in milliseconds. In live deployment, network latency to external LLM providers (Groq / NVIDIA NIM) and rate limiting introduce operational friction not captured offline.
3. **Groundedness as a Proxy for CSAT:** Our LLM judge scores groundedness at **4.66/5**. However, a draft can be 100% policy-grounded (*"Please submit your fare dispute in the app"*) while still frustrating an angry passenger who expects immediate compensation on Twitter.
4. **The False Dichotomy of High Automation vs. Safety:** Reaching 60%+ automation is simple if one auto-handles payment disputes. However, doing so raises unsafe auto-handling to dangerous levels. Our 24.5% automation coverage reflects an intentionally conservative, risk-averse stance.

---

## 🛠️ Tech Stack & Testing

* **Core Language:** Python 3.10+
* **ML & Information Retrieval:** `scikit-learn` (TF-IDF Vectorizer, Cosine Similarity, Logistic Regression), `pandas`, `numpy`
* **Interactive UI:** `streamlit` (Multi-tab support copilot, real-time analytics, data editor)
* **LLM Integration:** OpenAI-compatible API adapter supporting **Groq** (`llama-3.3-70b-versatile`) and **NVIDIA NIM** (`llama-3.1-8b-instruct`) with deterministic offline fallback
* **Testing:** `pytest` (11 unit tests covering agents, data ingestion, evaluation, safety invariants, and deliverables integrity)

---

## 📚 Citations & Acknowledgments

* **Primary Dataset:** Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`), CC0 public domain. Reconstructed into 41,185 multi-turn `@Uber_Support` conversation threads.
* **Intent Taxonomy Inspiration:** PolyAI Banking77 (Casanueva et al., 2020) for intent boundary structuring.
* **Calibration Methodology:** Expected Calibration Error (ECE) and reliability diagrams following Guo et al. (*"On Calibration of Modern Neural Networks"*, ICML 2017).
* **Evaluation Framework:** Multi-criteria rubric formulation following Zheng et al. (*"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"*, NeurIPS 2023).
* **Core Libraries:** `scikit-learn`, `pandas`, `numpy`, `streamlit`, `plotly`, `python-dotenv`, and `pytest`.



