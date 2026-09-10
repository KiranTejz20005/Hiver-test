# Golden Evaluation Set: Sampling & Labelling Methodology

## 1. Overview
The Golden Evaluation Set consists of **200 hand-curated and annotated customer support cases** extracted from the historical `@Uber_Support` Twitter conversation corpus. It serves as the primary benchmark for evaluating intent classification, escalation policy calibration, and draft response quality.

* **Total Cases:** 200
* **Class Balance:** Exact stratified distribution of 25 cases across all 8 canonical intent categories.
* **Storage Location:** `data/golden/uber_golden_set.csv` (and reviewed ground truth in `data/golden/uber_golden_set_reviewed.csv`)

---

## 2. Sampling Methodology

### 2.1 Stratification Strategy
In natural customer support traffic, over 70% of messages are routine status checks, generic greetings, or brief account questions. A random sample would result in single-digit representations of high-consequence edge cases (e.g., driver assault, reckless driving, double billing).

To evaluate safety recall and policy adherence with statistical rigor:
1. We partitioned the 41,185 Uber cases by primary intent signals.
2. We sampled **exactly 25 representative conversations** for each of the 8 canonical intents:
   - `safety_or_driver_conduct` (25 cases)
   - `payment_problem` (25 cases)
   - `refund_request` (25 cases)
   - `lost_item` (25 cases)
   - `account_or_login` (25 cases)
   - `promo_or_coupon` (25 cases)
   - `trip_or_pickup` (25 cases)
   - `other_or_unclear` (25 cases)

### 2.2 Text Length & Complexity Diversity
The sample was filtered to represent varied conversation structures:
- **Short (1–2 turns):** Direct inquiries (e.g., "Left my sunglasses in driver's car").
- **Medium (3–4 turns):** Multi-turn discussions requiring contextual turn separation.
- **Long (5+ turns):** Complex customer disputes involving repeated friction or multi-layered issues.

---

## 3. Hand-Labelling Protocol & Rubric

Each case was individually audited and annotated with three ground-truth labels:

1. **`expected_intent`:** The core operational category governing the inquiry. Ambiguous or conversational messages without specific problem declarations were assigned `other_or_unclear`.
2. **`expected_decision` (`escalate` vs `auto-handle`):**
   - **`escalate` (125 cases / 62.5%):** Assigned to all safety allegations, driver conduct violations, payment discrepancies (e.g., double charges, Paytm wallet deductions), account security risks, and complex fare disputes where financial backend verification is mandatory.
   - **`auto-handle` (75 cases / 37.5%):** Assigned only to low-risk, self-serve requests (e.g., lost item reporting instructions, promo terms, general app navigation).
3. **`human_notes`:** Explicit reasoning documenting the presence of risk triggers, ambiguity, or specific policy constraints.

---

## 4. Strict Retrieval Leakage Control
To prevent data leakage during evaluation:
- All 200 `conversation_id`s in this golden set are programmatically excluded from the 41,185-case TF-IDF search index at runtime (verified by `scripts/leakage_audit.py`).
- The agent is forced to retrieve evidence from the remaining 40,985 corpus cases, ensuring zero verbatim train-test memorization.
