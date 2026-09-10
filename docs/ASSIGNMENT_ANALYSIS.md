# Hiver SDE Intern Assignment: Understanding and Proposed Approach

Source brief: [Hiver SDE Intern - Take-Home Assignment](https://docs.google.com/document/d/18xREErNCLCGoH5fDvmciwrq7vuAPsJ-1nnt6LMtlJZ4/edit?tab=t.0)

## 1. What the assignment is really asking

This is not primarily an LLM demo. It is an applied machine-learning and evaluation exercise.

The expected system is a small AI support agent for **one brand** selected from the Customer Support on Twitter dataset. For every incoming customer message, the agent must:

1. Assign an intent from a compact taxonomy derived from that brand's data.
2. Draft a response grounded in the way that brand historically handled similar issues.
3. Decide whether to handle the message automatically or escalate it to a human, with a reason.

The central evaluation question is: **Would a support team trust the agent, and can the candidate prove that trust is justified?** The brief explicitly says that the proof matters more than the system itself.

## 2. What the hiring team is evaluating

The deliverables reveal the assessment criteria:

- **Problem framing:** Can you make a large, noisy dataset manageable and define what “good” means for a specific brand?
- **Data judgment:** Can you sample conversations sensibly, handle multi-turn threads, avoid leakage, and create a useful labelled evaluation set?
- **Baseline thinking:** Can you show that the proposed system improves over simple alternatives?
- **Evaluation quality:** Can you measure classification, response quality, and escalation decisions separately?
- **Reliability thinking:** Can you identify when the system should not answer automatically?
- **Failure analysis:** Can you inspect real bad cases and form plausible hypotheses instead of reporting only one score?
- **Scientific honesty:** Can you explain what the headline number hides or overstates?
- **Engineering communication:** Can another person reproduce the main result quickly and understand the key decisions?
- **Ownership:** Since AI coding assistants are allowed but the work will be discussed live, you must understand every important design choice.

## 3. Recommended scope

Choose one brand with enough volume and reasonably repetitive support patterns. The best choice is not necessarily the largest brand; it is the brand for which:

- customer and agent turns can be reconstructed reliably;
- there are several recurring issue types;
- historical replies contain useful resolution behaviour;
- the dataset has enough examples for train, validation, test, and a hand-labelled golden set;
- the brand's support policy is understandable from the conversations.

Keep the scope deliberately narrow. The submission should support a selected brand and a defined set of customer-support situations, not pretend to be a universal customer-service platform.

### Suggested non-goals

- No attempt to support every brand in the dataset.
- No claim that historical replies are always correct or policy-compliant.
- No autonomous handling of high-risk, ambiguous, abusive, legal, financial, security, or privacy-sensitive cases unless the evidence clearly supports it.
- No attempt to solve every possible intent; include an `other/unknown` or escalation path.
- No use of Banking77 as if it were representative of the selected brand. It can help with intent taxonomy thinking, but the primary evaluation should remain brand-specific.

## 4. Proposed conceptual solution

Treat the agent as a three-stage decision process with a shared evidence layer:

```text
Incoming customer message
          |
          v
Conversation and similar-case retrieval
          |
          +--> Intent classification
          |
          +--> Candidate reply grounded in retrieved historical resolutions
          |
          +--> Risk / confidence / escalation decision
                           |
              Auto-handle with draft  OR  Escalate with reason
```

The important design principle is that reply generation and escalation should use the same evidence. A confident-sounding reply without sufficiently similar historical support should be escalated rather than treated as safe.

### Intent taxonomy

Build the taxonomy from the selected brand's conversation data, not from an arbitrary generic list. Start with clusters of recurring customer problems, then merge near-duplicates until the labels are:

- mutually understandable;
- frequent enough to evaluate;
- actionable for support;
- distinct enough that a human annotator can apply them consistently.

Include an `unclear/other` category. A small, defensible taxonomy is better than many fragile labels.

### Historical grounding

Use resolved historical conversations as evidence for response drafting. A retrieved example should be similar in issue, product context, and resolution state. The proposed response should be checked against the evidence for:

- factual support;
- resolution completeness;
- tone and brand fit;
- absence of unsupported promises;
- appropriate next steps.

The report should make clear whether a response is copying a known resolution pattern, adapting one, or operating without a close precedent.

### Escalation policy

Escalation should be treated as a first-class output, not as a fallback added at the end. A reasonable policy can combine:

- intent confidence;
- similarity or evidence strength;
- whether the issue appears resolved in historical data;
- risk indicators;
- ambiguity or missing context;
- whether the generated response makes a consequential claim.

Every escalation should include a short reason such as “low evidence for this issue type,” “customer is reporting a possible account/security problem,” or “historical examples disagree.”

## 5. Evaluation plan

### Golden set

Create 150–250 hand-labelled examples from the selected brand. The sampling note should explain:

- how examples were sampled across time and conversation length;
- how common and rare intents were represented;
- how ambiguous and difficult cases were deliberately included;
- what each annotator labelled;
- how disagreements were resolved;
- which fields were hidden from the system during evaluation.

The set should contain complete enough context for a human to judge the intended support action, while preserving a clean separation from examples used for retrieval or prompt construction.

At minimum, label:

- intent;
- whether the historical conversation appears resolved;
- acceptable response characteristics or expected next step;
- auto-handle vs. escalate;
- escalation reason when applicable.

### Dataset split and leakage controls

Split by conversation or thread, never by individual tweet alone. Prefer a time-aware split if the goal is to simulate future support. Make sure near-duplicate messages, repeated campaigns, and the same conversation do not appear across train and test.

Document exactly what data the agent is allowed to retrieve at evaluation time. This is essential because retrieval from the test conversation itself would inflate results.

### Baselines

Include at least two baselines as required:

1. **Trivial baseline:** majority intent, a generic safe reply, and always escalate (or always auto-handle, depending on the metric being tested).
2. **Simple baseline:** keyword/rule intent matching plus a nearest historical example or template-based reply.

The proposed system should be compared against both on the same golden set and under the same leakage rules. The purpose is to show whether the additional complexity creates meaningful value.

### Metrics

Report separate metrics rather than one blended score:

- Intent macro-F1 and per-intent precision/recall, because common intents can hide rare-intent failures.
- Escalation precision, recall, and confusion matrix, with particular attention to unsafe auto-handling.
- Reply quality rubric scores for groundedness, correctness, completeness, tone, and actionability.
- Unsupported-claim rate or citation/evidence coverage for generated replies.
- Coverage: the percentage of examples the system chooses to auto-handle.
- Selective quality: response quality among auto-handled examples as the confidence threshold changes.

The key operational tradeoff is not simply “accuracy.” It is **quality at a chosen automation coverage and risk level**.

### LLM judge and human agreement

An LLM judge can score reply quality, but it must not be treated as ground truth automatically. Define a rubric with explicit 1–5 anchors and give the judge the customer message, relevant context, retrieved evidence, draft reply, and expected support outcome where available.

Have a human review a meaningful subset of the same examples. Report agreement using a simple statistic such as exact agreement, average score difference, or weighted Cohen's kappa. Include examples where the judge and human disagree, explain the likely reason, and state how the rubric was adjusted.

## 6. Failure analysis the report should contain

The report must show the top five failure modes using real examples. Strong categories to look for are:

1. **Intent overlap:** two issue types have similar language but require different actions.
2. **Insufficient context:** the customer message is too short, references an earlier turn, or omits a required identifier.
3. **Weak retrieval:** the system finds lexically similar but operationally different historical cases.
4. **Unsupported or overconfident replies:** the draft invents a policy, timeline, refund, or resolution not supported by evidence.
5. **Escalation calibration:** the agent auto-handles a risky case or escalates an ordinary case too often.

For each failure, include the input, the system output, the expected behaviour, the suspected cause, and a concrete improvement hypothesis. The point is to demonstrate debugging judgment, not merely to list errors.

## 7. The mandatory “misleading headline number” section

This section is an opportunity to show maturity. Possible caveats include:

- A high intent F1 may be driven by a few frequent intents.
- A high reply score may reflect fluent tone rather than factual correctness.
- An LLM judge may reward verbosity or agree with the system's framing.
- Results may be optimistic if the golden set is too easy or resembles the retrieval corpus.
- High quality may be achieved only at low automation coverage.
- Always escalating can look safe while providing no automation value.
- Historical resolutions may encode inconsistent or outdated support practices.

State the main number, then explain what it does not measure. A trustworthy conclusion might be that the system is promising for a limited class of low-risk cases, but not ready for unrestricted automation.

## 8. Suggested report structure

Keep the report within six pages, or use a clearly structured README section:

1. Executive summary and selected brand.
2. Problem definition, scope, and non-goals.
3. Data preparation, taxonomy, and leakage controls.
4. System design at a conceptual level.
5. Golden-set construction and annotation process.
6. Baselines and evaluation methodology.
7. Results, including automation coverage and risk tradeoffs.
8. LLM-judge rubric and human-agreement evidence.
9. Top five failure modes with examples.
10. What the headline number is hiding.
11. One-week follow-up plan.
12. Limitations and final recommendation.

## 9. Decision log: decisions worth recording

The final submission should include 10–15 non-obvious decisions. Good entries would cover:

- why this brand was selected;
- how thread boundaries were reconstructed;
- how the intent taxonomy was merged or split;
- why certain intents were excluded;
- how train/test leakage was prevented;
- how golden examples were sampled;
- what counts as a correct response;
- what evidence is sufficient for auto-handling;
- which cases always escalate;
- how the baselines were defined;
- why the chosen metrics match support risk;
- how LLM-judge bias was checked;
- how disagreements were handled;
- what the headline metric leaves out;
- why particular next-week improvements were prioritized.

Each entry should state the decision, the reason, and the tradeoff. This is likely to be useful during the live discussion because it shows that the design was intentional rather than accidental.

## 10. What “good” looks like to the evaluator

A strong submission will probably be modest in its claims and unusually clear about evidence. It will show a small, reproducible experiment with a brand-specific taxonomy, a defensible golden set, meaningful baselines, honest metrics, and concrete failure analysis.

The submission does not need to prove that an AI agent can replace a support team. It needs to prove that you can identify where automation is useful, where it is unsafe, and how you would measure the difference.

## 11. Practical sequence of work

1. Inspect the dataset and shortlist brands by volume, thread quality, and recurring issue structure.
2. Choose one brand and write its definition of “good support.”
3. Reconstruct threads and perform leakage-safe sampling.
4. Draft and refine a compact intent taxonomy.
5. Create the golden set and annotation guidelines.
6. Define the trivial and simple baselines before looking at final results.
7. Design the proposed agent around intent, evidence, reply, and escalation.
8. Define the reply rubric and human-agreement procedure.
9. Prepare the failure-analysis template before reviewing outputs.
10. Write the report around evidence and limitations, not only the best score.
11. Finish the README so another person can reproduce the headline results quickly.
12. Rehearse an explanation of every non-obvious decision for the live discussion.

## Bottom line

The hiring team is looking for an engineer who can turn messy support data into a carefully bounded AI workflow and then evaluate it honestly. The strongest strategy is to optimize for **credible evidence and clear reasoning**, with automation limited to cases where the historical data provides enough support.
