# Interview Notes

## What did I build?

An Uber support copilot that classifies a customer conversation, retrieves similar historical Uber resolutions, drafts a cautious reply, and routes uncertain or risky cases to a human.

## Why retrieval instead of fine-tuning?

The assignment asks for replies grounded in historical support behavior. Retrieval makes the evidence visible, updateable, and easier to audit than embedding a small noisy corpus into model weights.

## How did I avoid leakage?

Golden conversation IDs are excluded from the retrieval corpus at evaluation time. The leakage audit records the raw-source overlap separately from the runtime exclusion rule.

## Why macro F1?

Uber support intents are uneven. Macro F1 prevents common trip questions from hiding poor performance on rare but important safety or account cases.

## Why escalate?

Safety, account-specific payment, unknown, low-confidence, weak-evidence, injection-like, and failed-validation cases should not receive confident autonomous answers.

## What is the biggest weakness?

The draft golden labels are machine-assisted and not yet a human-validated benchmark. The report therefore does not claim human-judge agreement.

## What would I do with one more week?

Human-label the full benchmark, add resolution-level relevance labels, calibrate thresholds on a development split, and compare provider-backed responses against the deterministic fallback.
