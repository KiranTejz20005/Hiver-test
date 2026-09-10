from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import cohen_kappa_score


def classification_metrics(y_true, y_pred):
    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)), "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)}


def escalation_metrics(rows):
    tp = sum(r["expected_decision"] == r["predicted_decision"] == "escalate" for r in rows)
    fp = sum(r["expected_decision"] == "auto-handle" and r["predicted_decision"] == "escalate" for r in rows)
    fn = sum(r["expected_decision"] == "escalate" and r["predicted_decision"] == "auto-handle" for r in rows)
    tn = sum(r["expected_decision"] == r["predicted_decision"] == "auto-handle" for r in rows)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "unsafe_auto_handling_rate": fn / max(tp + fn, 1), "automation_coverage": (tn + fn) / max(len(rows), 1)}


def judge_agreement(human, judge):
    if not human or not judge:
        return {"n": 0, "weighted_kappa": None, "status": "no paired labels"}
    return {"n": len(human), "weighted_kappa": float(cohen_kappa_score(human, judge, weights="quadratic")), "status": "measured"}
