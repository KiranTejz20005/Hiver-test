import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import os
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.agent import UberSupportAgent

try:
    from src.data import extract_support_resolution, load_cases
except ImportError:
    from src.data import load_cases
    DEFAULT_INTENT_RESOLUTIONS = {
        "safety_or_driver_conduct": "Escalate immediately to Uber safety incident specialist for urgent investigation.",
        "payment_problem": "Review trip fare breakdown and payment method; route for billing adjustment.",
        "refund_request": "Examine trip cancellation timing and fare policy to process eligible refund.",
        "lost_item": "Guide rider through the lost-item reporting workflow in the Uber app.",
        "account_or_login": "Assist rider with identity verification and account credential recovery.",
        "promo_or_coupon": "Verify promotion terms and trip eligibility; apply fare credit if applicable.",
        "trip_or_pickup": "Troubleshoot booking/pickup issue and check driver status in the app.",
        "other_or_unclear": "Connect with rider through in-app support to investigate and resolve inquiry.",
    }

    def extract_support_resolution(conversation: str, intent: str = "other_or_unclear") -> str:
        support_lines = []
        seen = set()
        for line in str(conversation).split("\n"):
            line_clean = line.strip()
            if line_clean.startswith("Support:"):
                text = line_clean[len("Support:"):].strip()
                if text and text not in seen:
                    seen.add(text)
                    support_lines.append(text)
        if support_lines:
            res = " ".join(support_lines).strip()
            if len(res) >= 10 and re.search(r"[a-zA-Z]{3,}", res):
                return res
        return DEFAULT_INTENT_RESOLUTIONS.get(intent, DEFAULT_INTENT_RESOLUTIONS["other_or_unclear"])

load_dotenv()

st.set_page_config(page_title="Uber Support Copilot", page_icon="U", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 2rem;}
.decision {
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #cbd5e1;
    background: #f8fafc;
    color: #0f172a !important;
    font-weight: 500;
}
.decision strong {
    color: #0f172a !important;
    font-size: 1.1rem;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 0.35rem;
}
.safe {border-left: 6px solid #16a34a;}
.escalate {border-left: 6px solid #dc2626;}
</style>
""", unsafe_allow_html=True)

def parse_example_turns(raw_conversation: str) -> tuple[str, str]:
    """Isolate the customer's actual problem turn from setup tweets and conversation context."""
    lines = [line.strip() for line in str(raw_conversation).split("\n") if line.strip()]
    turns = []
    current_speaker = None
    current_lines = []

    for line in lines:
        if line.startswith("Customer:"):
            if current_speaker:
                turns.append((current_speaker, " ".join(current_lines).strip()))
            current_speaker = "Customer"
            current_lines = [line[len("Customer:"):].strip()]
        elif line.startswith("Support:"):
            if current_speaker:
                turns.append((current_speaker, " ".join(current_lines).strip()))
            current_speaker = "Support"
            current_lines = [line[len("Support:"):].strip()]
        else:
            current_lines.append(line)

    if current_speaker:
        turns.append((current_speaker, " ".join(current_lines).strip()))

    setup_patterns = [r"^just setting up my twitter", r"^#myfirsttweet\s*$", r"^@\w+\s+#myfirsttweet"]
    customer_indices = [i for i, (spk, _) in enumerate(turns) if spk == "Customer"]

    # Pick first substantive customer turn that isn't a Twitter setup artifact
    selected_turn_idx = -1
    for idx in customer_indices:
        content = turns[idx][1]
        is_setup = any(re.search(pat, content, re.IGNORECASE) for pat in setup_patterns)
        if not is_setup or len(customer_indices) == 1:
            selected_turn_idx = idx
            break

    if selected_turn_idx == -1 and customer_indices:
        selected_turn_idx = customer_indices[0]

    if selected_turn_idx != -1:
        target_message = turns[selected_turn_idx][1]
        other_turns = [f"{spk}: {content}" for i, (spk, content) in enumerate(turns) if i != selected_turn_idx]
        context_str = "\n".join(other_turns)
        return target_message, context_str

    return raw_conversation, ""

st.title("Uber Support Copilot")
st.caption("Evidence-grounded response drafting with explicit escalation decisions")

@st.cache_data(show_spinner="Loading Uber conversations...")
def get_cases(version: str = "v2_with_resolutions"):
    df = load_cases()
    if "resolution" in df.columns:
        mask_empty = df["resolution"].isna() | df["resolution"].astype(str).str.strip().eq("")
        if mask_empty.any():
            df.loc[mask_empty, "resolution"] = [
                extract_support_resolution(c, i)
                for c, i in zip(df.loc[mask_empty, "customer"], df.loc[mask_empty, "intent"])
            ]
    return df


@st.cache_resource(show_spinner="Building retrieval index...")
def get_agent(cases):
    return UberSupportAgent(cases)


cases = get_cases(version="v2_with_resolutions")
agent = get_agent(cases)

# Clean, descriptive examples for interactive testing
valid_mask = (
    cases["title"].str.strip().ne("") &
    ~cases["title"].str.startswith("@") &
    ~cases["title"].str.lower().str.startswith("customer: just setting up") &
    (cases["title"].str.len() > 12)
)
example_cases = cases[valid_mask].drop_duplicates(subset=["title"]).head(500)

with st.sidebar:
    st.header("Workspace")
    st.write(f"Knowledge cases: **{len(cases):,}**")
    st.write("Provider: **NVIDIA / Groq / local fallback**")
    st.divider()
    st.info("This prototype never sends a customer message. It drafts a response and explains whether a human should review it.")

tab_case, tab_eval, tab_review = st.tabs(["Analyze case", "Evaluation overview", "Golden set review"])

with tab_case:
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Customer case")
        example = st.selectbox("Load example", ["Custom message", *example_cases["title"].tolist()])
        if example == "Custom message":
            default_message, default_context = "", ""
        else:
            raw_conv = example_cases.loc[example_cases.title == example, "customer"].iloc[0]
            default_message, default_context = parse_example_turns(raw_conv)
        message = st.text_area("Incoming message", value=default_message, height=150, placeholder="Example: I was charged twice for the same ride.")
        context = st.text_area("Conversation context (optional)", value=default_context, height=110, placeholder="Paste earlier turns if available.")
        analyze = st.button("Analyze support case", type="primary", use_container_width=True)


    if analyze:
        result = agent.analyze(message, context)
        with right:
            st.subheader("Agent decision")
            if result["decision"] == "escalate":
                st.error(f"**ESCALATE**\n\n{result['reason']}", icon="🚨")
            else:
                st.success(f"**AUTO-HANDLE**\n\n{result['reason']}", icon="✅")

            c1, c2 = st.columns(2)
            c1.metric("Predicted intent", result["intent"].replace("_", " ").title())
            c2.metric("Confidence", f'{result["confidence"]:.0%}')
            st.write(f"**Provider:** {agent.provider.provider or 'local deterministic fallback'}")
            st.write("**Risk indicators**")
            if result["risk_flags"]:
                for flag in result["risk_flags"]:
                    st.markdown(f"- ⚠️ `{flag}`")
            else:
                st.markdown("None detected *(routine low-risk request)*")

        st.subheader("Suggested reply")
        st.text_area("Draft", value=result["reply"], height=130, label_visibility="collapsed")
        st.subheader("Historical evidence")
        evidence = pd.DataFrame(result["evidence"])
        if "resolution" in evidence.columns:
            evidence["resolution"] = evidence["resolution"].fillna("").astype(str)
            empty_mask = evidence["resolution"].str.strip().eq("")
            if empty_mask.any():
                evidence.loc[empty_mask, "resolution"] = [
                    extract_support_resolution(c, i)
                    for c, i in zip(evidence.loc[empty_mask, "customer"], evidence.loc[empty_mask, "intent"])
                ]
        st.dataframe(
            evidence[["title", "customer", "resolution"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "title": st.column_config.TextColumn("Title", width="medium"),
                "customer": st.column_config.TextColumn("Customer conversation", width="large"),
                "resolution": st.column_config.TextColumn("Historical resolution", width="large"),
            },
        )

with tab_eval:
    st.subheader("Evaluation foundation")
    evaluation_file = Path("reports/evaluation.json")
    if evaluation_file.exists():
        report = json.loads(evaluation_file.read_text(encoding="utf-8"))
        st.success(f"Loaded {report['n']} real Uber evaluation examples with retrieval leakage control.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Proposed macro-F1", f"{report['proposed']['intent']['macro_f1']:.3f}")
        c2.metric("Automation coverage", f"{report['proposed']['escalation']['automation_coverage']:.1%}")
        c3.metric("Unsafe auto-handling", f"{report['proposed']['escalation']['unsafe_auto_handling_rate']:.1%}")
        comparison = pd.DataFrame([
            {"System": "Proposed retrieval + policy", "Intent macro-F1": report["proposed"]["intent"]["macro_f1"], "Automation coverage": report["proposed"]["escalation"]["automation_coverage"]},
            {"System": "Majority intent baseline", "Intent macro-F1": report["baselines"]["majority_intent"]["macro_f1"], "Automation coverage": 0.0},
            {"System": "Always escalate baseline", "Intent macro-F1": 0.0, "Automation coverage": report["baselines"]["always_escalate"]["automation_coverage"]},
        ])
        st.dataframe(comparison, width="stretch", hide_index=True)
        judge_file = Path("reports/judge_scores.csv")
        failure_file = Path("reports/failure_analysis.csv")
        if judge_file.exists():
            judge = pd.read_csv(judge_file)
            st.metric("Average judge score", f"{judge['overall'].mean():.2f}/5")
        if failure_file.exists():
            failures = pd.read_csv(failure_file)
            st.write(f"**Failure analysis:** {len(failures)} disagreement cases saved for review.")
    else:
        st.info("Run the evaluation scripts to populate this dashboard.")
    metrics = pd.DataFrame([
        {"Quality check": "Real Uber conversations", "Status": "Complete: 41,185 cases"},
        {"Quality check": "Retrieval leakage control", "Status": "Complete: holdout excluded"},
        {"Quality check": "Golden-set human review", "Status": "Required before final submission"},
        {"Quality check": "Human / judge agreement", "Status": "Next evaluation milestone"},
    ])
    st.dataframe(metrics, width="stretch", hide_index=True)

with tab_review:
    st.subheader("Golden set review")
    golden_path = Path("data/golden/uber_golden_set.csv")
    reviewed_path = Path("data/golden/uber_golden_set_reviewed.csv")
    if golden_path.exists():
        review = pd.read_csv(reviewed_path if reviewed_path.exists() else golden_path).head(50).copy()
        review["human_notes"] = review.get("human_notes", "").fillna("").astype(str)
        for column in ["human_groundedness", "human_correctness", "human_completeness", "human_tone", "human_safety"]:
            source = review[column] if column in review.columns else pd.Series(0, index=review.index)
            review[column] = pd.to_numeric(source, errors="coerce").fillna(0).astype(int)
        st.caption("Review a representative queue, correct the inferred intent or decision, and save the reviewed labels for the final report.")
        review_columns = ["conversation_id", "customer", "resolution", "expected_intent", "expected_decision", "human_groundedness", "human_correctness", "human_completeness", "human_tone", "human_safety", "human_notes"]
        edited = st.data_editor(review[review_columns], width="stretch", hide_index=True, num_rows="fixed", column_config={
            "conversation_id": st.column_config.TextColumn("Conversation", disabled=True),
            "customer": st.column_config.TextColumn("Conversation text", disabled=True, width="large"),
            "resolution": st.column_config.TextColumn("Resolution", disabled=True, width="large"),
            "expected_intent": st.column_config.SelectboxColumn("Intent", options=sorted(cases["intent"].unique().tolist())),
            "expected_decision": st.column_config.SelectboxColumn("Decision", options=["auto-handle", "escalate"]),
            "human_groundedness": st.column_config.NumberColumn("Groundedness", min_value=1, max_value=5, step=1),
            "human_correctness": st.column_config.NumberColumn("Correctness", min_value=1, max_value=5, step=1),
            "human_completeness": st.column_config.NumberColumn("Completeness", min_value=1, max_value=5, step=1),
            "human_tone": st.column_config.NumberColumn("Tone", min_value=1, max_value=5, step=1),
            "human_safety": st.column_config.NumberColumn("Safety", min_value=1, max_value=5, step=1),
            "human_notes": st.column_config.TextColumn("Reviewer notes"),
        })
        if st.button("Save reviewed labels", type="primary"):
            full = pd.read_csv(golden_path, dtype=str)
            full = full.astype(object)
            full = full.set_index("conversation_id")
            edited_clean = edited.set_index("conversation_id").astype(object)
            for column in review_columns[3:]:
                if column not in full.columns:
                    full[column] = ""
                full.loc[edited_clean.index, column] = edited_clean[column].astype(str)
            full.reset_index().to_csv(reviewed_path, index=False)
            st.success(f"Saved reviewed labels to {reviewed_path}")

    else:
        st.info("Run scripts/create_golden_set.py before reviewing labels.")
