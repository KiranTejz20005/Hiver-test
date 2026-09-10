from src.data import infer_intent, load_cases


def test_real_uber_cases_are_available():
    cases = load_cases()
    assert len(cases) >= 1000
    assert {"customer", "resolution", "intent"}.issubset(cases.columns)
    assert (cases["resolution"].str.strip() != "").all()


def test_intent_rules_cover_core_uber_support_cases():
    assert infer_intent("My driver was dangerous and I felt unsafe") == "safety_or_driver_conduct"
    assert infer_intent("I was charged twice for one ride") == "payment_problem"
    assert infer_intent("I left my phone in the car") == "lost_item"
