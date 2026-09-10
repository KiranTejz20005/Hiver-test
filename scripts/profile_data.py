import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_cases

cases = load_cases()
profile = {"rows": len(cases), "intents": cases["intent"].value_counts().to_dict(), "source": "TNE-AI mirror of thoughtvector/customer-support-on-twitter"}
Path("reports").mkdir(exist_ok=True)
Path("reports/data_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
print(json.dumps(profile, indent=2))
