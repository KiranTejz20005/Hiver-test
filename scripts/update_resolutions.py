import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data import extract_support_resolution

def update_all():
    print("1. Updating data/uber_cases.parquet...")
    df = pd.read_parquet("data/uber_cases.parquet")
    df["resolution"] = [
        extract_support_resolution(c, i)
        for c, i in zip(df["customer"], df["intent"])
    ]
    empty_count = (df["resolution"].str.strip() == "").sum()
    assert empty_count == 0, f"Found {empty_count} empty resolutions!"
    df.to_parquet("data/uber_cases.parquet", index=False)
    print(f"Updated {len(df)} cases in uber_cases.parquet. All have valid resolutions.")

    print("2. Updating data/golden/uber_golden_set.csv...")
    golden = pd.read_csv("data/golden/uber_golden_set.csv")
    golden["resolution"] = [
        extract_support_resolution(c, i)
        for c, i in zip(golden["customer"], golden["intent"])
    ]
    assert (golden["resolution"].str.strip() != "").all()
    golden.to_csv("data/golden/uber_golden_set.csv", index=False)
    print(f"Updated {len(golden)} cases in uber_golden_set.csv.")

    print("3. Updating data/golden/uber_golden_set_reviewed.csv...")
    reviewed = pd.read_csv("data/golden/uber_golden_set_reviewed.csv")
    reviewed["resolution"] = [
        extract_support_resolution(c, i)
        for c, i in zip(reviewed["customer"], reviewed["intent"])
    ]
    assert (reviewed["resolution"].str.strip() != "").all()
    reviewed.to_csv("data/golden/uber_golden_set_reviewed.csv", index=False)
    print(f"Updated {len(reviewed)} cases in uber_golden_set_reviewed.csv.")

if __name__ == "__main__":
    update_all()
