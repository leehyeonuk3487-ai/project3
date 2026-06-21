"""
Step 1: Explore variable names and labels in KNHANES and CHS files.
Find candidates for each category using label keyword matching.
"""
import os
import pyreadstat
import json

BASE = "C:/Users/SEC/project3/data/raw"

KNHANES_FILES = {
    2021: f"{BASE}/knhanes/HN21_ALL.sav",
    2022: f"{BASE}/knhanes/HN22_ALL.sav",
    2023: f"{BASE}/knhanes/HN23_ALL.sav",
    2024: f"{BASE}/knhanes/HN24_ALL.sav",
}

CHS_FILES = {
    2021: f"{BASE}/chs/2021/chs21_all.sas7bdat",
    2022: f"{BASE}/chs/2022/chs22_all.sas7bdat",
    2023: f"{BASE}/chs/2023/chs23_all.sas7bdat",
    2024: f"{BASE}/chs/2024/chs24_all.sas7bdat",
    2025: f"{BASE}/chs/2025/chs25_all.sas7bdat",
}

# Keywords per category (Korean)
CATEGORIES = {
    "age":       ["연령", "나이", "age"],
    "sex":       ["성별", "sex", "gender"],
    "region":    ["시도", "시군구", "지역", "region", "sido", "sigungu"],
    "smoking":   ["흡연", "담배", "smok"],
    "alcohol":   ["음주", "알코올", "drink", "alcohol"],
    "physical":  ["신체활동", "걷기", "운동", "활동", "physical", "walk"],
    "hypertension": ["고혈압", "혈압진단", "hypertens"],
    "diabetes":  ["당뇨", "diabetes"],
    "height":    ["신장", "키", "height"],
    "weight":    ["체중", "몸무게", "weight"],
    "bmi":       ["체질량", "bmi"],
    "bp_sys":    ["수축기", "systol"],
    "bp_dia":    ["이완기", "diastol"],
    "weight_var":["가중치", "wt", "weight_var", "표본가중", "가중값"],
}


def try_read_meta(path, reader_fn):
    for enc in ["cp949", "euc-kr", "utf-8", None]:
        try:
            kwargs = {"metadataonly": True}
            if enc:
                kwargs["encoding"] = enc
            _, meta = reader_fn(path, **kwargs)
            return meta, enc
        except Exception as e:
            last_err = e
    raise last_err


def find_candidates(col_labels: dict) -> dict:
    """Return {category: [(varname, label), ...]} for all categories."""
    results = {}
    for cat, keywords in CATEGORIES.items():
        hits = []
        for col, label in col_labels.items():
            label_lower = str(label).lower()
            col_lower = col.lower()
            for kw in keywords:
                if kw.lower() in label_lower or kw.lower() in col_lower:
                    hits.append((col, label))
                    break
        results[cat] = hits
    return results


def explore(name, files, reader_fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Use first available file to find candidates (then verify across years)
    all_candidates = {}  # year -> {cat: [(col, label)]}

    for year, path in files.items():
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        print(f"\n  -- Year {year}: {os.path.basename(path)} --")
        try:
            meta, enc = try_read_meta(path, reader_fn)
        except Exception as e:
            print(f"  ERROR reading {path}: {e}")
            continue

        print(f"     Encoding: {enc} | Total vars: {len(meta.column_names)}")
        col_labels = meta.column_names_to_labels
        candidates = find_candidates(col_labels)
        all_candidates[year] = candidates

        for cat, hits in candidates.items():
            if hits:
                print(f"     [{cat}]")
                for col, lbl in hits:
                    print(f"       {col:20s} | {lbl}")
            else:
                print(f"     [{cat}] *** NO MATCH ***")

    return all_candidates


print("Exploring KNHANES files...")
kn_candidates = explore("KNHANES (.sav)", KNHANES_FILES, pyreadstat.read_sav)

print("\n\nExploring CHS files...")
chs_candidates = explore("CHS (.sas7bdat)", CHS_FILES, pyreadstat.read_sas7bdat)

# Save raw candidates to JSON for review
out = {
    "knhanes": {str(y): {cat: v for cat, v in cats.items()}
                for y, cats in kn_candidates.items()},
    "chs":     {str(y): {cat: v for cat, v in cats.items()}
                for y, cats in chs_candidates.items()},
}
with open("C:/Users/SEC/project3/scripts/candidates.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("\n\nDone. candidates.json saved.")
