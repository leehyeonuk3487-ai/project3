"""
Step 1 v2: Explore with correct encoding detection.
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

CATEGORIES = {
    "age":          ["연령", "나이", "age"],
    "sex":          ["성별", "sex", "gender"],
    "region":       ["시도", "시군구", "지역", "region", "sido", "sigungu"],
    "smoking":      ["흡연", "담배", "smok"],
    "alcohol":      ["음주", "알코올", "drink", "alcohol"],
    "physical":     ["신체활동", "걷기", "운동", "신체 활동", "physical", "walk"],
    "hypertension": ["고혈압", "혈압진단", "hypertens"],
    "diabetes":     ["당뇨", "diabetes"],
    "height":       ["신장", "키", "height"],
    "weight":       ["체중", "몸무게", "weight"],
    "bmi":          ["체질량", "bmi"],
    "bp_sys":       ["수축기", "systol"],
    "bp_dia":       ["이완기", "diastol"],
    "weight_var":   ["가중치", "wt_", "표본가중", "가중값"],
}


def try_read_meta_sav(path):
    """Try multiple encodings to get readable Korean labels from .sav"""
    for enc in ["cp949", "euc-kr", "utf-8"]:
        try:
            df, meta = pyreadstat.read_sav(path, metadataonly=True, encoding=enc)
            # Test if Korean labels are readable
            sample_labels = list(meta.column_names_to_labels.values())[:20]
            korean_ok = any(
                any('\uac00' <= ch <= '\ud7a3' for ch in str(lbl))
                for lbl in sample_labels
            )
            if korean_ok:
                return meta, enc
        except Exception:
            continue
    # Fallback: no encoding arg
    try:
        df, meta = pyreadstat.read_sav(path, metadataonly=True)
        return meta, "default"
    except Exception as e:
        raise e


def try_read_meta_sas(path):
    """Try multiple encodings for .sas7bdat"""
    for enc in ["cp949", "euc-kr", "utf-8"]:
        try:
            df, meta = pyreadstat.read_sas7bdat(path, metadataonly=True, encoding=enc)
            sample_labels = list(meta.column_names_to_labels.values())[:20]
            korean_ok = any(
                any('\uac00' <= ch <= '\ud7a3' for ch in str(lbl))
                for lbl in sample_labels
            )
            if korean_ok:
                return meta, enc
        except Exception:
            continue
    try:
        df, meta = pyreadstat.read_sas7bdat(path, metadataonly=True)
        return meta, "default"
    except Exception as e:
        raise e


def find_candidates(col_labels: dict) -> dict:
    results = {}
    for cat, keywords in CATEGORIES.items():
        hits = []
        for col, label in col_labels.items():
            label_lower = str(label).lower()
            col_lower = col.lower()
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in label_lower or kw_lower in col_lower:
                    hits.append((col, str(label)))
                    break
        results[cat] = hits
    return results


def explore(name, files, meta_fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    all_candidates = {}

    for year, path in files.items():
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        print(f"\n  -- Year {year}: {os.path.basename(path)} --")
        try:
            meta, enc = meta_fn(path)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"     Encoding: {enc} | Total vars: {len(meta.column_names)}")
        col_labels = meta.column_names_to_labels
        candidates = find_candidates(col_labels)
        all_candidates[year] = candidates

        for cat, hits in candidates.items():
            if hits:
                print(f"     [{cat}]")
                for col, lbl in hits[:8]:
                    print(f"       {col:25s} | {lbl}")
                if len(hits) > 8:
                    print(f"       ... ({len(hits)-8} more)")
            else:
                print(f"     [{cat}] *** NO MATCH ***")

    return all_candidates


print("=== KNHANES ===")
kn_cands = explore("KNHANES (.sav)", KNHANES_FILES, try_read_meta_sav)

print("\n=== CHS ===")
chs_cands = explore("CHS (.sas7bdat)", CHS_FILES, try_read_meta_sas)

# Serialize: tuples → lists
def serialize(d):
    return {
        str(y): {cat: [[c, l] for c, l in hits] for cat, hits in cats.items()}
        for y, cats in d.items()
    }

out = {"knhanes": serialize(kn_cands), "chs": serialize(chs_cands)}
with open("C:/Users/SEC/project3/scripts/candidates_v2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("\nDone. candidates_v2.json saved.")
