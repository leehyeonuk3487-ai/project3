"""
Filter KNHANES (.sav) and CHS (.sas7bdat) files:
- Keep only variables needed for military insurance risk analysis
- Filter age 15-39
- Merge all years, add 'year' column
- Save to data/processed/ as UTF-8-sig CSV
- Save codebook JSON (variable labels + value-label mappings)
"""

import os
import json
import pyreadstat
import pandas as pd
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE = "C:/Users/SEC/project3/data"
RAW = f"{BASE}/raw"
PROC = f"{BASE}/processed"
os.makedirs(PROC, exist_ok=True)

# ─── KNHANES ────────────────────────────────────────────────────────────────

KNHANES_FILES = {
    2021: f"{RAW}/knhanes/HN21_ALL.sav",
    2022: f"{RAW}/knhanes/HN22_ALL.sav",
    2023: f"{RAW}/knhanes/HN23_ALL.sav",
    2024: f"{RAW}/knhanes/HN24_ALL.sav",
}

KNHANES_VARS = [
    "age",        # 만나이
    "sex",        # 성별
    "region",     # 시도
    "BS3_1",      # 현재 일반담배(궐련) 흡연 여부  [smoking]
    "sm_presnt",  # 현재흡연율                     [smoking derived]
    "BD1_11",     # 1년간 음주빈도                  [alcohol]
    "dr_month",   # 월간음주율                     [alcohol derived]
    "BE3_71",     # 고강도 신체활동 여부: 일         [physical]
    "BE3_81",     # 중강도 신체활동 여부: 일         [physical]
    "BE3_31",     # 1주일간 걷기 일수               [physical]
    "pa_aerobic", # 유산소 신체활동 실천율           [physical derived]
    "DI1_dg",     # 고혈압 의사진단 여부             [hypertension]
    "DI1_pr",     # 고혈압 현재 유병 여부            [hypertension]
    "DE1_dg",     # 당뇨병 의사진단 여부             [diabetes]
    "DE1_pr",     # 당뇨병 현재 유병 여부            [diabetes]
    "HE_ht",      # 신장                           [height]
    "HE_wt",      # 체중                           [weight]
    "HE_BMI",     # 체질량지수                      [bmi]
    "HE_sbp",     # 최종 수축기 혈압(2,3차 평균)     [bp_sys]
    "HE_dbp",     # 최종 이완기 혈압(2,3차 평균)     [bp_dia]
    "wt_hs",      # 가구조사 가중치                  [weight_var]
    "wt_itvex",   # 건강설문,검진조사 가중치          [weight_var]
]

KNHANES_CATEGORIES = {
    "age":          ["age"],
    "sex":          ["sex"],
    "region":       ["region"],
    "smoking":      ["BS3_1", "sm_presnt"],
    "alcohol":      ["BD1_11", "dr_month"],
    "physical":     ["BE3_71", "BE3_81", "BE3_31", "pa_aerobic"],
    "hypertension": ["DI1_dg", "DI1_pr"],
    "diabetes":     ["DE1_dg", "DE1_pr"],
    "height":       ["HE_ht"],
    "weight":       ["HE_wt"],
    "bmi":          ["HE_BMI"],
    "bp_sys":       ["HE_sbp"],
    "bp_dia":       ["HE_dbp"],
    "weight_var":   ["wt_hs", "wt_itvex"],
}

# ─── CHS ─────────────────────────────────────────────────────────────────────

CHS_FILES = {
    2021: f"{RAW}/chs/2021/chs21_all.sas7bdat",
    2022: f"{RAW}/chs/2022/chs22_all.sas7bdat",
    2023: f"{RAW}/chs/2023/chs23_all.sas7bdat",
    2024: f"{RAW}/chs/2024/chs24_all.sas7bdat",
    2025: f"{RAW}/chs/2025/chs25_all.sas7bdat",
}

# CHS 2021 has lowercase ctprvn_code; 2022+ has CTPRVN_CODE (uppercase)
# We normalise column names to lowercase when reading CHS.
CHS_VARS = [
    "age",          # 만나이
    "sex",          # 성별
    "ctprvn_code",  # 시도번호  (normalised to lower)
    "signgu_code",  # 행정기관코드 (시군구)
    "sma_03z2",     # 일반담배(궐련) 현재흡연 여부   [smoking]
    "drb_01z3",     # 연간 음주 빈도               [alcohol]
    "drb_04z1",     # 월간 폭음 경험_남             [alcohol binge]
    "pha_04z1",     # 격렬한 신체활동 일수           [physical]
    "pha_07z1",     # 중등도 신체활동 일수           [physical]
    "phb_01z1",     # 걷기 실천 일수               [physical]
    "hya_04z1",     # 고혈압 진단 경험 여부          [hypertension]
    "dia_04z1",     # 당뇨병 진단 경험 여부          [diabetes]
    "oba_02z1",     # 키                           [height]
    "oba_03z1",     # 몸무게                       [weight]
    "wt_h",         # 가구가중치                    [weight_var]
    "wt_p",         # 개인가중치                    [weight_var]
]

CHS_CATEGORIES = {
    "age":          ["age"],
    "sex":          ["sex"],
    "region":       ["ctprvn_code", "signgu_code"],
    "smoking":      ["sma_03z2"],
    "alcohol":      ["drb_01z3", "drb_04z1"],
    "physical":     ["pha_04z1", "pha_07z1", "phb_01z1"],
    "hypertension": ["hya_04z1"],
    "diabetes":     ["dia_04z1"],
    "height":       ["oba_02z1"],
    "weight":       ["oba_03z1"],
    "bmi":          [],  # not available in CHS raw data
    "bp_sys":       [],  # not available in CHS
    "bp_dia":       [],  # not available in CHS
    "weight_var":   ["wt_h", "wt_p"],
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def build_codebook_entry(col, meta, original_col=None):
    """Build a single codebook entry dict for column `col`."""
    src_col = original_col or col
    label = meta.column_names_to_labels.get(src_col, "")
    entry = {"variable": src_col, "label": label}
    if hasattr(meta, "variable_value_labels") and src_col in meta.variable_value_labels:
        entry["value_labels"] = {
            str(k): v for k, v in meta.variable_value_labels[src_col].items()
        }
    return entry


def process_knhanes():
    print("\n" + "=" * 60)
    print("  KNHANES processing")
    print("=" * 60)

    dfs = []
    codebook = {}

    for year, path in KNHANES_FILES.items():
        print(f"\n  Year {year}: {os.path.basename(path)}")
        df, meta = pyreadstat.read_sav(path, encoding="utf-8",
                                       usecols=KNHANES_VARS)

        # Filter age 15-39
        df = df[df["age"].between(15, 39)].copy()
        df.insert(0, "year", year)

        print(f"    rows after age filter: {len(df):,}")
        dfs.append(df)

        # Build codebook from first year (structure identical across years)
        if year == 2021:
            for var in KNHANES_VARS:
                codebook[var] = build_codebook_entry(var, meta)

    merged = pd.concat(dfs, ignore_index=True)

    out_csv = f"{PROC}/knhanes_filtered_2021_2024.csv"
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_cb = f"{PROC}/knhanes_codebook.json"
    with open(out_cb, "w", encoding="utf-8") as f:
        json.dump(codebook, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(out_csv) / 1024 / 1024
    print(f"\n  >> Saved: {out_csv}")
    print(f"     Size: {size_mb:.2f} MB | Rows: {len(merged):,} | Cols: {len(merged.columns)}")
    print(f"  >> Codebook: {out_cb}")

    return merged, codebook


def process_chs():
    print("\n" + "=" * 60)
    print("  CHS processing")
    print("=" * 60)

    dfs = []
    codebook = {}
    first_year_done = False

    for year, path in CHS_FILES.items():
        print(f"\n  Year {year}: {os.path.basename(path)}")
        df, meta = pyreadstat.read_sas7bdat(path, encoding="cp949")

        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Select only available target columns
        available = [v for v in CHS_VARS if v in df.columns]
        missing_here = [v for v in CHS_VARS if v not in df.columns]
        if missing_here:
            print(f"    WARNING: missing vars {missing_here}")

        df = df[available].copy()
        df = df[df["age"].between(15, 39)].copy()
        df.insert(0, "year", year)

        print(f"    rows after age filter: {len(df):,}")
        dfs.append(df)

        if not first_year_done:
            # Build codebook: map normalised col name → original meta key
            meta_cols_lower = {c.lower(): c for c in meta.column_names}
            for var in CHS_VARS:
                orig = meta_cols_lower.get(var, var)
                codebook[var] = build_codebook_entry(var, meta, original_col=orig)
            first_year_done = True

    merged = pd.concat(dfs, ignore_index=True)

    out_csv = f"{PROC}/chs_filtered_2021_2025.csv"
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_cb = f"{PROC}/chs_codebook.json"
    with open(out_cb, "w", encoding="utf-8") as f:
        json.dump(codebook, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(out_csv) / 1024 / 1024
    print(f"\n  >> Saved: {out_csv}")
    print(f"     Size: {size_mb:.2f} MB | Rows: {len(merged):,} | Cols: {len(merged.columns)}")
    print(f"  >> Codebook: {out_cb}")

    return merged, codebook


def print_variable_table(title, categories, codebook):
    print(f"\n{'='*70}")
    print(f"  {title} — Variable Mapping Table")
    print(f"{'='*70}")
    print(f"  {'Category':<15} {'Variable':<15} {'Label'}")
    print(f"  {'-'*65}")
    for cat, vars_ in categories.items():
        if vars_:
            for v in vars_:
                lbl = codebook.get(v, {}).get("label", "(no label)")
                print(f"  {cat:<15} {v:<15} {lbl}")
        else:
            print(f"  {cat:<15} {'—':<15} (not available in this dataset)")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    kn_df, kn_cb = process_knhanes()
    chs_df, chs_cb = process_chs()

    print_variable_table("KNHANES", KNHANES_CATEGORIES, kn_cb)
    print_variable_table("CHS", CHS_CATEGORIES, chs_cb)

    print("\n\n=== SUMMARY ===")
    for fname in [
        "knhanes_filtered_2021_2024.csv",
        "chs_filtered_2021_2025.csv",
        "knhanes_codebook.json",
        "chs_codebook.json",
    ]:
        p = f"{PROC}/{fname}"
        mb = os.path.getsize(p) / 1024 / 1024
        print(f"  {fname:<40} {mb:.2f} MB")

    print("\nDone. Raw files untouched.")
