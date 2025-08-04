# atus_analysis/scripts/build_dataset.py
from utils.atus_io import read_atus, Path, RAW
import pandas as pd

PROC = RAW.parent / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 3. RESPONDENT-LEVEL TABLE
# ---------------------------------------------------------------------------

resp = read_atus("atusresp_0324",
                 dtypes={"TUCASEID": "int64", "TUFNWGTP": "float64"})
cps  = read_atus("atuscps_0324",
                 dtypes={"TUCASEID": "int64", "TULINENO": "int8"})
rost = read_atus("atusrost_0324",
                 dtypes={"TUCASEID": "int64", "TULINENO": "int8", "TEAGE": "int8"})

# keep respondent row only (TULINENO == 1)
cps_main  = cps.loc[cps["TULINENO"] == 1].drop(columns="TULINENO")
rost_main = rost.loc[rost["TULINENO"] == 1, ["TUCASEID", "TEAGE", "TESEX"]]

# indicator: any household child under 6 y
has_child_u6 = (
    rost.query("TEAGE.between(0, 5)")
        .groupby("TUCASEID")
        .size()
        .gt(0)
        .rename("HH_CHILD_U6")
)

respondent = (
    resp
      .merge(cps_main,  on="TUCASEID", how="left")
      .merge(rost_main, on="TUCASEID", how="left")
      .merge(has_child_u6, on="TUCASEID", how="left")
)

# ensure clean dtype
respondent["HH_CHILD_U6"] = (
    respondent["HH_CHILD_U6"]
        .fillna(False)
        .astype("boolean")        # or .astype("int8") for 0/1
)

respondent.to_parquet(PROC / "respondent.parquet", index=False)
print("✓ respondent.parquet written")

# ---------------------------------------------------------------------------
# 4. ACTIVITY-LEVEL TABLE
# ---------------------------------------------------------------------------

# --- 4.0  Load activity file -------------------------------------------------
act = read_atus("atusact_0324",
                dtypes={
                    "TUCASEID":      "int64",
                    "TUACTIVITY_N":  "int16",
                    "TUACTDUR24":    "int16"   # episode duration, minutes
                })

# --- 4A.  Pick or build the 6-digit activity code ---------------------------
if "TRCODEP" in act.columns:                      # modern multi-year bundles
    act["ACT_CODE"] = act["TRCODEP"].astype(str).str.zfill(6)

elif all(c in act.columns for c in
         ("TUTIER1CODE", "TUTIER2CODE", "TUTIER3CODE")):
    act["ACT_CODE"] = (
        act["TUTIER1CODE"].astype(str).str.zfill(2) +
        act["TUTIER2CODE"].astype(str).str.zfill(2) +
        act["TUTIER3CODE"].astype(str).str.zfill(2)
    )

else:
    raise KeyError(
        "Cannot locate activity-code variables. "
        "Expected TRCODEP or TUTIER1CODE/2CODE/3CODE.\n"
        f"Columns present: {list(act.columns)[:20]} …"
    )

# --- 4B.  Detect the episode start-time column ------------------------------
for cand in ("TUSTARTTIM", "TRSTTIME", "TRSTARTTIM"):
    if cand in act.columns:
        start_col = cand
        break
else:
    raise KeyError("Start-time column not found "
                   "(looked for TUSTARTTIM, TRSTTIME, TRSTARTTIM).")

# --- 4C.  Order episodes within each diary ----------------------------------
act = (
    act.sort_values(["TUCASEID", start_col], kind="mergesort")  # stable sort
       .reset_index(drop=True)
)

# ---------------------------------------------------------------------------
# 5. OPTIONAL “who-was-with” ENRICHMENT
# ---------------------------------------------------------------------------
who = read_atus("atuswho_0324",
                dtypes={"TUCASEID": "int64",
                        "TUACTIVITY_N": "int16",
                        "TULINENO": "int8"})

# merge Who + Roster (to get attributes of companions if needed)
who_rost = who.merge(rost, on=["TUCASEID", "TULINENO"], how="left")

act = act.merge(
        who_rost.groupby(["TUCASEID", "TUACTIVITY_N"])
                .size()
                .rename("NUM_PEOPLE"),
        on=["TUCASEID", "TUACTIVITY_N"],
        how="left"
)

act.to_parquet(PROC / "activity.parquet", index=False)
print("✓ activity.parquet written")
