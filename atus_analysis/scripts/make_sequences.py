from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
SEQ  = ROOT / "data" / "sequences"

SEQ.mkdir(parents=True, exist_ok=True)

activity = pd.read_parquet(DATA / "activity.parquet",
                           columns=["TUCASEID", "ACT_CODE",
                                    "TUACTIVITY_N", "TUACTDUR24"])
print(activity.shape)

SLOT = 10  # minutes
DAY_MIN = 24 * 60
NUM_SLOT = DAY_MIN // SLOT             # 144

def diary_to_grid(df):
    grid   = np.full(NUM_SLOT, 0, dtype=np.int32)   # 0 = idle
    cursor = 0

    for row in df.itertuples(index=False):
        dur   = int(row.TUACTDUR24)
        code  = np.int32(row.ACT_CODE)
        nslot = (dur + SLOT - 1) // SLOT            # ceil
        end   = min(cursor + nslot, NUM_SLOT)
        grid[cursor:end] = code
        cursor = end
        if cursor >= NUM_SLOT:
            break
    return grid

activity = activity.sort_values(["TUCASEID", "TUACTIVITY_N"])
grids = []
ids   = []

for case_id, grp in tqdm(activity.groupby("TUCASEID", sort=False)):
    ids.append(case_id)
    grids.append(diary_to_grid(grp))

X = np.stack(grids)          # shape (N, 144)
np.save(SEQ / "grid_10min.npy", X)
pd.Series(ids, name="TUCASEID").to_csv(SEQ / "grid_ids.csv", index=False)
print("✓ Saved fixed-length grids → grid_10min.npy")


def diary_to_ragged(df):
    return list(zip(df["ACT_CODE"].astype(np.int32),
                    df["TUACTDUR24"].astype(np.int32)))

ragged = (activity
          .groupby("TUCASEID", sort=False)
          .apply(diary_to_ragged)
          .to_frame(name="SEQ"))

ragged.to_parquet(SEQ / "ragged_sequences.parquet", index=True)
print("✓ Saved ragged list parquet")


