from pathlib import Path
import pandas as pd

RAW = Path(__file__).parents[1] / "data" / "raw"


def read_atus(name: str, dtypes: dict | None = None) -> pd.DataFrame:
    """
    Read a multi-year ATUS *.dat (actually CSV) file by stem name, e.g. 'atusact_0324'.
    """
    path = RAW / f"{name}.dat"
    return pd.read_csv(path, low_memory=False, dtype=dtypes)
