"""Load the raw WLB dataset (xlsx or csv)."""
import pandas as pd
from .config import RAW_PATH, RAW_SHEET


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    path = str(path)
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=RAW_SHEET)
    return pd.read_csv(path)
