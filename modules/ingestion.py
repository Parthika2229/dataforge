"""
InsOps · Data Ingestion Module
Loads CSV files and builds a column-level data profile.
"""
import pandas as pd
import numpy as np
from io import StringIO


def load_data(uploaded_file):
    """Read an uploaded Streamlit file object into a DataFrame."""
    try:
        content = uploaded_file.read()
        # Try common encodings
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(StringIO(content.decode(enc)))
                return df, None
            except UnicodeDecodeError:
                continue
        return None, "Could not decode file with utf-8, latin-1, or cp1252."
    except Exception as e:
        return None, str(e)


def detect_data_profile(df: pd.DataFrame) -> dict:
    """
    Analyse each column and return a profile dict with:
      - col_types  : detected semantic type per column
      - datetime_cols, numerical_cols, categorical_cols
    """
    col_types = {}
    datetime_cols = []
    numerical_cols = []
    categorical_cols = []

    for col in df.columns:
        series = df[col]
        dtype  = series.dtype

        # Try datetime parse on object columns
        if dtype == object:
            sample = series.dropna().head(100)
            try:
                pd.to_datetime(sample, infer_datetime_format=True)
                col_types[col] = "datetime"
                datetime_cols.append(col)
                continue
            except Exception:
                pass
            col_types[col] = "categorical"
            categorical_cols.append(col)

        else:
            # Use pandas kind checks to avoid StringDtype issues
            try:
                is_int   = pd.api.types.is_integer_dtype(dtype)
                is_float = pd.api.types.is_float_dtype(dtype)
            except Exception:
                is_int = is_float = False

            if is_int:
                if series.nunique() <= 15 and series.nunique() / max(len(series), 1) < 0.05:
                    col_types[col] = "categorical (int)"
                    categorical_cols.append(col)
                else:
                    col_types[col] = "numerical (int)"
                    numerical_cols.append(col)
            elif is_float:
                col_types[col] = "numerical (float)"
                numerical_cols.append(col)
            else:
                col_types[col] = str(dtype)
                categorical_cols.append(col)

    return {
        "col_types": col_types,
        "datetime_cols": datetime_cols,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
    }
