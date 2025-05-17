from datetime import datetime, timedelta
import json
import pandas as pd
import re
from config import TEMP_PROCESSED_JSON
import sys

def data_preprocessing(path):
    """
    Reads news.csv, normalises timestamps, shifts weekend entries to Monday 09:00,
    extracts ONLY yesterday’s rows, and writes them to TEMP_PROCESSED_JSON.

    Args
    ----
    path : str
        File path to news.csv
    """
    try:
        import pandas as pd
        import json, re, sys
        from datetime import datetime, timedelta
        from config import TEMP_PROCESSED_JSON

        # ───────────────────────── 1. LOAD & RENAME ──────────────────────────
        df = pd.read_csv(path, delimiter=";", encoding="utf-8", quotechar='"', on_bad_lines="skip")
        df = df.rename(columns={
            "Date and Timestamp": "datetime",
            "Title": "header",
            "Full Text": "content",
            "Source": "source",
            "SpecificSource": "specific_source",
            "Link": "link",
        })

        # ──────────────────────── 2. PARSE DATETIMES ─────────────────────────
        def _standardise(dt):
            # Match e.g. '16.3.2025 12:51' then convert to ISO-like string
            if isinstance(dt, str) and re.match(r"^\d{1,2}\.\d{1,2}\.\d{4} \d{1,2}:\d{2}$", dt):
                return pd.to_datetime(dt, dayfirst=True, errors="coerce")
            return pd.to_datetime(dt, errors="coerce")

        df["datetime"] = df["datetime"].apply(_standardise)
        df = df.dropna(subset=["datetime"])                       # discard unparsable rows

        # ─────────────────────── 3. WEEKEND → MONDAY 09:00 ───────────────────
        def _weekend_to_monday(dt):
            # 5 = Saturday, 6 = Sunday
            if dt.weekday() in (5, 6):
                return (dt + timedelta(days=7 - dt.weekday())
                       ).replace(hour=9, minute=0, second=0, microsecond=0)
            return dt

        df["datetime"] = df["datetime"].apply(_weekend_to_monday)

        # ───────────────────────── 4. SORT CHRONOLOGICALLY ───────────────────
        df = df.sort_values("datetime").reset_index(drop=True)

        # ──────────────────────── 5. FILTER “YESTERDAY” ──────────────────────
        start_date = (datetime.now() - timedelta(days=30)).date()   # 00:00 yesterday
        end_date   =  datetime.now().date()                        # 00:00 today

        df_subset = df[(df["datetime"].dt.date >= start_date) &
                       (df["datetime"].dt.date <  end_date)]

        if df_subset.empty:
            raise ValueError("No data available for yesterday’s window; df_subset is empty.")

        # ───────────────────────── 6. SAVE AS JSON ───────────────────────────
        with open(TEMP_PROCESSED_JSON, "w") as f:
            json.dump(df_subset.to_dict(orient="records"), f, indent=4, default=str)
            print(f"✅ Processed data saved to {TEMP_PROCESSED_JSON}")

    except Exception as e:
        print(f"[Error in data preprocessing pipeline] -> {e}")
        sys.exit(1)
