"""
fix_bad_merge.py
------------------
Removes rows from data.csv where the "text" is actually an ID/token
(no spaces at all) rather than a real message - this cleans up the
mpesa_synthetic.csv mis-merge (transaction_id got used as message text).

Run:  python fix_bad_merge.py
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data.csv")

df = pd.read_csv(DATA_PATH, encoding="utf-8")
before = len(df)

# a real SMS/email/WhatsApp message almost always has at least one space.
# a bare transaction ID / token does not.
looks_like_id = ~df["text"].astype(str).str.contains(r"\s", regex=True)
removed = df[looks_like_id]
df = df[~looks_like_id]

print(f"Removed {len(removed)} ID-like rows (no spaces) out of {before}.")
print("Sample of what was removed:")
print(removed["text"].head(5).to_string(index=False))

df.to_csv(DATA_PATH, index=False, encoding="utf-8")
print(f"\ndata.csv now has {len(df)} rows.")
