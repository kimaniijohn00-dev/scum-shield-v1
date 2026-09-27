"""
augment_data.py
-----------------
Appends new labeled scam messages into your existing data.csv, matching
whatever column format it already uses (v1/v2 from Kaggle, or text/label).

Run this BEFORE re-running train_scamshield.py, so the model actually
learns from the new examples.

Usage:  python augment_data.py
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data.csv")

if not os.path.exists(DATA_PATH):
    raise SystemExit(f"Could not find data.csv at {DATA_PATH}")

df = pd.read_csv(DATA_PATH, encoding="latin-1")

# match whatever column naming your dataset already uses
if "v1" in df.columns and "v2" in df.columns:
    label_col, text_col, scam_label = "v1", "v2", "spam"
else:
    label_col, text_col, scam_label = "label", "text", "scam"

# ---- ADD NEW SCAM EXAMPLES HERE ----
new_scam_messages = [
    "Hi.MADAM Rehema withdraw 4.8 million via bank of baroda acc:543216540 pin :563846. clear 6 containers at K.P.A Japan",
    "congratulations you have received ksh 100,000 from safaricom 25th anniversary promo. send ksh 500 processing fee to claim your cash",
    "Hallo madam REHEMA, have sent 6.8M via western union, clear containers (JX8213) secret word\"blue, M.TC.password is (5225685) (DKT DENNIS TZ. SMS 0101442617",
    "CONFIRMED! MPESA 254111532689 has been verified to receive a fund of KSH 25,200 on 2nd disbursement.Visit nyottakeplus.co.ke and withdraw. Complete now",
    "Received KES 450 registration bonus QBET balance: KES 450 Register & share user ID with customer care for awarding https://cutt.ly/3yivX2M6",
]
# -------------------------------------

new_rows = pd.DataFrame({
    text_col: new_scam_messages,
    label_col: [scam_label] * len(new_scam_messages),
})

# keep any extra columns the original file has (e.g. Kaggle's blank
# "Unnamed: 2/3/4" columns), filled empty so concat lines up cleanly
for col in df.columns:
    if col not in new_rows.columns:
        new_rows[col] = ""
new_rows = new_rows[df.columns]

df_augmented = pd.concat([df, new_rows], ignore_index=True)
df_augmented.to_csv(DATA_PATH, index=False, encoding="latin-1")

print(f"Added {len(new_scam_messages)} new scam examples.")
print(f"Total rows now: {len(df_augmented)} (was {len(df)})")
print("Now re-run: python train_scamshield.py")
