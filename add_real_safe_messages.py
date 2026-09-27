"""
add_real_safe_messages.py
----------------------------
Appends real, legitimate M-Pesa confirmation messages into data.csv,
labeled "safe". These are genuine examples pulled from an actual
phone, which helps the model learn what real M-Pesa texts look like -
as opposed to fakes that try to imitate them.

Run:  python add_real_safe_messages.py
Then: python train_scamshield.py
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data.csv")

if not os.path.exists(DATA_PATH):
    raise SystemExit(f"Could not find data.csv at {DATA_PATH}")

df = pd.read_csv(DATA_PATH, encoding="utf-8")

if "v1" in df.columns and "v2" in df.columns:
    label_col, text_col, safe_label = "v1", "v2", "ham"
else:
    label_col, text_col, safe_label = "label", "text", "safe"

new_safe_messages = [
    "UIOHT7MN6Y Confirmed.You have received Ksh2,000.00 from MENTOR SACCO B2C 302500 on 24/9/26 at 1:46 PM New M-PESA balance is Ksh2,865.40. See all your balances now https://saf.cx/kWQpy.",
    "UIO8O80P1K Confirmed.You have received Ksh2,000.00 from Equity Bulk Account 300600 on 24/9/26 at 3:35 PM New M-PESA balance is Ksh2,676.00. See all your balances now https://saf.cx/kWQpy.",
    "UIRKR8FLMV Confirmed. Ksh20.00 sent to SAFARICOM POSTPAID BUNDLES for account SAFARICOM DATA BUNDLES on 27/9/26 at 11:27 AM. New M-PESA balance is Ksh67.18. Transaction cost, Ksh0.00.",
    "UIRKR8FE1N Confirmed. Ksh130.00 paid to NEWTON KANG'ETHE WAMBUI. on 27/9/26 at 10:20 AM. New M-PESA balance is Ksh87.18. Transaction cost, Ksh0.00. Amount you can transact within the day is 499,730.00. See all your balances now https://saf.cx/kWQpy",
    "UIRKR8F88H Confirmed.You have received Ksh200.00 from ZIIDI on 27/9/26 10:17 AM. New M-PESA balance is Ksh217.18. See all your balances now https://saf.cx/kWQpy.",
    "UIRKR8F3MT Confirmed. Ksh140.00 sent to Kennedy Makokha 0759825317 on 27/9/26 at 9:17 AM. New M-PESA balance is Ksh17.18. Transaction cost, Ksh7.00. Amount you can transact within the day is 499,860.00. See all your balances now https://saf.cx/iqIzU",
    "UIQKR8EAI6 Confirmed.You have received Ksh50.00 from ZIIDI on 26/9/26 10:34 PM. New M-PESA balance is Ksh117.18. See all your balances now https://saf.cx/kWQpy.",
]

new_rows = pd.DataFrame({
    text_col: new_safe_messages,
    label_col: [safe_label] * len(new_safe_messages),
})

for col in df.columns:
    if col not in new_rows.columns:
        new_rows[col] = ""
new_rows = new_rows[df.columns]

df_augmented = pd.concat([df, new_rows], ignore_index=True).drop_duplicates(subset=text_col)
df_augmented.to_csv(DATA_PATH, index=False, encoding="utf-8")

print(f"Added {len(new_safe_messages)} real safe M-Pesa messages.")
print(f"Total rows now: {len(df_augmented)} (was {len(df)})")
print("Now re-run: python train_scamshield.py")
