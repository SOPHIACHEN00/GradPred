import os
import pandas as pd
import re

def parse_filename(filename):
    pattern = r"^attacking_time_(.*?)_(.*?)_([\d\.eE\+\-]+)_(\d+)_attacker(\d+)_trial(\d+)_attack"
    match = re.match(pattern, filename)
    if match:
        dataset, method, alpha, clients, attacker_id, trial_id = match.groups()
        return {
            'Filename': filename,
            'Dataset': dataset,
            'AttackMethod': method,
            'Alpha': float(alpha),
            'ClientNum': int(clients),
            'AttackerID': int(attacker_id),
            'TrialID': int(trial_id)
        }
    else:
        return None

def process_attack_time_folder(folder_path, output_summary_csv, output_group_csv):
    records = []

    for fname in os.listdir(folder_path):
        if not fname.endswith(".csv") or not fname.startswith("attacking_time_"):
            continue

        meta = parse_filename(fname)
        if not meta:
            print(f"Skipping unrecognized file: {fname}")
            continue

        full_path = os.path.join(folder_path, fname)
        try:
            df = pd.read_csv(full_path)
            # Look for the right column: "WithAttackerSec"
            if "WithAttackerSec" in df.columns:
                attack_time = float(df["WithAttackerSec"].values[0])
                meta["AttackingTime"] = attack_time
                records.append(meta)
            else:
                print(f"Missing WithAttackerSec in: {fname}")
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    df_all = pd.DataFrame(records)
    df_all.to_csv(output_summary_csv, index=False)

    df_group = df_all.groupby(["AttackMethod", "Alpha", "ClientNum", "TrialID"])["AttackingTime"].mean().reset_index()
    df_group.to_csv(output_group_csv, index=False)

    print(f"\nSaved detailed summary to: {output_summary_csv}")
    print(f"Saved grouped mean to: {output_group_csv}")

# === Configuration ===
input_folder = r"D:\xxx\xxx\TicTacToe" # Modify this line
output_csv_detail = os.path.join(input_folder, "q3_avg_attacking_time_summary.csv")
output_csv_group = os.path.join(input_folder, "q3_avg_attacking_time_summary_group_mean.csv")

process_attack_time_folder(input_folder, output_csv_detail, output_csv_group)
