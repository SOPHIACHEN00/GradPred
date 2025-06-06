import os
import pandas as pd
import re

def parse_filename(filename):
    pattern = r"(?i)^contributions_(.*?)_(.*?)_([\d\.eE\+\-]+)_(\d+)_attacker(\d+)_trial(\d+)_attack"
    match = re.match(pattern, filename)
    if match:
        dataset, method, alpha, client_num, attacker_id, trial_id = match.groups()
        return {
            'Filename': filename,
            'Dataset': dataset,
            'AttackMethod': method,
            'Alpha': float(alpha),
            'ClientNum': int(client_num),
            'AttackerID': int(attacker_id),
            'TrialID': int(trial_id),
        }
    else:
        return None

def compute_attacker_ratio(file_path, attacker_id):
    # Compute Q1: Attacker's ratio of total WithAttacker contribution.
    try:
        df = pd.read_csv(file_path)
        total_contribution = df['WithAttacker'].sum()
        attacker_value = df.loc[df['ClientID'] == attacker_id, 'WithAttacker'].values

        if len(attacker_value) != 1 or pd.isna(attacker_value[0]):
            return None

        ratio = attacker_value[0] / total_contribution
        return ratio
    except Exception as e:
        print(f"Error in file {file_path}: {e}")
        return None

def process_folder_for_q1(folder_path, output_csv_path):
    # Process all CSVs in a folder to compute attacker ratios (Q1) and save summary.
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            meta = parse_filename(filename)
            if meta is None:
                print(f"Skipping file: {filename}")
                continue

            file_path = os.path.join(folder_path, filename)
            ratio = compute_attacker_ratio(file_path, meta['AttackerID'])
            if ratio is not None:
                meta['Q1_AttackerRatio'] = ratio
                results.append(meta)

    df_result = pd.DataFrame(results)
    df_result.to_csv(output_csv_path, index=False)
    print(f"\n Q1 ratio summary saved to: {output_csv_path}")

# === Configuration ===
input_folder = r"D:\xxx\xxx\TicTacToe" # Modify this line
output_file = os.path.join(input_folder, "q1_attacker_ratio_summary.csv")

# Run
process_folder_for_q1(input_folder, output_file)
