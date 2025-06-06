import os
import pandas as pd
import re

def parse_filename(filename):
    pattern = r"^contributions_(.*?)_(.*?)_([\d\.eE\+\-]+)_(\d+)_attacker(\d+)_trial(\d+)_attack"
    match = re.match(pattern, filename)
    if match:
        dataset, attack_method, alpha, client_num, attacker_id, trial_id = match.groups()
        return {
            'Filename': filename,
            'Dataset': dataset,
            'AttackMethod': attack_method,
            'Alpha': float(alpha),
            'ClientNum': int(client_num),
            'AttackerID': int(attacker_id),
            'TrialID': int(trial_id),
        }
    else:
        print(f"Skipping unrecognized or noattack file: {filename}")
        return None

def compute_avg_impact(file_path):
    # Compute Q2: average absolute delta between WithAttacker and WithoutAttacker
    try:
        df = pd.read_csv(file_path)
        df_clean = df.dropna(subset=['WithAttacker', 'WithoutAttacker']).copy()
        df_clean['AbsDelta'] = (df_clean['WithAttacker'] - df_clean['WithoutAttacker']).abs()
        return df_clean['AbsDelta'].mean()
    except Exception as e:
        print(f"Error in file {file_path}: {e}")
        return None

def process_folder(folder_path, output_csv_path):
    # Process all valid CSV files in a folder, calculate Q2 for each, and save summary.
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            meta = parse_filename(filename)
            if meta is None:
                continue 

            file_path = os.path.join(folder_path, filename)
            meta['Q2_AvgImpact'] = compute_avg_impact(file_path)
            results.append(meta)

    df_result = pd.DataFrame(results)
    df_result.to_csv(output_csv_path, index=False)
    print(f"\n Summary saved to: {output_csv_path}\n")

# === Your folder and output file ===
input_folder = r"D:\xxx\xxx\TicTacToe" # Modify this line
output_file = os.path.join(input_folder, "q2_avg_impact_summary.csv")

# Run
process_folder(input_folder, output_file)
