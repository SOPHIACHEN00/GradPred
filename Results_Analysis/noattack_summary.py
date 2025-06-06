import os
import re
import pandas as pd

def parse_filename(filename):
    """
    Extract metadata from noattack filename.
    Example: contributions_tictactoe_arima_10.0_2_attacker0_trial0_noattack.csv
    """
    pattern = r"^contributions_(.*?)_(.*?)_([\d\.eE\+\-]+)_(\d+)_attacker\d+_trial\d+_noattack"
    match = re.match(pattern, filename)
    if match:
        dataset, attack_method, alpha, client_num = match.groups()
        return {
            'Dataset': dataset,
            'AttackMethod': attack_method,
            'Alpha': float(alpha),
            'ClientNum': int(client_num)
        }
    return None

def merge_noattack_contributions(folder_path, output_path):
    """
    Merge noattack contributions and keep only selected fields:
    Dataset, AttackMethod, Alpha, ClientNum, ClientID, Contribution (from WithoutAttacker)
    """
    merged_rows = []

    for filename in os.listdir(folder_path):
        if filename.startswith("contributions_") and filename.endswith("_noattack.csv"):
            meta = parse_filename(filename)
            if meta is None:
                print(f"Skipped: {filename}")
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)

                # Validate necessary columns exist
                if 'ClientID' not in df.columns or 'WithoutAttacker' not in df.columns:
                    print(f"Missing columns in {filename}, skipping.")
                    continue

                for _, row in df.iterrows():
                    merged_rows.append({
                        'Dataset': meta['Dataset'],
                        'AttackMethod': meta['AttackMethod'],
                        'Alpha': meta['Alpha'],
                        'ClientNum': meta['ClientNum'],
                        'ClientID': row['ClientID'],
                        'Contribution': row['WithoutAttacker']
                    })
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    merged_df = pd.DataFrame(merged_rows)
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged file saved to: {output_path}")

# === Your folder path ===
input_folder = r"D:\Study\Graduate_Study_WLU\Research\Attack_Analysis\Contribution\TicTacToe"
output_csv = os.path.join(input_folder, "noattack_contributions.csv")

# Run the merging process
merge_noattack_contributions(input_folder, output_csv)
