import pandas as pd

# Load your Q1 summary
df = pd.read_csv("q1_attacker_ratio_summary.csv")

# Group by attack_method, alpha, client_num and compute mean Q1 ratio
q1_group_mean = df.groupby(['AttackMethod', 'Alpha', 'ClientNum'])['Q1_AttackerRatio'].mean().reset_index()

q1_group_mean.to_csv("q1_attacker_ratio_group_mean.csv", index=False)

print(q1_group_mean)
