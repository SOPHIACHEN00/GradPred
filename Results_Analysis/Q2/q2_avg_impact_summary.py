import pandas as pd

# Load your Q2 summary
df = pd.read_csv("q2_avg_impact_summary.csv")

# Group by attack_method, alpha, client_num and take mean Q2
q2_group_mean = df.groupby(['AttackMethod', 'Alpha', 'ClientNum'])['Q2_AvgImpact'].mean().reset_index()

q2_group_mean.to_csv("q2_avg_impact_group_mean.csv", index=False)

print(q2_group_mean)
