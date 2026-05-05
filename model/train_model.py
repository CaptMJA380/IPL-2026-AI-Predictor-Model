import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import joblib
import kagglehub
import os

sns.set_style("whitegrid")

# Download dataset
path = kagglehub.dataset_download("chaitu20/ipl-dataset2008-2025")
csv_path = os.path.join(path, "IPL.csv")

raw_df = pd.read_csv(csv_path, low_memory=False)
df = raw_df[['match_id', 'batting_team', 'bowling_team', 'match_won_by']].drop_duplicates(subset=['match_id']).copy()
df = df.rename(columns={'batting_team': 'team1', 'bowling_team': 'team2', 'match_won_by': 'winner'})
df = df.dropna(subset=["winner"])

df['team1'] = df['team1'].str.strip()
df['team2'] = df['team2'].str.strip()
df['winner'] = df['winner'].str.strip()

teams = sorted(list(set(df['team1']).union(set(df['team2']))))
df = df[df['winner'].isin(teams)]

elo = {team:1500 for team in teams}

def update_elo(winner, loser, k=32):
    expected = 1 / (1 + 10 ** ((elo[loser] - elo[winner]) / 400))
    elo[winner] += k * (1 - expected)
    elo[loser] += k * (0 - (1 - expected))

elo_history = []

for _, row in df.iterrows():
    t1, t2, winner = row['team1'], row['team2'], row['winner']
    loser = t2 if winner == t1 else t1
    update_elo(winner, loser)
    elo_history.append((t1, elo[t1], t2, elo[t2]))

elo_df = pd.DataFrame(elo_history, columns=['team1','elo1','team2','elo2'])

df['team1_win'] = (df['winner'] == df['team1']).astype(int)

team_to_idx = {team:i for i,team in enumerate(teams)}

df['team1_idx'] = df['team1'].map(team_to_idx)
df['team2_idx'] = df['team2'].map(team_to_idx)

df['elo1'] = elo_df['elo1']
df['elo2'] = elo_df['elo2']

X = df[['team1_idx','team2_idx','elo1','elo2']]
y = df['team1_win']

model = XGBClassifier(n_estimators=300, learning_rate=0.05)
model.fit(X, y)

def predict_prob(t1, t2):
    return model.predict_proba([[team_to_idx[t1], team_to_idx[t2], elo[t1], elo[t2]]])[0][1]

# Pre-calculate win probabilities for efficiency
prob_matrix = np.zeros((len(teams), len(teams)))
for i in range(len(teams)):
    for j in range(i + 1, len(teams)):
        p = predict_prob(teams[i], teams[j])
        prob_matrix[i, j] = p
        prob_matrix[j, i] = 1 - p

win_counts = {team: 0 for team in teams}
SIMS = 1000

for _ in range(SIMS):
    points = {team: 0 for team in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            if np.random.rand() < prob_matrix[i, j]:
                points[teams[i]] += 2
            else:
                points[teams[j]] += 2
    winner = max(points, key=points.get)
    win_counts[winner] += 1

win_probs = {k: v / SIMS for k, v in win_counts.items()}
teams_sorted = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)

labels = [x[0] for x in teams_sorted]
values = [x[1] for x in teams_sorted]

plt.figure(figsize=(12, 6))
plt.bar(labels, values)
plt.xticks(rotation=45, ha='right')
plt.title("IPL 2026 Winning Probability")
plt.tight_layout()
plt.savefig("model/win_probs.png")
print("Plot saved to model/win_probs.png")

top2 = teams_sorted[:2]
print("Winner:", top2[0][0])
print("Runner-up:", top2[1][0])

joblib.dump(model, "model/model.pkl")
joblib.dump(team_to_idx, "model/team_index.pkl")
joblib.dump(elo, "model/elo.pkl")
print("Models saved successfully")
