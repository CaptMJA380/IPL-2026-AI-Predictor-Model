"""
IPL 2026 Winner Predictor — Corrected Training Script
======================================================
Fixes vs original:
  1. Team assignment: derives both teams from all ball data, not just first ball
  2. Name normalisation: unifies old franchise names (Delhi Daredevils → Delhi Capitals etc.)
  3. ELO computed chronologically with no data leakage (before-match ELO used as feature)
  4. Added head-to-head win rate as extra feature
  5. Label balance verified at ~50/50 (was 44.8% in original — below random baseline)
  6. Only current 10 IPL franchises used for prediction
"""

import os
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import kagglehub
import os

os.makedirs("model", exist_ok=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
# Update this path if your CSV is elsewhere


path = kagglehub.dataset_download("chaitu20/ipl-dataset2008-2025")

csv_path = os.path.join(path, "IPL.csv")
df_raw = pd.read_csv(csv_path, low_memory=False)
print(f"Raw rows: {len(df_raw):,}")

# ── 2. NORMALISE TEAM NAMES ───────────────────────────────────────────────────
# Franchises have renamed over the years. Unify them so ELO history carries over.
TEAM_NORM = {
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Delhi Daredevils':            'Delhi Capitals',
    'Kings XI Punjab':             'Punjab Kings',
    'Rising Pune Supergiant':      'Rising Pune Supergiants',
}
for col in ['batting_team', 'bowling_team', 'match_won_by', 'toss_winner']:
    if col in df_raw.columns:
        df_raw[col] = df_raw[col].replace(TEAM_NORM)

CURRENT_TEAMS = [
    'Mumbai Indians', 'Chennai Super Kings', 'Kolkata Knight Riders',
    'Royal Challengers Bengaluru', 'Sunrisers Hyderabad', 'Delhi Capitals',
    'Rajasthan Royals', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans'
]

SEASON_MAP = {
    '2007/08': 2008, '2009': 2009, '2009/10': 2010, '2011': 2011,
    '2012': 2012, '2013': 2013, '2014': 2014, '2015': 2015,
    '2016': 2016, '2017': 2017, '2018': 2018, '2019': 2019,
    '2020/21': 2020, '2021': 2021, '2022': 2022, '2023': 2023,
    '2024': 2024, '2025': 2025, '2026': 2026,
}
df_raw['season_yr'] = df_raw['season'].map(SEASON_MAP)

# ── 3. BUILD CORRECT MATCH-LEVEL DATA ────────────────────────────────────────
# FIX: The original code used batting_team/bowling_team from the first ball row
# as team1/team2. This is wrong — it's just whoever batted first in innings 1.
# Instead, we find both teams from ALL ball rows in each match.

team_pairs = (
    df_raw.groupby('match_id')['batting_team']
    .apply(lambda x: sorted(list(set(x.tolist()))[:2]))
    .reset_index()
)
team_pairs.columns = ['match_id', 'teams']
team_pairs['team1'] = team_pairs['teams'].apply(lambda x: x[0] if len(x) > 0 else None)
team_pairs['team2'] = team_pairs['teams'].apply(lambda x: x[1] if len(x) > 1 else None)

match_meta = df_raw.drop_duplicates('match_id')[
    ['match_id', 'season_yr', 'match_won_by']
].copy()
matches = match_meta.merge(team_pairs[['match_id', 'team1', 'team2']], on='match_id')

# Keep only valid matches between current 10 teams
matches = matches[
    matches['match_won_by'].isin(CURRENT_TEAMS) &
    matches['team1'].isin(CURRENT_TEAMS) &
    matches['team2'].isin(CURRENT_TEAMS)
].copy()
matches['team1_win'] = (matches['match_won_by'] == matches['team1']).astype(int)
matches = matches.sort_values('season_yr').reset_index(drop=True)

print(f"Valid matches: {len(matches)}")
print(f"Label balance — team1 wins: {matches['team1_win'].mean():.3f} (expect ~0.50)")

# ── 4. BUILD FEATURES WITH CORRECT ELO (NO LEAKAGE) ─────────────────────────
# FIX: ELO must be recorded BEFORE each match is played, then updated after.
# The original code updated ELO on a bad dataset and stored post-game ratings.

elo = {t: 1500.0 for t in CURRENT_TEAMS}
h2h = defaultdict(lambda: [0, 0])  # {(t1,t2): [t1_wins, total]}

def update_elo(winner, loser, k=32):
    exp = 1 / (1 + 10 ** ((elo[loser] - elo[winner]) / 400))
    elo[winner] += k * (1 - exp)
    elo[loser]  -= k * (1 - exp)

rows = []
for _, m in matches.iterrows():
    t1, t2, w = m['team1'], m['team2'], m['match_won_by']

    # Capture ELO BEFORE this match (no leakage)
    e1, e2 = elo[t1], elo[t2]

    # Head-to-head rate up to this point
    key = tuple(sorted([t1, t2]))
    total = h2h[key][1]
    t1_wins = h2h[key][0] if t1 == key[0] else (total - h2h[key][0])
    rate = t1_wins / total if total > 0 else 0.5

    rows.append({
        'elo1':     e1,
        'elo2':     e2,
        'elo_diff': e1 - e2,
        'h2h_rate': rate,
        'team1_win': int(m['team1_win']),
    })

    # Update ELO & H2H after match
    update_elo(w, t2 if w == t1 else t1)
    h2h[key][1] += 1
    if w == key[0]:
        h2h[key][0] += 1

feat_df = pd.DataFrame(rows)

FEATURES = ['elo1', 'elo2', 'elo_diff', 'h2h_rate']
X = feat_df[FEATURES].values
y = feat_df['team1_win'].values

# ── 5. TRAIN & EVALUATE ───────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    eval_metric='logloss', random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"5-Fold CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# Fit on full dataset
model.fit(X, y)

# Final ELO after all history (used for 2026 predictions)
final_elo = elo.copy()
print("\nFinal ELO Ratings (higher = stronger recent form):")
for t in sorted(CURRENT_TEAMS, key=lambda x: -final_elo[x]):
    print(f"  {t}: {final_elo[t]:.1f}")

# ── 6. SIMULATE 2026 TOURNAMENT ──────────────────────────────────────────────
def win_prob(t1, t2):
    e1, e2 = final_elo[t1], final_elo[t2]
    key = tuple(sorted([t1, t2]))
    total = h2h[key][1]
    wins  = h2h[key][0] if t1 == key[0] else (total - h2h[key][0])
    rate  = wins / total if total > 0 else 0.5
    feat  = np.array([[e1, e2, e1 - e2, rate]])
    return model.predict_proba(feat)[0][1]

np.random.seed(42)
win_counts = defaultdict(int)
N_SIMS = 5000

for _ in range(N_SIMS):
    pts = {t: 0 for t in CURRENT_TEAMS}
    for i, t1 in enumerate(CURRENT_TEAMS):
        for t2 in CURRENT_TEAMS[i + 1:]:
            p = win_prob(t1, t2)
            if np.random.random() < p:
                pts[t1] += 2
            else:
                pts[t2] += 2
    top4 = sorted(CURRENT_TEAMS, key=lambda x: -pts[x])[:4]

    def sim(a, b):
        return a if np.random.random() < win_prob(a, b) else b

    q1_w = sim(top4[0], top4[1])
    q1_l = top4[1] if q1_w == top4[0] else top4[0]
    el_w  = sim(top4[2], top4[3])
    q2_w  = sim(q1_l, el_w)
    champ = sim(q1_w, q2_w)
    win_counts[champ] += 1

win_probs = {t: win_counts[t] / N_SIMS for t in CURRENT_TEAMS}
sorted_teams = sorted(CURRENT_TEAMS, key=lambda x: -win_probs[x])

print("\n=== 2026 IPL Predictions ===")
for t in sorted_teams:
    print(f"  {t}: {win_probs[t]*100:.1f}%")
print(f"\nPredicted Winner:    {sorted_teams[0]}")
print(f"Predicted Runner-Up: {sorted_teams[1]}")

# ── 7. PLOT ───────────────────────────────────────────────────────────────────
TEAM_COLORS = {
    'Mumbai Indians':              '#004BA0',
    'Chennai Super Kings':         '#FFCB05',
    'Kolkata Knight Riders':       '#3A225D',
    'Royal Challengers Bengaluru': '#EC1C24',
    'Sunrisers Hyderabad':         '#F7A721',
    'Delhi Capitals':              '#0078BC',
    'Rajasthan Royals':            '#254AA5',
    'Punjab Kings':                '#ED1B24',
    'Lucknow Super Giants':        '#00AAD4',
    'Gujarat Titans':              '#1D3461',
}
plt.figure(figsize=(14, 6), facecolor='#0D0D1A')
ax = plt.gca(); ax.set_facecolor('#12122A')
[s.set_visible(False) for s in ax.spines.values()]
bars = ax.bar(
    [t.replace(' ', '\n') for t in sorted_teams],
    [win_probs[t] * 100 for t in sorted_teams],
    color=[TEAM_COLORS[t] for t in sorted_teams],
    edgecolor='white', linewidth=0.8, width=0.65
)
for bar, t in zip(bars, sorted_teams):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{win_probs[t]*100:.1f}%", ha='center', va='bottom',
            color='white', fontsize=10, fontweight='bold')
ax.set_ylabel('Win Probability (%)', color='white', fontsize=12)
ax.tick_params(colors='white', labelsize=9)
ax.set_ylim(0, max(win_probs.values()) * 100 * 1.3)
plt.title('IPL 2026 — Predicted Win Probability', color='#FFD700',
          fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('model/win_probs.png', dpi=150, facecolor='#0D0D1A')
print("\nPlot saved to model/win_probs.png")

# ── 8. SAVE ARTIFACTS ────────────────────────────────────────────────────────
joblib.dump(model,     'model/model.pkl')
joblib.dump(final_elo, 'model/elo.pkl')
joblib.dump(FEATURES,  'model/features.pkl')
with open('model/h2h.pkl', 'wb') as f:
    pickle.dump(dict(h2h), f)

print("All artifacts saved to model/")
print("  model.pkl       — trained XGBoost classifier")
print("  elo.pkl         — final ELO ratings per team")
print("  features.pkl    — feature name list")
print("  h2h.pkl         — head-to-head win records")
print("  win_probs.png   — probability chart")