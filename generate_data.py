import pandas as pd
import numpy as np

teams = [
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
    "Delhi Capitals", "Kolkata Knight Riders", "Punjab Kings",
    "Rajasthan Royals", "Sunrisers Hyderabad", "Gujarat Titans", "Lucknow Super Giants"
]

data = []
for _ in range(1000):
    t1, t2 = np.random.choice(teams, 2, replace=False)
    # Give some teams higher win prob just to make the model learn something
    prob_t1 = 0.5
    if t1 == "Chennai Super Kings": prob_t1 += 0.2
    if t2 == "Chennai Super Kings": prob_t1 -= 0.2
    if t1 == "Mumbai Indians": prob_t1 += 0.15
    if t2 == "Mumbai Indians": prob_t1 -= 0.15
    
    winner = t1 if np.random.rand() < prob_t1 else t2
    data.append([t1, t2, winner])

df = pd.DataFrame(data, columns=["team1", "team2", "winner"])
df.to_csv("data/IPL.csv", index=False)
print("data/IPL.csv created")
