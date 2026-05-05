import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import kagglehub
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="IPL 2026 Predictor", layout="wide")

# ---------------- LOGOS ----------------
LOGO_MAP = {
    "Chennai Super Kings": "https://documents.iplt20.com/ipl/CSK/Logos/Logooutline/CSKoutline.png",
    "Mumbai Indians": "https://documents.iplt20.com/ipl/MI/Logos/Logooutline/MIoutline.png",
    "Royal Challengers Bangalore": "https://documents.iplt20.com/ipl/RCB/Logos/Logooutline/RCBoutline.png",
    "Royal Challengers Bengaluru": "https://documents.iplt20.com/ipl/RCB/Logos/Logooutline/RCBoutline.png",
    "Kolkata Knight Riders": "https://documents.iplt20.com/ipl/KKR/Logos/Logooutline/KKRoutline.png",
    "Delhi Capitals": "https://documents.iplt20.com/ipl/DC/Logos/Logooutline/DCoutline.png",
    "Delhi Daredevils": "https://documents.iplt20.com/ipl/DC/Logos/Logooutline/DCoutline.png",
    "Punjab Kings": "https://documents.iplt20.com/ipl/PBKS/Logos/Logooutline/PBKSoutline.png",
    "Kings XI Punjab": "https://documents.iplt20.com/ipl/PBKS/Logos/Logooutline/PBKSoutline.png",
    "Rajasthan Royals": "https://documents.iplt20.com/ipl/RR/Logos/Logooutline/RRoutline.png",
    "Sunrisers Hyderabad": "https://documents.iplt20.com/ipl/SRH/Logos/Logooutline/SRHoutline.png",
    "Gujarat Titans": "https://documents.iplt20.com/ipl/GT/Logos/Logooutline/GToutline.png",
    "Lucknow Super Giants": "https://documents.iplt20.com/ipl/LSG/Logos/Logooutline/LSGoutline.png",
    "Deccan Chargers": "https://documents.iplt20.com/ipl/SRH/Logos/Logooutline/SRHoutline.png", # Fallback
    "Gujarat Lions": "https://documents.iplt20.com/ipl/GT/Logos/Logooutline/GToutline.png", # Fallback
    "Pune Warriors": "https://documents.iplt20.com/ipl/LSG/Logos/Logooutline/LSGoutline.png", # Fallback
    "Rising Pune Supergiant": "https://documents.iplt20.com/ipl/LSG/Logos/Logooutline/LSGoutline.png", # Fallback
    "Rising Pune Supergiants": "https://documents.iplt20.com/ipl/LSG/Logos/Logooutline/LSGoutline.png", # Fallback
    "Kochi Tuskers Kerala": "https://upload.wikimedia.org/wikipedia/en/thumb/f/f7/Kochi_Tuskers_Kerala_Logo.svg/200px-Kochi_Tuskers_Kerala_Logo.svg.png" # Wikimedia fallback
}
DEFAULT_LOGO = "https://documents.iplt20.com/ipl/IPL/Logos/Logooutline/IPLoutline.png"

# ---------------- CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700;900&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif;
        background: radial-gradient(circle at top left, #1e293b, #0f172a);
        color: #f8fafc;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 900 !important;
        font-size: 2.2rem !important;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        padding: 25px !important;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .stMetric:hover {
        transform: scale(1.02);
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(25px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebar"] section[data-testid="stSidebarNav"] span {
        color: #f8fafc !important;
    }
    
    h1 {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #fbbf24, #f59e0b, #ef4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        letter-spacing: -1px;
        text-align: center;
        padding: 20px 0;
    }

    .team-logo {
        width: 100px;
        height: 100px;
        object-fit: contain;
        margin-bottom: 15px;
        filter: drop-shadow(0 0 10px rgba(255,255,255,0.2));
    }

    .fade-in {
        animation: fadeIn 0.8s ease-out forwards;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    div[data-testid="stVerticalBlock"] > div {
        animation: fadeIn 0.8s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ----------------
model = joblib.load("model/model.pkl")
team_to_idx = joblib.load("model/team_index.pkl")
elo = joblib.load("model/elo.pkl")

# Download dataset
path = kagglehub.dataset_download("chaitu20/ipl-dataset2008-2025")
csv_path = os.path.join(path, "IPL.csv")

raw_df = pd.read_csv(csv_path, low_memory=False)
df = raw_df[['match_id', 'batting_team', 'bowling_team', 'match_won_by']].drop_duplicates(subset=['match_id']).copy()
df = df.rename(columns={'batting_team': 'team1', 'bowling_team': 'team2', 'match_won_by': 'winner'})
df = df.dropna(subset=["winner"])

# Clean team names
df['team1'] = df['team1'].str.strip()
df['team2'] = df['team2'].str.strip()
df['winner'] = df['winner'].str.strip()

teams = sorted(list(set(df['team1']).union(set(df['team2']))))
df = df[df['winner'].isin(teams)]

# ---------------- PREDICT FUNCTION ----------------
def predict_prob(t1, t2):
    if t1 not in elo or t2 not in elo:
        return 0.5  # fallback (neutral)

    i1 = team_to_idx.get(t1, 0)
    i2 = team_to_idx.get(t2, 0)

    e1 = elo[t1]
    e2 = elo[t2]

    return model.predict_proba([[i1, i2, e1, e2]])[0][1]

# ---------------- UI ----------------
st.markdown("<h1>🏏 IPL 2026 AI Predictor</h1>", unsafe_allow_html=True)

mode = st.sidebar.selectbox("Mode", ["🏆 Tournament", "⚔️ Match"])

# ================= TOURNAMENT =================
if mode == "🏆 Tournament":

    st.subheader("🏆 Tournament Winner Prediction")

    win_counts = {t: 0 for t in teams}

    SIMS = 500

    # Pre-calculate win probabilities for efficiency
    prob_matrix = np.zeros((len(teams), len(teams)))
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            prob_matrix[i, j] = predict_prob(teams[i], teams[j])
            prob_matrix[j, i] = 1 - prob_matrix[i, j]

    for _ in range(SIMS):
        pts = {t: 0 for t in teams}
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                if np.random.rand() < prob_matrix[i, j]:
                    pts[teams[i]] += 2
                else:
                    pts[teams[j]] += 2

        winner = max(pts, key=pts.get)
        win_counts[winner] += 1

    probs = {k: v / SIMS for k, v in win_counts.items()}
    sorted_probs = dict(sorted(probs.items(), key=lambda x: -x[1]))

    winner = list(sorted_probs.keys())[0]
    runner = list(sorted_probs.keys())[1]

    col1, col2 = st.columns(2)
    
    with col1:
        st.image(LOGO_MAP.get(winner, DEFAULT_LOGO), width=100)
        st.metric("🏆 Projected Winner", winner)
        
    with col2:
        st.image(LOGO_MAP.get(runner, DEFAULT_LOGO), width=100)
        st.metric("🥈 Projected Runner-Up", runner)

    # Plot
    fig = px.bar(
        x=list(sorted_probs.keys()),
        y=list(sorted_probs.values()),
        color=list(sorted_probs.keys()),
        title="Winning Probability",
    )

    fig.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ================= MATCH =================
elif mode == "⚔️ Match":

    st.subheader("⚔️ Match Predictor")

    t1 = st.selectbox("Team 1", teams)
    t2 = st.selectbox("Team 2", teams)

    if t1 != t2:
        prob = predict_prob(t1, t2)

        col1, col2 = st.columns(2)
        
        with col1:
            st.image(LOGO_MAP.get(t1, DEFAULT_LOGO), width=120)
            st.metric(t1, f"{prob*100:.2f}%")
            
        with col2:
            st.image(LOGO_MAP.get(t2, DEFAULT_LOGO), width=120)
            st.metric(t2, f"{(1-prob)*100:.2f}%")