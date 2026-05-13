import streamlit as st
import requests
import pandas as pd
import math

st.title("Advanced Prospect Analytics & Trajectory")
st.write("Multi-dimensional predictive modeling using NHLe, Age, and Shot Volume.")

test_players = {
    "Brayden Yager (WPG)": 8484242,
    "Connor Bedard (CHI)": 8484144,
    "Macklin Celebrini (SJS)": 8484801,
    "Will Smith (SJS)": 8484227 
}

selected_player_name = st.selectbox("Choose a Prospect:", list(test_players.keys()))
player_id = test_players[selected_player_name]

nhle_factors = {
    "WHL": 0.32, "OHL": 0.32, "QMJHL": 0.30, 
    "AHL": 0.47, "NCAA": 0.43, "NHL": 1.00,
    "SHL": 0.58, "Liiga": 0.43 # Added European leagues for future-proofing
}

# --- UPGRADED MULTI-DIMENSIONAL MOCK DATABASE ---
historical_comps = [
    {"Name": "Bo Horvat (Historical)", "NHLe": 38.0, "PPG": 1.10, "Age": 19, "SGP": 3.2, "Ceiling": "Top 6 Forward"},
    {"Name": "Vincent Trocheck (Historical)", "NHLe": 42.0, "PPG": 1.25, "Age": 19, "SGP": 3.5, "Ceiling": "Top 6 Forward"},
    {"Name": "AHL Journeyman (Historical)", "NHLe": 22.0, "PPG": 0.65, "Age": 20, "SGP": 1.8, "Ceiling": "AHL / Fringe NHL"},
    {"Name": "Elite Superstar (Historical)", "NHLe": 65.0, "PPG": 1.80, "Age": 18, "SGP": 4.5, "Ceiling": "Franchise Player"},
    {"Name": "Late Bloomer (Historical)", "NHLe": 45.0, "PPG": 1.30, "Age": 21, "SGP": 2.9, "Ceiling": "Middle 6 Forward"}
]

# --- ALGORITHM WEIGHTS ---
# We tell the AI which stats matter most. 
WEIGHT_NHLE = 1.0     # Baseline
WEIGHT_AGE = 3.0      # Age differences are penalized heavily
WEIGHT_SGP = 1.5      # Shot volume is a strong indicator of translatable offense

if st.button("Run Advanced Analytics"):
    
    url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        
        # Extract birth year to calculate age for each season
        birth_date = data.get("birthDate", "2000-01-01")
        birth_year = int(birth_date.split("-")[0])
        
        season_totals = data.get("seasonTotals", [])
        stats_list = []
        
        for season in season_totals:
            league = season.get("leagueAbbrev", "N/A")
            gp = season.get("gamesPlayed", 0)
            points = season.get("points", 0)
            shots = season.get("shots", 0) # Pulling raw shots
            
            if gp == 0: continue
                
            # Calculate the year the season started to determine the player's age
            season_year_str = str(season.get("season", "00000000"))
            if len(season_year_str) == 8:
                season_start_year = int(season_year_str[:4])
                player_age = season_start_year - birth_year
            else:
                player_age = "N/A"
                
            ppg = points / gp
            sgp = shots / gp if shots > 0 else 0 # Shots Per Game
            factor = nhle_factors.get(league, 0) 
            nhle = ppg * factor * 82
            
            stats_list.append({
                "Season": season_year_str,
                "Age": player_age,
                "League": league,
                "GP": gp, 
                "PTS": points, 
                "PPG": round(ppg, 2),
                "S/GP": round(sgp, 2) if sgp > 0 else "N/A", # Show N/A if league didn't track shots
                "NHLe": round(nhle, 1) if factor > 0 else None
            })
            
        df = pd.DataFrame(stats_list)
        df['Season'] = df['Season'].astype(str)

        st.subheader(f"Multi-Dimensional Data for {selected_player_name}")
        st.dataframe(df, use_container_width=True)
        
        # --- THE ADVANCED MACHINE LEARNING COMPONENT ---
        chart_data = df.dropna(subset=['NHLe'])
        if not chart_data.empty:
            
            st.write("---")
            st.write("### AI Projection: Weighted KNN Algorithm")
            
            # Grab the prospect's most recent valid season
            latest_season = chart_data.iloc[-1]
            prospect_nhle = latest_season['NHLe']
            prospect_age = latest_season['Age']
            prospect_sgp = latest_season['S/GP']
            
            # Fallback if shots weren't tracked in that specific league
            if prospect_sgp == "N/A": 
                prospect_sgp = 2.0 
            
            st.info(f"**Engine Input Vectors:** Age: {prospect_age} | NHLe: {prospect_nhle} | Shots/GP: {prospect_sgp}")
            
            for comp in historical_comps:
                # The Weighted Euclidean Distance Formula
                dist_nhle = WEIGHT_NHLE * (prospect_nhle - comp["NHLe"])**2
                dist_age = WEIGHT_AGE * (prospect_age - comp["Age"])**2
                dist_sgp = WEIGHT_SGP * (prospect_sgp - comp["SGP"])**2
                
                distance = math.sqrt(dist_nhle + dist_age + dist_sgp)
                comp["Distance"] = round(distance, 2)
            
            sorted_comps = sorted(historical_comps, key=lambda x: x["Distance"])
            
            match_1 = sorted_comps[0]
            st.success(f"**Top Historical Match:** {match_1['Name']} (Score: {match_1['Distance']})")
            st.write(f"**Projected Ceiling:** {match_1['Ceiling']}")
            
    else:
        st.error("Error connecting to the NHL API.")
