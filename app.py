import streamlit as st
import requests
import pandas as pd
import math
import os

st.title("Advanced Prospect Analytics & Trajectory")
st.write("Multi-dimensional predictive modeling for Drafted and Undrafted players.")

# We add our undrafted players to the test menu. We give them a dummy ID of "0" so 
# the code knows not to look for them in the NHL API.
test_players = {
    "Brayden Yager (WPG)": 8484242,
    "Macklin Celebrini (SJS)": 8484801,
    "Gavin McKenna (Undrafted)": 0,
    "Landon DuPont (Undrafted)": 0
}

selected_player_name = st.selectbox("Choose a Prospect:", list(test_players.keys()))
player_id = test_players[selected_player_name]

nhle_factors = {
    "WHL": 0.32, "OHL": 0.32, "QMJHL": 0.30, 
    "AHL": 0.47, "NCAA": 0.43, "NHL": 1.00, "CSSHL": 0.15
}

historical_comps = [
    {"Name": "Connor McDavid (Historical)", "NHLe": 68.0, "Age": 16, "SGP": 4.5, "Ceiling": "Generational"},
    {"Name": "Nathan MacKinnon (Historical)", "NHLe": 55.0, "Age": 16, "SGP": 4.2, "Ceiling": "Franchise Player"},
    {"Name": "Bo Horvat (Historical)", "NHLe": 38.0, "Age": 19, "SGP": 3.2, "Ceiling": "Top 6 Forward"},
    {"Name": "Vincent Trocheck (Historical)", "NHLe": 42.0, "Age": 19, "SGP": 3.5, "Ceiling": "Top 6 Forward"},
    {"Name": "AHL Journeyman (Historical)", "NHLe": 22.0, "Age": 20, "SGP": 1.8, "Ceiling": "AHL / Fringe NHL"}
]

WEIGHT_NHLE = 1.0     
WEIGHT_AGE = 3.0      
WEIGHT_SGP = 1.5      

if st.button("Run Advanced Analytics"):
    
    # --- HYBRID ROUTING ENGINE ---
    
    # ROUTE A: If ID is 0, they are undrafted. Look in our custom CSV database.
    if player_id == 0:
        st.info("Undrafted Player Detected. Routing to Custom Offline Database...")
        
        # Load our custom database
        if os.path.exists("prospect_db.csv"):
            db = pd.read_csv("prospect_db.csv")
            
            # Clean the selected name to match the database (removing the "(Undrafted)" tag)
            clean_name = selected_player_name.split(" (")[0]
            
            # Find the player in the database
            player_data = db[db['Name'] == clean_name]
            
            if not player_data.empty:
                # Extract their stats directly from our CSV
                prospect_age = int(player_data['Age'].values[0])
                league = player_data['League'].values[0]
                gp = int(player_data['GP'].values[0])
                pts = int(player_data['PTS'].values[0])
                prospect_sgp = float(player_data['SGP'].values[0])
                
                # Calculate NHLe
                ppg = pts / gp
                factor = nhle_factors.get(league, 0)
                prospect_nhle = round(ppg * factor * 82, 1)
                
                st.success(f"Successfully loaded data for {clean_name}")
                st.write(f"**Age:** {prospect_age} | **League:** {league} | **Points:** {pts} | **NHLe:** {prospect_nhle}")
                
            else:
                st.error("Player not found in custom database.")
                st.stop()
        else:
            st.error("Database file missing.")
            st.stop()

    # ROUTE B: If they have an ID, they are drafted. Hit the live NHL API.
    else:
        st.info("Drafted Player Detected. Routing to Live NHL API...")
        url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            birth_year = int(data.get("birthDate", "2000-01-01").split("-")[0])
            season_totals = data.get("seasonTotals", [])
            
            # Grab the most recent season for the math
            latest_season = season_totals[-1] 
            gp = latest_season.get("gamesPlayed", 1)
            pts = latest_season.get("points", 0)
            shots = latest_season.get("shots", 0)
            league = latest_season.get("leagueAbbrev", "N/A")
            
            season_year = int(str(latest_season.get("season", "00000000"))[:4])
            prospect_age = season_year - birth_year
            
            ppg = pts / gp
            prospect_sgp = round(shots / gp, 2) if shots > 0 else 2.0
            factor = nhle_factors.get(league, 0) 
            prospect_nhle = round(ppg * factor * 82, 1)
            
            st.success("Successfully pulled live NHL data.")
            st.write(f"**Age:** {prospect_age} | **League:** {league} | **Points:** {pts} | **NHLe:** {prospect_nhle}")
            
        else:
            st.error("API Error.")
            st.stop()

    # --- THE MACHINE LEARNING ENGINE (Runs for both Routes!) ---
    st.write("---")
    st.write("### AI Projection: Weighted KNN Algorithm")
    
    for comp in historical_comps:
        dist_nhle = WEIGHT_NHLE * (prospect_nhle - comp["NHLe"])**2
        dist_age = WEIGHT_AGE * (prospect_age - comp["Age"])**2
        dist_sgp = WEIGHT_SGP * (prospect_sgp - comp["SGP"])**2
        
        distance = math.sqrt(dist_nhle + dist_age + dist_sgp)
        comp["Distance"] = round(distance, 2)
    
    sorted_comps = sorted(historical_comps, key=lambda x: x["Distance"])
    match_1 = sorted_comps[0]
    
    st.success(f"**Top Historical Match:** {match_1['Name']} (Distance Score: {match_1['Distance']})")
    st.write(f"**Projected Ceiling:** {match_1['Ceiling']}")
