import streamlit as st
import requests
import pandas as pd
import math

st.title("Prospect Trajectory & NHLe Tracker")
st.write("Select a player below to calculate their NHLe and Machine Learning Similarity Scores.")

test_players = {
    "Brayden Yager (WPG)": 8484242,
    "Connor Bedard (CHI)": 8484144,
    "Macklin Celebrini (SJS)": 8484801,
    "Will Smith (SJS)": 8484227 # The corrected ID!
}

selected_player_name = st.selectbox("Choose a Prospect:", list(test_players.keys()))
player_id = test_players[selected_player_name]

nhle_factors = {
    "WHL": 0.32, "OHL": 0.32, "QMJHL": 0.30, 
    "AHL": 0.47, "NCAA": 0.43, "NHL": 1.00
}

# --- OUR MOCK HISTORICAL DATABASE FOR MACHINE LEARNING ---
# In a final app, this would be thousands of rows from a real database.
historical_comps = [
    {"Name": "Bo Horvat (Historical)", "NHLe": 38.0, "PPG": 1.10, "Ceiling": "Top 6 Forward"},
    {"Name": "Vincent Trocheck (Historical)", "NHLe": 42.0, "PPG": 1.25, "Ceiling": "Top 6 Forward"},
    {"Name": "AHL Journeyman (Historical)", "NHLe": 22.0, "PPG": 0.65, "Ceiling": "AHL / Fringe NHL"},
    {"Name": "Elite Superstar (Historical)", "NHLe": 65.0, "PPG": 1.80, "Ceiling": "Franchise Player"},
    {"Name": "Solid 3rd Liner (Historical)", "NHLe": 30.0, "PPG": 0.90, "Ceiling": "Bottom 6 Forward"}
]

if st.button("Run Analytics"):
    
    url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        season_totals = data.get("seasonTotals", [])
        
        stats_list = []
        
        for season in season_totals:
            league = season.get("leagueAbbrev", "N/A")
            gp = season.get("gamesPlayed", 0)
            points = season.get("points", 0)
            
            if gp == 0: continue
                
            ppg = points / gp
            factor = nhle_factors.get(league, 0) 
            nhle = ppg * factor * 82
            
            stats_list.append({
                "Season": season.get("season", "N/A"),
                "League": league,
                "Team": season.get("teamName", {}).get("default", "N/A"),
                "GP": gp, "PTS": points, "PPG": round(ppg, 2),
                "NHLe Projection": round(nhle, 1) if factor > 0 else None
            })
            
        df = pd.DataFrame(stats_list)
        df['Season'] = df['Season'].astype(str)

        st.subheader(f"Data for {selected_player_name}")
        st.dataframe(df, use_container_width=True)
        
        chart_data = df.dropna(subset=['NHLe Projection'])
        if not chart_data.empty:
            st.write("### NHLe Trajectory Over Time")
            st.line_chart(chart_data.set_index('Season')['NHLe Projection'])
            
            # --- THE MACHINE LEARNING COMPONENT ---
            st.write("---")
            st.write("### Predictive AI: Similarity Scores")
            st.write("Using K-Nearest Neighbors (KNN) to find historical matches based on their most recent season.")
            
            # Grab the prospect's most recent valid season for the calculation
            latest_season = chart_data.iloc[-1]
            prospect_nhle = latest_season['NHLe Projection']
            prospect_ppg = latest_season['PPG']
            
            st.info(f"**Target Metric:** Calculating distances using {selected_player_name}'s recent {prospect_nhle} NHLe and {prospect_ppg} PPG.")
            
            # Run the Euclidean Distance math against our historical database
            for comp in historical_comps:
                # The KNN Distance Formula
                distance = math.sqrt((prospect_nhle - comp["NHLe"])**2 + (prospect_ppg - comp["PPG"])**2)
                comp["Distance"] = round(distance, 2)
            
            # Sort the historical players so the smallest distance (closest match) is at the top
            sorted_comps = sorted(historical_comps, key=lambda x: x["Distance"])
            
            # Display the top 2 closest matches
            match_1 = sorted_comps[0]
            match_2 = sorted_comps[1]
            
            st.success(f"**Primary Match:** {match_1['Name']} (Distance Score: {match_1['Distance']})")
            st.write(f"**Projected Ceiling:** {match_1['Ceiling']}")
            
            st.warning(f"**Secondary Match:** {match_2['Name']} (Distance Score: {match_2['Distance']})")
            st.write(f"**Projected Floor:** {match_2['Ceiling']}")
            
    else:
        st.error("Error connecting to the NHL API.")
