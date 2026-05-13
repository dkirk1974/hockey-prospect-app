import streamlit as st
import requests
import pandas as pd

# Set the title of your web app
st.title("Prospect Trajectory & NHLe Tracker")
st.write("Select a player below to instantly calculate their NHL Equivalency (NHLe).")

# The updated test dictionary with correct IDs
test_players = {
    "Brayden Yager (WPG)": 8484242,
    "Connor Bedard (CHI)": 8484144,
    "Macklin Celebrini (SJS)": 8484801, # The real ID
    "Will Smith (SJS)": 8484183 # Swapping McKenna for another active prospect
}

# Create a dropdown menu
selected_player_name = st.selectbox("Choose a Prospect:", list(test_players.keys()))

# Look up the ID based on the name the user clicked
player_id = test_players[selected_player_name]

# Define our NHLe Translation Factors
nhle_factors = {
    "WHL": 0.32,
    "OHL": 0.32,
    "QMJHL": 0.30,
    "AHL": 0.47,
    "NCAA": 0.43,
    "NHL": 1.00
}

# Add a button to run the analytics
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
            
            if gp == 0:
                continue
                
            ppg = points / gp
            factor = nhle_factors.get(league, 0) 
            nhle = ppg * factor * 82
            
            stats_list.append({
                "Season": season.get("season", "N/A"),
                "League": league,
                "Team": season.get("teamName", {}).get("default", "N/A"),
                "GP": gp,
                "PTS": points,
                "PPG": round(ppg, 2),
                "NHLe Projection": round(nhle, 1) if factor > 0 else None
            })
            
        df = pd.DataFrame(stats_list)

        # THE FIX: This single line tells Pandas to treat the Season column as text (string) 
        # instead of a massive integer, so the graph plots it correctly.
        df['Season'] = df['Season'].astype(str)

        # Display the player's bio
        st.subheader(f"Data for {selected_player_name}")
        
        # Display the data as a table
        st.dataframe(df, use_container_width=True)
        
        # Draw the line chart
        chart_data = df.dropna(subset=['NHLe Projection'])
        if not chart_data.empty:
            st.write("### NHLe Trajectory Over Time")
            st.line_chart(chart_data.set_index('Season')['NHLe Projection'])
            
    else:
        st.error("Error connecting to the NHL API.")
