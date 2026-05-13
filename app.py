import streamlit as st
import requests
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
st.title("Elite Hockey Analytics & Trajectory Hub")
st.write("6-Dimensional predictive modeling (NHLe, Age, Volume, EV%, Goal Dependency, Size).")

# --- THE DYNAMIC SEARCH BAR ---
search_query = st.sidebar.text_input("Search Player Name (e.g., Sascha Boumedienne):", "")
analyze_btn = st.sidebar.button("Run Advanced Analytics")

nhle_factors = {
    "WHL": 0.32, "OHL": 0.32, "QMJHL": 0.30, 
    "AHL": 0.47, "NCAA": 0.43, "NHL": 1.00, "CSSHL": 0.15
}

# --- UPGRADED 6-DIMENSIONAL HISTORICAL DATABASE ---
historical_comps = [
    {"Name": "Connor McDavid (Historical)", "NHLe": 68.0, "Age": 16, "SGP": 4.5, "EV_Pct": 0.70, "Goal_Pct": 0.35, "Size": 90, "Ceiling": "Generational", "ImageURL": "https://assets.nhle.com/mugs/nhl/latest/8478402.png"},
    {"Name": "Nathan MacKinnon (Historical)", "NHLe": 55.0, "Age": 16, "SGP": 4.2, "EV_Pct": 0.65, "Goal_Pct": 0.42, "Size": 91, "Ceiling": "Franchise Player", "ImageURL": "https://assets.nhle.com/mugs/nhl/latest/8477492.png"},
    {"Name": "Bo Horvat (Historical)", "NHLe": 38.0, "Age": 19, "SGP": 3.2, "EV_Pct": 0.55, "Goal_Pct": 0.40, "Size": 93, "Ceiling": "Top 6 Forward", "ImageURL": "https://assets.nhle.com/mugs/nhl/latest/8477500.png"},
    {"Name": "Vincent Trocheck (Historical)", "NHLe": 42.0, "Age": 19, "SGP": 3.5, "EV_Pct": 0.60, "Goal_Pct": 0.45, "Size": 88, "Ceiling": "Top 6 Forward", "ImageURL": "https://assets.nhle.com/mugs/nhl/latest/8476389.png"},
    {"Name": "AHL Journeyman (Historical)", "NHLe": 22.0, "Age": 20, "SGP": 1.8, "EV_Pct": 0.40, "Goal_Pct": 0.25, "Size": 90, "Ceiling": "AHL / Fringe", "ImageURL": "https://upload.wikimedia.org/wikipedia/commons/e/e0/Generic_jersey_icon.png"}
]

WEIGHT_NHLE = 1.0     
WEIGHT_AGE = 3.0      
WEIGHT_SGP = 1.5 
WEIGHT_EV = 2.0      
WEIGHT_GOAL = 1.0
WEIGHT_SIZE = 0.5    

def create_hexagon_chart(prospect_profile, comp_profile):
    MAX_AGE = 22
    MAX_NHLE = 75
    MAX_SGP = 6.0
    MAX_SIZE = 100 
    
    categories = ['Youth Factor', 'NHLe Production', 'Offensive Volume', 'Even-Strength %', 'Goal Dependency', 'Physical Size']
    
    prospect_values = [
        (1 - (prospect_profile['Age'] / MAX_AGE)) * 100,
        (prospect_profile['NHLe'] / MAX_NHLE) * 100,
        (prospect_profile['SGP'] / MAX_SGP) * 100,
        prospect_profile['EV_Pct'] * 100,
        prospect_profile['Goal_Pct'] * 100,
        (prospect_profile['Size'] / MAX_SIZE) * 100
    ]
    
    comp_values = [
        (1 - (comp_profile['Age'] / MAX_AGE)) * 100,
        (comp_profile['NHLe'] / MAX_NHLE) * 100,
        (comp_profile['SGP'] / MAX_SGP) * 100,
        comp_profile['EV_Pct'] * 100,
        comp_profile['Goal_Pct'] * 100,
        (comp_profile['Size'] / MAX_SIZE) * 100
    ]

    prospect_values += prospect_values[:1]
    comp_values += comp_values[:1]
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, size=10)
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25%","50%","75%"], color="grey", size=8)
    plt.ylim(0,100)
    
    ax.plot(angles, prospect_values, linewidth=2, linestyle='solid', label='Selected Prospect', color='#004C97')
    ax.fill(angles, prospect_values, '#004C97', alpha=0.1)
    
    ax.plot(angles, comp_values, linewidth=2, linestyle='solid', label='Historical Comp', color='#FF6720')
    ax.fill(angles, comp_values, '#FF6720', alpha=0.3)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    return fig

if analyze_btn and search_query:
    prospect_image_url = "https://upload.wikimedia.org/wikipedia/commons/e/e0/Generic_jersey_icon.png"
    player_found = False
    is_undrafted = False
    
    # ROUTE A: Search Local Custom CSV First
    if os.path.exists("prospect_db.csv"):
        db = pd.read_csv("prospect_db.csv")
        # Exact or partial match check so you don't have to type perfect names
        match = db[db['Name'].str.contains(search_query, case=False, na=False)]
        if not match.empty:
            player_found = True
            is_undrafted = True
            player_data = match.iloc[0]
            
            selected_player_name = player_data['Name']
            prospect_age = int(player_data['Age'])
            league = player_data['League']
            gp = int(player_data['GP'])
            pts = int(player_data['PTS'])
            goals = int(player_data['G'])
            ppp = int(player_data['PPP'])
            prospect_sgp = float(player_data['SGP'])
            height_in = float(player_data['Height'])
            weight_lbs = float(player_data['Weight'])
            prospect_image_url = player_data['ImageURL']

            prospect_nhle = round((pts / gp) * nhle_factors.get(league, 0) * 82, 1)
            prospect_ev_pct = round((pts - ppp) / pts, 2) if pts > 0 else 0
            prospect_goal_pct = round(goals / pts, 2) if pts > 0 else 0
            prospect_size = height_in + (weight_lbs / 10)

    # ROUTE B: If not in local CSV, hit the NHL's internal Search API
    if not player_found:
        search_url = f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=5&q={search_query}"
        try:
            search_response = requests.get(search_url).json()
            if len(search_response) > 0:
                # The API hands us the hidden 7-digit ID dynamically!
                player_id = search_response[0]['playerId']
                selected_player_name = search_response[0]['name']
                player_found = True
                
                url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    birth_year = int(data.get("birthDate", "2000-01-01").split("-")[0])
                    height_in = data.get("heightInInches", 72)
                    weight_lbs = data.get("weightInPounds", 180)
                    season_totals = data.get("seasonTotals", [])
                    
                    if season_totals:
                        latest_season = season_totals[-1] 
                        gp = latest_season.get("gamesPlayed", 1)
                        pts = latest_season.get("points", 0)
                        goals = latest_season.get("goals", 0)
                        ppp = latest_season.get("powerPlayPoints", 0)
                        shots = latest_season.get("shots", 0)
                        league = latest_season.get("leagueAbbrev", "N/A")
                        
                        season_year = int(str(latest_season.get("season", "00000000"))[:4])
                        prospect_age = season_year - birth_year
                        
                        prospect_nhle = round((pts / gp) * nhle_factors.get(league, 0) * 82, 1)
                        prospect_sgp = round(shots / gp, 2) if shots > 0 else 2.0
                        prospect_ev_pct = round((pts - ppp) / pts, 2) if pts > 0 else 0
                        prospect_goal_pct = round(goals / pts, 2) if pts > 0 else 0
                        prospect_size = height_in + (weight_lbs / 10)
                        
                        prospect_image_url = f"https://assets.nhle.com/mugs/nhl/latest/{player_id}.png"
                    else:
                        st.error("No season data found for this player.")
                        st.stop()
                else:
                    st.error("Failed to connect to NHL player profile.")
                    st.stop()
            else:
                st.error("Player not found in local database or NHL API.")
                st.stop()
        except Exception as e:
            st.error("Error communicating with NHL Search API.")
            st.stop()

    if player_found:
        current_prospect_profile = {
            'Age': prospect_age, 'NHLe': prospect_nhle, 'SGP': prospect_sgp,
            'EV_Pct': prospect_ev_pct, 'Goal_Pct': prospect_goal_pct, 'Size': prospect_size
        }

        for comp in historical_comps:
            dist_nhle = WEIGHT_NHLE * (prospect_nhle - comp["NHLe"])**2
            dist_age = WEIGHT_AGE * (prospect_age - comp["Age"])**2
            dist_sgp = WEIGHT_SGP * (prospect_sgp - comp["SGP"])**2
            dist_ev = WEIGHT_EV * ((prospect_ev_pct * 100) - (comp["EV_Pct"] * 100))**2 
            dist_goal = WEIGHT_GOAL * ((prospect_goal_pct * 100) - (comp["Goal_Pct"] * 100))**2
            dist_size = WEIGHT_SIZE * (prospect_size - comp["Size"])**2
            
            distance = math.sqrt(dist_nhle + dist_age + dist_sgp + dist_ev + dist_goal + dist_size)
            comp["Distance"] = round(distance, 2)
        
        sorted_comps = sorted(historical_comps, key=lambda x: x["Distance"])
        top_match = sorted_comps[0] 

        st.write("---")
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            st.subheader("Selected Prospect")
            st.image(prospect_image_url, width=150)
            st.success(f"**{selected_player_name}** {'(Offline DB)' if is_undrafted else '(NHL API)'}")
            st.write(f"Age: {prospect_age} | League: {league}")
            st.write("vs.")
            st.subheader("Closest AI Match")
            st.image(top_match['ImageURL'], width=150)
            st.warning(f"**{top_match['Name']}**")
            st.write(f"Match Score: {top_match['Distance']}")

        with col2:
            st.subheader("Key Analytics Head-to-Head")
            comp_df = pd.DataFrame({
                "Metric": ["NHLe Projection", "EV Production %", "Goal Dependency %", "Shots / Game"],
                selected_player_name: [prospect_nhle, f"{int(prospect_ev_pct*100)}%", f"{int(prospect_goal_pct*100)}%", prospect_sgp],
                top_match['Name']: [top_match['NHLe'], f"{int(top_match['EV_Pct']*100)}%", f"{int(top_match['Goal_Pct']*100)}%", top_match['SGP']]
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            st.divider()
            st.write("### Consensus Projected Ceiling")
            ceiling = top_match['Ceiling']
            val = 100 if ceiling == "Generational" else (85 if ceiling == "Franchise Player" else (70 if ceiling == "Top 6 Forward" else 50))
            st.progress(val, text=f"**Current Ceiling Tier: {ceiling}**")

        with col3:
            st.subheader("6-Axis Comparative Profile")
            comp_profile = {
                'Age': top_match['Age'], 'NHLe': top_match['NHLe'], 'SGP': top_match['SGP'],
                'EV_Pct': top_match['EV_Pct'], 'Goal_Pct': top_match['Goal_Pct'], 'Size': top_match['Size']
            }
            spider_fig = create_hexagon_chart(current_prospect_profile, comp_profile)
            st.pyplot(spider_fig)
