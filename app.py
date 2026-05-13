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

# --- APP MEMORY ---
if 'search_results' not in st.session_state:
    st.session_state.search_results = {}
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False
if 'current_selection' not in st.session_state:
    st.session_state.current_selection = None

# --- LOAD HISTORICAL MATRIX ---
if os.path.exists("historical_matrix.csv"):
    historical_comps = pd.read_csv("historical_matrix.csv").to_dict('records')
else:
    st.error("Critical Error: historical_matrix.csv is missing. The AI has no brain!")
    st.stop()

# --- STEP 1: SEARCH ---
st.sidebar.subheader("Step 1: Search Directory")
search_query = st.sidebar.text_input("Enter Player Name (e.g., Aho, Mcdav, Iginla):", "")

if st.sidebar.button("Find Players"):
    st.session_state.search_results = {} 
    st.session_state.analyze_clicked = False 
    
    if search_query:
        if os.path.exists("prospect_db.csv"):
            db = pd.read_csv("prospect_db.csv")
            matches = db[db['Name'].str.contains(search_query, case=False, na=False)]
            for index, row in matches.iterrows():
                display_name = f"{row['Name']} (Offline DB)"
                st.session_state.search_results[display_name] = {"source": "csv", "name": row['Name']}
        
        search_url = f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=5&q={search_query}"
        try:
            search_response = requests.get(search_url).json()
            for player in search_response:
                team = player.get('teamAbbrev', None)
                if not team: team = "Unsigned/Prospect"
                display_name = f"{player['name']} ({team})"
                if display_name in st.session_state.search_results:
                    display_name += f" - ID:{player['playerId']}"
                st.session_state.search_results[display_name] = {
                    "source": "api", "id": player['playerId'], "name": player['name']
                }
        except Exception as e:
            st.sidebar.error("Error communicating with NHL Search API.")

# --- STEP 2: CONFIRM ---
if st.session_state.search_results:
    st.sidebar.divider()
    st.sidebar.subheader("Step 2: Confirm & Analyze")
    selected_option = st.sidebar.selectbox("Select Exact Match:", list(st.session_state.search_results.keys()))
    if st.sidebar.button("Run Advanced Analytics"):
        st.session_state.analyze_clicked = True
        st.session_state.current_selection = selected_option

nhle_factors = {"WHL": 0.32, "OHL": 0.32, "QMJHL": 0.30, "AHL": 0.47, "NCAA": 0.43, "NHL": 1.00, "CSSHL": 0.15, "USHL": 0.25, "SHL": 0.58, "Liiga": 0.43, "KHL": 0.77, "NTDP": 0.35}
WEIGHT_NHLE = 1.0; WEIGHT_AGE = 3.0; WEIGHT_SGP = 1.5; WEIGHT_EV = 2.0; WEIGHT_GOAL = 1.0; WEIGHT_SIZE = 0.5    

def create_hexagon_chart(prospect_profile, comp_profile):
    MAX_AGE = 22; MAX_NHLE = 75; MAX_SGP = 6.0; MAX_SIZE = 100 
    categories = ['Youth Factor', 'NHLe Production', 'Offensive Volume', 'Even-Strength %', 'Goal Dependency', 'Physical Size']
    
    prospect_values = [
        (1 - (prospect_profile['Age'] / MAX_AGE)) * 100, (prospect_profile['NHLe'] / MAX_NHLE) * 100,
        (prospect_profile['SGP'] / MAX_SGP) * 100, prospect_profile['EV_Pct'] * 100,
        prospect_profile['Goal_Pct'] * 100, (prospect_profile['Size'] / MAX_SIZE) * 100
    ]
    comp_values = [
        (1 - (comp_profile['Age'] / MAX_AGE)) * 100, (comp_profile['NHLe'] / MAX_NHLE) * 100,
        (comp_profile['SGP'] / MAX_SGP) * 100, comp_profile['EV_Pct'] * 100,
        comp_profile['Goal_Pct'] * 100, (comp_profile['Size'] / MAX_SIZE) * 100
    ]

    prospect_values += prospect_values[:1]; comp_values += comp_values[:1]
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, size=10); ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25%","50%","75%"], color="grey", size=8); plt.ylim(0,100)
    
    ax.plot(angles, prospect_values, linewidth=2, linestyle='solid', label='Selected Player (Draft Year)', color='#004C97')
    ax.fill(angles, prospect_values, '#004C97', alpha=0.1)
    ax.plot(angles, comp_values, linewidth=2, linestyle='solid', label='Historical Comp', color='#FF6720')
    ax.fill(angles, comp_values, '#FF6720', alpha=0.3)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    return fig

def calculate_ea_rating(profile):
    score_nhle = min(45, (profile['NHLe'] / 75.0) * 45); age_diff = max(0, 22 - profile['Age'])
    score_age = min(25, (age_diff / 5.0) * 25); score_sgp = min(15, (profile['SGP'] / 5.0) * 15)
    score_ev = min(10, profile['EV_Pct'] * 10); score_size = min(5, (profile['Size'] / 100.0) * 5)
    return min(99, int(40 + ((score_nhle + score_age + score_sgp + score_ev + score_size) * 0.6)))

# --- MAIN LOGIC ---
if st.session_state.analyze_clicked and st.session_state.current_selection:
    player_data_dict = st.session_state.search_results[st.session_state.current_selection]
    selected_player_name = player_data_dict["name"]
    
    app_mode = "prospect" 
    show_data_wall_warning = False
    nhl_trajectory_data = []

    if player_data_dict["source"] == "api":
        player_id = player_data_dict["id"]
        url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
        response = requests.get(url).json()
        
        birth_year = int(response.get("birthDate", "2000-01-01").split("-")[0])
        height_in = response.get("heightInInches", 72); weight_lbs = response.get("weightInPounds", 180)
        
        draft_details = response.get("draftDetails")
        if draft_details:
            draft_year = draft_details.get("year")
        else:
            draft_year = birth_year + 18 
            
        if draft_year < 2015:
            show_data_wall_warning = True

        season_totals = response.get("seasonTotals", [])
        
        target_season = f"{draft_year - 1}{draft_year}"
        draft_season_data = None
        
        for season in season_totals:
            if str(season.get('season')) == target_season:
                draft_season_data = season
                break
                
        if not draft_season_data and season_totals:
            draft_season_data = season_totals[-1]
            
        gp = max(1, draft_season_data.get("gamesPlayed", 1)) 
        pts = draft_season_data.get("points", 0); goals = draft_season_data.get("goals", 0)
        ppp = draft_season_data.get("powerPlayPoints", 0); shots = draft_season_data.get("shots", 0)
        league = draft_season_data.get("leagueAbbrev", "N/A")
        
        prospect_age = 18 
        prospect_nhle = round((pts / gp) * nhle_factors.get(league, 0.20) * 82, 1)
        prospect_sgp = round(shots / gp, 2) if shots > 0 else 2.0
        prospect_ev_pct = round((pts - ppp) / pts, 2) if pts > 0 else 0
        prospect_goal_pct = round(goals / pts, 2) if pts > 0 else 0
        prospect_size = height_in + (weight_lbs / 10)
        prospect_image_url = f"https://assets.nhle.com/mugs/nhl/latest/{player_id}.png"

        total_nhl_games = 0
        for season in season_totals:
            if season.get("leagueAbbrev") == "NHL":
                s_year = int(str(season.get("season"))[:4])
                if s_year >= draft_year:
                    s_gp = season.get("gamesPlayed", 0)
                    s_pts = season.get("points", 0)
                    total_nhl_games += s_gp
                    if s_gp > 5: 
                        nhl_trajectory_data.append({
                            "Age": s_year - birth_year,
                            "NHL Points Per Game": round(s_pts / s_gp, 2)
                        })
        
        if total_nhl_games > 20:
            app_mode = "trajectory"

    current_prospect_profile = {
        'Age': prospect_age, 'NHLe': prospect_nhle, 'SGP': prospect_sgp,
        'EV_Pct': prospect_ev_pct, 'Goal_Pct': prospect_goal_pct, 'Size': prospect_size
    }

    # --- THE CLONE FIX: Filter before sorting ---
    valid_comps = []
    for comp in historical_comps:
        # If the searched name is inside the CSV name (e.g., "Auston Matthews" inside "Auston Matthews (2016)"), skip it!
        if selected_player_name.lower() in comp["Name"].lower():
            continue
            
        dist_nhle = WEIGHT_NHLE * (prospect_nhle - comp["NHLe"])**2
        dist_age = WEIGHT_AGE * (prospect_age - comp["Age"])**2
        dist_sgp = WEIGHT_SGP * (prospect_sgp - comp["SGP"])**2
        dist_ev = WEIGHT_EV * ((prospect_ev_pct * 100) - (comp["EV_Pct"] * 100))**2 
        dist_goal = WEIGHT_GOAL * ((prospect_goal_pct * 100) - (comp["Goal_Pct"] * 100))**2
        dist_size = WEIGHT_SIZE * (prospect_size - comp["Size"])**2
        
        comp["Distance"] = round(math.sqrt(dist_nhle + dist_age + dist_sgp + dist_ev + dist_goal + dist_size), 2)
        valid_comps.append(comp)
    
    top_match = sorted(valid_comps, key=lambda x: x["Distance"])[0] 
    
    comp_profile = {
        'Age': top_match['Age'], 'NHLe': top_match['NHLe'], 'SGP': top_match['SGP'],
        'EV_Pct': top_match['EV_Pct'], 'Goal_Pct': top_match['Goal_Pct'], 'Size': top_match['Size']
    }

    st.write("---")
    
    if show_data_wall_warning:
        st.warning(f"⚠️ **Data Wall Warning:** {selected_player_name} was drafted before 2015. Historical minor-league data from that era rarely tracked Shots on Goal or Powerplay metrics accurately. The 6-axis math below is relying on incomplete data and may be highly skewed.")

    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        st.subheader("Draft Year Snapshot")
        ea_rating = calculate_ea_rating(current_prospect_profile)
        st.metric(label="D-0 Prospect Grade", value=f"{ea_rating} OVR")
        st.image(prospect_image_url, width=150)
        st.success(f"**{selected_player_name}**")
        st.write(f"Draft Year League: {league}")
        
        st.write("---")
        st.subheader("Closest AI Match")
        comp_ea_rating = calculate_ea_rating(comp_profile)
        st.metric(label="Historical OVR Grade", value=f"{comp_ea_rating} OVR", delta=f"{round(ea_rating - comp_ea_rating, 1)}", delta_color="normal")
        st.image(top_match['ImageURL'], width=150)
        st.warning(f"**{top_match['Name']}**")
        st.write(f"Match Score: {top_match['Distance']}")

    with col2:
        if app_mode == "trajectory":
            st.subheader("xActual NHL Trajectory")
            st.write("Has this player fulfilled their draft day expectations?")
            
            ceiling = top_match['Ceiling']
            if "Generational" in ceiling: expected_ppg = 1.10
            elif "Franchise" in ceiling: expected_ppg = 0.90
            elif "Top 6" in ceiling or "1st Round" in ceiling: expected_ppg = 0.65
            elif "Middle Round" in ceiling: expected_ppg = 0.40
            elif "Late Round" in ceiling: expected_ppg = 0.25
            else: expected_ppg = 0.50
            
            if nhl_trajectory_data:
                df_chart = pd.DataFrame(nhl_trajectory_data).set_index("Age")
                df_chart["Expected Baseline"] = expected_ppg
                st.line_chart(df_chart, color=["#FF6720", "#808080"])
            else:
                st.info("Not enough NHL data to plot trajectory.")
                
            st.write(f"**AI Projected Ceiling:** {ceiling}")
            
        else:
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
            val = 100 if "Generational" in ceiling else (85 if "Franchise" in ceiling else (75 if "Top 6" in ceiling or "1st Round" in ceiling else 50))
            st.progress(val, text=f"**Current Ceiling Tier: {ceiling}**")

    with col3:
        st.subheader("Draft Year 6-Axis Profile")
        spider_fig = create_hexagon_chart(current_prospect_profile, comp_profile)
        st.pyplot(spider_fig)
