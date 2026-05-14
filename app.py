import streamlit as st
import requests
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
st.title("Elite Hockey Analytics & Trajectory Hub")
st.write("6-Dimensional predictive modeling enhanced with Multi-Variable NHL Edge Physics (Speed, Shot, Stamina).")

# --- APP MEMORY ---
if 'search_results' not in st.session_state: st.session_state.search_results = {}
if 'analyze_clicked' not in st.session_state: st.session_state.analyze_clicked = False
if 'current_selection' not in st.session_state: st.session_state.current_selection = None
if 'watchlist' not in st.session_state: st.session_state.watchlist = [] 

if os.path.exists("historical_matrix.csv"):
    historical_comps = pd.read_csv("historical_matrix.csv").to_dict('records')
else:
    st.error("Critical Error: historical_matrix.csv is missing. The AI has no brain!")
    st.stop()

st.sidebar.subheader("Step 1: Search Directory")
search_query = st.sidebar.text_input("Enter Player Name:", "")

if st.sidebar.button("Find Players"):
    st.session_state.search_results = {} 
    st.session_state.analyze_clicked = False 
    if search_query:
        if os.path.exists("prospect_db.csv"):
            db = pd.read_csv("prospect_db.csv")
            matches = db[db['Name'].str.contains(search_query, case=False, na=False)]
            for index, row in matches.iterrows():
                st.session_state.search_results[f"{row['Name']} (Offline DB)"] = {"source": "csv", "name": row['Name']}
        
        search_url = f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=20&q={search_query}"
        try:
            search_response = requests.get(search_url).json()
            for player in search_response:
                team = player.get('teamAbbrev', "Unsigned/Prospect")
                if not team: team = "Unsigned/Prospect"
                display_name = f"{player['name']} ({team})"
                if display_name in st.session_state.search_results: display_name += f" - ID:{player['playerId']}"
                st.session_state.search_results[display_name] = {
                    "source": "api", "id": player['playerId'], "name": player['name'],
                    "pos": player.get("positionCode", "F")
                }
        except Exception as e:
            st.sidebar.error("Error communicating with NHL Search API.")

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
    prospect_values = [(1 - (prospect_profile['Age'] / MAX_AGE)) * 100, (prospect_profile['NHLe'] / MAX_NHLE) * 100, (prospect_profile['SGP'] / MAX_SGP) * 100, prospect_profile['EV_Pct'] * 100, prospect_profile['Goal_Pct'] * 100, (prospect_profile['Size'] / MAX_SIZE) * 100]
    comp_values = [(1 - (comp_profile['Age'] / MAX_AGE)) * 100, (comp_profile['NHLe'] / MAX_NHLE) * 100, (comp_profile['SGP'] / MAX_SGP) * 100, comp_profile['EV_Pct'] * 100, comp_profile['Goal_Pct'] * 100, (comp_profile['Size'] / MAX_SIZE) * 100]
    prospect_values += prospect_values[:1]; comp_values += comp_values[:1]
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, size=10); ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25%","50%","75%"], color="grey", size=8); plt.ylim(0,100)
    ax.plot(angles, prospect_values, linewidth=2, linestyle='solid', label='Selected Player', color='#004C97')
    ax.fill(angles, prospect_values, '#004C97', alpha=0.1)
    ax.plot(angles, comp_values, linewidth=2, linestyle='solid', label='Historical Comp', color='#FF6720')
    ax.fill(angles, comp_values, '#FF6720', alpha=0.3)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    return fig

def calculate_ea_rating(profile):
    return min(99, int(40 + ((min(45, (profile['NHLe'] / 75.0) * 45) + min(25, (max(0, 22 - profile['Age']) / 5.0) * 25) + min(15, (profile['SGP'] / 5.0) * 15) + min(10, profile['EV_Pct'] * 10) + min(5, (profile['Size'] / 100.0) * 5)) * 0.6)))

def fetch_edge_modifier(player_id):
    endpoints = {
        "speed": f"https://api-web.nhle.com/v1/edge/skater-skating-speed-detail/{player_id}/now",
        "shot": f"https://api-web.nhle.com/v1/edge/skater-shot-speed-detail/{player_id}/now",
        "distance": f"https://api-web.nhle.com/v1/edge/skater-skating-distance-detail/{player_id}/now"
    }
    
    edge_score = 0
    metrics_found = []
    
    try:
        res = requests.get(endpoints["speed"], timeout=3).json()
        speeds = res.get("topSkatingSpeeds", [])
        if speeds:
            top_speed = speeds[0].get("skatingSpeed", {}).get("imperial", 0.0)
            bursts = sum(1 for s in speeds if s.get("skatingSpeed", {}).get("imperial", 0.0) > 20.0)
            if top_speed >= 22.0 or bursts >= 5:
                edge_score += 1
                metrics_found.append(f"Elite Speed ({round(top_speed,1)} mph)")
            elif top_speed < 20.5 and bursts == 0:
                edge_score -= 1
    except: pass

    try:
        res = requests.get(endpoints["shot"], timeout=3).json()
        top_shot = res.get("shotSpeedDetails", {}).get("topShotSpeed", {}).get("imperial", 0.0)
        if top_shot >= 90.0:
            edge_score += 1
            metrics_found.append(f"Elite Power ({round(top_shot,1)} mph shot)")
    except: pass

    try:
        res = requests.get(endpoints["distance"], timeout=3).json()
        details = res.get("skatingDistanceDetails", [])
        for d in details:
            if d.get("strengthCode") == "all":
                percentile = d.get("distanceTotal", {}).get("percentile", 0.0)
                if percentile >= 0.90:
                    edge_score += 1
                    metrics_found.append(f"Elite Stamina ({int(percentile*100)}th %ile dist)")
                break
    except: pass

    if edge_score >= 2: return "SUPER_UPGRADE", metrics_found
    elif edge_score == 1: return "UPGRADE", metrics_found
    elif edge_score < 0: return "DOWNGRADE", ["Below Average Speed metrics detected"]
    else: return "NONE", []

def get_new_ceiling_base(base_ceiling, modifier, position):
    if modifier == "NONE": return base_ceiling
    
    if position == "Defenseman":
        tiers = ["Depth / AHL", "Bottom Pairing D", "Top 4 D", "Top Pairing D", "Franchise Defenseman"]
    else:
        tiers = ["Depth / AHL", "Bottom 6", "Middle 6", "Top 6", "1st Line", "Franchise"]
        
    bc_lower = base_ceiling.lower()
    if "franchise" in bc_lower: idx = len(tiers) - 1
    elif "1st line" in bc_lower or "top pairing" in bc_lower: idx = len(tiers) - 2
    elif "top 6" in bc_lower or "top 4" in bc_lower: idx = len(tiers) - 3
    elif "middle 6" in bc_lower: idx = len(tiers) - 4
    elif "bottom 6" in bc_lower or "bottom pairing" in bc_lower: idx = 1
    else: idx = 0 
    
    if modifier == "SUPER_UPGRADE": idx += 2
    elif modifier == "UPGRADE": idx += 1
    elif modifier == "DOWNGRADE": idx -= 1
    
    idx = max(0, min(idx, len(tiers) - 1))
    new_base = tiers[idx]
    
    if position != "Defenseman" and new_base not in ["Depth / AHL", "Franchise"]:
        new_base += f" {position}"
        
    return new_base

if st.session_state.analyze_clicked and st.session_state.current_selection:
    player_data_dict = st.session_state.search_results[st.session_state.current_selection]
    selected_player_name = player_data_dict["name"]
    app_mode = "prospect" 
    show_data_wall_warning = False
    nhl_trajectory_data = []
    
    edge_modifier = "NONE"
    edge_metrics = []
    
    # --- VETERAN OVERRIDE TRACKERS ---
    is_veteran = False
    max_nhl_ppg = 0.0
    total_nhl_games = 0

    if player_data_dict["source"] == "api":
        player_id = player_data_dict["id"]
        edge_modifier, edge_metrics = fetch_edge_modifier(player_id)
        
        url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
        response = requests.get(url).json()
        
        raw_pos = player_data_dict.get("pos", "F")
        if raw_pos == "C": prospect_position = "Center"
        elif raw_pos in ["L", "R", "W"]: prospect_position = "Winger"
        elif raw_pos == "D": prospect_position = "Defenseman"
        else: prospect_position = "Winger"

        birth_year = int(response.get("birthDate", "2000-01-01").split("-")[0])
        height_in = response.get("heightInInches", 72); weight_lbs = response.get("weightInPounds", 180)
        
        draft_details = response.get("draftDetails")
        draft_year = draft_details.get("year") if draft_details else birth_year + 18 
        if draft_year < 2015: show_data_wall_warning = True

        season_totals = response.get("seasonTotals", [])
        target_season = f"{draft_year - 1}{draft_year}"
        draft_season_data = next((s for s in season_totals if str(s.get('season')) == target_season), None)
        if not draft_season_data and season_totals: draft_season_data = season_totals[-1]
            
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

        for season in season_totals:
            if season.get("leagueAbbrev") == "NHL":
                s_year = int(str(season.get("season"))[:4])
                s_gp = season.get("gamesPlayed", 0); s_pts = season.get("points", 0)
                total_nhl_games += s_gp
                
                # Track career peak for Veteran Override
                if s_gp >= 20: 
                    ppg = s_pts / s_gp
                    if ppg > max_nhl_ppg: max_nhl_ppg = ppg
                
                if s_year >= draft_year:
                    if s_gp > 5: nhl_trajectory_data.append({"Age": s_year - birth_year, "NHL Points Per Game": round(s_pts / s_gp, 2)})
        
        if total_nhl_games > 20: app_mode = "trajectory"
        if total_nhl_games > 300: is_veteran = True

    current_prospect_profile = {'Age': prospect_age, 'NHLe': prospect_nhle, 'SGP': prospect_sgp, 'EV_Pct': prospect_ev_pct, 'Goal_Pct': prospect_goal_pct, 'Size': prospect_size}

    valid_comps = []
    for comp in historical_comps:
        if selected_player_name.lower() in comp["Name"].lower(): continue
        comp_pos = comp.get("Position", "Center") 
        if prospect_position != comp_pos: continue 
            
        dist_nhle = WEIGHT_NHLE * (prospect_nhle - comp["NHLe"])**2
        dist_age = WEIGHT_AGE * (prospect_age - comp["Age"])**2
        dist_sgp = WEIGHT_SGP * (prospect_sgp - comp["SGP"])**2
        dist_ev = WEIGHT_EV * ((prospect_ev_pct * 100) - (comp["EV_Pct"] * 100))**2 
        dist_goal = WEIGHT_GOAL * ((prospect_goal_pct * 100) - (comp["Goal_Pct"] * 100))**2
        dist_size = WEIGHT_SIZE * (prospect_size - comp["Size"])**2
        comp["Distance"] = round(math.sqrt(dist_nhle + dist_age + dist_sgp + dist_ev + dist_goal + dist_size), 2)
        valid_comps.append(comp)
    
    if not valid_comps:
        st.error(f"No matching historical comparisons found for {prospect_position}.")
        st.stop()
        
    top_match = sorted(valid_comps, key=lambda x: x["Distance"])[0] 
    comp_profile = {'Age': top_match['Age'], 'NHLe': top_match['NHLe'], 'SGP': top_match['SGP'], 'EV_Pct': top_match['EV_Pct'], 'Goal_Pct': top_match['Goal_Pct'], 'Size': top_match['Size']}
    ea_rating = calculate_ea_rating(current_prospect_profile)
    
    # --- CEILING LOGIC INCLUDING VETERAN OVERRIDE ---
    raw_ceiling = top_match['Ceiling']
    
    verified_ceiling = ""
    if is_veteran:
        if prospect_position in ["Center", "Winger"]:
            if max_nhl_ppg >= 0.95: verified_ceiling = f"Franchise {prospect_position}"
            elif max_nhl_ppg >= 0.75: verified_ceiling = f"1st Line {prospect_position}"
            elif max_nhl_ppg >= 0.55: verified_ceiling = f"Top 6 {prospect_position}"
            elif max_nhl_ppg >= 0.35: verified_ceiling = f"Middle 6 {prospect_position}"
            elif max_nhl_ppg >= 0.20: verified_ceiling = f"Bottom 6 {prospect_position}"
            else: verified_ceiling = "Fringe NHLer / AHL Depth"
        elif prospect_position == "Defenseman":
            if max_nhl_ppg >= 0.70: verified_ceiling = "Franchise Defenseman"
            elif max_nhl_ppg >= 0.50: verified_ceiling = "Top Pairing D"
            elif max_nhl_ppg >= 0.35: verified_ceiling = "Top 4 D"
            elif max_nhl_ppg >= 0.20: verified_ceiling = "Bottom Pairing D"
            else: verified_ceiling = "7th D / Fringe"
            
        final_ceiling = verified_ceiling
    else:
        final_ceiling = get_new_ceiling_base(raw_ceiling, edge_modifier, prospect_position)

    st.write("---")
    if show_data_wall_warning: st.warning(f"⚠️ **Data Wall Warning:** {selected_player_name} was drafted before 2015. Historical data may be incomplete.")

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        st.subheader("Draft Year Snapshot")
        st.metric(label="D-0 Prospect Grade", value=f"{ea_rating} OVR")
        st.image(prospect_image_url, width=150)
        st.success(f"**{selected_player_name}** ({prospect_position})")
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
            if edge_modifier != "NONE" and not is_veteran:
                st.info(f"⚡ **NHL Edge Dominance Detected:** {', '.join(edge_metrics)}")
                
            if "Franchise" in top_match['Ceiling']: expected_ppg = 1.00
            elif "1st Line" in top_match['Ceiling'] or "Top Pairing" in top_match['Ceiling']: expected_ppg = 0.80
            elif "Top 6" in top_match['Ceiling'] or "Top 4" in top_match['Ceiling']: expected_ppg = 0.60
            elif "Middle 6" in top_match['Ceiling'] or "Bottom Pairing" in top_match['Ceiling']: expected_ppg = 0.40
            elif "Bottom 6" in top_match['Ceiling'] or "7th D" in top_match['Ceiling']: expected_ppg = 0.25
            else: expected_ppg = 0.15
            
            if not is_veteran:
                if edge_modifier == "SUPER_UPGRADE": expected_ppg += 0.25
                elif edge_modifier == "UPGRADE": expected_ppg += 0.15 
                elif edge_modifier == "DOWNGRADE": expected_ppg -= 0.15
            
            if nhl_trajectory_data:
                df_chart = pd.DataFrame(nhl_trajectory_data).set_index("Age")
                df_chart["Expected Baseline"] = expected_ppg
                st.line_chart(df_chart, color=["#FF6720", "#808080"])
            else: st.info("Not enough NHL data to plot trajectory.")
            
            # --- THE VETERAN OVERRIDE DISPLAY ---
            if is_veteran:
                st.info(f"🏆 **Veteran Override Active:** With {total_nhl_games} NHL games played, future projections are obsolete.")
                st.success(f"**Verified Career Peak:** {verified_ceiling} *(Peak: {round(max_nhl_ppg, 2)} PPG)*")
            else:
                st.write(f"**Draft Day Consensus Ceiling:** {raw_ceiling}")
                if edge_modifier != "NONE":
                    tag = "🔥 SUPER UPGRADE" if edge_modifier == "SUPER_UPGRADE" else ("⬆️ UPGRADED" if edge_modifier == "UPGRADE" else "⬇️ DOWNGRADED")
                    st.success(f"**Current NHL Edge Ceiling:** {final_ceiling} ({tag})")
                
        else:
            st.subheader("Key Analytics Head-to-Head")
            comp_df = pd.DataFrame({
                "Metric": ["NHLe Projection", "EV Production %", "Goal Dependency %", "Shots / Game"],
                selected_player_name: [prospect_nhle, f"{int(prospect_ev_pct*100)}%", f"{int(prospect_goal_pct*100)}%", prospect_sgp],
                top_match['Name']: [top_match['NHLe'], f"{int(top_match['EV_Pct']*100)}%", f"{int(top_match['Goal_Pct']*100)}%", top_match['SGP']]
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.divider()
            
            # Progress Bar Section
            st.write("### Projected Roster Ceiling")
            
            if is_veteran:
                if "Franchise" in verified_ceiling: val = 100
                elif "1st Line" in verified_ceiling or "Top Pairing" in verified_ceiling: val = 85
                elif "Top 6" in verified_ceiling or "Top 4" in verified_ceiling: val = 70
                elif "Middle 6" in verified_ceiling or "Bottom Pairing" in verified_ceiling: val = 50
                elif "Bottom 6" in verified_ceiling or "7th D" in verified_ceiling: val = 35
                else: val = 20
                st.progress(val, text=f"**Verified Role:** {final_ceiling}")
            else:
                if "Franchise" in top_match['Ceiling']: val = 100
                elif "1st Line" in top_match['Ceiling'] or "Top Pairing" in top_match['Ceiling']: val = 85
                elif "Top 6" in top_match['Ceiling'] or "Top 4" in top_match['Ceiling']: val = 70
                elif "Middle 6" in top_match['Ceiling'] or "Bottom Pairing" in top_match['Ceiling']: val = 50
                elif "Bottom 6" in top_match['Ceiling'] or "7th D" in top_match['Ceiling']: val = 35
                else: val = 20
                
                if edge_modifier == "SUPER_UPGRADE": val = min(100, val + 25)
                elif edge_modifier == "UPGRADE": val = min(100, val + 15)
                elif edge_modifier == "DOWNGRADE": val = max(10, val - 15)
                
                st.progress(val, text=f"**Current Peak Role:** {final_ceiling}")

    with col3:
        st.subheader("Draft Year 6-Axis Profile")
        spider_fig = create_hexagon_chart(current_prospect_profile, comp_profile)
        st.pyplot(spider_fig)

    st.divider()
    
    colA, colB = st.columns([1, 4])
    with colA:
        if st.button("⭐ Save to Leaderboard", use_container_width=True):
            names = [p["Name"] for p in st.session_state.watchlist]
            if selected_player_name not in names:
                
                if is_veteran:
                    table_ceiling = f"🏆 {final_ceiling}"
                else:
                    table_ceiling = f"{raw_ceiling} ➡️ {final_ceiling}" if edge_modifier != "NONE" else raw_ceiling
                
                st.session_state.watchlist.append({
                    "Name": selected_player_name,
                    "Position": prospect_position,
                    "Draft League": league,
                    "OVR Grade": ea_rating,
                    "Closest Comp": top_match['Name'],
                    "Projected Peak": table_ceiling
                })
                st.rerun() 
            else:
                st.info("Already saved.")
                
    if st.session_state.watchlist:
        st.subheader("📋 Draft Day Leaderboard")
        wl_df = pd.DataFrame(st.session_state.watchlist).sort_values(by="OVR Grade", ascending=False)
        st.dataframe(wl_df, use_container_width=True, hide_index=True)
