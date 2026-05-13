import streamlit as st
import requests
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import numpy as np

# Set page to wide mode to hold multiple visual columns side-by-side
st.set_page_config(layout="wide")

st.title("Elite Hockey Analytics & Trajectory Hub")
st.write("Advanced predictive modeling with visual comparison metrics.")

test_players = {
    "Brayden Yager (WPG)": 8484242,
    "Macklin Celebrini (SJS)": 8484801,
    "Gavin McKenna (Undrafted)": 0,
    "Landon DuPont (Undrafted)": 0
}

# Move search dropdown to a sidebar to clean up main screen
selected_player_name = st.sidebar.selectbox("Choose a Prospect:", list(test_players.keys()))
player_id = test_players[selected_player_name]
analyze_btn = st.sidebar.button("Run Advanced Analytics")

nhle_factors = {
    "WHL": 0.32, "OHL": 0.32, "QMJHL": 0.30, 
    "AHL": 0.47, "NCAA": 0.43, "NHL": 1.00, "CSSHL": 0.15
}

# --- UPGRADED HISTORICAL DATABASE WITH IMAGES ---
# We use standard NHL asset URLs for historical headshots
historical_comps = [
    {"Name": "Connor McDavid (Historical)", "NHLe": 68.0, "Age": 16, "SGP": 4.5, "Ceiling": "Generational", "ImageURL": "https://assets.nhle.com/mugs/nhl/latest/8478402.png"},
    {"Name": "Nathan MacKinnon (Historical)", "NHLe": 55.0, "Age": 16, "SGP": 4.2, "Ceiling": "Franchise Player", "ImageURL": "https://assets.nhle.com/mugs/nhl/latest/8477444.png"},
    {"Name": "Bo Horvat (Historical)", "NHLe": 38.0, "Age": 19, "SGP": 3.2, "Ceiling": "Top 6 Forward", "ImageURL": "https://assets.nhle.com/mugs/nhl/latest/8477500.png"},
    {"Name": "Vincent Trocheck (Historical)", "NHLe": 42.0, "Age": 19, "SGP": 3.5, "Ceiling": "Top 6 Forward", "ImageURL": "https://assets.nhle.com/mugs/nhl/latest/8476389.png"},
    {"Name": "AHL Journeyman (Historical)", "NHLe": 22.0, "Age": 20, "SGP": 1.8, "Ceiling": "AHL / Fringe", "ImageURL": "https://upload.wikimedia.org/wikipedia/commons/e/e0/Generic_jersey_icon.png"}
]

WEIGHT_NHLE = 1.0     
WEIGHT_AGE = 3.0      
WEIGHT_SGP = 1.5      

# --- GRAPHING TOOL: Create Spider Chart Function ---
def create_spider_chart(prospect_profile, comp_profile):
    # Data Normalization parameters (Realistic max bounds to set 100% percentile)
    MAX_AGE = 22
    MAX_NHLE = 75
    MAX_SGP = 6.0
    
    categories = ['Youth Factor', 'NHLe Production', 'Offensive Volume']
    
    # Calculate Percentile Scores for both players
    # For Youth Factor, we inverse the calculation (Lower Age = Higher Score)
    prospect_values = [
        (1 - (prospect_profile['Age'] / MAX_AGE)) * 100,
        (prospect_profile['NHLe'] / MAX_NHLE) * 100,
        (prospect_profile['SGP'] / MAX_SGP) * 100
    ]
    
    comp_values = [
        (1 - (comp_profile['Age'] / MAX_AGE)) * 100,
        (comp_profile['NHLe'] / MAX_NHLE) * 100,
        (comp_profile['SGP'] / MAX_SGP) * 100
    ]

    # Close the shapes mathematically for matplotlib
    prospect_values += prospect_values[:1]
    comp_values += comp_values[:1]
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Matplotlib styling
    plt.xticks(angles[:-1], categories, size=12)
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25%","50%","75%"], color="grey", size=8)
    plt.ylim(0,100)
    
    # Plot Prospect (Blue, solid)
    ax.plot(angles, prospect_values, linewidth=2, linestyle='solid', label='Selected Prospect', color='#004C97')
    ax.fill(angles, prospect_values, '#004C97', alpha=0.1)
    
    # Plot Comparison (Orange, translucent)
    ax.plot(angles, comp_values, linewidth=2, linestyle='solid', label='Historical Comp', color='#FF6720')
    ax.fill(angles, comp_values, '#FF6720', alpha=0.3)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    return fig


# --- MAIN ANALYTICS LOGIC ---
if analyze_btn:
    
    prospect_image_url = "https://upload.wikimedia.org/wikipedia/commons/e/e0/Generic_jersey_icon.png" # Standard Placeholder

    # --- HYBRID ROUTING ENGINE WITH IMAGE EXTRACTION ---
    if player_id == 0:
        clean_name = selected_player_name.split(" (")[0]
        if os.path.exists("prospect_db.csv"):
            db = pd.read_csv("prospect_db.csv")
            player_data = db[db['Name'] == clean_name]
            
            if not player_data.empty:
                prospect_age = int(player_data['Age'].values[0])
                league = player_data['League'].values[0]
                gp = int(player_data['GP'].values[0])
                pts = int(player_data['PTS'].values[0])
                prospect_sgp = float(player_data['SGP'].values[0])
                prospect_image_url = player_data['ImageURL'].values[0] # EXTRACTING CSV IMAGE URL

                ppg = pts / gp
                factor = nhle_factors.get(league, 0)
                prospect_nhle = round(ppg * factor * 82, 1)
            else: st.stop()
        else: st.stop()

    else:
        # Pinging API for drafted players
        url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            birth_year = int(data.get("birthDate", "2000-01-01").split("-")[0])
            season_totals = data.get("seasonTotals", [])
            
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
            
            # EXTRACTING API IMAGE URL (Hidden endpoint)
            prospect_image_url = f"https://assets.nhle.com/mugs/nhl/latest/{player_id}.png"
            
        else: st.stop()

    # Create a simple profile dictionary for the graphing engine
    current_prospect_profile = {'Age': prospect_age, 'NHLe': prospect_nhle, 'SGP': prospect_sgp}

    # --- UPGRADED KNN MACHINE LEARNING ENGINE ---
    for comp in historical_comps:
        dist_nhle = WEIGHT_NHLE * (prospect_nhle - comp["NHLe"])**2
        dist_age = WEIGHT_AGE * (prospect_age - comp["Age"])**2
        dist_sgp = WEIGHT_SGP * (prospect_sgp - comp["SGP"])**2
        
        distance = math.sqrt(dist_nhle + dist_age + dist_sgp)
        comp["Distance"] = round(distance, 2)
    
    sorted_comps = sorted(historical_comps, key=lambda x: x["Distance"])
    top_match = sorted_comps[0] # Use top match for the visual comparison


    # --- NEW: VISUAL OUTPUT SECTION ---
    st.write("---")
    
    # Create three wide columns to hold all the output
    col1, col2, col3 = st.columns([1, 2, 2])
    
    # Column 1: Bios & Images (The "Mugshots")
    with col1:
        st.subheader("Selected Prospect")
        st.image(prospect_image_url, width=150)
        st.success(f"**{selected_player_name}**")
        st.write(f"Age: {prospect_age} | League: {league}")
        
        st.write("vs.")
        
        st.subheader("Closest AI Match")
        # Ensure 'historical_comps' list above was updated to include 'ImageURL'!
        st.image(top_match['ImageURL'], width=150)
        st.warning(f"**{top_match['Name']}**")
        st.write(f"Match Score: {top_match['Distance']}")

    # Column 2: Data Comparison Table
    with col2:
        st.subheader("Key Analytics Head-to-Head")
        
        # Build a temporary comparison table
        comp_df = pd.DataFrame({
            "Metric": ["Age", "Points Per Game", "Shots Per Game (SGP)", "NHLe Projection"],
            selected_player_name: [prospect_age, round(ppg, 2), prospect_sgp, prospect_nhle],
            top_match['Name']: [top_match['Age'], top_match['PPG'], top_match['SGP'], top_match['NHLe']]
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        st.divider()
        st.write("### Consensus Projected Ceiling")
        # Visualizing ceiling using a progress bar as a proxy for confidence/tier
        ceiling = top_match['Ceiling']
        if ceiling == "Generational": val = 100
        elif ceiling == "Franchise Player": val = 85
        elif ceiling == "Top 6 Forward": val = 70
        else: val = 50
        
        st.progress(val, text=f"**Current Ceiling Tier: {ceiling}**")


    # Column 3: The Spider Chart Visualization
    with col3:
        st.subheader("Comparative Attribute Profile")
        
        # Create a comp profile dictionary for the graph math
        comp_profile = {'Age': top_match['Age'], 'NHLe': top_match['NHLe'], 'SGP': top_match['SGP']}
        
        # Run the visualization function defined at the top
        spider_fig = create_spider_chart(current_prospect_profile, comp_profile)
        
        # Display the Matplotlib figure within Streamlit
        st.pyplot(spider_fig)
