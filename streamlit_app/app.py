# streamlit_app/app.py
# Marketing Mix Simulator - Interactive Dashboard (类似 consumer_choice_abm)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.simulator import MarketingSimulator
from data.generate_data import generate_marketing_data


# Page config
st.set_page_config(
    page_title="Marketing Mix Simulator",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        font-weight: bold;
    }
    .parameter-label {
        font-size: 12px;
        color: #666;
    }
    .parameter-value {
        font-size: 14px;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📊 Marketing Mix Simulator</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666; margin-bottom: 20px;">
    Simulate how marketing budgets affect consumer choices | 
    <span style="color: #1f77b4;">Bayesian MMM</span> + 
    <span style="color: #ff7f0e;">Agent-Based Modeling</span>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.simulator = None
    st.session_state.results = None
    st.session_state.step_count = 0
    st.session_state.is_running = False
    st.session_state.history = {'market_share': [], 'clustering': []}
    st.session_state.agents = []
    st.session_state.network = []
    st.session_state.scenario_results = {}

# ============ LAYOUT: 类似 consumer_choice_abm ============
# Left sidebar for controls
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # --- Community Settings ---
    st.markdown("### 🏘️ Community Settings")
    
    num_consumers = st.slider(
        "Number of Consumers",
        min_value=50,
        max_value=300,
        value=200,
        step=10,
        key="num_consumers"
    )
    
    avg_connections = st.slider(
        "Avg Connections",
        min_value=0.0,
        max_value=12.0,
        value=6.0,
        step=0.5,
        key="avg_connections"
    )
    
    init_orange_pct = st.slider(
        "Initial Orange %",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        key="init_orange_pct"
    )
    
    st.markdown("---")
    
    # --- Budget Settings ---
    st.markdown("### 💰 Budget Settings")
    
    total_budget = st.number_input(
        "Total Budget ($)",
        min_value=100000,
        max_value=2000000,
        value=500000,
        step=50000,
        key="total_budget"
    )
    
    st.markdown("#### Channel Allocation")
    
    col1, col2 = st.columns(2)
    with col1:
        tv_pct = st.slider("TV %", 0, 100, 25, 5, key="tv_pct")
        digital_pct = st.slider("Digital %", 0, 100, 30, 5, key="digital_pct")
    with col2:
        social_pct = st.slider("Social %", 0, 100, 25, 5, key="social_pct")
        influencer_pct = st.slider("Influencer %", 0, 100, 20, 5, key="influencer_pct")
    
    # Validate allocation
    total_pct = tv_pct + digital_pct + social_pct + influencer_pct
    if total_pct != 100:
        st.warning(f"⚠️ Total: {total_pct}% - Please adjust to 100%")
    else:
        budget_allocation = {
            'tv_spend': total_budget * tv_pct / 100,
            'digital_spend': total_budget * digital_pct / 100,
            'social_spend': total_budget * social_pct / 100,
            'influencer_spend': total_budget * influencer_pct / 100
        }
        st.success(f"✅ ${total_budget:,.0f} allocated")
    
    st.markdown("---")
    
    # --- Agent Settings ---
    st.markdown("### 🧠 Agent Settings")
    
    norm_influence = st.slider(
        "Norm Influence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="norm_influence"
    )
    
    info_exchange = st.slider(
        "Info Exchange",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="info_exchange"
    )
    
    exploration = st.slider(
        "Exploration",
        min_value=0.0,
        max_value=0.2,
        value=0.01,
        step=0.01,
        key="exploration"
    )
    
    st.markdown("---")
    
    # --- Buttons ---
    st.markdown("### 🎮 Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        setup_btn = st.button("🔄 Setup", type="primary", use_container_width=True)
    with col2:
        start_btn = st.button("▶️ Start", use_container_width=True)
    
    col3, col4, col5 = st.columns(3)
    with col3:
        pause_btn = st.button("⏸️ Pause", use_container_width=True)
    with col4:
        reset_btn = st.button("🔁 Reset", use_container_width=True)
    with col5:
        step_btn = st.button("👣 Step", use_container_width=True)
    
    st.markdown("---")
    
    # --- Status ---
    st.markdown("### 📊 Statistics")
    status_placeholder = st.empty()
    step_placeholder = st.empty()
    share_placeholder = st.empty()
    cluster_placeholder = st.empty()

# ============ MAIN CONTENT ============
# Top row: Agent Space (canvas like)
st.markdown("## 🌍 Agent Space")

# Create a placeholder for the agent visualization
agent_space = st.empty()

# Use plotly for agent visualization (like p5.js but with plotly)
def plot_agent_space(agents, network):
    """Create a scatter plot of agents with network connections"""
    if not agents:
        fig = go.Figure()
        fig.add_annotation(
            text="Click 'Setup' to initialize agents",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            xaxis=dict(range=[0, 850], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[0, 450], showgrid=False, zeroline=False, visible=False),
            plot_bgcolor='white',
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig
    
    # Draw network edges
    edge_x = []
    edge_y = []
    for edge in network:
        source = next((a for a in agents if a['id'] == edge['source']), None)
        target = next((a for a in agents if a['id'] == edge['target']), None)
        if source and target:
            edge_x.extend([source['x'], target['x'], None])
            edge_y.extend([source['y'], target['y'], None])
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(color='lightgray', width=1),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Add nodes
    colors = ['orange' if a['choice'] == 0 else 'blue' for a in agents]
    sizes = [10 + a['satisfaction'] * 10 for a in agents]  # Size based on satisfaction
    texts = [f"ID: {a['id']}<br>Choice: {'Orange' if a['choice'] == 0 else 'Blue'}<br>Satisfaction: {a['satisfaction']:.2f}" for a in agents]
    
    fig.add_trace(go.Scatter(
        x=[a['x'] for a in agents],
        y=[a['y'] for a in agents],
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=1, color='white'),
            opacity=0.8
        ),
        text=texts,
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        xaxis=dict(range=[0, 850], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 450], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='#f8f9fa',
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode='closest'
    )
    
    return fig

# Stats row (like consumer_choice_abm)
stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

with stats_col1:
    st.metric("⏱️ Time Step", st.session_state.step_count)

with stats_col2:
    share = st.session_state.history['market_share'][-1] if st.session_state.history['market_share'] else 0
    st.metric("🟠 Orange Share", f"{share:.1f}%")

with stats_col3:
    cluster = st.session_state.history['clustering'][-1] if st.session_state.history['clustering'] else 0
    st.metric("🔗 Clustering", f"{cluster:.3f}")

with stats_col4:
    if st.session_state.results:
        pred = st.session_state.results.get('mmm_prediction', {})
        pred_mean = pred.get('mean', 0)
        st.metric("📈 Predicted Sales", f"${pred_mean:,.0f}")
    else:
        st.metric("📈 Predicted Sales", "$0")

# Charts row (like consumer_choice_abm)
st.markdown("---")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### 📈 Market Shares")
    market_chart = st.empty()

with chart_col2:
    st.markdown("### 🔗 Clustering Over Time")
    cluster_chart = st.empty()

# ============ SIMULATION LOGIC ============

def initialize_simulator():
    """Initialize the simulator with current parameters"""
    try:
        # Generate data
        data = generate_marketing_data()
        
        # Create config
        class Config:
            NUM_CONSUMERS = st.session_state.num_consumers
            AVG_CONNECTIONS = st.session_state.avg_connections
            INIT_ORANGE_PCT = st.session_state.init_orange_pct
            QUALITY_ORANGE = 0.45
            QUALITY_BLUE = 0.50
            NORM_INFLUENCE = st.session_state.norm_influence
            INFO_EXCHANGE = st.session_state.info_exchange
            EXPLORATION = st.session_state.exploration
            SEED = 42
            NUM_STEPS = 100
        
        config = Config()
        simulator = MarketingSimulator(config, data)
        
        # Get initial agent data
        agents_data = []
        for c in simulator.abm.consumers:
            agents_data.append({
                'id': c.id,
                'x': c.x,
                'y': c.y,
                'choice': c.choice,
                'satisfaction': c.satisfaction[c.choice] if c.satisfaction[c.choice] else 0.5
            })
        
        # Get network data
        network_data = []
        seen = set()
        for c in simulator.abm.consumers:
            for n in c.neighbors:
                edge = tuple(sorted([c.id, n.id]))
                if edge not in seen:
                    seen.add(edge)
                    network_data.append({'source': c.id, 'target': n.id})
        
        st.session_state.simulator = simulator
        st.session_state.agents = agents_data
        st.session_state.network = network_data
        st.session_state.step_count = 0
        st.session_state.history = {'market_share': [simulator.abm.get_market_share()], 'clustering': [0]}
        st.session_state.initialized = True
        st.session_state.results = None
        
        return True
    except Exception as e:
        st.error(f"Initialization error: {e}")
        return False

def run_simulation_step():
    """Run one simulation step"""
    if st.session_state.simulator and st.session_state.is_running:
        # Run ABM step
        st.session_state.simulator.abm.step()
        st.session_state.step_count += 1
        
        # Update history
        st.session_state.history['market_share'].append(st.session_state.simulator.abm.get_market_share())
        st.session_state.history['clustering'].append(st.session_state.simulator.abm.history['clustering'][-1])
        
        # Update agents
        agents_data = []
        for c in st.session_state.simulator.abm.consumers:
            agents_data.append({
                'id': c.id,
                'x': c.x,
                'y': c.y,
                'choice': c.choice,
                'satisfaction': c.satisfaction[c.choice] if c.satisfaction[c.choice] else 0.5
            })
        st.session_state.agents = agents_data
        
        # Update network (if changed)
        network_data = []
        seen = set()
        for c in st.session_state.simulator.abm.consumers:
            for n in c.neighbors:
                edge = tuple(sorted([c.id, n.id]))
                if edge not in seen:
                    seen.add(edge)
                    network_data.append({'source': c.id, 'target': n.id})
        st.session_state.network = network_data
        
        # Update results
        st.session_state.results = {
            'final_market_share': st.session_state.history['market_share'][-1],
            'final_clustering': st.session_state.history['clustering'][-1],
            'mmm_prediction': {'mean': 0, 'std': 0}
        }
        
        return True
    return False

# ============ BUTTON HANDLERS ============

if setup_btn:
    with st.spinner("Setting up simulator..."):
        initialize_simulator()

if start_btn and st.session_state.initialized:
    st.session_state.is_running = True
    # Run simulation in a loop (Streamlit will handle the updates)
    # We'll use a while loop with small increments
    for _ in range(10):  # Run 10 steps per click for smoothness
        if st.session_state.is_running:
            run_simulation_step()
            time.sleep(0.05)  # Small delay for visualization

if pause_btn:
    st.session_state.is_running = False

if reset_btn:
    st.session_state.is_running = False
    with st.spinner("Resetting..."):
        initialize_simulator()

if step_btn:
    if st.session_state.initialized:
        run_simulation_step()

# ============ UPDATE VISUALIZATIONS ============

# Update agent space
if st.session_state.agents:
    fig = plot_agent_space(st.session_state.agents, st.session_state.network)
    agent_space.plotly_chart(fig, use_container_width=True)
else:
    fig = plot_agent_space([], [])
    agent_space.plotly_chart(fig, use_container_width=True)

# Update market share chart
if st.session_state.history['market_share']:
    df = pd.DataFrame({
        'Step': range(len(st.session_state.history['market_share'])),
        'Market Share (%)': st.session_state.history['market_share']
    })
    fig = px.line(
        df,
        x='Step',
        y='Market Share (%)',
        title='Orange Product Market Share'
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% Baseline")
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
    market_chart.plotly_chart(fig, use_container_width=True)

# Update clustering chart
if st.session_state.history['clustering']:
    df = pd.DataFrame({
        'Step': range(len(st.session_state.history['clustering'])),
        'Clustering': st.session_state.history['clustering']
    })
    fig = px.line(
        df,
        x='Step',
        y='Clustering',
        title='Clustering Coefficient'
    )
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
    cluster_chart.plotly_chart(fig, use_container_width=True)

# Update statistics in sidebar
status_placeholder.info("✅ Running" if st.session_state.is_running else "⏸️ Paused" if st.session_state.initialized else "🔴 Not initialized")
step_placeholder.markdown(f"**Step:** {st.session_state.step_count}")
share = st.session_state.history['market_share'][-1] if st.session_state.history['market_share'] else 0
share_placeholder.markdown(f"**Orange Share:** {share:.1f}%")
cluster = st.session_state.history['clustering'][-1] if st.session_state.history['clustering'] else 0
cluster_placeholder.markdown(f"**Clustering:** {cluster:.3f}")

# ============ AUTO-RUN ============
# Auto-run simulation if running
if st.session_state.is_running and st.session_state.initialized:
    # Run one step and trigger rerun
    run_simulation_step()
    time.sleep(0.05)
    st.rerun()

# ============ LEGEND ============
st.markdown("---")
legend_col1, legend_col2, legend_col3 = st.columns(3)
with legend_col1:
    st.markdown("🟠 Orange Product Consumer")
with legend_col2:
    st.markdown("🔵 Blue Product Consumer")
with legend_col3:
    st.markdown("🔗 Social Connection")