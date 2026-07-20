# streamlit_app/app.py
# Interactive Marketing Simulator Dashboard - Fixed Version

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import random
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import generate_marketing_data
from model.simulator import MarketingSimulator
from model.abm_model import ABM_Model_Marketing, Consumer
from config import Config

# Page config
st.set_page_config(
    page_title="Consumer Choice + Marketing Simulator",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Consumer Choice ABM with Marketing Simulation")
st.markdown("""
    **Agent-Based Modeling + Bayesian Marketing Mix Modeling**  
    Simulate consumer choices and marketing impact in one unified interface
""")

# Initialize session state
if 'simulator' not in st.session_state:
    with st.spinner("Initializing simulator..."):
        data = generate_marketing_data()
        config = Config()
        st.session_state.simulator = MarketingSimulator(config, data)
        st.session_state.results = {}
        st.session_state.scenarios = {}
        st.session_state.abm_initialized = False

# ============================================================
# LEFT SIDEBAR - Complete ABM + Marketing Controls
# ============================================================
with st.sidebar:
    st.header("⚙️ Controls")
    
    # ---- ABM Parameters ----
    st.subheader("🧠 Consumer ABM Parameters")
    
    # Community Settings
    with st.expander("🏘️ Community Settings", expanded=True):
        num_consumers = st.slider(
            "Number of Consumers",
            min_value=20,
            max_value=300,
            value=100,
            step=10,
            key="num_consumers"
        )
        
        avg_connections = st.slider(
            "Avg Connections",
            min_value=0.0,
            max_value=15.0,
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
    
    # Product Settings
    with st.expander("📦 Product Settings", expanded=True):
        quality_orange = st.slider(
            "Quality Orange",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.01,
            key="quality_orange"
        )
        
        quality_blue = st.slider(
            "Quality Blue",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.01,
            key="quality_blue"
        )
    
    # Agent Settings
    with st.expander("🎮 Agent Settings", expanded=True):
        norm_influence = st.slider(
            "Norm Influence",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            key="norm_influence"
        )
        
        info_exchange = st.slider(
            "Info Exchange",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            key="info_exchange"
        )
        
        exploration = st.slider(
            "Exploration",
            min_value=0.0,
            max_value=0.20,
            value=0.01,
            step=0.01,
            key="exploration"
        )
    
    st.markdown("---")
    
    # ---- Marketing Parameters ----
    st.subheader("📢 Marketing Budget")
    
    total_budget = st.number_input(
        "Total Budget ($)",
        min_value=100000,
        max_value=2000000,
        value=500000,
        step=50000,
        key="total_budget"
    )
    
    st.subheader("Channel Allocation")
    
    channels = ['tv_spend', 'digital_spend', 'social_spend', 'influencer_spend']
    channel_labels = ['📺 TV', '📱 Digital', '📢 Social', '🌟 Influencer']
    
    allocations = {}
    cols = st.columns(2)
    defaults = [30, 35, 20, 15]
    
    for i, (channel, label) in enumerate(zip(channels, channel_labels)):
        with cols[i % 2]:
            allocations[channel] = st.slider(
                f"{label} (%)",
                min_value=0,
                max_value=100,
                value=defaults[i],
                step=5,
                key=f"mkt_{channel}"
            )
    
    total_pct = sum(allocations.values())
    if total_pct != 100:
        st.warning(f"⚠️ Allocation sums to {total_pct}%. Adjust to 100%.")
    else:
        budget_allocation = {
            channel: total_budget * pct / 100
            for channel, pct in allocations.items()
        }
        
        # Display budget breakdown
        allocation_df = pd.DataFrame({
            'Channel': [label.replace('📺 ', '').replace('📱 ', '').replace('📢 ', '').replace('🌟 ', '') for label in channel_labels],
            'Spend ($)': [budget_allocation[c] for c in channels],
            '%': [allocations[c] for c in channels]
        })
        st.dataframe(allocation_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ---- Action Buttons ----
    col1, col2 = st.columns(2)
    
    with col1:
        setup_abm = st.button("🔄 Setup ABM", type="primary", use_container_width=True)
    
    with col2:
        run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    # Reset button
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.results = {}
        st.session_state.scenarios = {}
        st.session_state.abm_initialized = False
        st.rerun()

# ============================================================
# MAIN CONTENT AREA
# ============================================================

# Update config with current parameters
st.session_state.simulator.config.NUM_CONSUMERS = num_consumers
st.session_state.simulator.config.AVG_CONNECTIONS = avg_connections
st.session_state.simulator.config.INIT_ORANGE_PCT = init_orange_pct
st.session_state.simulator.config.QUALITY_ORANGE = quality_orange
st.session_state.simulator.config.QUALITY_BLUE = quality_blue
st.session_state.simulator.config.NORM_INFLUENCE = norm_influence
st.session_state.simulator.config.INFO_EXCHANGE = info_exchange
st.session_state.simulator.config.EXPLORATION = exploration

# Setup ABM
if setup_abm:
    with st.spinner("Setting up ABM..."):
        st.session_state.simulator.abm = ABM_Model_Marketing(
            st.session_state.simulator.config,
            st.session_state.simulator.marketing_effects
        )
        st.session_state.simulator.abm.setup()
        st.session_state.abm_initialized = True
        st.success(f"✅ ABM initialized with {num_consumers} consumers!")

# Run simulation
if run_button and total_pct == 100:
    with st.spinner("Running simulation..."):
        result = st.session_state.simulator.run_scenario(
            budget_allocation,
            steps=50,
            scenario_name=f"Budget_{total_budget}"
        )
        st.session_state.results = result
        st.session_state.abm_initialized = True
        st.success("✅ Simulation complete!")

# ============================================================
# TAB LAYOUT
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Agent Space & Metrics",
    "📈 Market Trends",
    "🔬 Marketing Efficiency",
    "🎯 Scenario Analysis"
])

# ============================================================
# TAB 1: AGENT SPACE & METRICS
# ============================================================
with tab1:
    # Get agent data
    if st.session_state.abm_initialized and st.session_state.simulator.abm.consumers:
        consumers = st.session_state.simulator.abm.consumers
        
        # ---- Agent Space Visualization (Plotly) ----
        st.subheader("🌍 Agent Space")
        
        # Prepare agent data for plotting
        agent_data = []
        for c in consumers:
            agent_data.append({
                'x': c.x,
                'y': c.y,
                'choice': 'Orange' if c.choice == 0 else 'Blue',
                'satisfaction': c.satisfaction[c.choice],
                'id': c.id
            })
        
        agent_df = pd.DataFrame(agent_data)
        
        # Create scatter plot
        fig = px.scatter(
            agent_df,
            x='x',
            y='y',
            color='choice',
            size='satisfaction',
            size_max=15,
            color_discrete_map={'Orange': 'orange', 'Blue': 'blue'},
            title='Consumer Network - Color = Product Choice, Size = Satisfaction',
            labels={'x': '', 'y': ''},
            hover_data={'id': True, 'satisfaction': True}
        )
        
        # Add network edges
        edges = []
        for c in consumers:
            for n in c.neighbors:
                if c.id < n.id:
                    edges.append({
                        'x0': c.x, 'y0': c.y,
                        'x1': n.x, 'y1': n.y
                    })
        
        for edge in edges:
            fig.add_shape(
                type='line',
                x0=edge['x0'], y0=edge['y0'],
                x1=edge['x1'], y1=edge['y1'],
                line=dict(color='lightgray', width=1),
                opacity=0.5
            )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ---- KPI Cards ----
        col1, col2, col3, col4 = st.columns(4)
        
        summary = st.session_state.simulator.abm.get_summary()
        
        with col1:
            st.metric(
                "Orange Market Share",
                f"{summary['market_share']:.1f}%"
            )
        
        with col2:
            st.metric(
                "Clustering Coefficient",
                f"{summary['clustering']:.3f}"
            )
        
        with col3:
            st.metric(
                "Orange Consumers",
                f"{summary['orange_count']}"
            )
        
        with col4:
            st.metric(
                "Blue Consumers",
                f"{summary['blue_count']}"
            )
        
        # ---- Additional agent statistics ----
        st.subheader("📊 Agent Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            # Satisfaction distribution
            fig = px.histogram(
                agent_df,
                x='satisfaction',
                color='choice',
                color_discrete_map={'Orange': 'orange', 'Blue': 'blue'},
                title='Satisfaction Distribution by Product Choice',
                labels={'satisfaction': 'Satisfaction', 'count': 'Number of Consumers'},
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Choice composition
            choice_counts = agent_df['choice'].value_counts().reset_index()
            choice_counts.columns = ['Choice', 'Count']
            
            fig = px.pie(
                choice_counts,
                values='Count',
                names='Choice',
                title='Current Product Choice Composition',
                color='Choice',
                color_discrete_map={'Orange': 'orange', 'Blue': 'blue'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👈 Click **Setup ABM** to initialize the agent-based model")

# ============================================================
# TAB 2: MARKET TRENDS
# ============================================================
with tab2:
    if st.session_state.results:
        result = st.session_state.results
        df = result['abm_results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Market Share Evolution")
            fig = px.line(
                df,
                x='step',
                y='market_share',
                title='Orange Product Market Share Over Time',
                labels={'step': 'Time Step', 'market_share': 'Market Share (%)'}
            )
            fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Baseline")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔗 Clustering Over Time")
            fig = px.line(
                df,
                x='step',
                y='clustering',
                title='Social Clustering Coefficient',
                labels={'step': 'Time Step', 'clustering': 'Clustering'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Consumer base
        st.subheader("👥 Consumer Base by Product")
        fig = px.line(
            df,
            x='step',
            y=['orange_count', 'blue_count'],
            title='Number of Consumers by Product Choice',
            labels={'step': 'Time Step', 'value': 'Number of Consumers'},
            color_discrete_map={'orange_count': 'orange', 'blue_count': 'blue'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👈 Run a simulation to see market trends")

# ============================================================
# TAB 3: MARKETING EFFICIENCY - FIXED VERSION
# ============================================================
with tab3:
    if st.session_state.results:
        result = st.session_state.results
        
        st.subheader("📊 ROI Analysis")
        
        # --- Calculate ROI ---
        channel_roi = {}
        roi_raw = {}
        
        for channel in channels:
            spend = result['budget'].get(channel, 0)
            effect = st.session_state.simulator.mmm.results['channel_contributions'].get(channel, {})
            contribution = effect.get('mean', 0)
            
            # Raw ROI
            raw_roi = contribution / spend if spend > 0 else 0
            
            # Scaled ROI for visualization (×100,000)
            scaled_roi = raw_roi * 100000
            
            channel_name = channel.replace('_spend', '').capitalize()
            channel_roi[channel_name] = scaled_roi
            roi_raw[channel_name] = raw_roi
        
        roi_df = pd.DataFrame({
            'Channel': list(channel_roi.keys()),
            'ROI (×100,000)': list(channel_roi.values()),
            'Raw ROI': list(roi_raw.values())
        })
        
        # --- Display ROI Table ---
        st.dataframe(
            roi_df.style.format({
                'ROI (×100,000)': '{:.2f}',
                'Raw ROI': '{:.6f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # --- ROI Bar Chart (Fixed) ---
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                roi_df,
                x='Channel',
                y='ROI (×100,000)',
                title='Channel ROI (Scaled for Visualization)',
                color='Channel',
                text_auto='.2f'
            )
            fig.update_traces(
                textposition='outside',
                marker=dict(line=dict(width=2, color='DarkSlateGray'))
            )
            fig.update_layout(
                yaxis_title="ROI (×100,000)",
                showlegend=False,
                height=400,
                yaxis=dict(gridcolor='lightgray')
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("📌 ROI = Channel Contribution / Channel Spend × 100,000 (scaled for visualization)")
            st.caption("💡 Higher ROI = More efficient channel")
        
        with col2:
            # --- Efficiency Score ---
            st.subheader("📊 Efficiency Score")
            
            max_roi = roi_df['ROI (×100,000)'].max()
            if max_roi > 0:
                roi_df['Efficiency Score'] = (roi_df['ROI (×100,000)'] / max_roi * 100)
            else:
                roi_df['Efficiency Score'] = 0
            
            fig = px.bar(
                roi_df,
                x='Channel',
                y='Efficiency Score',
                title='Channel Efficiency Score (%)',
                color='Channel',
                text_auto='.0f'
            )
            fig.update_traces(
                textposition='outside',
                marker=dict(line=dict(width=2, color='DarkSlateGray'))
            )
            fig.update_layout(
                yaxis_title="Efficiency Score (%)",
                showlegend=False,
                height=400,
                yaxis=dict(range=[0, 110], gridcolor='lightgray')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Best channel recommendation
            best_channel = roi_df.loc[roi_df['ROI (×100,000)'].idxmax(), 'Channel']
            st.success(f"🏆 Most Efficient Channel: **{best_channel}**")
        
        # --- Efficiency Summary Cards ---
        st.subheader("📈 Efficiency Summary")
        
        total_roi = result.get('roi', 0)
        share_gain = result.get('share_gain', 0)
        total_budget_val = result.get('total_budget', 0)
        
        # ROI rating
        if total_roi > 0.01:
            rating = "🌟 Excellent"
            rating_color = "green"
        elif total_roi > 0.005:
            rating = "✅ Good"
            rating_color = "blue"
        elif total_roi > 0:
            rating = "⚠️ Moderate"
            rating_color = "orange"
        else:
            rating = "❌ Poor"
            rating_color = "red"
        
        cols = st.columns(4)
        
        with cols[0]:
            st.metric(
                "Total ROI",
                f"{total_roi:.4f}",
                delta=rating,
                delta_color="normal"
            )
        
        with cols[1]:
            st.metric(
                "Share Gain",
                f"{share_gain:.1f}%",
                delta="vs baseline"
            )
        
        with cols[2]:
            st.metric(
                "Total Budget",
                f"${total_budget_val:,.0f}",
                delta=""
            )
        
        with cols[3]:
            st.metric(
                "Final Market Share",
                f"{result['final_market_share']:.1f}%",
                delta=""
            )
        
        # --- ROI Interpretation Guide ---
        with st.expander("📖 How to Interpret ROI Analysis", expanded=False):
            st.markdown("""
            ### 📊 ROI Analysis Guide
            
            **What is ROI?**
            - ROI = Channel Contribution / Channel Spend
            - Higher ROI = Better efficiency
            
            **How to Read the Charts:**
            1. **Channel ROI Chart**: Shows which channels generate the best return
            2. **Efficiency Score**: Relative performance (100% = best channel)
            3. **Total ROI**: Overall campaign effectiveness
            
            **Actionable Insights:**
            - ✅ **High ROI (> average)**: Increase budget allocation
            - ⚠️ **Medium ROI (≈ average)**: Maintain current budget
            - ❌ **Low ROI (< average)**: Reduce budget or optimize strategy
            
            **Common Patterns:**
            - **Digital > TV**: Performance marketing is more efficient than brand building
            - **Social > Influencer**: Organic social outperforms paid influencer
            - **All channels similar**: Market is saturated, focus on creative optimization
            """)
    
    else:
        st.info("👈 Run a simulation to see marketing efficiency metrics")

# ============================================================
# TAB 4: SCENARIO ANALYSIS
# ============================================================
with tab4:
    st.subheader("🎯 Scenario Analysis")
    
    # Scenario presets
    col1, col2, col3, col4 = st.columns(4)
    
    scenarios = {
        "📺 TV Focus": {'tv_spend': 400000, 'digital_spend': 150000, 'social_spend': 100000, 'influencer_spend': 50000},
        "📱 Digital Focus": {'tv_spend': 100000, 'digital_spend': 400000, 'social_spend': 150000, 'influencer_spend': 50000},
        "⚖️ Balanced": {'tv_spend': 200000, 'digital_spend': 200000, 'social_spend': 150000, 'influencer_spend': 100000},
        "🌟 Influencer": {'tv_spend': 100000, 'digital_spend': 150000, 'social_spend': 150000, 'influencer_spend': 350000}
    }
    
    with col1:
        if st.button("📺 TV Focus", use_container_width=True):
            st.session_state.simulator.run_scenario(
                scenarios["📺 TV Focus"], steps=40, scenario_name="TV Focus"
            )
    with col2:
        if st.button("📱 Digital Focus", use_container_width=True):
            st.session_state.simulator.run_scenario(
                scenarios["📱 Digital Focus"], steps=40, scenario_name="Digital Focus"
            )
    with col3:
        if st.button("⚖️ Balanced", use_container_width=True):
            st.session_state.simulator.run_scenario(
                scenarios["⚖️ Balanced"], steps=40, scenario_name="Balanced"
            )
    with col4:
        if st.button("🌟 Influencer", use_container_width=True):
            st.session_state.simulator.run_scenario(
                scenarios["🌟 Influencer"], steps=40, scenario_name="Influencer"
            )
    
    # Display scenario comparison
    if st.session_state.simulator.scenario_results:
        comparison = st.session_state.simulator.compare_scenarios()
        
        st.subheader("📊 Scenario Comparison")
        
        # Highlight best scenario
        best_idx = comparison['ROI'].idxmax()
        best_scenario = comparison.loc[best_idx]
        st.success(f"🏆 Best Scenario: **{best_scenario['Scenario']}** with ROI = {best_scenario['ROI']:.4f}")
        
        st.dataframe(comparison, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                comparison,
                x='Scenario',
                y='Final Market Share (%)',
                title='Final Market Share by Scenario',
                color='Scenario'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                comparison,
                x='Scenario',
                y='ROI',
                title='ROI by Scenario',
                color='Scenario'
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        # Budget optimization
        st.subheader("🎯 Budget Optimization")
        
        if st.button("🔍 Find Optimal Allocation", type="primary"):
            with st.spinner("Optimizing..."):
                optimal, optimal_roi = st.session_state.simulator.optimize_budget(
                    total_budget,
                    ['tv_spend', 'digital_spend', 'social_spend', 'influencer_spend'],
                    n_iterations=30
                )
                
                st.success(f"✅ Optimal allocation found! ROI = {optimal_roi:.4f}")
                
                opt_df = pd.DataFrame({
                    'Channel': ['TV', 'Digital', 'Social', 'Influencer'],
                    'Optimal Spend ($)': [optimal[c] for c in channels],
                    'Percentage': [optimal[c]/total_budget*100 for c in channels]
                })
                st.dataframe(opt_df, use_container_width=True)
                
                # Compare current vs optimal
                current_spend = [budget_allocation[c] for c in channels]
                optimal_spend = [optimal[c] for c in channels]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Current', x=['TV', 'Digital', 'Social', 'Influencer'], y=current_spend))
                fig.add_trace(go.Bar(name='Optimal', x=['TV', 'Digital', 'Social', 'Influencer'], y=optimal_spend))
                fig.update_layout(title='Current vs Optimal Budget Allocation', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👈 Run scenario analyses to compare strategies")

# Footer
st.markdown("---")
st.caption("Built with PyMC-Marketing, Agent-Based Modeling, and Streamlit | 🟠 Orange = Product A | 🔵 Blue = Product B")
