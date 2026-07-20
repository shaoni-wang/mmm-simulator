# Marketing Mix Simulator

## Overview

This project combines **Bayesian Media Mix Modeling (MMM)** with **Agent-Based Modeling (ABM)** to simulate how marketing spend influences consumer behavior and market outcomes. The integrated dashboard provides real-time visualization of consumer choices, social networks, and marketing efficiency metrics.

## Key Features

- **Agent-Based Model**: Simulates individual consumer decisions with social network effects
- **Bayesian MMM**: Quantifies channel contributions and ROI using pymc-marketing
- **Marketing Integration**: MMM effects feed into ABM consumer utilities
- **Real-time Visualization**: Interactive 2D agent space with network display
- **Scenario Analysis**: Compare different budget allocations
- **Budget Optimization**: Find optimal marketing mix

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/mmm_simulator.git
cd mmm_simulator

### Step 2: Install Dependencies
bash
pip install -r requirements.txt

### Step 3: Generate Sample Data
bash
python -m data.generate_data

### Step 4: Launch the Dashboard
bash
streamlit run streamlit_app/app.py

Open your browser at http://localhost:8501

Usage Guide
Dashboard Interface
### Left Panel: Controls
The left panel organizes all simulation controls into four sections. Community Settings (Number of Consumers, Avg Connections, Initial Orange %) control the agent population and network structure. Product Settings (Quality Orange, Quality Blue) define product quality attributes. Agent Settings (Norm Influence, Info Exchange, Exploration) determine consumer behavior. Marketing Budget lets you set total budget and allocate across four channels: 📺 TV (brand building), 📱 Digital (performance marketing), 📢 Social (engagement), and 🌟 Influencer (niche audiences). The Controls section provides Setup ABM, Run Simulation, and Reset All buttons.

### Main Area: Four Tabs
The main area features four analysis tabs. 🌍 Agent Space & Metrics displays a 2D network where Orange/Blue circles represent product choice, circle size indicates satisfaction, and grey lines show social connections. This tab also includes satisfaction distributions and KPI cards. 📈 Market Trends tracks market share, clustering, and consumer base over time. 🔬 Marketing Efficiency shows channel ROI, budget allocation, and efficiency metrics. 🎯 Scenario Analysis enables scenario comparison and budget optimization to find the optimal marketing mix.

### Getting Started
Click Setup ABM to initialize the agent population. Adjust marketing budget and allocation, then click Run Simulation to observe how marketing influences consumer choices. Monitor Market Trends for adoption patterns, Marketing Efficiency for channel performance, and use Scenario Analysis to compare budget strategies and find the optimal allocation.

## Model Logic
### Media Mix Modelling (MMM)
Simplified MMM with adstock and saturation

Y = α + Σ(βᵢ × Adstock(Xᵢ)) + ε

Adstock: Marketing carryover effect (X_t = spend_t + α × X_{t-1})

Saturation: Diminishing returns (effect = max_effect × (1 - exp(-X / half_saturation)))

### Agent-Based Model (ABM)
#### Consumer decision process each step:

Satisfaction Check: Probability to stick = satisfaction[current_product]

Exploration: Random switch with exploration_rate probability

Utility Comparison: Choose product with higher utility

Utility = (personal_experience + neighbor_info) / 2 + marketing_effect

#### Marketing Integration

Marketing effects modify consumer utility:

enhanced_utility = base_utility × (1 + marketing_boost × 0.2)

Where marketing_boost is derived from channel allocation.

## Example Scenarios
### Scenario 1: TV Focus
Budget: 70% TV, 20% Digital, 10% Social

Result: Slow but steady market share growth (brand building)

### Scenario 2: Digital Focus
Budget: 20% TV, 60% Digital, 20% Social

Result: Rapid initial adoption but diminishing returns

### Scenario 3: Balanced
Budget: 30% TV, 30% Digital, 25% Social, 15% Influencer

Result: Optimal mix with sustained growth

