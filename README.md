# Marketing Mix Simulator

## Overview

This project combines **Bayesian Media Mix Modeling (MMM)** with **Agent-Based Modeling (ABM)** to simulate how marketing spend influences consumer behavior and market outcomes.

## Installation
Prerequisites
Python 3.8 or higher

pip package manager

### Step 1: Clone the Repository
bash
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

## Usage Guide
Dashboard Interface

### Left Panel: Controls
Section	Parameters	Description

Community Settings	Number of Consumers, Avg Connections, Initial Orange %	Agent population and network structure

Budget Settings	Total Budget, Channel Allocation (%)	Marketing budget distribution

Agent Settings	Norm Influence, Info Exchange, Exploration	Consumer behaviour parameters

Controls	Setup, Start, Pause, Reset, Step	Simulation control buttons
### Main Area
Agent Space	Real-time visualisation of consumers (orange/blue circles) and social connections (grey lines)

Market Shares Chart	Tracks orange product adoption over time

Clustering Chart	Monitors social homophily dynamics

## Model Logic
### Media Mix Modelling (MMM)
Simplified MMM with adstock and saturation

Y = α + Σ(βᵢ × Adstock(Xᵢ)) + ε

Adstock: Marketing carryover effect (X_t = spend_t + α × X_{t-1})

Saturation: Diminishing returns (effect = max_effect × (1 - exp(-X / half_saturation)))

### Agent-Based Model (ABM)
Consumer decision process each step:

Satisfaction Check: Probability to stick = satisfaction[current_product]

Exploration: Random switch with exploration_rate probability

Utility Comparison: Choose product with higher utility

Utility = (personal_experience + neighbor_info) / 2 + marketing_effect

Marketing Integration

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

