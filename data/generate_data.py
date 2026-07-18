# data/generate_data.py
# Generate synthetic marketing and sales data

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_marketing_data(
    start_date='2023-01-01',
    weeks=104,  # 2 years
    seed=42
):
    """
    Generate synthetic marketing spend and sales data
    
    Returns:
        DataFrame with columns: date, tv_spend, digital_spend, social_spend,
        influencer_spend, competitors_spend, sales
    """
    np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=weeks, freq='W')
    
    # Base trends
    trend = np.linspace(0, 0.3, weeks)  # Market growth over time
    
    # Seasonality: annual pattern
    seasonality = 0.2 * np.sin(2 * np.pi * np.arange(weeks) / 52)
    
    # Marketing spend patterns with seasonality and trend
    tv_base = 100 + 20 * np.sin(2 * np.pi * np.arange(weeks) / 26) + trend * 50
    digital_base = 80 + 15 * np.sin(2 * np.pi * np.arange(weeks) / 13 + 0.5) + trend * 40
    social_base = 50 + 10 * np.sin(2 * np.pi * np.arange(weeks) / 8 + 1.0) + trend * 30
    influencer_base = 30 + 5 * np.sin(2 * np.pi * np.arange(weeks) / 6 + 2.0) + trend * 20
    
    # Add random noise
    tv_spend = tv_base * (1 + 0.2 * np.random.randn(weeks))
    digital_spend = digital_base * (1 + 0.2 * np.random.randn(weeks))
    social_spend = social_base * (1 + 0.2 * np.random.randn(weeks))
    influencer_spend = influencer_base * (1 + 0.2 * np.random.randn(weeks))
    
    # Ensure non-negative
    tv_spend = np.maximum(tv_spend, 10)
    digital_spend = np.maximum(digital_spend, 10)
    social_spend = np.maximum(social_spend, 10)
    influencer_spend = np.maximum(influencer_spend, 10)
    
    # Competitors spend (correlated with TV)
    competitors_spend = 0.7 * tv_spend + 20 * np.random.randn(weeks)
    competitors_spend = np.maximum(competitors_spend, 10)
    
    # Marketing effects with adstock (carryover)
    # Simulate adstock: current spend + 0.5 * previous spend
    adstock_factor = 0.5
    tv_effect = np.zeros(weeks)
    digital_effect = np.zeros(weeks)
    social_effect = np.zeros(weeks)
    influencer_effect = np.zeros(weeks)
    
    for i in range(1, weeks):
        tv_effect[i] = 0.3 * tv_spend[i] + adstock_factor * tv_effect[i-1]
        digital_effect[i] = 0.4 * digital_spend[i] + adstock_factor * digital_effect[i-1]
        social_effect[i] = 0.25 * social_spend[i] + adstock_factor * social_effect[i-1]
        influencer_effect[i] = 0.35 * influencer_spend[i] + adstock_factor * influencer_effect[i-1]
    
    # Saturation: diminishing returns with exponential function
    def saturation(x, max_effect=1000, half_saturation=200):
        return max_effect * (1 - np.exp(-x / half_saturation))
    
    tv_effect_sat = saturation(tv_effect, max_effect=800, half_saturation=250)
    digital_effect_sat = saturation(digital_effect, max_effect=600, half_saturation=180)
    social_effect_sat = saturation(social_effect, max_effect=400, half_saturation=120)
    influencer_effect_sat = saturation(influencer_effect, max_effect=300, half_saturation=100)
    
    # Competitors effect (negative)
    competitors_effect = -0.1 * competitors_spend
    
    # Calculate sales
    baseline = 500 + seasonality * 200 + trend * 300
    sales = (
        baseline +
        tv_effect_sat +
        digital_effect_sat +
        social_effect_sat +
        influencer_effect_sat +
        competitors_effect +
        50 * np.random.randn(weeks)  # Random noise
    )
    sales = np.maximum(sales, 100)  # Ensure non-negative
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'tv_spend': tv_spend,
        'digital_spend': digital_spend,
        'social_spend': social_spend,
        'influencer_spend': influencer_spend,
        'competitors_spend': competitors_spend,
        'sales': sales
    })
    
    return data

def generate_abm_simulation_data(n_consumers=200, n_steps=100, seed=42):
    """
    Generate synthetic ABM output data for testing integration
    """
    np.random.seed(seed)
    
    # Simulate market share evolution
    market_shares = []
    for t in range(n_steps):
        # S-curve adoption pattern
        if t < 30:
            share = 0.5 + 0.2 * np.random.randn()
        elif t < 70:
            share = 0.5 + 0.3 * np.sin(0.1 * t) + 0.1 * np.random.randn()
        else:
            share = 0.5 + 0.2 * np.sin(0.05 * t) + 0.05 * np.random.randn()
        market_shares.append(max(0, min(1, share)))
    
    return pd.DataFrame({
        'step': range(n_steps),
        'market_share': market_shares,
        'clustering': [0.5 + 0.3 * np.random.rand() for _ in range(n_steps)]
    })

if __name__ == '__main__':
    # Generate and save marketing data
    marketing_data = generate_marketing_data()
    marketing_data.to_csv('data/marketing_data.csv', index=False)
    print(f"Marketing data generated: {len(marketing_data)} weeks")
    
    # Generate and save ABM simulation data
    abm_data = generate_abm_simulation_data()
    abm_data.to_csv('data/abm_simulation.csv', index=False)
    print(f"ABM data generated: {len(abm_data)} steps")