# model/simulator.py
# Integration of MMM and ABM

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# 尝试导入 MMM
try:
    from .mmm_model import BayesianMMM
except ImportError:
    from .mmm_simple import SimpleMMM as BayesianMMM
    print("Using SimpleMMM (BayesianMMM not available)")

from .abm_model import ABM_Model_Marketing

class MarketingSimulator:
    """Complete marketing simulation integrating MMM and ABM"""
    
    def __init__(self, config, marketing_data: pd.DataFrame):
        self.config = config
        self.marketing_data = marketing_data
        
        # Initialize MMM
        self.mmm = BayesianMMM(marketing_data)
        self.mmm.build_model()
        self.mmm.fit(draws=500, tune=500, chains=2, cores=2)
        
        # Extract marketing effects
        self.marketing_effects = self.mmm.get_channel_effects()
        
        # Initialize ABM
        self.abm = ABM_Model_Marketing(config, self.marketing_effects)
        self.abm.setup()
        
        self.scenario_results = {}
        self.baseline_share = 50.0
    
    def run_scenario(
        self,
        budget_allocation: Dict[str, float],
        steps: int = 50,
        scenario_name: str = None
    ) -> Dict:
        """Run a marketing scenario simulation"""
        if scenario_name is None:
            scenario_name = f"Scenario_{len(self.scenario_results)}"
        
        print(f"\nRunning scenario: {scenario_name}")
        
        # Reset ABM
        self.abm.setup()
        
        # Apply marketing campaign
        self.abm.apply_marketing_campaign(budget_allocation)
        
        # Run simulation
        results = []
        for step in range(steps):
            self.abm.step()
            results.append({
                'step': step,
                'market_share': self.abm.get_market_share(),
                'clustering': self.abm.history['clustering'][-1],
                'orange_count': self.abm.history['orange_count'][-1],
                'blue_count': self.abm.history['blue_count'][-1]
            })
        
        # MMM prediction
        mmm_prediction = self.mmm.predict_scenario(budget_allocation)
        
        # Calculate ROI
        total_budget = sum(budget_allocation.values())
        share_gain = results[-1]['market_share'] - self.baseline_share
        roi = share_gain / total_budget if total_budget > 0 else 0
        
        scenario_result = {
            'name': scenario_name,
            'budget': budget_allocation,
            'total_budget': total_budget,
            'abm_results': pd.DataFrame(results),
            'mmm_prediction': mmm_prediction,
            'final_market_share': results[-1]['market_share'],
            'final_clustering': results[-1]['clustering'],
            'share_gain': share_gain,
            'roi': roi,
            'steps': steps
        }
        
        self.scenario_results[scenario_name] = scenario_result
        return scenario_result
    
    def compare_scenarios(self) -> pd.DataFrame:
        """Compare all run scenarios"""
        if not self.scenario_results:
            return pd.DataFrame()
        
        comparison = []
        for name, result in self.scenario_results.items():
            comparison.append({
                'Scenario': name,
                'Total Budget': result['total_budget'],
                'Final Market Share (%)': result['final_market_share'],
                'Share Gain (%)': result['share_gain'],
                'Clustering': result['final_clustering'],
                'ROI': result['roi'],
                'Predicted Sales': result['mmm_prediction']['mean']
            })
        
        return pd.DataFrame(comparison)
    
    def optimize_budget(
        self,
        total_budget: float,
        channels: List[str],
        n_iterations: int = 30
    ) -> Dict:
        """Simple budget optimization using random search"""
        best_roi = -float('inf')
        best_allocation = None
        
        for i in range(n_iterations):
            weights = np.random.dirichlet(np.ones(len(channels)))
            allocation = {
                channel: total_budget * weight
                for channel, weight in zip(channels, weights)
            }
            
            result = self.run_scenario(
                allocation,
                steps=20,
                scenario_name=f"Opt_{i}"
            )
            
            if result['roi'] > best_roi:
                best_roi = result['roi']
                best_allocation = allocation
        
        return best_allocation, best_roi
    
    def get_budget_share_curve(
        self,
        budget_range: tuple = (100000, 1500000),
        n_points: int = 15
    ) -> Dict:
        """Generate budget-market share curve"""
        budgets = np.linspace(budget_range[0], budget_range[1], n_points)
        results = []
        
        for b in budgets:
            per_channel = b / 4
            budget = {
                'tv_spend': per_channel,
                'digital_spend': per_channel,
                'social_spend': per_channel,
                'influencer_spend': per_channel
            }
            result = self.run_scenario(budget, steps=30, scenario_name=f"Curve_{b:.0f}")
            results.append({
                'budget': b,
                'final_share': result['final_market_share'],
                'roi': result['roi']
            })
        
        return {
            'budgets': [r['budget'] for r in results],
            'market_shares': [r['final_share'] for r in results],
            'roi': [r['roi'] for r in results]
        }
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary"""
        return {
            'mmm_summary': self.mmm.get_summary(),
            'scenarios': len(self.scenario_results),
            'marketing_effects': self.marketing_effects
        }


if __name__ == '__main__':
    from data.generate_data import generate_marketing_data
    from config import Config
    
    data = generate_marketing_data()
    config = Config()
    simulator = MarketingSimulator(config, data)
    
    # Run scenarios
    scenarios = [
        {'tv_spend': 200, 'digital_spend': 150, 'social_spend': 100, 'influencer_spend': 50},
        {'tv_spend': 100, 'digital_spend': 200, 'social_spend': 100, 'influencer_spend': 100},
        {'tv_spend': 150, 'digital_spend': 150, 'social_spend': 150, 'influencer_spend': 50},
    ]
    
    for i, budget in enumerate(scenarios):
        simulator.run_scenario(budget, steps=30, scenario_name=f"Scenario_{i+1}")
    
    comparison = simulator.compare_scenarios()
    print("\nScenario Comparison:")
    print(comparison)