import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from .mmm_model import BayesianMMM
from .abm_model import ABM_Model_Marketing


class MarketingSimulator:
    """
    Complete marketing simulation integrating MMM and ABM
    """
    
    def __init__(self, config, marketing_data: pd.DataFrame):
        """
        Initialize the simulator
        """
        self.config = config
        self.marketing_data = marketing_data
        
        # Initialize MMM
        self.mmm = BayesianMMM(marketing_data)
        self.mmm.fit()
        
        # Extract marketing effects
        self.marketing_effects = self.mmm.get_channel_effects()
        print(f"Marketing effects: {self.marketing_effects}")
        
        # Initialize ABM with marketing effects
        self.abm = ABM_Model_Marketing(config, self.marketing_effects)
        self.abm.setup()
        
        self.scenario_results = {}
        print("✅ Simulator ready!")
    
    def run_scenario(
        self,
        budget_allocation: Dict[str, float],
        steps: int = 50,
        scenario_name: str = None
    ) -> Dict:
        """
        Run a marketing scenario simulation
        """
        if scenario_name is None:
            scenario_name = f"Scenario_{len(self.scenario_results)}"
        
        print(f"\n▶️ Running scenario: {scenario_name}")
        print(f"   Budget: {budget_allocation}")
        
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
                'step_count': self.abm.step_count
            })
        
        # MMM prediction
        mmm_prediction = self.mmm.predict_scenario(budget_allocation)
        
        # Store results
        scenario_result = {
            'name': scenario_name,
            'budget': budget_allocation,
            'total_budget': sum(budget_allocation.values()),
            'abm_results': pd.DataFrame(results),
            'mmm_prediction': mmm_prediction,
            'final_market_share': results[-1]['market_share'],
            'final_clustering': results[-1]['clustering'],
            'roi': self._calculate_roi(budget_allocation, results[-1]['market_share'])
        }
        
        self.scenario_results[scenario_name] = scenario_result
        print(f"   ✅ Complete! Market share: {results[-1]['market_share']:.1f}%")
        
        return scenario_result
    
    def _calculate_roi(self, budget: Dict, market_share: float) -> float:
        """Calculate simple ROI"""
        total_budget = sum(budget.values())
        if total_budget == 0:
            return 0
        return market_share / total_budget * 100000  # Scaling factor
    
    def compare_scenarios(self) -> pd.DataFrame:
        """Compare all run scenarios"""
        if not self.scenario_results:
            return pd.DataFrame()
        
        comparison = []
        for name, result in self.scenario_results.items():
            comparison.append({
                'Scenario': name,
                'Total Budget': f"${result['total_budget']:,.0f}",
                'Final Market Share (%)': round(result['final_market_share'], 1),
                'Clustering': round(result['final_clustering'], 3),
                'ROI': round(result['roi'], 4)
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
        
        print(f"\n🔍 Optimizing budget (${total_budget:,.0f})...")
        
        for i in range(n_iterations):
            # Random allocation
            weights = np.random.dirichlet(np.ones(len(channels)))
            allocation = {
                channel: total_budget * weight
                for channel, weight in zip(channels, weights)
            }
            
            # Run scenario
            result = self.run_scenario(
                allocation,
                steps=20,
                scenario_name=f"Opt_{i}"
            )
            
            if result['roi'] > best_roi:
                best_roi = result['roi']
                best_allocation = allocation
        
        print(f"✅ Best ROI: {best_roi:.4f}")
        return best_allocation
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary"""
        return {
            'mmm_summary': self.mmm.get_summary(),
            'scenarios': len(self.scenario_results),
            'best_scenario': self._get_best_scenario(),
            'marketing_effects': self.marketing_effects
        }
    
    def _get_best_scenario(self) -> Optional[str]:
        """Find scenario with highest ROI"""
        if not self.scenario_results:
            return None
        best = max(self.scenario_results.items(), key=lambda x: x[1]['roi'])
        return best[0]


if __name__ == '__main__':
    from data.generate_data import generate_marketing_data
    
    # Config class
    class Config:
        NUM_CONSUMERS = 100
        AVG_CONNECTIONS = 6
        INIT_ORANGE_PCT = 50
        QUALITY_ORANGE = 0.45
        QUALITY_BLUE = 0.50
        NORM_INFLUENCE = 0.50
        INFO_EXCHANGE = 0.50
        EXPLORATION = 0.01
    
    # Generate data
    data = generate_marketing_data()
    
    # Create simulator
    config = Config()
    simulator = MarketingSimulator(config, data)
    
    # Run scenarios
    scenarios = [
        {'tv_spend': 200000, 'digital_spend': 150000, 'social_spend': 100000, 'influencer_spend': 50000},
        {'tv_spend': 100000, 'digital_spend': 200000, 'social_spend': 100000, 'influencer_spend': 100000},
        {'tv_spend': 150000, 'digital_spend': 150000, 'social_spend': 150000, 'influencer_spend': 50000},
    ]
    
    for i, budget in enumerate(scenarios):
        simulator.run_scenario(budget, steps=30, scenario_name=f"Scenario_{i+1}")
    
    # Compare results
    comparison = simulator.compare_scenarios()
    print("\n📊 Scenario Comparison:")
    print(comparison.to_string(index=False))
    
    # Optimize
    best = simulator.optimize_budget(500000, ['tv_spend', 'digital_spend', 'social_spend', 'influencer_spend'])
    print(f"\n🎯 Optimal allocation: {best}")
