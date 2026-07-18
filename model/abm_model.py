# model/abm_model.py
# Extended ABM with marketing integration

import random
import math
import numpy as np
from typing import List, Dict, Optional

class Consumer:
    """Consumer agent with marketing awareness"""
    
    def __init__(self, consumer_id: int, init_choice: int, qualities: List[float]):
        self.id = consumer_id
        self.choice = init_choice  # 0=orange, 1=blue
        
        # State variables
        self.satisfaction = [0.0, 0.0]
        self.personal_exp = [0.0, 0.0]
        self.utilities = [0.0, 0.0]
        
        # Position for network formation
        margin = 40
        self.x = random.uniform(margin, 850 - margin)
        self.y = random.uniform(margin, 450 - margin)
        
        # Social network
        self.neighbors = []
        
        # Marketing exposure
        self.marketing_exposure = {
            'tv': 0.0,
            'digital': 0.0,
            'social': 0.0,
            'influencer': 0.0
        }
        
        # Product qualities
        self.qualities = qualities
        
        # Initialize utilities
        if init_choice == 0:
            self.utilities = [1.0, 0.0]
        else:
            self.utilities = [0.0, 1.0]
        
        self.update_satisfaction()
    
    def update_satisfaction(self):
        """Update satisfaction based on current choice"""
        q = self.qualities[self.choice]
        self.satisfaction[self.choice] = math.log10(1 + 9 * q)
    
    def apply_marketing(self, channel_effects: Dict[str, float], budget_scaling: float = 1.0):
        """
        Apply marketing effects to consumer perception
        
        Args:
            channel_effects: Dict with channel names and effect sizes
            budget_scaling: Scaling factor for budget (0-2)
        """
        for channel, effect in channel_effects.items():
            if channel in self.marketing_exposure:
                # Marketing exposure increases based on effect and budget
                self.marketing_exposure[channel] += effect * budget_scaling * 0.1
    
    def decide(self, explore_rate: float, norm_influence: float = 0.5) -> int:
        """
        Make consumption decision with marketing influence
        """
        # Satisfaction-based sticking
        if random.random() < self.satisfaction[self.choice]:
            return self.choice
        
        # Exploration
        if random.random() < explore_rate:
            return 1 - self.choice
        
        # Marketing-influenced utility comparison
        # Marketing boosts perceived quality
        marketing_boost = sum(
            self.marketing_exposure.get(channel, 0) * effect
            for channel, effect in {
                'tv': 0.3, 'digital': 0.4, 'social': 0.2, 'influencer': 0.1
            }.items()
        )
        
        # Enhanced utilities with marketing boost
        enhanced_utilities = [
            self.utilities[0] * (1 + marketing_boost * 0.2),
            self.utilities[1] * (1 + marketing_boost * 0.2)
        ]
        
        if enhanced_utilities[0] > enhanced_utilities[1]:
            return 0
        elif enhanced_utilities[1] > enhanced_utilities[0]:
            return 1
        else:
            return self.choice
    
    def get_position(self):
        return (self.x, self.y)


class ABM_Model_Marketing:
    """Extended ABM with marketing capabilities"""
    
    def __init__(self, config, marketing_effects: Optional[Dict] = None):
        """
        Initialize ABM with marketing integration
        
        Args:
            config: Configuration object with model parameters
            marketing_effects: Dict of channel effect sizes from MMM
        """
        self.config = config
        self.marketing_effects = marketing_effects or {
            'tv': 0.3,
            'digital': 0.4,
            'social': 0.25,
            'influencer': 0.2
        }
        
        self.consumers: List[Consumer] = []
        self.qualities = [config.QUALITY_ORANGE, config.QUALITY_BLUE]
        self.step_count = 0
        self.history = {
            'market_share': [],
            'clustering': []
        }
        
        # Marketing tracking
        self.budget_allocation = {}
        self.marketing_exposure_history = []
        
    def setup(self):
        """Initialize consumers and network"""
        self._create_consumers()
        self._build_network()
        self.step_count = 0
        self.history = {'market_share': [], 'clustering': []}
        self._record_stats()
        
    def _create_consumers(self):
        """Create consumer agents"""
        self.consumers = []
        n = self.config.NUM_CONSUMERS
        init_orange_pct = self.config.INIT_ORANGE_PCT
        
        for i in range(n):
            choice = 0 if random.random() < init_orange_pct / 100 else 1
            consumer = Consumer(i, choice, self.qualities)
            self.consumers.append(consumer)
    
    def _build_network(self):
        """Build geometric network"""
        target_edges = int(self.config.AVG_CONNECTIONS * self.config.NUM_CONSUMERS / 2)
        edges = set()
        
        # Simple nearest-neighbor connection
        for i in range(len(self.consumers)):
            distances = []
            for j in range(len(self.consumers)):
                if i != j:
                    dist = self._distance(self.consumers[i], self.consumers[j])
                    distances.append((dist, j))
            distances.sort(key=lambda x: x[0])
            
            # Connect to nearest neighbors
            for _, j in distances[:int(self.config.AVG_CONNECTIONS)]:
                if len(edges) >= target_edges:
                    break
                edge = tuple(sorted([i, j]))
                if edge not in edges:
                    edges.add(edge)
                    self.consumers[i].neighbors.append(self.consumers[j])
                    self.consumers[j].neighbors.append(self.consumers[i])
    
    def _distance(self, c1, c2):
        return math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
    
    def apply_marketing_campaign(self, budget_allocation: Dict[str, float]):
        """
        Apply marketing campaign to all consumers
        
        Args:
            budget_allocation: Dict mapping channel to spend amount
        """
        self.budget_allocation = budget_allocation
        total_budget = sum(budget_allocation.values())
        
        if total_budget == 0:
            return
        
        # Normalize budget to get allocation percentages
        budget_pcts = {k: v/total_budget for k, v in budget_allocation.items()}
        
        # Apply marketing to each consumer
        for consumer in self.consumers:
            # Reset exposure
            consumer.marketing_exposure = {k: 0.0 for k in consumer.marketing_exposure}
            
            # Apply each channel based on budget
            for channel, pct in budget_pcts.items():
                if channel in self.marketing_effects:
                    # Random reach: not all consumers see all channels
                    if random.random() < 0.5:  # 50% reach probability
                        effect = self.marketing_effects[channel] * pct * 2
                        consumer.marketing_exposure[channel] = effect
    
    def step(self):
        """Execute one simulation step with marketing influence"""
        # Consumers make decisions
        for consumer in self.consumers:
            new_choice = consumer.decide(
                self.config.EXPLORATION,
                self.config.NORM_INFLUENCE
            )
            if new_choice != consumer.choice:
                consumer.choice = new_choice
                consumer.update_satisfaction()
        
        # Update utilities with marketing effects
        self._update_utilities_with_marketing()
        
        self.step_count += 1
        self._record_stats()
    
    def _update_utilities_with_marketing(self):
        """Update utilities including marketing effects"""
        for consumer in self.consumers:
            for option in [0, 1]:
                # Personal experience
                if consumer.choice == option:
                    consumer.personal_exp[option] = self.qualities[option]
                
                # Neighbor information
                neighbor_info = 0.0
                if random.random() < self.config.INFO_EXCHANGE:
                    if any(n.choice == option for n in consumer.neighbors):
                        neighbor_info = self.qualities[option]
                
                # Marketing boost
                marketing_boost = sum(
                    consumer.marketing_exposure.get(channel, 0) * effect
                    for channel, effect in self.marketing_effects.items()
                )
                
                # Combined utility
                self_info = consumer.personal_exp[option]
                prod_util = (self_info + neighbor_info) / 2.0
                
                # Enhanced utility with marketing
                consumer.utilities[option] = (
                    (1 - self.config.NORM_INFLUENCE) * prod_util +
                    self.config.NORM_INFLUENCE * 0.5 +  # Simplified norm
                    marketing_boost * 0.1  # Marketing influence
                )
    
    def _record_stats(self):
        """Record statistics"""
        orange_count = sum(1 for c in self.consumers if c.choice == 0)
        market_share = orange_count / self.config.NUM_CONSUMERS * 100
        self.history['market_share'].append(market_share)
        
        # Clustering
        cluster_sum = 0.0
        for c in self.consumers:
            if c.neighbors:
                same = sum(1 for n in c.neighbors if n.choice == c.choice)
                cluster_sum += same / len(c.neighbors)
        clustering = cluster_sum / self.config.NUM_CONSUMERS
        self.history['clustering'].append(clustering)
    
    def get_market_share(self) -> float:
        if not self.history['market_share']:
            return 0
        return self.history['market_share'][-1]
    
    def get_summary(self) -> Dict:
        return {
            'step': self.step_count,
            'market_share': self.get_market_share(),
            'clustering': self.history['clustering'][-1] if self.history['clustering'] else 0
        }

if __name__ == '__main__':
    # Simple test
    class Config:
        NUM_CONSUMERS = 100
        AVG_CONNECTIONS = 6
        INIT_ORANGE_PCT = 50
        QUALITY_ORANGE = 0.45
        QUALITY_BLUE = 0.50
        NORM_INFLUENCE = 0.5
        INFO_EXCHANGE = 0.5
        EXPLORATION = 0.01
    
    config = Config()
    marketing_effects = {'tv': 0.3, 'digital': 0.4, 'social': 0.25, 'influencer': 0.2}
    
    abm = ABM_Model_Marketing(config, marketing_effects)
    abm.setup()
    
    print(f"Initial market share: {abm.get_market_share():.1f}%")
    
    # Apply marketing
    budget = {'tv': 100, 'digital': 150, 'social': 75, 'influencer': 50}
    abm.apply_marketing_campaign(budget)
    
    # Run steps
    for _ in range(10):
        abm.step()
    
    print(f"Final market share: {abm.get_market_share():.1f}%")