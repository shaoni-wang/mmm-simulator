# model/mmm_model.py
# Simplified MMM using scipy (no arviz/pymc dependency)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class BayesianMMM:
    """
    Simplified MMM using scikit-learn instead of Bayesian methods
    This avoids arviz/pymc dependency issues
    """
    
    def __init__(self, data, channel_columns=None, control_columns=None, seed=42):
        self.data = data
        self.channel_columns = channel_columns or ['tv_spend', 'digital_spend', 
                                                    'social_spend', 'influencer_spend']
        self.control_columns = control_columns or ['competitors_spend']
        self.seed = seed
        self.model = None
        self.results = None
        
    def fit(self):
        """Fit model with adstock transformation"""
        X = self.data[self.channel_columns].values
        y = self.data['sales'].values
        
        # Adstock transformation (carryover effect)
        def adstock_transform(x, alpha=0.5):
            result = np.zeros_like(x)
            for i in range(1, len(x)):
                result[i] = x[i] + alpha * result[i-1]
            return result
        
        # Apply adstock to each channel
        X_adstock = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_adstock[:, i] = adstock_transform(X[:, i])
        
        # Fit linear model
        self.model = LinearRegression()
        self.model.fit(X_adstock, y)
        
        # Calculate contributions
        total_effect = np.sum(self.model.coef_ * X_adstock.mean(axis=0))
        contributions = {}
        for i, channel in enumerate(self.channel_columns):
            contrib = self.model.coef_[i] * X_adstock[:, i].mean()
            contributions[channel] = float(contrib)
        
        # Normalize contributions to 0-1 for ABM
        total_contrib = sum(contributions.values()) or 1
        normalized = {k: v/total_contrib for k, v in contributions.items()}
        
        self.results = {
            'coefficients': self.model.coef_,
            'intercept': float(self.model.intercept_),
            'r2': float(self.model.score(X_adstock, y)),
            'contributions': normalized,  # Normalized for ABM
            'raw_contributions': contributions,  # Original values
            'adstock_data': X_adstock
        }
        
        return self.results
    
    def get_channel_effects(self):
        """Get channel effects for ABM integration"""
        if self.results is None:
            return None
        return self.results['contributions']
    
    def predict_scenario(self, budget_allocation):
        """Predict sales for budget scenario"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        total_effect = self.model.intercept_
        for channel, spend in budget_allocation.items():
            if channel in self.channel_columns:
                idx = self.channel_columns.index(channel)
                total_effect += self.model.coef_[idx] * spend
        
        # Rough uncertainty estimate (20% standard deviation)
        std = abs(total_effect) * 0.2
        
        return {
            'mean': float(total_effect),
            'std': float(std),
            'lower': float(total_effect - 1.96 * std),
            'upper': float(total_effect + 1.96 * std)
        }
    
    def get_summary(self):
        """Get summary dataframe"""
        if self.results is None:
            return None
        
        summary = []
        for i, channel in enumerate(self.channel_columns):
            summary.append({
                'Channel': channel.capitalize(),
                'Coefficient': round(self.model.coef_[i], 3),
                'Contribution': round(self.results['raw_contributions'][channel], 0),
                'Effect_Pct': round(self.results['contributions'][channel] * 100, 1)
            })
        
        summary.append({
            'Channel': 'Intercept',
            'Coefficient': round(self.model.intercept_, 3),
            'Contribution': '-',
            'Effect_Pct': '-'
        })
        
        return pd.DataFrame(summary)
    
    def plot_results(self, figsize=(14, 6)):
        """Plot model results"""
        if self.results is None:
            raise ValueError("Model must be fitted first")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Channel contributions
        ax = axes[0]
        channels = list(self.results['contributions'].keys())
        values = [self.results['contributions'][c] for c in channels]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(channels)]
        
        bars = ax.bar([c.capitalize() for c in channels], values, color=colors)
        ax.set_title('Channel Effects (Normalized)')
        ax.set_ylabel('Relative Effect')
        ax.set_ylim(0, max(values) * 1.2)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=10)
        
        # 2. Actual vs Predicted
        ax = axes[1]
        X = self.results['adstock_data']
        y_pred = self.model.predict(X)
        
        ax.scatter(self.data['sales'], y_pred, alpha=0.5)
        ax.plot([self.data['sales'].min(), self.data['sales'].max()],
                [self.data['sales'].min(), self.data['sales'].max()],
                'r--', label='Perfect Prediction')
        ax.set_xlabel('Actual Sales')
        ax.set_ylabel('Predicted Sales')
        ax.set_title(f'Model Fit (R² = {self.results["r2"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


if __name__ == '__main__':
    from data.generate_data import generate_marketing_data
    
    # Generate data
    data = generate_marketing_data()
    print(f"Generated {len(data)} weeks of data")
    
    # Create and fit model
    mmm = BayesianMMM(data)
    mmm.fit()
    
    # Get summary
    summary = mmm.get_summary()
    print("\nModel Summary:")
    print(summary)
    
    # Get effects for ABM
    effects = mmm.get_channel_effects()
    print("\nChannel Effects for ABM:")
    for channel, effect in effects.items():
        print(f"  {channel}: {effect:.2%}")
    
    # Test scenario
    scenario = {
        'tv_spend': 200000,
        'digital_spend': 150000,
        'social_spend': 100000,
        'influencer_spend': 50000
    }
    prediction = mmm.predict_scenario(scenario)
    print(f"\nScenario Prediction: {prediction['mean']:.0f} ± {prediction['std']:.0f}")
    
    # Plot results
    fig = mmm.plot_results()
    plt.savefig('mmm_results.png', dpi=150)
    print("\n✅ Plot saved as 'mmm_results.png'")
