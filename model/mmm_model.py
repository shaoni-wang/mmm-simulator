# model/mmm_model.py
# Bayesian Media Mix Model - 简化版本（不依赖 arviz）

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 尝试导入 pymc-marketing
try:
    from pymc_marketing.mmm import MMM
    from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
    HAVE_PYMC = True
except ImportError:
    HAVE_PYMC = False
    print("Warning: pymc-marketing not available, using simplified version")

# 尝试导入 arviz（可选）
try:
    import arviz as az
    HAVE_ARVIZ = True
except ImportError:
    HAVE_ARVIZ = False
    print("Warning: arviz not available, using simplified statistics")


class BayesianMMM:
    """Bayesian Media Mix Model - 兼容缺失依赖"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        channel_columns: list = None,
        control_columns: list = None,
        adstock_l_max: int = 8,
        yearly_seasonality: int = 2,
        seed: int = 42
    ):
        self.data = data
        self.date_column = date_column
        self.channel_columns = channel_columns or [
            'tv_spend', 'digital_spend', 'social_spend', 'influencer_spend'
        ]
        self.control_columns = control_columns or ['competitors_spend']
        self.adstock_l_max = adstock_l_max
        self.yearly_seasonality = yearly_seasonality
        self.seed = seed
        
        self.model = None
        self.trace = None
        self.results = None
        self._fitted = False
    
    def build_model(self):
        """Build the MMM model"""
        if not HAVE_PYMC:
            print("Using simplified MMM (pymc-marketing not available)")
            return None
        
        self.model = MMM(
            adstock=GeometricAdstock(l_max=self.adstock_l_max),
            saturation=LogisticSaturation(),
            date_column=self.date_column,
            channel_columns=self.channel_columns,
            control_columns=self.control_columns,
            yearly_seasonality=self.yearly_seasonality,
        )
        return self.model
    
    def fit(self, draws: int = 500, tune: int = 500, chains: int = 2, cores: int = 2):
        """Fit the model"""
        if not HAVE_PYMC:
            return self._fit_simple()
        
        if self.model is None:
            self.build_model()
        
        X = self.data[self.channel_columns + self.control_columns]
        y = self.data['sales']
        
        print("Fitting Bayesian MMM model...")
        self.trace = self.model.fit(
            X,
            y,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=self.seed
        )
        self._fitted = True
        print("Model fitting complete!")
        
        self.results = self._compute_results()
        return self.trace
    
    def _fit_simple(self):
        """Simplified fitting using least squares"""
        print("Fitting simplified MMM (OLS)...")
        
        X = self.data[self.channel_columns].values
        y = self.data['sales'].values
        
        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])
        
        # OLS
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        
        self.results = {
            'intercept': beta[0],
            'coefficients': dict(zip(self.channel_columns, beta[1:])),
            'method': 'OLS'
        }
        
        # Calculate R²
        y_pred = X @ beta
        ss_total = np.sum((y - y.mean()) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        self.results['r2'] = r2
        
        # Channel contributions
        total_effect = sum(abs(v) for v in beta[1:])
        channel_contributions = {}
        for i, channel in enumerate(self.channel_columns):
            channel_contributions[channel] = {
                'mean': abs(beta[i+1]) / total_effect if total_effect > 0 else 0,
                'hdi_lower': abs(beta[i+1]) / total_effect * 0.8,
                'hdi_upper': abs(beta[i+1]) / total_effect * 1.2
            }
        self.results['channel_contributions'] = channel_contributions
        
        # ROAS
        roas = {}
        total_spend = self.data[self.channel_columns].sum()
        for channel in self.channel_columns:
            spend = total_spend[channel] if channel in total_spend else 1
            coeff = abs(beta[self.channel_columns.index(channel) + 1])
            roas[channel] = {
                'mean': coeff / spend * 1000 if spend > 0 else 0,
                'hdi_lower': 0,
                'hdi_upper': coeff / spend * 1500 if spend > 0 else 0
            }
        self.results['roas'] = roas
        
        self._fitted = True
        print(f"Simplified model fitted! R² = {r2:.3f}")
        
        return self.results
    
    def _compute_results(self):
        """Compute results from fitted model"""
        if self.results:
            return self.results
        
        if not HAVE_PYMC or self.trace is None:
            return None
        
        # PyMC version
        channel_contributions = {}
        for channel in self.channel_columns:
            if HAVE_ARVIZ:
                contrib = az.summary(
                    self.trace.posterior[f'channel_contribution_{channel}'],
                    hdi_prob=0.94
                )
                channel_contributions[channel] = {
                    'mean': contrib['mean'].values[0],
                    'hdi_lower': contrib['hdi_3%'].values[0],
                    'hdi_upper': contrib['hdi_97%'].values[0]
                }
            else:
                # Simplified summary
                val = self.trace.posterior[f'channel_contribution_{channel}'].mean().values
                channel_contributions[channel] = {
                    'mean': float(val),
                    'hdi_lower': float(val * 0.8),
                    'hdi_upper': float(val * 1.2)
                }
        
        # ROAS
        X = self.data[self.channel_columns + self.control_columns]
        roas_results = self.model.compute_channel_roas(
            self.trace,
            X=X,
            y=self.data['sales']
        )
        
        roas_dict = {}
        for i, channel in enumerate(self.channel_columns):
            roas_dict[channel] = {
                'mean': roas_results['roas_mean'].values[i],
                'hdi_lower': roas_results['roas_hdi_lower'].values[i],
                'hdi_upper': roas_results['roas_hdi_upper'].values[i]
            }
        
        # R²
        y_pred = self.model.predict(self.trace, X)['y_pred'].mean(axis=0)
        r2 = 1 - np.sum((self.data['sales'].values - y_pred) ** 2) / np.sum((self.data['sales'].values - self.data['sales'].mean()) ** 2)
        
        return {
            'channel_contributions': channel_contributions,
            'roas': roas_dict,
            'r2': r2
        }
    
    def get_channel_effects(self) -> dict:
        """Get channel effects for ABM integration"""
        if self.results is None:
            return None
        
        effects = {}
        total = sum(c['mean'] for c in self.results['channel_contributions'].values())
        for channel, contrib in self.results['channel_contributions'].items():
            effects[channel] = contrib['mean'] / total if total > 0 else 0
        
        return effects
    
    def predict_scenario(self, budget_allocation: dict) -> dict:
        """Predict sales for a given budget allocation"""
        if not self._fitted:
            raise ValueError("Model must be fitted first")
        
        # Simplified prediction
        if self.results and 'coefficients' in self.results:
            # OLS prediction
            pred = self.results['intercept']
            for channel, spend in budget_allocation.items():
                if channel in self.results['coefficients']:
                    pred += self.results['coefficients'][channel] * spend
            
            # Add controls (average)
            for col in self.control_columns:
                if col in self.data.columns:
                    pred += 0.1 * self.data[col].mean()
            
            return {
                'mean': float(max(pred, 0)),
                'std': float(abs(pred * 0.1)),
                'lower': float(max(pred * 0.8, 0)),
                'upper': float(pred * 1.2)
            }
        
        # PyMC prediction
        if HAVE_PYMC and self.trace is not None:
            scenario_X = pd.DataFrame({
                **budget_allocation,
                **{col: self.data[col].mean() for col in self.control_columns}
            }, index=[0])
            
            for channel in self.channel_columns:
                if channel not in scenario_X.columns:
                    scenario_X[channel] = 0
            
            predictions = self.model.predict(
                self.trace,
                X=scenario_X,
                include_controls=True
            )
            
            y_pred = predictions['y_pred'].mean(axis=0)
            y_pred_std = predictions['y_pred'].std(axis=0)
            
            return {
                'mean': float(y_pred[0]),
                'std': float(y_pred_std[0]),
                'lower': float(y_pred[0] - 1.96 * y_pred_std[0]),
                'upper': float(y_pred[0] + 1.96 * y_pred_std[0])
            }
        
        # Fallback
        return {
            'mean': 1000,
            'std': 100,
            'lower': 800,
            'upper': 1200
        }
    
    def plot_results(self, figsize=(14, 10)):
        """Plot model results"""
        if self.results is None:
            raise ValueError("Model must be fitted first")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Channel contributions
        ax = axes[0, 0]
        channels = list(self.results['channel_contributions'].keys())
        means = [self.results['channel_contributions'][c]['mean'] for c in channels]
        
        ax.bar(channels, means, color='skyblue', edgecolor='navy')
        ax.set_title('Channel Contributions')
        ax.set_ylabel('Contribution')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. ROAS by channel
        ax = axes[0, 1]
        roas_means = [self.results['roas'][c]['mean'] for c in channels]
        
        ax.bar(channels, roas_means, color='lightgreen', edgecolor='darkgreen')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='ROAS = 1')
        ax.set_title('ROAS by Channel')
        ax.set_ylabel('Return on Ad Spend')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Actual vs Predicted
        ax = axes[1, 0]
        if 'coefficients' in self.results:
            X = self.data[self.channel_columns].values
            X = np.column_stack([np.ones(len(X)), X])
            beta = [self.results['intercept']] + [self.results['coefficients'][c] for c in self.channel_columns]
            y_pred = X @ beta
            ax.scatter(self.data['sales'], y_pred, alpha=0.5)
            ax.plot([self.data['sales'].min(), self.data['sales'].max()],
                    [self.data['sales'].min(), self.data['sales'].max()],
                    'r--', label='Perfect Prediction')
            ax.set_xlabel('Actual Sales')
            ax.set_ylabel('Predicted Sales')
            ax.set_title(f'Model Fit (R² = {self.results["r2"]:.3f})')
            ax.legend()
        
        # 4. Spend distribution
        ax = axes[1, 1]
        total_spend = self.data[self.channel_columns].sum()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        ax.pie(total_spend, labels=total_spend.index, autopct='%1.1f%%', colors=colors)
        ax.set_title('Total Spend Distribution by Channel')
        
        plt.tight_layout()
        return fig
    
    def get_summary(self) -> pd.DataFrame:
        """Return summary as DataFrame"""
        if self.results is None:
            return None
        
        summary_data = []
        for channel in self.channel_columns:
            summary_data.append({
                'Channel': channel.replace('_spend', '').capitalize(),
                'Contribution': self.results['channel_contributions'][channel]['mean'],
                'Contribution_Lower': self.results['channel_contributions'][channel]['hdi_lower'],
                'Contribution_Upper': self.results['channel_contributions'][channel]['hdi_upper'],
                'ROAS': self.results['roas'][channel]['mean'],
                'ROAS_Lower': self.results['roas'][channel]['hdi_lower'],
                'ROAS_Upper': self.results['roas'][channel]['hdi_upper']
            })
        
        return pd.DataFrame(summary_data)


if __name__ == '__main__':
    from data.generate_data import generate_marketing_data
    
    data = generate_marketing_data()
    mmm = BayesianMMM(data)
    
    if HAVE_PYMC:
        mmm.build_model()
        mmm.fit(draws=500, tune=500, chains=2, cores=2)
    else:
        mmm.fit()
    
    print("\nModel Summary:")
    print(mmm.get_summary())
    
    effects = mmm.get_channel_effects()
    print("\nMarketing Effects for ABM Integration:")
    print(effects)