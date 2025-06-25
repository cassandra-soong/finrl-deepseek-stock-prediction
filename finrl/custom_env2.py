from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import pandas as pd


class RiskAwareStockTradingEnv1(StockTradingEnv):
    """
    This is a custom class built for Agent 2's Trading Environment to take the news sentiment risk scores into consideration during its trading steps.
    This class inherits the StockTradingEnv class from FinRL, and only overwrittes the step() function to give 'reward' or 'penalty' to the agent depending on the risk scores. 
    The amount of 'reward' or 'penalty' is indicated by a given weight value. 
    For example, a lower risk score signals optimistic chance in its trade, thus the agent's state[0] is multiplied with a high weight ('reward').
    
    Args:
        StockTradingEnv (class): provided by FinRL 
        
    """

    def __init__(self, df, **kwargs):
        self.risk_score_col = 'risk_score'
        self.turbulence_col = 'turbulence_bin'
        super().__init__(df, **kwargs)

    def step(self, actions):
        state, reward, done, truncated, info = super().step(actions)

        # Get the risk score
        day_idx = max(0, self.day - 1)
        risk_score = self.df.loc[day_idx, self.risk_score_col] if self.risk_score_col in self.df.columns else 0
        risk_score = risk_score if pd.notnull(risk_score) else 0
        
        # Inject risk into state indirectly 
        risk_weight = self._get_risk_scaling_factor(risk_score)

        # State[0] is a normalized or scaled value, part of the observation used by the agent.
        state[0] = state[0] * risk_weight

        # Get the turbulence bin
        turbulence_bin = self.df.loc[day_idx, self.turbulence_col] if self.turbulence_col in self.df.columns else "low"
        turbulence_bin = turbulence_bin if pd.notnull(turbulence_bin) else "low"

        # Inject turbulence into state indirectly
        turbulence_weight = self._get_turbulence_scaling_factor(turbulence_bin)


        # Adjust the state based on turbulence
        state[0] = state[0] * turbulence_weight


        return state, reward, done, truncated, info

    def _get_risk_scaling_factor(self, risk_score):
        if risk_score == 0:
            return 1  # No change
        elif 1 <= risk_score <= 2:
            return 15  # More optimistic
        elif risk_score == 3:
            return 10  # Slight optimistic
        elif 4 <= risk_score <= 5:
            return -5  # Slight caution
        else:
            return 1

    def _get_turbulence_scaling_factor(self, turbulence_bin):
        if turbulence_bin == "low":
            return 1
        elif turbulence_bin == "medium":
            return 1.10
        elif turbulence_bin == "high":
            return 1.05
        else:
            return 1