"""Strategy runner — runs all strategies and produces feature vectors.

Precomputes strategy signals for all bars in a DataFrame, adding 14
feature columns (presence + confidence per strategy, plus aggregates).
This runs ONCE before GA evolution — zero overhead per chromosome.
"""

import pandas as pd
import structlog

from aether_btc.signals.base import SignalStrategy, SignalType, StrategySignal

log = structlog.get_logger()

# Strategy column names for presence (1=BUY, -1=SHORT, 0=HOLD)
# and confidence [0, 1].
STRATEGY_COLUMNS = [
    "sig_momentum", "sig_momentum_conf",
    "sig_mean_reversion", "sig_mean_reversion_conf",
    "sig_trend_following", "sig_trend_following_conf",
    "sig_volatility_breakout", "sig_volatility_breakout_conf",
    "sig_funding_volume", "sig_funding_volume_conf",
]

AGGREGATE_COLUMNS = [
    "n_buy_signals",
    "n_sell_signals",
    "signal_consensus",
    "max_confidence",
]

ALL_STRATEGY_FEATURE_COLUMNS = STRATEGY_COLUMNS + AGGREGATE_COLUMNS


class StrategyRunner:
    """Runs all registered strategies and precomputes feature vectors."""

    def __init__(self, strategies: list[SignalStrategy]) -> None:
        self.strategies = strategies

    def precompute_all_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all strategies on every bar, adding 14 feature columns.

        This is called ONCE before GA evolution. The resulting columns
        become features in the sigmoid scoring function.

        Returns a new DataFrame with strategy columns added.
        """
        df = df.copy()
        n_bars = len(df)

        # Initialize per-strategy columns
        for strategy in self.strategies:
            sig_col = f"sig_{strategy.name}"
            conf_col = f"sig_{strategy.name}_conf"
            df[sig_col] = 0.0
            df[conf_col] = 0.0

        # Initialize aggregate columns
        for col in AGGREGATE_COLUMNS:
            df[col] = 0.0

        # Run each strategy on every bar
        for strategy in self.strategies:
            sig_col = f"sig_{strategy.name}"
            conf_col = f"sig_{strategy.name}_conf"

            sig_values = []
            conf_values = []

            for bar_idx in range(n_bars):
                signal = strategy.generate_signal(df, bar_idx)
                if signal.signal_type == SignalType.BUY:
                    sig_values.append(1.0)
                    conf_values.append(signal.confidence)
                elif signal.signal_type == SignalType.SHORT:
                    sig_values.append(-1.0)
                    conf_values.append(signal.confidence)
                else:
                    sig_values.append(0.0)
                    conf_values.append(0.0)

            df[sig_col] = sig_values
            df[conf_col] = conf_values

            log.debug("strategy_precomputed", strategy=strategy.name, bars=n_bars)

        # Compute aggregates
        sig_cols = [f"sig_{s.name}" for s in self.strategies]
        conf_cols = [f"sig_{s.name}_conf" for s in self.strategies]

        sig_matrix = df[sig_cols]
        conf_matrix = df[conf_cols]

        df["n_buy_signals"] = (sig_matrix > 0).sum(axis=1).astype(float)
        df["n_sell_signals"] = (sig_matrix < 0).sum(axis=1).astype(float)
        df["signal_consensus"] = sig_matrix.sum(axis=1)
        df["max_confidence"] = conf_matrix.max(axis=1)

        log.info(
            "strategies_precomputed",
            strategies=len(self.strategies),
            bars=n_bars,
            columns_added=len(ALL_STRATEGY_FEATURE_COLUMNS),
        )

        return df
