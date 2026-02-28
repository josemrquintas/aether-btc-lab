"""Chromosome representation for GA optimization.

Adapted from Aether v3 with crypto-specific genes:
- Leverage genes (base, confidence scale, volatility dampen, max cap)
- Funding rate awareness (threshold)
- Liquidation buffer
- Feature weight genes for 15-min crypto indicators
"""

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GeneSpec:
    """Specification for a single gene."""

    name: str
    min_val: float
    max_val: float
    default: float
    dtype: str = "float"
    description: str = ""

    def random_value(self) -> float | int:
        if self.dtype == "int":
            return random.randint(int(self.min_val), int(self.max_val))
        return random.uniform(self.min_val, self.max_val)


GENE_SPECS = [
    # === LEVERAGE GENES (crypto-specific) ===
    GeneSpec("leverage_base", 1.0, 10.0, 3.0, "float", "Base leverage for positions"),
    GeneSpec("leverage_confidence_scale", 0.0, 2.0, 1.0, "float", "Scale leverage by signal confidence"),
    GeneSpec("leverage_volatility_dampen", 0.0, 1.0, 0.5, "float", "Reduce leverage in high volatility"),
    GeneSpec("max_leverage_cap", 5.0, 20.0, 10.0, "float", "Hard cap on leverage"),

    # === CRYPTO ALLOCATION GENES ===
    GeneSpec("funding_rate_threshold", -0.001, 0.001, 0.0, "float", "Skip trade if funding rate too adverse"),
    GeneSpec("liquidation_buffer_pct", 0.05, 0.30, 0.15, "float", "Min distance to liquidation price"),
    GeneSpec("min_confidence_cutoff", 0.30, 0.80, 0.45, "float", "Below this confidence, skip signal"),
    GeneSpec("position_size_multiplier", 0.5, 3.0, 1.0, "float", "Position size scaling"),
    GeneSpec("min_hold_bars", 1, 32, 4, "int", "Min bars to hold before exit"),

    # === STOP / TARGET GENES ===
    GeneSpec("atr_stop_multiplier", 1.0, 5.0, 2.0, "float", "ATR multiplier for stop-loss"),
    GeneSpec("risk_reward_ratio", 1.0, 4.0, 2.0, "float", "Risk:reward ratio for take-profit"),

    # === FEATURE WEIGHT GENES (signal scoring) ===
    # Returns (5)
    GeneSpec("fw_return_1bar", -1.5, 1.5, 0.0, "float", "Weight: 1-bar return"),
    GeneSpec("fw_return_4bar", -1.5, 1.5, 0.0, "float", "Weight: 4-bar (1h) return"),
    GeneSpec("fw_return_16bar", -1.5, 1.5, 0.0, "float", "Weight: 16-bar (4h) return"),
    GeneSpec("fw_return_96bar", -1.5, 1.5, 0.0, "float", "Weight: 96-bar (1d) return"),
    GeneSpec("fw_momentum_acceleration", -1.5, 1.5, 0.0, "float", "Weight: momentum acceleration"),
    # Position (4)
    GeneSpec("fw_bb_position", -1.5, 1.5, 0.0, "float", "Weight: Bollinger Band position"),
    GeneSpec("fw_price_vs_sma50", -1.5, 1.5, 0.0, "float", "Weight: price vs SMA50"),
    GeneSpec("fw_price_vs_sma200", -1.5, 1.5, 0.0, "float", "Weight: price vs SMA200"),
    GeneSpec("fw_zscore_20d", -1.5, 1.5, 0.0, "float", "Weight: 20-period z-score"),
    # Volatility (4)
    GeneSpec("fw_atr_normalized", -1.5, 1.5, 0.0, "float", "Weight: ATR normalized"),
    GeneSpec("fw_volatility_ratio", -1.5, 1.5, 0.0, "float", "Weight: volatility ratio"),
    GeneSpec("fw_bb_width", -1.5, 1.5, 0.0, "float", "Weight: Bollinger Band width"),
    GeneSpec("fw_intraday_range", -1.5, 1.5, 0.0, "float", "Weight: intraday range"),
    # Volume (4)
    GeneSpec("fw_volume_ratio", -1.5, 1.5, 0.0, "float", "Weight: volume ratio"),
    GeneSpec("fw_cmf", -1.5, 1.5, 0.0, "float", "Weight: Chaikin Money Flow"),
    GeneSpec("fw_mfi", -1.5, 1.5, 0.0, "float", "Weight: Money Flow Index"),
    GeneSpec("fw_obv_slope", -1.5, 1.5, 0.0, "float", "Weight: OBV slope"),
    # Trend (7)
    GeneSpec("fw_rsi", -1.5, 1.5, 0.0, "float", "Weight: RSI"),
    GeneSpec("fw_macd_histogram", -1.5, 1.5, 0.0, "float", "Weight: MACD histogram"),
    GeneSpec("fw_adx", -1.5, 1.5, 0.0, "float", "Weight: ADX"),
    GeneSpec("fw_roc", -1.5, 1.5, 0.0, "float", "Weight: Rate of Change"),
    GeneSpec("fw_stoch_k", -1.5, 1.5, 0.0, "float", "Weight: Stochastic K"),
    GeneSpec("fw_supertrend_dir", -1.5, 1.5, 0.0, "float", "Weight: Supertrend direction"),
    GeneSpec("fw_cci", -1.5, 1.5, 0.0, "float", "Weight: CCI"),
    # Crypto-specific features (2)
    GeneSpec("fw_funding_rate", -1.5, 1.5, 0.0, "float", "Weight: funding rate"),
    GeneSpec("fw_quote_volume_ratio", -1.5, 1.5, 0.0, "float", "Weight: quote volume ratio"),
    # === STRATEGY SIGNAL GENES (14 new) ===
    # Per-strategy weights (10): presence + confidence for each of 5 strategies
    GeneSpec("fw_sig_momentum", -1.5, 1.5, 0.0, "float", "Weight: momentum signal"),
    GeneSpec("fw_sig_momentum_conf", -1.5, 1.5, 0.0, "float", "Weight: momentum confidence"),
    GeneSpec("fw_sig_mean_reversion", -1.5, 1.5, 0.0, "float", "Weight: mean reversion signal"),
    GeneSpec("fw_sig_mean_reversion_conf", -1.5, 1.5, 0.0, "float", "Weight: mean reversion confidence"),
    GeneSpec("fw_sig_trend_following", -1.5, 1.5, 0.0, "float", "Weight: trend following signal"),
    GeneSpec("fw_sig_trend_following_conf", -1.5, 1.5, 0.0, "float", "Weight: trend following confidence"),
    GeneSpec("fw_sig_volatility_breakout", -1.5, 1.5, 0.0, "float", "Weight: volatility breakout signal"),
    GeneSpec("fw_sig_volatility_breakout_conf", -1.5, 1.5, 0.0, "float", "Weight: volatility breakout confidence"),
    GeneSpec("fw_sig_funding_volume", -1.5, 1.5, 0.0, "float", "Weight: funding/volume signal"),
    GeneSpec("fw_sig_funding_volume_conf", -1.5, 1.5, 0.0, "float", "Weight: funding/volume confidence"),
    # Aggregate strategy features (4)
    GeneSpec("fw_n_buy_signals", -1.5, 1.5, 0.0, "float", "Weight: count of buy signals"),
    GeneSpec("fw_n_sell_signals", -1.5, 1.5, 0.0, "float", "Weight: count of sell signals"),
    GeneSpec("fw_signal_consensus", -1.5, 1.5, 0.0, "float", "Weight: net signal consensus"),
    GeneSpec("fw_max_confidence", -1.5, 1.5, 0.0, "float", "Weight: max strategy confidence"),
    # Bias term
    GeneSpec("fw_bias", -5.0, 5.0, 0.0, "float", "Scoring bias term"),
]

FEATURE_WEIGHT_NAMES = [
    spec.name for spec in GENE_SPECS
    if spec.name.startswith("fw_") and spec.name != "fw_bias"
]

# Map gene names to DataFrame column names for feature extraction
FEATURE_COLUMN_MAP = {
    "fw_return_1bar": "return_1bar",
    "fw_return_4bar": "return_4bar",
    "fw_return_16bar": "return_16bar",
    "fw_return_96bar": "return_96bar",
    "fw_momentum_acceleration": "momentum_acceleration",
    "fw_bb_position": "bb_position",
    "fw_price_vs_sma50": "price_vs_sma50",
    "fw_price_vs_sma200": "price_vs_sma200",
    "fw_zscore_20d": "zscore_20d",
    "fw_atr_normalized": "atr_normalized",
    "fw_volatility_ratio": "volatility_ratio",
    "fw_bb_width": "bb_width",
    "fw_intraday_range": "intraday_range",
    "fw_volume_ratio": "volume_ratio",
    "fw_cmf": "cmf",
    "fw_mfi": "mfi",
    "fw_obv_slope": "obv_slope",
    "fw_rsi": "rsi",
    "fw_macd_histogram": "macd_histogram",
    "fw_adx": "adx",
    "fw_roc": "roc",
    "fw_stoch_k": "stoch_k",
    "fw_supertrend_dir": "supertrend_direction",
    "fw_cci": "cci",
    "fw_funding_rate": "funding_rate",
    "fw_quote_volume_ratio": "quote_volume_ratio",
    # Strategy signal features (14)
    "fw_sig_momentum": "sig_momentum",
    "fw_sig_momentum_conf": "sig_momentum_conf",
    "fw_sig_mean_reversion": "sig_mean_reversion",
    "fw_sig_mean_reversion_conf": "sig_mean_reversion_conf",
    "fw_sig_trend_following": "sig_trend_following",
    "fw_sig_trend_following_conf": "sig_trend_following_conf",
    "fw_sig_volatility_breakout": "sig_volatility_breakout",
    "fw_sig_volatility_breakout_conf": "sig_volatility_breakout_conf",
    "fw_sig_funding_volume": "sig_funding_volume",
    "fw_sig_funding_volume_conf": "sig_funding_volume_conf",
    "fw_n_buy_signals": "n_buy_signals",
    "fw_n_sell_signals": "n_sell_signals",
    "fw_signal_consensus": "signal_consensus",
    "fw_max_confidence": "max_confidence",
}


@dataclass
class Chromosome:
    """A chromosome representing trading parameters evolved by the GA."""

    genes: dict[str, float | int] = field(default_factory=dict)
    fitness: float = 0.0

    @classmethod
    def random(cls) -> "Chromosome":
        genes = {spec.name: spec.random_value() for spec in GENE_SPECS}
        return cls(genes=genes)

    @classmethod
    def from_defaults(cls) -> "Chromosome":
        genes = {spec.name: spec.default for spec in GENE_SPECS}
        return cls(genes=genes)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chromosome":
        return cls(genes=data)

    def copy(self) -> "Chromosome":
        return Chromosome(genes=self.genes.copy(), fitness=self.fitness)

    def get(self, name: str) -> float | int:
        return self.genes[name]

    def set(self, name: str, value: float | int) -> None:
        self.genes[name] = value

    def mutate(self, mutation_rate: float = 0.1) -> "Chromosome":
        """Mutate genes with Gaussian perturbation (10% of gene range)."""
        child = self.copy()
        for spec in GENE_SPECS:
            if random.random() < mutation_rate:
                range_size = spec.max_val - spec.min_val
                noise = random.gauss(0, 0.1 * range_size)
                new_val = child.genes[spec.name] + noise
                new_val = max(spec.min_val, min(spec.max_val, new_val))
                if spec.dtype == "int":
                    new_val = round(new_val)
                child.genes[spec.name] = new_val
        return child

    def crossover(self, other: "Chromosome") -> tuple["Chromosome", "Chromosome"]:
        """Uniform crossover — each gene independently swaps with 50% probability."""
        child1_genes = {}
        child2_genes = {}
        for name in self.genes:
            if random.random() < 0.5:
                child1_genes[name] = self.genes[name]
                child2_genes[name] = other.genes[name]
            else:
                child1_genes[name] = other.genes[name]
                child2_genes[name] = self.genes[name]
        return Chromosome(genes=child1_genes), Chromosome(genes=child2_genes)
