"""Pydantic-based configuration for Aether BTC."""

from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    model_config = {"env_prefix": "AETHER_BTC_DB_"}

    host: str = "localhost"
    port: int = 5432
    name: str = "aether_btc"
    user: str = "aether_btc"
    password: str = ""

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class BinanceConfig(BaseSettings):
    """Binance API configuration."""

    model_config = {"env_prefix": "BINANCE_"}

    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True


class TradingConfig(BaseSettings):
    """Trading parameters for crypto."""

    model_config = {"env_prefix": "AETHER_BTC_TRADING_"}

    initial_capital: float = 1000.0
    commission_rate: float = 0.0004  # 0.04% taker fee (Binance Futures)
    slippage_rate: float = 0.0001  # 0.01% default slippage
    max_leverage: float = 20.0
    max_positions: int = 3  # Small account, limit concurrent positions
    max_position_size: float = 0.50  # 50% of capital per position (single pair)
    daily_loss_limit: float = 0.10  # 10% daily loss circuit breaker
    maintenance_margin_rate: float = 0.004  # Binance BTC/USDT bracket 1


class BacktestConfig(BaseSettings):
    """Backtesting configuration."""

    model_config = {"env_prefix": "AETHER_BTC_BACKTEST_"}

    slippage_model: str = "fixed_pct"
    walk_forward_train_days: int = 180  # 6 months
    walk_forward_test_days: int = 60  # 2 months
    forced_liquidation_slippage_multiplier: float = 2.0
    bars_per_funding: int = 32  # 8h / 15min = 32 bars between funding


class AlertConfig(BaseSettings):
    """Telegram alert configuration."""

    model_config = {"env_prefix": "AETHER_BTC_ALERT_"}

    telegram_token: str = ""
    telegram_chat_id: str = ""


class AetherBTCConfig(BaseSettings):
    """Root configuration combining all sub-configs."""

    model_config = {"env_prefix": "AETHER_BTC_"}

    environment: str = Field(default="development", description="development|paper|live")
    log_level: str = "INFO"
    log_format: str = "json"

    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
