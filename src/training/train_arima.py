"""Train ARIMA baseline for all 100 series."""

from src.config import get_settings
from src.models.baselines import ARIMABaseline
from src.training._baseline_trainer import train_single_baseline


def main() -> None:
    train_single_baseline(ARIMABaseline, get_settings())


if __name__ == "__main__":
    main()
