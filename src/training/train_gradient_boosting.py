"""Train Gradient Boosting baseline for all 100 series."""

from src.config import get_settings
from src.models.baselines import GradientBoostingBaseline
from src.training._baseline_trainer import train_single_baseline


def main() -> None:
    train_single_baseline(GradientBoostingBaseline, get_settings())


if __name__ == "__main__":
    main()
