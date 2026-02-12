"""
Optuna hyperparameter search for TFT.

Uses TPE sampler with Median Pruner. Each trial calls train_tft()
and retrieves the best_val_loss from MLflow. The best trial's
hyperparameters are saved to configs/best_hparams.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import optuna
from loguru import logger

from src.config import Settings, get_settings
from src.training.train import train_tft


def objective(trial: optuna.Trial, settings: Settings) -> float:
    """
    Optuna objective: train one TFT trial and return best_val_loss.

    Search space
    ------------
    hidden_size          : categorical [32, 64, 128, 256]
    attention_head_size  : categorical [1, 2, 4]
    dropout              : uniform [0.05, 0.3]
    hidden_continuous_size: categorical [8, 16, 32, 64]
    lstm_layers          : int [1, 3]
    learning_rate        : log-uniform [1e-5, 1e-2]
    gradient_clip_val    : log-uniform [0.01, 1.0]
    batch_size           : categorical [32, 64, 128]

    Note on max_epochs during tuning
    ---------------------------------
    We reduce max_epochs to 30 during search to speed up each trial.
    EarlyStopping will terminate bad trials quickly. The final model
    is retrained from scratch with the best params at full max_epochs.
    """
    overrides: dict[str, object] = {
        "tft.hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "tft.attention_head_size": trial.suggest_categorical(
            "attention_head_size", [1, 2, 4]
        ),
        "tft.dropout": trial.suggest_float("dropout", 0.05, 0.3),
        "tft.hidden_continuous_size": trial.suggest_categorical(
            "hidden_continuous_size", [8, 16, 32, 64]
        ),
        "tft.lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
        "tft.learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "tft.gradient_clip_val": trial.suggest_float(
            "gradient_clip_val", 0.01, 1.0, log=True
        ),
        "tft.batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        # Reduce epochs for search speed; EarlyStopping handles termination
        "tft.max_epochs": 30,
    }

    try:
        run_id = train_tft(
            settings=settings,
            experiment_name="jamuna_tft_optuna",
            run_name=f"trial_{trial.number}",
            overrides=overrides,
        )
    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()

    # Retrieve best_val_loss from MLflow
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    try:
        run = mlflow.get_run(run_id)
        val_loss = float(run.data.metrics.get("best_val_loss", float("inf")))
    except Exception as e:
        logger.warning(f"Could not retrieve val_loss for trial {trial.number}: {e}")
        val_loss = float("inf")

    logger.info(f"Trial {trial.number}: val_loss={val_loss:.4f} | params={overrides}")
    return val_loss


def run_hyperparameter_search(
    settings: Settings | None = None,
    n_trials: int | None = None,
    resume: bool = True,
) -> optuna.Study:
    """
    Run Optuna hyperparameter search.

    Parameters
    ----------
    settings : Settings instance
    n_trials : number of trials (overrides config)
    resume   : if True, resumes existing study from SQLite storage

    Returns
    -------
    Completed optuna.Study for analysis
    """
    if settings is None:
        settings = get_settings()

    n = n_trials or settings.optuna.n_trials

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
    )
    sampler = optuna.samplers.TPESampler(seed=settings.training.random_seed)

    study = optuna.create_study(
        study_name=settings.optuna.study_name,
        storage=settings.optuna.storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=resume,
    )

    logger.info(
        f"Starting Optuna search: {n} trials | "
        f"storage={settings.optuna.storage} | "
        f"study={settings.optuna.study_name}"
    )

    study.optimize(
        lambda trial: objective(trial, settings),
        n_trials=n,
        timeout=settings.optuna.timeout_seconds,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    # ── Save best hyperparameters ─────────────────────────────────────────────
    best = study.best_trial
    logger.info(
        f"Best trial: #{best.number} | "
        f"val_loss={best.value:.4f} | "
        f"params={best.params}"
    )

    hparams_path = Path("configs/best_hparams.json")
    hparams_path.parent.mkdir(parents=True, exist_ok=True)
    with hparams_path.open("w") as f:
        json.dump({"trial_number": best.number, "val_loss": best.value, **best.params}, f, indent=2)
    logger.info(f"Best hyperparameters saved to {hparams_path}")

    return study


def train_best_model(
    settings: Settings | None = None,
    hparams_path: str = "configs/best_hparams.json",
) -> str:
    """
    Retrain TFT from scratch using the best hyperparameters at full max_epochs.

    Call this after run_hyperparameter_search() completes.

    Returns
    -------
    MLflow run_id of the final best model
    """
    if settings is None:
        settings = get_settings()

    hparams_file = Path(hparams_path)
    if not hparams_file.exists():
        raise FileNotFoundError(
            f"Best hyperparameters not found at {hparams_file}. "
            "Run run_hyperparameter_search() first."
        )

    with hparams_file.open() as f:
        best_params = json.load(f)

    # Remove metadata keys; keep only hyperparameter keys
    meta_keys = {"trial_number", "val_loss"}
    overrides = {
        f"tft.{k}": v
        for k, v in best_params.items()
        if k not in meta_keys
    }
    # Restore full max_epochs (not the reduced 30 used during search)
    overrides.pop("tft.max_epochs", None)

    logger.info(f"Training best model with params: {overrides}")
    run_id = train_tft(
        settings=settings,
        experiment_name=settings.mlflow.experiment_name,
        run_name="best_model_final",
        overrides=overrides,
    )
    return run_id


def main() -> None:
    """CLI entry point: python -m src.training.tune"""
    import argparse

    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for TFT")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true", help="Start a fresh study")
    parser.add_argument("--train-best", action="store_true", help="Train best model after search")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    from src.config import load_settings

    settings = load_settings(args.config)
    study = run_hyperparameter_search(
        settings=settings,
        n_trials=args.n_trials,
        resume=not args.no_resume,
    )

    if args.train_best:
        run_id = train_best_model(settings)
        logger.info(f"Best model trained. MLflow run_id: {run_id}")


if __name__ == "__main__":
    main()
