from argparse import Namespace
import logging
import os
from pathlib import Path

import wandb


def wandb_running() -> bool:
    """Check if wandb is running."""
    return wandb.run is not None


def update_wandb_config(config: dict) -> None:
    """updates wandb config if wandb is running

    Args:
        config (dict): config to set
    """
    logging.debug(f"Updating Wandb config: {config}")
    if wandb_running():
        wandb.config.update(config)


def apply_wandb_sweep(args: Namespace) -> Namespace:
    """applies the wandb sweep configuration to the namespace object

    Args:
        args (Namespace): parsed arguments

    Returns:
        Namespace: arguments with sweep configuration applied (some are applied via hyperparams)
    """
    wandb.init()
    sweep_config = wandb.config
    args.__dict__.update(sweep_config)
    if args.hyperparams is None:
        args.hyperparams = []
    for key, value in sweep_config.items():
        args.hyperparams.append(f"{key}=" + (("'" + value + "'") if isinstance(value, str) else str(value)))
    logging.info(f"hyperparams after loading sweep config: {args.hyperparams}")
    return args


def wandb_log(log_dict):
    """logs metrics to wandb

    Args:
        log_dict (dict): metric dict to log
    """
    if wandb_running():
        metric_prefix = os.getenv("YAIB_WANDB_METRIC_PREFIX", "").strip().strip("/")
        if metric_prefix:
            wandb.log({f"{metric_prefix}/{key}": value for key, value in log_dict.items()})
        else:
            wandb.log(log_dict)


def set_wandb_experiment_name(args, mode):
    """stores the run name in wandb config

    Args:
        args (Namespace): parsed arguments
        mode (RunMode): run mode
    """
    if args.name is None:
        data_dir = Path(args.data_dir)
        args.name = data_dir.name
    run_name = f"{mode}_{args.model}_{args.name}"

    if args.fine_tune:
        run_name += f"_source_{args.source_name}_fine-tune_{args.fine_tune}_samples"
    elif args.eval:
        run_name += f"_source_{args.source_name}"
    elif args.samples:
        run_name += f"_train_size_{args.samples}_samples"
    elif args.complete_train:
        run_name += "_complete_training"

    if wandb_running():
        # When resuming a run id across multiple commands (e.g., train + eval),
        # keep the original run name instead of mutating config["run-name"].
        existing_run_name = wandb.config.get("run-name")
        if existing_run_name and existing_run_name != run_name:
            logging.info(
                f'Keeping existing wandb run-name "{existing_run_name}" (computed "{run_name}").'
            )
            return
        wandb.config.update({"run-name": run_name}, allow_val_change=True)
        wandb.run.name = run_name
        wandb.run.save()
