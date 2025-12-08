import json
from typing import Dict
from pathlib import Path
from datetime import timedelta
from enum import Enum
from json import JSONEncoder
import gin
import logging
import numpy as np
import torch

from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import Module
from torch.optim import Optimizer, Adam, SGD, RAdam
from typing import Optional, Union
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, MultiStepLR, ExponentialLR
import pdb

def save_config_file(log_dir):
    config_path = log_dir / "train_config.gin"
    with config_path.open("w") as f:
        f.write(gin.operative_config_str())


def create_optimizer(name: str, model: Module, lr: float, momentum: float = 0) -> Optimizer:
    """creates the specified optimizer with the given parameters

    Args:
        name (str): str name of optimizer
        model (Module): the model used for training
        lr (float): learning rate
        momentum (float): momentum (only for sgd optimizer)

    Raises:
        ValueError: thrown if optimizer name not known

    Returns:
        Optimizer: the model optimizer
    """
    name = name.lower()
    if name == "adam":
        return Adam(params=model.parameters(), lr=lr)
    elif name == "sgd":
        return SGD(params=model.parameters(), lr=lr, momentum=momentum)
    elif name == "radam":
        return RAdam(params=model.parameters(), lr=lr)
    else:
        raise ValueError(f"No optimizer with name {name} found!")


def create_scheduler(
    scheduler_name: Optional[str],
    optimizer: Optimizer,
    lr_factor: float,
    lr_steps: Optional[list],
    epochs: int,
) -> Union[_LRScheduler, None]:
    """creates a learning rate scheduler with the given parameters

    Args:
        scheduler_name (Optional[str]): str name of scheduler or None, in which case None will be returned
        optimizer (Optimizer): the learning optimizer
        lr_factor (float): the learning rate factor
        lr_steps (Optional[list]): learning rate steps for the scheduler to take (only supported for step scheduler)
        epochs (int): number of scheduler epochs (only supported for cosine scheduler)

    Raises:
        ValueError: thrown if step scheduler was chosen but no steps were passed
        ValueError: thrown if scheduler name not known and not None

    Returns:
        Union[_LRScheduler, None]: either the learning rate scheduler object or None if scheduler_name was None
    """
    if scheduler_name == "step":
        if not lr_steps:
            raise ValueError("step scheduler chosen but no lr steps passed!")
        return MultiStepLR(optimizer, lr_steps, lr_factor)
    elif scheduler_name == "exponential":
        return ExponentialLR(optimizer, lr_factor)
    elif scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, epochs)
    elif not scheduler_name:
        return None
    else:
        raise ValueError(f"no scheduler with name {scheduler_name} found!")


def export_catboost_feature_artifacts(model, feature_names, output_dir: Path, save_model_json: bool = False):
    """
    Persist CatBoost global feature importances and pairwise interactions in a readable form.
    Optionally dumps the full model as JSON to inspect tree splits.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(feature_names) if feature_names is not None else None
    def _name(idx: int) -> str:
        if names and idx < len(names):
            return names[idx]
        return f"feature_{idx}"

    try:
        importances = model.get_feature_importance(type="FeatureImportance")
        fi_pretty = [
            {"feature": _name(i), "importance": float(score)}
            for i, score in enumerate(importances)
        ]
        fi_pretty.sort(key=lambda item: item["importance"], reverse=True)
        (output_dir / "feature_importances.json").write_text(json.dumps(fi_pretty, indent=2))
    except Exception as e:
        logging.warning("Could not export CatBoost feature importances: %s", e)

    try:
        interactions = model.get_feature_importance(type="Interaction")
        inter_pretty = [
            {"feat1": _name(int(i)), "feat2": _name(int(j)), "score": float(score)}
            for i, j, score in interactions
        ]
        inter_pretty.sort(key=lambda item: abs(item["score"]), reverse=True)
        (output_dir / "feature_interactions.json").write_text(json.dumps(inter_pretty, indent=2))
    except Exception as e:
        logging.warning("Could not export CatBoost feature interactions: %s", e)

    if save_model_json:
        try:
            model.save_model(str(output_dir / "model.json"), format="json")
        except Exception as e:
            logging.warning("Could not export CatBoost model JSON: %s", e)


class JsonResultLoggingEncoder(JSONEncoder):
    """JSON converter for objects that are not serializable by default."""

    # Serializes foreign datatypes
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, tuple):
            if isinstance(obj)[0] is torch.Tensor or isinstance(obj)[0] is np.ndarray:
                return map(lambda item: item.tolist(), obj)
        if isinstance(obj, timedelta):
            return str(obj)
        return JSONEncoder.default(self, obj)


class Align(str, Enum):
    LEFT = "<"
    CENTER = "^"
    RIGHT = ">"


def log_table_row(
    cells: list,
    level: int = logging.INFO,
    widths: list[int] = None,
    header: list[str] = None,
    align: Align = Align.LEFT,
    highlight: bool = False,
):
    """Logs a table row.

    Args:
        cells: List of cells to log.
        level: Logging level.
        widths: List of widths for each cell.
        header: List of headers to calculate widths if widths not supplied.
        highlight: If set to true, highlight the row.
    """
    table_cells = cells
    if not widths and header:
        widths = [len(head) for head in header]
    if widths:
        table_cells = []
        for cell, width in zip(cells, widths):
            cell = str(cell)[:width]  # truncate cell if it is too long
            table_cells.append("{: {align}{width}}".format(cell, align=align.value, width=width))
    table_row = " | ".join([f"{cell}" for cell in table_cells])
    if highlight:
        table_row = f"\x1b[31;32m{table_row}\x1b[0m"
    logging.log(level, table_row)


class JSONMetricsLogger(Logger):
    def __init__(self, output_dir=None, **kwargs):
        super().__init__(**kwargs)
        if output_dir is None:
            output_dir = Path.cwd() / "metrics"
        logging.info(f"logging metrics to file: {str(output_dir.resolve())}")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self):
        return "json_metrics_logger"

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        old_metrics = {}
        stage_metrics = {
            "train": {"/".join(key.split("/")[1:]): value for key, value in metrics.items() if key.startswith("train/")},
            "val": {"/".join(key.split("/")[1:]): value for key, value in metrics.items() if key.startswith("val/")},
            "test": {"/".join(key.split("/")[1:]): value for key, value in metrics.items() if key.startswith("test/")},
        }
        for stage, metrics in stage_metrics.items():
            if metrics:
                output_file = self.output_dir / f"{stage}_metrics.json"
                old_metrics = {}
                if output_file.exists():
                    try:
                        with output_file.open("r") as f:
                            old_metrics = json.load(f)
                        logging.debug(f"updating {stage} metrics file...")
                    except json.decoder.JSONDecodeError:
                        logging.warning("could not decode json file, overwriting...")

                old_metrics.update(metrics)
                with output_file.open("w") as f:
                    json.dump(old_metrics, f, cls=JsonResultLoggingEncoder, indent=4)

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        pass
