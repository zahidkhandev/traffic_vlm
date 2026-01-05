# Task 31: TensorBoard/W&B Wrappers

import logging
import sys
from pathlib import Path


def setup_logger(name, save_dir=None, filename="train.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(save_dir / filename)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
