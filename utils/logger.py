"""
Lightweight metric logger. Writes CSV + optional TensorBoard.
"""

import os
import csv
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MetricLogger:
    def __init__(self, log_dir: str, use_tensorboard: bool = False):
        self.log_dir = log_dir
        self._csv_path = os.path.join(log_dir, "metrics.csv")
        self._file = open(self._csv_path, "w", newline="")
        self._writer: Optional[csv.writer] = None

        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=log_dir)
                logger.info("TensorBoard writer enabled.")
            except ImportError:
                logger.warning("TensorBoard not installed; skipping.")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is None:
            self._writer = csv.writer(self._file)
            self._writer.writerow(["step", "tag", "value"])

        self._writer.writerow([step, tag, f"{value:.6f}"])
        self._file.flush()

        if self._tb is not None:
            self._tb.add_scalar(tag, value, step)

    def close(self) -> None:
        self._file.close()
        if self._tb is not None:
            self._tb.close()
