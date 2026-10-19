# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base metric class and batch/aggregate helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch


class Metric:
    """Base class for metrics that accumulate results over multiple __call__ and expose
    aggregate()."""

    def __init__(self, **kwargs):
        self.clear()

    def __call__(self, *args, **kwargs):
        """Compute metric for current batch, append to saved_metrics, and return the batch
        result."""
        metrics = self._compute(*args, **kwargs)
        for key, val in metrics.items():
            self.saved_metrics[key].append(val.detach().cpu().float())
        return metrics

    def _compute(self, **kwargs):
        """Subclasses implement this to compute metric dict from batch inputs."""
        raise NotImplementedError()

    def clear(self):
        """Reset all accumulated metric values."""
        self.saved_metrics = defaultdict(list)

    def aggregate(self):
        """Return a dict of concatenated/stacked tensors over all accumulated batches."""
        output = {}
        for key, lst in self.saved_metrics.items():
            try:
                output[key] = torch.cat(lst)
            except RuntimeError:
                output[key] = torch.stack(lst)
        return output


def compute_metrics(metrics_list: List[Metric], metrics_in: Dict) -> Dict:
    """Run each metric on metrics_in and return the combined dict of batch results."""
    metrics_out = {}
    for metric in metrics_list:
        metrics_out.update(metric(**metrics_in))
    return metrics_out


def aggregate_metrics(metrics_list: List[Metric]) -> Dict:
    """Return combined aggregated results (concatenated over batches) for all metrics."""
    metrics_out = {}
    for metric in metrics_list:
        metrics_out.update(metric.aggregate())
    return metrics_out


def clear_metrics(metrics_list: List[Metric]) -> None:
    """Clear accumulated values for all metrics in the list."""
    for metric in metrics_list:
        metric.clear()
