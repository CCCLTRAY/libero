"""Minimal external-controller example for libero_scene2.evaluate."""

from __future__ import annotations

import numpy as np


class ZeroController:
    def reset(self, context) -> None:
        self.context = context

    def act(self, observation) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)


def make_controller(context):
    return ZeroController()
