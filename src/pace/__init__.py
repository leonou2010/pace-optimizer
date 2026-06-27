"""PACE — an AdamW variant with an lr-scaled pullback toward an EMA of the iterates.

The public entry point is the :class:`~pace.optimizer.PACE` optimizer. Setting
``lambda_pullback=0`` recovers plain AdamW; adding ``use_ema_eval=True`` recovers
the EMA-evaluation baseline.
"""

from pace.optimizer import PACE

__all__ = ["PACE"]
__version__ = "0.1.0"
