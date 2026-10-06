"""Training recipes built on the SDK primitives.

- ``supervised``: fine-tune on labeled rows with held-out evaluation. The default way to tune a model.
- ``distill``: label rows with a teacher of your choice, then fine-tune on its answer distributions.
"""

from . import distill, supervised
from .distill import DistillConfig, Teacher
from .supervised import SupervisedConfig, SupervisedResult

__all__ = ["supervised", "distill", "SupervisedConfig", "SupervisedResult", "DistillConfig", "Teacher"]
