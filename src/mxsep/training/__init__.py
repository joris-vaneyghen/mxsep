from .monitor import Monitor
from .trainer import Trainer
from .ddp_trainer import DDPTrainer

# Conditional import for XLA
try:
    from .xla_trainer import XLATrainer
except ModuleNotFoundError:
    # Define a placeholder or None if XLA not available
    XLATrainer = None
    # Optionally log a warning
    import warnings
    warnings.warn("torch_xla not installed. XLATrainer will not be available.")

# Optional: Define __all__ to control what's exported
__all__ = [
    'Monitor',
    'Trainer', 
    'DDPTrainer',
    'XLATrainer',  # Will be None if not available
]