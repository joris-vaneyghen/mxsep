import logging
from typing import Dict, Optional, Union, Iterable, Any
from contextlib import contextmanager

import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from mxsep.cfg import Config


class Monitor:
    def __init__(self, cfg: Config):
        cfg_monitoring = cfg.training.monitoring
        self.log_interval = cfg_monitoring.log_interval
        self.show_progress_bar = cfg_monitoring.show_progress_bar
        self.pbar = None
        self.use_wandb = cfg_monitoring.wandb is not None

        # Setup logger
        self.logger = logging.getLogger(__name__)

        # Initialize wandb if configured and enabled
        if self.use_wandb and cfg_monitoring.wandb:
            self._init_wandb(cfg)

        # For tracking global steps
        self.global_step = 0
        self.epoch = 0


    def _init_wandb(self, cfg: Config):
        """Initialize wandb logging"""
        wandb_cfg = cfg.training.monitoring.wandb
        self.wandb_watch = wandb_cfg.watch
        if wandb_cfg.api_key:
            wandb.login(key=wandb_cfg.api_key)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        wandb.init(
            project=wandb_cfg.project,
            name=wandb_cfg.name,
            job_type=wandb_cfg.job_type,
            tags=wandb_cfg.tags,
            notes=wandb_cfg.notes,
            config=cfg_dict,
            sync_tensorboard=True
        )
        self.logger.info("Weights & Biases initialized successfully")

    
    
    def watch(self, model, criterion):
        if self.use_wandb and self.wandb_watch:
            watch_args:dict = OmegaConf.to_container(self.wandb_watch, resolve=True)
            wandb.watch(model, criterion, **watch_args)
    
    def progress_bar(
        self, iterable: Iterable, desc: Optional[str] = None, **tqdm_kwargs
    ) -> Union[tqdm, Iterable]:
        """
        Enhanced progress bar with tqdm improvements

        Improvements:
        - Dynamic description updates
        - Postfix dictionary for metrics
        - Smooth display with leave=False for nested bars
        - Configurable position for multiple bars
        """
        if not self.show_progress_bar:
            return iterable

        # Default tqdm settings with improvements
        tqdm_defaults = {
            "desc": desc or "Processing",
            "dynamic_ncols": True,  # Adapt to terminal width
            "smoothing": 0.1,  # Exponential moving average smoothing
            "leave": True,  # Keep bar after completion
            "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            "unit": "it",
            **tqdm_kwargs,
        }

        self.pbar = tqdm(iterable, **tqdm_defaults)
        return self.pbar

    @contextmanager
    def progress_context(self, total: int, desc: str = "Processing", **tqdm_kwargs):
        """Context manager for progress bars with automatic cleanup"""
        self.pbar = None
        if self.show_progress_bar:
            self.pbar = tqdm(total=total, desc=desc, **tqdm_kwargs)
        try:
            yield self.pbar
        finally:
            if self.pbar:
                self.pbar.close()
                self.pbar = None


    def update_progress_bar(
        self, pbar: tqdm, metrics: Dict[str, float], step: Optional[int] = None
    ):
        """Update progress bar with metrics dynamically"""
        if pbar and self.show_progress_bar:
            if step is not None:
                pbar.update(step)
            pbar.set_postfix(**{k: f"{v:.4f}" for k, v in metrics.items()})

    def log_step(
        self, data: Dict[str, float], epoch: int, batch_idx: int
    ):
        """Log training step with improved formatting"""
        self.global_step += 1
        self.epoch = epoch


        # Prepare log dictionary
        log_dict = {
            "train_loss": data['loss'],
            "epoch": epoch,
            "global_step": self.global_step,
            **data,
        }

        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log(log_dict, step=self.global_step)

        self.log_data(data)

        # # Console logging at specified intervals
        # if self.log_interval and batch_idx % self.log_interval == 0:
        #     # Format metrics for display
        #     metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in data.items()])
        #     log_message = f"Epoch {epoch}, Step {batch_idx}, {metrics_str}"
        #
        #     if self.pbar:
        #         self.pbar.set_postfix(**{k: f"{v:.4f}" for k, v in data.items()})
        #     self.logger.info(log_message)

    def log_data(self, data: Dict[str, float]):
        batch_idx = data["batch_idx"]
        # Console logging at specified intervals
        if self.log_interval and batch_idx % self.log_interval == 0:
            # Format metrics for display
            data_formatted = {
                        k: f"{v:.4f}" if isinstance(v, float) else str(v)
                        for k, v in data.items()
                    }
            log_message = ", ".join([f"{k}:{v}" for k, v in data_formatted.items()])

            if self.pbar:
                self.pbar.set_postfix(data_formatted)
            self.logger.info(log_message)

    def log_epoch_summary(self, train_metrics, val_metrics, epoch):
        self.logger.info(f"\nEpoch {epoch} Summary:")
        self.logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}")
        if val_metrics:
            self.logger.info("Validation Metrics:")
            for key, value in val_metrics.items():
                self.logger.info(f"  {key}: {value:.4f}")

        # Log to wandb
        if self.use_wandb:
            data = {
                **train_metrics,
                **val_metrics,
            }
            wandb.log(data, step=self.global_step)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""
    ):
        """Generic method to log arbitrary metrics"""
        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}

        if self.use_wandb:
            wandb.log(metrics, step=step or self.global_step)

        # Log to console at INFO level
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Metrics: {metrics_str}")

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to wandb and console"""
        self.logger.info("Hyperparameters:")
        for key, value in params.items():
            self.logger.info(f"  └─ {key}: {value}")

        if self.use_wandb:
            wandb.config.update(params, allow_val_change=True)

    def log_text(self, text: str, key: str = "log", step: Optional[int] = None):
        """Log text to wandb (useful for debugging)"""
        if self.use_wandb:
            wandb.log({key: text}, step=step or self.global_step)
        self.logger.info(text)

    def finish(self):
        """Clean up wandb session"""
        if self.use_wandb:
            wandb.finish()
        self.logger.info("Monitoring session finished")

    
