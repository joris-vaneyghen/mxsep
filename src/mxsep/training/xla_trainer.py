# see https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist_amp.py
# turorial https://docs.pytorch.org/xla/release/r2.8/learn/pytorch-on-xla-devices.html#running-on-multiple-xla-devices-with-multi-processing
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch_xla.amp import autocast
from torch.utils.data import DataLoader

from mxsep.cfg import Config
from mxsep.data.dataset import PredefinedMixDataset
from mxsep.models import ISTFTModule, MusicSourceSeparationModel
from mxsep.training import Monitor


class XLATrainer:
    """Main training class"""

    def __init__(self, rank: int, world_size: int, cfg: Config):
        assert cfg.model
        assert cfg.training.optimizer

        self.rank = rank
        self.world_size = world_size

        assert cfg.training.device == 'xla'

        self.device = torch_xla.device()

        if os.environ.get('XLA_EAGER_MODE', default=0):
            torch_xla.experimental.eager_mode(True)

        if cfg.training.max_runtime:
            self.start_time = time.time()
            self.last_run = -1
            self.max_runtime = cfg.training.max_runtime
        else:
            self.max_runtime = None

        self.spectrogram_mode = cfg.training.stft_device == 'cpu'

        self.epochs = cfg.training.epochs
        self.use_amp = cfg.training.use_amp
        self.gradient_clip = cfg.training.gradient_clip
        self.seed = cfg.training.seed
        self.deterministic = cfg.training.deterministic
        self.target_sources = cfg.model.target_sources

        self.checkpoint_dir = Path(cfg.training.checkpoint_dir)


        if cfg.dataset.train.predefined_jsonl_path:
            self.multiple_predefined_mixes = Path(cfg.dataset.train.predefined_jsonl_path).suffix != ".jsonl"

            if self.spectrogram_mode:
                self.train_dataset = PredefinedMixDataset(cfg.dataset, split='train', stft_cfg=cfg.model.stft)

            else:
                self.train_dataset = PredefinedMixDataset(cfg.dataset, split='train')

            shuffle = not(self.multiple_predefined_mixes) # if single json file per epoch we should shuffle.
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=cfg.training.batch_size, shuffle=shuffle, pin_memory=True,
                                           num_workers=cfg.training.num_workers,  drop_last=True)
            self.train_loader = pl.MpDeviceLoader(self.train_loader, self.device)
        else:
            raise NotImplementedError("Only predefined mix dataset is implemented for now")


        if self.spectrogram_mode:
            self.istft = ISTFTModule(cfg.model.stft)

        if cfg.dataset.validation:
            if cfg.dataset.validation.predefined_jsonl_path:

                if self.spectrogram_mode:
                    self.validation_dataset = PredefinedMixDataset(cfg.dataset, split='validation', stft_cfg=cfg.model.stft)

                else:
                    self.validation_dataset = PredefinedMixDataset(cfg.dataset, split='validation')

                self.validation_loader = DataLoader(self.validation_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                                               num_workers=cfg.training.num_workers, drop_last=True)
                self.validation_loader = pl.MpDeviceLoader(self.validation_loader, self.device)
            else:
                raise NotImplementedError("Only predefined mix dataset is implemented for now")
        else:
            self.validation_loader = None


        self.init_randomizer()

        self.model = MusicSourceSeparationModel(cfg.model)
        self.model.to(self.device)
        xm.broadcast_master_param(self.model)
        
        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(cfg.training.optimizer, _partial_=True)(params=self.model.parameters())
        if cfg.training.lr_scheduler:
            self.lr_scheduler = hydra.utils.instantiate(cfg.training.lr_scheduler, _partial_=True)(
                optimizer=self.optimizer)
        else:
            self.lr_scheduler = None

        # Trackers
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float("inf") # todo if metric = sdr then +inf else -inf

        if cfg.training.resume_from_checkpoint:
            self._load_checkpoint(cfg.training.resume_from_checkpoint)
            self.epoch += 1

        self.loss_fn = hydra.utils.instantiate(cfg.training.loss)

        if xm.is_master_ordinal():
            self.monitor = Monitor(cfg)
            self.monitor.global_step = self.global_step

            self.monitor.watch(self.model, self.loss_fn)


    def init_randomizer(self):
        """Initialize all random number generators based on config."""
        # Python built-in random
        random.seed(self.seed)

        # NumPy random
        np.random.seed(self.seed)

        # PyTorch random
        torch.manual_seed(self.seed)

        # todo seed xla (~torch.cuda.manual_seed)

        # Optional: make CuDNN deterministic (slower but reproducible)
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        xm.master_print(f"Randomizer initialized with seed: {self.seed}")

    def train_epoch(self) -> dict[str, float] | None:
        """Train for one epoch"""
        if self.multiple_predefined_mixes:
            # Load new mixes for this epoch
            self.train_dataset.init_epoch(self.epoch)
            # reset sampler to allow variable dataset lengths
            # num_samples = len(self.train_dataset) // self.world_size
            # self.train_loader.sampler.num_samples = num_samples
            # self.train_loader.sampler.total_size = num_samples  * self.world_size

        self.model.train()
        epoch_loss = 0.0
        steps = 0
        epoch_metrics = {}
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            if self.spectrogram_mode:
                (x, y, y_waveform) = batch
            else:
                (x, y) = batch

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(self.device):
                pred = self.model(x, spectrogram_mode=self.spectrogram_mode)
                loss = self.loss_fn(torch.view_as_real(pred).contiguous(), torch.view_as_real(y).contiguous() )
                # if self.spectrogram_mode:
                #     pred = pred.cpu()
                #     pred = self.istft(pred)
                #     y = y_waveform

                # assert pred.shape == y.shape
                # loss = self.loss_fn(pred, y)
            # Backward pass

            loss.backward()

            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
            )

            xm.optimizer_step(self.optimizer)

            # Update scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Update trackers
            steps += 1
            self.global_step += 1
                
            batch_loss =  loss.item()
            batch_loss = xm.mesh_reduce("batch_loss", batch_loss, np.mean)

            if xm.is_master_ordinal():
                epoch_loss += batch_loss
                metrics = {
                    "batch_idx": batch_idx,
                    "epoch": self.epoch,
                    "loss": batch_loss,
                    "loss_avg": epoch_loss / steps,
                    "lr": self.optimizer.param_groups[0][
                        "lr"
                    ],  # or self.optimizer.defaults['lr'],
                }
                self.monitor.log_step(metrics, self.epoch, batch_idx)


            # # Save checkpoint
            # if self.global_step % self.config.save_interval == 0:
            #     self._save_checkpoint()

            # Debug memory
            # if self.memory_debugger:
            #     self.memory_debugger.snapshot(self.global_step)

        if xm.is_master_ordinal():
            # Calculate epoch averages
            avg_loss = epoch_loss / steps
            return {'avg_loss': avg_loss}
        else :
            return None

    def validate(self) -> dict[Any, Any] | dict[Any, float] | None:
        """Run validation"""
        if self.validation_loader is None:
            return {}

        self.model.eval()
        val_loss = 0.0
        all_metrics = {}

        with torch.no_grad():
        # with torch.inference_mode():
            for batch_idx, batch in enumerate(self.validation_loader):
                # Move to device
                if self.spectrogram_mode:
                    (x, y, y_waveform) = batch
                else:
                    (x, y) = batch

                x = x.to(self.device)
                y = y.to(self.device)

                with autocast(self.device):
                    outputs = self.model(x, spectrogram_mode=self.spectrogram_mode)
                    loss = self.loss_fn(outputs, y)
                
                val_loss = loss.item()
                val_loss = xm.mesh_reduce("val_loss", val_loss, np.mean)

                if xm.is_master_ordinal():
                    all_metrics["val_loss"].append(val_loss)
                    avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
                    data = {
                        **avg_metrics,
                        "batch_idx": batch_idx,
                    }
                    self.monitor.log_data(data)


        if xm.is_master_ordinal():
            # Average metrics
            avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}

            # Save best model
            if avg_metrics['val_loss'] < self.best_metric: #todo if sdr then '>'
                self.best_metric = avg_metrics['val_sdr']
                self._save_checkpoint('best_model.pt')
    
            self._save_checkpoint('latest_model.pt')

            return avg_metrics
        
        else :
            return None

    def _load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint_path = Path(self.checkpoint_dir) / filename
        if not checkpoint_path.exists():
            xm.master_print(f"Checkpoint {filename} not found, starting from scratch.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # if self.scheduler and checkpoint['scheduler_state_dict']:
        #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', float('inf'))

        xm.master_print(f"Loaded checkpoint {filename} (epoch {self.epoch}, global step {self.global_step})")


    def _save_checkpoint(self, filename: str = 'checkpoint.pt'):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.model.config,
        }
        xm.master_print(f"Save checkpoint {filename}")

        xm.save(checkpoint, checkpoint_dir / filename)
        # or xser.save(checkpoint, checkpoint_dir / filename)

    def train(self):
        """Main training loop"""
        for epoch in range(self.epoch, self.epochs):
            start_time = time.time()
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Print summary
            if xm.is_master_ordinal():
                self.monitor.log_epoch_summary(
                    train_metrics, val_metrics, self.epoch
                )

            self.last_run = time.time() - start_time
            if self.max_runtime:
                total_time = time.time() - self.start_time
                left =  self.max_runtime - total_time
                if left < self.last_run:
                    break

            # Save epoch checkpoint
            # self._save_checkpoint(f'epoch_{epoch}.pt')
