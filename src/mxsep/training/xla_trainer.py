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
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from mxsep.cfg import Config
from mxsep.data.dataset import PredefinedMixDataset
from mxsep.models import ISTFTModule, MusicSourceSeparationModel
from mxsep.training import Monitor
from mxsep.utils.metrics import calculate_sdr


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

            shuffle = not(self.multiple_predefined_mixes)  # if single json file per epoch we should shuffle.
            self.train_sampler = DistributedSampler(
                dataset=self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                seed=self.seed,
                drop_last=True,
            )
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=cfg.training.batch_size, shuffle=False, sampler=self.train_sampler, pin_memory=False,
                                           num_workers=cfg.training.num_workers,  drop_last=True)
            self.train_loader = pl.MpDeviceLoader(self.train_loader, self.device)

            if xm.is_master_ordinal():
                xm.master_print(f"Dataset size: {len(self.train_dataset)}")
                xm.master_print(f"World size: {self.world_size}")
                xm.master_print(
                    f"Expected steps per epoch: {len(self.train_dataset) // (self.world_size * cfg.training.batch_size)}"
                )
        else:
            raise NotImplementedError("Only predefined mix dataset is implemented for now")


        if cfg.dataset.validation:
            if cfg.dataset.validation.predefined_jsonl_path:

                if self.spectrogram_mode:
                    self.validation_dataset = PredefinedMixDataset(cfg.dataset, split='validation', stft_cfg=cfg.model.stft)

                else:
                    self.validation_dataset = PredefinedMixDataset(cfg.dataset, split='validation')

                sampler = DistributedSampler(
                    dataset=self.validation_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                    seed=self.seed,
                    drop_last=True,
                )
                self.validation_loader = DataLoader(
                    self.validation_dataset,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=cfg.training.num_workers,
                    drop_last=True,
                )
                self.validation_loader = pl.MpDeviceLoader(
                    self.validation_loader, self.device
                )
            else:
                raise NotImplementedError("Only predefined mix dataset is implemented for now")
        else:
            self.validation_loader = None


        self.init_randomizer()

        self.model = MusicSourceSeparationModel(cfg.model)

        def init_weights(m):
            # --- CONVOLUTIONAL LAYERS (Encoder/Decoder) ---
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                # Kaiming/He initialization for Conv layers with ReLU
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

            # --- TRANSFORMER BOTTLENECK ---
            elif isinstance(m, torch.nn.Linear):
                # For attention projections and feed-forward networks
                # Use a smaller scale for deeper transformers
                if hasattr(m, "in_features") and hasattr(m, "out_features"):
                    # Xavier/Glorot is still good for Linear layers
                    # but scale it for transformer depth
                    torch.nn.init.xavier_uniform_(
                        m.weight, gain=0.02
                    )  # Smaller gain for stability
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

            # --- NORMALIZATION LAYERS ---
            elif isinstance(
                m, (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.GroupNorm)
            ):
                torch.nn.init.ones_(m.weight)  # Scale to 1
                torch.nn.init.zeros_(m.bias)  # Shift to 0

        self.model.apply(init_weights)
        self.model.to(self.device)
        # self.model.train()
        # self.model = torch.compile(self.model, backend="openxla") # use TorchDynamo (see https://docs.pytorch.org/xla/release/r2.8/perf/dynamo.html)
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
        else:
            self.train_sampler.set_epoch(self.epoch)

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
                loss = self.loss_fn(pred, y)

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

            batch_loss = loss.item()
            batch_loss = xm.mesh_reduce("batch_loss", batch_loss, np.mean)

            if torch.isnan(loss) or torch.isinf(loss):
                xm.master_print(f"Warning: NaN/Inf loss detected for batch")
                for m in self.model.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        xm.master_print(f"Warning: reset BatchNorm2d")
                        m.reset_running_stats()

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
        val_sdr = 0.0
        # sdr = torch.zeros(len(self.target_sources), dtype=torch.float32, device=self.device)

        cnt  =  0

        with torch.no_grad():
        # with torch.inference_mode():
            for batch_idx, batch in enumerate(self.validation_loader):
                # Move to device
                if self.spectrogram_mode:
                    (x, y, y_waveform) = batch
                else:
                    (x, y) = batch

                with autocast(self.device):
                    outputs = self.model(x, spectrogram_mode=self.spectrogram_mode)
                    loss= self.loss_fn(outputs, y)
                
                sdr = calculate_sdr(outputs, y)
                val_sdr += torch.mean(sdr).item() # todo sdr per stem
                val_loss += loss.item()
                cnt += 1

        val_loss = val_loss / cnt
        val_sdr = val_sdr / cnt
        val_loss = xm.mesh_reduce("val_loss", val_loss, np.mean)
        val_sdr = xm.mesh_reduce("val_sdr", val_sdr, np.mean)

        metrics = {}
        metrics["val_loss"] = val_loss
        metrics["val_sdr"] = val_sdr

        if xm.is_master_ordinal():
            # Save best model
            if metrics['val_loss'] < self.best_metric:  # todo if sdr then '>'
                xm.master_print("Save best model with")
                # self.best_metric = metrics['val_loss']
                # self._save_checkpoint('best_model.pt')

            # self._save_checkpoint('latest_model.pt')
            return metrics
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
