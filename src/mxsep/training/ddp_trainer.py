# see example https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py


import random
import time
from pathlib import Path
from typing import Dict, Any
import hydra
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from mxsep.cfg import Config
from mxsep.data.dataset import PredefinedMixDataset
from mxsep.models import ISTFTModule, MusicSourceSeparationModel
from mxsep.training import Monitor
from mxsep.utils.metrics import calculate_sdr


class DDPTrainer:
    """Main training class"""

    def __init__(self, rank: int, world_size: int, cfg: Config):
        assert cfg.model
        assert cfg.training.optimizer

        self.rank = rank
        self.world_size = world_size

        assert cfg.training.device == 'cuda'
        assert torch.cuda.is_available()

        self.device = torch.device(f'cuda:{self.rank}')


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

        if self.use_amp:
            self.scaler = GradScaler()

        if cfg.dataset.train.predefined_jsonl_path:

            if self.spectrogram_mode:
                self.train_dataset = PredefinedMixDataset(cfg.dataset, split='train', stft_cfg=cfg.model.stft)

            else:
                self.train_dataset = PredefinedMixDataset(cfg.dataset, split='train')

            shuffle = Path(cfg.dataset.train.predefined_jsonl_path).suffix == '.jsonl'  # if single json file per epoch we should shuffle.
            sampler = DistributedSampler(dataset=self.train_dataset, shuffle=shuffle, seed=self.seed,  drop_last=True)
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=cfg.training.batch_size, shuffle=False, sampler=sampler, pin_memory=True,
                                           num_workers=cfg.training.num_workers)
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

                sampler = DistributedSampler(dataset=self.validation_dataset, shuffle=False, seed=self.seed,  drop_last=True)
                self.validation_loader = DataLoader(self.validation_dataset, sampler=sampler, batch_size=cfg.training.batch_size, shuffle=False,
                                               num_workers=cfg.training.num_workers, drop_last=True)
            else:
                raise NotImplementedError("Only predefined mix dataset is implemented for now")
        else:
            self.validation_loader = None


        self.init_randomizer()

        self.model = MusicSourceSeparationModel(cfg.model)
        self.model.to(self.device)
        
        self.optimizer = hydra.utils.instantiate(cfg.training.optimizer, _partial_=True)(params=self.model.parameters())
        if cfg.training.lr_scheduler:
            self.lr_scheduler = hydra.utils.instantiate(cfg.training.lr_scheduler, _partial_=True)(
                optimizer=self.optimizer)
        else:
            self.lr_scheduler = None

        # Trackers
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float("-inf") # todo if metric = sdr then -inf else +inf

        if cfg.training.resume_from_checkpoint:
            self._load_checkpoint(cfg.training.resume_from_checkpoint)
            self.epoch += 1

        self.model = DDP(self.model, device_ids=[self.rank])

        self.loss_fn = hydra.utils.instantiate(cfg.training.loss)

        if self.rank == 0:
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

        # If using CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)  # for multi-GPU

        # Optional: make CuDNN deterministic (slower but reproducible)
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Randomizer initialized with seed: {self.seed}")

    def train_epoch(self) -> dict[str, float] | None:
        """Train for one epoch"""
        self.train_dataset.init_epoch(self.epoch)  # Load new mixes for this epoch
        self.train_loader.sampler.set_epoch(self.epoch)
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
                y = y.to(self.device)

            x = x.to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp, device_type=self.device.type, dtype=torch.float16):
                pred = self.model(x, spectrogram_mode=self.spectrogram_mode)
                # loss = self.loss_fn(torch.view_as_real(pred).contiguous(), torch.view_as_real(y).contiguous() )
                if self.spectrogram_mode:
                    pred = pred.cpu()
                    pred = self.istft(pred)
                    y = y_waveform

                assert pred.shape == y.shape
                loss = self.loss_fn(pred, y)
            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp and self.device.type == 'cuda':
                self.scaler.scale(loss).backward()

                # Gradient clipping (see https://docs.pytorch.org/docs/main/notes/amp_examples.html#gradient-clipping)
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                self.optimizer.step()

            # Update scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Update trackers
            steps += 1
            self.global_step += 1
                
            dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)

            # Logging
            # if self.global_step % self.monitor.log_metrics_interval == 0:
            #     metrics = self._compute_metrics(outputs, outputs, y)
            # else :
            #     metrics = {}


            if self.rank == 0:
                batch_loss = loss.item()
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

        if self.rank == 0:
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

        with torch.inference_mode():
            for batch_idx, batch in enumerate(self.validation_loader):
                # Move to device
                if self.spectrogram_mode:
                    (x, y, y_waveform) = batch
                else:
                    (x, y) = batch

                x = x.to(self.device)
                y = y.to(self.device)

                with autocast(enabled=self.use_amp, device_type=self.device.type, dtype=torch.float16):
                    outputs = self.model(x, spectrogram_mode=self.spectrogram_mode)
                    loss = self.loss_fn(outputs, y)
                
                # Compute metrics
                if self.spectrogram_mode:
                    outputs = self.istft(outputs.detach().cpu())
                    y = y_waveform

                if self.rank == 0:
                    all_outputs = [
                        torch.zeros_like(outputs, device=self.device) for i in range(self.world_size)
                    ]
                    all_y  = [
                        torch.zeros_like(y, device=self.device) for i in range(self.world_size)
                    ]
                    all_losses = [
                        torch.zeros_like(loss, device=self.device) for i in range(self.world_size)
                    ]
                else:
                    all_outputs = None
                    all_y = None
                    all_losses = None

                dist.gather(outputs, gather_list=all_outputs, dst=0)
                dist.gather(outputs, gather_list=all_y, dst=0)
                dist.gather(loss, gather_list=all_losses, dst=0)

                if self.rank == 0:
                    for (output, y, loss) in zip(all_outputs, all_y, all_losses):
                        metrics = self._compute_metrics(outputs, y)
                        metrics = {f'val_{k}':v for k, v in metrics.items()}
                        val_loss = loss.item()
                        metrics["val_loss"] = val_loss
                        for key, value in metrics.items():
                            if key not in all_metrics:
                                all_metrics[key] = []
                            all_metrics[key].append(value)

                    avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
                    data = {
                        **avg_metrics,
                        "batch_idx": batch_idx,
                    }
                    self.monitor.log_data(data)


        if self.rank == 0:
            # Average metrics
            avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}

            # Save best model
            if avg_metrics['val_sdr'] > self.best_metric: #todo if not sdr then '<'
                self.best_metric = avg_metrics['val_sdr']
                self._save_checkpoint('best_model.pt')
    
            self._save_checkpoint('latest_model.pt')

            return avg_metrics
        
        else :
            return None

    # def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    #     """Prepare batch for training"""
    #     # Move mix to device
    #     batch['mix'] = batch['mix'].to(self.device)
    #
    #     # Move each source to device
    #     for source_name in batch['sources']:
    #         batch['sources'][source_name] = batch['sources'][source_name].to(self.device)
    #
    #     return batch

    # def _compute_loss(
    #         self,
    #         outputs: Dict[str, torch.Tensor],
    #         targets: Dict[str, torch.Tensor]
    # ) -> torch.Tensor:
    #     """Compute loss function"""
    #     loss = 0.0
    #
    #     for source_name in outputs:
    #         if source_name in targets:
    #             # Waveform loss
    #             if self.config.waveform_loss_weight > 0:
    #                 if self.config.loss_function == 'l1':
    #                     wave_loss = torch.nn.functional.l1_loss(
    #                         outputs[source_name], targets[source_name]
    #                     )
    #                 elif self.config.loss_function == 'l2':
    #                     wave_loss = torch.nn.functional.mse_loss(
    #                         outputs[source_name], targets[source_name]
    #                     )
    #                 elif self.config.loss_function == 'si-sdr':
    #                     wave_loss = -calculate_sisdr(
    #                         outputs[source_name], targets[source_name]
    #                     ).mean()
    #                 else:
    #                     wave_loss = torch.tensor(0.0, device=self.device)
    #
    #                 loss += self.config.waveform_loss_weight * wave_loss
    #
    #             # STFT loss (if model outputs spectrograms)
    #             if self.config.stft_loss_weight > 0 and hasattr(self.model, 'stft'):
    #                 with autocast(enabled=False):
    #                     output_spec = self.model.stft(outputs[source_name])
    #                     target_spec = self.model.stft(targets[source_name])
    #                     stft_loss = torch.nn.functional.l1_loss(
    #                         output_spec, target_spec
    #                     )
    #                     loss += self.config.stft_loss_weight * stft_loss
    #
    #     return loss

    def _compute_metrics(
            self,
            outputs: torch.Tensor,
            targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute evaluation metrics"""
        metrics = {}

        sdr = calculate_sdr(outputs, targets, self.target_sources)
        metrics = {**sdr}


        # SI-SDR
        # sisdr = calculate_sisdr(outputs[source_name], targets[source_name])
        # metrics[f'{source_name}_sisdr'] = sisdr.mean().item()

        return metrics


    def _load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint_path = Path(self.checkpoint_dir) / filename
        if not checkpoint_path.exists():
            print(f"Checkpoint {filename} not found, starting from scratch.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # if self.scheduler and checkpoint['scheduler_state_dict']:
        #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', float('inf'))

        print(f"Loaded checkpoint {filename} (epoch {self.epoch}, global step {self.global_step})")


    def _save_checkpoint(self, filename: str = 'checkpoint.pt'):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.model.module.config,
        }
        print(f"Save checkpoint {filename}")

        torch.save(checkpoint, checkpoint_dir / filename)

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
            if self.rank == 0:
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
