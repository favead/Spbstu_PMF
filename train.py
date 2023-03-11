from typing import Callable, Dict, Any
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR


class GlobalTrainer:
    def __init__(self, model: nn.Module, optim: Optimizer, loss_func: Callable, train_loader: DataLoader,
                 val_loader: DataLoader, wandb: object, device: torch.device, config: Dict[str, Any],
                 tqdm: Callable, scheduler: StepLR = None):
        self.model = model
        self.optim = optim
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wandb = wandb
        self.device = device
        self.config = config
        self.tqdm = tqdm
        self.scheduler = scheduler

    def fit_epoch(self) -> None:
        pass

    def val_epoch(self) -> None:
        pass

    def handle_checkpoint(self) -> None:
        pass

    def handle_weight_checkpoint(self) -> None:
        pass

    def train(self) -> None:
        for epoch in self.tqdm(range(1, self.config['epochs'] + 1)):
            self.fit_epoch()
            self.val_epoch()
            if self.scheduler:
                self.scheduler.step()
            if (epoch % self.config['checkpoint']) == 0:
                self.handle_checkpoint()
            if (epoch % self.config['weight_checkpoint']) == 0:
                self.handle_weight_checkpoint()
        return None
