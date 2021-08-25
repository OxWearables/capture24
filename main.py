import os
import random
import warnings
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import models
from augmentation import Augment
import utils


class ModelModule(pl.LightningModule):

    def __init__(self, model_cfg, optim_cfg):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg

        self.model = models.Resnet(
            model_cfg["n_channels"],
            model_cfg["outsize"],
            model_cfg["n_filters"],
            model_cfg["kernel_size"],
            model_cfg["n_resblocks"],
            model_cfg["resblock_kernel_size"],
            model_cfg["downfactor"],
            model_cfg["downorder"],
            model_cfg["drop1"],
            model_cfg["drop2"],
            model_cfg["fc_size"])

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.metrics_fn = utils.Metrics()

        self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
        self.is_swa_started = False

    def configure_optimizers(self):
        optim_cfg = self.optim_cfg

        self.num_batches_per_epoch = len(self.train_dataloader())

        if optim_cfg["method"] == 'adam':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optim_cfg["adam"]["lr"],
                amsgrad=optim_cfg["adam"]["amsgrad"])

            return optimizer

        elif optim_cfg["method"] == 'sgd':

            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=optim_cfg["sgd"]["lr"])

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=optim_cfg["cosine_annealing"]["T_0"],
                T_mult=optim_cfg["cosine_annealing"]["T_mult"],
                eta_min=optim_cfg["cosine_annealing"]["eta_min"])

            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def training_step(self, batch, batch_idx):

        x, target = batch

        optimizer = self.optimizers()

        y = self.model(x)
        loss = self.loss_fn(y, target)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if not self.is_swa_started:
            if self.optim_cfg["method"] == 'sgd':
                self.lr_schedulers().step(self.global_step / self.num_batches_per_epoch)

        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True)

    def training_epoch_end(self, outputs=None) -> None:
        self.is_swa_started = self.current_epoch + 1 >= self.optim_cfg["swa"]["start"]
        if self.is_swa_started:
            self._update_swa_model()
            if self.optim_cfg["method"] == 'sgd':
                self._adjust_learning_rate(self.optim_cfg["swa"]["lr"])

    def validation_step(self, batch, batch_idx):
        self._validation_test_step(batch, batch_idx, stage="valid")

    def validation_epoch_end(self, outputs=None):
        self._validation_test_epoch_end(outputs, stage="valid")

    def test_step(self, batch, batch_idx):
        self._validation_test_step(batch, batch_idx, stage="test")

    def test_epoch_end(self, outputs=None):
        self._validation_test_epoch_end(outputs, stage="test")

    def _validation_test_step(self, batch, batch_idx, stage=""):
        x, target = batch
        y = self(x)
        loss = self.loss_fn(y, target)
        self.log(f"{stage}/loss", loss)
        self.metrics_fn(y, target)
        return y, target

    def _validation_test_epoch_end(self, outputs=None, stage=""):
        scores = self.metrics_fn.compute()
        self.metrics_fn.reset()
        self.log(f"{stage}/f1", scores['f1'], prog_bar=True)
        self.log(f"{stage}/kappa", scores['kappa'], prog_bar=True)
        self.log(f"{stage}/phi", scores['phi'], prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        y = self(x)
        return y

    def forward(self, x):
        if self.is_swa_started:
            y = self.swa_model(x)
        else:
            y = self.model(x)
        return y

    def _update_swa_model(self):
        self.swa_model.update_parameters(self.model)

        def _loader():
            for batch in self.train_dataloader():
                x = batch[0].to(self.device)
                yield x

        with torch.no_grad(), torch.cuda.amp.autocast():
            torch.optim.swa_utils.update_bn(_loader(), self.swa_model)

    def _adjust_learning_rate(self, lr):
        for param_group in self.optimizers().optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class DataModule(pl.LightningDataModule):

    def __init__(self, data_cfg, dataloader_cfg, augment_cfg):
        super().__init__()

        self.data_cfg = data_cfg
        self.dataloader_cfg = dataloader_cfg
        self.augment_cfg = augment_cfg

        self.transform = Augment(
            augment_cfg["jitter"]["sigma"], augment_cfg["jitter"]["prob"],
            augment_cfg["shift"]["window"], augment_cfg["shift"]["prob"],
            augment_cfg["twarp"]["sigma"], augment_cfg["twarp"]["knots"], augment_cfg["twarp"]["prob"],
            augment_cfg["mwarp"]["sigma"], augment_cfg["mwarp"]["knots"], augment_cfg["mwarp"]["prob"],
        )

    def setup(self, stage=None):

        data_cfg = self.data_cfg

        X = np.load(os.path.join(data_cfg['datadir'], 'X.npy'))
        Y = np.load(os.path.join(data_cfg['datadir'], 'Y_willetts.npy'))
        pid = np.load(os.path.join(data_cfg['datadir'], 'pid.npy'))

        # deriv/test split: P001-P100 for derivation, the rest for testing
        whr_deriv = np.isin(pid, [f"P{i:03d}" for i in range(1,101)])
        X_deriv, Y_deriv, pid_deriv = X[whr_deriv], Y[whr_deriv], pid[whr_deriv]
        X_test, Y_test, pid_test = X[~whr_deriv], Y[~whr_deriv], pid[~whr_deriv]

        # further split deriv into train/val
        whr_val = np.isin(pid_deriv, np.random.choice(np.unique(pid_deriv), size=data_cfg['val_size']))
        X_val, Y_val, pid_val = X_deriv[whr_val], Y_deriv[whr_val], pid_deriv[whr_val]
        X_train, Y_train, pid_train = X_deriv[~whr_val], Y_deriv[~whr_val], pid_deriv[~whr_val]

        self.dataset_train = Dataset(X_train, Y_train, transform=self.transform)
        self.dataset_valid = Dataset(X_val, Y_val)
        self.dataset_test = Dataset(X_test, Y_test)

    def setup_predict(self, X=None, X_path=None):
        if X_path is not None:
            X = np.load(X_path)
        self.dataset_predict = Dataset(X)

    def train_dataloader(self):
        return DataModule.create_dataloader(self.dataset_train,
                                            **self.dataloader_cfg["train"],
                                            deterministic=False)

    def val_dataloader(self):
        return DataModule.create_dataloader(self.dataset_valid,
                                            **self.dataloader_cfg["valid"],
                                            deterministic=True)

    def test_dataloader(self):
        return DataModule.create_dataloader(self.dataset_test,
                                            **self.dataloader_cfg["test"],
                                            deterministic=True)

    def predict_dataloader(self):
        return DataModule.create_dataloader(self.dataset_predict,
                                            **self.dataloader_cfg["predict"],
                                            deterministic=True)

    @staticmethod
    def create_dataloader(dataset, seed=12345, batch_size=64, num_workers=1, deterministic=False):

        if deterministic:

            if not num_workers > 0:
                warnings.warn("Deterministic dataloader with num_workers=0 is not supported yet")

            def worker_init_fn(worker_id):
                """ Always start with same seed"""
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

        else:

            def worker_init_fn(worker_id):
                """ Ensure external RNG is randomly seeded """
                seed = torch.initial_seed() % (2**32)
                np.random.seed(seed)
                random.seed(seed)

        dataloader = DataLoader(
            dataset,
            shuffle=(not deterministic),
            batch_size=batch_size,
            worker_init_fn=worker_init_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return dataloader


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, Y=None, transform=None):

        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform is not None:
            x = self.transform(x)
        x = x.T  # channels first
        if self.Y is not None:
            y = self.Y[index]
            y = Dataset.to_class_idx(y)
            return x, y
        return x

    @staticmethod
    def to_class_idx(label):
        return [
            "sleep",
            "sit-stand",
            "vehicle",
            "walking",
            "mixed",
            "bicycling"
        ].index(label)


def create_lightning_modules(cfg: DictConfig):
    """ Create the data and model modules """

    # Data module
    datamodule = DataModule(cfg.data, cfg.dataloader, cfg.augment)

    # Model
    if cfg.ckpt_path is not None:
        model = ModelModule.load_from_checkpoint(
            cfg.ckpt_path,
            model_cfg=cfg.model,
            optim_cfg=cfg.optim)

    else:
        model = ModelModule(cfg.model, cfg.optim)

    return datamodule, model


def resolve_cfg_paths(cfg: DictConfig):
    cfg.data.datadir = os.path.expanduser(cfg.data.datadir)
    if cfg.ckpt_path is not None:
        cfg.ckpt_path = os.path.expanduser(cfg.ckpt_path)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    resolve_cfg_paths(cfg)

    # Lightning modules
    datamodule, model = create_lightning_modules(cfg)

    # Trainer
    early_stop_callback = EarlyStopping(monitor='valid/loss', patience=cfg.early_stop_patience, mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='valid/loss', mode='min', filename='best')
    trainer = pl.Trainer(gpus=1, auto_select_gpus=True,
                         precision=16,
                         max_epochs=cfg.n_epochs,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         log_every_n_steps=10,
                         num_sanity_val_steps=0,
                         limit_train_batches=cfg.limit_train_batches,
                         limit_val_batches=cfg.limit_val_batches,
                         limit_test_batches=cfg.limit_test_batches,
                         )

    if cfg.fit:
        trainer.fit(model, datamodule)

    if cfg.test:
        trainer.test(datamodule=datamodule)

    return


if __name__ == '__main__':
    main()
