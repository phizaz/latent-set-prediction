from trainer.start import *


@dataclass
class OptimizerConfig(BaseConfig):
    optimizer: str = 'adam'
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    wd: float = 0
    lr_term: float = 1e-6
    scheduler: str = 'rop'
    rop_patience: int = 1
    rop_factor: float = 0.2
    best_metric: str = 'val_loss'
    best_mode: str = 'min'
    n_ep_eval_cycle: float = 1

    @property
    def name(self):
        name = f'{self.optimizer}{self.lr}beta({self.beta1},{self.beta2})wd{self.wd}'
        if self.scheduler is None:
            pass
        elif self.scheduler == 'rop':
            name += f'-{self.scheduler}pat{self.rop_patience}fac{self.rop_factor}'
        else:
            raise NotImplementedError()
        name += f'_best{self.best_metric}'
        return name

    def make_opt(self, net):
        if self.optimizer == 'adam':
            return optim.Adam(net.parameters(),
                              lr=self.lr,
                              betas=(self.beta1, self.beta2),
                              weight_decay=self.wd)
        else:
            raise NotImplementedError()

    def make_scheduler(self):
        callbacks = []
        if self.scheduler is None:
            pass
        elif self.scheduler == 'rop':
            callbacks += [
                LRReducePlateauCb(self.best_metric,
                                  self.best_mode,
                                  n_ep_cycle=self.n_ep_eval_cycle,
                                  patience=self.rop_patience,
                                  factor=self.rop_factor)
            ]
        else:
            raise NotImplementedError()

        if self.lr_term > 0:
            callbacks.append(TerminateLRCb(self.lr_term))

        return callbacks


@dataclass
class AsymOptimizerConfig(OptimizerConfig):
    lr2: float = 2e-4

    @property
    def name(self):
        name = f'asym{self.optimizer}({self.lr},{self.lr2})wd{self.wd}'
        if self.scheduler is None:
            pass
        elif self.scheduler == 'rop':
            name += f'-{self.scheduler}pat{self.rop_patience}fac{self.rop_factor}'
        else:
            raise NotImplementedError()
        name += f'_best{self.best_metric}'
        return name

    def make_opt(self, net):
        if self.optimizer == 'adam':
            lr_params = {
                'params': net.enc_parameters(),
            }
            lr2_params = {
                'params': net.dec_parameters(),
                'lr': self.lr2,
            }
            return optim.Adam([lr_params, lr2_params],
                              lr=self.lr,
                              weight_decay=self.wd)
        else:
            raise NotImplementedError()
