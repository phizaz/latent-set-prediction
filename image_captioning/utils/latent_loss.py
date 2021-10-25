from trainer.start import *
from .assignment import *
from utils.utils import *


@dataclass
class LatentLossReturn:
    R_pi: Tensor
    R_i: Tensor
    loss: Tensor
    loss_RB: Tensor
    loss_BR: Tensor


@dataclass
class MSEGCRLatentLossConfig(BaseConfig):
    w_loss_br: float = 1
    w_loss_rb: float = 0.1
    loss: str = 'mse'
    gcr_mode: str = 'gcr'
    safe_coef: float = 0
    equal_instance_weight: bool = True

    @property
    def name(self):
        name = f'{self.loss}'
        if self.gcr_mode == 'none':
            name += f'-nogc'
        else:
            name += f'-{self.gcr_mode}'
            if self.gcr_mode == 'gcr':
                name += f'-d{self.safe_coef}'
        name += f'-w({self.w_loss_br},{self.w_loss_rb})'
        if self.equal_instance_weight:
            name += '-eq'
        return name

    def make_loss(self):
        return MSEGCRLatentLoss(self)


class MSEGCRLatentLoss:
    def __init__(self, conf: MSEGCRLatentLossConfig):
        self.conf = conf

    def forward(self, B, len_B, R, len_R):
        func = batch_hungarian_gcr(gcr_mode=self.conf.gcr_mode,
                                   safe_coef=self.conf.safe_coef)
        R_pi, B, R, R_i = func(B, len_B, R, len_R)

        loss_options = {
            'mse': F.mse_loss,
            'l1': F.l1_loss,
            'huber': F.smooth_l1_loss,
        }
        loss_fn = loss_options[self.conf.loss]

        # R => B must not push gradient to B
        loss_RB = loss_fn(R[R_i], B.detach(), reduction='none')
        loss_BR = loss_fn(B, R_pi.detach(), reduction='none')

        if self.conf.equal_instance_weight:
            loss_RB = mean_equal_by_instance(loss_RB, len_B)
            loss_BR = mean_equal_by_instance(loss_BR, len_B)
        else:
            loss_RB = loss_RB.mean()
            loss_BR = loss_BR.mean()

        loss = self.conf.w_loss_rb * loss_RB + self.conf.w_loss_br * loss_BR

        return LatentLossReturn(
            R_pi=R_pi,
            R_i=R_i,
            loss=loss,
            loss_BR=loss_BR,
            loss_RB=loss_RB,
        )
