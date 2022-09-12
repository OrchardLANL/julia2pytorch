import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from torch.autograd import grad

from pytorch_lightning.core.lightning import LightningModule

from src.setup_strings import singlephase_setup
from src.Julia_coupling import julia2pytorch


class DPmodel(LightningModule):
    def __init__(self,
                 domain_size=[256, 260], dx=0.0001,
                 LR=1e-4, batch_size=1):
        """
        Parameters
        ----------
        domain_size : TYPE, optional
            computational size of the domain. Currently, only 2D domains are supported.
            The default is [256, 260].
        dx : TYPE, optional
            resolution of the grid. The default is 0.0001.
        LR : TYPE, optional
            learning rate. The default is 1e-4.
        batch_size : TYPE, optional
            batch size. The default is 1.

        Returns
        -------
        None.

        """

        super().__init__()

        # setup DPFEHM
        setup_code = singlephase_setup(domain_size[::-1], dx)
        self.p_solver = julia2pytorch("solveforp", setup_code)

        # init DPFEHM
        x = torch.zeros(torch.multiply(*domain_size)).requires_grad_(True)
        p = self.p_solver(x)
        _ = grad(p.sum(), [x])[0]
        print('Python-Julia-DPFEHM coupling initialized correctly \n')
        
        # init an array of weights
        w_init = torch.ones(1, 1, *domain_size)*0.5
        self.weights = nn.Parameter(w_init,requires_grad=True)
        
        self.LR = LR
        self.batch_size = batch_size
        self.domain_size = domain_size
        # turn-off since we are getting the gradients from Julia
        self.automatic_optimization = False

    def forward(self, x):

        x = torch.log(torch.square(x))*F.relu(self.weights)
        # Julia is column major (as opposed to python)
        x = self.p_solver(x).reshape(self.domain_size[::-1]).T[None, None, ]

        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.LR)

    def training_step(self, batch, batch_idx):

        x = torch.as_tensor(batch[0][None,])
        y = torch.as_tensor(batch[1][None,])

        opt = self.optimizers()
        opt.zero_grad()

        yhat = self(x)

        loss = (y-yhat).pow(2).mean() # MSE

        self.manual_backward(loss)

        opt.step()

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=self.batch_size)


def load_data(file_loc):
    import h5py

    all_fracs = {}
    hf = h5py.File(file_loc, 'r')
    with hf as f:
        for key in f.keys():
            all_fracs[key] = {}
            for k in f[key].keys():
                all_fracs[key][k] = f[key][k][:]

    return all_fracs
