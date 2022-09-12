import numpy as np
import matplotlib.pyplot as plt

from pytorch_lightning import Trainer

from src.lightning_model import DPmodel, load_data


fracture = load_data('src/example_fracture.mat')
aperture = fracture['fracture_1']['aperture_field']
pressure = fracture['fracture_1']['pressure_field']

domain_size = list(aperture.shape[2:])

model = DPmodel(domain_size=domain_size, dx=0.0001, LR=1e-3)
trainer = Trainer(max_epochs=100, log_every_n_steps=1)

trainer.fit(model, [aperture, pressure])
