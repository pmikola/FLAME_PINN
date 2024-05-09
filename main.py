import os
import random
import sys
from os import path
import numpy as np
import torch
from torch import nn

from model import PINO
from lerning import teacher


from flameEngine import flame as fl
# Start ................
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2024)
np.random.seed(2024)
random.seed(2024)

f1 = fl.flame_sim(no_frames=1500,frame_skip=20)
model = PINO()
t1 = teacher(model)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# Note: Eon > Era > Period > Epoch
for period in range(0,1):
    t1.fsim = f1
    t1.generate_structure()
    t1.fsim.fuel_dens_modifier = 1/t1.fsim.dt
    t1.fsim.simulate(plot=1)
    # model = t1(dataloader,loss,optimizer,device)





