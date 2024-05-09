import os
import random
import sys
from os import path
import numpy as np
import torch
from torch import nn

from model import PINO
from teacher import teacher


from flameEngine import flame as fl
# Start ................
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2024)
np.random.seed(2024)
random.seed(2024)



model = PINO()
t1 = teacher(model,device)
first_frame,last_frame,frame_skip = 0,200,25
f1 = fl.flame_sim(no_frames=1000,frame_skip=frame_skip)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# Note: Eon > Era > Period > Epoch
for period in range(0,1):
    t1.fsim = f1
    t1.generate_structure()
    t1.fsim.fuel_dens_modifier = 1/t1.fsim.dt
    t1.fsim.simulate(simulate=0,save_rgb=1,save_alpha=1,save_fuel=1,delete_data=0)
    t1.data_preparation(first_frame,last_frame,frame_skip)
    #t1.learning_loop() # TODO : push data trough network





