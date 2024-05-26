import os
import random
import sys
from os import path
import numpy as np
import torch
from torch import nn

from model import Metamorph
from teacher import teacher


from flameEngine import flame as fl
# Start ................
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2024)
np.random.seed(2024)
random.seed(2024)

no_frame_samples = 15
batch_size = 256
input_window_size = 4

no_frames = 1000
first_frame,last_frame,frame_skip = 0,no_frames,10
model = Metamorph(no_frame_samples, batch_size, input_window_size, device).to(device)
t = teacher(model, device)
t.fsim = fl.flame_sim(no_frames=no_frames,frame_skip=frame_skip)
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)

# Note: Eon > Era > Period > Epoch
no_periods = 1
t.no_of_periods = no_periods
# t.fsim.simulate(simulate=0,delete_data=1)
for period in range(1,no_periods+1):
    t.period = period
    t.fsim = fl.flame_sim(no_frames=1000,frame_skip=frame_skip)
    t.fsim.igni_time = no_frames
    t.generate_structure()
    t.fsim.fuel_dens_modifier = 1/t.fsim.dt
    t.fsim.simulate(simulate=0,save_rgb=1,save_alpha=1,save_fuel=1,delete_data=0)
    t.learning_phase(no_frame_samples, batch_size, input_window_size, first_frame,
                     last_frame,frame_skip*2,criterion,optimizer,device,learning=0,num_epochs=1000)
    # t.fsim.simulate(simulate=0,delete_data=1)

# t.visualize_lerning()
#
t.examine(criterion,device)










