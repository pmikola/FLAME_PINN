import sys
from os import path
import torch

from model import PINO
from lerning import learning_loop
sys.path.append(r'C:/PRACA/ThreeSixty/Projects/flameEngine')
from ..flameEngine import flame as fl
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# f1 = fl.flame_sim()

