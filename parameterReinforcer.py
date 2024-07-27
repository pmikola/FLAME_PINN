import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.nn.functional as f


class Metamorph_parameterReinforcer(nn.Module):
    # Note : Buzzword - Metamorph will be better name here probably :) or HIPNO
    def __init__(self,device):
        super(Metamorph_parameterReinforcer, self).__init__()
        self.device = device
        self.modes = 1000

        # Definition of weights and weights as convolutions in FFT space used in FFTFeatures
        self.weights_data_0 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_0 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))
        self.weights_data_1 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_1 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))
        self.weights_data_2 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_2 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))

        # Definition of output dens layers
        self.lin1 = nn.Linear(self.modes,self.modes)
        self.lin2 = nn.Linear(self.modes,169)

        self.init_weights()

    def init_weights(self):
        if isinstance(self, nn.Linear):
            # torch.nn.init.xavier_uniform(self.weight)
            self.weight.data.normal_(mean=0.0, std=1.0)
            self.bias.data.fill_(0.01)

        if isinstance(self, nn.Conv1d):
            self.weight.data.normal_(mean=0.0, std=1.0)
            self.bias.data.fill_(0.01)

        if isinstance(self, nn.Parameter):
            self.data.normal_(mean=0.0, std=1.0)

    def weight_reset(self: nn.Module):
        reset_parameters = getattr(self, "reset_parameters", None)
        if callable(reset_parameters):
            self.reset_parameters()
        # NOTE : Making sure that nn.conv2d and nn.linear will be reset
        if isinstance(self, nn.Conv2d) or isinstance(self, nn.Linear):
            self.reset_parameters()

    def forward(self, params):
        x = self.SpaceTimeFFTFeature(params,self.weights_data_0,self.weights_data_fft_0)
        x = self.activate(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        return x

    def shapeShift(self,x, h):
        if x.dim() == 3:
            coefficients = h.reshape(self.batch_size,x.shape[1],x.shape[2],self.shifterCoefficients)
            x_powers = torch.pow(x[0:self.batch_size,:,:].unsqueeze(3), self.exponents.unsqueeze(0).unsqueeze(1))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=3)
            craftedPolynomial = self.activate(craftedPolynomial)
            return craftedPolynomial
        elif x.dim() == 2:
            coefficients = h.reshape(self.batch_size,x.shape[1],self.shifterCoefficients)
            x_powers = torch.pow(x[0:self.batch_size, :].unsqueeze(2), self.exponents.unsqueeze(0).unsqueeze(1))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=2)
            craftedPolynomial = self.activate(craftedPolynomial)
            return craftedPolynomial
        elif x.dim() == 4:
            coefficients = h.reshape(self.batch_size, x.shape[1], x.shape[2], x.shape[3], self.shifterCoefficients)
            x_powers = torch.pow(x[0:self.batch_size, :, :, :].unsqueeze(4),self.exponents.unsqueeze(0).unsqueeze(1).unsqueeze(2))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=4)
            craftedPolynomial = self.activate(craftedPolynomial)
            return craftedPolynomial
        else:
            raise ValueError("Unsupported input dimensions")

    def SpaceTimeFFTFeature(self,data,weights_data,weights_data_fft):
        # Attention :  Below is implemented simplified FNO LAYER
        fft_data = torch.fft.fft(data,norm='forward')
        FFTwithW = torch.einsum("ab,xy->by",fft_data.unsqueeze(-1), weights_data_fft)
        iFFW= torch.fft.ifft(FFTwithW, norm='forward')

        data = self.activate(iFFW)
        # Attention :  Above is implemented simplified FNO LAYER
        #dimag = data.imag
        dreal = data.real
        data = dreal
        # data = torch.tanh(data)
        return data

    def activate(self,x):
        return torch.tanh(x)#*2#*self.activation_weight

