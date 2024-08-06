import copy
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.nn.functional as f
from collections import deque
from teacher import teacher


class Metamorph_parameterReinforcer(nn.Module):
    # Note : On-Policy method (for now)
    def __init__(self,no_layers,batch_size,modes,memory_size,device):
        super(Metamorph_parameterReinforcer, self).__init__()
        self.device = device
        self.no_layers = no_layers
        self.batch_size = batch_size
        self.modes = modes
        self.memory_size = memory_size

        self.actions = deque(maxlen=self.memory_size)
        self.states = deque(maxlen=self.memory_size)
        self.losses = deque(maxlen=self.memory_size)
        self.RLosses = deque(maxlen=self.memory_size)
        self.next_actions = deque(maxlen=self.memory_size)
        self.next_states = deque(maxlen=self.memory_size)
        self.rewards = deque(maxlen=self.memory_size)
        self.next_rewards = deque(maxlen=self.memory_size)

        # Definition of weights and weights as convolutions in FFT space used in FFTFeatures
        self.weights_data_0 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_0 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))
        self.weights_data_1 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_1 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))
        self.weights_data_2 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_2 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))

        # Definition of output dens layers
        self.lin1 = nn.Linear(self.no_layers,self.modes)
        self.lin2 = nn.Linear(self.modes,self.no_layers)
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

    def forward(self,model_p):
        x = self.SpaceTimeFFTFeature(model_p.unsqueeze(0),self.weights_data_0,self.weights_data_fft_0)
        x = self.SpaceTimeFFTFeature(x, self.weights_data_1, self.weights_data_fft_1)
        x = self.SpaceTimeFFTFeature(x, self.weights_data_2, self.weights_data_fft_2)
        x = torch.flatten(x,start_dim=1)
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
        FFTwithW = torch.einsum("bfp,mn->bfm",fft_data, weights_data_fft)
        iFFW= torch.fft.ifft(FFTwithW, norm='forward')
        data = self.activate(iFFW)
        # Attention :  Above is implemented simplified FNO LAYER
        #dimag = data.imag
        dreal = data.real
        data = dreal
        # data = torch.tanh(data)
        return data

    def save_state(self,model):
        i = 0
        model_parameters = torch.zeros((self.no_layers, self.modes),
                                   requires_grad=True).to(self.device)
        for name, param in model.named_parameters():
            fparam = torch.flatten(param)
            fpshape = fparam.shape[0]
            if fpshape < self.modes:
                fparamWHF = torch.fft.fft(fparam)[:fpshape].real
                model_parameters[i,:fpshape] = fparamWHF
                i+=1
            else:
                fparamWHF = torch.fft.fft(fparam)[:self.modes].real
                model_parameters[i] = fparamWHF
                i += 1
        self.states.append(model_parameters.detach())
        return model_parameters

    def save_next_state(self,model):
        i = 0
        model_parameters = torch.zeros((self.no_layers, self.modes),
                                   requires_grad=True).to(self.device)
        for name, param in model.named_parameters():
            fparam = torch.flatten(param)
            fpshape = fparam.shape[0]
            if fpshape < self.modes:
                fparamWHF = torch.fft.fft(fparam)[:fpshape].real
                model_parameters[i,:fpshape] = fparamWHF
                i+=1
            else:
                fparamWHF = torch.fft.fft(fparam)[:self.modes].real
                model_parameters[i] = fparamWHF
                i += 1
        self.next_states.append(model_parameters.detach())
        return model_parameters

    def save_action(self,action):
        self.actions.append(action)

    def save_next_action(self,action):
        self.next_actions.append(action)

    def calculate_reward(self,loss,RLoss):
        reward = torch.tensor([0.]).to(self.device)
        if loss < RLoss:
            reward -=2*RLoss/loss
        elif loss == RLoss:
            pass
        else: reward +=2*loss/RLoss
        self.rewards.append(reward.detach())

    def calculate_next_reward(self,loss,RLoss):
        reward = torch.tensor([0.]).to(self.device)
        if loss < RLoss:
            reward = RLoss/loss
        elif loss == RLoss:
            pass
        else: reward +=2*loss/RLoss
        self.next_rewards.append(reward.detach())

    def PolicyFunctionLoss(self,gamma=0.1):
        # TODO: Calculate policy loss function from state action space using output action that
        # TODO : corresponds to number of activations of main neural network (so the output of policy network
        # TODO: should be - softmax(features,no_layers x actions_per_layer)
        # TODO: use differentiable torchtopk algorithm or something similar (maby shapeshift again)
        return loss

    def weight_mutation(self,model,action):
        i = 0
        for (name, param) in model.named_parameters():
            p = param * action[0,i]
            param.copy_(p)
            i += 1
        return model


    def activate(self,x):
        return torch.tanh(x)#*2#*self.activation_weight

