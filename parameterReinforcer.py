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
    def __init__(self,no_layers,batch_size,modes,action_per_layer,memory_size,device):
        super(Metamorph_parameterReinforcer, self).__init__()
        self.device = device
        self.no_layers = no_layers
        self.batch_size = batch_size
        self.modes = modes
        self.memory_size = memory_size
        self.action_per_layer = action_per_layer
        self.masks = deque(maxlen=self.no_layers*self.action_per_layer)

        self.actions = deque(maxlen=self.memory_size)
        self.states = deque(maxlen=self.memory_size)
        self.losses = deque(maxlen=self.memory_size)
        self.RLosses = deque(maxlen=self.memory_size)
        self.next_actions = deque(maxlen=self.memory_size)
        self.next_states = deque(maxlen=self.memory_size)
        self.rewards = deque(maxlen=self.memory_size)
        self.next_rewards = deque(maxlen=self.memory_size)

        # Definition of target policy function
        self.target_policy = torch.rand(self.batch_size, self.no_layers, self.action_per_layer,requires_grad=True).to(self.device)
        self.pred_policy = torch.rand(self.batch_size, self.no_layers, self.action_per_layer,requires_grad=True).to(self.device)

        # Definition of weights and weights as convolutions in FFT space used in FFTFeatures
        self.weights_data_0 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_0 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))
        self.weights_data_1 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_1 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))
        self.weights_data_2 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_2 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))

        # Definition of output dens layers
        self.lin1 = nn.Linear(self.no_layers,self.modes)
        self.lin2 = nn.Linear(self.modes,self.no_layers*self.action_per_layer)
        self.softmax = nn.LogSoftmax(dim=2)
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
        #x = torch.cat([model_p,self.masks])

        x = self.SpaceTimeFFTFeature(model_p.unsqueeze(0),self.weights_data_0,self.weights_data_fft_0)
        x = self.SpaceTimeFFTFeature(x, self.weights_data_1, self.weights_data_fft_1)
        x = self.SpaceTimeFFTFeature(x, self.weights_data_2, self.weights_data_fft_2)
        x = torch.flatten(x,start_dim=1)
        x = self.activate(self.lin1(x))
        x = self.lin2(x).view(self.batch_size,self.no_layers,self.action_per_layer)
        # x = self.softmax(x)
        return x

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
        model_parameters = torch.zeros((self.no_layers, self.modes)).to(self.device)
        for name, param in model.named_parameters():
            fparam = torch.flatten(param)
            fpshape = fparam.shape[0]
            if fpshape < self.modes:
                fparamWHF = torch.fft.fft(fparam)[:fpshape].real
                model_parameters[i,:fpshape] = fparamWHF.detach().clone()
                i+=1
            else:
                fparamWHF = torch.fft.fft(fparam)[:self.modes].real
                model_parameters[i] = fparamWHF.detach().clone()
                i += 1
        self.states.append(model_parameters.detach())
        return model_parameters

    def save_next_state(self,model):
        i = 0
        model_parameters = torch.zeros((self.no_layers, self.modes)).to(self.device)
        for name, param in model.named_parameters():
            fparam = torch.flatten(param)
            fpshape = fparam.shape[0]
            if fpshape < self.modes:
                fparamWHF = torch.fft.fft(fparam)[:fpshape].real
                model_parameters[i,:fpshape] = fparamWHF.detach().clone()
                i+=1
            else:
                fparamWHF = torch.fft.fft(fparam)[:self.modes].real
                model_parameters[i] = fparamWHF.detach().clone()
                i += 1
        self.next_states.append(model_parameters.detach())
        return model_parameters

    def create_masks(self,model):
        for (name, param) in model.named_parameters():
            for i in range(self.action_per_layer):
                self.masks.append(torch.rand_like(param.detach()))

    def weight_mutation(self,model,action):
        i = 0
        p_action = torch.exp(action.detach())
        p_action_value, p_action_idx = torch.max(p_action, dim=-1)
        for (name, param) in model.named_parameters():
            p = param * self.masks[self.action_per_layer*i+p_action_idx[:,i]]
            param.copy_(p)
            i += 1
        return model

    def save_action(self,action):
        self.actions.append(action.detach())

    def save_next_action(self,action):
        self.next_actions.append(action.detach())

    def calculate_reward(self,loss,MLoss):
        reward = torch.tensor([0.]).to(self.device).detach()
        if loss < MLoss:
            reward = reward + 1/1 * (loss/MLoss)

        elif loss == MLoss:
            pass
        else: reward =reward + 1 * (loss/MLoss)
        self.rewards.append(reward.detach())

    def calculate_next_reward(self,loss,MLoss):
        reward = torch.tensor([0.]).to(self.device).detach()
        if loss < MLoss:
            reward = reward + 1/1 * (loss/MLoss)
        elif loss == MLoss:
            pass
        else:
            reward =reward + 1 * (loss / MLoss)
        self.next_rewards.append(reward.detach())

    def PolicyFunctionLoss(self,alpha=0.5,gamma=0.1):
        target_policy = self.target_policy + alpha*(self.next_rewards[-1]+gamma*self.next_actions[-1]- self.actions[-1])
        # self.target_policy.data.copy_(target_policy.detach())
        return target_policy

    def next_to_current(self):
        self.actions,self.next_actions = self.next_actions,self.actions
        self.states, self.next_states = self.next_states, self.states
        self.rewards, self.next_rewards = self.next_rewards, self.rewards


    def activate(self,x):
        return torch.tanh(x)#*2#*self.activation_weight

