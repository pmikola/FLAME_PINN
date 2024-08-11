import copy
import random
import time
from statistics import mean

import numpy as np
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
        self.MLosses = deque(maxlen=self.memory_size)
        self.next_losses = deque(maxlen=self.memory_size)
        self.next_MLosses = deque(maxlen=self.memory_size)
        self.next_actions = deque(maxlen=self.memory_size)
        self.next_states = deque(maxlen=self.memory_size)
        self.rewards = deque(maxlen=self.memory_size)
        self.next_rewards = deque(maxlen=self.memory_size)
        self.reward = torch.tensor([0.]).to(self.device)
        self.next_reward = torch.tensor([0.]).to(self.device)

        # Definition of target policy function
        self.q_target = torch.rand(self.batch_size, self.no_layers, self.action_per_layer,requires_grad=True).to(self.device)

        # Definition of weights and weights as convolutions in FFT space used in FFTFeatures
        self.weights_data_0 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_0 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))
        self.weights_data_1 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_1 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))
        self.weights_data_2 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_2 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))

        # Definition of output dens layers
        self.lin1 = nn.Linear(self.no_layers*self.modes,self.modes)
        self.lin2 = nn.Linear(self.modes,self.no_layers*self.action_per_layer)
        self.softmax = nn.Softmax(dim=2)
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
        if model_p.shape[1] == self.modes:
            model_p = model_p.unsqueeze(0)
        else:pass

        self.batch_size = model_p.shape[0]
        x = self.SpaceTimeFFTFeature(model_p,self.weights_data_0,self.weights_data_fft_0)
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
        FFTwithW = torch.einsum("bfp,an->bfn",fft_data, weights_data_fft)
        iFFW= torch.fft.ifft(FFTwithW, norm='forward')
        data = self.activate(iFFW.real)
        # Attention :  Above is implemented simplified FNO LAYER
        # data = torch.tanh(data)
        return data

    def save_state(self, model):
        i = 0
        model_parameters = torch.zeros((self.no_layers, self.modes)).to(self.device)
        for name, param in model.named_parameters():
            fparam = torch.flatten(param)
            fpshape = fparam.shape[0]
            if fpshape < self.modes:
                fparamWHF = torch.fft.fft(fparam)[:fpshape].real
                model_parameters[i, :fpshape] = fparamWHF.detach().clone()
                i += 1
            else:
                fparamWHF = torch.fft.fft(fparam)[:self.modes].real
                model_parameters[i] = fparamWHF.detach().clone()
                i += 1
        self.states.append(model_parameters.detach())
        return model_parameters

    def save_next_state(self, model):
        i = 0
        model_parameters = torch.zeros((self.no_layers, self.modes)).to(self.device)
        for name, param in model.named_parameters():
            fparam = torch.flatten(param)
            fpshape = fparam.shape[0]
            if fpshape < self.modes:
                fparamWHF = torch.fft.fft(fparam)[:fpshape].real
                model_parameters[i, :fpshape] = fparamWHF.detach().clone()
                i += 1
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

    def mutate_parameters(self, model, action):

        p_action = action
        if p_action.shape[0] > 1:
            p_a_values  = torch.mean(p_action, dim=0)
            p_action_idx = torch.argmax(p_a_values, dim=-1).unsqueeze(0).unsqueeze(-1)
        else:
            p_action_idx = torch.argmax(p_action, dim=-1)
        i = 0

        for (name, param) in model.named_parameters():
            p = param * self.masks[self.action_per_layer*i+p_action_idx[:,i]]
            param.copy_(p)
            i += 1
        return model

    def exploit_explore_action_selector(self,action,p=0.3):
        selector = torch.randint(1,10,(1,))
        if  selector > 10*p:
            action = torch.rand(self.batch_size, self.no_layers, self.action_per_layer,requires_grad=True).to(self.device)
        else:
            pass
        return action

    def save_action(self,action):
        self.actions.append(action.detach())

    def save_next_action(self,action):
        self.next_actions.append(action.detach())

    def save_losses(self,loss,MLoss):
        self.losses.append(loss.detach())
        self.MLosses.append(MLoss.detach())

    def save_next_losses(self,loss,MLoss):
        self.next_losses.append(loss.detach())
        self.next_MLosses.append(MLoss.detach())

    def calculate_reward(self,loss,MLoss,reiterate=1):
        if not reiterate:
            nr_mean = torch.tensor(mean(self.reward)).to(self.device)
            if self.rewards[-1] < nr_mean:
                self.rewards[-1] -= 100
            self.reward = torch.tensor([0.]).to(self.device)
        else:pass
        self.save_losses(loss,MLoss)
        multipliers = torch.linspace(1, 10, 100).tolist()
        for multiplier in multipliers:
            if torch.mean(torch.tensor(self.losses))*multiplier > torch.mean(torch.tensor(self.MLosses)):
                self.reward = self.reward - 1.
            if MLoss * multiplier < loss:
                self.reward = self.reward + 1.
            else:
                break
        self.rewards.append(self.reward.detach())

    def calculate_next_reward(self,loss,MLoss,reiterate=1):
        if not reiterate:
            nr_mean = torch.tensor(mean(self.next_reward)).to(self.device)
            if self.next_rewards[-1] < nr_mean:
                self.next_rewards[-1] = 100
            self.next_reward = torch.tensor([0.]).to(self.device)
        else:pass
        multipliers = torch.linspace(1, 10, 100).tolist()
        self.save_next_losses(loss, MLoss)
        for multiplier in multipliers:
            if torch.mean(torch.tensor(self.next_losses))*multiplier > torch.mean(torch.tensor(self.next_MLosses)):
                self.next_reward = self.next_reward - 1.
            if MLoss * multiplier < loss:
                self.next_reward  = self.next_reward + 1.
            else:
                break
        self.next_rewards.append(self.next_reward.detach())

    def Q_Value(self,sa_index,alpha=1.,gamma=0.99):
        best_action_indices  =  torch.argmax(self.next_actions[sa_index],dim=2)
        best_next_q_values = torch.gather(self.next_actions[sa_index], 1, best_action_indices.unsqueeze(-1))
        td_target = (self.next_rewards[sa_index] + gamma * best_next_q_values)
        td_error = td_target - self.actions[sa_index]
        self.q_target = self.actions[sa_index] + alpha * td_error
        return self.q_target,self.actions[sa_index]

    def Q_Value_experience_replay(self,teacher,RL_optimizer,criterion_RL,idx,data_input,data_output,structure_input,structure_output,
                                                 criterion_model,norm,model_b,RLmodel,dataset,dataset_idx,
                                                 no_samples,alpha=0.1,gamma=0.99):

        # model_output = model_b(dataset)
        # loss = teacher.loss_calculation(dataset_idx, model_output, data_input, data_output, structure_input,
        #                              structure_output, criterion_model, norm)

        states = [self.states[i] for i in idx]
        states = torch.stack(states,dim=0)
        action = RLmodel(states)
        with torch.no_grad():
            model_r = RLmodel.mutate_parameters(model_b, action)
            model_mutated_output = model_r(dataset)
            mutation_loss = teacher.loss_calculation(dataset_idx, model_mutated_output, data_input,
                                                  data_output,
                                                  structure_input, structure_output,
                                                  criterion_model,
                                                  norm)
        next_states = [self.next_states[i] for i in idx]
        next_states = torch.stack(next_states, dim=0)
        next_action = RLmodel(next_states)

        with torch.no_grad():
            model_r = RLmodel.mutate_parameters(model_r, next_action)
            model_mutated_output = model_r(dataset)
            next_mutation_loss = teacher.loss_calculation(dataset_idx, model_mutated_output, data_input,
                                                     data_output,
                                                     structure_input, structure_output,
                                                     criterion_model,
                                                     norm)


        RLoss = criterion_RL((mutation_loss+next_mutation_loss).unsqueeze(0),torch.tensor([0.]).to(self.device))

        return RLoss

    def experience_replay(self,teacher,RL_optimizer,criterion_RL, data_input,data_output,
                          structure_input,structure_output,
                          criterion_model,norm,
                          model_r,RLmodel,dataset,
                          dataset_idx,no_samples):
        idx = range(0,len(self.actions),1)
        idx = random.choices(idx, k=no_samples)
        RLoss = self.Q_Value_experience_replay(teacher,RL_optimizer,criterion_RL,idx,data_input,data_output,structure_input,structure_output,
                                                 criterion_model,norm,model_r,RLmodel,dataset,dataset_idx,
                                                 no_samples,alpha=0.1,gamma=0.99)
        return RLoss

    def next_to_current(self):
        self.actions = self.next_actions
        self.states = self.next_states
        self.rewards = self.next_rewards


    def activate(self,x):
        return torch.tanh(x)#*2#*self.activation_weight

