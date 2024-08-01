import copy
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.nn.functional as f

from teacher import teacher


class Metamorph_parameterReinforcer(nn.Module):
    # Note : Buzzword - Metamorph will be better name here probably :) or HIPNO
    def __init__(self,no_layers,batch_size,device):
        super(Metamorph_parameterReinforcer, self).__init__()



        self.device = device
        self.no_layers = no_layers
        self.modes = 32
        self.memory_size = 128
        # self.points_size = int(self.memory_size/6)
        self.batch_size = batch_size

        # # self.reward = torch.zeros(self.batch_size, requires_grad=True).to(self.device)
        # self.model_parameter_memory = torch.zeros((self.no_layers, self.modes,1),
        #                            requires_grad=True).to(self.device)
        # self.loss_memory = torch.zeros(1,requires_grad=True).to(self.device)
        # self.action_memory = torch.zeros((self.batch_size,self.no_layers, 1),requires_grad=True).to(self.device)
        # self.rewards_memory = torch.zeros((self.batch_size,1),requires_grad=True).to(self.device)
        # self.loss_calculation = teacher.loss_calculation
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
        model_p = torch.zeros((self.no_layers, self.modes),
                                   requires_grad=True).to(self.device)
        for name, param in model.named_parameters():
            fparam = torch.flatten(param)
            fpshape = fparam.shape[0]
            if fpshape < self.modes:
                fparamWHF = torch.fft.fft(fparam)[:fpshape].real
                model_p[i,:fpshape] = fparamWHF
                i+=1
            else:
                fparamWHF = torch.fft.fft(fparam)[:self.modes].real
                model_p[i] = fparamWHF
                i += 1
        return model_p


    def simple_weight_mutation(self,model,actions):
        # with torch.no_grad():
        i = 0
        for (name, param) in model.named_parameters():
            p = param * actions[0,i]
            param.data = p.detach()
            i += 1
        return model

    # def execute_and_evaluate_actions(self,t,model,loss,dataset,idx,data_input,data_output,
    #                                  structure_input,structure_output,criterion, norm):
    #     # with torch.no_grad():
    #     losses = torch.full((1,), 100, device=self.device)
    #     model_test = copy.deepcopy(model)
    #     # with torch.no_grad():
    #     for j in range(self.n.shape[0]):
    #         i = 0
    #         with torch.no_grad():
    #             for (name,param), (test_name,test_param) in zip(model.named_parameters(),model_test.named_parameters()):
    #                 p = param * self.action_memory[j,i,self.n[j]]
    #                 test_param.copy_(p)
    #                 i += 1
    #             model_output  = model_test(dataset)
    #             # model_output = r.detach(), g.detach(), b.detach(), a.detach(), s.detach()
    #             partial_loss = t.loss_calculation(idx,model_output,data_input,data_output,structure_input,
    #                                                  structure_output,criterion,norm)
    #             losses = torch.cat([losses,partial_loss.unsqueeze(0)]).detach()
    #     min_loss_idx = self.caluclate_reward(loss.unsqueeze(0),losses)
    #     with torch.no_grad():
    #         i = 0
    #         for (name, param) in model.named_parameters():
    #             p = param * self.action_memory[min_loss_idx, i, self.n[min_loss_idx]]
    #             param.copy_(p)
    #             i += 1
    #     return model
    #
    # def caluclate_reward(self,base_loss,losses):
    #     loss_vector = (losses < base_loss).float()[1:]
    #     min_loss_idx = torch.argmin(losses[1:])
    #     delta_loss = (base_loss - losses[1:])
    #     reward = delta_loss / (delta_loss.min())
    #     if loss_vector[min_loss_idx]:
    #         reward[min_loss_idx] = reward[min_loss_idx]* 2
    #     self.save_rewards(reward)
    #     return min_loss_idx
    #
    #
    # def save_rewards(self,reward):
    #     if self.model_parameter_memory.shape[2] > self.memory_size:
    #         temp_r = self.rewards_memory[:,1:]
    #         self.rewards_memory[:,0:-1] = temp_r
    #         self.rewards_memory[:,-1] = reward.unsqueeze(-1)
    #     else:
    #         self.rewards_memory = torch.cat([self.rewards_memory, reward.unsqueeze(-1)], dim=1)
    #
    # def RL_loss(self,criterion,gamma):
    #     # q_values = self.action_memory[:, :, self.n[-1]]
    #     # q_values.data[:,:] = 1.
    #     # self.action_memory.data[:, :, self.n[-1]] = q_values
    #     # next_q_values,idx_next_q_values = self.action_memory.max(dim=2,keepdim=True)
    #     # self.action_memory.data[idx_next_q_values] = 1.
    #     # rewards_values = self.rewards_memory
    #     # target_q_values = rewards_values.unsqueeze(1) + gamma * self.action_memory
    #
    #     RLoss = criterion(self.action_memory[:, :, self.n[-1]].unsqueeze(2), self.action_memory[:, :, self.n[-1]].unsqueeze(2))
    #     print(RLoss.grad_fn)
    #     return RLoss
    def activate(self,x):
        return torch.tanh(x)#*2#*self.activation_weight

