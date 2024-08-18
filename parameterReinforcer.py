import copy
import random
import time
from operator import itemgetter
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
        self.masks = deque(maxlen=self.action_per_layer)
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
        self.reward = torch.zeros(self.batch_size).to(self.device)
        self.next_reward = torch.zeros(self.batch_size).to(self.device)

        # Definition of target policy function
        self.q_target = torch.rand(self.batch_size, self.action_per_layer,requires_grad=True).to(self.device)

        # Definition of weights and weights as convolutions in FFT space used in FFTFeatures
        self.weights_data_0 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_0 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))
        self.weights_data_1 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_1 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))
        self.weights_data_2 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.float))
        self.weights_data_fft_2 = nn.Parameter(torch.rand((1,self.modes), dtype=torch.cfloat))

        self.weights_MASK_0 = nn.Parameter(torch.rand((1, self.modes), dtype=torch.float))
        self.weights_MASK_fft_0 = nn.Parameter(torch.rand((1, self.modes), dtype=torch.cfloat))
        self.weights_MASK_1 = nn.Parameter(torch.rand((1, self.modes), dtype=torch.float))
        self.weights_MASK_fft_1 = nn.Parameter(torch.rand((1, self.modes), dtype=torch.cfloat))
        self.weights_MASK_2 = nn.Parameter(torch.rand((1, self.modes), dtype=torch.float))
        self.weights_MASK_fft_2 = nn.Parameter(torch.rand((1, self.modes), dtype=torch.cfloat))
        # Definition of output dens layers
        self.lin1 = nn.Linear(self.no_layers*self.modes,self.modes)
        self.lin2 = nn.Linear(self.modes,self.action_per_layer)
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
        masks = torch.stack(list(self.masks))

        self.batch_size = model_p.shape[0]
        x,masks = self.MaskedFFTFeature(model_p,masks,self.weights_data_0,self.weights_MASK_0,
                                        self.weights_data_fft_0,self.weights_MASK_fft_0)
        x,masks = self.MaskedFFTFeature(x,masks, self.weights_data_1,self.weights_MASK_1
                                        ,self.weights_MASK_fft_1, self.weights_data_fft_1)
        x,masks = self.MaskedFFTFeature(x,masks, self.weights_data_2,self.weights_MASK_2,self.weights_MASK_fft_2, self.weights_data_fft_2)
        x = torch.flatten(x,start_dim=1)

        x = self.activate(self.lin1(x))
        x = self.lin2(x).view(self.batch_size,self.action_per_layer)
        # x = self.softmax(x)
        return x

    def MaskedFFTFeature(self,data,MASK,weights_data,weights_mask,weights_data_fft,weights_mask_fft):
        # Attention :  Below is implemented simplified FNO LAYER
        fft_data = torch.fft.fft(data,norm='forward')
        fft_mask = torch.fft.fft(MASK, norm='forward')
        FFTwithW = torch.einsum("bfp,an->bfn",fft_data, weights_data_fft)
        FFTwithWM = torch.einsum("bwch,an->bwcn", fft_mask, weights_mask_fft)
        FFTwithW = torch.einsum("bwch,pfk->pfh",FFTwithWM,FFTwithW)
        iFFW= torch.fft.ifft(FFTwithW, norm='forward')
        iFFWM = torch.fft.ifft(FFTwithWM, norm='forward')
        data = self.activate(iFFW.real)
        mask = self.activate(iFFWM.real)
        # Attention :  Above is implemented simplified FNO LAYER
        # data = torch.tanh(data)
        return data,mask

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

    def create_masks(self,data):
        for i in range(self.action_per_layer):
            MASK = torch.rand_like(data)
            new_zeros = torch.zeros_like(MASK)
            new_ones = torch.ones_like(MASK)
            treshold = 0.5
            mask = MASK.real <= treshold
            MASK[mask] = new_zeros[mask]
            MASK[~mask] = new_ones[~mask]
            self.masks.append(MASK)
            # self.masks.append(torch.nn.parameter.Parameter(data=MASK,requires_grad=True))

    def mutate(self, data_in, action):
        p_action_idx = torch.argmax(action, dim=1)
        (data_input, structure_input, meta_input_h1, meta_input_h2, meta_input_h3,
         meta_input_h4, meta_input_h5, noise_var_in, meta_output_h1, meta_output_h2,
         meta_output_h3, meta_output_h4, meta_output_h5, noise_var_out) = data_in
        mask =torch.stack([self.masks[i] for i in p_action_idx])
        rng  = torch.rand(mask.shape).to(self.device).detach()
        mask = mask.float() * 1 + (1 - mask) * rng

        structure_input = structure_input.unsqueeze(0).expand(self.batch_size, -1, -1, -1)
        meta_input_h1 = meta_input_h1.unsqueeze(0).expand(self.batch_size, -1,-1)
        meta_input_h2 = meta_input_h2.unsqueeze(0).expand(self.batch_size, -1,-1)
        meta_input_h3 = meta_input_h3.unsqueeze(0).expand(self.batch_size, -1,-1)
        meta_input_h4 = meta_input_h4.unsqueeze(0).expand(self.batch_size, -1,-1)
        meta_input_h5 = meta_input_h5.unsqueeze(0).expand(self.batch_size,-1)
        noise_var_in = noise_var_in.unsqueeze(0).expand(self.batch_size, -1,-1)
        meta_output_h1 = meta_output_h1.unsqueeze(0).expand(self.batch_size, -1,-1)
        meta_output_h2 = meta_output_h2.unsqueeze(0).expand(self.batch_size, -1,-1)
        meta_output_h3 = meta_output_h3.unsqueeze(0).expand(self.batch_size, -1,-1)
        meta_output_h4 = meta_output_h4.unsqueeze(0).expand(self.batch_size, -1,-1)
        meta_output_h5 = meta_output_h5.unsqueeze(0).expand(self.batch_size,-1)
        noise_var_out = noise_var_out.unsqueeze(0).expand(self.batch_size, -1,-1)

        return data_input * mask, structure_input, meta_input_h1, meta_input_h2, meta_input_h3,meta_input_h4, meta_input_h5, noise_var_in, meta_output_h1, meta_output_h2,meta_output_h3, meta_output_h4, meta_output_h5, noise_var_out


    def exploit_explore_action_selector(self,action,p=0.1):
        selector = torch.randint(1,10,(1,))
        if  selector > 10*p:
            action = torch.rand(self.batch_size, self.action_per_layer,requires_grad=True).to(self.device)
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
        MLoss = torch.stack(MLoss).to(self.device)
        # if not reiterate:
        self.reward = torch.zeros_like(MLoss).to(self.device)
        # else:
        #     pass
        self.save_losses(loss, MLoss)
        #MLosses_tensor = torch.stack([ml.clone().detach().to(self.device) for ml in self.MLosses])
        multipliers = torch.linspace(1, 10, 100).to(self.device)
        #losses_mean = torch.mean(torch.tensor(self.losses)).to(self.device)
        #MLosses_mean = torch.mean(MLosses_tensor).to(self.device)
        loss_min = torch.min(torch.stack(list(self.losses))).to(self.device)
        multipliers = multipliers.view(1, -1)
        #condition_1 = (losses_mean * multipliers > MLosses_mean)
        condition_2 = (MLoss.view(-1, 1) * multipliers.permute(0, 1) < loss_min)
        self.reward += torch.sum(condition_2.float(),dim=1)*0.1
        # self.reward -= torch.sum(condition_1.float())*0.01
        self.rewards.append(self.reward.detach())

    def calculate_next_reward(self,loss,MLoss,reiterate=1):
        MLoss = torch.stack(MLoss).to(self.device)
        # if not reiterate:
        self.next_reward = torch.zeros_like(MLoss).to(self.device)
        # else:
        #     pass
        self.save_next_losses(loss, MLoss)
        #MLosses_tensor = torch.stack([ml.clone().detach().to(self.device) for ml in self.MLosses])
        multipliers = torch.linspace(1, 10, 100).to(self.device)
        # losses_mean = torch.mean(torch.tensor(self.losses)).to(self.device)
        loss_min = torch.min(torch.stack(list(self.losses))).to(self.device)
        #MLosses_mean = torch.mean(MLosses_tensor).to(self.device)
        multipliers = multipliers.view(1, -1)
        #condition_1 = (losses_mean * multipliers > MLosses_mean)
        condition_2 = (MLoss.view(-1, 1) * multipliers.permute(0, 1) < loss_min)

        self.next_reward += torch.sum(condition_2.float(), dim=1) * 0.01
        # self.next_reward -= torch.sum(condition_1.float()) * 0.01
        self.next_rewards.append(self.next_reward.detach())

    def Q_Value(self,sa_index=-1,alpha=1.,gamma=0.99):
        # print(self.next_actions[sa_index].shape,self.next_rewards[sa_index].shape)
        best_action_indices  =  torch.argmax(self.next_actions[sa_index],dim=1)
        best_next_q_values = torch.gather(self.next_actions[sa_index], 1, best_action_indices.unsqueeze(-1))
        q_values = torch.gather(self.actions[sa_index], 1, best_action_indices.unsqueeze(-1))
        # q_values = self.actions[sa_index]
        # best_next_q_values = self.next_actions[sa_index]
        next_rewards = self.next_rewards[sa_index].permute(1, 0)
        td_target = next_rewards + gamma * best_next_q_values
        td_error = td_target - q_values
        q_target = q_values + alpha * td_error
        return q_target,best_action_indices

    def mutations_batch(self,teacher,model,dataset,action,loss_calc_data):
        dataset_idx, data_input,data_output,structure_input, structure_output,criterion_model,norm = loss_calc_data
        mutation_losses = []
        di, si, mih1, mih2, mih3, mih4, mih5, nvi, moh1, moh2, moh3, moh4, moh5, nvo = self.mutate(dataset, action)
        mutated_dataset = di, si, mih1, mih2, mih3, mih4, mih5, nvi, moh1, moh2, moh3, moh4, moh5, nvo
        dataset_m = (
            di[:self.batch_size], si[:self.batch_size], mih1[:self.batch_size], mih2[:self.batch_size],
            mih3[:self.batch_size], mih4[:self.batch_size], mih5[:self.batch_size], nvi[:self.batch_size],
            moh1[:self.batch_size], moh2[:self.batch_size], moh3[:self.batch_size], moh4[:self.batch_size],
            moh5[:self.batch_size], nvo[:self.batch_size])


        mutated_output = model(dataset_m)
        mutation_loss = teacher.loss_calculation(dataset_idx,
                                                     mutated_output,
                                                     data_input,
                                                     data_output,
                                                     structure_input,
                                                     structure_output,
                                                     criterion_model,
                                                     norm)
        mutation_losses.append(mutation_loss.detach())
        return mutation_losses,mutated_dataset

    def Q_Value_experience_replay(self,teacher,RL_optimizer,criterion_RL,idx,data_input,data_output,structure_input,structure_output,
                                                 criterion_model,norm,model_b,RLmodel,dataset,dataset_idx,
                                                 no_samples,alpha=0.1,gamma=0.99):
        loss_calc_data = dataset_idx, data_input, data_output, structure_input, structure_output, criterion_model, norm
        model_output = model_b(dataset)
        loss = teacher.loss_calculation(dataset_idx, model_output, data_input, data_output, structure_input,
                                     structure_output, criterion_model, norm)
        model = copy.deepcopy(model_b)
        _ = self.save_state(model)
        states = [self.states[i] for i in idx]
        states = torch.stack(states,dim=0)
        action = RLmodel(states)
        action = self.exploit_explore_action_selector(action)
        self.save_action(action)
        mutation_losses,mutated_dataset = self.mutations_batch(teacher,model,dataset,action,loss_calc_data)
            # mutation_loss_idx, mutation_loss = min(enumerate(mutation_losses), key=itemgetter(1))
        self.calculate_reward(loss, mutation_losses)
        _ = self.save_next_state(model)
        next_states = [self.next_states[i] for i in idx]
        next_states = torch.stack(next_states, dim=0)
        next_action = RLmodel(next_states)
        next_action = self.exploit_explore_action_selector(next_action)
        self.save_next_action(next_action)
        next_mutation_losses,next_mutated_dataset = self.mutations_batch(teacher,model,dataset,next_action,loss_calc_data)
            # next_mutation_loss_idx, next_mutation_loss = min(enumerate(next_mutation_losses), key=itemgetter(1))
        self.calculate_next_reward(loss, next_mutation_losses)
        Q_target, q_idx = self.Q_Value()
        q_values = torch.gather(action, 1, q_idx.unsqueeze(-1))
        RLoss = criterion_RL(q_values, Q_target)
        RL_optimizer.zero_grad(set_to_none=True)
        RLoss.backward()
        RL_optimizer.step()

        best_idx = torch.argmin(torch.stack(next_mutation_losses))
        di, si, mih1, mih2, mih3, mih4, mih5, nvi, moh1, moh2, moh3, moh4, moh5, nvo = mutated_dataset
        dataset_mutated = (
        di[best_idx], si[best_idx], mih1[best_idx], mih2[best_idx], mih3[best_idx], mih4[best_idx], mih5[best_idx], nvi[best_idx], moh1[best_idx], moh2[best_idx], moh3[best_idx], moh4[best_idx], moh5[best_idx],
        nvo[best_idx])
        del model_b,model
        return RLmodel,RLoss,dataset_mutated

    def experience_replay(self,teacher,RL_optimizer,criterion_RL, data_input,data_output,
                          structure_input,structure_output,
                          criterion_model,norm,
                          model_r,RLmodel,dataset,
                          dataset_idx):
        if len(self.actions) + 1 <2:
            idx = range(0,1,1)
        else:
            idx = range(0, len(self.actions), 1)
        no_samples = self.batch_size
        idx = random.choices(idx, k=no_samples)
        RLmodel,RLoss,dataset_mutated = self.Q_Value_experience_replay(teacher,RL_optimizer,criterion_RL,idx,data_input,
                                                       data_output,structure_input,structure_output,
                                                 criterion_model,norm,model_r,RLmodel,dataset,dataset_idx,
                                                 no_samples,alpha=0.1,gamma=0.99)
        return RLmodel,RLoss,dataset_mutated

    def next_to_current(self):
        self.actions = self.next_actions
        self.states = self.next_states
        self.rewards = self.next_rewards


    def activate(self,x):
        return torch.tanh(x)#*2#*self.activation_weight

