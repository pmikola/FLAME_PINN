import time

import torch
import torch.nn as nn
import torch.optim as optim

class PINO(nn.Module):
    # Note : Buzzword - Metamorph will be better name here probably :) or HIPNO
    def __init__(self,no_frame_samples,batch_size,input_window_size,device):
        super(PINO, self).__init__()
        self.device = device
        self.no_frame_samples = no_frame_samples
        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.in_scale = (1+self.input_window_size*2)
        self.no_subslice_in_tensors = 5


        # NOTE : on Hierarchy 0 flows data and on higher levels flows metas
        self.no_meta_h3 = 20 * 2
        self.no_meta_h2 = 32 * 2
        self.no_meta_h1 = 224 * 2
        # Definition of non-linear shifting activation function with parameters
        self.shifterCoefficients = 10 # No. of polynomial coefficients
        self.exponents = torch.arange(1, self.shifterCoefficients+1 , 1, device=self.device) # Check : frome 0 to n or from 1 to n +1?
        # PLACEHOLDER for layers in non-linear activation function or others stuff

        # Definition of layer 0,1,2 for lvl 3 in hierarchy
        self.l0h3 = nn.Linear(in_features=self.no_meta_h3, out_features=int(self.no_meta_h1 / 4))
        self.l1h3 = nn.Linear(in_features=int(self.no_meta_h1 / 4), out_features=int(self.no_meta_h1 / 4))
        self.l2h3 = nn.Linear(in_features=int(self.no_meta_h1 / 4), out_features=self.shifterCoefficients)

        # Definition of layer 0,1,2 for lvl 2 in hierarchy
        self.l0h2 = nn.Linear(in_features=self.no_meta_h2, out_features=int(self.no_meta_h1 / 4))
        self.l1h2 = nn.Linear(in_features=int(self.no_meta_h1 / 4), out_features=int(self.no_meta_h1 / 4))
        self.l2h2 = nn.Linear(in_features=int(self.no_meta_h1 / 4), out_features=self.shifterCoefficients)

        # Definition of layer 0,1,2 for lvl 1 in hierarchy
        self.l0h1 = nn.Linear(in_features=self.no_meta_h1, out_features=int(self.no_meta_h1 / 2))
        self.l1h1 = nn.Linear(in_features=int(self.no_meta_h1 / 2), out_features=int(self.no_meta_h1 / 4))
        self.l2h1 = nn.Linear(in_features=int(self.no_meta_h1 / 4), out_features=self.shifterCoefficients)

        # Definition of input layer 0 for lvl 0 in hierarchy
        self.l0h0_small = nn.Conv1d(in_channels=self.no_subslice_in_tensors*self.in_scale,
                                  out_channels=self.no_subslice_in_tensors*self.in_scale, kernel_size=1)
        self.l0h0_medium = nn.Conv1d(in_channels=self.no_subslice_in_tensors*self.in_scale,
                                   out_channels=self.no_subslice_in_tensors*self.in_scale, kernel_size=2)
        self.l0h0_large = nn.Conv1d(in_channels=self.no_subslice_in_tensors*self.in_scale,
                                  out_channels=self.no_subslice_in_tensors*self.in_scale, kernel_size=3)

        # Question : if we change same input to fft (k space) representation and
        #  change configuration of input not by kernel size but by modes
        #  from fft with learnable parameters - do we will have better results?
        # TODO : check if above query gives better result

        # Definition of input layer 1,2,3 for lvl 0 in hierarchy
        self.l1h0= nn.Conv1d(in_channels=self.no_subslice_in_tensors*self.in_scale,
                                  out_channels=self.no_subslice_in_tensors*self.in_scale, kernel_size=3)
        self.l2h0 = nn.Conv1d(in_channels=self.no_subslice_in_tensors*self.in_scale,
                                  out_channels=int(self.no_subslice_in_tensors*self.in_scale), kernel_size=3)
        self.flat_size = 200
        self.l3h0 = nn.Linear(in_features=self.flat_size,out_features=int(self.flat_size/2))

        # Definition of Heads for red, green, blue and alpha output channels
        self.l4_h0_r = nn.Linear(in_features=int(self.flat_size/2),out_features=1,bias=False)
        self.l4_h0_g = nn.Linear(in_features=int(self.flat_size/2),out_features=1,bias=False)
        self.l4_h0_b = nn.Linear(in_features=int(self.flat_size/2),out_features=1,bias=False)
        self.l4_h0_a = nn.Linear(in_features=int(self.flat_size/2),out_features=1,bias=False)



    def forward(self, din):
        (data_input,meta_input_h1,meta_input_h2,meta_input_h3,
         meta_output_h1,meta_output_h2,meta_output_h3) = din
        meta_h3 = torch.cat([meta_input_h3.float(), meta_output_h3.float()], dim=1)
        gamma = torch.tanh(self.l0h3(meta_h3))
        gamma_l1 = self.shapeShift(self.l1h3(gamma), gamma) # TODO : change gamma this for nn.parameter in init
        gamma_l2 = self.shapeShift(self.l2h3(gamma), gamma) # TODO : change gamma this for nn.parameter in init

        meta_h2 = torch.cat([meta_input_h2.float(),meta_output_h2.float()],dim=1)
        beta = torch.tanh(self.l0h2(meta_h2))
        beta_l1 = self.shapeShift(self.l1h2(beta), gamma_l1)
        beta_l2 = self.shapeShift(self.l2h2(beta), gamma_l2)

        meta_h1 = torch.cat([meta_input_h1.float(),meta_output_h1.float()],dim=1)
        alpha = torch.tanh(self.l0h1(meta_h1))
        alpha_l1 = self.shapeShift(self.l1h1(alpha),beta_l1)
        alpha_l2 =  self.shapeShift(self.l2h1(alpha_l1),beta_l2)

        a = self.l0h0_small(data_input)
        b = self.l0h0_medium(data_input)
        c = self.l0h0_large(data_input)

        # Assumption : static relu on input to make fixed stable embedded/hidden
        #  representation - also all values for used tensor should be above 0.
        a = torch.relu(a)
        b = torch.relu(b)
        c = torch.relu(c)
        x = torch.cat([a, b, c], dim=2)
        x = self.shapeShift(self.l1h0(x),alpha_l1)
        x = self.shapeShift(self.l2h0(x),alpha_l2)
        x = torch.flatten(x,start_dim=1)
        x = torch.relu(self.l3h0(x))

        r = self.l4_h0_r(x)
        g = self.l4_h0_g(x)
        b = self.l4_h0_b(x)
        a = self.l4_h0_a(x)
        return r,g,b,a


    def shapeShift(self,x, h):
        if x.dim() == 3:
            coefficients = h[0:self.batch_size, 0:self.shifterCoefficients].unsqueeze(1).unsqueeze(2)
            x_powers = torch.pow(x[0:self.batch_size,:,:].unsqueeze(3), self.exponents.unsqueeze(0).unsqueeze(1))
            craftedPolynomial = torch.sum(coefficients*x_powers,dim=3)
            return craftedPolynomial
        elif x.dim() == 2:
            coefficients = h[0:self.batch_size, 0:self.shifterCoefficients].unsqueeze(1)
            x_powers = torch.pow(x[0:self.batch_size, :].unsqueeze(2), self.exponents.unsqueeze(0).unsqueeze(1))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=2)
            return craftedPolynomial
        elif x.dim() == 4:
            coefficients = h[0:self.batch_size, 0:self.shifterCoefficients].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            x_powers = torch.pow(x[0:self.batch_size, :, :, :].unsqueeze(4),
                                 self.exponents.unsqueeze(0).unsqueeze(1).unsqueeze(2))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=4)
            return craftedPolynomial
        else:
            raise ValueError("Unsupported input dimensions")

