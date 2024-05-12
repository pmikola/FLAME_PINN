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
        # Definition of input layer 0 for lvl 0 in hierarchy
        self.in_small = nn.Conv1d(in_channels=self.no_subslice_in_tensors*self.in_scale,
                                  out_channels=self.no_subslice_in_tensors*self.in_scale, kernel_size=1)
        self.in_medium = nn.Conv1d(in_channels=self.no_subslice_in_tensors*self.in_scale,
                                   out_channels=self.no_subslice_in_tensors*self.in_scale, kernel_size=2)
        self.in_large = nn.Conv1d(in_channels=self.no_subslice_in_tensors*self.in_scale,
                                  out_channels=self.no_subslice_in_tensors*self.in_scale, kernel_size=3)

        # Question : if we change same input to fft (k space) representation and
        #  change configuration of input not by kernel size but by modes
        #  from fft with learnable parameters - do we will have better results?
        # TODO : check if above query gives better result

        # Definition of input layer 1 for lvl 0 in hierarchy
        self.l1h0= nn.Conv1d(in_channels=self.no_subslice_in_tensors*self.in_scale,
                                  out_channels=self.no_subslice_in_tensors*self.in_scale, kernel_size=1)

        # Definition of non-linear shifting activation function with parameters
        self.no_shifter_params = 10
        # PLACEHOLDER for layers in non-linear activation function

        # Definition of layer 0,1,2 for lvl 1 in hierarchy
        self.no_meta_h1 = 224
        self.l0h1 = nn.Linear(in_features=self.no_meta_h1, out_features=int(self.no_meta_h1 / 2))
        self.l1h1 = nn.Linear(in_features=int(self.no_meta_h1 / 2), out_features=int(self.no_meta_h1 / 4))
        self.l2h1 = nn.Linear(in_features=int(self.no_meta_h1 / 4), out_features=self.no_shifter_params)
    def forward(self, din):
        (data_input,meta_input_h1,meta_input_h2,meta_input_h3,data_output,
         meta_output_h1,meta_output_h2,meta_output_h3) = din
        print(data_input.shape,data_output.shape,meta_input_h1.shape,meta_input_h2.shape,meta_input_h3.shape,
         meta_output_h1.shape,meta_output_h2.shape,meta_output_h3.shape)


        a = self.in_small(data_input)
        b = self.in_medium(data_input)
        c = self.in_large(data_input)

        # Assumption : static relu on input to make fixed stable embedded/hidden
        #  representation - also all values for used tensor should be above 0.
        a = torch.relu(a)
        b = torch.relu(b)
        c = torch.relu(c)
        x = torch.cat([a,b,c],dim=2)

        alpha = torch.relu(self.l0h1(meta_input_h1.float()))
        alpha = self.shapeShift(self.l1h1(alpha),alpha)
        alpha =  self.shapeShift(self.l2h1(alpha),alpha)
        x = self.l1h0(x)
        x = self.shapeShift(x,alpha)
        print(x.shape, a.shape, b.shape, c.shape,'xabc')
        time.sleep(1100)



        return x


    def shapeShift(self,x, h):
        if x.dim() == 3:
            coefficients = h[0:self.batch_size, 0:self.no_shifter_params].unsqueeze(1).unsqueeze(2)
            exponents = torch.arange(1,self.no_shifter_params+1,1,device=self.device)
            x_powers = torch.pow(x[0:self.batch_size,:,:].unsqueeze(3), exponents.unsqueeze(0).unsqueeze(1))
            craftedPolynomial = torch.sum(coefficients*x_powers,dim=3)
            return craftedPolynomial
        elif x.dim() == 2:
            coefficients = h[0:self.batch_size, 0:self.no_shifter_params].unsqueeze(1)
            exponents = torch.arange(1, self.no_shifter_params + 1, 1, device=self.device)
            x_powers = torch.pow(x[0:self.batch_size, :].unsqueeze(2), exponents.unsqueeze(0).unsqueeze(1))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=2)
            return craftedPolynomial
        elif x.dim() == 4:
            coefficients = h[0:self.batch_size, 0:self.no_shifter_params].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            exponents = torch.arange(1, self.no_shifter_params + 1, 1, device=self.device)
            x_powers = torch.pow(x[0:self.batch_size, :, :, :].unsqueeze(4),
                                 exponents.unsqueeze(0).unsqueeze(1).unsqueeze(2))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=4)
            return craftedPolynomial
        else:
            raise ValueError("Unsupported input dimensions")

