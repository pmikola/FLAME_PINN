import random
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
        self.in_data = 25

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

        # Definition of intermediate layer/parameters that transforms input into Fourier Feature with positional encoding and gaussian Gate
        self.weights_fd1 = nn.Parameter(torch.rand(1,self.no_subslice_in_tensors*self.in_scale, 5, dtype=torch.cfloat))
        self.m = 16 # No o of modes for SpaceTime Encoding
        self.ii = torch.arange(start=0, end=self.m, step=1, device=self.device)

        self.kernel_0 = torch.zeros(self.in_data, self.in_data, 1, 2, 2,device=self.device)
        self.k0_init = nn.Parameter(torch.tensor([[random.uniform(-1,1),random.uniform(-1,1)],
                                                       [random.uniform(-1,1),random.uniform(-1,1)]],device=self.device))
        self.kernel_0[0,:,0,:,:] = self.k0_init

        self.kernel_1 = torch.zeros(self.in_data, self.in_data, 1, 2, 2, device=self.device)
        self.k1_init = nn.Parameter(torch.tensor([[random.uniform(-1, 1), random.uniform(-1, 1)],
                                                       [random.uniform(-1, 1), random.uniform(-1, 1)]],device=self.device))
        self.kernel_1[0, :, 0, :, :] = self.k1_init

        self.kernel_2 = torch.zeros(self.in_data, self.in_data, 1, 2, 2, device=self.device)
        self.k2_init = nn.Parameter(torch.tensor([[random.uniform(-1, 1), random.uniform(-1, 1)],
                                                       [random.uniform(-1, 1), random.uniform(-1, 1)]],device=self.device))
        self.kernel_2[0, :, 0, :, :] = self.k2_init
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

        self.init_weights()
    def init_weights(self):
        if isinstance(self, nn.Linear):
            #torch.nn.init.xavier_uniform(self.weight)
            self.weight.data.normal_(mean=0.0, std=1.0)
            self.bias.data.fill_(0.01)

        if isinstance(self,nn.Conv1d):
            self.weight.data.normal_(mean=0.0, std=1.0)
            self.bias.data.fill_(0.01)

        if isinstance(self,nn.Parameter):
            self.data.normal_(mean=0.0, std=1.0)

    def forward(self, din):
        (data_input,meta_input_h1,meta_input_h2,meta_input_h3,
         meta_input_h4,meta_input_h5,meta_output_h1,meta_output_h2,
         meta_output_h3,meta_output_h4,meta_output_h5) = din
        meta_central_points = torch.cat([meta_input_h3.float(), meta_output_h3.float()], dim=1)

        # Question : Do highest hierarchy should have parameters that are learning
        #  or just be top layer without any additional coefss (regarding polyNonlinear)
        gamma = torch.tanh(self.l0h3(meta_central_points))
        gamma_l1 = torch.tanh(self.l1h3(gamma))
        gamma_l2 = torch.tanh(self.l2h3(gamma_l1))

        meta_step = torch.cat([meta_input_h2.float(), meta_output_h2.float()], dim=1)
        beta = torch.tanh(self.l0h2(meta_step))
        beta_l1 = self.shapeShift(self.l1h2(beta), gamma_l1)
        beta_l2 = self.shapeShift(self.l2h2(beta), gamma_l2)

        meta_h1 = torch.cat([meta_input_h1.float(),meta_output_h1.float()],dim=1)
        alpha = torch.tanh(self.l0h1(meta_h1))
        alpha_l1 = self.shapeShift(self.l1h1(alpha),beta_l1)
        alpha_l2 =  self.shapeShift(self.l2h1(alpha_l1),beta_l2)

        x = self.SpaceTimeFFTFeature(data_input, meta_input_h4, meta_input_h5, meta_output_h5)

        a = self.l0h0_small(x)
        b = self.l0h0_medium(x)
        c = self.l0h0_large(x)

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
            craftedPolynomial = nn.functional.hardtanh(craftedPolynomial,-2,2)
            return craftedPolynomial
        elif x.dim() == 2:
            coefficients = h[0:self.batch_size, 0:self.shifterCoefficients].unsqueeze(1)
            x_powers = torch.pow(x[0:self.batch_size, :].unsqueeze(2), self.exponents.unsqueeze(0).unsqueeze(1))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=2)
            craftedPolynomial = nn.functional.hardtanh(craftedPolynomial, -2, 2)
            return craftedPolynomial
        elif x.dim() == 4:
            coefficients = h[0:self.batch_size, 0:self.shifterCoefficients].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            x_powers = torch.pow(x[0:self.batch_size, :, :, :].unsqueeze(4),
                                 self.exponents.unsqueeze(0).unsqueeze(1).unsqueeze(2))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=4)
            craftedPolynomial = nn.functional.hardtanh(craftedPolynomial, -2, 2)
            return craftedPolynomial
        else:
            raise ValueError("Unsupported input dimensions")

    def SpaceTimeFFTFeature(self,data,meta_space,meta_step_in,meta_step_out):

        # print(self.weights_fd1.shape)
        # print(data.shape)
        x_grid, y_grid = torch.meshgrid(meta_space[0, 0:self.in_scale], meta_space[0, self.in_scale:], indexing='ij')
        x_grid = x_grid.unsqueeze(0)
        y_grid = y_grid.unsqueeze(0)

        for b in range(1,self.batch_size):
            xx_grid,yy_grid = torch.meshgrid(meta_space[b,0:self.in_scale],meta_space[0,self.in_scale:],indexing='ij')
            xx_grid = xx_grid.unsqueeze(0)
            yy_grid = yy_grid.unsqueeze(0)
            x_grid = torch.cat([x_grid,xx_grid],dim=0)
            y_grid = torch.cat([y_grid,yy_grid],dim=0)

        fseries_space_x = torch.pow(x_grid.unsqueeze(3), (2 * self.ii // 2))
        fseries_space_y = torch.pow(y_grid.unsqueeze(3), (2 * self.ii // 2))
        PosEncSin_x = torch.sin(fseries_space_x)
        PosEncSin_y = torch.sin(fseries_space_y)
        PosEncCos_x = torch.cos(fseries_space_x)
        PosEncCos_y = torch.cos(fseries_space_y)
        PosEnc_x = (PosEncCos_x + PosEncSin_x)/4
        PosEnc_y = (PosEncCos_y + PosEncSin_y)/4

        fseries_step_in = torch.pow(meta_step_in.unsqueeze(1).unsqueeze(1).unsqueeze(1), (2 * self.ii // 2))
        fseries_step_out = torch.pow(meta_step_out.unsqueeze(1).unsqueeze(1).unsqueeze(1), (2 * self.ii // 2))
        TimePosEncSin_step_in = torch.sin(fseries_step_in)
        TimeEncSin_step_out = torch.sin(fseries_step_out)
        TimeEncCos_step_in = torch.cos(fseries_step_in)
        TimeEncCos_step_out = torch.cos(fseries_step_out)
        TimeEnc_step_in = (TimeEncSin_step_out + TimePosEncSin_step_in) / 4
        TimeEnc_step_out = (TimeEncCos_step_out + TimeEncCos_step_in) / 4

        SpaceTimeEncodings = PosEnc_x + PosEnc_y + TimeEnc_step_in + TimeEnc_step_out
        SpaceTimeEncodings = SpaceTimeEncodings.repeat(1,5,1,1)
        st = list(SpaceTimeEncodings.size())
        n = int(torch.sqrt(torch.tensor(self.m)).item())

        SpaceTimeEncodings = (SpaceTimeEncodings.unsqueeze(4).
                              reshape(int(st[0]), int(st[1]), int(st[2]), n,n))

        SpaceTimeEncodings  = torch.tanh(nn.functional.conv3d(SpaceTimeEncodings,self.kernel_0))
        SpaceTimeEncodings = torch.tanh(nn.functional.conv3d(SpaceTimeEncodings, self.kernel_1))
        SpaceTimeEncodings = torch.tanh(nn.functional.conv3d(SpaceTimeEncodings, self.kernel_2))

        SpaceTimeEncodings = torch.flatten(SpaceTimeEncodings,start_dim=2)
        data = data+SpaceTimeEncodings
        # Attention :  Below is implemented simplified FNO LAYER
        # question : using only real gives better results than using real and imag in sum or concat manner?
        fft_data = torch.fft.fftshift(torch.fft.fftn(data))
        # question : is "bij,own->bin" give same outcome as "bij,own->bwj" ?
        FFwithWeights = torch.einsum("bij,own->bin", fft_data, self.weights_fd1)
        data = torch.tanh(torch.fft.ifftshift(torch.fft.ifftn(FFwithWeights))).real
        # Attention :  Above is implemented simplified FNO LAYER
        return data




