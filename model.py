import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt


class Metamorph(nn.Module):
    # Note : Buzzword - Metamorph will be better name here probably :) or HIPNO
    def __init__(self,no_frame_samples,batch_size,input_window_size,device):
        super(Metamorph, self).__init__()
        self.device = device
        self.no_frame_samples = no_frame_samples
        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.in_scale = (1+self.input_window_size*2)
        self.no_subslice_in_tensors = 4
        self.in_data = 20

        # Definition of non-linear shifting activation function with parameters
        self.shifterCoefficients = 6  # No. of polynomial coefficients
        self.exponents = torch.arange(0, self.shifterCoefficients, 1,
                                      device=self.device)  # Check : from 0 to n or from 1 to n +1?

        # Definition of intermediate layer/parameters that transforms input into Fourier Feature with positional encoding and TODO: gaussian Gate
        self.modes = 16  # No o of modes for SpaceTime Encoding
        self.ii = torch.arange(start=0, end=self.modes, step=1, device=self.device)

        self.weights_data_fft = nn.Parameter(torch.rand(1, self.no_subslice_in_tensors * self.in_scale, self.in_scale, dtype=torch.cfloat))
        self.weights_space_time_encoding_fft = nn.Parameter(torch.rand(1,self.no_subslice_in_tensors * self.in_scale, self.in_scale, self.modes, dtype=torch.cfloat))
        self.weights_data = nn.Parameter(torch.rand(1, self.no_subslice_in_tensors * self.in_scale, self.in_scale, dtype=torch.float))

        # Problem with the kernels below when changing size that are used in FFT
        # self.kernel_0 = torch.zeros(self.in_data, self.in_data, 1, 2, 2, device=self.device)
        # self.k0_init = nn.Parameter(torch.tensor([[random.uniform(-1, 1), random.uniform(-1, 1)],
        #                                           [random.uniform(-1, 1), random.uniform(-1, 1)]], device=self.device))
        # self.kernel_0[0, :, 0, :, :] = self.k0_init
        #
        # self.kernel_1 = torch.zeros(self.in_data, self.in_data, 1, 2, 2, device=self.device)
        # self.k1_init = nn.Parameter(torch.tensor([[random.uniform(-1, 1), random.uniform(-1, 1)],
        #                                           [random.uniform(-1, 1), random.uniform(-1, 1)]], device=self.device))
        # self.kernel_1[0, :, 0, :, :] = self.k1_init
        #
        # self.kernel_2 = torch.zeros(self.in_data, self.in_data, 1, 2, 2, device=self.device)
        # self.k2_init = nn.Parameter(torch.tensor([[random.uniform(-1, 1), random.uniform(-1, 1)],
        #                                           [random.uniform(-1, 1), random.uniform(-1, 1)]], device=self.device))
        # self.kernel_2[0, :, 0, :, :] = self.k2_init

        # NOTE : on Hierarchy 0 flows data and on higher levels flows metas
        self.no_meta_h3 = 20 * 2
        self.no_meta_h2 = 32 * 2
        self.no_meta_h1 = 224 * 2
        self.dens_width = 10 * self.shifterCoefficients
        self.flat_size = 200

        # Definition of layer 0,1,2 for lvl 3 in hierarchy
        self.l0h3 = nn.Linear(in_features=self.no_meta_h3, out_features=self.dens_width*self.shifterCoefficients)
        self.l1h3 = nn.Linear(in_features=self.dens_width*self.shifterCoefficients, out_features=self.dens_width*(self.shifterCoefficients**2))
        self.l2h3 = nn.Linear(in_features=self.dens_width*(self.shifterCoefficients**2), out_features=self.dens_width*(self.shifterCoefficients**3))


        # Definition of layer 0,1,2 for lvl 2 in hierarchy
        self.l0h2 = nn.Linear(in_features=self.no_meta_h2, out_features=self.dens_width*self.shifterCoefficients)
        self.l1h2 = nn.Linear(in_features=self.dens_width*self.shifterCoefficients, out_features=self.dens_width*self.shifterCoefficients)
        self.l2h2 = nn.Linear(in_features=self.dens_width*self.shifterCoefficients, out_features=self.dens_width*(self.shifterCoefficients**2))

        # Definition of layer 0,1,2 for lvl 1 in hierarchy
        self.l0h1 = nn.Linear(in_features=self.no_meta_h1, out_features=self.dens_width)
        self.l1h1 = nn.Linear(in_features=self.dens_width, out_features=self.dens_width)
        self.l2h1 = nn.Linear(in_features=self.dens_width, out_features=self.dens_width*self.shifterCoefficients)

        # Definition of intermidiet layer between lvl 0 and 1 for dimension matching
        self.l1h01 = nn.Linear(in_features=self.dens_width, out_features=(self.in_scale**2) * self.shifterCoefficients*self.no_subslice_in_tensors)
        self.l2h01 = nn.Linear(in_features=self.dens_width*self.shifterCoefficients, out_features=(self.in_scale**2) * self.shifterCoefficients*self.no_subslice_in_tensors)

        # Definition of input layer 0 for lvl 0 in hierarchy
        # rmv of 3 paralled layers for conv 1d k=1,2,3
        # Question : if we change same input to fft (k space) representation and
        #  change configuration of input not by kernel size but by modes
        #  from fft with learnable parameters - do we will have better results?
        # TODO : check if above query gives better result

        # Definition of input layer 1,2,3 for lvl 0 in hierarchy
        self.l1h0= nn.Conv1d(in_channels=self.no_subslice_in_tensors*self.in_scale,
                                  out_channels=self.no_subslice_in_tensors*self.in_scale, kernel_size=1)
        self.l2h0 = nn.Conv1d(in_channels=self.no_subslice_in_tensors*self.in_scale,
                                  out_channels=int(self.no_subslice_in_tensors*self.in_scale), kernel_size=1)
        self.l3h0 = nn.Linear(in_features=self.no_subslice_in_tensors*self.in_scale**2,out_features=int(self.flat_size/2))

        # Definition of the structure density distribution
        self.l1h0s = nn.Conv1d(in_channels=714,out_channels=200,kernel_size=1)
        self.l2h0s = nn.Conv1d(in_channels=414, out_channels=200, kernel_size=1)
        self.l3h0s = nn.Conv1d(in_channels=200, out_channels=self.in_scale*self.no_subslice_in_tensors, kernel_size=1)
        self.l4h0s = nn.Conv1d(in_channels=200, out_channels=self.in_scale, kernel_size=1)

        # Definition of Heads for red, green, blue and alpha output channels
        self.l4_h0_r = nn.Linear(in_features=int(self.flat_size/2),out_features=int(self.in_scale**2),bias=True)
        self.l4_h0_g = nn.Linear(in_features=int(self.flat_size/2),out_features=int(self.in_scale**2),bias=True)
        self.l4_h0_b = nn.Linear(in_features=int(self.flat_size/2),out_features=int(self.in_scale**2),bias=True)
        self.l4_h0_a = nn.Linear(in_features=int(self.flat_size/2),out_features=int(self.in_scale**2),bias=True)

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
        (data_input,structure_input,meta_input_h1,meta_input_h2,meta_input_h3,
         meta_input_h4,meta_input_h5,meta_output_h1,meta_output_h2,
         meta_output_h3,meta_output_h4,meta_output_h5) = din
        meta_central_points = torch.cat([meta_input_h3.float(), meta_output_h3.float()], dim=1)
        # print(data_input.shape,meta_input_h1.shape,meta_input_h2.shape,meta_input_h3.shape,
        #  meta_input_h4.shape,meta_input_h5.shape,meta_output_h1.shape,meta_output_h2.shape,
        #  meta_output_h3.shape,meta_output_h4.shape,meta_output_h5.shape,meta_central_points.shape)
        # Question : Do highest hierarchy should have parameters that are learning
        #  or just be top layer without any additional coefss (regarding polyNonlinear)

        gamma = torch.tanh(self.l0h3(meta_central_points))
        gamma_l1 = torch.tanh(self.l1h3(gamma))
        gamma_l2 = torch.tanh(self.l2h3(gamma_l1))

        meta_step = torch.cat([meta_input_h2.float(), meta_output_h2.float()], dim=1)
        beta = torch.tanh(self.l0h2(meta_step))
        beta_l1 = self.shapeShift(self.l1h2(beta), gamma_l1)
        beta_l2 = self.shapeShift(self.l2h2(beta_l1), gamma_l2)

        meta_h1 = torch.cat([meta_input_h1.float(),meta_output_h1.float()],dim=1)
        alpha = torch.tanh(self.l0h1(meta_h1))
        alpha_l1 = self.shapeShift(self.l1h1(alpha),beta_l1)
        alpha_l2 =  self.shapeShift(self.l2h1(alpha_l1),beta_l2)

        x_alpha_l1 = torch.tanh(self.l1h01(alpha_l1))
        x_alpha_l2 = torch.tanh(self.l2h01(alpha_l2))

        #s = torch.tanh(self.l1h0s(structure_input).permute(0, 2, 1))
        #s = torch.tanh(self.l2h0s(s).permute(0, 2, 1))
        #s = torch.tanh(self.l3h0s(s).permute(0, 2, 1))
        #s = torch.tanh(self.l4h0s(s).permute(0, 2, 1))
        # sfft = torch.tanh(self.l5h0s(s))
        x = self.SpaceTimeFFTFeature(data_input, meta_input_h4, meta_input_h5, meta_output_h5)
        x = x #+ s

        x = self.shapeShift(self.l1h0(x),x_alpha_l1)
        x = self.shapeShift(self.l2h0(x),x_alpha_l2)

        x = torch.flatten(x,start_dim=1)
        x = torch.tanh(self.l3h0(x))

        r = self.l4_h0_r(x).reshape(self.batch_size,self.in_scale,self.in_scale)
        g = self.l4_h0_g(x).reshape(self.batch_size,self.in_scale,self.in_scale)
        b = self.l4_h0_b(x).reshape(self.batch_size,self.in_scale,self.in_scale)
        a = self.l4_h0_a(x).reshape(self.batch_size,self.in_scale,self.in_scale)
        return r,g,b,a


    def shapeShift(self,x, h):
        if x.dim() == 3:
            coefficients = h.reshape(self.batch_size,x.shape[1],x.shape[2],self.shifterCoefficients)
            x_powers = torch.pow(x[0:self.batch_size,:,:].unsqueeze(3), self.exponents.unsqueeze(0).unsqueeze(1))
            craftedPolynomial = torch.sum(coefficients*x_powers,dim=3)
            # craftedPolynomial = nn.functional.hardtanh(craftedPolynomial,-2,2)
            return craftedPolynomial
        elif x.dim() == 2:
            coefficients = h.reshape(self.batch_size,x.shape[1],self.shifterCoefficients)
            x_powers = torch.pow(x[0:self.batch_size, :].unsqueeze(2), self.exponents.unsqueeze(0).unsqueeze(1))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=2)
            # craftedPolynomial = nn.functional.hardtanh(craftedPolynomial, -2, 2)
            return craftedPolynomial
        elif x.dim() == 4:
            coefficients = h.reshape(self.batch_size, x.shape[1], x.shape[2], x.shape[3], self.shifterCoefficients)
            x_powers = torch.pow(x[0:self.batch_size, :, :, :].unsqueeze(4),self.exponents.unsqueeze(0).unsqueeze(1).unsqueeze(2))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=4)
            # craftedPolynomial = nn.functional.hardtanh(craftedPolynomial, -2, 2)
            return craftedPolynomial
        else:
            raise ValueError("Unsupported input dimensions")

    def SpaceTimeFFTFeature(self,data,meta_space,meta_step_in,meta_step_out):
        # print(self.weights_fd1.shape)
        # print(data.shape)
        # x_grid, y_grid = torch.meshgrid(meta_space[0, 0:self.in_scale], meta_space[0, self.in_scale:], indexing='ij')
        # x_grid = x_grid.unsqueeze(0)
        # y_grid = y_grid.unsqueeze(0)
        #
        # for b in range(1,self.batch_size):
        #     xx_grid,yy_grid = torch.meshgrid(meta_space[b,0:self.in_scale],meta_space[0,self.in_scale:],indexing='ij')
        #     xx_grid = xx_grid.unsqueeze(0)
        #     yy_grid = yy_grid.unsqueeze(0)
        #     x_grid = torch.cat([x_grid,xx_grid],dim=0)
        #     y_grid = torch.cat([y_grid,yy_grid],dim=0)
        #
        # # NOTE : SPACE CODING
        # fseries_space_x = torch.pow(x_grid.unsqueeze(3), (2 * self.ii // 2))
        # fseries_space_y = torch.pow(y_grid.unsqueeze(3), (2 * self.ii // 2))
        # PosEncSin_x = torch.sin(fseries_space_x)
        # PosEncSin_y = torch.sin(fseries_space_y)
        # PosEncCos_x = torch.cos(fseries_space_x)
        # PosEncCos_y = torch.cos(fseries_space_y)
        # PosEnc_x = (PosEncCos_x + PosEncSin_x)/4
        # PosEnc_y = (PosEncCos_y + PosEncSin_y)/4
        #
        # # NOTE : TIME CODING
        # fseries_step_in = torch.pow(meta_step_in.unsqueeze(1).unsqueeze(1).unsqueeze(1), (2 * self.ii // 2))
        # fseries_step_out = torch.pow(meta_step_out.unsqueeze(1).unsqueeze(1).unsqueeze(1), (2 * self.ii // 2))
        # TimePosEncSin_step_in = torch.sin(fseries_step_in)
        # TimeEncSin_step_out = torch.sin(fseries_step_out)
        # TimeEncCos_step_in = torch.cos(fseries_step_in)
        # TimeEncCos_step_out = torch.cos(fseries_step_out)
        # TimeEnc_step_in = (TimeEncSin_step_out + TimePosEncSin_step_in)/4
        # TimeEnc_step_out = (TimeEncCos_step_out + TimeEncCos_step_in)/4
        #
        # TimeEnc = TimeEnc_step_in + TimeEnc_step_out
        # PosEnc = PosEnc_x + PosEnc_y
        # SpaceTimeEncodings = TimeEnc + PosEnc
        # # STER = torch.mean(SpaceTimeEncodings, dim=3).repeat(1,4,1)
        # # data = data + STER
        #print(data.shape,torch.mean(SpaceTimeEncodings,dim=3).shape,fseries_space_x.shape,fseries_step_in.shape)
        # print(SpaceTimeEncodings.shape)
        # SpaceTimeEncodings = SpaceTimeEncodings.repeat(1,4,1,1)
        # st = list(SpaceTimeEncodings.size())
        # n = int(torch.sqrt(torch.tensor(self.modes)).item())
        # print(SpaceTimeEncodings.shape)

        # SpaceTimeEncodings = (SpaceTimeEncodings.unsqueeze(4).reshape(int(st[0]), int(st[1]), int(st[2]), n,n))
        # SpaceTimeEncodings  = torch.tanh(nn.functional.conv3d(SpaceTimeEncodings,self.kernel_0))
        # SpaceTimeEncodings = torch.tanh(nn.functional.conv3d(SpaceTimeEncodings, self.kernel_1))
        # SpaceTimeEncodings = torch.tanh(nn.functional.conv3d(SpaceTimeEncodings, self.kernel_2))
        #SpaceTimeEncodings = torch.flatten(SpaceTimeEncodings,start_dim=2)
        # data_space_time = data#+SpaceTimeEncodings

        # Attention :  Below is implemented simplified FNO LAYER
        # # question : using only real gives better results than using real and imag in sum or concat manner?
        fft_data = torch.fft.fftn(data,norm='forward')
        extracted_data_modes = fft_data[:,:self.modes]
        padded_data_modes = torch.zeros_like(fft_data)
        padded_data_modes[:, :self.modes] = extracted_data_modes
        # question : is "bij,own->bin" give same outcome as "bij,own->bwj"?
        FFwithWeights = torch.einsum("bij,own->bin", padded_data_modes, self.weights_data_fft)

        # print(SpaceTimeEncodings.shape)
        #fft_space_time_encoding = torch.fft.fftn(SpaceTimeEncodings, norm='forward')
        #extracted_space_time_encoding_modes = fft_space_time_encoding[:, :self.modes]
        #padded_space_time_encoding_modes = torch.zeros_like(fft_space_time_encoding)
        #padded_space_time_encoding_modes[:, :self.modes] = extracted_space_time_encoding_modes
        #SpaceTimeEncFFwithWeights = torch.einsum("bijm,owkn->bwi", padded_space_time_encoding_modes, self.weights_space_time_encoding_fft)

        # plt.imshow(fft_data[4].real.cpu().tolist())
        # plt.show()
        # iSTFFWW = torch.fft.ifftn(SpaceTimeEncFFwithWeights, norm='forward')
        # iSTFFWW_real = iSTFFWW.real
        # iSTFFWW_imag = iSTFFWW.imag
        fft_dataWSpaceTime = FFwithWeights#+SpaceTimeEncFFwithWeights
        # plt.imshow(SpaceTimeEncFFwithWeights[4].real.cpu().tolist())
        # plt.show()
        iFFWW = torch.fft.ifftn(fft_dataWSpaceTime, norm='forward')
        iFFWW_real = iFFWW.real
        iFFWW_imag = iFFWW.imag

        skip = self.weights_data*data
        ifft_data = iFFWW_real+iFFWW_imag
        #ifft_space_time_encoding = iSTFFWW_real + iSTFFWW_imag
        data = torch.tanh(skip+ifft_data)#+ifft_space_time_encoding)
        # Attention :  Above is implemented simplified FNO LAYER
        # data = torch.tanh(data)
        return data




