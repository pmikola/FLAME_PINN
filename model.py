import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.nn.functional as f


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
        self.exponents = torch.arange(1, self.shifterCoefficients+1, 1,
                                      device=self.device)  # Check : from 0 to n or from 1 to n +1?

        # Definition of intermediate layer/parameters that transforms input into Fourier Feature with positional encoding and TODO: gaussian Gate
        self.weights_data_0 = nn.Parameter(torch.rand(5 * self.in_scale**2, dtype=torch.float))
        self.weights_data_1 = nn.Parameter(torch.rand(5 * self.in_scale**2, dtype=torch.float))
        self.weights_data_2 = nn.Parameter(torch.rand(5 * self.in_scale**2, dtype=torch.float))

        self.weights_data_fft_0 = nn.Parameter(torch.rand(5 * self.in_scale**2, dtype=torch.cfloat))
        self.weights_data_fft_1 = nn.Parameter(torch.rand(5 * self.in_scale**2, dtype=torch.cfloat))
        self.weights_data_fft_2 = nn.Parameter(torch.rand(5 * self.in_scale**2, dtype=torch.cfloat))

        self.multiplier = (torch.arange(start=1,end=105,step=1)).to(self.device)

        # Definition of Walsh-Hadamard rescale layers
        self.Walsh_Hadamard_rescaler_l0s0= nn.Linear(in_features=128, out_features=(5 * self.in_scale**2))


        # NOTE : on Hierarchy 0 flows data and on higher levels flows metas
        self.no_meta_h3 = 20 * 2
        self.no_meta_h2 = 32 * 2
        self.no_meta_h1 = 224 * 2
        self.dens_width = 10 * self.shifterCoefficients
        self.flat_size = 20*self.in_scale**2 # Note: n neurons per every pixel

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
        self.l1h01 = nn.Linear(in_features=self.dens_width, out_features=(self.in_scale**2) * self.shifterCoefficients*5)
        self.l2h01 = nn.Linear(in_features=self.dens_width*self.shifterCoefficients, out_features=(self.in_scale**2) * self.shifterCoefficients*5)

        # Definition of input layer 0 for lvl 0 in hierarchy
        # rmv of 3 paralleled layers for conv 1d k=1,2,3
        # Question : if we change same input to fft (k space) representation and
        #  change configuration of input not by kernel size but by modes
        #  from fft with learnable parameters - do we will have better results?
        # TODO : check if above query gives better result
        self.l0h0rx = nn.Linear(in_features=int(self.in_scale**2),out_features=int(self.in_scale**2))
        self.l0h0ry = nn.Linear(in_features=int(self.in_scale**2),out_features=int(self.in_scale**2))
        self.l0h0gx = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0gy = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0bx = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0by = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0ax = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0ay = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0sx = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0sy = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))


        # Definition of input layer 1,2,3 for lvl 0 in hierarchy
        self.l1h0 = nn.Linear(in_features=int(self.in_scale ** 2)*5,
                  out_features=int(self.in_scale ** 2)*5)
        self.l2h0 = nn.Linear(in_features=int(self.in_scale ** 2)*5,
                  out_features=int(self.in_scale ** 2)*5)
        self.l3h0 = nn.Linear(in_features=int(self.in_scale ** 2)*5,out_features=int(self.in_scale ** 2)*5)

        # Definition of the structure density distribution
        self.l0h0s = nn.Conv1d(in_channels=self.in_scale,
                              out_channels=self.no_subslice_in_tensors * self.in_scale, kernel_size=1)
        self.l1h0s = nn.Conv1d(in_channels=self.no_subslice_in_tensors * self.in_scale,
                              out_channels=int(self.in_scale), kernel_size=1)

        # Definition of Heads for red, green, blue, alpha and structure output channels
        self.l4_h0_r = nn.Linear(in_features=int(self.in_scale ** 2)*5, out_features=int(self.flat_size / 2))
        self.l4_h0_g = nn.Linear(in_features=int(self.in_scale ** 2)*5, out_features=int(self.flat_size / 2))
        self.l4_h0_b = nn.Linear(in_features=int(self.in_scale ** 2)*5, out_features=int(self.flat_size / 2))
        self.l4_h0_a = nn.Linear(in_features=int(self.in_scale ** 2)*5, out_features=int(self.flat_size / 2))
        self.l4_h0_s = nn.Linear(in_features=int(self.in_scale ** 2)*5, out_features=int(self.flat_size / 2))

        self.l5_h0_r = nn.Linear(in_features=int(self.flat_size/2),out_features=int(self.flat_size/2))
        self.l5_h0_g = nn.Linear(in_features=int(self.flat_size/2),out_features=int(self.flat_size/2))
        self.l5_h0_b = nn.Linear(in_features=int(self.flat_size/2),out_features=int(self.flat_size/2))
        self.l5_h0_a = nn.Linear(in_features=int(self.flat_size/2),out_features=int(self.flat_size/2))
        self.l5_h0_s = nn.Linear(in_features=int(self.flat_size/2),out_features=int(self.flat_size/2))

        self.l6_h0_r = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size/2))
        self.l6_h0_g = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size/2))
        self.l6_h0_b = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size/2))
        self.l6_h0_a = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size/2))
        self.l6_h0_s = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size/2))

        self.l7_h0_r = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size/2))
        self.l7_h0_g = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size/2))
        self.l7_h0_b = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size/2))
        self.l7_h0_a = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size/2))
        self.l7_h0_s = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size/2))

        self.l8_h0_r = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l8_h0_g = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l8_h0_b = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l8_h0_a = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l8_h0_s = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))

        self.l9_h0_r = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l9_h0_g = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l9_h0_b = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l9_h0_a = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l9_h0_s = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))

        self.l10_h0_r = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l10_h0_g = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l10_h0_b = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l10_h0_a = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l10_h0_s = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))

        self.l11_h0_r = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.in_scale ** 2), bias=True)
        self.l11_h0_g = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.in_scale ** 2), bias=True)
        self.l11_h0_b = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.in_scale ** 2), bias=True)
        self.l11_h0_a = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.in_scale ** 2), bias=True)
        self.l11_h0_s = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.in_scale ** 2), bias=True)
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
        #
        meta_step = torch.cat([meta_input_h2.float(), meta_output_h2.float()], dim=1)
        beta = torch.tanh(self.l0h2(meta_step))
        beta_l1 = self.shapeShift(torch.tanh(self.l1h2(beta)), gamma_l1)
        beta_l2 = self.shapeShift(torch.tanh(self.l2h2(beta_l1)), gamma_l2)

        meta_h1 = torch.cat([meta_input_h1.float(),meta_output_h1.float()],dim=1)
        alpha = torch.tanh(self.l0h1(meta_h1))
        alpha_l1 = self.shapeShift(torch.tanh(self.l1h1(alpha)),beta_l1)
        alpha_l2 =  self.shapeShift(torch.tanh(self.l2h1(alpha_l1)),beta_l2)

        x_alpha_l1 = torch.tanh(self.l1h01(alpha_l1))
        x_alpha_l2 = torch.tanh(self.l2h01(alpha_l2))

        r_along_x = data_input[:, 0:self.in_scale, :].view(self.batch_size, self.in_scale * self.in_scale)
        r_along_y = data_input[:, 0:self.in_scale, :].transpose(1, 2).contiguous().view(self.batch_size, self.in_scale * self.in_scale)
        r_along_x = f.gelu(self.l0h0rx(r_along_x))
        r_along_y = f.gelu(self.l0h0ry(r_along_y))

        g_along_x = data_input[:, self.in_scale:self.in_scale * 2, :].view(self.batch_size, self.in_scale * self.in_scale)
        g_along_y = data_input[:, self.in_scale:self.in_scale * 2, :].transpose(1, 2).contiguous().view(self.batch_size,self.in_scale * self.in_scale)
        g_along_x = f.gelu(self.l0h0gx(g_along_x))
        g_along_y = f.gelu(self.l0h0gy(g_along_y))

        b_along_x = data_input[:, self.in_scale * 2:self.in_scale * 3, :].view(self.batch_size, self.in_scale * self.in_scale)
        b_along_y = data_input[:, self.in_scale * 2:self.in_scale * 3, :].transpose(1, 2).contiguous().view(self.batch_size,self.in_scale * self.in_scale)
        b_along_x = f.gelu(self.l0h0bx(b_along_x))
        b_along_y = f.gelu(self.l0h0by(b_along_y))

        a_along_x = data_input[:, self.in_scale * 3:self.in_scale * 4, :].view(self.batch_size, self.in_scale * self.in_scale)
        a_along_y = data_input[:, self.in_scale * 3:self.in_scale * 4, :].transpose(1, 2).contiguous().view(self.batch_size,self.in_scale * self.in_scale)
        a_along_x = f.gelu(self.l0h0ax(a_along_x))
        a_along_y = f.gelu(self.l0h0ay(a_along_y))

        s_along_x = structure_input.view(self.batch_size, self.in_scale * self.in_scale)
        s_along_y = structure_input.transpose(1, 2).contiguous().view(self.batch_size,self.in_scale * self.in_scale)
        s_along_x = f.gelu(self.l0h0sx(s_along_x))
        s_along_y = f.gelu(self.l0h0sy(s_along_y))
        # print(r_along_x.shape,r_along_y.shape)
        rr = torch.cat([r_along_x*r_along_y],dim=1)
        gg = torch.cat([g_along_x*g_along_y],dim=1)
        bb = torch.cat([b_along_x*b_along_y],dim=1)
        aa = torch.cat([a_along_x*a_along_y],dim=1)
        ss = torch.cat([s_along_x*s_along_y],dim=1)
        rgbas = torch.cat([rr, gg, bb, aa, ss], dim=1)

        space_time = self.WalshHadamardSpaceTimeFeature(meta_central_points, meta_step)

        stff_in = torch.flatten(torch.cat([data_input,structure_input],dim=1),start_dim=1)
        x = self.SpaceTimeFFTFeature(stff_in,self.weights_data_0,self.weights_data_fft_0, space_time)
        x = self.SpaceTimeFFTFeature(x,self.weights_data_1,self.weights_data_fft_1, space_time)
        x = self.SpaceTimeFFTFeature(x,self.weights_data_2,self.weights_data_fft_2, space_time)

        # x = rgbas + x
        x_mod = self.shapeShift(torch.tanh(self.l1h0(rgbas)), x_alpha_l1)
        x_mod = self.shapeShift(torch.tanh(self.l2h0(x_mod)), x_alpha_l2)

        x = f.gelu(self.l3h0(x))+rgbas+x_mod
        rres = f.gelu(self.l4_h0_r(x))
        gres = f.gelu(self.l4_h0_g(x))
        bres = f.gelu(self.l4_h0_b(x))
        ares = f.gelu(self.l4_h0_a(x))
        sres = f.gelu(self.l4_h0_s(x))

        r = f.gelu(self.l5_h0_r(rres))
        g = f.gelu(self.l5_h0_g(gres))
        b = f.gelu(self.l5_h0_b(bres))
        a = f.gelu(self.l5_h0_a(ares))
        s = f.gelu(self.l5_h0_s(sres))

        r = f.gelu(self.l6_h0_r(r))
        g = f.gelu(self.l6_h0_g(g))
        b = f.gelu(self.l6_h0_b(b))
        a = f.gelu(self.l6_h0_a(a))
        s = f.gelu(self.l6_h0_s(s))

        rres = f.gelu(self.l7_h0_r(r))+rres
        gres = f.gelu(self.l7_h0_g(g))+gres
        bres = f.gelu(self.l7_h0_b(b))+bres
        ares = f.gelu(self.l7_h0_a(a))+ares
        sres = f.gelu(self.l7_h0_s(s))+sres

        r = f.gelu(self.l8_h0_r(rres))
        g = f.gelu(self.l8_h0_g(gres))
        b = f.gelu(self.l8_h0_b(bres))
        a = f.gelu(self.l8_h0_a(ares))
        s = f.gelu(self.l8_h0_s(sres))
        #
        # r = f.gelu(self.l9_h0_r(r))
        # g = f.gelu(self.l9_h0_g(g))
        # b = f.gelu(self.l9_h0_b(b))
        # a = f.gelu(self.l9_h0_a(a))
        # s = f.gelu(self.l9_h0_s(s))
        #
        # r = f.gelu(self.l10_h0_r(r))+rres
        # g = f.gelu(self.l10_h0_g(g))+gres
        # b = f.gelu(self.l10_h0_b(b))+bres
        # a = f.gelu(self.l10_h0_a(a))+ares
        # s = f.gelu(self.l10_h0_s(s))+sres

        r = self.l11_h0_r(r).reshape(self.batch_size, self.in_scale, self.in_scale)
        g = self.l11_h0_g(g).reshape(self.batch_size, self.in_scale, self.in_scale)
        b = self.l11_h0_b(b).reshape(self.batch_size, self.in_scale, self.in_scale)
        a = self.l11_h0_a(a).reshape(self.batch_size, self.in_scale, self.in_scale)
        s = self.l11_h0_s(s).reshape(self.batch_size, self.in_scale, self.in_scale)
        return r,g,b,a,s


    def shapeShift(self,x, h):
        if x.dim() == 3:
            coefficients = h.reshape(self.batch_size,x.shape[1],x.shape[2],self.shifterCoefficients)
            x_powers = torch.pow(x[0:self.batch_size,:,:].unsqueeze(3), self.exponents.unsqueeze(0).unsqueeze(1))
            craftedPolynomial = torch.sum((coefficients)*x_powers,dim=3)
            return craftedPolynomial
        elif x.dim() == 2:
            coefficients = h.reshape(self.batch_size,x.shape[1],self.shifterCoefficients)
            x_powers = torch.pow(x[0:self.batch_size, :].unsqueeze(2), self.exponents.unsqueeze(0).unsqueeze(1))
            craftedPolynomial = torch.sum((coefficients) * x_powers, dim=2)
            return craftedPolynomial
        elif x.dim() == 4:
            coefficients = h.reshape(self.batch_size, x.shape[1], x.shape[2], x.shape[3], self.shifterCoefficients)
            x_powers = torch.pow(x[0:self.batch_size, :, :, :].unsqueeze(4),self.exponents.unsqueeze(0).unsqueeze(1).unsqueeze(2))
            craftedPolynomial = torch.sum((coefficients) * x_powers, dim=4)
            return craftedPolynomial
        else:
            raise ValueError("Unsupported input dimensions")

    def SpaceTimeFFTFeature(self,data,weights_data,weights_data_fft, space_time):
        # Attention :  Below is implemented simplified FNO LAYER
        # # question : using only real gives better results than using real and imag in sum or concat manner?
        fft_data = torch.fft.fft(data,norm='forward')
        fft_data /= torch.tensor(fft_data.shape[1]) # normalize
        data_modes = fft_data + space_time

        # question : is "bij,own->bin" give same outcome as "bij,own->bwj"?
        FFwithWeights = torch.einsum("bi,j->bj",data_modes, weights_data_fft)
        iFFWW = torch.fft.ifft(FFwithWeights, norm='forward')
        iFFWW_real = iFFWW.real
        iFFWW_imag = iFFWW.imag
        ifft_data = iFFWW_real+iFFWW_imag
        data = f.gelu(ifft_data)+f.gelu(weights_data*data)
        # Attention :  Above is implemented simplified FNO LAYER
        # data = torch.tanh(data)
        return data

    def WalshHadamardSpaceTimeFeature(self,meta_central_points, meta_step):
        # NOTE: Walsh-Hadamard transform for space and time coding
        space_time = torch.cat([meta_central_points, meta_step], dim=1)
        bit_padding = torch.zeros((self.batch_size, 128 - space_time.shape[1])).to(self.device)
        space_time = torch.cat([space_time, bit_padding], dim=1)
        length = space_time.shape[1]
        assert (length & (length - 1)) == 0, "Length must be a power of 2"
        bit = length
        len_tens = torch.tensor(length)
        stages = torch.log2(len_tens)
        stages = torch.arange(0, stages.int())
        for _ in stages:
            bit >>= 1
            indices = torch.arange(length).view(1, -1)
            indices_i = (indices & ~bit).flatten()
            indices_j = (indices | bit).flatten()
            result_i = space_time[:, indices_i]
            result_j = space_time[:, indices_j]
            space_time[:, indices_i] = result_i + result_j
            space_time[:, indices_j] = result_i - result_j

        space_time /= len_tens  # normalize
        space_time = self.Walsh_Hadamard_rescaler_l0s0(space_time)
        space_time = f.gelu(space_time)
        return space_time


