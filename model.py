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
        self.activation_weight = nn.Parameter(torch.rand(1, dtype=torch.float))

        # Definition of non-linear shifting activation function with parameters
        self.shifterCoefficients = 4  # No. of polynomial coefficients
        self.exponents = torch.arange(1, self.shifterCoefficients+1, 1,
                                      device=self.device)  # Check : from 0 to n or from 1 to n +1?

        # Definition of intermediate layer/parameters that transforms input into Fourier Feature with positional encoding and TODO: gaussian Gate
        self.weights_data_0 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.float))
        self.weights_data_1 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.float))
        self.weights_data_2 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.float))
        self.weights_data_3 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.float))
        self.weights_data_4 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.float))
        self.weights_data_5 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.float))
        self.weights_data_6 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.float))
        self.weights_data_7 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.float))
        self.weights_data_8 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.float))
        self.weights_data_9 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.float))

        self.weights_data_fft_0 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.cfloat))
        self.weights_data_fft_1 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.cfloat))
        self.weights_data_fft_2 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.cfloat))
        self.weights_data_fft_3 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.cfloat))
        self.weights_data_fft_4 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.cfloat))
        self.weights_data_fft_5 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.cfloat))
        self.weights_data_fft_6 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.cfloat))
        self.weights_data_fft_7 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.cfloat))
        self.weights_data_fft_8 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.cfloat))
        self.weights_data_fft_9 = nn.Parameter(torch.rand((self.in_scale,self.in_scale,5), dtype=torch.cfloat))

        self.space_time_0 = nn.Parameter(torch.rand((self.in_scale, self.in_scale, 5), dtype=torch.cfloat))
        self.space_time_1 = nn.Parameter(torch.rand((self.in_scale, self.in_scale, 5), dtype=torch.cfloat))
        self.space_time_2 = nn.Parameter(torch.rand((self.in_scale, self.in_scale, 5), dtype=torch.cfloat))
        self.space_time_3 = nn.Parameter(torch.rand((self.in_scale, self.in_scale, 5), dtype=torch.cfloat))
        self.space_time_4 = nn.Parameter(torch.rand((self.in_scale, self.in_scale, 5), dtype=torch.cfloat))
        self.space_time_5 = nn.Parameter(torch.rand((self.in_scale, self.in_scale, 5), dtype=torch.cfloat))
        self.space_time_6 = nn.Parameter(torch.rand((self.in_scale, self.in_scale, 5), dtype=torch.cfloat))
        self.space_time_7 = nn.Parameter(torch.rand((self.in_scale, self.in_scale, 5), dtype=torch.cfloat))
        self.space_time_8 = nn.Parameter(torch.rand((self.in_scale, self.in_scale, 5), dtype=torch.cfloat))
        self.space_time_9 = nn.Parameter(torch.rand((self.in_scale, self.in_scale, 5), dtype=torch.cfloat))

        # Definition of Walsh-Hadamard rescale layers
        self.Walsh_Hadamard_rescaler_l0wh= nn.Linear(in_features=256, out_features=(self.in_scale**2))
        self.Walsh_Hadamard_rescaler_l1wh= nn.Linear(in_features=(self.in_scale**2), out_features=(self.in_scale**2))

        # NOTE : on Hierarchy 0 flows data and on higher levels flows metas
        self.no_meta_h3 = 20 * 2
        self.no_meta_h2 = 32 * 2
        self.no_meta_h1 = 224 * 2
        self.dens_width = 2 * self.shifterCoefficients
        self.flat_size = 10*self.in_scale**2 # Note: n neurons per every pixel
        self.diffiusion_context = 32*2

        # Definition of layer 0,1,2 for lvl 4 in hierarchy - theta - diffusion noise context
        self.l0h4 = nn.Linear(in_features=self.diffiusion_context,
                              out_features=self.dens_width * self.shifterCoefficients ** 2).to(torch.cfloat)
        self.l1h4 = nn.Linear(in_features=self.dens_width * self.shifterCoefficients ** 2,
                              out_features=self.dens_width * self.shifterCoefficients ** 4).to(torch.cfloat)
        self.l2h4 = nn.Linear(in_features=self.dens_width * self.shifterCoefficients ** 4,
                              out_features=self.dens_width * self.shifterCoefficients ** 4).to(torch.cfloat)

        # Definition of layer 0,1,2 for lvl 3 in hierarchy - gamma
        self.l0h3 = nn.Linear(in_features=self.no_meta_h3, out_features=self.dens_width * self.shifterCoefficients).to(torch.cfloat)
        self.l1h3 = nn.Linear(in_features=self.dens_width * self.shifterCoefficients,
                              out_features=self.dens_width * self.shifterCoefficients ** 3).to(torch.cfloat)
        self.l2h3 = nn.Linear(in_features=self.dens_width * self.shifterCoefficients ** 3,
                              out_features=self.dens_width * self.shifterCoefficients ** 3).to(torch.cfloat)

        # Definition of layer 0,1,2 for lvl 2 in hierarchy - beta
        self.l0h2 = nn.Linear(in_features=self.no_meta_h2, out_features=self.dens_width * self.shifterCoefficients).to(torch.cfloat)
        self.l1h2 = nn.Linear(in_features=self.dens_width * self.shifterCoefficients,
                              out_features=self.dens_width * self.shifterCoefficients ** 2).to(torch.cfloat)
        self.l2h2 = nn.Linear(in_features=self.dens_width * self.shifterCoefficients ** 2,
                              out_features=self.dens_width * self.shifterCoefficients ** 2).to(torch.cfloat)

        # Definition of layer 0,1,2 for lvl 1 in hierarchy - alpha
        self.l0h1 = nn.Linear(in_features=self.no_meta_h1,
                              out_features=int((self.dens_width * self.shifterCoefficients))).to(torch.cfloat)
        self.l1h1 = nn.Linear(in_features=int((self.dens_width * self.shifterCoefficients)),
                              out_features=int(self.dens_width * self.shifterCoefficients ** 1)).to(torch.cfloat)
        self.l2h1 = nn.Linear(in_features=int(self.dens_width * self.shifterCoefficients ** 1),
                              out_features=int(self.dens_width * self.shifterCoefficients ** 1)).to(torch.cfloat)

        # Definition of intermediate layer between lvl 0 and 1 for dimension matching
        self.l1h01 = nn.Linear(in_features=int(self.dens_width * self.shifterCoefficients ** 1),
                               out_features=(self.in_scale ** 2) * self.shifterCoefficients * 5).to(torch.cfloat)
        self.l2h01 = nn.Linear(in_features=int(self.dens_width * self.shifterCoefficients ** 1),
                               out_features=(self.in_scale ** 2) * self.shifterCoefficients * 5).to(torch.cfloat)

        # Definition of input layer 0 for lvl 0 in hierarchy
        # rmv of 3 paralleled layers for conv 1d k=1,2,3
        # Question : if we change same input to fft (k space) representation and
        #  change configuration of input not by kernel size but by modes
        #  from fft with learnable parameters - do we will have better results?
        # TODO : check if above query gives better result
        self.l0h0rx = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0ry = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0gx = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0gy = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0bx = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0by = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0ax = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0ay = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0sx = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))
        self.l0h0sy = nn.Linear(in_features=int(self.in_scale ** 2), out_features=int(self.in_scale ** 2))

        # Definition of input layer 1,2,3 for lvl 0 in hierarchy
        self.l1h0 = nn.Linear(in_features=int(self.in_scale ** 2) * 5,
                              out_features=int(self.in_scale ** 2) * 5).to(torch.cfloat)
        self.l2h0 = nn.Linear(in_features=int(self.in_scale ** 2) * 5,
                              out_features=int(self.in_scale ** 2) * 5).to(torch.cfloat)
        self.l3h0 = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.in_scale ** 2) * 5).to(torch.cfloat)

        # Definition of Heads for red, green, blue, alpha and structure output channels
        self.l4_h0_r = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.flat_size / 2))
        self.l4_h0_g = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.flat_size / 2))
        self.l4_h0_b = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.flat_size / 2))
        self.l4_h0_a = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.flat_size / 2))
        self.l4_h0_s = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.flat_size / 2))

        self.l5_h0_r = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l5_h0_g = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l5_h0_b = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l5_h0_a = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l5_h0_s = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))

        self.l6_h0_r = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l6_h0_g = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l6_h0_b = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l6_h0_a = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l6_h0_s = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))

        self.l7_h0_r = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l7_h0_g = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l7_h0_b = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l7_h0_a = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l7_h0_s = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))

        self.l8_h0_r = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l8_h0_g = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l8_h0_b = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l8_h0_a = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))
        self.l8_h0_s = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.flat_size / 2))

        self.l9_h0_r = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.in_scale ** 2), bias=True)
        self.l9_h0_g = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.in_scale ** 2), bias=True)
        self.l9_h0_b = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.in_scale ** 2), bias=True)
        self.l9_h0_a = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.in_scale ** 2), bias=True)
        self.l9_h0_s = nn.Linear(in_features=int(self.flat_size / 2), out_features=int(self.in_scale ** 2), bias=True)
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

    def forward(self, din):
        (data_input, structure_input, meta_input_h1, meta_input_h2, meta_input_h3,
         meta_input_h4, meta_input_h5, noise_var_in, meta_output_h1, meta_output_h2,
         meta_output_h3, meta_output_h4, meta_output_h5, noise_var_out) = din

        # Question : Do highest hierarchy should have parameters that are learning
        #  or just be top layer without any additional coefss (regarding polyNonlinear)
        meta_central_points = torch.cat([meta_input_h3.float(), meta_output_h3.float()], dim=1)
        noise_var = torch.cat([noise_var_in, noise_var_out], dim=1)
        meta_step = torch.cat([meta_input_h2.float(), meta_output_h2.float()], dim=1)
        meta_h1 = torch.cat([meta_input_h1.float(), meta_output_h1.float()], dim=1)

        theta = self.activate(self.l0h4(noise_var.to(torch.cfloat)))
        theta_l1 = self.activate(self.l1h4(theta))
        theta_l2 = self.activate(self.l2h4(theta_l1))

        gamma = self.activate(self.l0h3(meta_central_points.to(torch.cfloat)))
        gamma_l1 = self.shapeShift(self.l1h3(gamma), theta_l1)
        gamma_l2 = self.shapeShift(self.l2h3(gamma_l1), theta_l2)
        #
        beta = self.activate(self.l0h2(meta_step.to(torch.cfloat)))
        beta_l1 = self.shapeShift(self.l1h2(beta), gamma_l1)
        beta_l2 = self.shapeShift(self.l2h2(beta_l1), gamma_l2)

        alpha = self.activate(self.l0h1(meta_h1.to(torch.cfloat)))
        alpha_l1 = self.shapeShift(self.l1h1(alpha), beta_l1)
        alpha_l2 = self.shapeShift(self.l2h1(alpha_l1), beta_l2)

        x_alpha_l1 = self.activate(self.l1h01(alpha_l1))
        x_alpha_l2 = self.activate(self.l2h01(alpha_l2))

        # Note: Factorisation for dense layers

        rr = data_input[:, 0:self.in_scale, :]
        gg = data_input[:, self.in_scale:self.in_scale * 2, :]
        bb = data_input[:, self.in_scale * 2:self.in_scale * 3, :]
        aa = data_input[:, self.in_scale * 3:self.in_scale * 4, :]
        ss = structure_input
        r_along_x = rr.view(self.batch_size, self.in_scale * self.in_scale)
        r_along_y = rr.transpose(1, 2).contiguous().view(self.batch_size,self.in_scale * self.in_scale)
        r_along_x = self.activate(self.l0h0rx(r_along_x)).view(self.batch_size,self.in_scale , self.in_scale)
        r_along_y = self.activate(self.l0h0ry(r_along_y)).view(self.batch_size,self.in_scale , self.in_scale)

        g_along_x = gg.view(self.batch_size,self.in_scale * self.in_scale)
        g_along_y = gg.transpose(1, 2).contiguous().view(self.batch_size,self.in_scale * self.in_scale)
        g_along_x = self.activate(self.l0h0gx(g_along_x)).view(self.batch_size,self.in_scale , self.in_scale)
        g_along_y = self.activate(self.l0h0gy(g_along_y)).view(self.batch_size,self.in_scale , self.in_scale)

        b_along_x = bb.view(self.batch_size,self.in_scale * self.in_scale)
        b_along_y = bb.transpose(1, 2).contiguous().view(
            self.batch_size, self.in_scale * self.in_scale)
        b_along_x = self.activate(self.l0h0bx(b_along_x)).view(self.batch_size,self.in_scale , self.in_scale)
        b_along_y = self.activate(self.l0h0by(b_along_y)).view(self.batch_size,self.in_scale , self.in_scale)

        a_along_x = aa.view(self.batch_size,self.in_scale * self.in_scale)
        a_along_y = aa.transpose(1, 2).contiguous().view(
            self.batch_size, self.in_scale * self.in_scale)
        a_along_x = self.activate(self.l0h0ax(a_along_x)).view(self.batch_size,self.in_scale , self.in_scale)
        a_along_y = self.activate(self.l0h0ay(a_along_y)).view(self.batch_size,self.in_scale , self.in_scale)

        s_along_x = ss.view(self.batch_size, self.in_scale * self.in_scale)
        s_along_y = ss.transpose(1, 2).contiguous().view(self.batch_size, self.in_scale * self.in_scale)
        s_along_x = self.activate(self.l0h0sx(s_along_x)).view(self.batch_size,self.in_scale , self.in_scale)
        s_along_y = self.activate(self.l0h0sy(s_along_y)).view(self.batch_size,self.in_scale , self.in_scale)

        stff_in = torch.cat([rr.unsqueeze(-1), gg.unsqueeze(-1), bb.unsqueeze(-1), aa.unsqueeze(-1), ss.unsqueeze(-1)], dim=3)

        rr = r_along_x * r_along_y
        gg = g_along_x * g_along_y
        bb = b_along_x * b_along_y
        aa = a_along_x * a_along_y
        ss = s_along_x * s_along_y
        rgbas_prod = torch.cat([rr.unsqueeze(-1), gg.unsqueeze(-1), bb.unsqueeze(-1), aa.unsqueeze(-1), ss.unsqueeze(-1)], dim=3)
        space_time = self.WalshHadamardSpaceTimeFeature(meta_central_points, meta_step, noise_var)
        x0 = self.SpaceTimeFFTFeature(stff_in, self.weights_data_0, self.weights_data_fft_0,self.space_time_0, space_time)
        x1 = self.SpaceTimeFFTFeature(rgbas_prod, self.weights_data_1, self.weights_data_fft_1,self.space_time_1, space_time)
        x0 = self.SpaceTimeFFTFeature(x0, self.weights_data_2, self.weights_data_fft_2,self.space_time_2, space_time)
        x1 = self.SpaceTimeFFTFeature(x1, self.weights_data_3, self.weights_data_fft_3,self.space_time_3, space_time)
        x0 = self.SpaceTimeFFTFeature(x0, self.weights_data_4, self.weights_data_fft_4,self.space_time_4, space_time)
        x1 = self.SpaceTimeFFTFeature(x1, self.weights_data_5, self.weights_data_fft_5,self.space_time_5, space_time)
        x0 = self.SpaceTimeFFTFeature(x0, self.weights_data_6, self.weights_data_fft_6,self.space_time_6, space_time)
        x1 = self.SpaceTimeFFTFeature(x1, self.weights_data_7, self.weights_data_fft_7,self.space_time_7, space_time)

        x = x0 + x1
        x = torch.flatten(x,start_dim=1)
        rgbas_prod = torch.flatten(rgbas_prod,start_dim=1)
        x_mod = self.shapeShift(self.l1h0(rgbas_prod.to(torch.cfloat)), x_alpha_l1)
        x_mod = self.shapeShift(self.l2h0(x_mod), x_alpha_l2)

        x = self.activate(self.l3h0(x_mod))+x+rgbas_prod
        x_real = x.real
        x_imag = x.imag
        x = x_real + x_imag
        rres = self.activate(self.l4_h0_r(x))
        gres = self.activate(self.l4_h0_g(x))
        bres = self.activate(self.l4_h0_b(x))
        ares = self.activate(self.l4_h0_a(x))
        sres = self.activate(self.l4_h0_s(x))

        r = self.activate(self.l5_h0_r(rres))
        g = self.activate(self.l5_h0_g(gres))
        b = self.activate(self.l5_h0_b(bres))
        a = self.activate(self.l5_h0_a(ares))
        s = self.activate(self.l5_h0_s(sres))

        r = self.activate(self.l6_h0_r(r))
        g = self.activate(self.l6_h0_g(g))
        b = self.activate(self.l6_h0_b(b))
        a = self.activate(self.l6_h0_a(a))
        s = self.activate(self.l6_h0_s(s))

        rres = self.activate(self.l7_h0_r(r))+rres
        gres = self.activate(self.l7_h0_g(g))+gres
        bres = self.activate(self.l7_h0_b(b))+bres
        ares = self.activate(self.l7_h0_a(a))+ares
        sres = self.activate(self.l7_h0_s(s))+sres

        r = self.activate(self.l8_h0_r(rres))
        g = self.activate(self.l8_h0_g(gres))
        b = self.activate(self.l8_h0_b(bres))
        a = self.activate(self.l8_h0_a(ares))
        s = self.activate(self.l8_h0_s(sres))

        r = self.l9_h0_r(r).reshape(self.batch_size, self.in_scale, self.in_scale)
        g = self.l9_h0_g(g).reshape(self.batch_size, self.in_scale, self.in_scale)
        b = self.l9_h0_b(b).reshape(self.batch_size, self.in_scale, self.in_scale)
        a = self.l9_h0_a(a).reshape(self.batch_size, self.in_scale, self.in_scale)
        s = self.l9_h0_s(s).reshape(self.batch_size, self.in_scale, self.in_scale)
        return r,g,b,a,s


    def shapeShift(self,x, h, ):
        if x.dim() == 3:
            coefficients = h.reshape(self.batch_size,x.shape[1],x.shape[2],self.shifterCoefficients)
            x_powers = torch.pow(x[0:self.batch_size,:,:].unsqueeze(3), self.exponents.unsqueeze(0).unsqueeze(1))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=3)
            craftedPolynomial = self.activate(craftedPolynomial)
            return craftedPolynomial
        elif x.dim() == 2:
            s = h.reshape(self.batch_size,x.shape[1],self.shifterCoefficients) # Note: Conversion steering of the transform
            t = x.unsqueeze(-1) # TODO make an meaningful t
            dt = t[:,0:t.shape[1]-1,:] - t[:,1:t.shape[1]]
            dt_last = dt[:, -1, :].unsqueeze(1)
            dt = torch.cat([dt, dt_last], dim=1)
            expotential = torch.exp(-s*t)
            ffunc = t * expotential[:,:,0:self.shifterCoefficients] * dt
            ffunc_last = ffunc[:,-1,:].unsqueeze(1)
            ffunc = torch.cat([ffunc,ffunc_last],dim=1)
            FLaplace = torch.cumulative_trapezoid(ffunc,dx=float(1/255),dim=-2)
            FLaplace = torch.sum(FLaplace,dim=-1)
            craftedFLaplace = self.activate(FLaplace)
            return craftedFLaplace
        elif x.dim() == 4:
            coefficients = h.reshape(self.batch_size, x.shape[1], x.shape[2], x.shape[3], self.shifterCoefficients)
            x_powers = torch.pow(x[0:self.batch_size, :, :, :].unsqueeze(4),self.exponents.unsqueeze(0).unsqueeze(1).unsqueeze(2))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=4)
            craftedPolynomial = self.activate(craftedPolynomial)
            return craftedPolynomial
        else:
            raise ValueError("Unsupported input dimensions")

    def SpaceTimeFFTFeature(self,data,weights_data,weights_data_fft,weights_space_time_noise, space_time_noise):
        # Attention :  Below is implemented simplified FNO LAYER
        fft_data = torch.fft.fftn(data*weights_data,dim=(1,2,3),norm='forward')
        FFTwithW = torch.einsum("bijk,nmo->bimo",fft_data, weights_data_fft)
        Wspace_time_noise = space_time_noise * weights_space_time_noise
        FFTWeightSpaceTimeNoise = torch.einsum("bimo,bprs->birs",FFTwithW,Wspace_time_noise)
        iFFW= torch.fft.ifftn(FFTWeightSpaceTimeNoise,dim=(1,2,3), norm='forward')
        data = self.activate(iFFW)+data
        # Attention :  Above is implemented simplified FNO LAYER
        dimag = data.imag
        dreal = data.real
        data = dimag + dreal
        # data = torch.tanh(data)
        return data

    def WalshHadamardSpaceTimeFeature(self,meta_central_points, meta_step,noise_var):
        # NOTE: Walsh-Hadamard transform for space and time coding
        space_time = torch.cat([meta_central_points, meta_step,noise_var], dim=1)
        bit_padding = torch.zeros((self.batch_size, 256 - space_time.shape[1])).to(self.device)
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
        space_time = self.activate(self.Walsh_Hadamard_rescaler_l0wh(space_time))
        space_time = self.activate(self.Walsh_Hadamard_rescaler_l1wh(space_time))
        return space_time.view(self.batch_size,self.in_scale , self.in_scale).unsqueeze(-1).real

    def activate(self,x):
        return torch.tanh(x)#*2#*self.activation_weight

