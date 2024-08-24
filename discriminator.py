import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.nn.functional as f


class Metamorph_discriminator(nn.Module):
    def __init__(self, no_frame_samples, batch_size, input_window_size, device):
        super(Metamorph_discriminator, self).__init__()
        self.device = device
        self.no_frame_samples = no_frame_samples
        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.in_scale = (1 + self.input_window_size * 2)
        self.no_subslice_in_tensors = 4
        self.in_data = 20
        self.activation_weight = nn.Parameter(torch.rand(1, dtype=torch.float))

        # Definition of non-linear shifting activation function with parameters
        self.shifterCoefficients = 2  # No. of polynomial coefficients
        self.exponents = torch.arange(1, self.shifterCoefficients + 1, 1,
                                      device=self.device)  # Check : from 0 to n or from 1 to n +1?

        # Definition of intermediate layer/parameters that transforms input into Fourier Feature with positional encoding and TODO: gaussian Gate
        self.weights_data_0 = nn.Parameter(torch.rand(5 * self.in_scale ** 2, dtype=torch.float))
        self.weights_data_1 = nn.Parameter(torch.rand(5 * self.in_scale ** 2, dtype=torch.float))
        self.weights_data_2 = nn.Parameter(torch.rand(5 * self.in_scale ** 2, dtype=torch.float))
        self.weights_data_3 = nn.Parameter(torch.rand(5 * self.in_scale ** 2, dtype=torch.float))
        self.weights_data_4 = nn.Parameter(torch.rand(5 * self.in_scale ** 2, dtype=torch.float))
        self.weights_data_5 = nn.Parameter(torch.rand(5 * self.in_scale ** 2, dtype=torch.float))

        self.weights_data_fft_0 = nn.Parameter(torch.rand(5 * self.in_scale ** 2, dtype=torch.cfloat))
        self.weights_data_fft_1 = nn.Parameter(torch.rand(5 * self.in_scale ** 2, dtype=torch.cfloat))
        self.weights_data_fft_2 = nn.Parameter(torch.rand(5 * self.in_scale ** 2, dtype=torch.cfloat))
        self.weights_data_fft_3 = nn.Parameter(torch.rand(5 * self.in_scale ** 2, dtype=torch.cfloat))
        self.weights_data_fft_4 = nn.Parameter(torch.rand(5 * self.in_scale ** 2, dtype=torch.cfloat))
        self.weights_data_fft_5 = nn.Parameter(torch.rand(5 * self.in_scale ** 2, dtype=torch.cfloat))

        # Definition of Walsh-Hadamard rescale layers
        self.Walsh_Hadamard_rescaler_l0wh = nn.Linear(in_features=256, out_features=(5 * self.in_scale ** 2))
        self.Walsh_Hadamard_rescaler_l1wh = nn.Linear(in_features=(5 * self.in_scale ** 2),
                                                      out_features=(5 * self.in_scale ** 2))

        # NOTE : on Hierarchy 0 flows data and on higher levels flows metas
        self.no_meta_h3 = 20 * 2
        self.no_meta_h2 = 32 * 2
        self.no_meta_h1 = 224 * 2
        self.dens_width = self.shifterCoefficients
        self.flat_size = 1 * self.in_scale ** 2  # Note: n neurons per every pixel
        self.diffiusion_context = 32 * 2
        self.flow_matching_context = 32

        # Definition of layer 0,1,2 for lvl 4 in hierarchy - theta - diffusion noise context
        self.l0h4 = nn.Linear(in_features=self.diffiusion_context + self.flow_matching_context,
                              out_features=self.shifterCoefficients * self.shifterCoefficients ** 2)
        self.l1h4 = nn.Linear(in_features=self.shifterCoefficients * self.shifterCoefficients ** 2,
                              out_features=self.shifterCoefficients * self.shifterCoefficients ** 3)
        self.l2h4 = nn.Linear(in_features=self.shifterCoefficients * self.shifterCoefficients ** 3,
                              out_features=self.shifterCoefficients * self.shifterCoefficients ** 3)

        # Definition of layer 0,1,2 for lvl 3 in hierarchy - gamma
        self.l0h3 = nn.Linear(in_features=self.no_meta_h3,
                              out_features=self.shifterCoefficients * self.shifterCoefficients)
        self.l1h3 = nn.Linear(in_features=self.shifterCoefficients * self.shifterCoefficients,
                              out_features=self.shifterCoefficients * self.shifterCoefficients ** 2)
        self.l2h3 = nn.Linear(in_features=self.shifterCoefficients * self.shifterCoefficients ** 2,
                              out_features=self.shifterCoefficients * self.shifterCoefficients ** 2)

        # Definition of layer 0,1,2 for lvl 2 in hierarchy - beta
        self.l0h2 = nn.Linear(in_features=self.no_meta_h2,
                              out_features=self.shifterCoefficients * self.shifterCoefficients)
        self.l1h2 = nn.Linear(in_features=self.shifterCoefficients * self.shifterCoefficients,
                              out_features=self.shifterCoefficients * self.shifterCoefficients ** 1)
        self.l2h2 = nn.Linear(in_features=self.shifterCoefficients * self.shifterCoefficients ** 1,
                              out_features=self.shifterCoefficients * self.shifterCoefficients ** 1)

        # Definition of layer 0,1,2 for lvl 1 in hierarchy - alpha
        self.l0h1 = nn.Linear(in_features=self.no_meta_h1,
                              out_features=int((self.shifterCoefficients * self.shifterCoefficients) ** (1 / 2)))
        self.l1h1 = nn.Linear(in_features=int((self.shifterCoefficients * self.shifterCoefficients) ** (1 / 2)),
                              out_features=int((self.shifterCoefficients * self.shifterCoefficients) ** (1 / 2)))
        self.l2h1 = nn.Linear(in_features=int((self.shifterCoefficients * self.shifterCoefficients) ** (1 / 2)),
                              out_features=int((self.shifterCoefficients * self.shifterCoefficients) ** (1 / 2)))

        # Definition of intermediate layer between lvl 0 and 1 for dimension matching
        self.l1h01 = nn.Linear(in_features=int((self.shifterCoefficients * self.shifterCoefficients) ** (1 / 2)),
                               out_features=(self.in_scale ** 2) * self.shifterCoefficients * 5)
        self.l2h01 = nn.Linear(in_features=int((self.shifterCoefficients * self.shifterCoefficients) ** (1 / 2)),
                               out_features=(self.in_scale ** 2) * self.shifterCoefficients * 5)

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
                              out_features=int(self.in_scale ** 2) * 5)
        self.l2h0 = nn.Linear(in_features=int(self.in_scale ** 2) * 5,
                              out_features=int(self.in_scale ** 2) * 5)
        self.l3h0 = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.in_scale ** 2) * 5)

        # Definition of Heads for red, green, blue, alpha and structure output channels
        self.l4_h0_r = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.in_scale ** 2))
        self.l4_h0_g = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.in_scale ** 2))
        self.l4_h0_b = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.in_scale ** 2))
        self.l4_h0_a = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.in_scale ** 2))
        self.l4_h0_s = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=int(self.in_scale ** 2))

        self.l12_disc_output = nn.Linear(in_features=int(self.in_scale ** 2) * 5, out_features=1, bias=True)
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

    def forward(self, disc_data, g_model_data, shuffle_idx):
        (_, _, meta_input_h1, meta_input_h2, meta_input_h3,
         _, _, noise_var_in, fmot_in, meta_output_h1, meta_output_h2,
         meta_output_h3, _, _, noise_var_out) = g_model_data

        meta_input_h2 = torch.cat([meta_input_h2, meta_input_h2], dim=0)[shuffle_idx]
        meta_input_h3 = torch.cat([meta_input_h3, meta_input_h3], dim=0)[shuffle_idx]
        noise_var_in = torch.cat([noise_var_in, fmot_in, noise_var_in], dim=0)[shuffle_idx]
        meta_output_h2 = torch.cat([meta_output_h2, meta_output_h2], dim=0)[shuffle_idx]
        meta_output_h3 = torch.cat([meta_output_h3, meta_output_h3], dim=0)[shuffle_idx]
        noise_var_out = torch.cat([noise_var_out, noise_var_out], dim=0)[shuffle_idx]

        meta_central_points = torch.cat([meta_input_h3.float(), meta_output_h3.float()], dim=1)
        noise_var = torch.cat([noise_var_in, noise_var_out], dim=1)
        meta_step = torch.cat([meta_input_h2.float(), meta_output_h2.float()], dim=1)
        noise_variance = 0.1
        disc_data = disc_data[shuffle_idx] + torch.nan_to_num(noise_variance * torch.rand_like(disc_data[shuffle_idx]),nan=0.0)
        space_time = self.WalshHadamardSpaceTimeFeature(meta_central_points, meta_step, noise_var)
        stff_in = torch.flatten(disc_data, start_dim=1)
        x = self.SpaceTimeFFTFeature(stff_in, self.weights_data_0, self.weights_data_fft_0, space_time)
        # Note : Delete space x time x noise contexts of generator data

        x = self.activate(self.l3h0(x))
        r = self.activate(self.l4_h0_r(x))
        g = self.activate(self.l4_h0_g(x))
        b = self.activate(self.l4_h0_b(x))
        a = self.activate(self.l4_h0_a(x))
        s = self.activate(self.l4_h0_s(x))
        out = torch.cat([r, g, b, a, s], dim=1)
        out = torch.sigmoid(self.l12_disc_output(out))
        return out

    def shapeShift(self, x, h):
        if x.dim() == 3:
            coefficients = h.reshape(self.batch_size, x.shape[1], x.shape[2], self.shifterCoefficients)
            x_powers = torch.pow(x[0:self.batch_size, :, :].unsqueeze(3), self.exponents.unsqueeze(0).unsqueeze(1))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=3)
            craftedPolynomial = self.activate(craftedPolynomial)
            return craftedPolynomial
        elif x.dim() == 2:
            coefficients = h.reshape(self.batch_size, x.shape[1], self.shifterCoefficients)
            x_powers = torch.pow(x[0:self.batch_size, :].unsqueeze(2), self.exponents.unsqueeze(0).unsqueeze(1))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=2)
            craftedPolynomial = self.activate(craftedPolynomial)
            return craftedPolynomial
        elif x.dim() == 4:
            coefficients = h.reshape(self.batch_size, x.shape[1], x.shape[2], x.shape[3], self.shifterCoefficients)
            x_powers = torch.pow(x[0:self.batch_size, :, :, :].unsqueeze(4),
                                 self.exponents.unsqueeze(0).unsqueeze(1).unsqueeze(2))
            craftedPolynomial = torch.sum(coefficients * x_powers, dim=4)
            craftedPolynomial = self.activate(craftedPolynomial)
            return craftedPolynomial
        else:
            raise ValueError("Unsupported input dimensions")

    def SpaceTimeFFTFeature(self, data, weights_data, weights_data_fft, space_time):
        # Attention :  Below is implemented simplified FNO LAYER
        # # question : using only real gives better results than using real and imag in sum or concat manner?
        fft_data = torch.fft.fft(data, norm='forward')
        fft_data /= torch.tensor(fft_data.shape[1])
        # question : is "bij,own->bin" give same outcome as "bij,own->bwj"?
        FFwithWeights = torch.einsum("bi,j->bj", fft_data, weights_data_fft)
        iFFWW = torch.fft.ifft(FFwithWeights, norm='forward')
        iFFWW_real = iFFWW.real
        iFFWW_imag = iFFWW.imag
        ifft_data = iFFWW_real + iFFWW_imag
        data = self.activate(ifft_data) + self.activate(
            weights_data * data)  #+space_time # Note : Delete space x time x noise context
        # Attention :  Above is implemented simplified FNO LAYER
        # data = torch.tanh(data)
        return data.real

    def WalshHadamardSpaceTimeFeature(self, meta_central_points, meta_step, noise_var):
        # NOTE: Walsh-Hadamard transform for space and time coding
        space_time = torch.cat([meta_central_points, meta_step, noise_var], dim=1)
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
        return space_time.real

    def activate(self, x):
        return torch.relu(x)  #*2#*self.activation_weight
