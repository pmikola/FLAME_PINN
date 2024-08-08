import copy
import os.path
import random
import struct
import time
from statistics import mean

import kornia
import numpy as np
import torch
from matplotlib import pyplot as plt, animation
from torch.autograd import grad


class teacher(object):
    def __init__(self,models,discriminator,parameterReinforcer,device):
        super(teacher, self).__init__()
        #self.t = None
        self.validation_dataset = None
        self.max_seed = int(1e2)
        self.model = models[0]
        self.expert_0 = models[1]#copy.deepcopy(models[1])
        self.expert_1 = models[2]#copy.deepcopy(models[2])
        self.expert_2 = models[3]  # copy.deepcopy(models[2])
        self.discriminator = discriminator
        self.parameterReinforcer = parameterReinforcer
        self.device = device
        self.fsim = None
        self.period = 1
        self.no_of_periods = 1
        self.data_tensor = None
        self.meta_tensor = None
        self.meta_binary = None
        self.field_names = None
        self.no_frame_samples,self.first_frame, self.last_frame, self.frame_skip =None, None, None, None

        self.data_input = torch.zeros((self.model.batch_size,self.model.in_scale*4,self.model.in_scale),requires_grad=True).to(device)
        self.structure_input = None#torch.zeros((self.model.batch_size,self.model.in_scale,self.model.in_scale),requires_grad=True)
        self.meta_input_h1 = None
        self.meta_input_h2 = None
        self.meta_input_h3 = None
        self.meta_input_h4 = None
        self.meta_input_h5 = None
        self.noise_var_in = None
        self.data_output = torch.zeros((self.model.batch_size,self.model.in_scale*4,self.model.in_scale),requires_grad=True).to(device)
        self.structure_output = None#torch.zeros((self.model.batch_size,self.model.in_scale,self.model.in_scale),requires_grad=True)
        self.meta_output_h1 = None
        self.meta_output_h2 = None
        self.meta_output_h3 = None
        self.meta_output_h4 = None
        self.meta_output_h5 = None
        self.noise_var_out = None

        self.data_input_val = None
        self.data_output_val = None
        self.structure_input_val = None
        self.structure_output_val = None
        self.meta_input_h1_val = None
        self.meta_input_h2_val = None
        self.meta_input_h3_val = None
        self.meta_input_h4_val = None
        self.meta_input_h5_val = None
        self.noise_var_in_val = None
        self.meta_output_h1_val = None
        self.meta_output_h2_val = None
        self.meta_output_h3_val = None
        self.meta_output_h4_val = None
        self.meta_output_h5_val = None
        self.noise_var_out_val = None
        self.epoch = 0
        self.num_of_epochs = 0
        self.train_loss = []
        self.val_loss = []
    def generate_structure(self):
        no_structure = random.randint(0, self.fsim.grid_size_y - self.fsim.N_boundary)
        self.fsim.idx = torch.randint(low=self.fsim.N_boundary,high=self.fsim.grid_size_x-self.fsim.N_boundary,size=(no_structure,))
        self.fsim.idy = torch.randint(low=self.fsim.N_boundary,high=self.fsim.grid_size_y-self.fsim.N_boundary,size=(no_structure,))
        # self.fsim.idx = random.sample(range(self.fsim.N_boundary, self.fsim.grid_size_x - self.fsim.N_boundary), no_structure)
        # self.fsim.idy = random.sample(range(self.fsim.N_boundary, self.fsim.grid_size_y - self.fsim.N_boundary),no_structure)
        self.fsim.idx_u = self.fsim.idx
        self.fsim.idy_u = self.fsim.idy
        self.fsim.idx_v = self.fsim.idx
        self.fsim.idy_v = self.fsim.idy
    def generate_sim_params(self):
        pass

    def data_preparation(self,create_val_dataset=0):
        folder_names = ['v','u','velocity_magnitude','fuel_density','oxidizer_density',
                        'product_density','pressure','temperature','rgb','alpha']
        data_tensor = []
        meta_tensor = []
        meta_binary = []
        field_names = []
        for name in folder_names:
            if os.path.exists(name):
                for i in range(self.first_frame, self.last_frame, self.frame_skip):
                    if name == 'rgb':
                        ptfile = torch.load(name+'\\'+'t{}.pt'.format(i))
                        for j in range(0,3):
                            data_tensor.append(ptfile['data'][:,:,j]/255)
                            meta_tensor.append(ptfile['metadata'])
                            field_names.append(ptfile['name'])
                    else:
                        ptfile = torch.load(name+'\\'+'t{}.pt'.format(i))
                        data_tensor.append(ptfile['data'])
                        meta_tensor.append(ptfile['metadata'])
                        field_names.append(ptfile['name'])

        self.data_tensor = torch.stack(data_tensor,dim=0)
        self.meta_tensor = torch.stack(meta_tensor,dim=0)
        for i in range(self.meta_tensor.shape[0]):
            meta_temp = []
            for j in range(self.meta_tensor.shape[1]):
                binary_var = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', self.meta_tensor[i,j]))
                # Note : '!f' The '!' ensures that
                #     it's in network byte order (big-endian) and the 'f' says that it should be
                #     packed as a float. Use d for double precision
                binary_var = np.frombuffer(binary_var.encode("ascii"), dtype='u1') - 48
                # binary_var = torch.tensor([int(bit) for bit in binary_var], dtype=torch.uint8) - 48
                meta_temp.append(binary_var)
            meta_binary.append(meta_temp)
        self.meta_binary =  torch.from_numpy(np.array(meta_binary))
        self.field_names = field_names
        fdens_idx = np.array([i for i, x in enumerate(self.field_names) if x == "fuel_density"])
        frame_samples = random.sample(list(set(fdens_idx)), k=self.no_frame_samples)
        f_dens_pos = len(fdens_idx)
        fdens_idx = frame_samples
        # TODO : RGB is not 000000,111111,222222 but 012,012,012,012...
        rgb_idx = np.array([i for i, x in enumerate(self.field_names) if x == "rgb"])
        r_idx = rgb_idx[::3][frame_samples]
        g_idx = (rgb_idx[::3]+1)[frame_samples]
        b_idx = (rgb_idx[::3]+2)[frame_samples]
        # fs = np.array(frame_samples)+150

        alpha_idx = np.array([i for i, x in enumerate(self.field_names) if x == "alpha"])[frame_samples]
        fuel_slices = self.data_tensor[fdens_idx]
        min_val = fuel_slices.min()
        max_val = fuel_slices.max()
        fuel_slices = (fuel_slices - min_val) / ((max_val - min_val) + 1e-10)

        r_slices = self.data_tensor[r_idx]
        g_slices = self.data_tensor[g_idx]
        b_slices = self.data_tensor[b_idx]
        alpha_slices = self.data_tensor[alpha_idx]
        meta_binary_slices = self.meta_binary[fdens_idx]

        # gt = np.stack((r_slices[0].cpu().numpy(), g_slices[0].cpu().numpy(), b_slices[0].cpu().numpy()), axis=2)
        # print(gt.shape)
        # plt.imshow(gt.astype(np.uint8) , alpha=alpha_slices[0].cpu().numpy())
        # plt.show()
        x_range = range(self.fsim.N_boundary + self.input_window_size,
                        fuel_slices[0].shape[0] - self.fsim.N_boundary - self.input_window_size)
        y_range = range(self.fsim.N_boundary + self.input_window_size,
                        fuel_slices[0].shape[1] - self.fsim.N_boundary - self.input_window_size)
        data_input = []
        structure_input = []
        meta_input_h1 = []
        meta_input_h2 = []
        meta_input_h3 = []
        meta_input_h4 = []
        meta_input_h5 = []
        noise_var_in = []
        data_output = []
        structure_output = []
        meta_output_h1 = []
        meta_output_h2 = []
        meta_output_h3 = []
        meta_output_h4 = []
        meta_output_h5 = []
        noise_var_out = []
        frame = 0

        while not frame >= self.batch_size*2:
            choose_diffrent_frame = 0
            noise_flag = torch.randint(low=0, high=10, size=(1,))
            # noise_variance_in = torch.tensor(0.).to(self.device)
            # noise_variance_out = torch.tensor(0.).to(self.device)
            # noise_variance_in_binary = torch.zeros(32).to(self.device)
            # noise_variance_out_binary = torch.zeros(32).to(self.device)
            noise_mod = 1.
            if noise_flag < 3:
                noise_variance_in = torch.tensor(0.).to(self.device)
                noise_variance_out = torch.tensor(0.).to(self.device)
                noise_variance_in_binary = torch.zeros(32).to(self.device)
                noise_variance_out_binary = torch.zeros(32).to(self.device)
            elif 3 < noise_flag < 8:
                noise_variance_in = torch.randn(size=(1,))*noise_mod
                noise_variance_in_binary = ''.join(f'{c:08b}' for c in np.float32(noise_variance_in).tobytes())
                noise_variance_in = noise_variance_in.to(self.device)
                noise_variance_in_binary = [int(noise_variance_in_binary[i], 2) for i in range(0, len(noise_variance_in_binary), 1)]
                noise_variance_in_binary = torch.tensor(np.array(noise_variance_in_binary)).to(self.device)
                noise_variance_out = torch.tensor(0.).to(self.device)
                noise_variance_out_binary = torch.zeros(32).to(self.device)
            elif 8 < noise_flag<10:
                noise_variance_out = torch.randn(size=(1,))*noise_mod
                noise_variance_out_binary = ''.join(f'{c:08b}' for c in np.float32(noise_variance_out).tobytes())
                noise_variance_out = noise_variance_out.to(self.device)
                noise_variance_out_binary = [int(noise_variance_out_binary[i], 2) for i in range(0, len(noise_variance_out_binary), 1)]
                noise_variance_out_binary = torch.tensor(np.array(noise_variance_out_binary)).to(self.device)
                noise_variance_in = torch.tensor(0.).to(self.device)
                noise_variance_in_binary = torch.zeros(32).to(self.device)
            else:
                noise_variance_in = torch.randn(size=(1,))*noise_mod
                noise_variance_in_binary = ''.join(f'{c:08b}' for c in np.float32(noise_variance_in).tobytes())
                noise_variance_in = noise_variance_in.to(self.device)
                noise_variance_in_binary = [int(noise_variance_in_binary[i], 2) for i in range(0, len(noise_variance_in_binary), 1)]
                noise_variance_in_binary = torch.tensor(np.array(noise_variance_in_binary)).to(self.device)
                noise_variance_out = torch.randn(size=(1,))*noise_mod
                noise_variance_out_binary = ''.join(f'{c:08b}' for c in np.float32(noise_variance_out).tobytes())
                noise_variance_out = noise_variance_out.to(self.device)
                noise_variance_out_binary = [int(noise_variance_out_binary[i], 2) for i in range(0, len(noise_variance_out_binary), 1)]
                noise_variance_out_binary = torch.tensor(np.array(noise_variance_out_binary)).to(self.device)


            idx_input = random.choice(range(0, fuel_slices.shape[0]))
            central_point_x_in = random.sample(x_range, 1)[0]
            central_point_y_in = random.sample(y_range, 1)[0]
            window_x_in = np.array(range(central_point_x_in - self.input_window_size, central_point_x_in + self.input_window_size + 1))
            window_y_in = np.array(range(central_point_y_in - self.input_window_size, central_point_y_in + self.input_window_size + 1))
            central_point_x_binary_in = "{0:010b}".format(central_point_x_in)
            central_point_x_binary_in = torch.tensor(np.array([int(d) for d in central_point_x_binary_in]))
            central_point_y_binary_in = "{0:010b}".format(central_point_y_in)
            central_point_y_binary_in = torch.tensor(np.array([int(d) for d in central_point_y_binary_in]))
            slice_x_in = slice(window_x_in[0], window_x_in[-1] + 1)
            slice_y_in = slice(window_y_in[0], window_y_in[-1] + 1)

            idx_output = random.choice(range(0, fuel_slices.shape[0]))
            offset_x = random.randint(int(-self.input_window_size/2),int(self.input_window_size/2))
            offset_y = random.randint(int(-self.input_window_size/2),int(self.input_window_size/2))
            central_point_x_out = central_point_x_in + offset_x
            central_point_y_out = central_point_y_in + offset_y

            window_x_out = np.array(range(central_point_x_out - self.input_window_size, central_point_x_out + self.input_window_size + 1))
            window_y_out = np.array(range(central_point_y_out - self.input_window_size, central_point_y_out + self.input_window_size + 1))
            central_point_x_binary_out = "{0:010b}".format(central_point_x_out)
            central_point_x_binary_out = torch.tensor(np.array([int(d) for d in central_point_x_binary_out]))
            central_point_y_binary_out = "{0:010b}".format(central_point_y_out)
            central_point_y_binary_out = torch.tensor(np.array([int(d) for d in central_point_y_binary_out]))
            slice_x_out = slice(window_x_out[0], window_x_out[-1] + 1)
            slice_y_out = slice(window_y_out[0], window_y_out[-1] + 1)

            # Note : Input data
            fuel_subslice_in = fuel_slices[idx_input,slice_x_in, slice_y_in] + torch.nan_to_num(noise_variance_in*torch.rand_like(fuel_slices[idx_input,slice_x_in, slice_y_in]).to(self.device),nan=0.0)
            r_subslice_in = r_slices[idx_input,slice_x_in, slice_y_in] + torch.nan_to_num(noise_variance_in*torch.rand_like(r_slices[idx_input,slice_x_in, slice_y_in]).to(self.device),nan=0.0)
            g_subslice_in = g_slices[idx_input,slice_x_in, slice_y_in] + torch.nan_to_num(noise_variance_in*torch.rand_like(g_slices[idx_input,slice_x_in, slice_y_in]).to(self.device),nan=0.0)
            b_subslice_in = b_slices[idx_input,slice_x_in, slice_y_in] + torch.nan_to_num(noise_variance_in*torch.rand_like(b_slices[idx_input,slice_x_in, slice_y_in]).to(self.device),nan=0.0)
            alpha_subslice_in = alpha_slices[idx_input,slice_x_in, slice_y_in] + torch.nan_to_num(noise_variance_in*torch.rand_like(alpha_slices[idx_input,slice_x_in, slice_y_in]).to(self.device),nan=0.0)
            data_input_subslice = torch.cat([r_subslice_in,g_subslice_in, b_subslice_in, alpha_subslice_in], dim=0)

            meta_step_in = meta_binary_slices[idx_input][0]
            meta_step_in_numeric = self.meta_tensor[idx_input][0]

            meta_fuel_initial_speed_in = meta_binary_slices[idx_input][1]
            meta_fuel_cut_off_time_in = meta_binary_slices[idx_input][2]
            meta_igni_time_in = meta_binary_slices[idx_input][3]
            meta_ignition_temp_in = meta_binary_slices[idx_input][4]
            meta_viscosity_in = meta_binary_slices[idx_input][14]
            meta_diff_in = meta_binary_slices[idx_input][15]
            meta_input_subslice = torch.cat([meta_step_in, meta_fuel_initial_speed_in,
                                             meta_fuel_cut_off_time_in, meta_igni_time_in,
                                             meta_ignition_temp_in, meta_viscosity_in, meta_diff_in], dim=0)


            # Note : Output data
            fuel_subslice_out = fuel_slices[idx_output,slice_x_out, slice_y_out] + torch.nan_to_num(noise_variance_out*torch.rand_like(fuel_slices[idx_output,slice_x_out, slice_y_out]).to(self.device),nan=0.0)
            r_subslice_out = r_slices[idx_output,slice_x_out, slice_y_out] + torch.nan_to_num(noise_variance_out*torch.rand_like(r_slices[idx_output,slice_x_out, slice_y_out]).to(self.device),nan=0.0)
            g_subslice_out = g_slices[idx_output,slice_x_out, slice_y_out] + torch.nan_to_num(noise_variance_out*torch.rand_like(g_slices[idx_output,slice_x_out, slice_y_out]).to(self.device),nan=0.0)
            b_subslice_out = b_slices[idx_output,slice_x_out, slice_y_out] + torch.nan_to_num(noise_variance_out*torch.rand_like(b_slices[idx_output,slice_x_out, slice_y_out]),nan=0.0)
            alpha_subslice_out = alpha_slices[idx_output,slice_x_out, slice_y_out] + torch.nan_to_num(noise_variance_out*torch.rand_like(alpha_slices[idx_output,slice_x_out, slice_y_out]).to(self.device),nan=0.0)
            data_output_subslice = torch.cat([r_subslice_out,g_subslice_out, b_subslice_out, alpha_subslice_out], dim=0)

            meta_step_out = meta_binary_slices[idx_output][0]
            meta_step_out_numeric = self.meta_tensor[idx_output][0]
            meta_fuel_initial_speed_out = meta_binary_slices[idx_output][1]
            meta_fuel_cut_off_time_out = meta_binary_slices[idx_output][2]
            meta_igni_time_out = meta_binary_slices[idx_output][3]
            meta_ignition_temp_out = meta_binary_slices[idx_output][4]
            meta_viscosity_out = meta_binary_slices[idx_output][14]
            meta_diff_out = meta_binary_slices[idx_output][15]
            meta_output_subslice = torch.cat([meta_step_out, meta_fuel_initial_speed_out,
                                              meta_fuel_cut_off_time_out, meta_igni_time_out,
                                              meta_ignition_temp_out, meta_viscosity_out, meta_diff_out], dim=0)


            r_in_is_zero = torch.count_nonzero(r_subslice_in)
            r_out_is_zero = torch.count_nonzero(r_subslice_out)
            g_in_is_zero = torch.count_nonzero(g_subslice_in)
            g_out_is_zero = torch.count_nonzero(g_subslice_out)
            b_in_is_zero = torch.count_nonzero(b_subslice_in)
            b_out_is_zero = torch.count_nonzero(b_subslice_out)
            a_in_is_zero = torch.count_nonzero(alpha_subslice_out)
            a_out_is_zero = torch.count_nonzero(alpha_subslice_out)
            f_in_is_zero = torch.count_nonzero(fuel_subslice_out)
            f_out_is_zero = torch.count_nonzero(fuel_subslice_out)
            r_zero = r_in_is_zero==r_out_is_zero
            r_i0 = r_in_is_zero == 0
            r_o0 = r_out_is_zero == 0
            g_zero = g_in_is_zero==g_out_is_zero
            g_i0 = g_in_is_zero == 0
            g_o0 = g_out_is_zero == 0
            b_zero = b_in_is_zero == b_out_is_zero
            b_i0 = b_in_is_zero == 0
            b_o0 = b_out_is_zero == 0
            a_zero = a_in_is_zero == a_out_is_zero
            a_i0 = a_in_is_zero == 0
            a_o0 = a_out_is_zero == 0
            f_zero = f_in_is_zero == f_out_is_zero
            f_i0 = f_in_is_zero == 0
            f_o0 = f_out_is_zero == 0
            rzero = r_zero and r_i0 and r_o0
            gzero = g_zero and g_i0 and g_o0
            bzero = b_zero and b_i0 and b_o0
            azero = a_zero and a_i0 and a_o0
            fzero = f_zero and f_i0 and f_o0

            central_points_in = torch.cat([central_point_x_binary_in, central_point_y_binary_in], dim=0).to(self.device)
            central_points_out = torch.cat([central_point_x_binary_out, central_point_y_binary_out], dim=0).to(self.device)

            if create_val_dataset == 0:
                matches_points_in = (self.meta_input_h3_val == central_points_in).all(dim=1)
                matches_points_out = (self.meta_output_h3_val == central_points_out).all(dim=1)
                matches_time_in = (self.meta_input_h2_val == meta_step_in.to(self.device)).all(dim=1)
                matches_time_out = (self.meta_output_h2_val == meta_step_out.to(self.device)).all(dim=1)
                if True in matches_points_in and True in matches_time_in and True in matches_time_out and True in matches_points_out:
                    choose_diffrent_frame = 1

            mod = 4
            if self.epoch > self.num_of_epochs*0.03 or create_val_dataset == 1:
                pass
            else:
                data_in_cnz = torch.count_nonzero(data_input_subslice)
                fuel_in_cnz = torch.count_nonzero(fuel_subslice_in)
                data_out_cnz = torch.count_nonzero(data_output_subslice)
                fuel_out_cnz = torch.count_nonzero(fuel_subslice_out)

                if (data_in_cnz < mod*int(data_input_subslice.shape[0]*data_input_subslice.shape[1]/(self.epoch+mod)) or
                        fuel_in_cnz < mod*int(data_input_subslice.shape[0]*data_input_subslice.shape[1]/(self.epoch+mod)) or
                        data_out_cnz < mod*int(data_output_subslice.shape[0]*data_output_subslice.shape[1]/(self.epoch+mod)) or
                        fuel_out_cnz < mod*int(data_output_subslice.shape[0]*data_output_subslice.shape[1]/(self.epoch+mod))):
                    choose_diffrent_frame = 1
            frame += 1
            if  rzero and gzero and bzero and azero and fzero and idx_input > idx_output and choose_diffrent_frame:
                frame -=1
            else:
                # Note: Data for the different layers
                data_input.append(data_input_subslice)
                structure_input.append(fuel_subslice_in)
                meta_input_h1.append(meta_input_subslice)
                meta_input_h2.append(meta_step_in)
                meta_input_h3.append(central_points_in)
                meta_input_h4.append(torch.cat([torch.tensor(window_x_in),torch.tensor(window_y_in)]))
                meta_input_h5.append(meta_step_in_numeric)
                noise_var_in.append(noise_variance_in_binary.to(torch.float))
                data_output.append(data_output_subslice)
                structure_output.append(fuel_subslice_out)
                meta_output_h1.append(meta_output_subslice)
                meta_output_h2.append(meta_step_out)
                meta_output_h3.append(central_points_out)
                meta_output_h4.append(torch.cat([torch.tensor(window_x_out), torch.tensor(window_y_out)]))
                meta_output_h5.append(meta_step_out_numeric)
                noise_var_out.append(noise_variance_out_binary.to(torch.float))

        if create_val_dataset == 1:
            self.data_input_val = torch.stack(data_input, dim=0)[0:self.batch_size].to(self.device)
            self.structure_input_val = torch.stack(structure_input, dim=0)[0:self.batch_size].to(self.device)
            self.meta_input_h1_val = torch.stack(meta_input_h1, dim=0)[0:self.batch_size].to(self.device)
            self.meta_input_h2_val = torch.stack(meta_input_h2, dim=0)[0:self.batch_size].to(self.device)
            self.meta_input_h3_val = torch.stack(meta_input_h3, dim=0)[0:self.batch_size].to(self.device)
            self.meta_input_h4_val = torch.stack(meta_input_h4, dim=0)[0:self.batch_size].to(self.device)
            self.meta_input_h5_val = torch.stack(meta_input_h5, dim=0)[0:self.batch_size].to(self.device)
            self.noise_var_in_val = torch.stack(noise_var_in, dim=0)[0:self.batch_size].to(self.device)

            self.data_output_val = torch.stack(data_output, dim=0)[0:self.batch_size].to(self.device)
            self.structure_output_val = torch.stack(structure_output, dim=0)[0:self.batch_size].to(self.device)
            self.meta_output_h1_val = torch.stack(meta_output_h1, dim=0)[0:self.batch_size].to(self.device)
            self.meta_output_h2_val = torch.stack(meta_output_h2, dim=0)[0:self.batch_size].to(self.device)
            self.meta_output_h3_val = torch.stack(meta_output_h3, dim=0)[0:self.batch_size].to(self.device)
            self.meta_output_h4_val = torch.stack(meta_output_h4, dim=0)[0:self.batch_size].to(self.device)
            self.meta_output_h5_val = torch.stack(meta_output_h5, dim=0)[0:self.batch_size].to(self.device)
            self.noise_var_out_val = torch.stack(noise_var_out, dim=0)[0:self.batch_size].to(self.device)
        else:
            self.data_input = torch.stack(data_input,dim=0).to(self.device)
            self.structure_input = torch.stack(structure_input,dim=0).to(self.device)
            self.meta_input_h1 = torch.stack(meta_input_h1,dim=0).to(self.device)
            self.meta_input_h2 = torch.stack(meta_input_h2,dim=0).to(self.device)
            self.meta_input_h3 = torch.stack(meta_input_h3,dim=0).to(self.device)
            self.meta_input_h4 = torch.stack(meta_input_h4,dim=0).to(self.device)
            self.meta_input_h5 = torch.stack(meta_input_h5,dim=0).to(self.device)
            self.noise_var_in = torch.stack(noise_var_in,dim=0).to(self.device)

            self.data_output = torch.stack(data_output,dim=0).to(self.device)
            self.structure_output = torch.stack(structure_output, dim=0).to(self.device)
            self.meta_output_h1 = torch.stack(meta_output_h1,dim=0).to(self.device)
            self.meta_output_h2 = torch.stack(meta_output_h2,dim=0).to(self.device)
            self.meta_output_h3 = torch.stack(meta_output_h3,dim=0).to(self.device)
            self.meta_output_h4 = torch.stack(meta_output_h4,dim=0).to(self.device)
            self.meta_output_h5 = torch.stack(meta_output_h5,dim=0).to(self.device)
            self.noise_var_out = torch.stack(noise_var_out,dim=0).to(self.device)
    def examine(self,criterion,device,plot=0):
        self.model.load_state_dict(torch.load('model.pt'))
        folder_names = ['v', 'u', 'velocity_magnitude', 'fuel_density', 'oxidizer_density',
                        'product_density', 'pressure', 'temperature', 'rgb', 'alpha']
        data_tensor = []
        meta_tensor = []
        meta_binary = []
        field_names = []

        for name in folder_names:
            if os.path.exists(name):
                for i in range(self.first_frame, self.last_frame, self.frame_skip):
                    if name == 'rgb':
                        ptfile = torch.load(name + '\\' + 't{}.pt'.format(i))
                        for j in range(0, 3):
                            data_tensor.append(ptfile['data'][:, :, j] / 255.)
                            meta_tensor.append(ptfile['metadata'])
                            field_names.append(ptfile['name'])
                    else:
                        ptfile = torch.load(name + '\\' + 't{}.pt'.format(i))
                        data_tensor.append(ptfile['data'])
                        meta_tensor.append(ptfile['metadata'])
                        field_names.append(ptfile['name'])

        self.data_tensor = torch.stack(data_tensor, dim=0)
        self.meta_tensor = torch.stack(meta_tensor, dim=0)

        for i in range(self.meta_tensor.shape[0]):
            meta_temp = []
            for j in range(self.meta_tensor.shape[1]):
                binary_var = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', self.meta_tensor[i, j]))
                # Note : '!f' The '!' ensures that
                #     it's in network byte order (big-endian) and the 'f' says that it should be
                #     packed as a float. Use d for double precision
                binary_var = np.frombuffer(binary_var.encode("ascii"), dtype='u1') - 48
                # binary_var = torch.tensor([int(bit) for bit in binary_var], dtype=torch.uint8) - 48
                meta_temp.append(binary_var)
            meta_binary.append(meta_temp)
        self.meta_binary = torch.from_numpy(np.array(meta_binary))
        self.field_names = field_names
        fdens_idx = np.array([i for i, x in enumerate(self.field_names) if x == "fuel_density"])
        #frame_samples = random.sample(list(set(fdens_idx)), k=self.no_frame_samples)
        f_dens_pos = len(fdens_idx)

        #fdens_idx = fdens_idx[frame_samples]
        rgb_idx = np.array([i for i, x in enumerate(self.field_names) if x == "rgb"])
        r_idx = rgb_idx[::3]#[frame_samples]
        g_idx = rgb_idx[::3]+1#[frame_samples]
        b_idx = rgb_idx[::3]+2#[frame_samples]
        alpha_idx = np.array([i for i, x in enumerate(self.field_names) if x == "alpha"])#[frame_samples]
        fuel_slices = self.data_tensor[fdens_idx]
        min_val = fuel_slices.min()
        max_val = fuel_slices.max()
        fuel_slices = (fuel_slices - min_val) / ((max_val - min_val) + 1e-12)

        r_slices = self.data_tensor[r_idx]
        g_slices = self.data_tensor[g_idx]
        b_slices = self.data_tensor[b_idx]
        alpha_slices = self.data_tensor[alpha_idx]
        meta_binary_slices = self.meta_binary[fdens_idx]

        # Note: IDX preparation
        central_points_x = np.arange(self.input_window_size, fuel_slices.shape[1] - self.input_window_size+1)
        central_points_y = np.arange(self.input_window_size, fuel_slices.shape[2] - self.input_window_size+1)

        central_points_x_pos = central_points_x + self.input_window_size
        central_points_x_neg = central_points_x - self.input_window_size
        central_points_y_pos = central_points_y + self.input_window_size
        central_points_y_neg = central_points_y - self.input_window_size


        windows_x = []
        windows_y = []

        central_points_x_binary = []
        central_points_y_binary = []
        v = int(central_points_x_pos.shape[0] / self.model.in_scale + 1)
        h = int(central_points_y_pos.shape[0] / self.model.in_scale + 1)
        j = 0
        for m in range(0, v):
            k = 0
            for n in range(0, h):
                wx_range = np.array(range(int(central_points_x_neg[j]),int(central_points_x_pos[j]) +2))
                windows_x.append(wx_range)
                central_point_x_binary_pre = "{0:010b}".format(central_points_x[j])
                central_points_x_binary.append(torch.tensor([torch.tensor(int(d), dtype=torch.int8) for d in central_point_x_binary_pre]))
                wy_range = np.array(range(int(central_points_y_neg[k]),int(central_points_y_pos[k]) + 2))
                windows_y.append(wy_range)
                central_point_y_binary_pre = "{0:010b}".format(central_points_y[k])
                central_points_y_binary.append(torch.tensor([torch.tensor(int(d), dtype=torch.int8) for d in central_point_y_binary_pre]))
                k +=self.model.in_scale
            j+=self.model.in_scale


        central_points_x_binary = torch.tensor(np.array(central_points_x_binary))
        central_points_y_binary = torch.tensor(np.array(central_points_y_binary))
        central_points_xy_binary = []
        for g in range(len(central_points_x_binary)):
            xy_binary = torch.cat([central_points_x_binary[g], central_points_y_binary[g]])
            central_points_xy_binary.append(xy_binary)


        x_idx = torch.tensor(np.array(windows_x))
        y_idx = torch.tensor(np.array(windows_y))
        x_idx_start = np.array([sublist[0] for sublist in x_idx])
        x_idx_end = np.array([sublist[-1] for sublist in x_idx])
        # print('START\n',x_idx_start[0:150],x_idx_start[-150:-1],'START \n')
        # print('STOP \n',x_idx_end[0:150], x_idx_end[-150:-1],'STOP \n')
        y_idx_start = np.array([sublist[0] for sublist in y_idx])
        y_idx_end = np.array([sublist[-1] for sublist in y_idx])
        # print('START\n', y_idx_start[0:150], y_idx_start[-150:-1], 'START \n')
        # print('STOP \n', y_idx_end[0:150], y_idx_end[-150:-1], 'STOP \n')
        t = 0.
        ims = []
        fig = plt.figure(figsize=(10, 6))
        grid = (1, 3)
        ax1 = plt.subplot2grid(grid, (0, 0))
        ax2 = plt.subplot2grid(grid, (0, 1))
        ax3 = plt.subplot2grid(grid, (0, 2))
        # ax3 = plt.subplot2grid(grid, (1, 0))
        # ax4 = plt.subplot2grid(grid, (1, 1))
        for ax in [ax1, ax2,ax3]:
            ax.set_axis_off()


        for i in range(0,fuel_slices.shape[0]-1):
            idx_input = i
            idx_output = i+1

            # Note : Input data
            fsin = []
            rsin = []
            gsin = []
            bsin = []
            asin = []

            fsout = []
            rsout = []
            gsout = []
            bsout = []
            asout = []

            for ii in range(len(x_idx_start)):
                fsin.append(fuel_slices[idx_input, x_idx_start[ii]:x_idx_end[ii],y_idx_start[ii]:y_idx_end[ii]])
                rsin.append(r_slices[idx_input, x_idx_start[ii]:x_idx_end[ii],y_idx_start[ii]:y_idx_end[ii]])
                gsin.append(g_slices[idx_input, x_idx_start[ii]:x_idx_end[ii],y_idx_start[ii]:y_idx_end[ii]])
                bsin.append(b_slices[idx_input, x_idx_start[ii]:x_idx_end[ii],y_idx_start[ii]:y_idx_end[ii]])
                asin.append(alpha_slices[idx_input, x_idx_start[ii]:x_idx_end[ii],y_idx_start[ii]:y_idx_end[ii]])

                fsout.append(fuel_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                rsout.append(r_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                gsout.append(g_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                bsout.append(b_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])
                asout.append(alpha_slices[idx_output, x_idx_start[ii]:x_idx_end[ii], y_idx_start[ii]:y_idx_end[ii]])


            fuel_subslice_in =  torch.stack(fsin,dim=0)
            r_subslice_in = torch.stack(rsin,dim=0)
            g_subslice_in = torch.stack(gsin,dim=0)
            b_subslice_in = torch.stack(bsin,dim=0)
            alpha_subslice_in = torch.stack(asin,dim=0)
            data_input_subslice = torch.cat([r_subslice_in,g_subslice_in, b_subslice_in,alpha_subslice_in], dim=1)
            meta_step_in = meta_binary_slices[idx_input][0]
            meta_step_in_numeric = self.meta_tensor[idx_input][0]
            meta_fuel_initial_speed_in = meta_binary_slices[idx_input][1]
            meta_fuel_cut_off_time_in = meta_binary_slices[idx_input][2]
            meta_igni_time_in = meta_binary_slices[idx_input][3]
            meta_ignition_temp_in = meta_binary_slices[idx_input][4]

            meta_viscosity_in = meta_binary_slices[idx_input][14]
            meta_diff_in = meta_binary_slices[idx_input][15]
            meta_input_subslice = torch.cat([meta_step_in, meta_fuel_initial_speed_in,
                                             meta_fuel_cut_off_time_in, meta_igni_time_in,
                                             meta_ignition_temp_in, meta_viscosity_in, meta_diff_in], dim=0)
            # Note : Output data
            f_subslice_out = torch.stack(fsout, dim=0)
            r_subslice_out = torch.stack(rsout,dim=0)
            g_subslice_out = torch.stack(gsout,dim=0)
            b_subslice_out = torch.stack(bsout,dim=0)
            alpha_subslice_out = torch.stack(asout,dim=0)

            data_output_subslice = torch.cat([r_subslice_out, g_subslice_out, b_subslice_out, alpha_subslice_out],dim=1)
            meta_step_out = meta_binary_slices[idx_output][0]
            meta_step_out_numeric = self.meta_tensor[idx_output][0]
            meta_fuel_initial_speed_out = meta_binary_slices[idx_output][1]
            meta_fuel_cut_off_time_out = meta_binary_slices[idx_output][2]
            meta_igni_time_out = meta_binary_slices[idx_output][3]
            meta_ignition_temp_out = meta_binary_slices[idx_output][4]
            meta_viscosity_out = meta_binary_slices[idx_output][14]
            meta_diff_out = meta_binary_slices[idx_output][15]
            meta_output_subslice = torch.cat([meta_step_out, meta_fuel_initial_speed_out,
                                              meta_fuel_cut_off_time_out, meta_igni_time_out,
                                              meta_ignition_temp_out, meta_viscosity_out, meta_diff_out], dim=0)
            # Note: Data for the different layers
            data_input = data_input_subslice
            self.model.batch_size = data_input.shape[0]
            structure_input = fuel_subslice_in
            meta_input_h1 = meta_input_subslice.unsqueeze(0).repeat(data_input.shape[0],1)
            meta_input_h2 = meta_step_in.unsqueeze(0).repeat(data_input.shape[0],1)
            meta_input_h3 = torch.tensor(np.array(central_points_xy_binary))
            meta_input_h4 = torch.cat([x_idx[:,0:-1], y_idx[:,0:-1]],dim=1)
            noise_var_in = torch.zeros((data_input.shape[0], 32))

            meta_input_h5 = meta_step_in_numeric.repeat(data_input.shape[0],1).squeeze(1)
            data_output = data_output_subslice
            meta_output_h1 = meta_output_subslice.unsqueeze(0).repeat(data_input.shape[0],1)
            meta_output_h2 = meta_step_out.unsqueeze(0).repeat(data_input.shape[0],1)
            meta_output_h3 = torch.tensor(np.array(central_points_xy_binary))
            meta_output_h4 = torch.cat([x_idx[:,0:-1], y_idx[:,0:-1]],dim=1)
            meta_output_h5 = meta_step_out_numeric.repeat(data_input.shape[0],1).squeeze(1)
            noise_var_out = torch.zeros((data_input.shape[0],32))

            self.model.eval()

            (data_input, structure_input, meta_input_h1, meta_input_h2,
             meta_input_h3, meta_input_h4, meta_input_h5,noise_var_in, meta_output_h1,
             meta_output_h2, meta_output_h3, meta_output_h4, meta_output_h5,noise_var_out) = \
                (data_input.to(device),
                 structure_input.to(device),
                 meta_input_h1.to(device),
                 meta_input_h2.to(device),
                 meta_input_h3.to(device),
                 meta_input_h4.to(device),
                 meta_input_h5.to(device),
                 noise_var_in.to(device),
                 meta_output_h1.to(device),
                 meta_output_h2.to(device),
                 meta_output_h3.to(device),
                 meta_output_h4.to(device),
                 meta_output_h5.to(device),
                 noise_var_out.to(device))

            # data_output = data_output.to(device)
            dataset = (data_input, structure_input, meta_input_h1, meta_input_h2,
                       meta_input_h3, meta_input_h4, meta_input_h5,noise_var_in, meta_output_h1,
                       meta_output_h2, meta_output_h3, meta_output_h4, meta_output_h5,noise_var_out)

            t_start = time.perf_counter()
            pred_r, pred_g, pred_b, pred_a,pred_s = self.model(dataset)
            t_pred = time.perf_counter()

            t = t_pred - t_start
            print(f'Pred Time: {t*1e6:.4f} [us]')

            r_v_true = np.array([]).reshape(0, h * self.model.in_scale)
            g_v_true = np.array([]).reshape(0, h * self.model.in_scale)
            b_v_true = np.array([]).reshape(0, h * self.model.in_scale)
            a_v_true = np.array([]).reshape(0, h * self.model.in_scale)

            r_v_pred = np.array([]).reshape(0, h * self.model.in_scale)
            g_v_pred = np.array([]).reshape(0, h * self.model.in_scale)
            b_v_pred = np.array([]).reshape(0, h * self.model.in_scale)
            a_v_pred = np.array([]).reshape(0, h * self.model.in_scale)
            idx = 0
            for m in range(0, v):
                r_h_true = np.array([]).reshape(self.model.in_scale, 0)
                g_h_true = np.array([]).reshape(self.model.in_scale, 0)
                b_h_true = np.array([]).reshape(self.model.in_scale, 0)
                a_h_true = np.array([]).reshape(self.model.in_scale, 0)

                r_h_pred = np.array([]).reshape(self.model.in_scale, 0)
                g_h_pred = np.array([]).reshape(self.model.in_scale, 0)
                b_h_pred = np.array([]).reshape(self.model.in_scale, 0)
                a_h_pred = np.array([]).reshape(self.model.in_scale, 0)
                for n in range(0, h):
                    iter_index = idx#(n + m * h)
                    r_h_true = np.hstack([r_h_true, r_subslice_out[iter_index].cpu().detach().numpy()])
                    g_h_true = np.hstack([g_h_true, g_subslice_out[iter_index].cpu().detach().numpy()])
                    b_h_true = np.hstack([b_h_true, b_subslice_out[iter_index].cpu().detach().numpy()])
                    a_h_true = np.hstack([a_h_true, alpha_subslice_out[iter_index].cpu().detach().numpy()])

                    r_h_pred = np.hstack([r_h_pred, pred_r[iter_index].cpu().detach().numpy()])
                    g_h_pred = np.hstack([g_h_pred, pred_g[iter_index].cpu().detach().numpy()])
                    b_h_pred = np.hstack([b_h_pred, pred_b[iter_index].cpu().detach().numpy()])
                    a_h_pred = np.hstack([a_h_pred, pred_a[iter_index].cpu().detach().numpy()])
                    idx+=1
                r_v_true = np.vstack([r_v_true, r_h_true])
                g_v_true = np.vstack([g_v_true, g_h_true])
                b_v_true = np.vstack([b_v_true, b_h_true])
                a_v_true = np.vstack([a_v_true, a_h_true])


                r_v_pred = np.clip(abs(np.vstack([r_v_pred, r_h_pred])), 0., 1.)
                g_v_pred = np.clip(abs(np.vstack([g_v_pred, g_h_pred])), 0., 1.)
                b_v_pred = np.clip(abs(np.vstack([b_v_pred, b_h_pred])), 0., 1.)
                a_v_pred = np.clip(abs(np.vstack([a_v_pred, a_h_pred])), 0., 1.)


            prediction = np.stack((r_v_pred, g_v_pred, b_v_pred), axis=2)
            ground_truth = np.stack((r_v_true, g_v_true, b_v_true),axis=2)

            title_pred = ax1.set_title("Prediction")
            title_true = ax2.set_title("Ground Truth")
            title_rms = ax3.set_title("rms")

            rgb_pred_anim = ax1.imshow(prediction.astype(np.uint8)*255,alpha=a_v_pred)
            rgb_true_anim = ax2.imshow(ground_truth.astype(np.uint8)*255,alpha=a_v_true)

            rms = np.mean(np.sqrt(abs(prediction**2 - ground_truth**2)),axis=2)
            rms_anim = ax3.imshow(rms,cmap= 'RdBu',vmin=0,vmax=1)

            ims.append([rgb_pred_anim, rgb_true_anim,rms_anim,title_pred,title_true,title_rms])
        ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=100)
        # ani.save("flame_animation.gif")
        fig.colorbar(rms_anim, ax=ax3)
        plt.show()

    def learning_phase(self,no_frame_samples, batch_size, input_window_size, first_frame, last_frame,
                                      frame_skip,criterion,optimizer,disc_optimizer,RL_optimizer,device,learning=1,num_epochs=1500):
            (self.no_frame_samples,self.batch_size,self.input_window_size,self.first_frame,
             self.last_frame,self.frame_skip) = (no_frame_samples, batch_size,
                                                 input_window_size, first_frame, last_frame,frame_skip)

            criterion_model,criterion_e0,criterion_e1,criterion_e2,criterion_disc,criterion_RL = criterion
            self.num_of_epochs = num_epochs
            model_to_Save = self.model
            if learning == 1:
                best_loss = float('inf')
                num_epochs = num_epochs
                t = 0.
                t_epoch = 0.
                grad_counter = 0
                reiterate_data = 0
                reiterate_counter = 0
                norm = 'forward'
                print_every_nth_frame=10
                best_models = []
                best_losses = []
                zero = torch.tensor([0.],requires_grad=True).to(device).float()

                self.data_preparation(1)
                val_idx = torch.arange(self.data_input_val.shape[0])
                self.validation_dataset = (
                self.data_input_val, self.structure_input_val, self.meta_input_h1_val,self.meta_input_h2_val,
                self.meta_input_h3_val, self.meta_input_h4_val, self.meta_input_h5_val,self.noise_var_in_val,
                self.meta_output_h1_val,self.meta_output_h2_val, self.meta_output_h3_val, self.meta_output_h4_val,
                self.meta_output_h5_val,self.noise_var_out_val)

                for epoch in range(num_epochs):

                    self.epoch = epoch
                    t_epoch_start = time.perf_counter()
                    self.seed_setter(int(epoch+1))
                    if reiterate_data == 0:
                        self.data_preparation()
                    else:
                        reiterate_counter +=1

                    m_idx = torch.arange(int(self.data_input.shape[0]/2))
                    e0_idx = torch.randint(int(self.data_input.shape[0]),(int(self.data_input.shape[0]/2),))
                    e1_idx = torch.randint(int(self.data_input.shape[0]),(int(self.data_input.shape[0]/2),))
                    e2_idx = torch.arange(start=int(self.data_input.shape[0] / 2),end=int(self.data_input.shape[0]))

                    dataset = (self.data_input[m_idx],self.structure_input[m_idx],self.meta_input_h1[m_idx],self.meta_input_h2[m_idx],
                               self.meta_input_h3[m_idx],self.meta_input_h4[m_idx],self.meta_input_h5[m_idx],self.noise_var_in[m_idx],self.meta_output_h1[m_idx],
                               self.meta_output_h2[m_idx],self.meta_output_h3[m_idx],self.meta_output_h4[m_idx],self.meta_output_h5[m_idx],self.noise_var_out[m_idx])



                    dataset_e0 = (self.data_input[e0_idx], self.structure_input[e0_idx], self.meta_input_h1[e0_idx], self.meta_input_h2[e0_idx],
                               self.meta_input_h3[e0_idx], self.meta_input_h4[e0_idx], self.meta_input_h5[e0_idx], self.noise_var_in[e0_idx],
                               self.meta_output_h1[e0_idx],
                               self.meta_output_h2[e0_idx], self.meta_output_h3[e0_idx], self.meta_output_h4[e0_idx], self.meta_output_h5[e0_idx],
                               self.noise_var_out[e0_idx])

                    dataset_e1 = (self.data_input[e1_idx], self.structure_input[e1_idx], self.meta_input_h1[e1_idx], self.meta_input_h2[e1_idx],
                                  self.meta_input_h3[e1_idx], self.meta_input_h4[e1_idx], self.meta_input_h5[e1_idx], self.noise_var_in[e1_idx],
                                  self.meta_output_h1[e1_idx],
                                  self.meta_output_h2[e1_idx], self.meta_output_h3[e1_idx], self.meta_output_h4[e1_idx], self.meta_output_h5[e1_idx],
                                  self.noise_var_out[e1_idx])

                    dataset_e2 = (self.data_input[e2_idx], self.structure_input[e2_idx], self.meta_input_h1[e2_idx],
                                  self.meta_input_h2[e2_idx],
                                  self.meta_input_h3[e2_idx], self.meta_input_h4[e2_idx], self.meta_input_h5[e2_idx],
                                  self.noise_var_in[e2_idx],
                                  self.meta_output_h1[e2_idx],
                                  self.meta_output_h2[e2_idx], self.meta_output_h3[e2_idx], self.meta_output_h4[e2_idx],
                                  self.meta_output_h5[e2_idx],
                                  self.noise_var_out[e2_idx])


                    t_start = time.perf_counter()
                    self.seed_setter(int((epoch+1)*2))
                    model_output = self.model(dataset)
                    self.seed_setter(int((epoch+1) * 3))
                    expert_0_output = self.expert_0(dataset_e0)
                    self.seed_setter(int((epoch+1) * 4))
                    expert_1_output = self.expert_1(dataset_e1)
                    self.seed_setter(int((epoch + 1) * 5))
                    expert_2_output = self.expert_2(dataset_e2)
                    t_pred = time.perf_counter()

                    disc_loss = self.discriminator_loss(m_idx, model_output, self.data_output, self.structure_output,criterion_disc)
                    # disc_loss = self.discriminator_loss(e0_idx, expert_0_output, self.data_output, self.structure_output,criterion_disc)
                    # disc_loss = self.discriminator_loss(e1_idx, expert_1_output, self.data_output, self.structure_output,criterion_disc)+disc_loss
                    # disc_loss = self.discriminator_loss(e2_idx, expert_2_output, self.data_output, self.structure_output,criterion_disc)+disc_loss
                    #
                    disc_optimizer.zero_grad(set_to_none=True)
                    disc_loss.backward()
                    disc_optimizer.step()

                    loss = self.loss_calculation(m_idx,model_output,self.data_input,self.data_output,self.structure_input,self.structure_output,criterion_model, norm)
                    e0loss = self.loss_calculation(e0_idx,expert_0_output,self.data_input,self.data_output,self.structure_input,self.structure_output, criterion_e0, norm)
                    e1loss = self.loss_calculation(e1_idx,expert_1_output,self.data_input,self.data_output,self.structure_input,self.structure_output, criterion_e1, norm)
                    e2loss = self.loss_calculation(e2_idx,expert_2_output,self.data_input,self.data_output,self.structure_input,self.structure_output, criterion_e1, norm)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    e0loss.backward()
                    e1loss.backward()
                    e2loss.backward()
                    optimizer.step()
                    # if (epoch + 1) % 5 == 0:
                    # UnderConstruction! UnderConstruction! UnderConstruction!

                    self.model.eval()
                    model = copy.deepcopy(self.model)
                    state = self.parameterReinforcer.save_state(model)
                    action = self.parameterReinforcer(state)
                    action = self.parameterReinforcer.exploit_explore_action_selector(action)
                    self.parameterReinforcer.save_action(action)
                    with torch.no_grad():
                        model = self.parameterReinforcer.mutate_parameters(model, action)
                        model_mutated_output = model(dataset)
                        mutation_loss = self.loss_calculation(m_idx, model_mutated_output, self.data_input, self.data_output,
                                                      self.structure_input, self.structure_output, criterion_model,
                                                      norm)
                        self.parameterReinforcer.calculate_reward(loss.detach(), mutation_loss.detach())
                        if (epoch + 1) % 100 == 0:
                            for (name, param),(post_action_names,post_action_params) in zip(self.model.named_parameters(),model.named_parameters()):
                                param.copy_(post_action_params)
                            print("parameters mutated!")

                    next_model = copy.deepcopy(self.model)
                    next_state = self.parameterReinforcer.save_next_state(model)
                    next_action = self.parameterReinforcer(next_state)
                    next_action = self.parameterReinforcer.exploit_explore_action_selector(next_action)
                    self.parameterReinforcer.save_next_action(next_action)
                    with torch.no_grad():
                        next_model = self.parameterReinforcer.mutate_parameters(next_model, next_action)
                        model_mutated_output = next_model(dataset)
                        next_mutation_loss = self.loss_calculation(m_idx, model_mutated_output, self.data_input, self.data_output,
                                                      self.structure_input, self.structure_output, criterion_model,
                                                      norm)
                        self.parameterReinforcer.calculate_next_reward(loss.detach(), next_mutation_loss.detach())
                    Q_target = self.parameterReinforcer.Q_Value()
                    # print(torch.flatten(action[-1]).max(),torch.flatten(target_policy.squeeze(0).max()))
                    RLoss = criterion_RL(action,Q_target)
                    RL_optimizer.zero_grad(set_to_none=True)
                    RLoss.backward()
                    RL_optimizer.step()
                    del model,next_model
                    # self.parameterReinforcer.next_to_current()
                    self.model.train()
                    # UnderConstruction! UnderConstruction! UnderConstruction!

                    if self.validation_dataset is not None:
                        self.model.eval()
                        with torch.no_grad():
                            val_model_output = self.model(self.validation_dataset)
                            val_loss = self.loss_calculation(val_idx,val_model_output,self.data_input_val,self.data_output_val,self.structure_input_val,self.structure_output_val, criterion_model, norm)
                        self.model.train()

                    self.train_loss.append(loss.item())
                    self.val_loss.append(val_loss.item())

                    # t_stop = time.perf_counter()
                    t += (t_pred - t_start)/4
                    if  epoch > 25:
                        if val_loss < min(self.val_loss[:-1]):
                            model_to_Save = self.model
                            print('saved_checkpoint')

                    if len(self.train_loss) > 10:
                        loss_recent_history = np.array(self.train_loss)[-10:-1]
                        val_loss_recent_history = np.array(self.val_loss)[-10:-1]
                        mean_hist_losses = np.mean(loss_recent_history)
                        if loss_recent_history[-1] > loss_recent_history[-2] or reiterate_counter < 50 or loss_recent_history[-1] < loss_recent_history[-2]*0.9 or loss_recent_history[-1] > 0.3:
                            reiterate_data = 1
                        else:
                            reiterate_counter = 0
                            reiterate_data = 0
                        gloss = abs(np.sum(np.gradient(loss_recent_history)))
                        g_val_loss= np.sum(np.gradient(val_loss_recent_history))
                        if g_val_loss > 5e1:
                            reiterate_data = 0
                        if gloss > 5e-1:
                            grad_counter =0
                        else:
                            grad_counter += 1
                        # NOTE: lowering lr for  better performance and reset lr within conditions
                        if grad_counter == 3 or reiterate_data == 0:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = param_group['lr']*0.95
                                if param_group['lr'] < 1e-5 or reiterate_data == 0:
                                    param_group['lr'] = 1e-3
                                    reiterate_counter = 0
                                    reiterate_data = 0
                                    print('optimizer -> lr back to starting point')
                            self.discriminator.weight_reset()
                            self.discriminator.init_weights()
                            disc_optimizer =  torch.optim.Adam(self.discriminator.parameters(),lr=5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=True)

                               # noise_amplitude = noise_amplitude*0.5
                                # if noise_amplitude < 1e-3:
                                #     # print('noise amplitude')
                                #     noise_amplitude = 0.
                            grad_counter = 0
                        # NOTE: Averaging models with best loss results within all models

                        if (epoch + 1) % 1 == 0 or reiterate_data == 0:
                            if loss.item() < mean_hist_losses  or e0loss.item() < mean_hist_losses or e1loss.item() < mean_hist_losses or e2loss.item() < mean_hist_losses or best_loss < mean_hist_losses or reiterate_data == 0:
                                if loss < e0loss and loss<e1loss and loss<e2loss:
                                    best_loss = loss.item()
                                    best_losses.append(best_loss)
                                    best_models.append(self.model)
                                elif e0loss < loss and e0loss < e1loss and e0loss < e2loss:
                                    best_loss = e0loss.item()
                                    best_losses.append(best_loss)
                                    best_models.append(self.expert_0)
                                elif e1loss < loss and e1loss < e0loss and e1loss < e2loss:
                                    best_loss = e1loss.item()
                                    best_losses.append(best_loss)
                                    best_models.append(self.expert_1)
                                else:
                                    best_loss = e2loss.item()
                                    best_losses.append(best_loss)
                                    best_models.append(self.expert_2)

                                if len(best_models) > 200:
                                    best_losses = torch.tensor(np.array(best_losses))
                                    n = 5
                                    _,best_n_losses_idx = torch.topk(best_losses,n,largest=False)
                                    #best_loss_pos = best_losses.argmin()
                                    best_losses_norm = 1/(best_losses / best_losses.min())
                                    # with torch.no_grad():
                                    model_avg_damping = self.model
                                    model_avg_enhance = self.model
                                    param_sum_damping = {name: torch.zeros_like(param) for name, param in
                                                 model_avg_damping.named_parameters()}
                                    param_sum_enhance = {name: torch.zeros_like(param) for name, param in
                                                         model_avg_enhance.named_parameters()}

                                    for m in best_n_losses_idx:
                                        for (name_best, param_best), (name, param) in zip(best_models[best_n_losses_idx[0]].named_parameters(),best_models[m].named_parameters()):
                                            param_best_sign = torch.sgn(param_best)
                                            param_sign = torch.sgn(param)

                                            opposite_sign_mask = param_best_sign != param_sign
                                            same_sign_mask = param_best_sign == param_sign

                                            param_sum_damping[name][opposite_sign_mask] += param[opposite_sign_mask] * best_losses_norm[m]
                                            param_sum_enhance[name][same_sign_mask] += param[same_sign_mask] * best_losses_norm[m]

                                    with torch.no_grad():
                                        for name, param in model_avg_damping.named_parameters():
                                            param_avg = param_sum_damping[name] / n
                                            param.copy_(param_avg)

                                        for name, param in model_avg_enhance.named_parameters():
                                            param_avg = param_sum_enhance[name] / n
                                            param.copy_(param_avg)

                                        for (name, param),(name_enh, param_enh), (name_damp, param_damp) in zip(self.model.named_parameters(),model_avg_enhance.named_parameters(), model_avg_damping.named_parameters()):
                                            param_selector = random.randint(0,4) # Note : 20% chance to enhance or damp parameter
                                            if param_selector == 0:
                                                param.copy_(param_enh)
                                            elif param_selector == 1:
                                                param.copy_(param_damp)
                                            else:
                                                pass

                                    self.discriminator.weight_reset()
                                    self.discriminator.init_weights()
                                    disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=5e-4,
                                                                      betas=(0.9, 0.999), eps=1e-08,
                                                                      weight_decay=1e-6, amsgrad=True)
                                    print('model_avg -> weighted average -> main')
                                    best_models = []
                                    best_losses = []
                                    best_loss = mean_hist_losses



                    t_epoch_stop = time.perf_counter()
                    t_epoch +=(t_epoch_stop - t_epoch_start)

                    if (epoch + 1) % print_every_nth_frame == 0:
                        t_epoch_total = num_epochs * t_epoch
                        t_epoch_current = epoch * t_epoch
                        print(f'P: {self.period}/{self.no_of_periods} | E: {((t_epoch_total-t_epoch_current)/(print_every_nth_frame*60)):.2f} [min], '
                              f'vL: {val_loss.item():.3f}, '
                              f'mL: {loss.item():.3f}, '
                              f'R: {self.parameterReinforcer.rewards[-1].item():.0f}, '
                              f'RLoss {RLoss.item():.5f} '
                              f'dL: {disc_loss.item():.3f}, '
                              f'e0L: {e0loss.item():.2f}, '
                              f'e1L: {e1loss.item():.2f}, '
                              f'e2L: {e2loss.item():.2f}, '
                              f'tpf: {((self.fsim.grid_size_x * self.fsim.grid_size_y) / (self.model.in_scale ** 2)) * (t * 1e3 / print_every_nth_frame / self.batch_size):.2f} [ms]')
                        t = 0.
                        t_epoch=0.

                # if best_model_state is None:
                #     pass
                # else:
                #     self.model.load_state_dict(torch.load('model.pt'))
            else:
                pass
            torch.save(model_to_Save.state_dict(), 'model.pt')
            print('model_saved on disk')
            return self.model

    def dreaming_phase(self):
        pass


    def visualize_lerning(self):
        plt.plot(self.train_loss)
        plt.plot(self.val_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()



    def loss_calculation(self,idx,model_output,data_input,data_output,structure_input,structure_output,criterion,norm='forward'):
        pred_r,pred_g,pred_b,pred_a,pred_s = model_output

        r_in = data_input[:, 0:self.model.in_scale, :][idx]
        g_in = data_input[:,self.model.in_scale:self.model.in_scale * 2,:][idx]
        b_in = data_input[:,self.model.in_scale * 2:self.model.in_scale * 3,:][idx]
        a_in = data_input[:, self.model.in_scale * 3:self.model.in_scale * 4, :][idx]
        s_in = structure_input[idx]

        r_out = data_output[:, 0:self.model.in_scale, :][idx]
        g_out = data_output[:, self.model.in_scale:self.model.in_scale * 2, :][idx]
        b_out = data_output[:, self.model.in_scale * 2:self.model.in_scale * 3, :][idx]
        a_out = data_output[:, self.model.in_scale * 3:self.model.in_scale * 4, :][idx]
        s_out = structure_output[idx]

        # Solution for learning of the dynamics in loss calculation
        # NOTE: Firs order difference
        diff_r_true = r_out - r_in
        diff_r_pred = pred_r - r_in
        # diff_r_pred = torch.gradient(torch.cat([data_in_r.unsqueeze(3),pred_r.unsqueeze(3)],dim=3),dim=3)[0][:, :, :, 1]
        loss_diff_r = criterion(diff_r_pred, diff_r_true)
        diff_g_true = g_out- g_in
        diff_g_pred = pred_g - g_in
        loss_diff_g = criterion(diff_g_pred, diff_g_true)
        diff_b_true = b_out - b_in
        diff_b_pred = pred_b - b_in
        loss_diff_b = criterion(diff_b_pred, diff_b_true)
        diff_a_true = a_out - a_in
        diff_a_pred = pred_a - a_in
        loss_diff_a = criterion(diff_a_pred, diff_a_true)
        diff_s_true = s_out - s_in
        diff_s_pred = pred_s - s_in
        loss_diff_s = criterion(diff_s_pred, diff_s_true)
        diff_loss = loss_diff_r + loss_diff_g + loss_diff_b + loss_diff_a + loss_diff_s

        # Note: Gradient loss
        grad_r_true = torch.gradient(r_out, dim=[1, 2])[0]
        grad_r_pred = torch.gradient(pred_r)[0]
        grad_r = criterion(grad_r_pred, grad_r_true)
        grad_g_true = torch.gradient(g_out, dim=[1, 2])[0]
        grad_g_pred = torch.gradient(pred_g)[0]
        grad_g = criterion(grad_g_pred, grad_g_true)
        grad_b_true = torch.gradient(b_out, dim=[1, 2])[0]
        grad_b_pred = torch.gradient(pred_b)[0]
        grad_b = criterion(grad_b_pred, grad_b_true)
        grad_a_true = torch.gradient(a_out, dim=[1, 2])[0]
        grad_a_pred = torch.gradient(pred_a)[0]
        grad_a = criterion(grad_a_pred, grad_a_true)
        grad_s_true = torch.gradient(s_out, dim=[1, 2])[0]
        grad_s_pred = torch.gradient(pred_s)[0]
        grad_s = criterion(grad_s_pred, grad_s_true)

        grad_loss = grad_r + grad_g + grad_b + grad_a + grad_s



        # Note: Fourier loss
        fft_out_pred_r = torch.real(torch.fft.rfft2(pred_r, norm=norm))
        fft_out_true_r = torch.real(torch.fft.rfft2(r_out, norm=norm))
        fft_out_pred_g = torch.real(torch.fft.rfft2(pred_g, norm=norm))
        fft_out_true_g = torch.real(torch.fft.rfft2(g_out, norm=norm))
        fft_out_pred_b = torch.real(torch.fft.rfft2(pred_b, norm=norm))
        fft_out_true_b = torch.real(torch.fft.rfft2(b_out, norm=norm))
        fft_out_pred_a = torch.real(torch.fft.rfft2(pred_a, norm=norm))
        fft_out_true_a = torch.real(torch.fft.rfft2(a_out, norm=norm))
        fft_out_pred_s = torch.real(torch.fft.rfft2(pred_s, norm=norm))
        fft_out_true_s = torch.real(torch.fft.rfft2(s_out, norm=norm))

        fft_in_true_r = torch.real(torch.fft.rfft2(r_in, norm=norm))
        fft_in_true_g = torch.real(torch.fft.rfft2(g_in,norm=norm))
        fft_in_true_b = torch.real(torch.fft.rfft2(b_in,norm=norm))
        fft_in_true_a = torch.real(torch.fft.rfft2(a_in,norm=norm))
        fft_in_true_s = torch.real(torch.fft.rfft2(s_in, norm=norm))

        fft_loss_r = criterion(fft_out_pred_r, fft_out_true_r)
        fft_loss_g = criterion(fft_out_pred_g, fft_out_true_g)
        fft_loss_b = criterion(fft_out_pred_b, fft_out_true_b)
        fft_loss_a = criterion(fft_out_pred_a, fft_out_true_a)
        fft_loss_s = criterion(fft_out_pred_s, fft_out_true_s)
        fft_loss = fft_loss_r + fft_loss_g + fft_loss_b + fft_loss_a + fft_loss_s

        # Note: Fourier Gradient Loss
        diff_fft_true_r = fft_out_true_r - fft_in_true_r
        diff_fft_pred_r = fft_out_pred_r - fft_in_true_r
        diff_fft_loss_r = criterion(diff_fft_pred_r, diff_fft_true_r)
        diff_fft_true_g = fft_out_true_g - fft_in_true_g
        diff_fft_pred_g = fft_out_pred_g - fft_in_true_g
        diff_fft_loss_g = criterion(diff_fft_pred_g, diff_fft_true_g)
        diff_fft_true_b = fft_out_true_b - fft_in_true_b
        diff_fft_pred_b = fft_out_pred_b - fft_in_true_b
        diff_fft_loss_b = criterion(diff_fft_pred_b, diff_fft_true_b)
        diff_fft_true_a = fft_out_true_a - fft_in_true_a
        diff_fft_pred_a = fft_out_pred_a - fft_in_true_a
        diff_fft_loss_a = criterion(diff_fft_pred_a, diff_fft_true_a)
        diff_fft_true_s = fft_out_true_s - fft_in_true_s
        diff_fft_pred_s = fft_out_pred_s - fft_in_true_s
        diff_fft_loss_s = criterion(diff_fft_pred_s, diff_fft_true_s)
        diff_fft_loss = diff_fft_loss_r + diff_fft_loss_g + diff_fft_loss_b + diff_fft_loss_a + diff_fft_loss_s

        # Note : Exact value loss
        loss_r = criterion(pred_r, r_out)
        loss_g = criterion(pred_g, g_out)
        loss_b = criterion(pred_b, b_out)
        loss_alpha = criterion(pred_a, a_out)
        loss_s = criterion(pred_s, s_out)
        value_loss = loss_r + loss_g + loss_b + loss_alpha + loss_s

        # Solution for learning and maintaining of the proper color and other element space
        bandwidth = torch.tensor(0.1).to(self.device) # Note: Higher value less noise (gaussian smothing)
        bins = 255
        r_out = torch.flatten(r_out,start_dim=1)
        pred_r = torch.flatten(pred_r, start_dim=1)
        bins_true = torch.linspace(r_out.min(), r_out.max(), bins).to(self.device)
        bins_pred = torch.linspace(pred_r.min().tolist(), pred_r.max().tolist(), bins).to(self.device)
        r_true_hist = kornia.enhance.histogram(r_out,bins=bins_true,bandwidth=bandwidth)
        r_pred_hist = kornia.enhance.histogram(pred_r,bins=bins_pred,bandwidth=bandwidth)
        r_hist_loss = criterion(r_pred_hist,r_true_hist)

        g_out = torch.flatten(g_out, start_dim=1)
        pred_g = torch.flatten(pred_g, start_dim=1)
        bins_true = torch.linspace(g_out.min(), g_out.max(), bins).to(self.device)
        bins_pred = torch.linspace(pred_g.min().tolist(), pred_g.max().tolist(), bins).to(self.device)
        g_true_hist = kornia.enhance.histogram(g_out, bins=bins_true, bandwidth=bandwidth)
        g_pred_hist = kornia.enhance.histogram(pred_g, bins=bins_pred, bandwidth=bandwidth)
        g_hist_loss = criterion(g_pred_hist, g_true_hist)

        b_out = torch.flatten(b_out, start_dim=1)
        pred_b = torch.flatten(pred_b, start_dim=1)
        bins_true = torch.linspace(b_out.min(), b_out.max(), bins).to(self.device)
        bins_pred = torch.linspace(pred_b.min().tolist(), pred_b.max().tolist(), bins).to(self.device)
        b_true_hist = kornia.enhance.histogram(b_out, bins=bins_true, bandwidth=bandwidth)
        b_pred_hist = kornia.enhance.histogram(pred_b, bins=bins_pred, bandwidth=bandwidth)
        b_hist_loss = criterion(b_pred_hist, b_true_hist)

        a_out = torch.flatten(a_out, start_dim=1)
        pred_a = torch.flatten(pred_a, start_dim=1)
        bins_true = torch.linspace(a_out.min(), a_out.max(), bins).to(self.device)
        bins_pred = torch.linspace(pred_a.min().tolist(), pred_a.max().tolist(), bins).to(self.device)
        a_true_hist = kornia.enhance.histogram(a_out, bins=bins_true, bandwidth=bandwidth)
        a_pred_hist = kornia.enhance.histogram(pred_a, bins=bins_pred, bandwidth=bandwidth)
        a_hist_loss = criterion(a_pred_hist, a_true_hist)

        s_out = torch.flatten(s_out, start_dim=1)
        pred_s = torch.flatten(pred_s, start_dim=1)
        bins_true = torch.linspace(s_out.min(), s_out.max(), bins).to(self.device)
        bins_pred = torch.linspace(pred_s.min().tolist(), pred_s.max().tolist(), bins).to(self.device)
        s_true_hist = kornia.enhance.histogram(s_out, bins=bins_true, bandwidth=bandwidth)
        s_pred_hist = kornia.enhance.histogram(pred_s, bins=bins_pred, bandwidth=bandwidth)
        s_hist_loss = criterion(s_pred_hist, s_true_hist)

        hist_loss = r_hist_loss + b_hist_loss + g_hist_loss + a_hist_loss +s_hist_loss
        hist_loss = hist_loss*10
        LOSS = value_loss + diff_loss + grad_loss + fft_loss + diff_fft_loss+hist_loss # Attention: Aggregate all losses here
        return LOSS

    def discriminator_loss(self,idx,model_output,data_output,structure_output,criterion):

        dataset = (
        self.data_input[idx], self.structure_input[idx], self.meta_input_h1[idx], self.meta_input_h2[idx],
        self.meta_input_h3[idx], self.meta_input_h4[idx], self.meta_input_h5[idx], self.noise_var_in[idx],
        self.meta_output_h1[idx],
        self.meta_output_h2[idx], self.meta_output_h3[idx], self.meta_output_h4[idx], self.meta_output_h5[idx],
        self.noise_var_out[idx])

        pred_r, pred_g, pred_b, pred_a, pred_s = model_output
        r_out = data_output[:, 0:self.model.in_scale, :][idx]
        g_out = data_output[:, self.model.in_scale:self.model.in_scale * 2, :][idx]
        b_out = data_output[:, self.model.in_scale * 2:self.model.in_scale * 3, :][idx]
        a_out = data_output[:, self.model.in_scale * 3:self.model.in_scale * 4, :][idx]
        s_out = structure_output[idx]
        pred = torch.cat([pred_r.detach(), pred_g.detach(), pred_b.detach(), pred_a.detach(), pred_s.detach()],dim=1)
        true = torch.cat([r_out, g_out, b_out, a_out,s_out],dim=1)
        fake_labels = torch.zeros((pred.shape[0],1)).to(self.device)
        true_labels = torch.ones((true.shape[0], 1)).to(self.device)
        combined_data = torch.cat([pred,true],dim=0)
        combined_labels = torch.cat([fake_labels,true_labels],dim=0)
        shuffle_idx = torch.randint(0,combined_data.shape[0],(int(combined_data.shape[0]/2),)).to(self.device)
        shuffled_labels = combined_labels[shuffle_idx]
        disc_pred = self.discriminator(combined_data,dataset,shuffle_idx)
        disc_loss = criterion(disc_pred, shuffled_labels)
        return disc_loss


    @staticmethod
    def seed_setter(seed):
        s = torch.randint(0,seed,(1,))
        torch.manual_seed(2024 + s)
        s = np.random.randint(0,seed)
        np.random.seed(2024 + s)
        s = random.randint(0, seed)
        random.seed(2024 + s)






