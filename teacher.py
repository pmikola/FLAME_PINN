import os.path
import random
import struct
import time

import numpy as np
import torch


class teacher(object):
    def __init__(self,model,device):
        super(teacher, self).__init__()
        self.model = model
        self.device = device
        self.fsim = None
        self.data_tensor = None
        self.meta_tensor = None
        self.meta_binary = None
        self.field_names = None
        self.first_frame, self.last_frame, self.frame_skip = None, None, None

        self.data_input = None
        self.meta_input_h1 = None
        self.meta_input_h2 = None
        self.meta_input_h3 = None
        self.data_output = None
        self.meta_output_h1 = None
        self.meta_output_h2 = None
        self.meta_output_h3 = None
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

    def data_preparation(self,no_frame_samples,batch_size,input_window_size,first_frame,last_frame,frame_skip):
        self.first_frame, self.last_frame, self.frame_skip = first_frame, last_frame, frame_skip
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
                            data_tensor.append(ptfile['data'][:,:,j]/255.)
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
        frame_samples = random.sample(list(set(fdens_idx)), k=no_frame_samples)
        f_dens_pos = len(fdens_idx)
        fdens_idx = fdens_idx[frame_samples]
        rgb_idx = np.array([i for i, x in enumerate(self.field_names) if x == "rgb"])
        r_idx = rgb_idx[0:f_dens_pos][frame_samples]
        g_idx = rgb_idx[f_dens_pos:f_dens_pos * 2][frame_samples]
        b_idx = rgb_idx[f_dens_pos * 2:f_dens_pos * 3][frame_samples]
        alpha_idx = np.array([i for i, x in enumerate(self.field_names) if x == "alpha"])[frame_samples]
        fuel_slices = self.data_tensor[fdens_idx]
        r_slices = self.data_tensor[r_idx]
        g_slices = self.data_tensor[g_idx]
        b_slices = self.data_tensor[b_idx]
        alpha_slices = self.data_tensor[alpha_idx]
        meta_binary_slices = self.meta_binary[fdens_idx]
        x_range = range(self.fsim.N_boundary + input_window_size,
                        fuel_slices[0].shape[0] - self.fsim.N_boundary - input_window_size)
        y_range = range(self.fsim.N_boundary + input_window_size,
                        fuel_slices[0].shape[1] - self.fsim.N_boundary - input_window_size)
        data_input = []
        meta_input_h1 = []
        meta_input_h2 = []
        meta_input_h3 = []
        data_output = []
        meta_output_h1 = []
        meta_output_h2 = []
        meta_output_h3 = []
        for _ in range(batch_size):
            idx_input = random.choice(range(0, fuel_slices.shape[0]))
            idx_output = random.choice(range(0, fuel_slices.shape[0]))

            central_point_x = random.sample(x_range, 1)[0]
            central_point_y = random.sample(y_range, 1)[0]
            window_x = np.array(range(central_point_x - input_window_size, central_point_x + input_window_size + 1))
            window_y = np.array(range(central_point_y - input_window_size, central_point_y + input_window_size + 1))
            central_point_x_binary = "{0:010b}".format(central_point_x)
            central_point_x_binary = torch.tensor(np.array([int(d) for d in central_point_x_binary]))
            central_point_y_binary = "{0:010b}".format(central_point_y)
            central_point_y_binary = torch.tensor(np.array([int(d) for d in central_point_y_binary]))
            slice_x = slice(window_x[0], window_x[-1] + 1)
            slice_y = slice(window_y[0], window_y[-1] + 1)
            # Note : Input data
            fuel_subslice_in = fuel_slices[idx_input][slice_x, slice_y]
            r_subslice_in = r_slices[idx_input][slice_x, slice_y]
            g_subslice_in = g_slices[idx_input][slice_x, slice_y]
            b_subslice_in = b_slices[idx_input][slice_x, slice_y]
            alpha_subslice_in = alpha_slices[idx_input][slice_x, slice_y]
            data_input_subslice = torch.cat([fuel_subslice_in, r_subslice_in,
                                             g_subslice_in, b_subslice_in, alpha_subslice_in], dim=0)
            meta_step_in = meta_binary_slices[idx_input][0]
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
            fuel_subslice_out = fuel_slices[idx_output][central_point_x, central_point_y].reshape(1)

            r_subslice_out = r_slices[idx_output][central_point_x, central_point_y].reshape(1)
            g_subslice_out = g_slices[idx_output][central_point_x, central_point_y].reshape(1)
            b_subslice_out = b_slices[idx_output][central_point_x, central_point_y].reshape(1)
            alpha_subslice_out = alpha_slices[idx_output][central_point_x, central_point_y].reshape(1)
            data_output_subslice = torch.cat([fuel_subslice_out, r_subslice_out,
                                              g_subslice_out, b_subslice_out, alpha_subslice_out], dim=0)
            meta_step_out = meta_binary_slices[idx_output][0]
            meta_fuel_initial_speed_out = meta_binary_slices[idx_output][1]
            meta_fuel_cut_off_time_out = meta_binary_slices[idx_output][2]
            meta_igni_time_out = meta_binary_slices[idx_output][3]
            meta_ignition_temp_out = meta_binary_slices[idx_output][4]
            meta_viscosity_out = meta_binary_slices[idx_output][14]
            meta_diff_out = meta_binary_slices[idx_output][15]
            meta_output_subslice = torch.cat([meta_step_out, meta_fuel_initial_speed_out,
                                              meta_fuel_cut_off_time_out, meta_igni_time_out,
                                              meta_ignition_temp_out, meta_viscosity_out, meta_diff_out], dim=0)

            central_points = torch.cat([central_point_x_binary, central_point_y_binary], dim=0)
            data_input.append(data_input_subslice)
            meta_input_h1.append(meta_input_subslice)
            meta_input_h2.append(meta_step_in)
            meta_input_h3.append(central_points)
            data_output.append(data_output_subslice)
            meta_output_h1.append(meta_output_subslice)
            meta_output_h2.append(data_output_subslice)
            meta_output_h3.append(central_points)


        self.data_input = torch.stack(data_input,dim=0)
        self.meta_input_h1 = meta_input_h1
        self.meta_input_h2 = meta_input_h2
        self.meta_input_h3 = meta_input_h3
        self.data_output = data_output
        self.meta_output_h1 = meta_output_h1
        self.meta_output_h2 = meta_output_h2
        self.meta_output_h3 = meta_output_h3

        time.sleep(100)


    def learning_phase(self,criterion,optimizer,device,num_epochs=100):
            global loss
            best_loss = float('inf')
            best_model_state = None
            num_epochs = num_epochs
            for epoch in range(num_epochs):
                (self.data_input,self.meta_input_h1,self.meta_input_h2,
                 self.meta_input_h3,self.data_output,self.meta_output_h1,
                 self.meta_output_h2,self.meta_output_h3) =(self.data_input.to(device),
                                                            self.meta_input_h1.to(device),
                                                            self.meta_input_h2.to(device),
                                                            self.meta_input_h3.to(device),
                                                            self.data_output.to(device),
                                                            self.meta_output_h1.to(device),
                                                            self.meta_output_h2.to(device),
                                                            self.meta_output_h3.to(device))
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

                if (epoch + 1) % 10 == 0:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_model_state = self.model.state_dict()
            if best_model_state is None:
                pass
            else:
                self.model = best_model_state
            return self.model

    def dreaming_phase(self):
        pass