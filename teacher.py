import os.path
import random
import struct
import time

import numpy as np
import torch
from matplotlib import pyplot as plt


class teacher(object):
    def __init__(self,model,device):
        super(teacher, self).__init__()
        self.model = model
        self.device = device
        self.fsim = None
        self.period = 1
        self.no_of_periods = 1
        self.data_tensor = None
        self.meta_tensor = None
        self.meta_binary = None
        self.field_names = None
        self.no_frame_samples,self.first_frame, self.last_frame, self.frame_skip =None, None, None, None

        self.data_input = None
        self.meta_input_h1 = None
        self.meta_input_h2 = None
        self.meta_input_h3 = None
        self.meta_input_h4 = None
        self.meta_input_h5 = None
        self.data_output = None
        self.meta_output_h1 = None
        self.meta_output_h2 = None
        self.meta_output_h3 = None
        self.meta_output_h4 = None
        self.meta_output_h5 = None

        self.saved_loss = []
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

    def data_preparation(self):
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
        frame_samples = random.sample(list(set(fdens_idx)), k=self.no_frame_samples)
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
        x_range = range(self.fsim.N_boundary + self.input_window_size,
                        fuel_slices[0].shape[0] - self.fsim.N_boundary - self.input_window_size)
        y_range = range(self.fsim.N_boundary + self.input_window_size,
                        fuel_slices[0].shape[1] - self.fsim.N_boundary - self.input_window_size)
        data_input = []
        meta_input_h1 = []
        meta_input_h2 = []
        meta_input_h3 = []
        meta_input_h4 = []
        meta_input_h5 = []
        data_output = []
        meta_output_h1 = []
        meta_output_h2 = []
        meta_output_h3 = []
        meta_output_h4 = []
        meta_output_h5 = []
        for _ in range(self.batch_size):
            idx_input = random.choice(range(0, fuel_slices.shape[0]))
            idx_output = random.choice(range(0, fuel_slices.shape[0]))
            central_point_x = random.sample(x_range, 1)[0]
            central_point_y = random.sample(y_range, 1)[0]
            window_x = np.array(range(central_point_x - self.input_window_size, central_point_x + self.input_window_size + 1))
            window_y = np.array(range(central_point_y - self.input_window_size, central_point_y + self.input_window_size + 1))
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
            fuel_subslice_out = fuel_slices[idx_output][slice_x, slice_y]#.reshape(1)
            r_subslice_out = r_slices[idx_output][slice_x, slice_y]#.reshape(1)
            g_subslice_out = g_slices[idx_output][slice_x, slice_y]#.reshape(1)
            b_subslice_out = b_slices[idx_output][slice_x, slice_y]#.reshape(1)
            alpha_subslice_out = alpha_slices[idx_output][slice_x, slice_y]#.reshape(1)
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

            # Note: Data for the different layers
            central_points = torch.cat([central_point_x_binary, central_point_y_binary], dim=0)
            data_input.append(data_input_subslice)
            meta_input_h1.append(meta_input_subslice)
            meta_input_h2.append(meta_step_in)
            meta_input_h3.append(central_points)
            meta_input_h4.append(torch.cat([torch.tensor(window_x),torch.tensor(window_y)]))
            meta_input_h5.append(meta_step_in_numeric)
            data_output.append(data_output_subslice)
            meta_output_h1.append(meta_output_subslice)
            meta_output_h2.append(meta_step_out)
            meta_output_h3.append(central_points)
            meta_output_h4.append(torch.cat([torch.tensor(window_x), torch.tensor(window_y)]))
            meta_output_h5.append(meta_step_out_numeric)

        self.data_input = torch.stack(data_input,dim=0)
        self.meta_input_h1 = torch.stack(meta_input_h1,dim=0)
        self.meta_input_h2 = torch.stack(meta_input_h2,dim=0)
        self.meta_input_h3 = torch.stack(meta_input_h3,dim=0)
        self.meta_input_h4 = torch.stack(meta_input_h4,dim=0)
        self.meta_input_h5 = torch.stack(meta_input_h5,dim=0)

        self.data_output = torch.stack(data_output,dim=0)
        self.meta_output_h1 = torch.stack(meta_output_h1,dim=0)
        self.meta_output_h2 = torch.stack(meta_output_h2,dim=0)
        self.meta_output_h3 = torch.stack(meta_output_h3,dim=0)
        self.meta_output_h4 = torch.stack(meta_output_h4,dim=0)
        self.meta_output_h5 = torch.stack(meta_output_h5,dim=0)


    def learning_phase(self,no_frame_samples, batch_size, input_window_size, first_frame, last_frame,
                                      frame_skip,criterion,optimizer,device,learning=1,num_epochs=1500):
            (self.no_frame_samples,self.batch_size,self.input_window_size,self.first_frame,
             self.last_frame,self.frame_skip) = (no_frame_samples, batch_size,
                                                 input_window_size, first_frame, last_frame,frame_skip)
            global loss
            if learning == 1:
                best_loss = float('inf')
                best_model_state = None
                num_epochs = num_epochs
                for epoch in range(num_epochs):
                    self.data_preparation()
                    (self.data_input,self.meta_input_h1,self.meta_input_h2,
                     self.meta_input_h3,self.meta_input_h4,self.meta_input_h5,self.meta_output_h1,
                     self.meta_output_h2,self.meta_output_h3,self.meta_output_h4,self.meta_output_h5) =\
                                                                (self.data_input.to(device),
                                                                self.meta_input_h1.to(device),
                                                                self.meta_input_h2.to(device),
                                                                self.meta_input_h3.to(device),
                                                                self.meta_input_h4.to(device),
                                                                self.meta_input_h5.to(device),
                                                                self.meta_output_h1.to(device),
                                                                self.meta_output_h2.to(device),
                                                                self.meta_output_h3.to(device),
                                                                self.meta_output_h4.to(device),
                                                                self.meta_output_h5.to(device))


                    self.data_output = self.data_output.to(device)
                    dataset = (self.data_input,self.meta_input_h1,self.meta_input_h2,
                               self.meta_input_h3,self.meta_input_h4,self.meta_input_h5,self.meta_output_h1,
                               self.meta_output_h2,self.meta_output_h3,self.meta_output_h4,self.meta_output_h5)
                    pred_r,pred_g,pred_b,pred_a = self.model(dataset)

                    loss_r = criterion(pred_r, self.data_output[:,0:5,:])
                    loss_g = criterion(pred_g, self.data_output[:,5:10,:])
                    loss_b = criterion(pred_b, self.data_output[:,10:15,:])
                    loss_alpha = criterion(pred_a, self.data_output[:,15:20,:])

                    loss = loss_r + loss_g + loss_b + loss_alpha # TODO : Add Endropy loss + diversity loss + intermidiete velocity vectors loss + casual loss + grad pinn rgb loss
                    self.saved_loss.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    if (epoch+1) % 10 == 0:
                        print(f'Period: {self.period}/{self.no_of_periods} | Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

                    if (epoch + 1) % 10 == 0:
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            torch.save(self.model.state_dict(), 'model.pt')
                if best_model_state is None:
                    pass
                else:
                    self.model.load_state_dict(torch.load('model.pt'))
            else:
                pass
            return self.model

    def dreaming_phase(self):
        pass


    def visualize_lerning(self):
        plt.plot(self.saved_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    def examine(self,criterion,device):
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
        r_idx = rgb_idx[0:f_dens_pos]#[frame_samples]
        g_idx = rgb_idx[f_dens_pos:f_dens_pos * 2]#[frame_samples]
        b_idx = rgb_idx[f_dens_pos * 2:f_dens_pos * 3]#[frame_samples]
        alpha_idx = np.array([i for i, x in enumerate(self.field_names) if x == "alpha"])#[frame_samples]
        fuel_slices = self.data_tensor[fdens_idx]

        r_slices = self.data_tensor[r_idx]
        g_slices = self.data_tensor[g_idx]
        b_slices = self.data_tensor[b_idx]
        alpha_slices = self.data_tensor[alpha_idx]
        meta_binary_slices = self.meta_binary[fdens_idx]
        data_input = []
        meta_input_h1 = []
        meta_input_h2 = []
        meta_input_h3 = []
        meta_input_h4 = []
        meta_input_h5 = []
        data_output = []
        meta_output_h1 = []
        meta_output_h2 = []
        meta_output_h3 = []
        meta_output_h4 = []
        meta_output_h5 = []
        # Note: IDX preparation
        central_points_x = np.arange(self.input_window_size, fuel_slices.shape[1] - self.input_window_size)
        central_points_y = np.arange(self.input_window_size, fuel_slices.shape[2] - self.input_window_size)
        central_points_x_pos = central_points_x + self.input_window_size
        central_points_x_neg = central_points_x - self.input_window_size
        central_points_y_pos = central_points_y + self.input_window_size
        central_points_y_neg = central_points_y - self.input_window_size
        windows_x = []
        windows_y = []
        central_point_x_binary = []
        central_point_y_binary = []
        for j in range(0, central_points_x_pos.shape[0]):
            wx_range = range(int(central_points_x_neg[j]), int(central_points_x_pos[j]) + 1)
            windows_x.append(wx_range)
            central_point_x_binary.append("{0:010b}".format(central_points_x[j]))
            central_point_x_binary[j] = torch.tensor([torch.tensor(int(d), dtype=torch.int8) for d in central_point_x_binary[j]])
            if j >= central_points_y_neg.shape[0]:
                pass
            else:
                wy_range = range(int(central_points_y_neg[j]), int(central_points_y_pos[j]) + 1)
                windows_y.append(wy_range)
                central_point_y_binary.append("{0:010b}".format(central_points_y[j]))
                central_point_y_binary[j] = torch.tensor([torch.tensor(int(d), dtype=torch.int8) for d in central_point_y_binary[j]])

        windows_x = torch.tensor(np.array(windows_x))
        windows_y = torch.tensor(np.array(windows_y))

        wx_idx = torch.arange(0, len(windows_x))
        wy_idx = torch.arange(0, len(windows_y))
        # slice_x = [torch.arange(windows_x[k][0], windows_x[k][-1]+1) for k in wx_idx]
        # slice_y = [torch.arange(windows_y[k][0], windows_y[k][-1]+1) for k in wy_idx]
        slice_x = torch.stack([torch.arange(w[0], w[-1] + 1) for w in [windows_x[k] for k in wx_idx]], dim=0)
        slice_y = torch.stack([torch.arange(w[0], w[-1] + 1) for w in [windows_y[k] for k in wy_idx]], dim=0)

        n_x, n_y = slice_x.shape[0], slice_y.shape[0]

        mesh_x, mesh_y = torch.meshgrid(torch.arange(n_x), torch.arange(n_y), indexing='ij')

        x = slice_x[mesh_x]
        y = slice_y[mesh_y]
        addim = 4
        adim_half = int(addim / 2)
        nx = n_x + addim
        ny = n_y + addim
        cprod = torch.empty(n_x, n_y, self.model.in_scale ** 2, 2)
        central_points_x_binary = torch.stack(central_point_x_binary, dim=0)
        central_points_y_binary = torch.stack(central_point_y_binary, dim=0)
        central_points_xy_binary = []
        windows_xy = []
        for xx in range(n_x):
            central_points_yy_binary = []
            windows_yy = []
            for yy in range(n_y):
                x_slice = x[xx, yy, :]
                y_slice = y[xx, yy, :]
                xy_binary = torch.cat([central_points_x_binary[xx], central_points_y_binary[yy]])
                central_points_yy_binary.append(xy_binary)
                windows_yy.append(torch.cat([windows_x[xx], windows_y[yy]]))
                cart_prod = torch.cartesian_prod(x_slice, y_slice)
                cprod[xx, yy, :, :] = cart_prod
            central_points_xy_binary.append(central_points_yy_binary)
            windows_xy.append(windows_yy)
        windows_xy = torch.tensor(np.array(windows_xy))
        x_idx = cprod[:, :, :, 0].int()

        xx_idx = torch.zeros((n_x+addim,n_y+addim,self.model.in_scale ** 2),device=self.device)
        xx_idx[adim_half:n_x+adim_half,adim_half:n_y+adim_half,:] = x_idx
        xx_idx[:adim_half,:adim_half,:] = x_idx[:adim_half,:adim_half,:]
        xx_idx[n_x+adim_half:, n_y+adim_half:, :] = x_idx[n_x-adim_half:, n_y-adim_half:, :]
        xx_idx[n_x+adim_half:, :adim_half, :] = x_idx[n_x-adim_half:, :adim_half, :]
        xx_idx[:adim_half, n_y+adim_half:, :] = x_idx[:adim_half, n_y-adim_half:, :]
        y_idx = cprod[:, :, :, 1].int()
        yy_idx = torch.zeros((n_x+addim,n_y+addim,self.model.in_scale ** 2), device=self.device)
        yy_idx[adim_half:n_x+adim_half,adim_half:n_y+adim_half,:] = y_idx
        yy_idx[:adim_half,:adim_half,:] = y_idx[:adim_half,:adim_half,:]
        yy_idx[n_x+adim_half:, n_y+adim_half:, :] = y_idx[n_x-adim_half:, n_y-adim_half:, :]
        yy_idx[n_x+adim_half:, :adim_half, :] = y_idx[n_x-adim_half:, :adim_half, :]
        yy_idx[:adim_half, n_y+adim_half:, :] = y_idx[:adim_half, n_y-adim_half:, :]
        x_idx = xx_idx.int()
        y_idx = yy_idx.int()

        central_points_xy_binary = torch.tensor(np.array(central_points_xy_binary))
        cp_xy_bin = torch.zeros((nx,ny,central_points_xy_binary.shape[2]),device=self.device)
        cp_xy_bin[adim_half:n_x + adim_half, adim_half:n_y + adim_half, :] = central_points_xy_binary
        cp_xy_bin[:adim_half, :adim_half, :] = central_points_xy_binary[:adim_half, :adim_half, :]
        cp_xy_bin[n_x + adim_half:, n_y + adim_half:, :] = central_points_xy_binary[n_x - adim_half:, n_y - adim_half:, :]
        cp_xy_bin[n_x + adim_half:, :adim_half, :] = central_points_xy_binary[n_x - adim_half:, :adim_half, :]
        cp_xy_bin[:adim_half, n_y + adim_half:, :] = central_points_xy_binary[:adim_half, n_y - adim_half:, :]

        windows_xy = torch.tensor(np.array(windows_xy))
        w_xy = torch.zeros((nx, ny, windows_xy.shape[2]), device=self.device)
        w_xy[adim_half:n_x + adim_half, adim_half:n_y + adim_half, :] = windows_xy
        w_xy[:adim_half, :adim_half, :] = windows_xy[:adim_half, :adim_half, :]
        w_xy[n_x + adim_half:, n_y + adim_half:, :] = windows_xy[n_x - adim_half:, n_y - adim_half:,:]
        w_xy[n_x + adim_half:, :adim_half, :] = windows_xy[n_x - adim_half:, :adim_half, :]
        w_xy[:adim_half, n_y + adim_half:, :] = windows_xy[:adim_half, n_y - adim_half:, :]
        idx_center = int(self.model.in_scale ** 2 // 2)
        for i in range(0,fuel_slices.shape[0]-1):
            idx_input = i
            idx_output = i+1
            # Note : Input data
            fuel_subslice_in = fuel_slices[idx_input, x_idx,y_idx]
            r_subslice_in = r_slices[idx_input, x_idx,y_idx]
            g_subslice_in = g_slices[idx_input, x_idx,y_idx]
            b_subslice_in = b_slices[idx_input, x_idx,y_idx]
            alpha_subslice_in = alpha_slices[idx_input, x_idx,y_idx]

            data_input_subslice = torch.cat([fuel_subslice_in.unsqueeze(3), r_subslice_in.unsqueeze(3),
                                             g_subslice_in.unsqueeze(3), b_subslice_in.unsqueeze(3),
                                             alpha_subslice_in.unsqueeze(3)], dim=3)
            data_input_subslice = data_input_subslice
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
            #fuel_subslice_out = fuel_slices[idx_output,x_idx[:,:,idx_center].unsqueeze(2), y_idx[:,:,idx_center].unsqueeze(2)]
            r_subslice_out = r_slices[idx_output,x_idx[:,:,idx_center].unsqueeze(2), y_idx[:,:,idx_center].unsqueeze(2)]
            g_subslice_out = g_slices[idx_output,x_idx[:,:,idx_center].unsqueeze(2), y_idx[:,:,idx_center].unsqueeze(2)]
            b_subslice_out = b_slices[idx_output,x_idx[:,:,idx_center].unsqueeze(2), y_idx[:,:,idx_center].unsqueeze(2)]
            alpha_subslice_out = alpha_slices[idx_output,x_idx[:,:,idx_center].unsqueeze(2), y_idx[:,:,idx_center].unsqueeze(2)]
            data_output_subslice = torch.cat([r_subslice_out.unsqueeze(3), g_subslice_out.unsqueeze(3), b_subslice_out.unsqueeze(3), alpha_subslice_out.unsqueeze(3)],
                                             dim=3)
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
            data_input.append(data_input_subslice)
            meta_input_h1.append(meta_input_subslice)
            meta_input_h2.append(meta_step_in)
            meta_input_h3.append(cp_xy_bin)
            meta_input_h4.append(w_xy)
            meta_input_h5.append(meta_step_in_numeric)
            data_output.append(data_output_subslice)
            meta_output_h1.append(meta_output_subslice)
            meta_output_h2.append(meta_step_out)
            meta_output_h3.append(cp_xy_bin)
            meta_output_h4.append(w_xy)
            meta_output_h5.append(meta_step_out_numeric)

        rshape = (nx * ny, 25, 5)
        kshape = (nx, ny, 25, 5)
        self.model.eval()
        self.model.batch_size = 295596
        for idx in range(len(data_input)):
            dataset = (data_input[idx].reshape(rshape).to(device),
                       meta_input_h1[idx].unsqueeze(0).unsqueeze(0).repeat(nx,ny,1).reshape(nx * ny,len(meta_input_h1[0])).to(device),
                       meta_input_h2[idx].unsqueeze(0).unsqueeze(0).repeat(nx,ny,1).reshape(nx * ny,len(meta_input_h2[0])).to(device),
                       meta_input_h3[idx].reshape(nx * ny,len(meta_input_h3[0][0][0])).to(device),
                       meta_input_h4[idx].reshape(nx * ny,len(meta_input_h4[0][0][0])).to(device),
                       meta_input_h5[idx].unsqueeze(0).unsqueeze(0).repeat(nx,ny).reshape(nx * ny).to(device),
                       meta_output_h1[idx].unsqueeze(0).unsqueeze(0).repeat(nx,ny,1).reshape(nx * ny,len(meta_output_h1[0])).to(device),
                       meta_output_h2[idx].unsqueeze(0).unsqueeze(0).repeat(nx,ny,1).reshape(nx * ny,len(meta_output_h2[0])).to(device),
                       meta_output_h3[idx].reshape(nx * ny,len(meta_input_h3[0][0][0])).to(device),
                       meta_output_h4[idx].reshape(nx * ny,len(meta_output_h4[0][0][0])).to(device),
                       meta_output_h5[idx].unsqueeze(0).unsqueeze(0).repeat(nx,ny).reshape(nx * ny).to(device))

            pred_r, pred_g, pred_b, pred_a = self.model(dataset)
            print(pred_r.shape)
            print('dupa')
            time.sleep(10000)
            loss_r = criterion(pred_r, self.data_output[idx][:, 0].unsqueeze(1))
            loss_g = criterion(pred_g, self.data_output[idx][:, 1].unsqueeze(1))
            loss_b = criterion(pred_b, self.data_output[idx][:, 2].unsqueeze(1))
            loss_alpha = criterion(pred_a, self.data_output[idx][:, 3].unsqueeze(1))

            loss = loss_r + loss_g + loss_b + loss_alpha
            self.saved_loss.append(loss.item())
            print(f'Period: {self.period}/{self.no_of_periods} | Frame: {idx + 1}/{len(data_input)}, Loss: {loss.item():.4f}')
