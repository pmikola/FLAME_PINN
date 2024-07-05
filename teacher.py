import os.path
import random
import struct
import time

import numpy as np
import torch
from matplotlib import pyplot as plt, animation


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
        self.structure_input = None
        self.meta_input_h1 = None
        self.meta_input_h2 = None
        self.meta_input_h3 = None
        self.meta_input_h4 = None
        self.meta_input_h5 = None
        self.data_output = None
        self.structure_output = None
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
        data_output = []
        structure_output = []
        meta_output_h1 = []
        meta_output_h2 = []
        meta_output_h3 = []
        meta_output_h4 = []
        meta_output_h5 = []
        frame = 0
        while not frame >= self.batch_size:
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
            fuel_subslice_in = fuel_slices[idx_input,slice_x, slice_y]
            r_subslice_in = r_slices[idx_input,slice_x, slice_y]
            g_subslice_in = g_slices[idx_input,slice_x, slice_y]
            b_subslice_in = b_slices[idx_input,slice_x, slice_y]
            alpha_subslice_in = alpha_slices[idx_input,slice_x, slice_y]
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
            fuel_subslice_out = fuel_slices[idx_output,slice_x, slice_y]#.reshape(1)
            r_subslice_out = r_slices[idx_output,slice_x, slice_y]#.reshape(1)
            g_subslice_out = g_slices[idx_output,slice_x, slice_y]#.reshape(1)
            b_subslice_out = b_slices[idx_output,slice_x, slice_y]#.reshape(1)
            alpha_subslice_out = alpha_slices[idx_output,slice_x, slice_y]#.reshape(1)
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

            # NOTE : Entropy additional
            # r_in_entropy = torch.special.entr(r_subslice_in).sum()
            # g_in_entropy = torch.special.entr(g_subslice_in).sum()
            # b_in_entropy = torch.special.entr(b_subslice_in).sum()
            # a_in_entropy = torch.special.entr(alpha_subslice_in).sum()
            # s_in_entropy = torch.special.entr(fuel_subslice_in).sum()
            frame += 1
            if  rzero and gzero and bzero and azero and fzero:
                frame -=1
            # elif r_in_entropy > 1. or g_in_entropy > 1. or b_in_entropy>1. or a_in_entropy > 1. or s_in_entropy >1.:
            #     frame -= 1
            #     if frame > self.batch_size - 2:
            #         pass
            #     else:
            #         for i in range(0,2):
            #             central_points = torch.cat([central_point_x_binary, central_point_y_binary], dim=0)
            #             data_input.append(data_input_subslice)
            #             structure_input.append(fuel_subslice_in)
            #             meta_input_h1.append(meta_input_subslice)
            #             meta_input_h2.append(meta_step_in)
            #             meta_input_h3.append(central_points)
            #             meta_input_h4.append(torch.cat([torch.tensor(window_x), torch.tensor(window_y)]))
            #             meta_input_h5.append(meta_step_in_numeric)
            #             data_output.append(data_output_subslice)
            #             structure_output.append(fuel_subslice_out)
            #             meta_output_h1.append(meta_output_subslice)
            #             meta_output_h2.append(meta_step_out)
            #             meta_output_h3.append(central_points)
            #             meta_output_h4.append(torch.cat([torch.tensor(window_x), torch.tensor(window_y)]))
            #             meta_output_h5.append(meta_step_out_numeric)
            #             frame+=1
            else:
                # Note: Data for the different layers
                central_points = torch.cat([central_point_x_binary, central_point_y_binary], dim=0)
                data_input.append(data_input_subslice)
                structure_input.append(fuel_subslice_in)
                meta_input_h1.append(meta_input_subslice)
                meta_input_h2.append(meta_step_in)
                meta_input_h3.append(central_points)
                meta_input_h4.append(torch.cat([torch.tensor(window_x),torch.tensor(window_y)]))
                meta_input_h5.append(meta_step_in_numeric)
                data_output.append(data_output_subslice)
                structure_output.append(fuel_subslice_out)
                meta_output_h1.append(meta_output_subslice)
                meta_output_h2.append(meta_step_out)
                meta_output_h3.append(central_points)
                meta_output_h4.append(torch.cat([torch.tensor(window_x), torch.tensor(window_y)]))
                meta_output_h5.append(meta_step_out_numeric)


        self.data_input = torch.stack(data_input,dim=0)
        self.structure_input = torch.stack(structure_input,dim=0)
        self.meta_input_h1 = torch.stack(meta_input_h1,dim=0)
        self.meta_input_h2 = torch.stack(meta_input_h2,dim=0)
        self.meta_input_h3 = torch.stack(meta_input_h3,dim=0)
        self.meta_input_h4 = torch.stack(meta_input_h4,dim=0)
        self.meta_input_h5 = torch.stack(meta_input_h5,dim=0)

        self.data_output = torch.stack(data_output,dim=0)
        self.structure_output = torch.stack(structure_output, dim=0)
        self.meta_output_h1 = torch.stack(meta_output_h1,dim=0)
        self.meta_output_h2 = torch.stack(meta_output_h2,dim=0)
        self.meta_output_h3 = torch.stack(meta_output_h3,dim=0)
        self.meta_output_h4 = torch.stack(meta_output_h4,dim=0)
        self.meta_output_h5 = torch.stack(meta_output_h5,dim=0)

        # print(self.data_input.shape, self.structure_input.shape, self.meta_input_h1.shape, self.meta_input_h2.shape,
        #       self.meta_input_h3.shape, self.meta_input_h4.shape, self.meta_input_h5.shape, self.meta_output_h1.shape,
        #       self.meta_output_h2.shape, self.meta_output_h3.shape, self.meta_output_h4.shape, self.meta_output_h5.shape)


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
                t = 0.
                grad_counter = 0
                noise_amplitude = 1.
                print_every_nth_frame=10
                for epoch in range(num_epochs):
                    self.data_preparation()
                    (self.data_input,self.structure_input,self.meta_input_h1,self.meta_input_h2,
                     self.meta_input_h3,self.meta_input_h4,self.meta_input_h5,self.meta_output_h1,
                     self.meta_output_h2,self.meta_output_h3,self.meta_output_h4,self.meta_output_h5) =\
                                                                (self.data_input.to(device),
                                                                self.structure_input.to(device),
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


                    self.data_output = self.data_output.to(device) #+ (torch.randn_like(self.data_output)*2-0.5)*noise_amplitude
                    self.data_input = self.data_input + (torch.randn_like(self.data_input)*2-0.5)*noise_amplitude
                    self.structure_input = self.structure_input + (torch.randn_like(self.structure_input)*2-0.5)*noise_amplitude
                    dataset = (self.data_input,self.structure_input,self.meta_input_h1,self.meta_input_h2,
                               self.meta_input_h3,self.meta_input_h4,self.meta_input_h5,self.meta_output_h1,
                               self.meta_output_h2,self.meta_output_h3,self.meta_output_h4,self.meta_output_h5)
                    t_start = time.perf_counter()
                    pred_r,pred_g,pred_b,pred_a,pred_s = self.model(dataset)
                    t_pred = time.perf_counter()

                    diff_r_true = self.data_output[:,0:self.model.in_scale,:]-self.data_input[:,0:self.model.in_scale,:]
                    diff_r_pred = pred_r-self.data_input[:,0:self.model.in_scale,:]
                    loss_diff_r = criterion(diff_r_pred,diff_r_true)
                    diff_g_true =  self.data_output[:, self.model.in_scale:self.model.in_scale*2, :]-self.data_input[:, self.model.in_scale:self.model.in_scale*2, :]
                    diff_g_pred = pred_g-self.data_input[:, self.model.in_scale:self.model.in_scale*2, :]
                    loss_diff_g = criterion(diff_g_pred, diff_g_true)
                    diff_b_true = self.data_output[:, self.model.in_scale*2:self.model.in_scale*3, :]-self.data_input[:, self.model.in_scale*2:self.model.in_scale*3, :]
                    diff_b_pred = pred_b-self.data_input[:, self.model.in_scale*2:self.model.in_scale*3, :]
                    loss_diff_b = criterion(diff_b_pred, diff_b_true)
                    diff_a_true = self.data_output[:, self.model.in_scale*3:self.model.in_scale*4, :]-self.data_input[:, self.model.in_scale*3:self.model.in_scale*4, :]
                    diff_a_pred = pred_a-self.data_input[:, self.model.in_scale*3:self.model.in_scale*4, :]
                    loss_diff_a = criterion(diff_a_pred, diff_a_true)
                    diff_s_true = self.structure_output-self.structure_input
                    diff_s_pred = pred_s-self.structure_input
                    loss_diff_s = criterion(diff_s_pred, diff_s_true)
                    diff_loss = loss_diff_r+loss_diff_g+loss_diff_b+loss_diff_a+loss_diff_s

                    loss_r = criterion(pred_r, self.data_output[:,0:self.model.in_scale,:])
                    loss_g = criterion(pred_g, self.data_output[:,self.model.in_scale:self.model.in_scale*2,:])
                    loss_b = criterion(pred_b, self.data_output[:,self.model.in_scale*2:self.model.in_scale*3,:])
                    loss_alpha = criterion(pred_a, self.data_output[:,self.model.in_scale*3:self.model.in_scale*4,:])
                    loss_s = criterion(pred_s, self.structure_output)
                    value_loss = loss_r + loss_g + loss_b + loss_alpha+loss_s

                    # r_entropy_loss = criterion(torch.special.entr(torch.nn.functional.relu(pred_r)),torch.special.entr(self.data_output[:,0:self.model.in_scale,:]))
                    # g_entropy_loss = criterion(torch.special.entr(torch.nn.functional.relu(pred_g)),torch.special.entr(self.data_output[:,self.model.in_scale:self.model.in_scale*2,:]))
                    # b_entropy_loss = criterion(torch.special.entr(torch.nn.functional.relu(pred_b)),torch.special.entr(self.data_output[:,self.model.in_scale*2:self.model.in_scale*3,:]))
                    # a_entropy_loss = criterion(torch.special.entr(torch.nn.functional.relu(pred_a)),torch.special.entr(self.data_output[:,self.model.in_scale*3:self.model.in_scale*4,:]))
                    # s_entropy_loss = criterion(torch.special.entr(torch.nn.functional.relu(pred_s)),torch.special.entr(self.structure_output))
                    # entropy_loss = r_entropy_loss + g_entropy_loss+b_entropy_loss+a_entropy_loss+s_entropy_loss

                    loss = value_loss+diff_loss#+entropy_loss # TODO : diversity loss + intermidiete velocity vectors loss + casual loss

                    self.saved_loss.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # t_stop = time.perf_counter()
                    t += t_pred - t_start

                    if len(self.saved_loss) > 100:
                        gloss = np.array(self.saved_loss)[-25:-1]
                        gloss = abs(np.sum(np.gradient(gloss)))
                        if gloss > 0.1: grad_counter +=1
                        else: grad_counter = 0
                        if grad_counter == 10:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = param_group['lr']*0.99
                                if param_group['lr'] < 0.5e-5:
                                    param_group['lr'] = 1e-4
                                    print('lr back to starting point')
                                noise_amplitude = noise_amplitude*0.5
                                # if noise_amplitude < 1e-3:
                                #     # print('noise amplitude')
                                #     noise_amplitude = 0.
                            grad_counter = 0
                    if (epoch+1) % print_every_nth_frame == 0:
                        print(f'Period: {self.period}/{self.no_of_periods} | Epoch: {epoch+1}/{num_epochs}, '
                              f'Loss: {loss.item():.4f}, '
                              f'Avg. Time per frame: {((self.fsim.grid_size_x*self.fsim.grid_size_y)/(self.model.in_scale**2))*(t*1e6/print_every_nth_frame/self.batch_size):.4f} [us]')
                        t = 0.
                    if (epoch + 1) % 10 == 0:
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            torch.save(self.model.state_dict(), 'model.pt')
                # if best_model_state is None:
                #     pass
                # else:
                #     self.model.load_state_dict(torch.load('model.pt'))
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
                # print(int(central_points_x_neg[j]),int(central_points_x_pos[j]))
                # print(int(central_points_y_neg[j]), int(central_points_y_pos[j]))
                # time.sleep(0.3)
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
            structure_input = fuel_subslice_in
            meta_input_h1 = meta_input_subslice.unsqueeze(0).repeat(data_input.shape[0],1)
            meta_input_h2 = meta_step_in.unsqueeze(0).repeat(data_input.shape[0],1)
            meta_input_h3 = torch.tensor(np.array(central_points_xy_binary))
            meta_input_h4 = torch.cat([x_idx[:,0:-1], y_idx[:,0:-1]],dim=1)

            meta_input_h5 = meta_step_in_numeric.repeat(data_input.shape[0],1).squeeze(1)
            data_output = data_output_subslice
            meta_output_h1 = meta_output_subslice.unsqueeze(0).repeat(data_input.shape[0],1)
            meta_output_h2 = meta_step_out.unsqueeze(0).repeat(data_input.shape[0],1)
            meta_output_h3 = torch.tensor(np.array(central_points_xy_binary))
            meta_output_h4 = torch.cat([x_idx[:,0:-1], y_idx[:,0:-1]],dim=1)
            meta_output_h5 = meta_step_out_numeric.repeat(data_input.shape[0],1).squeeze(1)

            self.model.batch_size = data_input.shape[0]
            self.model.eval()

            (data_input, structure_input, meta_input_h1, meta_input_h2,
             meta_input_h3, meta_input_h4, meta_input_h5, meta_output_h1,
             meta_output_h2, meta_output_h3, meta_output_h4, meta_output_h5) = \
                (data_input.to(device),
                 structure_input.to(device),
                 meta_input_h1.to(device),
                 meta_input_h2.to(device),
                 meta_input_h3.to(device),
                 meta_input_h4.to(device),
                 meta_input_h5.to(device),
                 meta_output_h1.to(device),
                 meta_output_h2.to(device),
                 meta_output_h3.to(device),
                 meta_output_h4.to(device),
                 meta_output_h5.to(device))

            # data_output = data_output.to(device)
            dataset = (data_input, structure_input, meta_input_h1, meta_input_h2,
                       meta_input_h3, meta_input_h4, meta_input_h5, meta_output_h1,
                       meta_output_h2, meta_output_h3, meta_output_h4, meta_output_h5)
            # print(data_input.shape, structure_input.shape, meta_input_h1.shape, meta_input_h2.shape,
            #            meta_input_h3.shape, meta_input_h4.shape, meta_input_h5.shape, meta_output_h1.shape,
            #            meta_output_h2.shape, meta_output_h3.shape, meta_output_h4.shape, meta_output_h5.shape)
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
            # ims.append([rgb_true_anim,title_pred,title_true,title_rms])

            # grad_r_true = data_input[:, 0:self.model.in_scale, :] - data_output[:, 0:self.model.in_scale, :]
            # grad_r_pred = data_input[:, 0:self.model.in_scale, :] - pred_r
            # loss_grad_r = criterion(grad_r_pred, grad_r_true)
            # grad_g_true = data_input[:, self.model.in_scale:self.model.in_scale * 2, :] - data_output[:,self.model.in_scale:self.model.in_scale * 2,:]
            # grad_g_pred = data_input[:, self.model.in_scale:self.model.in_scale * 2, :] - pred_g
            # loss_grad_g = criterion(grad_g_pred, grad_g_true)
            # grad_b_true = data_input[:, self.model.in_scale * 2:self.model.in_scale * 3, :] - data_output[:,
            #                                                                                        self.model.in_scale * 2:self.model.in_scale * 3,
            #                                                                                        :]
            # grad_b_pred = data_input[:, self.model.in_scale * 2:self.model.in_scale * 3, :] - pred_b
            # loss_grad_b = criterion(grad_b_pred, grad_b_true)
            # grad_a_true = data_input[:, self.model.in_scale * 3:self.model.in_scale * 4, :] - data_output[:,self.model.in_scale * 3:self.model.in_scale * 4,:]
            # grad_a_pred = data_input[:, self.model.in_scale * 3:self.model.in_scale * 4, :] - pred_a
            # loss_grad_a = criterion(grad_a_pred, grad_a_true)
            # grad_loss = loss_grad_r + loss_grad_g + loss_grad_b + loss_grad_a
            #
            # loss_r = criterion(pred_r, data_output[:, 0:self.model.in_scale, :])
            # loss_g = criterion(pred_g, data_output[:, self.model.in_scale:self.model.in_scale * 2, :])
            # loss_b = criterion(pred_b, data_output[:, self.model.in_scale * 2:self.model.in_scale * 3, :])
            # loss_alpha = criterion(pred_a, data_output[:, self.model.in_scale * 3:self.model.in_scale * 4, :])
            # value_loss = loss_r + loss_g + loss_b + loss_alpha
            #
            # loss = value_loss + grad_loss  # TODO : Add Endropy loss + diversity loss + intermidiete velocity vectors loss + casual loss
            # self.saved_loss.append(loss.item())
        ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=100)
        # ani.save("flame_animation_ground_t_vs_preds.gif")
        fig.colorbar(rms_anim, ax=ax3)
        plt.show()






