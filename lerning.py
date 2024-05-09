import random

import torch


class teacher(object):
    def __init__(self,model):
        super(teacher, self).__init__()
        self.model = model
        self.fsim = None

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

def learning_loop(self,dataloader,criterion,optimizer,device,num_epochs=100):
        global loss
        best_loss = float('inf')
        best_model_state = None
        num_epochs = num_epochs
        for epoch in range(num_epochs):
            for inputs, targets in dataloader:
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
