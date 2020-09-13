import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model_vs import *
from util_vscontroller import ReplayMemory_vs_rep
import numpy as np

device = torch.device('cuda:0')


class vs_rep:
    def __init__(self, trained_model_path=None, num_actions=2, input_channels=3, hidden_size=256, max_memory_size=30000):
        # Params
        self.num_actions = num_actions

        # Networks
        self.perception = Perception(input_channels).to(device)
        self.VS_correlation = VS_correlation(self.perception).to(device)
        self.Forward_Dynamics = Forward_Dynamics(self.VS_correlation).to(device)
        self.Inverse_Dynamics = Inverse_Dynamics(self.VS_correlation).to(device)

        '''
        if trained_model_path != None:
            self.Forward_Dynamics.load_state_dict(torch.load('{}/vs_representation_epoch_200000.pt'.format(trained_model_path)))
        '''
        if trained_model_path != None:
            self.Forward_Dynamics.load_state_dict(torch.load('{}/vs_representation_Forward_Dynamics_epoch_200000.pt'.format(trained_model_path)))
            self.Inverse_Dynamics.load_state_dict(torch.load('{}/vs_representation_Inverse_Dynamics_epoch_200000.pt'.format(trained_model_path)))


        # Training
        self.memory = ReplayMemory_vs_rep(max_memory_size)        
        
        self.forward_criterion = nn.MSELoss()
        self.inverse_criterion = nn.CrossEntropyLoss()

        #self.perception_params_to_learn = self.perception.parameters()
        #print('perception_params: {}'.format(self.perception_params_to_learn))
        self.forward_dynamics_params_to_learn = list(self.Forward_Dynamics.conv.parameters()) + list(self.Forward_Dynamics.bn.parameters())
        self.inverse_dynamics_params_to_learn = self.Inverse_Dynamics.parameters()

        #self.perception_optimizer = optim.Adam(self.perception_params_to_learn, lr=1e-4)
        self.forward_dynamics_optimizer = optim.Adam(self.forward_dynamics_params_to_learn, lr=1e-4)
        self.inverse_dynamics_optimizer = optim.Adam(self.inverse_dynamics_params_to_learn, lr=1e-4)
        
    def update(self, batch_size):
        ## extract sampled data from memory
        left_batch, right_batch, goal_batch, action_batch = self.memory.sample(batch_size)

        np_left_imgs = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
        for i in range(batch_size):
            np_left_imgs[i] = left_batch[i][0]
        tensor_left_imgs = torch.tensor(np_left_imgs, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        np_right_imgs = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
        for i in range(batch_size):
            np_right_imgs[i] = right_batch[i][0]
        tensor_right_imgs = torch.tensor(np_right_imgs, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        np_goal_imgs = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
        for i in range(batch_size):
            np_goal_imgs[i] = goal_batch[i][0]
        tensor_goal_imgs = torch.tensor(np_goal_imgs, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        np_actions = np.zeros((batch_size, 1), dtype=np.float32)
        for i in range(batch_size):
            np_actions[i] = action_batch[i]
        tensor_actions = torch.tensor(np_actions, dtype=torch.float32).to(device)
        gt_a_hat = torch.tensor(np_actions, dtype=torch.long).to(device).squeeze()

        ## compute forward_dynamics loss
        predicted_ft = self.Forward_Dynamics(tensor_left_imgs, tensor_goal_imgs, tensor_actions)
        goal_ft = self.Forward_Dynamics.correlation(tensor_right_imgs, tensor_goal_imgs) #batch_size x 16 x 4 x 4
        
        ## compute inverse_dynamics loss
        output_a_tilde= self.Inverse_Dynamics(tensor_left_imgs, tensor_right_imgs, tensor_goal_imgs).squeeze()
        #print('output_a_tilde.shape = {}'.format(output_a_tilde.shape))
        print('output_a_tilde = {}'.format(output_a_tilde))
        print('gt_a_hat = {}'.format(gt_a_hat))
        #print('gt_a_hat.shape = {}'.format(gt_a_hat.shape))
        self.forward_dynamics_optimizer.zero_grad()
        self.inverse_dynamics_optimizer.zero_grad()
        forward_loss = self.forward_criterion(predicted_ft, goal_ft)
        inverse_loss = self.inverse_criterion(output_a_tilde, gt_a_hat)
        loss = 0.0*forward_loss + inverse_loss
        print('forward_loss = {}, inverse_loss = {}'.format(forward_loss.item(), inverse_loss.item()))
        
        loss.backward()
        '''
        for param in self.perception_params_to_learn:
            param.grad.data.clamp_(-1, 1)
        self.perception_optimizer.step()
        '''
        for param in self.forward_dynamics_params_to_learn:
            param.grad.data.clamp_(-1, 1)
        self.forward_dynamics_optimizer.step()
        for param in self.inverse_dynamics_params_to_learn:
            param.grad.data.clamp_(-1, 1)
        self.inverse_dynamics_optimizer.step()

        #print('loss = {:.2f}'.format(loss))
        return loss.item()
        
class vs_rep_error_state:
    def __init__(self, trained_model_path=None, num_actions=2, input_channels=3, hidden_size=256, max_memory_size=30000):
        # Params
        self.num_actions = num_actions

        # Networks
        self.perception = Perception_multiscale(input_channels).to(device)
        self.VS_correlation = VS_correlation(self.perception).to(device)
        self.Inverse_Dynamics = Inverse_Dynamics_error_state(self.VS_correlation).to(device)

        if trained_model_path != None:
            self.Inverse_Dynamics.load_state_dict(torch.load('{}/vs_representation_Inverse_Dynamics_epoch_200000.pt'.format(trained_model_path)))

        # Training
        self.memory = ReplayMemory_vs_rep(max_memory_size)        

        self.inverse_criterion = nn.CrossEntropyLoss()

        self.inverse_dynamics_params_to_learn = self.Inverse_Dynamics.parameters()

        self.inverse_dynamics_optimizer = optim.Adam(self.inverse_dynamics_params_to_learn, lr=1e-4)
        
    def update(self, batch_size):
        ## extract sampled data from memory
        left_batch, right_batch, goal_batch, action_batch = self.memory.sample(batch_size)

        np_left_imgs = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
        for i in range(batch_size):
            np_left_imgs[i] = left_batch[i][0]
        tensor_left_imgs = torch.tensor(np_left_imgs, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        np_right_imgs = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
        for i in range(batch_size):
            np_right_imgs[i] = right_batch[i][0]
        tensor_right_imgs = torch.tensor(np_right_imgs, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        np_goal_imgs = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
        for i in range(batch_size):
            np_goal_imgs[i] = goal_batch[i][0]
        tensor_goal_imgs = torch.tensor(np_goal_imgs, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        np_actions = np.zeros((batch_size, 1), dtype=np.float32)
        for i in range(batch_size):
            np_actions[i] = action_batch[i]
        tensor_actions = torch.tensor(np_actions, dtype=torch.float32).to(device)
        gt_a_hat = torch.tensor(np_actions, dtype=torch.long).to(device).squeeze()

        ## compute inverse_dynamics loss
        output_a_tilde= self.Inverse_Dynamics(tensor_left_imgs, tensor_right_imgs, tensor_goal_imgs).squeeze()
        #print('output_a_tilde.shape = {}'.format(output_a_tilde.shape))
        print('output_a_tilde = {}'.format(output_a_tilde))
        print('gt_a_hat = {}'.format(gt_a_hat))
        #print('gt_a_hat.shape = {}'.format(gt_a_hat.shape))

        self.inverse_dynamics_optimizer.zero_grad()
        inverse_loss = self.inverse_criterion(output_a_tilde, gt_a_hat)
        loss = inverse_loss
        print('inverse_loss = {}'.format(inverse_loss.item()))
        
        loss.backward()

        for param in self.inverse_dynamics_params_to_learn:
            param.grad.data.clamp_(-1, 1)
        self.inverse_dynamics_optimizer.step()

        #print('loss = {:.2f}'.format(loss))
        return loss.item()