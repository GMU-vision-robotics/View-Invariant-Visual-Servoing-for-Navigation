import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model_vs import *
from util_vscontroller import ReplayMemory_vs_dqn, ReplayMemory_overlap_dqn, ReplayMemory_overlap_dqn_recurrent
import random
import numpy as np
import math

device = torch.device('cuda:0')

class DQN_vs:
    def __init__(self, trained_model_path=None, num_actions=2, input_channels=3, actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=0.99, tau=1e-2, max_memory_size=30000):
        # Params
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.steps_done = 0
        self.input_channels = input_channels

        # Networks
        '''
        self.perception = Perception(input_channels).to(device)
        self.VS_correlation = VS_correlation(self.perception).to(device)
        self.VS_correlation.load_state_dict(torch.load('/home/reza/Datasets/GibsonEnv/my_code/vs_controller/trained_vs_representation/vs_representation_correlation_sixth_try.pt'))
        '''
        self.perception = Perception_multiscale(input_channels).to(device)
        self.VS_correlation = VS_correlation(self.perception).to(device)

        self.actor = DQN_VS_Controller(self.VS_correlation, self.num_actions, input_size=256).to(device)
        self.critic = DQN_VS_Controller(self.VS_correlation, self.num_actions, input_size=256).to(device)

        if trained_model_path != None:
            self.actor.load_state_dict(torch.load('{}/dqn_epoch_200000.pt'.format(trained_model_path)))
            print('*********************************successfully read the model ...')

        ## copy params from actor.parameters to critic.parameters.
        ## when calling the copy_(),  the argument param.data is the src.
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()

        # Training
        self.memory = ReplayMemory_vs_dqn(max_memory_size)        
        ## only update weights of actor's linear layer
        '''
        for param in self.actor.correlation.parameters():
            param.requires_grad = False
        self.actor_optimizer = optim.RMSprop(self.actor.linear.parameters(), lr=actor_learning_rate)
        '''
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=actor_learning_rate)

    def update_critic(self):
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()
    
    ## for collecting (state, action, next_state) tuples
    def select_action(self, left_img, goal_img, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=10000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        ## take action with the maximum reward by using the actor
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                tensor_left = torch.tensor(left_img, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2)
                tensor_goal = torch.tensor(goal_img, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2)
                return self.actor(tensor_left, tensor_goal).max(1)[1].view(1, 1)
        else:
            ## take random actions, do exploration
            return torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)
    
    def update(self, batch_size):
        left_imgs, goal_imgs, actions, rewards, next_left_imgs, next_goal_imgs, done = self.memory.sample(batch_size)
        
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(1)
        
        np_left_imgs = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
        for i in range(batch_size):
            np_left_imgs[i] = left_imgs[i][0]
        tensor_left_imgs = torch.tensor(np_left_imgs, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        np_goal_imgs = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
        for i in range(batch_size):
            np_goal_imgs[i] = goal_imgs[i][0]
        tensor_goal_imgs = torch.tensor(np_goal_imgs, dtype=torch.float32).to(device).permute(0, 3, 1, 2)

        np_next_left_imgs = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
        for i in range(batch_size):
            np_next_left_imgs[i] = next_left_imgs[i][0]
        tensor_next_left_imgs = torch.tensor(np_next_left_imgs, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        np_next_goal_imgs = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
        for i in range(batch_size):
            np_next_goal_imgs[i] = next_goal_imgs[i][0]
        tensor_next_goal_imgs = torch.tensor(np_next_goal_imgs, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
    
        # Critic loss (value function loss)  
        ## Get predicted next-state actions and Q values from target models
        ## gather() accumulates values at given index
        Qvals = self.actor.forward(tensor_left_imgs, tensor_goal_imgs).gather(1, actions.unsqueeze(1))
        next_Q = self.critic.forward(tensor_next_left_imgs, tensor_next_goal_imgs).max(1)[0].detach().unsqueeze(1)
        # Compute Q targets for current states (y_i)
        Qprime = rewards + self.gamma * next_Q * (1-done)
        loss = F.smooth_l1_loss(Qvals, Qprime)
        #print('loss = {}'.format(loss))
        
        # update networks
        self.actor_optimizer.zero_grad()
        loss.backward()
        '''
        for param in self.actor.linear.parameters():
            param.grad.data.clamp_(-1, 1)
        '''
        self.actor_optimizer.step()

class DQN_vs_overlap:
    def __init__(self, trained_model_path=None, num_actions=2, input_channels=2, actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=0.97, tau=1e-2, max_memory_size=50000):
        # Params
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.steps_done = 0
        self.input_channels = input_channels

        # Networks
        self.perception_actor = Perception_overlap(input_channels).to(device)
        self.actor = DQN_OVERLAP_Controller(self.perception_actor, self.num_actions, input_size=256).to(device)
        self.perception_critic = Perception_overlap(input_channels).to(device)
        self.critic = DQN_OVERLAP_Controller(self.perception_critic, self.num_actions, input_size=256).to(device)

        if trained_model_path != None:
            self.actor.load_state_dict(torch.load('{}/dqn_epoch_200000.pt'.format(trained_model_path)))
            print('*********************************successfully read the model ...')

        ## copy params from actor.parameters to critic.parameters.
        ## when calling the copy_(),  the argument param.data is the src.
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()

        # Training
        self.memory = ReplayMemory_overlap_dqn(max_memory_size)        
        ## only update weights of actor's linear layer
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=actor_learning_rate)

    def update_critic(self):
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()
    
    ## for collecting (state, action, next_state) tuples
    def select_action(self, state, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=10000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        ## take action with the maximum reward by using the actor
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = state[0] ## 256 x 256 x 1
                print('obs.shape = {}'.format(obs.shape))
                #print('obs.shape = {}'.format(obs.shape))
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2)
                action = self.actor(obs).max(1)[1].view(1, 1)
                return action
        else:
            ## take random actions, do exploration
            return torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)
    
    def update(self, batch_size):
        states, actions, rewards, next_states, done = self.memory.sample(batch_size)
        
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(1)
        #print('actions.shape = {}'.format(actions.shape))
        #print('rewards.shape = {}'.format(rewards.shape))
        #print('done.shape = {}'.format(done.shape))
        
        tensor_states = np.zeros((batch_size, 256, 256, self.input_channels), dtype=np.float32)
        for i in range(batch_size):
            tensor_states[i] = states[i][0]
        states = torch.tensor(tensor_states, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        tensor_next_states = np.zeros((batch_size, 256, 256, self.input_channels), dtype=np.float32)
        for i in range(batch_size):
            tensor_next_states[i] = next_states[i][0]
        next_states = torch.tensor(tensor_next_states, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        # Critic loss (value function loss)  
        ## Get predicted next-state actions and Q values from target models
        ## gather() accumulates values at given index
        Qvals = self.actor.forward(states).gather(1, actions.unsqueeze(1))
        #print('Qvals.shape = {}'.format(Qvals.shape))
        next_Q = self.critic.forward(next_states).max(1)[0].detach().unsqueeze(1)
        #print('next_Q.shape = {}'.format(next_Q.shape))
        # Compute Q targets for current states (y_i)
        Qprime = rewards + self.gamma * next_Q * (1-done)
        #print('Qprime.shape = {}'.format(Qprime.shape))
        loss = F.smooth_l1_loss(Qvals, Qprime)
        #print('loss = {}'.format(loss))
        
        # update networks
        self.actor_optimizer.zero_grad()
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

class DQN_vs_siamese:
    def __init__(self, trained_model_path=None, num_actions=2, input_channels=2, actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=0.97, tau=1e-2, max_memory_size=50000):
        # Params
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.steps_done = 0
        self.input_channels = input_channels

        # Networks
        self.perception_actor = Perception_siamese(input_channels).to(device)
        self.actor = DQN_OVERLAP_Controller(self.perception_actor, self.num_actions, input_size=512).to(device)
        self.perception_critic = Perception_siamese(input_channels).to(device)
        self.critic = DQN_OVERLAP_Controller(self.perception_critic, self.num_actions, input_size=512).to(device)

        if trained_model_path != None:
            self.actor.load_state_dict(torch.load('{}/dqn_epoch_200000.pt'.format(trained_model_path)))
            print('*********************************successfully read the model ...')

        ## copy params from actor.parameters to critic.parameters.
        ## when calling the copy_(),  the argument param.data is the src.
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()

        # Training
        self.memory = ReplayMemory_overlap_dqn(max_memory_size)        
        ## only update weights of actor's linear layer
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=actor_learning_rate)

    def update_critic(self):
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()
    
    ## for collecting (state, action, next_state) tuples
    def select_action(self, state, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=10000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        ## take action with the maximum reward by using the actor
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = state[0] ## 256 x 256 x 1
                print('obs.shape = {}'.format(obs.shape))
                #print('obs.shape = {}'.format(obs.shape))
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2)
                return self.actor(obs).max(1)[1].view(1, 1)
        else:
            ## take random actions, do exploration
            return torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)
    
    def update(self, batch_size):
        states, actions, rewards, next_states, done = self.memory.sample(batch_size)
        
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(1)
        
        tensor_states = np.zeros((batch_size, 256, 256, self.input_channels), dtype=np.float32)
        for i in range(batch_size):
            tensor_states[i] = states[i][0]
        states = torch.tensor(tensor_states, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        tensor_next_states = np.zeros((batch_size, 256, 256, self.input_channels), dtype=np.float32)
        for i in range(batch_size):
            tensor_next_states[i] = next_states[i][0]
        next_states = torch.tensor(tensor_next_states, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        # Critic loss (value function loss)  
        ## Get predicted next-state actions and Q values from target models
        ## gather() accumulates values at given index
        Qvals = self.actor.forward(states).gather(1, actions.unsqueeze(1))
        next_Q = self.critic.forward(next_states).max(1)[0].detach().unsqueeze(1)
        # Compute Q targets for current states (y_i)
        Qprime = rewards + self.gamma * next_Q * (1-done)
        #print('Qprime = {}'.format(Qprime))
        loss = F.smooth_l1_loss(Qvals, Qprime)
        #print('loss = {}'.format(loss))
        
        # update networks
        self.actor_optimizer.zero_grad()
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

class DQN_vs_triplet:
    def __init__(self, trained_model_path=None, num_actions=2, input_channels=2, actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=0.97, tau=1e-2, max_memory_size=50000):
        # Params
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.steps_done = 0
        self.input_channels = input_channels

        # Networks
        self.perception_actor = Perception_triplet(input_channels).to(device)
        self.actor = DQN_OVERLAP_Controller(self.perception_actor, self.num_actions, input_size=768).to(device)
        self.perception_critic = Perception_triplet(input_channels).to(device)
        self.critic = DQN_OVERLAP_Controller(self.perception_critic, self.num_actions, input_size=768).to(device)

        if trained_model_path != None:
            self.actor.load_state_dict(torch.load('{}/dqn_epoch_200000.pt'.format(trained_model_path)))
            print('*********************************successfully read the model ...')

        ## copy params from actor.parameters to critic.parameters.
        ## when calling the copy_(),  the argument param.data is the src.
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()

        # Training
        self.memory = ReplayMemory_overlap_dqn(max_memory_size)        
        ## only update weights of actor's linear layer
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=actor_learning_rate)

    def update_critic(self):
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()
    
    ## for collecting (state, action, next_state) tuples
    def select_action(self, state, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=10000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        ## take action with the maximum reward by using the actor
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = state[0] ## 256 x 256 x 1
                print('obs.shape = {}'.format(obs.shape))
                #print('obs.shape = {}'.format(obs.shape))
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2)
                return self.actor(obs).max(1)[1].view(1, 1)
        else:
            ## take random actions, do exploration
            return torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)
    
    def update(self, batch_size):
        states, actions, rewards, next_states, done = self.memory.sample(batch_size)
        
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(1)
        
        tensor_states = np.zeros((batch_size, 256, 256, self.input_channels), dtype=np.float32)
        for i in range(batch_size):
            tensor_states[i] = states[i][0]
        states = torch.tensor(tensor_states, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        tensor_next_states = np.zeros((batch_size, 256, 256, self.input_channels), dtype=np.float32)
        for i in range(batch_size):
            tensor_next_states[i] = next_states[i][0]
        next_states = torch.tensor(tensor_next_states, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        # Critic loss (value function loss)  
        ## Get predicted next-state actions and Q values from target models
        ## gather() accumulates values at given index
        Qvals = self.actor.forward(states).gather(1, actions.unsqueeze(1))
        next_Q = self.critic.forward(next_states).max(1)[0].detach().unsqueeze(1)
        # Compute Q targets for current states (y_i)
        Qprime = rewards + self.gamma * next_Q * (1-done)
        #print('Qprime = {}'.format(Qprime))
        loss = F.smooth_l1_loss(Qvals, Qprime)
        #print('loss = {}'.format(loss))
        
        # update networks
        self.actor_optimizer.zero_grad()
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

class DQN_vs_siamese:
    def __init__(self, trained_model_path=None, num_actions=2, input_channels=2, actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=0.97, tau=1e-2, max_memory_size=50000):
        # Params
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.steps_done = 0
        self.input_channels = input_channels

        # Networks
        self.perception_actor = Perception_siamese(input_channels).to(device)
        self.actor = DQN_OVERLAP_Controller(self.perception_actor, self.num_actions, input_size=512).to(device)
        self.perception_critic = Perception_siamese(input_channels).to(device)
        self.critic = DQN_OVERLAP_Controller(self.perception_critic, self.num_actions, input_size=512).to(device)

        if trained_model_path != None:
            self.actor.load_state_dict(torch.load('{}/dqn_epoch_200000.pt'.format(trained_model_path)))
            print('*********************************successfully read the model ...')

        ## copy params from actor.parameters to critic.parameters.
        ## when calling the copy_(),  the argument param.data is the src.
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()

        # Training
        self.memory = ReplayMemory_overlap_dqn(max_memory_size)        
        ## only update weights of actor's linear layer
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=actor_learning_rate)

    def update_critic(self):
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()
    
    ## for collecting (state, action, next_state) tuples
    def select_action(self, state, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=10000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        ## take action with the maximum reward by using the actor
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = state[0] ## 256 x 256 x 1
                print('obs.shape = {}'.format(obs.shape))
                #print('obs.shape = {}'.format(obs.shape))
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2)
                return self.actor(obs).max(1)[1].view(1, 1)
        else:
            ## take random actions, do exploration
            return torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)
    
    def update(self, batch_size):
        states, actions, rewards, next_states, done = self.memory.sample(batch_size)
        
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(1)
        
        tensor_states = np.zeros((batch_size, 256, 256, self.input_channels), dtype=np.float32)
        for i in range(batch_size):
            tensor_states[i] = states[i][0]
        states = torch.tensor(tensor_states, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        tensor_next_states = np.zeros((batch_size, 256, 256, self.input_channels), dtype=np.float32)
        for i in range(batch_size):
            tensor_next_states[i] = next_states[i][0]
        next_states = torch.tensor(tensor_next_states, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        # Critic loss (value function loss)  
        ## Get predicted next-state actions and Q values from target models
        ## gather() accumulates values at given index
        Qvals = self.actor.forward(states).gather(1, actions.unsqueeze(1))
        next_Q = self.critic.forward(next_states).max(1)[0].detach().unsqueeze(1)
        # Compute Q targets for current states (y_i)
        Qprime = rewards + self.gamma * next_Q * (1-done)
        #print('Qprime = {}'.format(Qprime))
        loss = F.smooth_l1_loss(Qvals, Qprime)
        #print('loss = {}'.format(loss))
        
        # update networks
        self.actor_optimizer.zero_grad()
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

class DQN_vs_siamese_fusion:
    def __init__(self, trained_model_path=None, num_actions=2, input_channels=2, actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=0.97, tau=1e-2, max_memory_size=50000):
        # Params
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.steps_done = 0
        self.input_channels = input_channels

        # Networks
        self.perception_actor = Perception_siamese_fusion_new(input_channels).to(device)
        self.actor = DQN_OVERLAP_Controller(self.perception_actor, self.num_actions, input_size=256).to(device)
        self.perception_critic = Perception_siamese_fusion_new(input_channels).to(device)
        self.critic = DQN_OVERLAP_Controller(self.perception_critic, self.num_actions, input_size=256).to(device)

        if trained_model_path != None:
            self.actor.load_state_dict(torch.load('{}/dqn_epoch_200000.pt'.format(trained_model_path)))
            print('*********************************successfully read the model ...')

        ## copy params from actor.parameters to critic.parameters.
        ## when calling the copy_(),  the argument param.data is the src.
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()

        # Training
        self.memory = ReplayMemory_overlap_dqn(max_memory_size)        
        ## only update weights of actor's linear layer
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=actor_learning_rate)

    def update_critic(self):
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()
    
    ## for collecting (state, action, next_state) tuples
    def select_action(self, state, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=10000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        ## take action with the maximum reward by using the actor
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = state[0] ## 256 x 256 x 1
                print('obs.shape = {}'.format(obs.shape))
                #print('obs.shape = {}'.format(obs.shape))
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2)
                return self.actor(obs).max(1)[1].view(1, 1)
        else:
            ## take random actions, do exploration
            return torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)
    
    def update(self, batch_size):
        states, actions, rewards, next_states, done = self.memory.sample(batch_size)
        
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(1)
        
        tensor_states = np.zeros((batch_size, 256, 256, self.input_channels), dtype=np.float32)
        for i in range(batch_size):
            tensor_states[i] = states[i][0]
        states = torch.tensor(tensor_states, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        tensor_next_states = np.zeros((batch_size, 256, 256, self.input_channels), dtype=np.float32)
        for i in range(batch_size):
            tensor_next_states[i] = next_states[i][0]
        next_states = torch.tensor(tensor_next_states, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        # Critic loss (value function loss)  
        ## Get predicted next-state actions and Q values from target models
        ## gather() accumulates values at given index
        Qvals = self.actor.forward(states).gather(1, actions.unsqueeze(1))
        next_Q = self.critic.forward(next_states).max(1)[0].detach().unsqueeze(1)
        # Compute Q targets for current states (y_i)
        Qprime = rewards + self.gamma * next_Q * (1-done)
        #print('Qprime = {}'.format(Qprime))
        loss = F.smooth_l1_loss(Qvals, Qprime)
        #print('loss = {}'.format(loss))
        
        # update networks
        self.actor_optimizer.zero_grad()
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

class DQN_vs_overlap_resnet:
    def __init__(self, trained_model_path=None, num_actions=2, input_channels=2, actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=0.97, tau=1e-2, max_memory_size=50000):
        # Params
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.steps_done = 0
        self.input_channels = input_channels

        # Networks
        self.perception_actor = Preception_overlap_resnet(input_channels).to(device)
        self.actor = DQN_OVERLAP_RESNET_Controller(self.perception_actor, self.num_actions, input_size=512).to(device)
        self.perception_critic = Preception_overlap_resnet(input_channels).to(device)
        self.critic = DQN_OVERLAP_RESNET_Controller(self.perception_critic, self.num_actions, input_size=512).to(device)

        if trained_model_path != None:
            self.actor.load_state_dict(torch.load('{}/dqn_epoch_200000.pt'.format(trained_model_path)))
            print('*********************************successfully read the model ...')

        ## copy params from actor.parameters to critic.parameters.
        ## when calling the copy_(),  the argument param.data is the src.
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()

        # Training
        self.memory = ReplayMemory_overlap_dqn(max_memory_size)        
        ## only update weights of actor's linear layer
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=actor_learning_rate)

    def update_critic(self):
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()
    
    ## for collecting (state, action, next_state) tuples
    def select_action(self, state, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=10000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        ## take action with the maximum reward by using the actor
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = state[0] ## 256 x 256 x 1
                print('obs.shape = {}'.format(obs.shape))
                #print('obs.shape = {}'.format(obs.shape))
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2)
                return self.actor(obs).max(1)[1].view(1, 1)
        else:
            ## take random actions, do exploration
            return torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)
    
    def update(self, batch_size):
        states, actions, rewards, next_states, done = self.memory.sample(batch_size)
        
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(1)
        
        tensor_states = np.zeros((batch_size, 256, 256, self.input_channels), dtype=np.float32)
        for i in range(batch_size):
            tensor_states[i] = states[i][0]
        states = torch.tensor(tensor_states, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        tensor_next_states = np.zeros((batch_size, 256, 256, self.input_channels), dtype=np.float32)
        for i in range(batch_size):
            tensor_next_states[i] = next_states[i][0]
        next_states = torch.tensor(tensor_next_states, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        # Critic loss (value function loss)  
        ## Get predicted next-state actions and Q values from target models
        ## gather() accumulates values at given index
        Qvals = self.actor.forward(states).gather(1, actions.unsqueeze(1))
        next_Q = self.critic.forward(next_states).max(1)[0].detach().unsqueeze(1)
        # Compute Q targets for current states (y_i)
        Qprime = rewards + self.gamma * next_Q * (1-done)
        #print('Qprime = {}'.format(Qprime))
        loss = F.smooth_l1_loss(Qvals, Qprime)
        #print('loss = {}'.format(loss))
        
        # update networks
        self.actor_optimizer.zero_grad()
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

class DQN_vs_overlap_rnn:
    def __init__(self, trained_model_path=None, num_actions=2, input_channels=2, actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=0.97, tau=1e-2, max_memory_size=2000):
        # Params
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.steps_done = 0
        self.input_channels = input_channels

        # Networks
        self.perception_actor = Perception_overlap_recurrent(input_channels).to(device)
        self.actor = DQN_OVERLAP_Recurrent_Controller(self.perception_actor, self.num_actions, input_size=256).to(device)
        self.perception_critic = Perception_overlap_recurrent(input_channels).to(device)
        self.critic = DQN_OVERLAP_Recurrent_Controller(self.perception_critic, self.num_actions, input_size=256).to(device)

        if trained_model_path != None:
            self.actor.load_state_dict(torch.load('{}/dqn_epoch_200000.pt'.format(trained_model_path)))
            print('*********************************successfully read the model ...')

        ## copy params from actor.parameters to critic.parameters.
        ## when calling the copy_(),  the argument param.data is the src.
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()

        # Training
        self.memory = ReplayMemory_overlap_dqn_recurrent(max_memory_size)        
        ## only update weights of actor's linear layer
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=actor_learning_rate)

    def update_critic(self):
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()
    
    ## for collecting (state, action, next_state) tuples
    #def select_action(self, state, hidden_state, cell_state, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=10000):
    def select_action(self, state, hidden_state, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=10000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        ## take action with the maximum reward by using the actor
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = state#[0] ## 256 x 256 x 1
                print('obs.shape = {}'.format(obs.shape))
                #print('obs.shape = {}'.format(obs.shape))
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
                #print('obs.shape = {}'.format(obs.shape))
                #model_out = self.actor.forward(obs, hidden_state, cell_state, batch_size=1, time_step=1)
                model_out = self.actor.forward(obs, hidden_state, batch_size=1, time_step=1)
                action = model_out[0].max(1)[1].view(1, 1)
                #hidden_state = model_out[1][0]
                #cell_state = model_out[1][1]
                hidden_state = model_out[1]
                #return action, hidden_state, cell_state
                return action, hidden_state
        else:
            ## take random actions, do exploration
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = state#[0] ## 256 x 256 x 1
                #print('obs.shape = {}'.format(obs.shape))
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
                #print('obs.shape = {}'.format(obs.shape))
                #model_out = self.actor.forward(obs, hidden_state, cell_state, batch_size=1, time_step=1)
                model_out = self.actor.forward(obs, hidden_state, batch_size=1, time_step=1)
                action = torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)
                #hidden_state = model_out[1][0]
                #cell_state = model_out[1][1]
                hidden_state = model_out[1]
                #return action, hidden_state, cell_state
                return action, hidden_state
    
    def update(self, batch_size, time_step=5):
        print('batch_size = {}'.format(batch_size))
        #hidden_batch, cell_batch = self.actor.init_hidden_states(batch_size=batch_size)
        hidden_batch = self.actor.init_hidden_states(batch_size=batch_size)
        batches = self.memory.sample(batch_size, time_step=time_step)
        
        #print('batches.shape = {}'.format(batches.shape))
        states = torch.zeros((batch_size, time_step, 256, 256, 2), dtype=torch.float32).to(device)
        actions = torch.zeros((batch_size, time_step), dtype=torch.long).to(device)
        rewards = torch.zeros((batch_size, time_step), dtype=torch.float32).to(device)
        next_states = torch.zeros((batch_size, time_step, 256, 256, 2), dtype=torch.float32).to(device)
        done = torch.zeros((batch_size, time_step), dtype=torch.float32).to(device)

        for i, b in enumerate(batches):
            ac, rw, do = [], [], []
            for j, elem in enumerate(b):
                states[i, j] = torch.tensor(elem[0], dtype=torch.float32)
                ac.append(elem[1])
                rw.append(elem[2])
                #print('elem[3].shape = {}'.format(elem[3].shape))
                #print('next_states[i,j].shape = {}'.format(next_states[i,j].shape))
                next_states[i, j] = torch.tensor(elem[3], dtype=torch.float32)
                do.append(elem[4])
            actions[i] = torch.tensor(ac, dtype=torch.long)
            rewards[i] = torch.tensor(rw, dtype=torch.float32)
            done[i] = torch.tensor(do, dtype=torch.float32)

        # Critic loss (value function loss)  
        ## Get predicted next-state actions and Q values from target models
        ## gather() accumulates values at given index
        #Qvals, _ = self.actor.forward(states, hidden_batch, cell_batch, batch_size=batch_size, time_step=time_step) ## batch_size x action_space
        #print('Qvals = {}'.format(Qvals))
        Qvals, _ = self.actor.forward(states, hidden_batch, batch_size=batch_size, time_step=time_step) ## batch_size x action_space
        Qvals = Qvals.gather(1, actions[:, time_step-1].unsqueeze(1)).squeeze(1) ## batch_size
        #print('actions = {}'.format(actions))
        #print('Qvals = {}'.format(Qvals))

        #next_Q, _ = self.critic.forward(next_states, hidden_batch, cell_batch, batch_size=batch_size, time_step=time_step)##batch_size x action_space
        #print('next_Q = {}'.format(next_Q))
        next_Q, _ = self.critic.forward(next_states, hidden_batch, batch_size=batch_size, time_step=time_step)##batch_size x action_space
        next_Q = next_Q.max(1)[0].detach() ##batch_size
        #print('next_Q = {}'.format(next_Q))
        # Compute Q targets for current states (y_i)
        #print('rewards.shape = {}'.format(rewards[:, time_step-1].shape))
        #print('rewards = {}'.format(rewards))
        #print('done = {}'.format(done))
        Qprime = rewards[:, time_step-1] + self.gamma * next_Q * (1-done[:, time_step-1])
        #print('Qprime = {}'.format(Qprime))

        loss = F.smooth_l1_loss(Qvals, Qprime)
        #print('loss = {}'.format(loss))
        #assert 1==2
        # update networks
        self.actor_optimizer.zero_grad()
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

class DQN_vs_overlap_rnn_no_perception:
    def __init__(self, trained_model_path=None, num_actions=2, input_channels=2, actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=0.97, tau=1e-2, max_memory_size=1000):
        # Params
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.steps_done = 0
        self.input_channels = input_channels

        # Networks
        self.actor = DQN_OVERLAP_Recurrent_Controller_no_perception(self.num_actions, input_size=256).to(device)
        self.critic = DQN_OVERLAP_Recurrent_Controller_no_perception(self.num_actions, input_size=256).to(device)

        if trained_model_path != None:
            self.actor.load_state_dict(torch.load('{}/dqn_epoch_200000.pt'.format(trained_model_path)))
            print('*********************************successfully read the model ...')

        ## copy params from actor.parameters to critic.parameters.
        ## when calling the copy_(),  the argument param.data is the src.
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()

        # Training
        self.memory = ReplayMemory_overlap_dqn_recurrent(max_memory_size)        
        ## only update weights of actor's linear layer
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=actor_learning_rate)

    def update_critic(self):
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()
    
    ## for collecting (state, action, next_state) tuples
    #def select_action(self, state, hidden_state, cell_state, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=10000):
    def select_action(self, state, hidden_state, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=10000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        ## take action with the maximum reward by using the actor
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = state#[0] ## 256 x 256 x 1
                print('obs.shape = {}'.format(obs.shape))
                #print('obs.shape = {}'.format(obs.shape))
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
                #print('obs.shape = {}'.format(obs.shape))
                #model_out = self.actor.forward(obs, hidden_state, cell_state, batch_size=1, time_step=1)
                model_out = self.actor.forward(obs, hidden_state, batch_size=1, time_step=1)
                action = model_out[0].max(1)[1].view(1, 1)
                #hidden_state = model_out[1][0]
                #cell_state = model_out[1][1]
                hidden_state = model_out[1]
                #return action, hidden_state, cell_state
                return action, hidden_state
        else:
            ## take random actions, do exploration
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = state#[0] ## 256 x 256 x 1
                #print('obs.shape = {}'.format(obs.shape))
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
                #print('obs.shape = {}'.format(obs.shape))
                #model_out = self.actor.forward(obs, hidden_state, cell_state, batch_size=1, time_step=1)
                model_out = self.actor.forward(obs, hidden_state, batch_size=1, time_step=1)
                action = torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)
                #hidden_state = model_out[1][0]
                #cell_state = model_out[1][1]
                hidden_state = model_out[1]
                #return action, hidden_state, cell_state
                return action, hidden_state
    
    def update(self, batch_size, time_step=5):
        print('batch_size = {}'.format(batch_size))
        #hidden_batch, cell_batch = self.actor.init_hidden_states(batch_size=batch_size)
        hidden_batch = self.actor.init_hidden_states(batch_size=batch_size)
        batches = self.memory.sample(batch_size, time_step=time_step)
        
        #print('batches.shape = {}'.format(batches.shape))
        states = torch.zeros((batch_size, time_step, 256), dtype=torch.float32).to(device)
        actions = torch.zeros((batch_size, time_step), dtype=torch.long).to(device)
        rewards = torch.zeros((batch_size, time_step), dtype=torch.float32).to(device)
        next_states = torch.zeros((batch_size, time_step, 256), dtype=torch.float32).to(device)
        done = torch.zeros((batch_size, time_step), dtype=torch.float32).to(device)

        for i, b in enumerate(batches):
            ac, rw, do = [], [], []
            for j, elem in enumerate(b):
                states[i, j] = torch.tensor(elem[0], dtype=torch.float32)
                ac.append(elem[1])
                rw.append(elem[2])
                #print('elem[3].shape = {}'.format(elem[3].shape))
                #print('next_states[i,j].shape = {}'.format(next_states[i,j].shape))
                next_states[i, j] = torch.tensor(elem[3], dtype=torch.float32)
                do.append(elem[4])
            for j in range(1, len(rw)):
                rw[j] = rw[j-1] + rw[j]
            actions[i] = torch.tensor(ac, dtype=torch.long)
            rewards[i] = torch.tensor(rw, dtype=torch.float32)
            done[i] = torch.tensor(do, dtype=torch.float32)
        
        '''
        print('actions.shape = {}'.format(actions.shape))
        print('actions = {}'.format(actions))
        print('rewards.shape = {}'.format(rewards.shape))
        print('rewards = {}'.format(rewards))
        print('done.shape = {}'.format(done.shape))
        print('done = {}'.format(done))
        print('states.shape = {}'.format(states.shape))
        print('states = {}'.format(states))
        print('next_states.shape = {}'.format(next_states.shape))
        print('next_states = {}'.format(next_states))
        '''
        #assert 1==2
        '''
        for b in batches:
            cs, ac, rw, ns, do = [], [], [], [], []
            for elem in b:
                cs.append(elem[0])
                ac.append(elem[1])
                rw.append(elem[2])
                ns.append(elem[3])
                do.append(elem[4])
            states.append(cs)
            actions.append(ac)
            rewards.append(rw)
            next_states.append(ns)
            done.append(do)

        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        #print('rewards.shape = {}'.format(rewards.shape))
        #print('rewards = {}'.format(rewards))
        done = torch.tensor(done, dtype=torch.float32).to(device)
        
        print('actions.shape = {}'.format(actions.shape))
        print('actions = {}'.format(actions))
        print('rewards.shape = {}'.format(rewards.shape))
        print('rewards = {}'.format(rewards))
        print('done.shape = {}'.format(done.shape))
        print('done = {}'.format(done))

        tensor_states = np.zeros((batch_size, time_step, 256), dtype=np.float32)
        for i in range(batch_size):
            for j in range(time_step):
                #print('states.shape = {}'.format(states[i][j].shape))
                tensor_states[i, j] = states[i][j]
        states = torch.tensor(tensor_states, dtype=torch.float32).to(device)
        #print('states = {}'.format(states))
        
        tensor_next_states = np.zeros((batch_size, time_step, 256), dtype=np.float32)
        for i in range(batch_size):
            for j in range(time_step):
                tensor_next_states[i, j] = next_states[i][j]
                #print('next_states.shape = {}'.format(next_states[i][j].shape))
        next_states = torch.tensor(tensor_next_states, dtype=torch.float32).to(device)
        #print('next_states = {}'.format(next_states))
        #assert 1==2
        '''

        # Critic loss (value function loss)  
        ## Get predicted next-state actions and Q values from target models
        ## gather() accumulates values at given index
        #Qvals, _ = self.actor.forward(states, hidden_batch, cell_batch, batch_size=batch_size, time_step=time_step) ## batch_size x action_space
        #print('Qvals = {}'.format(Qvals))
        Qvals, _ = self.actor.forward(states, hidden_batch, batch_size=batch_size, time_step=time_step) ## batch_size x action_space
        Qvals = Qvals.gather(1, actions[:, time_step-1].unsqueeze(1)).squeeze(1) ## batch_size
        #print('actions = {}'.format(actions))
        #print('Qvals = {}'.format(Qvals))

        #next_Q, _ = self.critic.forward(next_states, hidden_batch, cell_batch, batch_size=batch_size, time_step=time_step)##batch_size x action_space
        #print('next_Q = {}'.format(next_Q))
        next_Q, _ = self.critic.forward(next_states, hidden_batch, batch_size=batch_size, time_step=time_step)##batch_size x action_space
        next_Q = next_Q.max(1)[0].detach() ##batch_size
        #print('next_Q = {}'.format(next_Q))
        # Compute Q targets for current states (y_i)
        #print('rewards.shape = {}'.format(rewards[:, time_step-1].shape))
        #print('rewards = {}'.format(rewards))
        #print('done = {}'.format(done))
        Qprime = rewards[:, time_step-1] + self.gamma * next_Q * (1-done[:, time_step-1])
        #print('Qprime = {}'.format(Qprime))

        loss = F.smooth_l1_loss(Qvals, Qprime)
        #print('loss = {}'.format(loss))
        #assert 1==2
        # update networks
        self.actor_optimizer.zero_grad()
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

class DQN_vs_overlap_rnn_new:
    def __init__(self, trained_model_path=None, num_actions=2, input_channels=2, actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=0.97, tau=1e-2, max_memory_size=2000):
        # Params
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.steps_done = 0
        self.input_channels = input_channels

        # Networks
        self.perception_actor = Perception_overlap_recurrent(input_channels).to(device)
        self.actor = DQN_OVERLAP_Recurrent_Controller(self.perception_actor, self.num_actions, input_size=256).to(device)
        self.perception_critic = Perception_overlap_recurrent(input_channels).to(device)
        self.critic = DQN_OVERLAP_Recurrent_Controller(self.perception_critic, self.num_actions, input_size=256).to(device)

        if trained_model_path != None:
            self.actor.load_state_dict(torch.load('{}/dqn_epoch_200000.pt'.format(trained_model_path)))
            print('*********************************successfully read the model ...')

        ## copy params from actor.parameters to critic.parameters.
        ## when calling the copy_(),  the argument param.data is the src.
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()

        # Training
        self.memory = ReplayMemory_overlap_dqn_recurrent(max_memory_size)        
        ## only update weights of actor's linear layer
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=actor_learning_rate)

    def update_critic(self):
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()
    
    ## for collecting (state, action, next_state) tuples
    #def select_action(self, state, hidden_state, cell_state, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=10000):
    def select_action(self, state, hidden_state, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=10000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        ## take action with the maximum reward by using the actor
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = state#[0] ## 256 x 256 x 1
                print('obs.shape = {}'.format(obs.shape))
                #print('obs.shape = {}'.format(obs.shape))
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
                #print('obs.shape = {}'.format(obs.shape))
                #model_out = self.actor.forward(obs, hidden_state, cell_state, batch_size=1, time_step=1)
                model_out = self.actor.forward(obs, hidden_state, batch_size=1, time_step=1)
                action = model_out[0].max(1)[1].view(1, 1)
                #hidden_state = model_out[1][0]
                #cell_state = model_out[1][1]
                hidden_state = model_out[1]
                #return action, hidden_state, cell_state
                return action, hidden_state
        else:
            ## take random actions, do exploration
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = state#[0] ## 256 x 256 x 1
                #print('obs.shape = {}'.format(obs.shape))
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
                #print('obs.shape = {}'.format(obs.shape))
                #model_out = self.actor.forward(obs, hidden_state, cell_state, batch_size=1, time_step=1)
                model_out = self.actor.forward(obs, hidden_state, batch_size=1, time_step=1)
                action = torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)
                #hidden_state = model_out[1][0]
                #cell_state = model_out[1][1]
                hidden_state = model_out[1]
                #return action, hidden_state, cell_state
                return action, hidden_state
    
    def update(self, batch_size, time_step=5):
        print('batch_size = {}'.format(batch_size))
        #hidden_batch, cell_batch = self.actor.init_hidden_states(batch_size=batch_size)
        hidden_batch = self.actor.init_hidden_states(batch_size=batch_size)
        batches = self.memory.sample(batch_size, time_step=time_step)
        
        #print('batches.shape = {}'.format(batches.shape))
        states = torch.zeros((batch_size, time_step, 256, 256, 2), dtype=torch.float32).to(device)
        actions = torch.zeros((batch_size, time_step), dtype=torch.long).to(device)
        rewards = torch.zeros((batch_size, time_step), dtype=torch.float32).to(device)
        next_states = torch.zeros((batch_size, time_step, 256, 256, 2), dtype=torch.float32).to(device)
        done = torch.zeros((batch_size, time_step), dtype=torch.float32).to(device)

        for i, b in enumerate(batches):
            ac, rw, do = [], [], []
            for j, elem in enumerate(b):
                states[i, j] = torch.tensor(elem[0], dtype=torch.float32)
                ac.append(elem[1])
                rw.append(elem[2])
                #print('elem[3].shape = {}'.format(elem[3].shape))
                #print('next_states[i,j].shape = {}'.format(next_states[i,j].shape))
                next_states[i, j] = torch.tensor(elem[3], dtype=torch.float32)
                do.append(elem[4])
            for j in range(1, len(rw)):
                rw[j] = rw[j-1] + rw[j]
            actions[i] = torch.tensor(ac, dtype=torch.long)
            rewards[i] = torch.tensor(rw, dtype=torch.float32)
            #print('rewards[{}] = {}'.format(i, rewards[i]))
            done[i] = torch.tensor(do, dtype=torch.float32)

        # Critic loss (value function loss)  
        ## Get predicted next-state actions and Q values from target models
        ## gather() accumulates values at given index
        #Qvals, _ = self.actor.forward(states, hidden_batch, cell_batch, batch_size=batch_size, time_step=time_step) ## batch_size x action_space
        #print('Qvals = {}'.format(Qvals))
        Qvals, _ = self.actor.forward(states, hidden_batch, batch_size=batch_size, time_step=time_step) ## batch_size x action_space
        Qvals = Qvals.gather(1, actions[:, time_step-1].unsqueeze(1)).squeeze(1) ## batch_size
        #print('actions = {}'.format(actions))
        #print('Qvals = {}'.format(Qvals))

        #next_Q, _ = self.critic.forward(next_states, hidden_batch, cell_batch, batch_size=batch_size, time_step=time_step)##batch_size x action_space
        #print('next_Q = {}'.format(next_Q))
        next_Q, _ = self.critic.forward(next_states, hidden_batch, batch_size=batch_size, time_step=time_step)##batch_size x action_space
        next_Q = next_Q.max(1)[0].detach() ##batch_size
        #print('next_Q = {}'.format(next_Q))
        # Compute Q targets for current states (y_i)
        #print('rewards.shape = {}'.format(rewards[:, time_step-1].shape))
        #print('rewards = {}'.format(rewards))
        #print('done = {}'.format(done))
        Qprime = rewards[:, time_step-1] + self.gamma * next_Q * (1-done[:, time_step-1])
        #print('Qprime = {}'.format(Qprime))

        loss = F.smooth_l1_loss(Qvals, Qprime)
        #print('loss = {}'.format(loss))
        #assert 1==2
        # update networks
        self.actor_optimizer.zero_grad()
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

class DDPG_overlap:
    def __init__(self, trained_model_path=None, num_actions=2, input_channels=2, hidden_size=128, actor_learning_rate=1e-5, critic_learning_rate=1e-5, gamma=0.97, tau=1e-2, max_memory_size=50000):
        # Params
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.steps_done = 0
        self.input_channels = input_channels

        # Networks
        self.perception_actor = Perception_overlap(input_channels).to(device)
        self.actor = DDPG_Actor(self.perception_actor, hidden_size, self.num_actions).to(device)
        self.perception_actor_target = Perception_overlap(input_channels).to(device)
        self.actor_target = DDPG_Actor(self.perception_actor_target, hidden_size, self.num_actions).to(device)
        self.perception_critic = Perception_overlap(input_channels).to(device)
        self.critic = DDPG_Critic(self.perception_critic, hidden_size, self.num_actions, input_size=256+self.num_actions).to(device)
        self.perception_critic_target = Perception_overlap(input_channels).to(device)
        self.critic_target = DDPG_Critic(self.perception_critic_target, hidden_size, self.num_actions, input_size=256+self.num_actions).to(device)

        if trained_model_path != None:
            self.actor.load_state_dict(torch.load('{}/actor_epoch_200000.pt'.format(trained_model_path)))
            self.critic.load_state_dict(torch.load('{}/critic_epoch_200000.pt'.format(trained_model_path)))

        ## copy params from actor.parameters to actor_target.parameters.
        ## when calling the copy_(),  the argument param.data is the src.
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = ReplayMemory_overlap_dqn(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        ## only update weights of actor and critic
        ## weights for actor_target and critic_target are copied rather than learned
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    ## for collecting (state, action, next_state) tuples
    def select_action(self, state, max_sigma=0.3, min_sigma=0.01, decay_period=3000):
        with torch.no_grad():
            self.steps_done += 1
            obs = state[0] ## 256 x 256 x 1
            #print('obs.shape = {}'.format(obs.shape))
            obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2)
            #print('obs.shape = {}'.format(obs.shape))
            action = self.actor.forward(obs)
            ## index is 0 because batch size is 1
            action = action.tolist()[0]
            print('network action = {}'.format(action))
            sigma = min_sigma + (max_sigma - min_sigma) * math.exp(-1. * self.steps_done / decay_period)
            print('sigma = {}'.format(sigma))
            noise = sigma * np.random.randn(1)
            action += noise
            action = np.clip(action, -1, +1)
            return action

    def update_target(self):
        # update target networks 
        # TD (temporal difference) update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    
    def update(self, batch_size):
        states, actions, rewards, next_states, done = self.memory.sample(batch_size)
        
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(1)
        
        tensor_states = np.zeros((batch_size, 256, 256, self.input_channels), dtype=np.float32)
        for i in range(batch_size):
            tensor_states[i] = states[i][0]
        states = torch.tensor(tensor_states, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
        
        tensor_next_states = np.zeros((batch_size, 256, 256, self.input_channels), dtype=np.float32)
        for i in range(batch_size):
            tensor_next_states[i] = next_states[i][0]
        next_states = torch.tensor(tensor_next_states, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
    
        # Critic loss (value function loss)  
        ## Get predicted next-state actions and Q values from target models     
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        # Compute Q targets for current states (y_i)
        Qprime = rewards + self.gamma * next_Q * (1-done)
        critic_loss = self.critic_criterion(Qvals, Qprime)
        print('critic_loss = {}'.format(critic_loss))

        # Actor loss (policy loss)
        actor_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        print('actor_loss = {}'.format(actor_loss))
        
        # update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

        ## only critic_params_to_learn parameters are learned
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()

        self.update_target()
