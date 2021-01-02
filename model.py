import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim=8, action_dim=3):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        self.action_x_layer = nn.Linear(64, action_dim)
        self.action_y_layer = nn.Linear(64, action_dim)
        self.value_layer = nn.Linear(128, 1)
        
        self.temp_x_logprobs = []
        self.temp_y_logprobs = []
        self.temp_state_values = []
        self.temp_rewards = []
        
        self.x_logprobs = []
        self.y_logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        state = torch.from_numpy(state.flatten()).float()
        x1 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x1))
        
        state_value = self.value_layer(x1)

        x_action_probs = F.softmax(self.action_x_layer(x2), dim=0)
        x_action_distribution = Categorical(x_action_probs)
        x_action = x_action_distribution.sample()
        
        y_action_probs = F.softmax(self.action_y_layer(x2), dim=0)
        y_action_distribution = Categorical(y_action_probs)
        y_action = y_action_distribution.sample()
        
        self.temp_x_logprobs.append(x_action_distribution.log_prob(x_action))
        self.temp_y_logprobs.append(y_action_distribution.log_prob(y_action))
        self.temp_state_values.append(state_value)
        
        return np.array([x_action.item(), y_action.item()])
    
    def calculateLoss(self):                
        # normalizing the rewards:
        rewards = torch.tensor(self.rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for x_logprob, y_logprob, value, reward in zip(self.x_logprobs, self.y_logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_x_loss = -x_logprob * advantage / 2
            action_y_loss = -y_logprob * advantage / 2
            value_loss = F.smooth_l1_loss(value[0], reward)
            loss += (action_x_loss + action_y_loss + value_loss)   
        return loss
    
    def updateMemory(self, gamma=0.95):
        rewards = []
        dis_reward = 0
        for reward in self.temp_rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
        self.x_logprobs += self.temp_x_logprobs
        self.y_logprobs += self.temp_y_logprobs
        self.state_values += self.temp_state_values
        self.rewards += rewards
        self.clearTempMemory()
        
    def clearTempMemory(self):
        del self.temp_x_logprobs[:]
        del self.temp_y_logprobs[:]
        del self.temp_state_values[:]
        del self.temp_rewards[:]
    
    def clearMemory(self):
        del self.x_logprobs[:]
        del self.y_logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
        
    def clip_grads(self, eps=1):
        self.fc1.weight.grad.data.clamp_(-eps, eps)
        self.fc2.weight.grad.data.clamp_(-eps, eps)
        self.action_x_layer.weight.grad.data.clamp_(-eps, eps)
        self.action_y_layer.weight.grad.data.clamp_(-eps, eps)
        self.value_layer.weight.grad.data.clamp_(-eps, eps)
