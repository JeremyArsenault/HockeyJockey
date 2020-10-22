import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=3):
        super(ActorCritic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim,128)
        torch.nn.init.xavier_uniform_(self.fc1.weight)      
        self.fc2 = nn.Linear(128,64)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fcx = nn.Linear(64, action_dim)
        torch.nn.init.xavier_uniform_(self.fcx.weight)
        self.fcy = nn.Linear(64, action_dim)
        torch.nn.init.xavier_uniform_(self.fcy.weight)
        self.fcc = nn.Linear(128, 1)
        torch.nn.init.xavier_uniform_(self.fcc.weight)
        
    def forward(self, state):
        x1 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x1))
        logit_x = F.softmax(self.fcx(x2), dim=1)
        logit_y = F.softmax(self.fcy(x2), dim=1)
        critic = self.fcc(x1)
        return logit_x, logit_y, critic
    
    def actor(self, state):
        logit_x, logit_y, _ = self.forward(state)
        dist_x = torch.distributions.Categorical(logit_x)
        dist_y = torch.distributions.Categorical(logit_y)
        return dist_x, dist_y
    
    def critic(self, state):
        _, _, critic = self.forward(state)
        return critic
    
    def get_action(self, state):
        with torch.no_grad():
            dist_x, dist_y = self.actor(state)
        x, y = dist_x.sample(), dist_y.sample()
        action = np.array([x,y])
        return action
    
    def clip_grads(self, eps=1):
        self.fc1.weight.grad.data.clamp_(-eps, eps)
        self.fc2.weight.grad.data.clamp_(-eps, eps)
        self.fcx.weight.grad.data.clamp_(-eps, eps)
        self.fcy.weight.grad.data.clamp_(-eps, eps)
        self.fcc.weight.grad.data.clamp_(-eps, eps)
        
class Actor(nn.Module):
    """
    Actor: Returns distribution over discrete action spac
    """
    def __init__(self, state_dim, action_dim=3):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim,64)
        torch.nn.init.xavier_uniform_(self.fc1.weight)      
        self.fc2 = nn.Linear(64,32)
        torch.nn.init.xavier_uniform_(self.fc2.weight)      
        self.fc3 = nn.Linear(32,16)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.fcx = nn.Linear(16,action_dim)
        torch.nn.init.xavier_uniform_(self.fcx.weight)
        self.fcy = nn.Linear(16, action_dim)
        torch.nn.init.xavier_uniform_(self.fcy.weight)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logit_x = F.softmax(self.fcx(x), dim=1)
        logit_y = F.softmax(self.fcy(x), dim=1)
        dist_x = torch.distributions.Categorical(logit_x)
        dist_y = torch.distributions.Categorical(logit_y)
        return dist_x, dist_y
    
    def get_action(self, state):
        with torch.no_grad():
            dist_x, dist_y = self.forward(state)
        x, y = dist_x.sample(), dist_y.sample()
        action = np.array([x,y])
        return action
    
    def clip_grads(self, eps=1):
        self.fc1.weight.grad.data.clamp_(-eps, eps)
        self.fc2.weight.grad.data.clamp_(-eps, eps)
        self.fc3.weight.grad.data.clamp_(-eps, eps)
        self.fcx.weight.grad.data.clamp_(-eps, eps)
        self.fcy.weight.grad.data.clamp_(-eps, eps)
        
class Critic(nn.Module):
    """
    Q(s): Evaluates expect discounded sum of rewards of state independent from action
    """
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim,64)
        torch.nn.init.xavier_uniform_(self.fc1.weight) 
        self.fc2 = nn.Linear(64,32)
        torch.nn.init.xavier_uniform_(self.fc2.weight) 
        self.fc3 = nn.Linear(32,16)
        torch.nn.init.xavier_uniform_(self.fc3.weight) 
        self.fc4 = nn.Linear(16,1)
        torch.nn.init.xavier_uniform_(self.fc4.weight) 

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def clip_grads(self, eps=1):
        self.fc1.weight.grad.data.clamp_(-eps, eps)
        self.fc2.weight.grad.data.clamp_(-eps, eps)
        self.fc3.weight.grad.data.clamp_(-eps, eps)
        self.fc4.weight.grad.data.clamp_(-eps, eps)
