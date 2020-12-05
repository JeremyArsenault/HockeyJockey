from model import ActorCritic
import torch
import torch.optim as optim
from simulation_env import AirHockey, SinglePlayerReturn
from robots import DiscreteActionBotSim
import copy
import gym
import numpy as np

def train_single_player_return():
    max_episodes = 50000
    episodes_per_update = 10
    render = False
    gamma = 0.99
    lr = 0.005
    betas = (0.9, 0.999)
    path = 'models/single_player.pkl'
    
    env = SinglePlayerReturn(DiscreteActionBotSim())
    
    policy = ActorCritic()
    
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    
    r = []
    i_episode = 0
    while i_episode<max_episodes:
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            state, reward, done, i = env.step(action)
            policy.temp_rewards.append(reward) 
            
        i_episode += 1
        r.append(reward)
        
        if i_episode%10:
            policy.updateMemory(gamma)
            continue

        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss()
        loss.backward()
        policy.clip_grads()
        optimizer.step()
        policy.clearMemory()
    
        if i_episode%500==0:
            if len(r)>50:
                print('Episode ',i_episode,': avg reward ',sum(r)/len(r))
            else:
                print('Episode ',i_episode,': avg reward n/a')
                
        if i_episode%1000==0:
            torch.save(policy, path)

        if len(r)>100:
            r.pop(0)
            if sum(r)/len(r)>0.8:
                r = []
                v += 1
                print('CONVERGED v',v)
                break

    torch.save(policy, path)

def train():

    max_episodes = 200000
    episodes_per_update = 5
    render = False
    gamma = 0.99
    lr = 0.0015
    betas = (0.9, 0.999)
    convergence_thresh = 0.1
    
    env = AirHockey(DiscreteActionBotSim())
    
    policy = ActorCritic()
    old_policy = ActorCritic()
    #policy = torch.load('models/actor_critic.pkl')
    #old_policy = torch.load('models/actor_critic.pkl')
    old_policy.eval()
    
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    
    v = 0
    r = []
    i_episode = 0
    while i_episode<max_episodes:
        state, s = env.reset()
        done = False
        rew = 0
        while not done:
            action = policy(state)
            state, reward, s, _, done, draw = env.step(action, old_policy(s))
            policy.temp_rewards.append(reward)
            rew += reward   
            
        if draw and np.random.rand()>0.5:
            policy.clearTempMemory()
            continue
            
        i_episode += 1
        r.append(rew)
        
        if i_episode%episodes_per_update:
            policy.updateMemory(gamma)
            continue

        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss()
        loss.backward()
        policy.clip_grads()
        optimizer.step()
        policy.clearMemory()
        old_policy.clearTempMemory()
    
        if i_episode%500==0:
            if len(r)>50:
                print('Episode ',i_episode,' v'+str(v)+': avg reward ',sum(r)/len(r))
            else:
                print('Episode ',i_episode,' v'+str(v)+': avg reward n/a')

        if len(r)>100:
            r.pop(0)
            if sum(r)/len(r)>convergence_thresh:
                r = []
                v += 1
                i_episode = 0
                print('CONVERGED v',v)
                path = 'models/actor_critic.pkl'
                torch.save(policy, path)
                old_policy = torch.load(path)
                old_policy.eval()
            
    path = 'models/actor_critic.pkl'
    torch.save(policy, path)
                
if __name__ == '__main__':
    #train_single_player_return()
    train()
