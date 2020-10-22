import numpy as np

def uniform_sample(buffer, batch_size):
    """
    Sample uniformly from buffer
    """
    inds = np.random.choice(len(buffer), batch_size, replace=True)
    sample = [buffer[x] for x in inds]
    return sample, inds

def priority_sample(buffer, batch_size):
    """
    Sample highest TD priority batch from buffer
    """
    priorities = np.array([x[8] for x in buffer])
    inds = np.argsort(priorities)[-batch_size:]
    sample = [buffer[x] for x in inds]
    return sample, inds

def stoch_priority_sample(buffer, batch_size):
    """
    Sample from buffer using naive stochastic priority sampling
    """
    priorities = np.array([x[8] for x in buffer])*np.random.uniform(size=len(buffer))
    inds = np.argsort(priorities)[-batch_size:]
    sample = [buffer[x] for x in inds]
    return sample, inds

def sample_states(env, num_states=50, buff_size=100):
    """
    Sample random states from environment
    """
    buffer = []
    state, _ = env.reset()
    while len(buffer)<buff_size:
        buffer.append(state)
        state, _, _, _, done = env.step(env.action_space.sample(), env.action_space.sample())
        if done:
            state, _ = env.reset()
    sample, _ = uniform_sample(buffer, num_states)
    return torch.Tensor(sample)

def compute_max_q(actor_critic, states):
    """
    Return the max model evaluation of sampled states
    """
    actor_critic.eval()
    with torch.no_grad():
        q_vals = actor_critic.critic(states)
    return float(q_vals.max())

def simulate_rp(env, actor_critic, episodes=50):
    """
    Return avg reward vs random policy
    """
    rewards = 0
    for e in range(episodes):
        done = False
        state, _ = env.reset()
        while not done:
            action_1 = actor_critic.get_action(torch.Tensor([state]))
            action_2 = env.action_space.sample()
            state, reward, _, _, done = env.step(action_1, action_2)
            rewards += reward
    return rewards / episodes

def update(actor_critic, buffer, batch_size, optimiser):
    # sample buffer and preprocess batch
    #critic.eval()
    #actor.eval()
    
    #batch, batch_inds = uniform_sample(buffer, batch_size)
    batch = buffer
    batch_states = torch.Tensor([x[0] for x in batch])
    batch_next_states = torch.Tensor([x[1] for x in batch])
    batch_x_actions = torch.Tensor([x[2] for x in batch])
    batch_y_actions = torch.Tensor([x[3] for x in batch])
    batch_x_logprobs = torch.cat([x[4] for x in batch])
    batch_y_logprobs = torch.cat([x[5] for x in batch])
    batch_rewards = torch.Tensor([x[6] for x in batch])
    batch_mask = torch.Tensor([x[7] for x in batch])
    # batch_exp_rewards = batch_rewards + discount_factor*batch_mask*critic(batch_next_states)
    
    # compute losses and update models 
    #critic.train()
    #actor.train()
    actor_critic.train()
    
    dists_x, dists_y = actor_critic.actor(batch_states)
    #batch_x_logprobs = dists_x.log_prob(batch_x_actions)
    #batch_y_logprobs = dists_y.log_prob(batch_y_actions)
    values = actor_critic.critic(batch_states)
    advantage = batch_rewards - values
    
    actor_x_loss = -(batch_x_logprobs * advantage).mean()
    actor_y_loss = -(batch_y_logprobs * advantage).mean()
    #actor_loss = actor_x_loss + actor_y_loss
    critic_loss = advantage.pow(2).mean()
    loss = critic_loss + actor_x_loss + actor_y_loss
    

    optimiser.zero_grad()
    loss.backward()
    actor_critic.clip_grads()
    optimiser.step()
    
    """
    critic_optimiser.zero_grad()
    actor_optimiser.zero_grad()
    critic_loss.backward()
    actor_loss.backward()
    actor.clip_grads()
    critic.clip_grads()
    actor_optimiser.step()
    critic_optimiser.step()
    """
    
    # update buffer priorities
    """
    td = torch.abs(advantage.detach())
    for batch_i, buff_i in enumerate(batch_inds):
        b = list(buffer[buff_i])
        b[8] = float(td[batch_i][0])
        buffer[buff_i] = tuple(b)
    """

def train(env, actor_critic, optimiser, episodes, buffer_size=1000, batch_size=500, discount_factor=0.95, verbose=True):
    """
    Simulate and train model to maximize reward from env
    """
    q_states = sample_states(env)
    q_history = []
    rp_history = []
    initial_priority = 10

    buffer = []
    for episode in range(episodes):
        p1_buff, p2_buff = [], []
        done = False
        state_1, state_2 = env.reset()
        frames = [env.render()]
        while not done:
            # take step and append to buffer
            # with torch.no_grad():
            dist_x_1, dist_y_1 = actor_critic.actor(torch.Tensor([state_1]))
            dist_x_2, dist_y_2 = actor_critic.actor(torch.Tensor([state_2]))
            action_x_1, action_y_1 = dist_x_1.sample(), dist_y_1.sample()
            action_x_2, action_y_2 = dist_x_2.sample(), dist_y_2.sample()
            logprob_x_1, logprob_y_1 = dist_x_1.log_prob(action_x_1).unsqueeze(0), dist_y_1.log_prob(action_y_1).unsqueeze(0)
            logprob_x_2, logprob_y_2 = dist_x_2.log_prob(action_x_2).unsqueeze(0), dist_y_2.log_prob(action_y_2).unsqueeze(0)
            #logprob_x_1, logprob_y_1 = None, None
            #logprob_x_2, logprob_y_2 = None, None
            action_1 = torch.stack((action_x_1, action_y_1), dim=1).numpy()[0]
            action_2 = torch.stack((action_x_2, action_y_2), dim=1).numpy()[0]
            
            next_state_1, reward_1, next_state_2, reward_2, done  = env.step(action_1, action_2)
            
            frames.append(env.render())
            
            mask = 0 if done else 1
                
            p1_buff.append([state_1, next_state_1, action_x_1, action_y_1, logprob_x_1, logprob_y_1, reward_1, mask, initial_priority])
            p2_buff.append([state_2, next_state_2, action_x_2, action_y_2, logprob_x_2, logprob_y_2, reward_2, mask, initial_priority])
            state_1 = next_state_1
            state_2 = next_state_2
            
        with open('models/current_game.npy', 'wb') as f:
            np.save(f, np.array(frames))
            
        for buff in [p1_buff, p2_buff]:
            r = 0
            for transition in reversed(buff):
                r = transition[6] + discount_factor*r
                transition[6] = r
            
        buffer += p1_buff
        buffer += p2_buff
        """
        # if buffer isn't populated keep simulating               
        if len(buffer)>buffer_size:
            buffer.pop(0)
        elif len(buffer)<batch_size:
            continue
        """
        # do a training step
        # update(actor, critic, buffer, batch_size, actor_optimiser, critic_optimiser)
        update(actor_critic, buffer, batch_size, optimiser)
        buffer = []
            
        # update performance metrics
        q, rp_reward = None, None
        if episode%10==0:
            q = compute_max_q(actor_critic, q_states)
            q_history.append(q)

            rp_reward = simulate_rp(env, actor_critic)
            rp_history.append(rp_reward)
            
        if verbose==True and episode%250==0:
            print('Episode: ',episode,'  max q: ',q, '  rp avg: ',rp_reward)
            
    return q_history, rp_history

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from models import Actor, Critic, ActorCritic
    from simulation_env import AirHockey
    from robots import DiscreteActionBotSim
    import torch
    
    num_episodes = 5000
    actor_lr = 7e-4
    critic_lr = 7e-4
    lr = 2e-3
    
    env = AirHockey(DiscreteActionBotSim())
    
    actor_critic = ActorCritic(12)
    #actor = Actor(12)
    #critic = Critic(12)
    #actor = torch.load('models/wide_actor.pkl')
    #critic = torch.load('models/wide_critic.pkl')

    #actor_optim = torch.optim.RMSprop(actor.parameters(), lr=actor_lr)
    #critic_optim = torch.optim.RMSprop(critic.parameters(), lr=critic_lr)
    optim = torch.optim.RMSprop(actor_critic.parameters(), lr=lr)

    #q_history, rp_history = train(env, actor, critic, actor_optim, critic_optim, num_episodes)
    q_history, rp_history = train(env, actor_critic, optim, num_episodes)
    torch.save(actor_critic, 'models/wide_actor_critic.pkl')
    #torch.save(critic, 'models/wide_critic.pkl')
    with open('models/q_history.npy', 'wb') as f:
        np.save(f, np.array(q_history))
    with open('models/rp_history.npy', 'wb') as f:
        np.save(f, np.array(rp_history))