import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import torch
from simulation_env import AirHockey
from robots import DiscreteActionBotSim
import model

#from array2gif import write_gif

MODEL_PATH = 'models/actor_critic.pkl'

if __name__=='__main__':
    env = AirHockey(DiscreteActionBotSim())
    actor = torch.load(MODEL_PATH)

    state1, state2 = env.reset()
    frames = [env.render()]
    done = False
    
    while not done:
        a1 = actor(state1)
        a2 = actor(state2)
        state1, reward, state2, _, done, info = env.step(a1, a2)
        frames.append(env.render())
    print(reward)
    
    #write_gif(np.array(frames)[:50], '.images/sample_simulation.gif', fps=15)
    
    for frame in frames:
        cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.waitKey(25)
    cv2.destroyAllWindows()