import torch
from image_processing import Camera
from model import ActorCritic
from robots import DiscreteActionBot as Bot
import time

MODEL_PATH = 'models/actor_critic.pkl'
FREQUENCY = 0.2

if __name__=="__main__":
    camera = Camera(0)
    agent = Bot()
    
    model = torch.load(MODEL_PATH)
    
    #while True:
    for i in range(5):
        t = time.time()
        
        state = camera.get_state()
        #action = model(state)
        action = ([0,1])
        agent.execute(action)
        
        elapsed = time.time()-t
        print(elapsed)
        if elapsed < FREQUENCY:
            time.sleep(FREQUENCY - elapsed)
    
    