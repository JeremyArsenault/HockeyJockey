import torch
from image_processing import Camera
from model import ActorCritic
from env import Env

MODEL_PATH = 'models/actor_critic.pkl'
FREQUENCY = 0.2

if __name__=="__main__":
    camera = Camera()
    env = Env(camera, FREQUENCY)
    
    model = torch.load(MODEL_PATH)
    
    while True:
        t = time.time()
        
        state = env.get_state()
        action = model(state)
        env.step(action)
        
        elapsed = time.time()-t
        if elapsed < FREQUENCY:
            time.sleep(FREQUENCY - elapsed)
    
    