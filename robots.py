import numpy as np
from ctypes import cdll

class DiscreteActionBotSim:
    """
    Model of robot dynamics in discrete action space
    """
    def __init__(self):
        self.yvel = (-0.5, 0, 0.5)
        self.xvel = (0.7, 0, -0.7)
        self.std = 0.1
        
    def sample(self):
        """
        sample from action space
        """
        return np.random.randint(0,3, size=(2,)) # 3 actions
    
    def execute(self, action, v0, p):
        """
        return striker velocity after action
        """
        vx = np.random.normal(self.xvel[action[0]], self.std)
        vy = np.random.normal(self.yvel[action[1]], self.std)
        return np.array([vx,vy])
    
class DiscreteActionBot:
    """
    Model of robot dynamics in discrete action space
    """
    def __init__(self):
        # consts
        self.pulse_up = 0.125
        self.pulse_down = 0.18
        self.pulse_left = 0.075
        self.pulse_right = 0.09
        
        # see motor_control.c for shared compilation instructions
        self.mc_lib = cdll.LoadLibrary('./motor_control.so')
        self.mc_lib.pinSetup()
        
    def __del__(self):
        self.mc_lib.pinSetup()
            
    def execute(self, action):
        """
        execute move action. Does not block :)
        """
        if action[0]==0:
            # move up
            self.mc_lib.moveUp(int(self.pulse_up*1000))

        elif action[0]==2:
            # move down
            self.mc_lib.moveDown(int(self.pulse_down*1000))

        if action[1]==0:
            # move left
            self.mc_lib.moveLeft(int(self.pulse_left*1000))
        
        elif action[1]==2:
            # move right
            self.mc_lib.moveRight(int(self.pulse_right*1000))
            