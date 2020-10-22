import numpy as np

class DiscreteActionBotSim:
    """
    Model of robot dynamics in discrete action space
    """
    def __init__(self):
        self.acc = np.array([0.07, 0.05])
        self.drag = np.array([0.8, 0.8])
        self.ylim = (-0.68, -0.25)
        self.xlim = (-0.32, 0.32)
        
    def sample(self):
        """
        sample from action space
        """
        # return np.random.randint(0,3, size=(2,)) # 3 actions
        return np.random.randint(0,2, size=(2,)) # 2 actions
    
    def execute(self, action, v0, p):
        """
        return striker velocity after action
        """
        # v = self.drag * (v0 + (action-1)*self.acc)  # 3 actions
        v = self.drag * (v0 + (2*action-1)*self.acc) # 2 actions
        if p[1]>self.ylim[1] and v[1]>0:
            v[1] = 0
        elif p[1]<self.ylim[0] and v[1]<0:
            v[1] = 0
        if p[0]>self.xlim[1] and v[0]>0:
            v[0] = 0
        elif p[0]<self.xlim[0] and v[0]<0:
            v[0] = 0
        return v
    
class DiscreteActionBot:
    """
    Model of robot dynamics in discrete action space
    """