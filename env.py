import time

class Env:
    def __init__(self, camera, freq):
        self.freq = freq
        self.camera = camera
        
        self.time = time.time()
        s = camera.get_state()
        self.striker1_pos = s[1]
        self.striker2_pos = s[2]
        self.puck_pos = s[0]
        
    def get_state(self):
        """
        Wait until time=freq has passed and return game state
        """
        while time.time()-self.time < self.freq:
            time.sleep(0.01)
        time_delt = time.time()-self.time()
        self.time = time.time()
        
        self.corners = camera.get_corners()
        striker1_pos = camera.get_striker1_pos(self.corners)
        striker2_pos = camera.get_striker2_pos(self.corners)
        puck_pos = camera.get_puck_pos(self.corners)
        
        striker1_vel = (striker1_pos-self.striker1_pos)/time_delt
        striker2_vel = (striker2_pos-self.striker2_pos)/time_delt
        puck_vel = (puck_pos-self.puck_pos)/time_delt
        
        self.striker1_pos = striker1_pos
        self.striker2_pos = striker2_pos
        self.puck_pos = puck_pos

        return np.array([puck_pos, puck_vel, striker1_pos, striker1_vel, striker2_pos, striker2_vel])
    
    def step(self, action):
        """
        Move robot according to action
        """
        pass