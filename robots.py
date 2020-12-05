import numpy as np
#import RPi.GPIO as GPIO
import time

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
        
        # label pins
        self.pins = {'x_up':11, 'x_down':12,
                     'y1_up':16, 'y1_down':15,
                     'y2_up':32, 'y2_down':31,
                     }
        # pin setup
        GPIO.setmode(GPIO.BOARD)
        for pin in self.pins.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin,GPIO.HIGH)            
        
            
    def execute(self, action):
        """
        execute action. blocks until action complete
        """
        x_sleep = 0
        y_sleep = 0
        start_time = time.time()
        
        if action[0]==0:
            # move up
            GPIO.output(self.pins['y1_up'],GPIO.LOW)
            GPIO.output(self.pins['y2_up'],GPIO.LOW)
            y_sleep = self.pulse_up

        elif action[0]==2:
            # move down
            GPIO.output(self.pins['y1_down'],GPIO.LOW)
            GPIO.output(self.pins['y2_down'],GPIO.LOW)
            y_sleep = self.pulse_down

        if action[1]==0:
            # move left
            GPIO.output(self.pins['x_down'],GPIO.LOW)
            x_sleep = self.pulse_left
        
        elif action[1]==2:
            # move right
            GPIO.output(self.pins['x_up'],GPIO.LOW)
            x_sleep = self.pulse_right
        
        # kill x
        time.sleep(x_sleep)
        GPIO.output(self.pins['x_up'],GPIO.HIGH)
        GPIO.output(self.pins['x_down'],GPIO.HIGH)
        
        # kill y
        time.sleep(max(0,y_sleep-x_sleep))
        GPIO.output(self.pins['y1_up'],GPIO.HIGH)
        GPIO.output(self.pins['y2_up'],GPIO.HIGH)
        GPIO.output(self.pins['y1_down'],GPIO.HIGH)
        GPIO.output(self.pins['y2_down'],GPIO.HIGH)
        
            