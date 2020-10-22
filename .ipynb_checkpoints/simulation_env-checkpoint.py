import numpy as np
import math

class HockeyPhysics:
    """
    Deterministic physics model of air hockey game
    
    Coordinates are in form (x,y) meters with (0,0) located at the center of the table surface
    Velocities are in form (dx,dy) meters/second

    Only collisions between the ball and env will be handled.
    
    Table bumpers will absorb 10% of normal velocity
    Striker will absorb 20% of normal velocity
    
    Assume table is frictionless w/ no drag
    
    https://en.wikipedia.org/wiki/Air_hockey
    """
    
    def __init__(self):
        # state vector
        self.state = {
            'puck' : {
                'position':np.array([0,0]),
                'velocity':np.array([0,0]),
            },
            'striker1' : {
                'position':np.array([0,-1.4]),
                'velocity':np.array([0,0]),
            },
            'striker2' : {
                'position':np.array([0,0.6]),
                'velocity':np.array([-.1,-.1]),
            },
        }
        
        # useful world constants
        self.consts = {
            'table_width' : 0.75,
            'table_length' : 1.5,
            'table_corner_radius' : 0.01,
            'table_goal_width' : 0.65,
            'table_bounce' : 0.8,
            
            'puck_radius' : 0.038,
            
            'striker_radius' : 0.07,
            'striker_bounce' : 0.7,
        }
    
    def is_goal(self):
        """
        Return 1 for s1 goal, 0 for no goal, -1 for s2 goal
        """
        py = self.state['puck']['position'][1]
        length = self.consts['table_length']
        if py>(length/2):
            return 1
        elif py<(-length/2):
            return -1
        return 0
        
    def simulate(self, time):
        """
        Step forward in time and update state vector
        Return all collisions in order of occurance
        """
        collisions = []
        while time:
            # check all possible collisions
            all_collisions = np.array([
                self.bumper_collision_time(time),
                self.striker_collision_time('striker1', time),
                self.striker_collision_time('striker2', time),
                time,
            ], dtype=np.float)
            min_time = np.nanmin(all_collisions)
        
            # handle nearest collision
            self.update_positions(min_time)
            if min_time == all_collisions[0]:
                self.bumper_collision()
                collisions.append('bumper')
            elif min_time == all_collisions[1]:
                self.striker_collision('striker1')
                collisions.append('striker1')
            elif min_time == all_collisions[2]:
                self.striker_collision('striker2')
                collisions.append('striker2')
                
            if len(collisions)>10000:
                print('Bounce loop')
                return collisions
            if len(collisions)>2 and (collisions[-1]==collisions[-2]) and (collisions[-1][0]=='s'):
                print('Double striker bounce')
                return collisions
                
            # update time remaining in simulation step
            time -= min_time
        return collisions
    
    def update_positions(self, t):
        """
        Update positions of objects over time interval
        """
        table_w = self.consts['table_width']
        table_l  = self.consts['table_length']
        r = self.consts['puck_radius']
        p_vel = self.state['puck']['velocity']
        p_pos = self.state['puck']['position']   
        s1_vel = self.state['striker1']['velocity']
        s1_pos = self.state['striker1']['position']  
        s2_vel = self.state['striker2']['velocity']
        s2_pos = self.state['striker2']['position']
        
        p_vel = min(1, 1e8/np.linalg.norm(p_vel)) * p_vel
        self.state['puck']['velocity'] = p_vel
        
        s_ylim = (-table_l/2+r, -r)
        s_xlim = (-table_w/2+r, table_w/2-r)
        p1 = s1_pos + s1_vel*t
        p2 = s2_pos + s2_vel*t
        if not (s_xlim[0]<=p1[0]<=s_xlim[1]):
            s1_vel[0] = 0
        if not (s_xlim[0]<=-p2[0]<=s_xlim[1]):
            s2_vel[0] = 0
        if not (s_ylim[0]<=p1[1]<=s_ylim[1]):
            s1_vel[1] = 0
        if not (s_ylim[0]<=-p2[1]<=s_ylim[1]):
            s2_vel[1] = 0        
        
        self.state['puck']['position'] = p_pos + p_vel*t
        self.state['striker1']['position'] = s1_pos + s1_vel*t
        self.state['striker2']['position'] = s2_pos + s2_vel*t
        
    def circle_intersection_time(self, pos, vel, center, r, order):
        """
        Return the time a point w/ pos, vel intersects with a circle w/
        center, r either entering or exiting the circle
        """
        a = vel[0]**2+vel[1]**2
        b = 2*(pos[0]*vel[0]+pos[1]*vel[1]-vel[0]*center[0]-vel[1]*center[1])
        c = pos[0]**2+pos[1]**2+center[0]**2+center[1]**2-2*(pos[0]*center[0]+pos[1]*center[1])-r**2
        
        if a==0:
            return None
        """
        if a>1e25 or b>1e25 or c>1e25:
            print('Weird shit: ',pos,vel)
            return None
        """
        
        discrim = b**2-4*a*c
            
        if discrim <= 0:
            return None
        else:
            x1 = (-b+math.sqrt(discrim))/(2*a)
            x2 = (-b-math.sqrt(discrim))/(2*a)
            if order=='first':
                t0 = min(x1,x2)
            elif order=='second':
                t0 = max(x1,x2)
        
        if t0<=0:
            return None
        return t0
    
    def bumper_collision_time(self, time):
        """
        If puck is on collision course w/ table return expected collision time
        else return None
        """           
        table_w = self.consts['table_width']
        table_l  = self.consts['table_length']
        goal_w = self.consts['table_goal_width']
        table_r = self.consts['table_corner_radius']
        r = self.consts['puck_radius']
            
        vel = self.state['puck']['velocity']
        pos = self.state['puck']['position']
        
        if np.linalg.norm(vel)==0:
            return None
        
        t0s = [time+1]
        
        # side bumpers
        if vel[0]>0:
            t0s.append((table_w/2 - r - pos[0])/vel[0])
        elif vel[0]<0:
            t0s.append((-table_w/2 + r - pos[0])/vel[0])
        
        # end bumpers
        if vel[1]>0:
            t0 = (table_l/2 - r - pos[1])/vel[1]
            if t0>0 and abs(pos[0]+vel[0]*t0)>=(goal_w/2):
                t0s.append(t0)
        elif vel[1]<0:
            t0 = (-table_l/2 + r - pos[1])/vel[1]
            if t0>0 and abs(pos[0]+vel[0]*t0)>=(goal_w/2):
                t0s.append(t0)
             
        # table corners
        centers = [(table_w/2-table_r, -table_l/2+table_r),(-table_w/2+table_r, table_l/2-table_r),\
                   (table_w/2-table_r, table_l/2-table_r),(-table_w/2+table_r, -table_l/2+table_r)]
        for c in centers:
            t0 = self.circle_intersection_time(pos, vel, c, max(0,table_r-r), 'second')
            if t0 is not None:
                p = pos+vel*t0
                if abs(p[0])>abs(c[0]) and abs(p[1])>abs(c[1]):
                    t0s.append(t0)
            
        # goal corners
        centers = [(goal_w/2, -table_l/2),(-goal_w/2, table_l/2),\
                   (goal_w/2, table_l/2),(-goal_w/2, -table_l/2)]
        for c in centers:
            t0s.append(self.circle_intersection_time(pos, vel, c, r, 'first'))

        return np.nanmin(np.array(t0s, dtype=np.float))
    
    def striker_collision_time(self, striker, time):
        """
        If puck is on collision course w/ striker return expected collision time
        else return None
        """
        striker_r = self.consts['striker_radius']
        r = self.consts['puck_radius']
        
        p_v = self.state['puck']['velocity']
        p_p = self.state['puck']['position']
        s_v = self.state[striker]['velocity']
        s_p = self.state[striker]['position']
        
        vel = p_v-s_v
        pos = p_p-s_p
        t0 = self.circle_intersection_time(pos, vel, (0,0), striker_r+r, 'first')
        return t0
    
    def bounce(self, vel, n, f):
        """
        Return velocity after bounce
        """
        n1 = n * np.linalg.norm(vel*n)
        n2 = vel + n1
        v = n1*f + n2
        return v
    
    def circle_norm(self, center, pos, order):
        """
        Return vector normal to circle at bounce point
        """
        ab = pos-np.array(center)
        h = np.linalg.norm(ab)
        u = np.array([ab[0]/h,ab[1]/h])
        if order=='first':
            return u
        elif order=='second':
            return -u
    
    def bumper_collision(self):
        """
        Update puck velocity after bumper collision
        """
        f = self.consts['table_bounce']
        table_w = self.consts['table_width']
        table_l  = self.consts['table_length']
        goal_w = self.consts['table_goal_width']
        table_r = self.consts['table_corner_radius']
        r = self.consts['puck_radius']
            
        v0 = self.state['puck']['velocity']
        p0 = self.state['puck']['position']
        
        # side bumpers
        if abs(p0[0]-(table_w/2-r))<1e-5:
            n = np.array([-1,0])
        elif abs(p0[0]+(table_w/2-r))<1e-5:
            n = np.array([1,0])
            
        # end bumpers
        if abs(p0[1]-(table_l/2-r))<1e-5:
            n = np.array([0,-1])
        elif abs(p0[1]+(table_l/2-r))<1e-5:
            n = np.array([0,1])
            
        # table corners
        centers = [(table_w/2-table_r, -table_l/2+table_r),(-table_w/2+table_r, table_l/2-table_r),\
                   (table_w/2-table_r, table_l/2-table_r),(-table_w/2+table_r, -table_l/2+table_r)]
        for c in centers:
            if table_r>r and abs(np.linalg.norm(p0-c)-(table_r-r))<1e-5:
                n = self.circle_norm(c, p0, 'second')
        
        # end corners
        centers = [(goal_w/2, -table_l/2),(-goal_w/2, table_l/2),\
                   (goal_w/2, table_l/2),(-goal_w/2, -table_l/2)]
        for c in centers:
            if abs(np.linalg.norm(p0-c)-r)<1e-5:
                n = self.circle_norm(c, p0, 'first')
        
        v = self.bounce(v0, n, f)
        self.state['puck']['velocity'] = v
    
    def striker_collision(self, striker):
        """
        Update puck velocity after striker collision
        """
        f = self.consts['striker_bounce']
        striker_r = self.consts['striker_radius']
        r = self.consts['puck_radius']
            
        p_v = self.state['puck']['velocity']
        p_p = self.state['puck']['position']
        s_v = self.state[striker]['velocity']
        s_p = self.state[striker]['position']
        
        v0 = p_v-s_v
        n = self.circle_norm(s_p, p_p, 'first')
        
        vp = self.bounce(v0, n, f) + s_v
        vs = 0.2*self.bounce(-v0, -n, (1-f)) + s_v
        self.state['puck']['velocity'] = vp
        self.state[striker]['velocity'] = vs
        
    def render(self):
        """
        return np array image game repr
        """
        def fill_circle(img, loc, r):
            w, h = img.shape[1], img.shape[0]
            for y in range(loc[0]-r, loc[0]+r+1):
                for x in range(loc[1]-r, loc[1]+r+1):
                    d = np.sqrt((y-loc[0])**2+(x-loc[1])**2)
                    if 0<=x<w and 0<=y<h and r>=d:
                        img[y,x] = 0
            return img
            
        scale = 100
        w = int(self.consts['table_width']*scale)
        l = int(self.consts['table_length']*scale)
        shift = np.array([w/2,l/2])
        p = (self.state['puck']['position']*scale + shift).astype(int)
        s1 = (self.state['striker1']['position']*scale + shift).astype(int)
        s2 = (self.state['striker2']['position']*scale + shift).astype(int)
        p_r = int(self.consts['puck_radius']*scale)
        s_r = int(self.consts['striker_radius']*scale)
        
        img = np.ones((w,l,3))
        img[:,:,0] = fill_circle(img[:,:,0], p, p_r)
        img[:,:,1] = fill_circle(img[:,:,1], p, p_r)
        img[:,:,2] = fill_circle(img[:,:,2], p, p_r)
        img[:,:,1] = fill_circle(img[:,:,1], s1, s_r)
        img[:,:,1] = fill_circle(img[:,:,1], s2, s_r)
        img[:,:,2] = fill_circle(img[:,:,2], s1, s_r)
        img[:,:,2] = fill_circle(img[:,:,2], s2, s_r)
        return (255*img).astype(np.float32)
        
class AirHockey(HockeyPhysics):
    """
    OpenAI gym style air hockey env to simulate playing from both sides of the table
    """
    def __init__(self, action_space, freq=0.1, max_length=60):
        HockeyPhysics.__init__(self)
        self.action_space = action_space
        self.freq = freq
        self.max_length = max_length
        self.side_penalty = 0.05
        self.reset()
    
    def get_observation(self):
        """
        observation : [pp, pv, s1p, s1v, s2p, s2v]
        :return: striker1 observation
        """
        p_p = self.state['puck']['position']
        p_v = self.state['puck']['velocity']
        s1_p = self.state['striker1']['position']
        s1_v = self.state['striker1']['velocity']
        s2_p = self.state['striker2']['position']
        s2_v = self.state['striker2']['velocity']
        return np.array([p_p,p_v,s1_p,s1_v,s2_p,s2_v]).flatten()
    
    def reset(self):
        """
        reset env state
        :return: observation1, observation2
        """
        self.time = 0
        striker_y = -self.consts['table_length']/2+np.random.uniform(0.1, 0.3)
        striker_x = np.random.uniform(-0.1,0.1)
        puck_vx = np.random.uniform(-0.5,0.5)
        puck_vy = np.random.uniform(0.1, 0.4)
        puck_x = np.random.uniform(-0.2,0.2)
        self.state['puck']['position'] = np.array([puck_x, 0])
        self.state['puck']['velocity'] = np.array([puck_vx,puck_vy])
        self.state['striker1']['position'] = np.array([striker_x,striker_y])
        self.state['striker1']['velocity'] = np.zeros(2)
        self.state['striker2']['position'] = np.array([-striker_x,-striker_y])
        self.state['striker2']['velocity'] = np.zeros(2)
        
        ob1 = self.get_observation()
        ob2 = -1 * ob1
        return ob1, ob2
        
    def step(self, s1_action, s2_action):
        """
        Take action and step forward in time
        :return: observ1, reward1, observ2, reward2, done
        """
        v0 = self.state['striker1']['velocity']
        p = self.state['striker1']['position']
        v = self.action_space.execute(s1_action, v0, p)
        self.state['striker1']['velocity'] = v
        
        v0 = -self.state['striker2']['velocity']
        p = -self.state['striker2']['position']
        v = -self.action_space.execute(s2_action, v0, p)
        self.state['striker2']['velocity'] = v
        
        side = np.sign(self.state['puck']['position'][1])
        sp = self.side_penalty
        
        self.time += self.freq
        self.simulate(self.freq)
        
        o = self.get_observation()
        r = self.is_goal() + side*sp
        d = self.is_goal() or (self.time>self.max_length)
        
        return o, r, -o, -r, d