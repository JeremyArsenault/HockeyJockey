import cv2
import numpy as np
import imutils
import math
import time

class Camera:
    """
    Class to handle all image processing
    """
    def __init__(self, device=1):
        # TABLE SIZE
        self.table_width = 1
        self.table_length = 2
        self.rescaled_dim = (150, 270)
        
        # IMPORTANT COLOR CODES:
        self.color_green = np.array([97,130,90]) 
        self.color_pink = np.array([140,40,150])
        self.color_orange = np.array([9,66,149])
        
        # connect to camera
        self.cam = cv2.VideoCapture(device)
        if not self.cam.isOpened():
            raise Exception('Failed to connect to webcam')
            
        # set corners
        self.set_corners()
        
    def __del__(self):
        self.cam.release()
        
    def get_image(self):
        """
        get image from camera
        :return: raw image, perspective shifted image
        """
        self.flush_buffer()
        _, frame = self.cam.read()
        shift_frame = self.perspective_shift(frame)
        return frame, shift_frame
    
    def flush_buffer(self):
        """
        flush buffer and update clock
        open cv is annoying
        """
        t1 = time.time()
        while True:
            t2 = time.time()
            if t2-t1>0.03:
                break
            t1 = t2
            self.cam.read()
        
    def color_mask(self, img, color, thresh=20, blur=3):
        """
        Mask image according to color
        :return: mask (0:other colors, 1:selected color)
        """
        dist = np.linalg.norm(img-color, axis=2)
        mask = dist <= (np.min(dist) + thresh)
        smooth_mask = cv2.blur(np.uint8(mask), (blur,blur))
        return np.float32(smooth_mask)
    
    def perspective_shift(self, img):
        """
        Warp and resize image into top down to-scale view of table
        :return: warped image
        """
        dst = np.array([[0, 0],
                        [self.rescaled_dim[0]-1, 0],
                        [self.rescaled_dim[0]-1, self.rescaled_dim[1]-1],
                        [0, self.rescaled_dim[1]-1]])
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(np.float32(self.corners), np.float32(dst))
        warped = cv2.warpPerspective(img, M, self.rescaled_dim)

        return warped

        
    def find_centroids(self, img, n=1):
        """
        Detect n largest contours of mask and find centers
        If no objects detected, return None
        :return: size sorted centriod list, True if all objects were detected
        """
        # Find contours
        contours = cv2.findContours(np.uint8(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # Get centers and areas
        centers = []
        areas = []
        for c in contours:
            M = cv2.moments(c)
            cX = int(M["m10"] / max(M["m00"],1))
            cY = int(M["m01"] / max(M["m00"],1))
            centers.append([cX,cY])
            areas.append(cv2.contourArea(c))
            
        # Make sure we have enough contours
        detected=True
        while len(areas)<n:
            centers.append(None)
            areas.append(0)
            detected=False
            
        # Find top n sorted contours
        sorted_centers = []
        for i in np.argsort(-1*np.array(areas))[:n]:
            sorted_centers.append(centers[i])
            
        return np.array(sorted_centers), detected


    def set_corners(self):
        """
        Get the corners of the table for perspective shifts
        """
        self.flush_buffer()
        _, frame = self.cam.read()
        corner_mask = self.color_mask(frame, self.color_pink, thresh=30, blur=5)
        pts, detected = self.find_centroids(corner_mask, 4)
        if not detected:
            raise Exception("Failed to detect all four table corners")
            
        # put pts in correct order
        corners = np.zeros((4, 2))
        s = pts.sum(axis = 1)
        corners[0] = pts[np.argmin(s)]
        corners[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        corners[1] = pts[np.argmin(diff)]
        corners[3] = pts[np.argmax(diff)]
        
        self.corners = corners
        
    def abs_to_meter(self, loc):
        """
        convert location of centroid into absolute position in meters
        :return: position
        """
        if loc is None:
            return None
        pos_y = -(self.table_length/self.rescaled_dim[1])*(loc[1]-self.rescaled_dim[1]/2)
        pos_x = (self.table_width/self.rescaled_dim[0])*(loc[0]-self.rescaled_dim[0]/2)
        return [pos_x, pos_y]
    
    def get_pos(self, frame):
        """
        find position state vector or everything on table
        pos (x,y) / None
        :return: [puck pos, s1 pos, s2 pos]
        """
        frame = self.perspective_shift(frame)
        
        puck_mask = self.color_mask(frame, self.color_green, thresh=13)
        striker_mask = self.color_mask(frame, self.color_orange, thresh=25, blur = 5)
        
        puck_loc, _ = self.find_centroids(puck_mask)
        striker_locs, _ = self.find_centroids(striker_mask, 2)
        
        p_pos = self.abs_to_meter(puck_loc[0])
        # cases: (pos,pos), (pos,None), (None,None)
        if striker_locs[0] is not None:
            pos_1 = self.abs_to_meter(striker_locs[0])
            pos_2 = self.abs_to_meter(striker_locs[1])
            s1_pos, s2_pos = pos_1, pos_2 if pos_1[1]<0 else pos_2, pos_1
        else:
            s1_pos, s2_pos = None, None          
        
        return [p_pos, s1_pos, s2_pos]
        
    def get_state(self, time, frames=4):
        """
        Capture frame and find state vector of everything on the table.
        :frames: number of frames captured to determine velocity
        :return: [puck pos, s1 pos, s2 pos]
                 returns zeros if no history was recoverable 
        """
        if frames<2:
            raise ValueError('Needs at least 2 frames to determine velocity')
        self.flush_buffer()
        start_t = time.time()
        
        # time this to make sure we aren't blocking on get_pos for too long
        puck_history = []
        time_history = []
        p_pos, p_vel = [0,0], [0,0]
        s1_pos, s2_pos = [0,0], [0,0]
        for i in range(frames):
            _, frame = self.cam.read()
            t = time.time()-start_t
            p = self.get_pos(frame)
            
            if p[0] is not None:
                puck_history.append(p[0])
                time_history.append(t)
            # choose last nonzero striker locations
            if p[1] is not None:
                s1_pos = p[1]
            if p[2] is not None:
                s2_pos = p[2]
            
        # estimate puck position at current time
        if len(puck_history)==1:
            p_pos = puck_history[0]
        else:
            # do linear regression
            a = np.array([[t,1] for t in time_history])
            b = np.array(puck_history)
            m = np.linalg.lstsq(a, b, rcond=None)[0]
            
            t = np.array([[time.time()-start_t, 1]])
            p_pos = np.dot(t,m)[0]
            p_vel = m[:,0]
            
        return np.array([p_pos, p_vel, s1_pos, s2_pos])

        
        