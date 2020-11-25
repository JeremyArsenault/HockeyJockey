import cv2
import numpy as np
import imutils

class Camera:
    """
    Class to handle all image processing
    """
    def __init__(self, device=0):
        # TABLE SIZE
        self.table_width = 1.5
        self.table_length = 2.7
        self.rescaled_dim = (150, 270)
        
        # IMPORTANT COLOR CODES:
        self.color_green = np.array([119,134,98]) 
        self.color_pink = np.array([140,40,150])
        self.color_orange = np.array([68,87,163])
        
        # connect to camera
        self.cam = cv2.VideoCapture(device)
        if not self.cam.isOpened():
            raise Exception('Failed to connect to webcam')
            
        # set corners
        self.set_corners()
        
    def color_mask(self, img, color, thresh=10, blur=3):
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
        M = cv2.getPerspectiveTransform(self.corners, dst)
        warped = cv2.warpPerspective(img, M, self.rescaled_dim)

        return warped

        
    def find_centroids(self, img, n=1):
        """
        Detect n largest contours of mask and find centers
        If no objects detected, return center of table
        :return: size sorted centriod list
        """
        # Find contours
        contours = cv2.findContours(np.uint8(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # Get centers and areas
        centers = []
        areas = []
        for c in contours:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append([cX,cY])
            areas.append(cv2.contourArea(c))
            
        # Make sure we have enough contours
        while len(areas)<n:
            centers.append([self.rescaled_dim[0]/2, self.rescaled_dim[1]/2])
            areas.append(0)
            
        # Find top n sorted contours
        sorted_centers = []
        for i in np.argsort(-1*np.array(areas))[:n]:
            sorted_centers.append(centers[i])
            
        return np.array(sorted_centers)


    def set_corners(self):
        """
        Get the corners of the table for perspective shifts
        """
        _, frame = self.cam.read()
        corner_mask = self.color_mask(frame, self.color_pink)
        pts = self.find_centroids(corner_mask, 4)
        
        # put pts in correct order
        corners = np.zeros((4, 2))
        s = pts.sum(axis = 1)
        corners[0] = pts[np.argmin(s)]
        corners[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        corners[1] = pts[np.argmin(diff)]
        corners[3] = pts[np.argmax(diff)]
        
        self.corners = corners
        
    def get_pos(self, loc):
        """
        convert location of centroid into absolute position in meters
        :return: position
        """
        pos_y = (self.table_length/self.rescaled_dim[1])*(loc[0]-self.rescaled_dim[1]/2)
        pos_x = (self.table_width/self.rescaled_dim[0])*(loc[1]-self.rescaled_dim[0]/2)
        return [pos_x, pos_y]
        
    def get_state(self):
        """
        Capture frame and find position state vector or everything on table
        :return: [puck pos, s1 pos, s2 pos]
        """
        _, frame = self.cam.read()
        
        frame = self.perspective_shift(frame)
        
        puck_mask = self.color_mask(frame, self.color_green)
        striker_mask = self.color_mask(frame, self.color_orange)
        
        puck_loc = self.find_centroids(puck_mask)
        striker_locs = self.find_centroids(striker_mask, 2)
        
        p_pos = get_pos(puck_loc[0])
        if striker_locs[0][1]<striker_locs[1][1]:
            s1_pos = get_pos(striker_locs[0])
            s2_pos = get_pos(striker_locs[1])
        else:
            s1_pos = get_pos(striker_locs[0])
            s2_pos = get_pos(striker_locs[1])            
        
        return [p_pos, s1_pos, s2_pos]