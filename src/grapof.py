# -*- coding: utf-8 -*-
import cv2
import numpy as np
from cv_bridge import CvBridge
import flapnet
from numpy.linalg import inv


class ImageProcessing:
    # Extract coordinates of a line given two points
    def bresenham_line(self, x0, y0, x1, y1):
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0  
            x1, y1 = y1, x1
        
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        
        if y0 < y1: 
            ystep = 1
        else:
            ystep = -1
        
        deltax = x1 - x0
        deltay = abs(y1 - y0)
        error = -deltax / 2
        y = y0
        
        line = []    
        for x in range(x0, x1 + 1):
            if steep:
                line.append((y,x))
            else:
                line.append((x,y))
        
            error = error + deltay
            if error > 0:
                y = y + ystep
                error = error - deltax
        return line
    
    # Finds closest white point to target point
    def find_nearest_white(self, img, target):
        nonzero = cv2.findNonZero(img)
        distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
        nearest_index = np.argmin(distances)
        return nonzero[nearest_index]
    
    # Finds first white point of a line in a binary image
    def find_first_border_point(self, img, line):
        if img[line[0][1],line[0][0]] == 0:
            for i in range(len(line)):
                if img[line[i][1],line[i][0]] > 204:
                    return (line[i][0], line[i][1])
        else:
            for i in reversed(range(len(line))):
                if img[line[i][1],line[i][0]] > 204:
                    return (line[i][0], line[i][1])
                
    # Find a a single centroid in region
    def find_single_centroid(self, img):
        M = cv2.moments(img)
        if M['m00'] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return cX, cY
    
    # Detects multiple centroids in the image and return them into an array
    def find_multiple_centroids(self, img):
        centres = []
        contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
          moments = cv2.moments(contours[i])
          if moments['m00'] != 0:
            centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
          
        return centres
    
    # Detect grasping points given
    def find_grasping_points(self, I_bin, centres, cX,cY,):
        gp = []
        for i in range(len(centres)):
            line = self.bresenham_line(centres[i][0], centres[i][1], cX, cY)
            if self.find_first_border_point(I_bin,line) != None:
                gp.append(self.find_first_border_point(I_bin, line))
        return gp

    # Clean binary disparity map
    def clean_disparity_map(self, I_bin, size_th=2000):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(I_bin, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        clean_img = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= size_th:
                clean_img[output == i + 1] = 255
        return clean_img

    # Prints centroids of cordinates <centres> on image I
    def print_tissue_centroids(self, I, centres):
        multi_cent = I
        for i in range(len(centres)):
            multi_cent = cv2.circle(multi_cent, (centres[i][0], centres[i][1]), 2, (200, 150, 90), 5)
        
        return multi_cent
    
    # Prints grasping points of coordinates <gp> on image I
    def print_grasping_points(self, I, gp):
        multi_gp = I
        for i in range(len(gp)):
            multi_gp = cv2.circle(multi_gp, (gp[i][0], gp[i][1]), 2, (0, 255, 0), 5)
        
        return multi_gp
    
    # Prints single background centroid
    def print_background_centroid(self, img, cX, cY):
        disp_centroid = cv2.circle(img, (cX, cY), 2, (0, 0, 255), 5)
        return disp_centroid


class Geometry:
    def __init__(self):
        return

    # Estimate distance from camera of point
    def estimate_distance(self, focal_len, baseline, disp):
        return (focal_len * baseline)/disp

    def project_on_canvas(self, point, offset_x, offset_y):
        return point[0] + offset_x, point[1] + offset_y

    def pix2word(self, disp_mat, cam_mat, foc_len, baseline, pj_point):
        disp = disp_mat[pj_point[1], pj_point[0]]
        gp_Z = self.estimate_distance(foc_len, baseline, disp)
        img_gp_coord = (pj_point[1], pj_point[0], 1)
        cam_mat = np.asarray(cam_mat).reshape((3,4))
        world_gp_coord = (np.matmul(inv(cam_mat[:, 0:3]), img_gp_coord)) * gp_Z
        return world_gp_coord


class TopicsSubscription:
    def __init__(self, nn_target_size=(64,64), crop_window=((55, 521), (159, 665))):
        self.disp_x_offset = 0
        self.disp_y_offset = 0
        self.disp_mat = np.zeros(1)
        self.foc_len = 0
        self.baseline = 0
        self.bridge = CvBridge()
        self.fn_preproc = flapnet.Preprocessing()
        self.camera_mat = np.zeros((3, 4), dtype=float)
        self.cam_disp = np.zeros(1)
        self.target_size = nn_target_size
        self.crop_win = crop_window

    def disp_callback(self, disp_msg):
        self.disp_mat = self.bridge.imgmsg_to_cv2(disp_msg.image)
        self.foc_len = disp_msg.f
        self.baseline = disp_msg.T
        self.disp_x_offset = disp_msg.valid_window.x_offset
        self.disp_y_offset = disp_msg.valid_window.y_offset

    def caminfo_callback(self, cam_info_msg):
        self.camera_mat = cam_info_msg.P

    def img_callback(self, image_msg):
        # convert image to a compatible format
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        cv_image = cv_image[self.crop_win[0][0]:self.crop_win[0][1], self.crop_win[1][0]:self.crop_win[1][1]]
        self.cam_disp = self.fn_preproc.image_preproc(cv_image, self.target_size)