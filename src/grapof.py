# -*- coding: utf-8 -*-
import cv2
import numpy as np


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
                if img[line[i][0],line[i][1]] > 50:
                    return line[i]
        else:
            for i in reversed(range(len(line))):
                if img[line[i][1],line[i][0]] > 50:
                    return line[i]
                
    # Find a a single centroid in region
    def find_single_centroid(self, img):
        M = cv2.moments(img)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    
    # Detects multiple centroids in the image and return them into an array
    def find_multiple_centroids(self, img):
        centres = []
        _, contours, _= cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
          moments = cv2.moments(contours[i])
          centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
          
        return centres
    
    # Detect grasping points given
    def find_grasping_points(self, I_bin, centres, cX,cY,):
        gp = []
        for i in range(len(centres)):
            line = self.bresenham_line(centres[i][0], centres[i][1], cX, cY)
            gp.append(self.find_first_border_point(I_bin, line))
        return gp

    
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
    
    
    

    

        