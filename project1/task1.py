###############
##Design the function "findRotMat" to  return 
# 1) rotMat1: a 2D numpy array which indicates the rotation matrix from xyz to XYZ 
# 2) rotMat2: a 2D numpy array which indicates the rotation matrix from XYZ to xyz 
#It is ok to add other functions if you need
###############

import numpy as np
import cv2

def findRotMat(alpha, beta, gamma):
    #......

    a=np.radians(alpha)
    b=np.radians(beta)
    c=np.radians(gamma)

    rz= np.array(((np.cos(a), -np.sin(a), 0),
                (np.sin(a), np.cos(a), 0),
                (0,0,1)))
    #print(rz)

    rx= np.array(((1,0,0),
                (0, np.cos(b), -np.sin(b)),
                (0, np.sin(b), np.cos(b))))
    #print(rx)

    ry= np.array(((np.cos(c), -np.sin(c), 0),
                (np.sin(c), np.cos(c), 0),
                (0,0,1)))
    #print(rz1)

    s=np.matmul (ry, rx)
    #print(s)
    
    p=np.matmul (s,rz)
    #print(p)

    p1=p.transpose()
    #print(p1)
    return p,p1

if __name__ == "__main__":
    alpha = 45
    beta = 30
    gamma = 60
    rotMat1, rotMat2= findRotMat(alpha, beta, gamma)
    print(rotMat1)
    print(rotMat2)









