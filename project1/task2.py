###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
#It is ok to add other functions if you need
###############
import numpy as np
import cv2
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

def calibrate(imgname):
    #......
    world_coordinates=[[40, 0, 40], [40, 0, 30], [40, 0, 20], [40, 0, 10],
                       [30, 0, 40], [30, 0, 30], [30, 0, 20], [30, 0, 10],
                       [20, 0, 40], [20, 0, 30], [20, 0, 20], [20, 0, 10],
                       [10, 0, 40], [10, 0, 30], [10, 0, 20], [10, 0, 10],
                       [0, 0, 40],  [0, 0, 30],  [0, 0, 20],  [0, 0, 10],
                       [0, 10, 40], [0, 10, 30], [0, 10, 20], [0, 10, 10],
                       [0, 20, 40], [0, 20, 30], [0, 20, 20], [0, 20, 10],
                       [0, 30, 40], [0, 30, 30], [0, 30, 20], [0, 30, 10],
                       [0, 40, 40], [0, 40, 30], [0, 40, 20], [0, 40, 10]]

    #CHECKERBOARD = (5, 5)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    image=imread(imgname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners=cv2.findChessboardCorners(gray,(4,9),None)
    array1=[]
    if ret == True:
        #objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #imgpoints.append(corners2)

        # Draw and display the corners
        image = cv2.drawChessboardCorners(image, (4,9), corners2, ret)

        #cv2.imshow('img', image)
        #cv2.waitKey(0)

    #world_coordinates = [[40,0,40],[40,0,30],[40,0,20],[40,0,10],[30,0,40],[30,0,30],[30,0,20],[30,0,10],[20,0,40],[20,0,30],[20,0,20],[20,0,10],[10,0,40],[10,0,30],[10,0,20],[10,0,10],[0,0,40],[0,0,30],[0,0,20],[0,0,10],[0,10,40],[0,10,30],[0,10,20],[0,10,10],[0,20,40],[0,20,30],[0,20,20],[0,20,10],[0,30,40],[0,30,30],[0,30,20],[0,30,10],[0,40,40],[0,40,30],[0,40,20],[0,40,10]]
    array1.append(corners2)
    #print(corners2.shape)
    matrix=[]
    cornersNew=array1[0].reshape(36,2)
    #print(cornersNew)
    for i in range(0,35):
        var1=[world_coordinates[i][0],world_coordinates[i][1], world_coordinates[i][2],1,0,0,0,0,-cornersNew[i][0]*world_coordinates[i][0],-cornersNew[i][0]*world_coordinates[i][1],-cornersNew[i][0]*world_coordinates[i][2],-cornersNew[i][0]]
        matrix.append(var1)
        var2=[0,0,0,0,world_coordinates[i][0],world_coordinates[i][1], world_coordinates[i][2],1,-cornersNew[i][1]*world_coordinates[i][0],-cornersNew[i][1]*world_coordinates[i][1],-cornersNew[i][1]*world_coordinates[i][2],-cornersNew[i][1]]
        matrix.append(var2)

    u, s, v = np.linalg.svd(matrix,full_matrices=True)
    x=v[11]

    #lambda_l=np.sqrt(1/(x[8]*x[8]+x[9]*x[9]+x[10]*x[10]))
    lambda_l = np.sqrt(1/np.sum([x[8] ** 2, x[9] ** 2, x[10] ** 2]))
    #print (lambda_l)
    m=lambda_l*x

    matrix1 = [m[0],m[1],m[2]]
    matrix2 = [m[4], m[5], m[6]]
    matrix3 = [m[8], m[9], m[10]]
    #print(matrix1)
    #matrix1 = np.matrix(matrix10)
    #matrix3 = np.matrix(matrix30)
    #matrix2=np.matrix(matrix20)3x1 1x3
    #Ox=np.matmul(matrix1.transpose(),matrix3)
    #Ox = matrix1[0] * matrix3[0] + matrix1[1] * matrix3[1] + matrix1[2] * matrix3[2]
    Ox = np.sum([np.dot(matrix1[0], matrix3[0]),np.dot(matrix1[1], matrix3[1]),np.dot(matrix1[2], matrix3[2])])
    Oy = matrix2[0] * matrix3[0] + matrix2[1] * matrix3[1] + matrix2[2] * matrix3[2]
    #Oy=np.matmul(matrix2.transpose(),matrix3)
    #print(Ox,Oy)
    #print(matrix1,matrix2,matrix3)
    #print(np.matmul(matrix1.transpose(),matrix1) - np.square(Ox))
    Fx=np.sqrt(matrix1[0] * matrix1[0] + matrix1[1] * matrix1[1] + matrix1[2] * matrix1[2] - np.square(Ox))
    Fy=np.sqrt(matrix2[0] * matrix2[0] + matrix2[1] * matrix2[1] + matrix2[2] * matrix2[2] - np.square(Oy))

    return ([Fx,Fy,Ox,Oy],True)

    
if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)