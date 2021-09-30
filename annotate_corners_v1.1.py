import copy
import numpy as np
import cv2
from copy import deepcopy
import os
import glob
import sys


def annotate_corners(videopath):
    global pointIndex
    global img
    pointIndex = 0

    cam = cv2.VideoCapture(videopath)

    success,img = cam.read()


    frame_id = 0
    final_frame = None
    image_buffer = [img]
    cv2.imshow('image',img)
    while success:
        frame_id += 1
        
        success,img = cam.read()
        image_buffer.append(deepcopy(img))
    i=1
    while (i<len(image_buffer)-1):
        img = image_buffer[i]

        key = cv2.waitKey()
        
        if key == 3:
            # print(len(image_buffer), key)
            final_frame = image_buffer[i]
            cv2.imshow('image',img)
            i+=1
        elif key == 2:
            i-=1
            # img1 = deepcopy(image_buffer.pop())
            if i<0:
                i=0
            img1 = image_buffer[i]
            cv2.imshow('image',img1)
        elif key == ord('s'):
            final_frame = image_buffer[i-1]
            break
        elif key == 27: #esc key:
            quit()
        
    img = final_frame

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (200, 50)
    
    # fontScale
    fontScale = 1
    
    color = (0, 255, 0)
    
    thickness = 2
    x_offset=y_offset=60
    text_img = deepcopy(img)
    s_img = cv2.imread("mock_cabinet.png")

    text_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img


    def draw_circle(event,x,y,flags,param):
        global img
        global pointIndex
        global pts

        if event == cv2.EVENT_LBUTTONDOWN:

            cv2.circle(img,(x,y),3,(0,0,255),-1)
            cv2.putText(img,str(pointIndex),(x,y),font,1,(0,255,0),1,cv2.LINE_AA)
            pts[pointIndex] = (x,y)
            pointIndex = pointIndex + 1

    def selectnPoints():
        global img
        global pointIndex
        global pts
        # get number of points as input
        # nPoints = input("Enter number of visible points: ")
        nPoints = 8
        pts = [(0,0) for i in range(nPoints)]
        print ("Annotate visible points, by clicking on each of them")
        # pointIndex = 0	
        while(pointIndex != nPoints):
            local_img = copy.deepcopy(img)
            img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
            local_img = cv2.putText(local_img, f'Annotate {pointIndex}', org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('image',local_img)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                return False
            elif key == 32:
                pts[pointIndex] = (999,999)
                pointIndex = pointIndex + 1
                continue

        cv2.imshow('image',img)
        cv2.waitKey(500)
        return True

    def get_points_from_csv(filepath):
        filepath = os.path.dirname(filepath)
        # print(filepath)
        list_of_files = glob.glob(f'{filepath}/*.csv') # * means all if need specific format then *.csv
        if len(list_of_files) == 0:
            print("No csv files found")
            return False
        latest_file = max(list_of_files, key=os.path.getctime)
        "load points from csv"
        # global pts
        pts = []
        with open(latest_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split(',')
                pts.append((int(line[0]), int(line[1])))
        return pts
    
    def plot_points(img,pts):

        "plot points"
        for i in range(len(pts)):
            cv2.circle(img,pts[i],3,(0,0,255),-1)
            cv2.putText(img,str(i),pts[i],font,1,(0,255,0),1,cv2.LINE_AA)

    pts = get_points_from_csv(videopath)
    if pts:
        plot_points(text_img,pts)

    image = cv2.putText(text_img, 'Save previous annotations?', org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('image',image)
    key = cv2.waitKey()
    if key == ord('y'):
        filename = videopath.split('.')[0] + '_corners.csv'
        np.savetxt(filename, pts, fmt='%d', delimiter=',')
        cv2.destroyAllWindows()
    else:

        cv2.namedWindow('image')
        cv2.setMouseCallback('image',draw_circle)

        if (selectnPoints()):
            # save pts to csv
            filename = videopath.split('.')[0] + '_corners.csv'
            np.savetxt(filename, pts, fmt='%d', delimiter=',')

        
# annotate_corners("117.mp4")

if __name__ == "__main__":
    session = sys.argv[1]
    for item in os.listdir(f"{session}/right"):
        if item.endswith(".mp4"):
            annotate_corners(f"{session}/right/"+item)

