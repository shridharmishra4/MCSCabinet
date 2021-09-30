from collections import defaultdict
import pickle
import cv2
import os
import numpy as np
import torch
import copy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import scipy.linalg as lin
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import shapely
from shapely.geometry import Polygon
from glob import glob

x_dim, y_dim, z_dim = 8.125/11.0, 6.0/11.0, 1.0

# load points from the csv file
def load_points_from_file(file_name):
    data = {}


    x_dim, y_dim, z_dim = 8.125/11.0, 6.0/11.0, 1.0
    with open(file_name, 'r') as f:
        for i,line in enumerate(f):
            line = line.strip()
            if line == "": continue
            x, y = line.split(",")
            data[i] = (int(x),int(y))
    d = {(0.0,0.0,0.0):data[3],
            (x_dim,0.0,0.0):data[2],
            (0,y_dim,0.0): [],
            (0.0,0.0,z_dim):data[0],
            (x_dim,y_dim,0.0):data[6],
            (0.0,y_dim,z_dim): [],
            (x_dim,0.0,z_dim): data[1],
            (x_dim,y_dim,z_dim):data[5]}

    side_box = [data[0], data[1], data[2], data[3]]
    return d,side_box
    
def load_bbox_from_file(file_name):
    data = {}

    with open(file_name, 'r') as f:
        for i,line in enumerate(f):
            line = line.strip()
            if line == "": continue
            x, y = line.split(",")
            data[i] = (x,y)
    return data

    

# DATA = {
#         0 : {(0.0,0.0,0.0):      [(332,434), (160,50)], # just use first column
#                 (x_dim,0.0,0.0):    [(509,459), (350, 44)],
#                 (0,y_dim,0.0): [],
#                 (0.0,0.0,z_dim):    [(331,128), (168, 17) ],
#                 (x_dim,y_dim,0.0):  [(639,450) , (250, -27)],
#                 (0.0,y_dim,z_dim): [],   
#                 (x_dim,0.0,z_dim):  [(509,110), (380, 12)],
#                 (x_dim,y_dim,z_dim):[(642,117), (379, -62)]},

#         1 : {(0.0,0.0,0.0):      [(324,394), (332,434)], 
#                 (x_dim,0.0,0.0):    [(500,416), (509,459)],
#                 #(0,y_dim,0.0):     [(257, -33), (463, 425)],
#                 (0,y_dim,0.0): [],
#                 (0.0,0.0,z_dim):    [(347, 80), (331,128)],
#                 (x_dim,y_dim,0.0):  [(627,422), (639,450)],
#                 #(0.0,y_dim,z_dim): [(174, -64), (461,129)], 
#                 (0.0,y_dim,z_dim): [],   
#                 (x_dim,0.0,z_dim):  [(529, 52), (509,110)],
#                 (x_dim,y_dim,z_dim):[(657, 85), (642,117)]},

#         2 : {(0.0,0.0,0.0):      [(339, 424), (332,434)], 
#                 (x_dim,0.0,0.0):    [(515, 460), (509,459)],
#                 #(0,y_dim,0.0):     [(257, -33), (463, 425)],
#                 (0,y_dim,0.0): [],
#                 (0.0,0.0,z_dim):    [(348, 129), (331,128)],
#                 (x_dim,y_dim,0.0):  [(641, 451), (639,450)],
#                 #(0.0,y_dim,z_dim): [(174, -64), (461,129)], 
#                 (0.0,y_dim,z_dim): [],   
#                 (x_dim,0.0,z_dim):  [(524, 110), (509,110)],
#                 (x_dim,y_dim,z_dim):[(655, 130), (642,117)]},

#         3 : {(0.0,0.0,0.0):      [(343,414), (332,434)], 
#                 (x_dim,0.0,0.0):    [(521, 433), (509,459)],
#                 #(0,y_dim,0.0):     [(257, -33), (463, 425)],
#                 (0,y_dim,0.0): [],
#                 (0.0,0.0,z_dim):    [(334, 96), (331,128)],
#                 (x_dim,y_dim,0.0):  [(644, 407), (639,450)],
#                 #(0.0,y_dim,z_dim): [(174, -64), (461,129)], 
#                 (0.0,y_dim,z_dim): [],   
#                 (x_dim,0.0,z_dim):  [(503, 70), (509,110)],
#                 (x_dim,y_dim,z_dim):[(635, 70), (642,117)]}
#         }

# TOP_BOX = { 0:[162, 11, 377, 59], 
#             1:[163, 0, 370, 55],
#             2:[160, 0, 450, 67],
#             3:[162, 6, 375, 35]}

# SIDE_BOX = {0:[331,110, 509,459], 
#             1:[324, 52, 529, 416],
#             2:[339, 110, 524, 460],
#             3:[334, 70, 521, 433]}

def read_track(top, front):
    # reading the tensors
    # top
    # "/nfs/hpc/share/mishrash/Projects/hand_tracking_v1/videos/nik/tracking_outputs/151953-Hidden_AffS_117_top_states.pkl"
    # "/nfs/hpc/share/mishrash/Projects/hand_tracking_v1/videos/nik/tracking_outputs/151953-Hidden_AffS_117_front_states.pkl"
    
    top_data = {}
    for t in os.listdir(top):
        if t.endswith("states.pkl"):
            name = t.split("_")
            start, end = int(name[3]), int(name[4])

            d = torch.load(os.path.join(top, t), map_location=torch.device('cpu'))
            top_data[(start, end)] = d

    front_data = {}
    for t in os.listdir(front):
        if t.endswith("states.pkl"):
            name = t.split("_")
            start, end = int(name[3]), int(name[4])
            d = torch.load(os.path.join(front, t), map_location=torch.device('cpu'))
            front_data[(start, end)] = d
    
    print(top_data.keys(), front_data.keys())
    return top_data, front_data

def read_points_bounding_boxes(top, front):
    bbox_points = {}
    front_points = {}
    side_points = {}

    # files  = glob(top+"/*bbox.csv")
    # print("files engind with box",files)

    # for t in os.listdir(top):
    #     if t.endswith("bbox.csv"):
    #         bbox_points[t] = load_bbox_from_file(os.path.join(top, t))
    # for i,t in enumerate(os.listdir(front)):
    #     if t.endswith("corners.csv"):
    #         front_points[i],side_points[i] = load_points_from_file(os.path.join(front, t))

    for i,t in enumerate(glob(top+"/*bbox.csv")):
        bbox_points[i] = load_bbox_from_file(t)
    for i,t in enumerate(glob(front+"/*corners.csv")):
        front_points[i],side_points[i] = load_points_from_file(t)
    
    return bbox_points, front_points,side_points


"""def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
 
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi"""

def drawCube(ax, Z):

    ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2], c='r')
    x, y, z = Z[0, 0], Z[0, 1], Z[0, 2]
    label = '(%.1f, %.1f, %.1f)' % (x, y, z)
    ax.text(x, y, z, label)
    x, y, z = Z[-1, 0], Z[-1, 1], Z[-1, 2]
    label = '(%.1f, %.1f, %.1f)' % (x, y, z)
    ax.text(x, y, z, label)

    verts = [[Z[0],Z[1],Z[6],Z[3]],
    [Z[0],Z[2],Z[5],Z[3]], 
    [Z[2],Z[4],Z[7],Z[5]], 
    [Z[1],Z[4],Z[7],Z[6]], 
    [Z[0],Z[1],Z[4],Z[2]],
    [Z[3],Z[6],Z[7],Z[5]]]

    # plot sides
    ax.add_collection3d(Poly3DCollection(verts, 
        facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

def check_parameters(K, dist, R, T, p_gt, p3d_gt):
    tot_error=0
    total_points=0
    for i in range(len(p3d_gt)):
        reprojected_points, _ = cv2.projectPoints(p3d_gt[i], R, T, K, dist)
        reprojected_points=reprojected_points.reshape(-1,2)
        tot_error+=np.sum(np.abs(p_gt[i]-reprojected_points)**2)
        total_points+=len(p3d_gt[i])

    mean_error=np.sqrt(tot_error/total_points)
    print("Mean reprojection error: ", mean_error)

    return

def get_hand_pose(data, handpose,  ratio_x=1,  ratio_y=1):
    if handpose is None or type(handpose) != dict: 
        return []
    min_x, min_y, max_x, max_y = None, None, None, None
    for _, p2d in data.items():
        if len(p2d) == 0: continue
        p = p2d
        x, y = p[0], p[1]
        if min_x is None:
            min_x, min_y = x, y
            max_x, max_y = x, y
        if x < min_x: min_x = x
        if x > max_x: max_x = x
        if y < min_y: min_y = y
        if y > max_y: max_y = y

    positions = []
    for tip in ["THUMB_TIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_TIP", "RING_FINGER_TIP", "PINKY_TIP"]:
        if tip not in handpose: continue
        if handpose[tip] is None: continue
        x, y = handpose[tip]
        x *= ratio_x
        y *= ratio_y
        if y < min_y or y > max_y: continue  
        if x < min_x or x > max_x: continue
        positions.append([int(x),int(y)])
    return positions

def swap_hand(handpose_right, handpose_left, front_right, front_left):
    if front_left is None and front_right is None:
        return handpose_right, handpose_left
    if len(handpose_left) == 0 and len(handpose_right) == 0:
        return handpose_right, handpose_left
    if front_left is not None :
        xcenter_l, ycenter_l = (front_left[2] + front_left[0])/2.0, (front_left[3] + front_left[1])/2.0 
        if len(handpose_left) == 0 and len(handpose_right) > 0 and front_right is None: 
            return handpose_left, handpose_right
        if len(handpose_right) == 0 and len(handpose_left) > 0 and front_right is None: 
            return handpose_right, handpose_left
        if len(handpose_right) > 0 and len(handpose_left) > 0 and front_right is None:
            x_l, y_l = handpose_left[0]
            x_r, y_r = handpose_right[0]
            drl = np.sqrt((x_r - xcenter_l)**2 + (y_r - ycenter_l)**2)
            dll = np.sqrt((x_l - xcenter_l)**2 + (y_l - ycenter_l)**2)
            if drl < dll: return handpose_left, handpose_right
            return handpose_right, handpose_left
        xcenter_r, ycenter_r = (front_right[2] + front_right[0])/2.0, (front_right[3] + front_right[1])/2.0
        if len(handpose_right) > 0 and len(handpose_left) == 0:
            x_r, y_r = handpose_right[0]
            drr = np.sqrt((x_r - xcenter_r)**2 + (y_r - ycenter_r)**2)
            drl = np.sqrt((x_r - xcenter_l)**2 + (y_r - ycenter_l)**2)
            if drl < drr: return handpose_left, handpose_right
            return handpose_right, handpose_left
        if len(handpose_right) == 0 and len(handpose_left) > 0:
            x_l, y_l = handpose_left[0]
            dlr = np.sqrt((x_l - xcenter_r)**2 + (y_l - ycenter_r)**2)
            dll = np.sqrt((x_l - xcenter_l)**2 + (y_l - ycenter_l)**2)
            if dlr < dll: return handpose_left, handpose_right
            return handpose_right, handpose_left
        x_l, y_l = handpose_left[0]
        x_r, y_r = handpose_right[0]
        drr = np.sqrt((x_r - xcenter_r)**2 + (y_r - ycenter_r)**2)
        dlr = np.sqrt((x_l - xcenter_r)**2 + (y_l - ycenter_r)**2)
        drl = np.sqrt((x_r - xcenter_l)**2 + (y_r - ycenter_l)**2)
        dll = np.sqrt((x_l - xcenter_l)**2 + (y_l - ycenter_l)**2)
        if dlr < drr or drl < dll: return handpose_left, handpose_right
    else:
        if len(handpose_left) == 0 and len(handpose_right) > 0: return handpose_right, handpose_left
        if len(handpose_right) == 0 and len(handpose_left) > 0: return handpose_left , handpose_right   
        x_l, y_l = handpose_left[0]
        x_r, y_r = handpose_right[0]
        xcenter_r, ycenter_r = (front_right[2] + front_right[0])/2.0, (front_right[3] + front_right[1])/2.0
        drr = np.sqrt((x_r - xcenter_r)**2 + (y_r - ycenter_r)**2)
        dlr = np.sqrt((x_l - xcenter_r)**2 + (y_l - ycenter_r)**2)
        if dlr < drr: return handpose_left, handpose_right

    return handpose_right, handpose_left

def get_track(video, top_data, front_data, save = True):
    
    name = video.split("/")[-1]
    name = name.split('.')[0]
    # print(name)

    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    h, w, _ = frame.shape
    frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    dirs = "output/"
    if not os.path.exists(dirs):
        os.mkdir(dirs)

    limit_x, limit_y = 360, 512

    ratio_y_front = h / 360.0
    ratio_x_front = (w - limit_y) / 512.0

    # print("top data keys",top_data.keys())
    # print("top data keys",top_data.keys())
    # print("front data keys",front_data.keys())



    inter_data = {}
    for i, frames in enumerate(top_data.keys()):
        # print(i,frames)
        top = top_data[frames]
        front = front_data[frames]
        intersection = []
        count = 0
        
        for count in range(frame_number):
            intersect_right, intersect_left = False, False
            #
            #front_right, front_left, top_right, top_left = None, None, None, None
            #if count not in top['right']: continue
            if count >= len(top['right']): break
            top_right = top['right'][count]
            if top_right is not None:
                top_right = top_right['xyxy']
            #
            top_left = top['left'][count]
            if top_left is not None:
                top_left = top_left['xyxy']
            #
            if front['right'][count] is not None:
                front_right = front['right'][count]
                handpose_right = get_hand_pose(DATA[i], front_right['hand_pose'], ratio_x_front, ratio_y_front)
            if front_right is not None: 
                # print(front_right)              
                front_right_ = front_right['xyxy']
                front_right_[0] *= ratio_x_front
                front_right_[1] *= ratio_y_front
                front_right_[2] *= ratio_x_front
                front_right_[3] *= ratio_y_front
                front_right_ = [int(x) for x in front_right_]
                if front_right_[2] > (w - limit_y) or front_right_[3] > h:
                    front_right_ = None

            #
            if front['left'][count] is not None:
                front_left = front['left'][count]
                handpose_left = get_hand_pose(DATA[i], front_left['hand_pose'], ratio_x_front, ratio_y_front)
            if front_left is not None:
                # print(front_left)
                front_left_ = front_left['xyxy']
                front_left_[0] *= ratio_x_front
                front_left_[1] *= ratio_y_front
                front_left_[2] *= ratio_x_front
                front_left_[3] *= ratio_y_front
                front_left_ = [int(x) for x in front_left_]
                if front_left_[2] > (w - limit_y) or front_left_[3] > h:
                    front_left_ = None

            handpose_right, handpose_left = swap_hand(handpose_right, handpose_left, front_right_, front_left_) 
            intersect_right, intersect_left = False, False
            if top_right is not None:
                intersect_right = intersect_top(top_right, box2=TOP_BOX[i])
            if top_left is not None:
                intersect_left = intersect_top(top_left, box2=TOP_BOX[i])
            if front_right is not None:
                intersect_right &= intersect_side(front_right_, box2=SIDE_BOX[i])
            if front_left is not None:
                intersect_left &= intersect_side(front_left_, box2=SIDE_BOX[i])

            record = {"left":intersect_left, "right":intersect_right, "pos_left":front_left_, "top_left":top_left, "top_right":top_right,
                    "pos_right":front_right_, "handpose_right":handpose_right, "handpose_left":handpose_left}

            intersection.append(record)
        inter_data[str(frames)] = intersection

    return inter_data

# def intersect(box1, box2=[165, 0, 375, 50], threshold = 0.0):
#     """x1, y1, x2, y2 = box2
#     #print(box1,  x1, y1, x2, y2)
#     if x1 < box1[0] < x2 and y1 < box1[1] < y2:
#         return True
#     if x1 < box1[2] < x2 and y1 < box1[1] < y2:
#         return True"""
#     xA = max(box1[0], box2[0])
#     yA = max(box1[1], box2[1])
#     xB = min(box1[2], box2[2])
#     yB = min(box1[3], box2[3])

#     # compute the area of intersection rectangle
#     interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
#     if interArea == 0:
#         return 0
#     # compute the area of both the prediction and ground-truth
#     # rectangles
#     boxAArea = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
#     boxBArea = abs((box2[2] - box2[0]) * (box2[3] - box2[1]))

#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     iou = interArea / float(boxAArea + boxBArea - interArea)
#     return iou > threshold

def intersect_top(box1, box2=[165, 0, 375, 50], threshold = 0.0):

    box2 = [box2[0], box2[2], box2[3], box2[1]]
    box2 = Polygon([[int(p[0]), int(p[1])] for p in box2])
    polygon1 = shapely.geometry.box(*box1, ccw=True)
    polygon2 = box2


    def IOU2(polygon1, polygon2):
    # Define each polygon
        polygon1_shape = Polygon(polygon1)
        polygon2_shape = Polygon(polygon2)
        try:
        # Calculate intersection and union, and tne IOU
            polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
            polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
            return polygon_intersection / polygon_union
        except:
            print(polygon1,polygon2)
            import matplotlib.pyplot as plt

            x,y = polygon1.exterior.xy
            plt.plot(x, y, color='#6699cc', alpha=0.7)
            x1,y1 = polygon2.exterior.xy
            plt.plot(x1, y1, color='#8899cc', alpha=0.7)
            # plt.plot(x,y)
            # plt.waitforbuttonpress()
            pass


    return IOU2(polygon1, polygon2) > threshold

def intersect_side(box1, box2=[165, 0, 375, 50], threshold = 0.0):
    if box1 is None:
        return False

    box2 = [box2[0], box2[1], box2[2], box2[3]]
    box2 = Polygon([[int(p[0]), int(p[1])] for p in box2])
    polygon1 = shapely.geometry.box(*box1, ccw=True)
    polygon2 = box2


    def IOU2(polygon1, polygon2):
    # Define each polygon
        polygon1_shape = Polygon(polygon1)
        polygon2_shape = Polygon(polygon2)
        try:
        # Calculate intersection and union, and tne IOU
            polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
            polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
            return polygon_intersection / polygon_union
        except:
            # print(polygon1,polygon2)
            import matplotlib.pyplot as plt

            x,y = polygon1.exterior.xy
            plt.plot(x, y, color='#6699cc', alpha=0.7)
            x1,y1 = polygon2.exterior.xy
            plt.plot(x1, y1, color='#8899cc', alpha=0.7)
            # plt.plot(x,y)
            # plt.waitforbuttonpress()
            pass

    return IOU2(polygon1, polygon2) > threshold

def bounding_box_from_polygon(points):
    x_coordinates, y_coordinates = zip(*points)

    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def cal_cam(video, inter_data):
    path = os.path.dirname(video)
    name = video.split("/")[-1]
    name = name.split('.')[0]
    # print(name)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    h, w, _ = frame.shape
    frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(frame_number)
    fps = cap.get(cv2.CAP_PROP_FPS)

    dirs = f"{path}/output/"
    if not os.path.exists(dirs):
        os.mkdir(dirs)

    limit_x, limit_y = 360, 512

    video_clip = 0
    for frame_ids, intersection in inter_data.items():  
        print("Processing clip ", video_clip)  
        start, end = eval(frame_ids)
        start_frame = start * fps
        interations = defaultdict(dict)

        objpoints0, objpoints1, imgpoints0, imgpoints1 = [], [], [], []
        objpointsall = []
        for p3d, p2d in DATA[video_clip].items():
            x, y, z = p3d
            p3d_ = [x,y,z]
            objpointsall.append(list(p3d_))
            if len(p2d) > 0:
                objpoints1.append(list(p3d_))
                imgpoints1.append(list(p2d))

        front_polygon = SIDE_BOX[video_clip]
        front_bbox = bounding_box_from_polygon(front_polygon)

        video_clip += 1
        objpoints1 = np.array(objpoints1).astype('float32')
        imgpoints1 = np.array(imgpoints1).astype('float32')
        objpointsall = np.array(objpointsall).astype('float32')

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, im1 = cap.read()
        # print(im1.shape)
        im1 = im1[:, limit_y:]
        size = im1.shape
        dim = (size[0], size[1])
        # interations["shape"] = dim
        interations["shape"] = (180,256)
        mtx1 = cv2.initCameraMatrix2D([objpoints1],[imgpoints1], dim)
        #calibrate cameras
        err, mtx1, dist, R1, T1 = cv2.calibrateCamera([objpoints1], [imgpoints1], dim, mtx1, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS) 

        R1, T1 = R1[0], T1[0]
        
        check_parameters(mtx1, dist, R1, T1, imgpoints1, objpoints1)
        #project a 3D point in the pointcloud onto a 2D image
        R1 = cv2.Rodrigues(R1)[0]
        c1 = -np.matrix(R1).T * np.matrix(T1)
        c1 = np.array(c1.T)[0]
        
        print("Camera position 1", c1)
        P1 = mtx1.dot(np.hstack((R1,T1)))

        h_w, h_h = 180, 256
        # pts1 = np.float32([[331,128],[509,110],[332,434],[509,459]])
        pts1 = np.float32(front_polygon)
        # pts2 = np.float32([[0,0],[h_w,0],[0,h_h],[h_w, h_h]])
        # pts2 = np.float32([[0,h_h],[h_w, h_h],[h_w,0],[0,0],]) #upside down
        # pts2 = np.float32([[h_w,0],[0,0],[0,h_h],[h_w, h_h]]) #flipped
        pts2 = np.float32([[0,0],[h_w,0],[h_w, h_h],[0,h_h]])

        # pts2 = np.float32([[0,h_h],[h_w, h_h],[h_w,0],[0,0],]) 

        def save_to_file(file_name,M):
            pickle.dump(M, open(file_name, "wb"))
            print("[INFO] saved model to file: {}".format(file_name))
        

        M = cv2.getPerspectiveTransform(pts1,pts2)
        save_to_file(f"{dirs}{name}_{start}_{end}_M.p".format(name),M)
        background = cv2.warpPerspective(im1,M,(h_w, h_h))

        # cv2.imshow("img",background)
        # cv2.waitKey(0)
        # exit(0)
        
        heatmap = np.zeros((h_h, h_w))
        print(f'{dirs} + {name} + _{start}_{end}_heat.mp4')
        cap = cv2.VideoCapture(video)
        output_cap = cv2.VideoWriter(dirs + name + '_{}_{}_heat.mp4'.format(start, end),fourcc, int(fps), (h_w*4, h_h))
        output_cap_cummulative = cv2.VideoWriter(dirs + name + '_{}_{}_heat_cummulative.mp4'.format(start, end),fourcc, int(fps), (h_w*3, h_h))
        interactions_file = dirs + name + '_{}_{}_heat.pkl'.format(start, end)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for i, record in enumerate(intersection):
            ret, frame = cap.read()
            if not ret: break
            frame_heatmap = np.zeros((h_h, h_w))
            left_frame_heatmap = np.zeros((h_h, h_w))
            right_frame_heatmap = np.zeros((h_h, h_w))
            frame = frame[:, limit_y:]
            front_left = record["pos_left"]
            front_right = record["pos_right"]   
            background = cv2.warpPerspective(frame,M,(h_w, h_h)) 

            for hand in ["left", "right"]:
                if not record[hand]: continue
                p = record["pos_" + hand]
                if p is None: 
                    continue
                p_list = get_point(p,front_bbox)
                handpose = record["handpose_" + hand]
                if handpose is None:
                    p_list = [p_list]
                else:
                    p_list = [[p[0],p[1],1.0] for p in handpose]

                inter_results = []
                for j, p1 in enumerate(p_list):          
                    X = np.dot(lin.pinv(P1),p1)
                    X = X / X[3]
                    #XX  = np.copy(X)
                    #XX[1] = X[2]; XX[2] = X[1]; XX[2] = -XX[2]
                    xvec = np.copy(X)
                    xvec[0] = c1[0]-xvec[0]
                    xvec[1] = c1[1]-xvec[1]
                    xvec[2] = c1[2]-xvec[2]
                    xvec = -xvec
                    planeNormal = np.array([0.0, -1.0, 0.0])
                    planePoint = np.array([0.0, 0.0, 0.0])
                    X = LinePlaneCollision(planeNormal, planePoint, xvec[:3], c1, epsilon=1e-6)
                    x, y, z = X[0], -0.1, X[2]
                    inter_results.append([x, z])
                    
                    #ax.scatter3D(x, y, z, c='k')

                    x_map, y_map = z*h_h/z_dim, x*h_w/x_dim
                    map_x = int(x_map) 
                    map_y = int(y_map)
                    # print("frame,x,y",i, map_x, map_y)
                    interations[i][hand] = (map_y,map_x)

                    if 0 <= map_x < h_h and 0 <= map_y < h_w:
                        frame_heatmap[h_h - map_x-1, map_y] = 1
                        frame_heatmap = cv2.GaussianBlur(frame_heatmap,(7,7),0)
                        if hand == "left":
                            left_frame_heatmap = copy.deepcopy(frame_heatmap)
                        else:
                            right_frame_heatmap = copy.deepcopy(frame_heatmap)
                        heatmap += frame_heatmap
                    else:
                        continue   
                    cv2.circle(frame, (int(p1[0]), int(p1[1])), 5, (0, 255, 0), 2)
                    # print(frame.shape)
                    # interations[i][hand] = (int(p1[0]),int(p1[1]))

                    #cv2.putText(frame, hand, (int(p1[0]), int(p1[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255 ,255), 2, cv2.LINE_AA)
                record["inter_" + hand] = inter_results
                    
            if front_right is not None:
                if record["right"]:
                    cv2.rectangle(frame, (front_right[0], front_right[1]), (front_right[2], front_right[3]), (0,255,0), 2)
                    # interations[i]["right"] = (int((front_right[0]+front_right[2])/2),int((front_right[1]+front_right[3])/2)) 
                else:
                    cv2.rectangle(frame, (front_right[0], front_right[1]), (front_right[2], front_right[3]), (0,0,255), 2)         
            if front_left is not None:
                if record["left"]:
                    # interations[i]["left"] = (int((front_left[0]+front_left[2])/2),int((front_left[1]+front_left[3])/2))
                    cv2.rectangle(frame, (front_left[0], front_left[1]), (front_left[2], front_left[3]), (0,255,0), 2)
                else:
                    cv2.rectangle(frame, (front_left[0], front_left[1]), (front_left[2], front_left[3]), (0,0,255), 2)


            "pickle dump interactions"
            with open(interactions_file, 'wb') as f:
                pickle.dump(interations, f)


            frame = cv2.resize(frame, (h_w*2, h_h))     
            if np.max(heatmap) > 0.01:
                heatmap_s = cv2.applyColorMap(np.uint8(255 * (heatmap/np.max(heatmap))), cv2.COLORMAP_JET)
            else:
                heatmap_s = np.zeros((h_h, h_w, 3), dtype="uint8")
            
            heatmap_s[heatmap < 0.01] = 0
            heatmap_s = cv2.addWeighted(heatmap_s, 0.5, background, 0.5, 0)

            if np.max(frame_heatmap) > 0.01:
                heatmap_is = cv2.applyColorMap(np.uint8(255 * (frame_heatmap/np.max(frame_heatmap))), cv2.COLORMAP_JET)
            else:
                heatmap_is = np.zeros((h_h, h_w, 3), dtype="uint8")
            heatmap_is[frame_heatmap < 0.01] = 0
            heatmap_is = cv2.addWeighted(heatmap_is, 0.5, background, 0.5, 0) 

            if np.max(left_frame_heatmap) > 0.01:
                heatmap_left_coloured = cv2.applyColorMap(np.uint8(255 * (left_frame_heatmap/np.max(left_frame_heatmap))), cv2.COLORMAP_JET)
            else:
                heatmap_left_coloured = np.zeros((h_h, h_w, 3), dtype="uint8")
            heatmap_left_coloured[left_frame_heatmap < 0.01] = 0
            heatmap_left_coloured = cv2.addWeighted(heatmap_left_coloured, 0.5, background, 0.5, 0) 

            if np.max(right_frame_heatmap) > 0.01:
                heatmap_right_coloured = cv2.applyColorMap(np.uint8(255 * (right_frame_heatmap/np.max(right_frame_heatmap))), cv2.COLORMAP_JET)
            else:
                heatmap_right_coloured = np.zeros((h_h, h_w, 3), dtype="uint8")
            heatmap_right_coloured[right_frame_heatmap < 0.01] = 0
            heatmap_right_coloured = cv2.addWeighted(heatmap_right_coloured, 0.5, background, 0.5, 0) 



            # frame_o = np.zeros((h_h, h_w*4, 3), dtype="uint8")
            # frame_o[:, :2*h_w] = frame.astype("uint8")
            # frame_o[:, 2*h_w:3*h_w] = heatmap_is.astype("uint8")
            # frame_o[:, 3*h_w:] = heatmap_s.astype("uint8")

            frame_o = np.zeros((h_h, h_w*4, 3), dtype="uint8")
            frame_o[:, :1*h_w] = heatmap_left_coloured.astype("uint8")
            frame_o[:, h_w :3*h_w] = frame.astype("uint8")
            frame_o[:, 3*h_w:] = heatmap_right_coloured.astype("uint8")

            frame_cummalative = np.zeros((h_h, h_w*3, 3), dtype="uint8")
            frame_cummalative[:, :2*h_w] = frame.astype("uint8")
            frame_cummalative[:, 2*h_w:] = heatmap_s.astype("uint8")


            #print(frame_o.shape, heatmap_is.shape, heatmap_s.shape, frame.shape)
            output_cap.write(frame_o)
            output_cap_cummulative.write(frame_cummalative)
            
        output_cap.release()
        output_cap_cummulative.release()
    
    return inter_data

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
 
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi

def get_point(bb1, bb2):
    bb2 = [bb2[0][0], bb2[0][1], bb2[1][0], bb2[1][1]]
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    mid_x = (x_left + x_right) / 2.0
    mid_y = (y_top + y_bottom) / 2.0
    return [mid_x, mid_y, 1.0]

    




def smooth_inter(inter_data):
    for _, inter in inter_data.items():
        old_r, old_l = None, None
        for i, r in enumerate(inter):
            if i == len(inter)-1: continue
            if i == 0:
                old_r = r["right"]
                old_f = r["left"]
                continue
            next_r = inter[i+1]["right"]
            next_l = inter[i+1]["left"]
            if old_r and next_r:
                r["right"] = copy.deepcopy(old_r)
            if old_l and next_l:
                r["left"] = copy.deepcopy(old_l)
            old_r = copy.deepcopy(r["right"])
            old_l = copy.deepcopy(r["left"])          
    return inter_data

def draw_inter(video, inter_data):
    name = video.split("/")[-1]
    name = name.split('.')[0]

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    h, w, _ = frame.shape
    frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    limit_x, limit_y = 360, 512
    dirs = "output/"
    if not os.path.exists(dirs):
        os.mkdir(dirs)

    current_clip = 0
    for frame_ids, inter in inter_data.items():
        cap = cv2.VideoCapture(video)
        start, end = eval(frame_ids)
        output_front = cv2.VideoWriter(dirs + name + '_{}_{}_front.mp4'.format(start, end),fourcc, int(fps), (w - limit_y, h))
        output_top = cv2.VideoWriter(dirs + name + '_{}_{}_top.mp4'.format(start, end),fourcc, int(fps), (limit_y, h - limit_x))

        start_frame = start * fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for r in inter:
            ret, frame = cap.read()


            if not ret: continue
            # Image
            f_top = frame[limit_x:, :limit_y]
            f_front = frame[:, limit_y:]
            top_left = r["top_left"]
            top_right = r["top_right"]
            intersect_left = r["left"]    
            intersect_right = r["right"]      
            front_left = r["pos_left"]
            front_right = r["pos_right"]
            handpose_right = r["handpose_right"]
            handpose_left = r["handpose_left"]

            cv2.rectangle(f_top, (TOP_BOX[current_clip][0],TOP_BOX[current_clip][1]), (TOP_BOX[current_clip][2],TOP_BOX[current_clip][3]), (0,0,0), 2)
            cr, cl = (0,0,255), (0,0,255)
            if intersect_right: cr = (0,255,0)
            if intersect_left: cl = (0,255,0)
            if top_right is not None:
                cv2.rectangle(f_top, (top_right[0], top_right[1]), (top_right[2], top_right[3]), cr, 2)
                cv2.putText(f_top, "RIGHT", (top_right[0], top_right[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255 ,255), 2, cv2.LINE_AA)    
            if top_left is not None:
                cv2.rectangle(f_top, (top_left[0], top_left[1]), (top_left[2], top_left[3]), cl, 2)
                cv2.putText(f_top, "LEFT", (top_left[0], top_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255 ,255), 2, cv2.LINE_AA)
            if front_right is not None:
                cv2.rectangle(f_front, (front_right[0], front_right[1]), (front_right[2], front_right[3]), cr, 2)         
                cv2.putText(f_front, "RIGHT", (front_right[0], front_right[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255 ,255), 2, cv2.LINE_AA)
            if front_left is not None:
                cv2.rectangle(f_front, (front_left[0], front_left[1]), (front_left[2], front_left[3]), cl, 2)
                cv2.putText(f_front, "LEFT", (front_left[0], front_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255 ,255), 2, cv2.LINE_AA)
            if handpose_right is not None:
                for p in handpose_right:
                    cv2.circle(f_front, (p[0], p[1]), 5, cr, 2)
                    #cv2.putText(f_front, "R", (p[0], p[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255 ,255), 2, cv2.LINE_AA)
            if handpose_left is not None:
                for p in handpose_left:
                    cv2.circle(f_front, (p[0], p[1]), 5, cl, 2)  
                    #cv2.putText(f_front, "L", (p[0], p[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255 ,255), 2, cv2.LINE_AA)


            output_top.write(f_top)
            output_front.write(f_front)
        
        output_front.release()
        output_top.release()   
        cap.release()
        current_clip += 1
    return

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def write_results_datavyu(video, gt_path, inter_data):
    pathname = os.path.dirname(video)
    
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    h, w, _ = frame.shape
    frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    """with open(gt_path, "r") as f:
        content = f.readlines()

    print(content[0])"""
    seg_list_left_start = ["interaction_left.onset"]
    seg_list_left_end = ["interaction_left.offset"]
    seg_list_right_start = ["interaction_right.onset"]
    seg_list_right_end = ["interaction_right.offset"]
    hand_left_pos = ["hand_left.pos"]
    hand_left_touch = ["hand_left.touch"]
    hand_right_pos = ["hand_right.pos"]
    hand_right_touch = ["hand_right.touch"]

    for frame_ids, inter in inter_data.items():
        start, end = eval(frame_ids)
        frame_start = int(start*fps)
        frame_end = int(end*fps)
        ms_start = (start+1) * 1000
        ms_end = end*1000

        seg_left, seg_right = [] ,[]
        for i, record in enumerate(inter):
            time_step = int(ms_start + 1000*i/fps)
            if record["pos_left"] is None: 
                seg_left.append(0)
            else:
                seg_left.append(1)                             

            if record["pos_right"] is None: 
                seg_right.append(0)
            else:
                seg_right.append(1)

            if record['left']: 
                if record["pos_left"] is not None: 
                    hand_left_pos.append([time_step, record["pos_left"]])        
                if "inter_left" in record: 
                    hand_left_touch.append([time_step, record["inter_left"]])
            if record['right']: 
                if record["pos_right"] is not None: 
                    hand_right_pos.append([time_step, record["pos_right"]])   
                if "inter_right" in record: 
                    hand_right_touch.append([time_step, record["inter_right"]])                    


        seg_left = [1 if x > 0.5 else 0 for x in smooth(seg_left, 5)]
        seg_right = [1 if x > 0.5 else 0 for x in smooth(seg_right, 5)]
        start_l, start_r = None, None

        for i in range(len(seg_left)):
            if start_l is None and seg_left[i] == 1: start_l = i
            if start_r is None and seg_right[i] == 1: start_r = i
            if i + 1 >= len(seg_left):
                continue
            if seg_left[i] != seg_left[i+1] and seg_left[i] == 1:
                seg_list_left_start.append(ms_start + 1000*start_l/fps)
                seg_list_left_end.append(ms_start + 1000*i/fps)
                start_l = None
            if seg_right[i] != seg_right[i+1] and seg_right[i] == 1:
                seg_list_right_start.append(ms_start + 1000*start_r/fps)
                seg_list_right_end.append(ms_start + 1000*i/fps)
                start_r = None
        if seg_left[-1] == 1: 
            seg_list_left_start.append(ms_start + 1000*start_l/fps)
            seg_list_left_end.append(ms_start + 1000*i/fps)         
        if seg_right[-1] == 1: 
            seg_list_right_start.append(ms_start + 1000*start_r/fps)
            seg_list_right_end.append(ms_start + 1000*i/fps)

    with open(video.replace(".mp4",".csv"), "w") as f:
        nlines = max(len(seg_list_left_start), len(seg_list_left_end), len(seg_list_right_end), len(seg_list_right_end),
                len(hand_left_pos), len(hand_left_touch), len(hand_right_pos), len(hand_right_touch))
        for i in range(nlines):
            line = []
            if i >= len(seg_list_left_start):
                line.append(",")
            else:
                line.append(str(seg_list_left_start[i]))
            if i >= len(seg_list_left_end):
                line.append(",")
            else:
                line.append(str(seg_list_left_end[i]))

            if i >= len(seg_list_right_start):
                line.append(",")
            else:
                line.append(str(seg_list_right_start[i]))
            if i >= len(seg_list_right_end):
                line.append(",")
            else:
                line.append(str(seg_list_right_end[i])) 

            if i >= len(hand_left_pos):
                line.append(",")
            else:
                line.append(str(hand_left_pos[i]))
            if i >= len(hand_right_pos):
                line.append(",")
            else:
                line.append(str(hand_right_pos[i]))   

            if i >= len(hand_left_touch):
                line.append(",")
            else:
                line.append(str(hand_left_touch[i]))
            if i >= len(hand_right_touch):
                line.append(",")
            else:
                line.append(str(hand_right_touch[i]))   

            line = ",".join(line)
            f.write(line + "\n")

 
    return

if __name__ == "__main__":
    import json
    import sys
    session = sys.argv[1]

    session_path = f"{session}"
    video = glob(f'{session_path}/*.mp4')[0]
    # print(video)

    gt_path = f"{session_path}/datavyu.csv"
    top_states = f"{session_path}/top/output/states"
    front_states = f"{session_path}/right/output/states"

    top_bbox_path = f"{session_path}/top/"
    front_bbox_path = f"{session_path}/right/"

    # cal_cam(video)
    top, front = read_track(top_states, front_states)
    TOP_BOX,DATA,SIDE_BOX = read_points_bounding_boxes(top_bbox_path, front_bbox_path)
    # print(DATA[0])
    inter = get_track(video, top, front)
    inter = smooth_inter(inter)
    # draw_inter(video, inter)
    inter = cal_cam(video, inter )
    # with open(f'{session_path}/data.json') as outfile:
    #     json.dump(inter, outfile)
    
    # with open(f'{session_path}/data.json') as outfile:
    #     inter = json.load(outfile)   
    write_results_datavyu(video, gt_path, inter)
    # os.system("ruby ./wpscan.rb -u www.mysite.com")