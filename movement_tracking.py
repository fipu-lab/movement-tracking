from utils import utils

import csv

import os
import sys

from PIL import Image

import pandas as pd
import math
import threading
import argparse
import re 


parser = argparse.ArgumentParser()
parser.add_argument("--video", help="Enter the name of your video file")
parser.add_argument("--output", help="Use if you want to keep csv file")
parser.add_argument("--model", help="Detection model to use (yolo, faster_rcnn)")
args = parser.parse_args()

if args.model == "yolo":
    print("Using yolo model for detection")
    from models.yolo import detect, to_tlwh
else:
    print("Using Faster-RCNN model for detection")
    from models.faster_rcnn import detect, to_tlwh

if args.output:
    csv_name = str(args.output)
else:
    csv_name = "movement_data.csv"

#Remove csv file with same name if exists
if os.path.exists(csv_name):
    os.remove(csv_name)

def start_visualization(file_name):
    python = sys.executable
    if sys.platform == "win32":
        if re.search("\s", python): 
            python = "\""+python+"\""
    print(python)
    os.system(str(python + " visualization.py "+ file_name))


def write_to_csv(data, file_name):
    if os.path.exists(file_name):
        with open(file_name, "a", newline='') as file:
            fieldnames = ['timestamp', "id",'result']
            writer = csv.DictWriter(file, fieldnames = fieldnames)
            for key in data:
                writer.writerow(key)
    else:
        with open(file_name, 'w', newline='') as file:
            fieldnames = ['timestamp', "id",'result']
            writer = csv.DictWriter(file, fieldnames = fieldnames)
            writer.writeheader()
            for key in data:
                writer.writerow(key)

def calculate_movement(time, data):
    """
    time: current frame
    data: data gathered until time
    """
    data = pd.DataFrame(data)
    timeWindow = 60
    timeEnd = time     
    dataInWindow = data[(data["timestamp"]<timeEnd) & (data["timestamp"]>(timeEnd- timeWindow))]
    vEach = list()
    vFinal = 0
    temp = dict()
    movement_results = list()

    for a in data[data["timestamp"] == timeEnd]["id"].unique():
        vTilda = math.sqrt((math.pow(dataInWindow[dataInWindow["id"] == a]["X"].diff().mean(), 2) + math.pow(dataInWindow[dataInWindow["id"] == a]["Y"].diff().mean(),2 )))
        if math.isnan(vTilda):
            vTilda = 0
        vEach.append(vTilda)

    vFinal = sum(vEach)

    if len(vEach) == 0:
        print("vEach = 0")
    else:
        vFinal = vFinal/len(vEach)

    temp['timestamp'] = timeEnd
    temp["id"] = data[data["timestamp"] == timeEnd]["id"].unique()
    temp['result'] = vFinal
    movement_results.append(temp)
    #print("Frame : ", timeEnd)
    #print(vFinal)

    return movement_results


class_path='config/coco.names'
classes = utils.load_classes(class_path)

if args.video:
    #Video file
    videopath = args.video 
else:
    #Live camera
    videopath = 0


import cv2
from sort import *
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800,600))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame=vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)

frames = 0
starttime = time.time()

#dict which consists of persone and location
personDict = {}
#list for saving data
data = []
#flag for visualization
viz_flag = True


while(True):
    ret, frame = vid.read()
    if not ret:
        break
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect(pilimg)

    #print(frames)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)

    if detections is not None:

        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:

            x1, y1, box_w, box_h = to_tlwh(img, x1, y1, x2, y2)

            color = colors[int(obj_id) % len(colors)]
            #ovdje nalazi koja je klasa detektani objekt 
            cls = classes[int(cls_pred)]
            if cls == "person":

                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                
                #centar pravokutnika
                rectCentar = (int((x1+(x1+box_w))/2) , int((y1+(y1+box_h))/2))

                #ovdje nacrta kruzic u sredini 
                cv2.circle(frame, rectCentar, 2, color, 4)

                #ak vec postoji 
                if obj_id in personDict:
                    #ak se centar pravokutnika nije pomakao vise od 5px  
                    if rectCentar[0] in range(personDict[obj_id][0]-5, personDict[obj_id][0]+5) and rectCentar[1] in range(personDict[obj_id][1]-5, personDict[obj_id][1]+5) :
                        cv2.putText(frame, cls + "-" + str(int(obj_id)) + ":(", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                        personDict[obj_id] = rectCentar
                    #ak se micao
                    else:
                        cv2.putText(frame, cls + "-" + str(int(obj_id)) + ":)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                        personDict[obj_id] = rectCentar
                #ak ne postoji
                else:
                    #sprema poziciju u dict
                    personDict[obj_id] = rectCentar
                    cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

                
                #print(obj_id ,":",personDict[obj_id], " in frame ", frames)
                temp = {}
                temp['timestamp'] =  frames
                temp['id'] = obj_id
                temp['X'] = personDict[obj_id][0]
                temp["Y"] = personDict[obj_id][1]
                data.append(temp)

                if frames > 60:
                    write_to_csv(calculate_movement(frames, data), file_name = csv_name)
                    if viz_flag:
                        t = threading.Thread(target=start_visualization, args=(csv_name, ))
                        t.daemon = True
                        t.start()
                        viz_flag = False

    cv2.imshow('Stream', frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
cv2.destroyAllWindows()
if not args.output:
    os.remove(csv_name)
