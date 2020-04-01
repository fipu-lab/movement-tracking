from models import *
from utils import *

import csv

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image
from threading import Timer

import pandas as pd
import numpy as np
import math
import threading
import argparse
import re 


parser = argparse.ArgumentParser()
parser.add_argument("--video", help="Enter the name of your video file")
parser.add_argument("--output", help= "Use if you want to keep csv file")
args = parser.parse_args()

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

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

# load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor


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
    detections = detect_image(pilimg)

    #print(frames)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))

    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:

        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
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
