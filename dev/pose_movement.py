"""
#######################
######## Setup ########
#######################

```bash

pip install -U torch==1.4 torchvision==0.5 -f https://download.pytorch.org/whl/cu101/torch_stable.html

git clone -q https://github.com/MVIG-SJTU/AlphaPose.git
pip install -q youtube-dl cython gdown
pip install -q -U PyYAML
apt-get install -y -q libyaml-dev
cd AlphaPose && python setup.py build develop --user

python -m pip install git+https://github.com/yanfengliu/cython_bbox.git gdown

mkdir -p AlphaPose/detector/yolo/data
gdown -O AlphaPose/detector/yolo/data/yolov3-spp.weights https://drive.google.com/uc?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC

mkdir -p AlphaPose/detector/tracker/data
gdown -O AlphaPose/detector/tracker/data/jde.1088x608.uncertainty.pt https://drive.google.com/uc?id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA

gdown -O AlphaPose/pretrained_models/fast_dcn_res50_256x192.pth https://drive.google.com/uc?id=1zUz9YIk6eALCbZrukxD7kQ554nhi1pVv

mkdir AlphaPose/trackers/weights
# dowload model: osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth

import sys
sys.path.append("AlphaPose")

```

"""


import os
import cv2
import math
import json
import time
import pickle
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.vis import getTime
from alphapose.utils.writer import DataWriter
from PoseFlow.poseflow_infer import get_box
from alphapose.utils.vis import vis_frame
# from alphapose.utils.pPose_nms import write_json


class PoseMovement:

    def __init__(self, video, kp_score_treshold=.7):

        self.video = video
        self.kp_score_treshold = kp_score_treshold

        self.detector = "yolo"
        self.outputpath = os.path.dirname(self.video) + os.path.sep + "AlphaPose" + os.path.sep
        self.vis = False
        self.profile = False
        self.format = None # coco/cmu/open
        self.min_box_area = 0
        self.detbatch = 1 # 5
        self.posebatch = 10 # 80
        self.eval = False
        self.gpus = [0]
        self.flip = False
        self.qsize = 64 # 1024
        self.debug = False
        self.save_video = False
        self.vis_fast = False
        self.pose_flow = False
        self.pose_track = True
        self.sp = True
        self.save_img = False

        assert not (self.pose_flow and self.pose_track), "Pick only PoseFlow or Pose Track"

        self.device = torch.device("cuda:0")
        self.detbatch = self.detbatch * len(self.gpus)
        self.posebatch = self.posebatch * len(self.gpus)
        self.tracking = (self.pose_track or self.pose_flow or self.detector == 'tracker')

        self.cfg = "configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml"
        self.checkpoint = "pretrained_models/fast_dcn_res50_256x192.pth"

        self.cfg = update_config(self.cfg)

        self.all_results = []

    def _save(self):
        with open(self.outputpath + str(os.path.basename(self.video).split(".")[0]) + "_results.pkl", 'wb') as handle:
            pickle.dump(self.all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load(self):
        with open(self.outputpath + str(os.path.basename(self.video).split(".")[0]) + "_results.pkl", 'rb') as handle:
            self.all_results = pickle.load(handle)

    def run(self):

        if os.path.isfile(self.video):
            mode, input_source = 'video', self.video
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')

        if not os.path.exists(self.outputpath):
            os.makedirs(self.outputpath)

        det_loader = DetectionLoader(input_source, get_detector(self), self.cfg, self, batchSize=self.detbatch, mode=mode, queueSize=self.qsize)
        det_worker = det_loader.start()

        # Load pose model
        pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)

        print(f'Loading pose model from {self.checkpoint}...')
        pose_model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))

        if self.pose_track:
            tracker = Tracker(tcfg, self)

        pose_model.to(self.device)
        pose_model.eval()

        if self.save_video:
            from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
            video_save_opt['savepath'] = self.outputpath + os.path.basename(self.video)
            video_save_opt.update(det_loader.videoinfo)
            writer = DataWriter(self.cfg, self, save_video=True, video_save_opt=video_save_opt, queueSize=self.qsize).start()
        else:
            writer = DataWriter(self.cfg, self, save_video=False, queueSize=self.qsize).start()

        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

        batchSize = self.posebatch

        try:
            for i in im_names_desc:
                start_time = getTime()
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                    if orig_img is None:
                        break
                    if boxes is None or boxes.nelement() == 0:
                        writer.save(None, None, None, None, None, orig_img, os.path.basename(im_name))
                        continue

                    # Pose Estimation
                    inps = inps.to(self.device)
                    datalen = inps.size(0)

                    leftover = 0
                    if (datalen) % batchSize:
                        leftover = 1
                    num_batches = datalen // batchSize + leftover

                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                        hm_j = pose_model(inps_j)
                        hm.append(hm_j)

                    hm = torch.cat(hm)
                    #hm = hm.cpu()
                    if self.pose_track:
                        boxes,scores,ids,hm,cropped_boxes = track(tracker, self, orig_img, inps, boxes, hm, cropped_boxes, im_name, scores)
                    writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, os.path.basename(im_name))

            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
            writer.stop()
            det_loader.stop()

        except KeyboardInterrupt:
            det_loader.terminate()
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
            writer.stop()

        self.all_results = writer.results()
        self._save()

    @staticmethod
    def get_pose_bbox(img, human):

        height, width = img.shape[:2]
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
        kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
        keypoints = []

        for n in range(kp_scores.shape[0]):
            keypoints.append(float(kp_preds[n, 0]))
            keypoints.append(float(kp_preds[n, 1]))
            keypoints.append(float(kp_scores[n]))

        bbox = get_box(keypoints, height, width)

        return bbox

    @staticmethod
    def find_pose(poses, idx):

        for pose in poses:
            if idx == pose.get("idx"):
                return pose

        return None

    @staticmethod
    def compute_movement(pose1, pose2, kp_score_treshold=.5):

        COCO_PERSON_KEYPOINT_NAMES = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle')

        keypoints1 = pose1.get("keypoints")
        keypoints2 = pose2.get("keypoints")

        def dist(p1, p2):
            d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if d > 50: # probably a miss detection
                return 0
            return d

        if keypoints1.shape != keypoints2.shape or np.any(np.isnan(keypoints1.numpy().flatten())) or np.any(np.isnan(keypoints2.numpy().flatten())):
            print("Shape: ", keypoints1.shape, keypoints2.shape)
            print(keypoints1, keypoints2)

        # TODO: find midpoints for hips and sholders to replace two values with one
        points_of_interest = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        k1_poi = keypoints1[points_of_interest]
        k2_poi = keypoints2[points_of_interest]

        # filter only confident detections
        sc1_poi = pose1.get("kp_score")[points_of_interest]
        sc2_poi = pose2.get("kp_score")[points_of_interest]

        idx = np.where(sc1_poi.flatten() >= kp_score_treshold)[0]
        k1_poi = k1_poi[idx]
        k2_poi = k2_poi[idx]
        sc1_poi = sc1_poi[idx]
        sc2_poi = sc2_poi[idx]

        idx = np.where(sc2_poi.flatten() >= kp_score_treshold)[0]
        k1_poi = k1_poi[idx]
        k2_poi = k2_poi[idx]
        # /filter only confident detections

        total_dist = sum([dist(p1, p2) for p1, p2 in zip(k1_poi, k2_poi)])

        return total_dist

    def play(self, show_boxes=True, show_plot=True):

        self.showbox = show_boxes

        self._load()

        total_movement = {}
        for i in range(200): # create ids in advance
            total_movement[i] = []

        cap = cv2.VideoCapture(self.video)
        i = -1

        while cap.isOpened() and i < len(self.all_results):
            i += 1

            success, frame = cap.read()

            if not success:
                break

            if i == 0: # skip first frame
                continue

            result = self.all_results[i]

            img = vis_frame(frame, result, self)

            for idx in total_movement.keys():

                pose = self.find_pose(result["result"], idx)
                prev_pose = self.find_pose(self.all_results[i-1]["result"], idx)

                if pose is not None and prev_pose is not None:
                    total_movement[idx].append(self.compute_movement(pose, prev_pose, self.kp_score_treshold))
                else:
                    total_movement[idx].append(None)
            # TODO: divide frame_movement / num_of_persons_in_frame
            # show text
            for pose in result["result"]:

                look_back, activity_treshold = 60, 60

                ms = np.array(total_movement[pose.get("idx")][-look_back:])
                ms = ms[np.where(ms != None)]
                movement = sum(ms)/len(ms) if len(ms) > 0 else 0

                bbox = self.get_pose_bbox(img, pose)

                if movement > activity_treshold:
                    cv2.putText(img, ":)", (int(bbox[0]), int((bbox[2] - 15))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(img, ":(", (int(bbox[0]), int((bbox[2] - 15))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Demo", img)
            if cv2.waitKey(10) == 27: # Esc
                break

            if show_plot:
                plt.clf()
                x = np.array([[total_movement[idx][i] if total_movement[idx][i] != None else 0 for idx in total_movement.keys()] for i in range(len(total_movement[0]))])
                x = x.sum(axis=1)
                plt.plot(x.flatten())

        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def extract_movement(self, agg="sec"):

        assert agg in ("sec", None)

        self._load()
        total_movement = []

        for i in range(len(self.all_results)):

            if i == 0: # skip first frame
                continue

            prev_result = self.all_results[i-1]
            curr_result = self.all_results[i]
            frame_movement = []

            for pose in curr_result["result"]:

                prev_pose = self.find_pose(prev_result["result"], pose.get("idx"))

                if pose is not None and prev_pose is not None:
                    frame_movement.append(self.compute_movement(pose, prev_pose, self.kp_score_treshold))

            s = 0 if len(curr_result["result"]) == 0 else sum(frame_movement) / len(curr_result["result"])
            total_movement.append(s)

        if agg == "sec":
            cap = cv2.VideoCapture(self.video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            f_len = int(len(total_movement) / fps)
            leftover = int(len(total_movement) - f_len * fps)

            tm_reshaped = np.array(total_movement[:int(f_len * fps)]).reshape(-1, int(fps))
            out = np.sum(tm_reshaped, axis=1)

            if leftover > 0:
                out = np.concatenate([out, [sum(total_movement[-leftover:])]])

            total_movement = out

        return np.array(total_movement)


def collect_movement(video_list, duration):

    all_mvms = []

    for video in video_list:

        pm = PoseMovement(video=video)

        mvm = pm.extract_movement()

        mvm = np.concatenate([mvm[:duration], np.repeat(0, max(0, duration - len(mvm)))])

        all_mvms.append(mvm.reshape(-1, 1))

    return np.sum(np.concatenate(all_mvms, axis=1), axis=1)


def plot_active_inactive():
    duration = 20 * 60 # 20 min

    active = collect_movement(["../video/aktivni_kamera_1.mp4", "../video/aktivni_kamera_2.mp4", "../video/aktivni_kamera_3.mp4"], duration)
    inactive = collect_movement(["../video/neaktivni_kamera_1.mp4", "../video/neaktivni_kamera_2.mp4", "../video/neaktivni_kamera_3.mp4"], duration)

    plt.plot(active)
    plt.plot(inactive)

    plt.tight_layout()
    plt.legend(["Aktivni", "Neaktivni"])
    plt.title("Aktivnost prema praćenju poza")
    plt.xlabel("Sekunde")
    plt.ylabel("Pikseli")


if __name__ == "__main__":

    import sys
    print(sys.argv[1])
    pm = PoseMovement(sys.argv[1])
    pm.run()

    # pm = PoseMovement(video="../video/neaktivni_kamera_1.mp4")
    # pm.run()
    # pm.play()
    # plt.plot(pm.extract_movement())
