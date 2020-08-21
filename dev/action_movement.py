"""
#######################
######## Setup ########
#######################

```bash

conda install -yq python=3.6 pytorch==1.4 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch --yes
conda install -yq python=3.6 av -c conda-forge --yes
conda install -yq python=3.6 tqdm ipykernel --yes
pip install yacs opencv-python tensorboardX
conda install -yq python=3.6 Cython SciPy matplotlib --yes
pip install cython-bbox easydict gdown

git clone https://github.com/MVIG-SJTU/AlphAction.git
cd AlphAction
python setup.py build develop

# Download models
mkdir models
cd models
gdown --id 1DKHo0XoBjrTO2fHTToxbV0mAPzgmNH3x
cd ..
mkdir -p ./data/models/detector_models
cd data/models/detector_models
gdown --id 1T13mXnPLu8JRelwh60BRR21f2TlGWBAM
gdown --id 1IJSp_t5SRlQarFClrRolQzSJ4K5xZIqm
cd ..
mkdir aia_models
cd aia_models
gdown --id 1DKHo0XoBjrTO2fHTToxbV0mAPzgmNH3x
gdown --id 1CudK8w0d2_5r73_tnyAY1Fnwd78hce3M

```

"""

from time import sleep
from itertools import count
from tqdm import tqdm

import torch
import pickle
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

from AlphAction.demo.visualizer import AVAVisualizer
from AlphAction.demo.action_predictor import AVAPredictorWorker

#pytorch issuse #973
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))

# METs table https://partner.ergotron.com/portals/0/literature/compendium-of-physical-activities.pdf
CATEGORIES = {
    "bend/bow": 1.2,
    "crawl": 2.0,
    "crouch/kneel": 1.2,
    "dance": 4.5,
    "fall down": 1.0,
    "get up": 1.3,
    "jump/leap": 3.0,
    "lie/sleep": 1.0,
    "martial art": 4.5,
    "run/jog": 4.5,
    "sit": 1.0,
    "stand": 1.0,
    "swim": 4.5,
    "walk": 2.0,
    "answer phone": 1.2,
    "brush teeth": 1.5,
    "carry/hold sth.": 1.2,
    "catch sth.": 1.5,
    "chop": 1.2,
    "climb": 3.0,
    "clink glass": 1.2,
    "close": 1.2,
    "cook": 2.5,
    "cut": 1.2,
    "dig": 3.0,
    "dress/put on clothing": 1.5,
    "drink": 1.2,
    "drive": 1.5,
    "eat": 1.5,
    "enter": 1.2,
    "exit": 1.2,
    "extract": 1.2,
    "fishing": 2.5,
    "hit sth.": 2.0,
    "kick sth.": 2.0,
    "lift/pick up": 3.0,
    "listen to sth.": 1.5,
    "open": 1.2,
    "paint": 1.5,
    "play board game": 1.5,
    "play musical instrument": 1.5,
    "play with pets": 2.8,
    "point to sth.": 1.1,
    "press": 1.2,
    "pull sth.": 1.2,
    "push sth.": 1.2,
    "put down": 1.2,
    "read": 1.2,
    "ride": 3.0,
    "row boat": 3.0,
    "sail boat": 3.0,
    "shoot": 1.2,
    "shovel": 2.0,
    "smoke": 1.0,
    "stir": 1.0,
    "take a photo": 1.0,
    "look at a cellphone": 1.0,
    "throw": 1.2,
    "touch sth.": 1.2,
    "turn": 1.2,
    "watch screen": 1.0,
    "work on a computer": 1.2,
    "write": 1.2,
    "fight/hit sb.": 2.5,
    "give/serve sth. to sb.": 1.2,
    "grab sb.": 1.1,
    "hand clap": 1.1,
    "hand shake": 1.1,
    "hand wave": 1.1,
    "hug sb.": 1.1,
    "kick sb.": 1.2,
    "kiss sb.": 1.2,
    "lift sb.": 1.2,
    "listen to sb.": 1.1,
    "play with kids": 1.5,
    "push sb.": 1.2,
    "sing": 2.0,
    "take sth. from sb.": 1.2,
    "talk": 1.8,
    "watch sb.": 1.2,
}
MET_PER_S = [CATEGORIES[x]/60 for x in CATEGORIES.keys()]


class ActionMovement:

    def __init__(self, video):
        self.video_path = video
        self.action_confidence_threshold = .5

        self.webcam = False
        self.cpu = False
        self.cfg_path = "../config_files/resnet101_8x8f_denseserial.yaml"
        self.weight_path = "../data/models/aia_models/resnet101_8x8f_denseserial.pth"
        self.visual_threshold = 0.5
        self.start = 0
        self.duration = -1
        self.detect_rate = 4
        self.common_cate = False
        self.hide_time = False
        self.tracker_box_thres = 0.1
        self.tracker_nms_thres = 0.4

        self.input_path = 0 if self.webcam else self.video_path
        self.device = torch.device("cpu" if self.cpu else "cuda")
        self.realtime = True # if self.webcam else False

        # Configuration for Tracker. Currently Multi-gpu is not supported
        self.gpus = "0"
        self.gpus = [int(i) for i in self.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
        self.min_box_area = 0
        self.tracking = True
        self.detector = "tracker"
        self.debug = False

        self.all_results = []

        base_dir = os.path.dirname(self.input_path) + os.path.sep + "AlphAction" + os.path.sep
        self.output_path = base_dir + os.path.splitext(os.path.basename(self.input_path))[0] + ".mp4"
        self.results_output_path = base_dir + str(os.path.basename(self.input_path).split(".")[0]) + "_results.json"

        os.makedirs(base_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.input_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    def _save(self):
        pass

    def _load(self):

        self.all_results = json.load(open(self.results_output_path, "r"))

    def _fill_gaps(self):

        step = int((1000 / self.detect_rate * self.fps / 1000))

        for i, res in enumerate(self.all_results):

            if res["timestamp"] is not None:

                actions = torch.nonzero(torch.FloatTensor(res["action_scores"]) >= self.action_confidence_threshold).squeeze(1) # , as_tuple=False
                res["actions"] = actions[:, 1].numpy()

                last = i+1+step if self.all_results[min(len(self.all_results)-1, i+1+step)]["timestamp"] is not None else i+2+step
                for f_res in self.all_results[i+1:last]:

                    assert f_res["timestamp"] is None, "Timestamp in step is not None"

                    for k in ["actions", "action_scores", "action_boxes", "action_ids", "boxes", "scores", "ids"]:
                        f_res[k] = res[k]

    def run(self):

        fh = open(self.results_output_path, 'w')
        fh.write("[")
        fh.flush()

        print('Starting video demo, video path: {}'.format(self.input_path))

        # Initialise Visualizer
        video_writer = AVAVisualizer(
            self.input_path,
            self.output_path,
            self.realtime,
            self.start,
            self.duration,
            (not self.hide_time),
            exclude_class=[],
            confidence_threshold=self.visual_threshold,
            common_cate=self.common_cate,
        )

        torch.multiprocessing.set_start_method('forkserver', force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')

        ava_predictor_worker = AVAPredictorWorker(self)

        def l(t):
            return [] if t is None else t.tolist()

        try:
            for i in tqdm(count()):
                with torch.no_grad():
                    (orig_img, boxes, scores, ids) = ava_predictor_worker.read_track()

                    if orig_img is None:
                        if not self.realtime:
                            ava_predictor_worker.compute_prediction()
                        break

                    if self.realtime:
                        result = ava_predictor_worker.read()

                        if i > 0:
                            fh.write(",")

                        action_boxes, action_scores, timestamp, action_ids = None, None, None, None
                        if result is not None:
                            predictions, timestamp, action_ids = result
                            action_boxes = predictions.bbox
                            action_scores = predictions.get_field("scores")

                        json.dump({"i": i,
                                   "boxes": l(boxes),
                                   "scores": l(scores),
                                   "ids": l(ids),
                                   "action_boxes": l(action_boxes),
                                   "action_scores": l(action_scores),
                                   "timestamp": timestamp,
                                   "action_ids": l(action_ids)},
                                  fh)
                        fh.flush()

                        flag = video_writer.realtime_write_frame(result, orig_img, boxes, scores, ids)
                        if not flag:
                            break
                    else:
                        video_writer.send_track((boxes, ids))

        except KeyboardInterrupt:
            print("Keyboard Interrupted")

        except Exception as e:
            print("Unhadled exception", e)

        finally:
            print("Close json results")
            fh.write("]")
            fh.flush()
            fh.close()

        if not self.realtime:
            video_writer.send_track("DONE")
            while True:
                result = ava_predictor_worker.read()
                if result is None:
                    sleep(0.1)
                    continue
                if result == "done":
                    break

                video_writer.send(result)

            video_writer.send("DONE")
            print("Wait for writer process to finish...")
            video_writer.progress_bar(i)

        video_writer.close()
        ava_predictor_worker.terminate()

    def play(self, show_boxes=True, show_plot=True):

        self._load()
        self._fill_gaps()

        total_movement = []

        cap = cv2.VideoCapture(self.input_path)
        i = -1

        while cap.isOpened() and i < len(self.all_results):
            i += 1

            success, frame = cap.read()

            if not success:
                break

            res = self.all_results[i]

            img = frame.copy()

            if res.get("actions", None) is not None:
                # print(len(res["boxes"]), len(res["actions"]), len(res["action_boxes"])) TODO: prati boxeve i akcije prema id-evima
                for box, action in zip(res["boxes"], res["actions"]):

                    box = [int(a) for a in box]
                    img = cv2.rectangle(img=img, rec=(box[0], box[1], box[2]-box[0], box[3]-box[1]), color=(0, 0, 255), thickness=2)
                    cv2.putText(img, list(CATEGORIES.keys())[action], (int(box[0]), int((box[1] - 15))), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

            cv2.imshow("Demo", img)
            if cv2.waitKey(10) == 27: # Esc
                break

        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def extract_movement(self, agg="sec"):

        MET_PER_FRAME = np.array([x/self.fps for x in MET_PER_S])

        self._load()
        self._fill_gaps()
        total_movement = []

        for i, res in enumerate(self.all_results):

            if res.get("actions", None) is not None:
                s = 0 if len(res["actions"]) == 0 else sum(MET_PER_FRAME[res["actions"]]) / len(res["actions"])
                total_movement.append(s)

        if agg == "sec":

            f_len = int(len(total_movement) / self.fps)
            leftover = int(len(total_movement) - f_len * self.fps)

            tm_reshaped = np.array(total_movement[:int(f_len * self.fps)]).reshape(-1, int(self.fps))
            out = np.sum(tm_reshaped, axis=1)

            if leftover > 0:
                out = np.concatenate([out, [sum(total_movement[-leftover:])]])

            total_movement = out

        return np.array(total_movement)


def collect_movement(video_list, duration):

    all_mvms = []

    for video in video_list:

        am = ActionMovement(video=video)

        mvm = am.extract_movement()

        mvm = np.concatenate([mvm[:duration], np.repeat(0, max(0, duration - len(mvm)))])

        all_mvms.append(mvm.reshape(-1, 1))

    return np.sum(np.concatenate(all_mvms, axis=1), axis=1)


def plot_active_inactive():

    duration = 20 * 60 # 20 min

    active = collect_movement(["../video/aktivni_kamera_1.mp4", "../video/aktivni_kamera_2.mp4", "../video/aktivni_kamera_3_0.mp4"], duration)
    inactive = collect_movement(["../video/neaktivni_kamera_1.mp4", "../video/neaktivni_kamera_2.mp4", "../video/neaktivni_kamera_3.mp4"], duration)

    plt.plot(active)
    plt.plot(inactive)

    plt.tight_layout()
    plt.legend(["Aktivni", "Neaktivni"])
    plt.title("Aktivnost prema praćenju METs vrijednosti (detekcija radnje)")
    plt.xlabel("Sekunde")
    plt.ylabel("MET")


if __name__ == "__main__":

    import sys
    print(sys.argv[1])
    pm = ActionMovement(sys.argv[1])
    pm.run()

    # am = ActionMovement("../video/neaktivni_kamera_3.mp4")
    # am.run()
