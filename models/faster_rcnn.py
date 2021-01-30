import torch
import numpy as np

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

MODEL_CONFIG = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG)
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)


def detect(img):
    """
    out of shape (x1, y1, x2, y2, object_conf, class_score, class_pred)
    :param img:
    :return:
    """

    #img = pilimg.copy()
    img = np.array(img)

    outputs = predictor(img)
    instances = outputs["instances"]

    out = torch.cat((
             instances.pred_boxes.tensor,
             instances.scores.reshape((-1, 1)),
             instances.scores.reshape((-1, 1)),
             instances.pred_classes.float().reshape((-1, 1))
            ), 1)

    return out


def to_tlwh(img, x1, y1, x2, y2):

    box_w = int(x2 - x1)
    box_h = int(y2 - y1)
    x1 = int(x1)
    y1 = int(y1)

    return x1, y1, box_w, box_h
