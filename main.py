import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from PIL import Image
import numpy as np
from util import visualize, set_background


set_background('./bg.png')
# set title
st.title('Brain MRI tumor detection')

# set header
st.header('Please upload an image')

# upload file
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

# load model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = './train/output/model_0002999.pth'
cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)

# load image
if file:
    image = Image.open(file).convert('RGB')

    image_array = np.asarray(image)

    # detect objects
    outputs = predictor(image_array)

    threshold = 0.5

    # Display predictions
    preds = outputs["instances"].pred_classes.tolist()
    scores = outputs["instances"].scores.tolist()
    bboxes = outputs["instances"].pred_boxes

    bboxes_ = []
    for j, bbox in enumerate(bboxes):
        bbox = bbox.tolist()

        score = scores[j]
        pred = preds[j]

        if score > threshold:
            x1, y1, x2, y2 = [int(i) for i in bbox]
            bboxes_.append([x1, y1, x2, y2])

    # visualize
    visualize(image, bboxes_)