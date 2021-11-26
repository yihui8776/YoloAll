
import cv2
from PIL import Image
import numpy as np
import os
import time

import torch
from torch import nn
# from torchvision.models import resnet50
import torchvision.transforms as T
from hubconf import detr_resnet50, detr_resnet50_panoptic ,_make_detr
import ssl

import matplotlib.pyplot as plt
from models.detr import DETR
from torchvision.ops.boxes import batched_nms
from common_utils import vis,visual

model = None
device = 'cpu'
size=(800,800)
infer_size=(800,800)
conf_thres =0.7
nms_thres=0.5
mean=[0.,0.,0.] # rbr
std=[255., 255., 255.] #bgr

LABEL = (
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
)

# 图像数据处理
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 将xywh转xyxy
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


# 将0-1映射到图像
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b




def get_support_models():
    model_list=[]
    now_dir = os.path.dirname(os.path.realpath(__file__))
    for file in os.listdir(now_dir):
        if str(file).endswith('.pth') and 'detr' in str(file):
            model_list.append(str(file).replace('.pth', ''))
    return model_list



def create_model(model_name='detr-r50-e632da11', dev='cpu'):
    global model
    global device
    model = None
    device = dev

    # Load model

    pre_train = os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s.pth' % (model_name))

    model = detr_resnet50(pretrained=False, num_classes=90 + 1).eval()
    #model = _make_detr("resnet50", dilation=False, num_classes=80 + 1)
    state_dict = torch.load(pre_train, map_location=device)
    model.load_state_dict(state_dict["model"])
    if device == 'cuda':
        model.cuda()


def inference(img):
    global model
    global device
    global size
    global conf_thres
    global nms_thres
    global mean
    global std

    map_result = {'type': 'img'}
    img2 =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #标准化等
    img2 = Image.fromarray(img2)
    img3 = transform(img2).unsqueeze(0)


    if img3.mode == 'RGBA':
        r, g, b, a = img3.split()
        img3 = Image.merge("RGB", (r, g, b))



    if device == "cuda":
        img3 = img3.cuda()

    with torch.no_grad():
        outputs = model(img3)
        #nms
        #pred = non_max_suppression(output, infer_conf, nms_thre, classes=None, agnostic=False)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > conf_thres

    probas = probas.cpu().detach().numpy()
    keep = keep.cpu().detach().numpy()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep],  img2.size)

    # draw det
    if  probas[keep] is None:
        map_result['result'] = img
    else:
        scores = []
        boxes = []
        cls = []
        for prob, (xmin, ymin, xmax, ymax) in zip(probas[keep], bboxes_scaled):  # 对各个类的概率和位置
            cl = prob.argmax()  # 选择概率最大值
            #scores.append(round(prob[cl] * 100, 2))
            scores.append(prob[cl])
            cls.append(cl)
            boxes.append([xmin,ymin,xmax,ymax])


        clss = np.array(cls, dtype = int)
        usescores = np.array(scores,dtype = float)
        useboxes = np.array(boxes,dtype = int)

        #vis_res = vis(img, useboxes, usescores, clss, conf=conf_thres)
        vis_res = visual(img, useboxes, usescores, clss, conf=conf_thres, class_names=LABEL)
        map_result['result'] = vis_res

    return map_result


if __name__ == '__main__':
    create_model(model_name='detr-r50-e632da11', dev='cpu')
    image = cv2.imread('people.jpg', cv2.IMREAD_COLOR)
    ret = inference(image)