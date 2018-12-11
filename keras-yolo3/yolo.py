# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
import pickle
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from pathwaytest import dualway
from maxpathway import findmax_score_frame
from mining import drap_change,getmotionrects
from Bbox import bbox

class YOLO(object):
    _defaults = {
        "model_path": 'logs/24e_mining.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/my_classes.txt',
        "score": 0.45,
        "iou": .5,
        "model_image_size": (416, 416),
        "gpu_num": 2,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(
            '.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors//3, num_classes)
            # make sure model, anchors and classes match
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        # Shuffle colors to decorrelate adjacent classes.
        np.random.shuffle(self.colors)
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(
                self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(
                image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)

        return image,out_boxes,out_scores,out_classes

    def close_session(self):
        self.sess.close()

# def flow2hsv(flow):
#     import cv2
#     man, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     hsv = np.zeros((flow.shape[0],flow.shape[1],3),dtype=np.uint8)
#     hsv[..., 1] = 255
#     hsv[..., 0] = ang*180/np.pi/2
#     hsv[..., 2] = cv2.normalize(man, None, 0, 255, cv2.NORM_MINMAX)
#     return hsv

def detect_video(yolo, video_path, output_path=""):

    import cv2
##################################################################################
#video parameter setup
###################################################################################
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")


    # video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')
    # video_fps = vid.get(cv2.CAP_PROP_FPS)
    # video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))*2,
    #               int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # isOutput = True if output_path != "" else False
    # if isOutput:
    #     print("!!! TYPE:", type(output_path), type(
    #         video_FourCC), type(video_fps), type(video_size))
    #     out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
################################################################################
    
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

#####################################################################################
#optical flow
#####################################################################################
    return_value, frame=vid.read()
    prev=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    tvl1otf=cv2.DualTVL1OpticalFlow_create()
#######################################################################################

    bboxes_list=[]
    frame_index=1

    while True:
        return_value, frame = vid.read()
        if return_value:
            bboxes=[]
            nextf=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            flow= tvl1otf.calc(prev,nextf,None)
            intensity=np.sqrt(flow[...,0]**2,flow[...,1]**2)
            draped_intensity=drap_change(intensity)
            _,boxes_motion=getmotionrects(draped_intensity)

            #motion bbox generation
            for box in boxes_motion:
                top,left,bottom,right=box
                top=max(0,int(top))
                left=max(0,int(left))
                bottom=min(intensity.shape[0],int(bottom))
                right=min(intensity.shape[1],int(right))
                area_int=intensity[top:bottom,left:right]
                if len(area_int)==0:
                    motion_score=0
                else:
                    motion_score=np.mean(area_int)
                for cls in [0,1]:
                    bboxes.append(bbox(motion_score,0.5,cls,box,frame_index))
                    
            #yolo bbox generation
            image = Image.fromarray(frame)
            image,outboxes,outscores,outclass = yolo.detect_image(image)

            for index,box in enumerate(outboxes):
                top,left,bottom,right=box
                top=max(0,int(top))
                left=max(0,int(left))
                bottom=min(intensity.shape[0],int(bottom))
                right=min(intensity.shape[1],int(right))
                area_int=intensity[top:bottom,left:right]
                if len(area_int)==0:
                    motion_score=0
                else:
                    motion_score=np.mean(area_int)
                bboxes.append(bbox(motion_score,outscores[index],outclass[index],box,frame_index))
            bboxes_list.append(bboxes)
            prev=nextf
            frame_index+=1
        else:
            break
    vid.release()
    return bboxes_list

def findbestway(video_path,output_path,bbox_list):
    print(bbox_list)
    import cv2
    vid = cv2.VideoCapture(video_path)
    video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("!!! TYPE:", type(output_path), type(
        video_FourCC), type(video_fps), type(video_size))

    out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    # print('start to find best way')
    best_box_list=[]
    best_socre_list=[]
    for whichcls in [0,1]:
        best_bbox,score_max,max_bbox_index,max_bbox_frame=findmax_score_frame(bbox_list,whichcls)
        best_box_list.append(best_bbox)
        best_socre_list.append(score_max)
    video_class=best_box_list[np.argmax(best_socre_list)].cls
    print(video_class)
    del best_box_list
    del best_socre_list

    # bestpathway=dualway(best_frame_index,best_box_index,motion_scores,boxes,frame_cls_classcores,alpha=1,gama=2)
    print('finded')
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    _,frame=vid.read()
    ret,frame=vid.read()
    count=0
    while(True):
        ret,frame=vid.read()
        #print(ret)
        #print('inside')
        if ret:
            bboxs= bbox_list[count]
            print(count)
            print(len(bboxs))
            if len(bboxs)==0:
                pass
            else:
                for box in bboxs:
                    if box.cls==video_class:
                        motion_score=box.motion
                        clS_score=box.score
                        txt='cls:{:.2f} motion:{:.2f},class:{}'.format(clS_score,motion_score,'chimp' if video_class==0 else 'gorilla')
                        print('for frame ',count,txt)
                        top,left,bottom,right=box.box
                        top=max(0,int(top))
                        left=max(0,int(left))
                        bottom=min(frame.shape[0],int(bottom))
                        right=min(frame.shape[1],int(right))
                        print(top,left,bottom,right)
                        cv2.rectangle(frame, (int(left),int(top)), ( int(right),int(bottom)), (0, 255, 0), 3)
                        cv2.putText(frame, text=txt, org=(left, top), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(255, 0, 0), thickness=2)
            out.write(frame)
            count+=1
        else:
            break
    vid.release()
    out.release()
    return
    # yolo.close_session()
