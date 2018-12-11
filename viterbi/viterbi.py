from utils import find_max_class_video,findmax_score_frame,filter_by_bestcls,addnone,interp_agumentation,split,findway,check_not_none
import pickle
import cv2
import numpy as np
import Bbox
import os
import sys
def assertnone(bboxes_list):
    for i,bboxes in enumerate(bboxes_list):
        c=0
        for b in bboxes:
            if b.isnone:
                c+=1
            assert b.frame_index==i+1
        assert c==1
def preprocess(bboxes_list):
    cls=find_max_class_video(bboxes_list)
    bboxes_list=filter_by_bestcls(bboxes_list,cls)
    bboxes_list=addnone(bboxes_list)
    bboxes_list= interp_agumentation(bboxes_list)
    assertnone(bboxes_list)
    return bboxes_list,cls


def viterbi(bboxes_list,cls):
    tracklet=[]
    best_box,mixscore,whichindex,whichframe=findmax_score_frame(bboxes_list,cls)

    forw,backw=split(bboxes_list,whichframe)
    forwardpath=findway(forw,mixscore)
    backwardpath=findway(backw,mixscore)
    backwardpath=list(reversed(backwardpath))
    backwardpath.pop()
    forwardpath.pop(0)
    allpath=backwardpath+[whichindex]+forwardpath
    
    for i in range(len(allpath)):
        if allpath[i]>=len(bboxes_list[i]):
            c=0
        else:
            c=allpath[i]
        tracklet.append(bboxes_list[i][c])
        bboxes_list[i].pop(c)
    bboxes_list=addnone(bboxes_list)
    restnum=check_not_none(bboxes_list)
    return tracklet,bboxes_list,restnum
def gettracklets(bboxes_list,cls):
    tracklets=[]
    while True:
        tracklet,bboxes_list,restum=viterbi(bboxes_list,cls)
        assertnone(bboxes_list)
        tracklets.append(tracklet)
        print(len(tracklet),restum)
        if restum < 50:
            break
    return tracklets

if __name__ == "__main__":
    for l in os.listdir('../../representative_bbox_pickle_output/'):
        l=l.split('.')[0]
        with open('../../representative_bbox_pickle_output/{}.pickle'.format(l),'rb') as file:
            bboxes_list=pickle.load(file)
        bboxes_list,cls= preprocess(bboxes_list)
        tracklets=gettracklets(bboxes_list,cls)
        print('total tracklets:',len(tracklets))
        cap=cv2.VideoCapture('../original_data/representative/{}.mp4'.format(l))
        video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        writer = cv2.VideoWriter('../representative_test/processed_{}.mp4'.format(l), video_FourCC, video_fps, video_size)
        _,_=cap.read()
        frame_index=0
        while(True):
            ret,frame=cap.read()
            if ret:
                np.random.seed(34)
                for tracklet in tracklets:
                    color=(np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
                    bbox=tracklet[frame_index]
                    if bbox.isnone:
                        continue
                    else:
                        top,left,bottom,right=tuple(map(int,bbox.box))
                        txt='cls:{} s:{}, m:{}'.format('chimp' if bbox.cls==0 else 'gorilla',bbox.score,bbox.motion)
                        cv2.rectangle(frame,(left,top),(right,bottom),color,3)
                        cv2.putText(frame,txt,(left,top),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.70, color=color, thickness=2)
                writer.write(frame)
                frame_index+=1
            else:
                break
        cap.release()
        writer.release()


    
    


