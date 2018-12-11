import cv2
import numpy as np
import pickle
from maxpathway import findmax_score_frame
def trackboxes(frames,frame_index,bbox,track_fun):

    bbox=(int(bbox[1]),int(bbox[0]),int(bbox[3]-bbox[1]),int(bbox[2]-bbox[0]))
    bboxes=np.zeros((len(frames),4))
    bboxes[frame_index]=np.array(bbox)

    tracker=track_fun()
    ok = tracker.init(frames[frame_index],bbox)

    current_i=frame_index

    for frame in frames[frame_index+1:]:
        current_i+=1
        ok,box=tracker.update(frame)
        #print(box)
        if True:
            bboxes[current_i]=np.array(list(box))
        else:
            bboxes[current_i]=np.array([None,None,None,None])
    del tracker
    tracker2=track_fun()
    ok = tracker2.init(frames[frame_index],bbox)
    current_i=frame_index
    for frame in reversed(frames[:frame_index]):
        current_i-=1
        ok,box=tracker2.update(frame)
        if True:
            bboxes[current_i]=np.array(list(box))
        else:
            bboxes[current_i]=np.array([None,None,None,None])

    return bboxes      

if __name__ == "__main__":
    a = 'FgJpFLxSmHboxes.txt FgJpFLxSmHmotion_scores.txt  FgJpFLxSmHframe_cls_classcores.txt'
    a = a.split()
    with open('../test_out/'+a[0], 'rb') as file:
        boxes = pickle.load(file)

    with open('../test_out/'+a[1], 'rb') as file:
        motions = pickle.load(file)

    with open('../test_out/'+a[2], 'rb') as file:
        frame_cls_clsscores = pickle.load(file)


    box_index, max_frame_index = findmax_score_frame(
        1, frame_cls_clsscores, motions)
    bbox=boxes[max_frame_index][box_index]

    cap=cv2.VideoCapture('../test_gorilla/FgJpFLxSmH.mp4')
    cap.read()
    frames=[]
    while True:
        ret,frame=cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    track_fun=cv2.TrackerGOTURN_create
    bboxes=trackboxes(frames,max_frame_index,bbox,track_fun)

    for index,frame in enumerate(frames):
        bbox=bboxes[index]
        if np.all(bbox==0):
            pass
        else:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame,p1,p2,(244,0,0),2,1)
        cv2.imshow('frame',frame)
        if cv2.waitKey(15) & 0xff ==ord('q'):
            break
        
    



        

