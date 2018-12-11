import numpy as np
import Bbox
def findmax_score_frame(bbox_list,cls,alpha=1,beta=1):
    mixscore=0
    best_box=None
    whichindex=None
    whichframe=None
    
    for i, bboxes in enumerate(bbox_list):
        for j,bbox in enumerate(bboxes):
            if bbox.isnone:
                continue
            if bbox.cls==cls:
                candidate=alpha*bbox.motion*beta*bbox.score
                if candidate>mixscore:
                    mixscore=candidate
                    best_box=bbox
                    whichframe=i
                    whichindex=j
    return best_box,mixscore,whichindex,whichframe

def find_max_class_video(bboxes_list):
    n=np.zeros(2)
    for bboxes in bboxes_list:
        for bbox in bboxes:
            if bbox.isnone:
                continue
            n[bbox.cls]+=bbox.motion*bbox.score
    return np.argmax(n)

def addnone(bboxes_list):
    for i,bboxes in enumerate(bboxes_list):
        judge=np.any([bbox.isnone for bbox in bboxes]) 
        if judge:
            pass
        else:
            bboxes.append(Bbox.bbox(None,None,None,None,i+1,isnone=True))
    return bboxes_list

def filter_by_bestcls(bboxes_list,bestcls):
    for j,bboxes in enumerate(bboxes_list):
        for i,bbox in enumerate(bboxes):
            if bbox.isnone:
                continue
            else:
                if bbox.cls!=bestcls:
                    bboxes_list[j].pop(i)
    return bboxes_list

def interpotation(bboxes_list):
    def interp(bbox1,bbox2):
        interpnum=abs(bbox1.frame_index-bbox2.frame_index)
        def linefun(x,y,num):
            out=np.linspace(x,y,num=num,endpoint=False)
            out=out[1:]
            return out
        scores=linefun(bbox1.score,bbox2.score,interpnum)
        motions=linefun(bbox1.motion,bbox2.motion,interpnum)
        frames=linefun(bbox1.frame_index,bbox2.frame_index,interpnum)
        boxes=np.array((list(map(linefun,bbox1.box,bbox2.box,[interpnum]*4)))).transpose(1,0)
        bboxes=[]
        for i in range(interpnum-1):
            bboxes.append(Bbox.bbox(motions[i],scores[i],bbox1.cls,boxes[i],int(frames[i])))
        return bboxes

    pointer=None
    counter=0
    allinterbox=[]
    for i in range(len(bboxes_list)):
        bboxes=bboxes_list[i]
        isnone=True if len(bboxes)==1 else False
        if isnone:
            counter+=1
            continue
        else:
            if counter==0:
                pointer=i
            else:
                if counter<10:
                    # could interp
                    if pointer is not None:
                        cad1=bboxes_list[pointer]
                        cad2=bboxes_list[i]
                        possible_score=[]
                        possible_pair=[]
                        for bbox1 in cad1:
                            if bbox1.isnone:
                                continue
                            for bbox2 in cad2:
                                if bbox2.isnone:
                                    continue
                                possible_score.append(bbox1.iouscore(bbox2))
                                possible_pair.append((bbox1,bbox2))
                        for i,score in enumerate(possible_score):
                            if score >-0.:
                                bbox1=possible_pair[i][0]
                                bbox2=possible_pair[i][1]
                                bboxes=interp(bbox1,bbox2)
                                allinterbox.append(bboxes)               
            pointer=i
            counter=0
    return allinterbox

def interp_agumentation(bboxes_list):
    aggumentation=interpotation(bboxes_list)
    for eaches in aggumentation:
        for each in eaches:
            i=each.frame_index-1
            bboxes_list[i].append(each)
    return bboxes_list
def findway(process_list,bestbbox_score):
    max_score=[]
    max_score.append([bestbbox_score])
    max_score_index=[]
    max_score_index.append([0])
    
    for i,bboxes in enumerate(process_list[1:]):
        i=i+1
        score_list=[]
        index_list=[]
        if len(bboxes)==1:
            #current is none
            score_list.append(max(max_score[i-1]))
            if len(process_list[i-1])==1: # prev is also none
                index_list.append(0)
                
            else:
                index_list.append(np.argmax(max_score[i-1]))          
        else:
            if len(process_list[i-1])!=1 or i==1: #previous not None
                for bbox in bboxes:
                    if bbox.isnone:
                        emission=0
                    else:
                        emission=bbox.motion*bbox.score
                    transmissions=[]
                    # i==1 exception
                    
                    if i==1:
                        if bbox.isnone:
                            transmission=max_score[0][0]
                        else:
                            pre_bbox=process_list[0][0]
                            prev_bbox_index=0
                            transmission=bbox.iouscore(pre_bbox)+max_score[i-1][prev_bbox_index]
                        transmissions.append(transmission)
                    else:
                        for prev_bbox_index,pre_bbox in enumerate (process_list[i-1]):
                            if pre_bbox.isnone or bbox.isnone:
                                transmission=max_score[i-1]
                                transmission=transmission[prev_bbox_index]
                            else:
                                transmission=bbox.iouscore(pre_bbox)+max_score[i-1][prev_bbox_index]
                        transmissions.append(transmission)
                    score_list.append(np.max(transmissions)+emission) #could include a alpha or belta
                    index_list.append(np.argmax(transmissions))
            else:#previous is None
                for bbox in bboxes:
                    if bbox.isnone:
                        emission=0
                    else:
                        emission=bbox.motion*bbox.score
                    transmission=max_score[i-1][0]+0#iou is 0 for this case 
                    score_list.append(transmission+emission)
                    index_list.append(0)
        max_score.append(score_list)
        max_score_index.append(index_list)
    
    
    endpoint_index=np.argmax(max_score[-1])
    endpoint=max_score_index[-1][endpoint_index]
    best_path=[]
    best_path.append(endpoint)

    processed_max_index=list(reversed(max_score_index))[1:]
    for frame_index in range(len(processed_max_index)):

        endpoint=processed_max_index[frame_index][endpoint]
        best_path.append(endpoint)
    return list(reversed(best_path))

def split(bboxes_list,frame_index):
    forward_bboxes_list=bboxes_list[frame_index:]
    backward_bboxes_list=bboxes_list[:frame_index+1]
    backward_bboxes_list=list(reversed(backward_bboxes_list))
    return forward_bboxes_list,backward_bboxes_list


def check_not_none(bboxes_list):
    allcount=0
    for bboxes in bboxes_list:
        assert len(bboxes)>=1
        allcount+=len(bboxes)-1
    return allcount