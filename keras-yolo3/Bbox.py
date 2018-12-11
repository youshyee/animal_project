class bbox(object):
    def __init__(self,motion,score,cls,box,frame_index,isnone=False):
        if not isnone:
            self.motion=motion
            self.score=score
            self.cls=cls
            self.box=box#top,left,bottom,right
            self.isnone=isnone
            self.frame_index=frame_index
        else:
            self.isnone=True
            self.frame_index=frame_index

    def getemission(self):
        return self.score*self.motion
    def iouscore(self,bbox):
        iou=self.iou(self.box,bbox.box)
        return self.iou_fun(iou)
    def iou_fun(self,iou):
        if iou<=0.7:
            return iou*20/0.7-20
        elif iou>0.7 and iou<0.9:
            a=(0.9*np.e**-1-0.7)/(1-np.e**-1)
            b=-np.log(a+0.7)
            return np.log(iou+a)+b
        else:
            return 1

    def iou(self,box1,box2):
        '''
        top,left, bottom, right = box
        '''
        newtop=max(box1[0],box2[0])
        newbottom=min(box1[2],box2[2])

        newleft=max(box1[1],box2[1])
        newright=min(box1[3],box2[3])

        inter_area=max(0,newbottom-newtop)*max(0,newright-newleft)
        b1_area=(box1[2]-box1[0])*(box1[3]-box1[1])
        b2_area=(box2[2]-box2[0])*(box2[3]-box2[1])
        return inter_area/(b1_area+b2_area-inter_area)



