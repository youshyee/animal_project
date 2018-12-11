import cv2
import os
import numpy as np
import time
import torch
from math import ceil
from torch.autograd import Variable
import models
from multiprocessing import Pool, current_process

def getmotionrects(motionframe):
    frame=motionframe.copy()
    hight=motionframe.shape[0]
    width=motionframe.shape[1]
    ret, thresh = cv2.threshold(motionframe, 5, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #print(len(contours))
    intensity=[]
    boxes=[]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        #print(x,y,w,h)
        if w*h<=2000:
            pass
        elif w*h > hight*width*0.65:
            pass
        else:
            top=y
            left=x
            bottom=min(y+h,hight)
            right=min(x+w,width)
            #print(x,y,w,h)
            tens=np.mean(frame[top:bottom,left:right])
            if tens>40:
                #cv2.rectangle(motionframe,(x,y),(x+w,y+h),(255,255,0),2)
                boxes.append(np.array([top,left,bottom,right]))
                intensity.append(tens)
    return intensity,boxes

def drap_change(intensity,threthold=3.7784260511398315,lowthre=0.04486689880490303):
    intensity[intensity>threthold]=threthold
    intensity[intensity<lowthre]=0.
    mal=255/threthold
    intensity=np.floor(intensity*mal)
    return np.array(intensity,dtype='uint8')
    
def check_dir(dir):
    if os.path.exists(dir):
        return dir
    else:
        os.mkdir(dir)
        return dir
def pwc(prev,nextf):
	pwc_model_fn = './pwc_net.pth.tar'
	im_all=[prev,nextf]
	divisor = 64.
	H = im_all[0].shape[0]
	W = im_all[0].shape[1]

	H_ = int(ceil(H/divisor) * divisor)
	W_ = int(ceil(W/divisor) * divisor)
	for i in range(len(im_all)):
		im_all[i] = cv2.resize(im_all[i], (W_, H_))

	for _i, _inputs in enumerate(im_all):
		im_all[_i] = 1.0 * im_all[_i]/255.0
		im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
		im_all[_i] = torch.from_numpy(im_all[_i])
		im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
		im_all[_i] = im_all[_i].float()
		
	im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)

	net = models.pwc_dc_net(pwc_model_fn)
	net = net.cuda()
	net.eval()

	flo = net(im_all)
	flo = flo[0] * 20.0
	flo = flo.cpu().data.numpy()

	# scale the flow back to the input size 
	flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) # 
	u_ = cv2.resize(flo[:,:,0],(W,H))
	v_ = cv2.resize(flo[:,:,1],(W,H))
	u_ *= W/ float(W_)
	v_ *= H/ float(H_)
	flo = np.dstack((u_,v_))
	return flo

def handle_each_video(videopath,cls,savetrainpath=check_dir('/mnt/storage/home/rn18510/space/mining_train_pwc'),savevalpath=check_dir('/mnt/storage/home/rn18510/space/mining_validation_pwc')):
    starttime=time.time()
    file=open('/mnt/storage/home/rn18510/space/mingingdata_pwc.txt','a+')
    videoname=videopath.split('/')[-1].split('.')[0]
    cap=cv2.VideoCapture(videopath)
    ret,prev=cap.read()
    #prev=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_count=1
    while(True):
        ret, frame = cap.read()
        if ret :
            #nextf=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            flow=pwc(prev,frame)
            intensity = np.sqrt(flow[...,0]**2+flow[...,1]**2)
            intensity=drap_change(intensity)
            nums,boxes=getmotionrects(intensity)
            prev=frame
            if len(boxes)==0:
                continue
            else:
                save_train=videoname+'_frame_{}.jpg'.format(frame_count)
                cv2.imwrite(os.path.join(savetrainpath,save_train),frame) #save training frames
                boxes_txt=''
                for i,box in enumerate(boxes):
                    top=box[0]
                    left=box[1]
                    bottom=box[2]
                    right=box[3]
                    cv2.rectangle(frame,(left,top),(right,bottom),(255,243,0),2)
                    cv2.putText(frame, text=str(nums[i]), org=(right, bottom), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(255, 0, 0), thickness=2)
                    boxes_txt+=str(left)+','+str(top)+','+str(right)+','+str(bottom)+','+str(cls)+' '
                print('boxestxt:',boxes_txt)
                #print(os.path.join(savetrainpath,save_train),boxes_txt,sep=' ',file=file)
                cv2.imwrite(os.path.join(savevalpath,save_train),frame)
        else:
            break
        frame_count+=1
    cap.release()
    endtime=time.time()
    file.close()
    print('cost time: ',endtime-starttime)
def wrapfun(para):
    print('current process pid:',current_process().pid)
    return handle_each_video(*para)

if __name__ == "__main__":
    chimpdir='/mnt/storage/home/rn18510/space/data_species/chimpanzee'
    gridir='/mnt/storage/home/rn18510/space/data_species/gorilla'
    chimvideos=os.listdir(chimpdir)
    grivideos=os.listdir(gridir)
    path1=[os.path.join(chimpdir,i) for i in chimvideos]
    path2=[os.path.join(gridir,i) for i in grivideos]

    path=path1+path2
    print(len(path))

    clslist=[0]*len(path1)+[1]*len(path2)

    testp=path[55:69]
    clsp=clslist[55:69]


    for i in range(len(clsp)):
        handle_each_video(testp[i],clsp[i])



                
