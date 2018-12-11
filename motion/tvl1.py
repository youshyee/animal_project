
import cv2
import numpy as np
import os 
#import matplotlib.pyplot as plt
from multiprocessing import Pool,current_process
from PIL import Image

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
def cal_optical_tvl1(prev,nextf):
    tvl1otf=cv2.DualTVL1OpticalFlow_create()
    flow= tvl1otf.calc(prev,nextf,None)
    return flow
# 
def visual_tvl1(videopath,savepath):
    check_dir(savepath)
    #################
    print('current processing{} and the PID is {}'.format(videopath.split()[-1],current_process().pid))
    cap = cv2.VideoCapture(videopath)
    _,frame=cap.read()
    prev=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    count=1
    while(True):
        ret, frame = cap.read()
        if ret :
            nextf=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            flow=cal_optical_tvl1(prev,nextf)
            intensity = np.sqrt(flow[...,0]**2+flow[...,1]**2)
            intensity=drap_change(intensity)
            im=Image.fromarray(intensity)
            im.save(os.path.join(savepath,'frame_{}.png'.format(count)))
            #plt.imsave(os.path.join(savepath,'frame_{}.jpg'.format(count)),intensity)
            
        else:
            break
        prev=nextf
        count+=1
    cap.release()

    
    # print('the actual frames {}, all possible frames {}'.format(len(optical_list),all_frames))
    # for i,optical in enumerate(optical_list):
    #     maxvalue=maxvalue*0.7
    #     optical[optical>maxvalue]=maxvalue
    #     optical=np.array(optical/maxvalue*255,dtype='uint8')
    #     if os.path.exists('frames'):
    #         pass
    #     else :
    #         os.mkdir('frames')
    #     print('writing ',i)
    #     cv2.imwrite('frames/frame_{}.jpg'.format(i),optical)
        
        # optical=cv2.cvtColor(optical,cv2.COLOR_GRAY2BGR)
        # for rect in rects:
        #     x,y,w,h=rect
        #     cv2.rectangle(optical,(x,y),(x+w,y+h),(0,0,255),1,cv2.LINE_AA)
        # hstack=np.hstack((frames[i],optical))
        # writer.write(hstack)
    # writer.release()
def multiprocessing_wrapper(args):
    return visual_tvl1(*args)
if __name__ == "__main__":
    pathdir='/mnt/storage/home/rn18510/space/data_species/gorilla'
    targets=os.listdir(pathdir)
    outpath=check_dir('/mnt/storage/home/rn18510/space/data_species/'+'tvl1frames')
    inputpath=[]
    outputpath=[]
    for target in targets: 
        inputpath.append(os.path.join(pathdir,target))
        outputpath.append(os.path.join(outpath,target.split('.')[0]))
    with Pool() as pool:
        pool.map(multiprocessing_wrapper,zip(inputpath,outputpath))
    
    
    
