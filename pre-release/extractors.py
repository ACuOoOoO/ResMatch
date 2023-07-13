import cv2
import numpy as np
import torch
import os
from extract_patches.core import extract_patches
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from superpoint import SuperPoint
from hardnet import HardNet

def resize(img,resize):
    img_h,img_w=img.shape[0],img.shape[1]
    cur_size=max(img_h,img_w)
    if len(resize)==1: 
      scale1,scale2=resize[0]/cur_size,resize[0]/cur_size
    else:
      scale1,scale2=resize[0]/img_h,resize[1]/img_w
    new_h,new_w=int(img_h*scale1),int(img_w*scale2)
    new_img=cv2.resize(img.astype('float32'),(new_w,new_h)).astype('uint8')
    scale=np.asarray([scale2,scale1])
    return new_img,scale


class ExtractSIFT:
    def __init__(self,config,root=True):
        self.num_kp=config['num_kpt']
        self.contrastThreshold=config['det_th']
        self.resize=config['resize']
        self.root=root

    def run(self, img_path):
        self.sift = cv2.SIFT_create(nfeatures=self.num_kp, contrastThreshold=self.contrastThreshold)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        scale=[1,1]
        if self.resize[0]!=-1:
            img,scale=resize(img,self.resize)
        cv_kp, desc = self.sift.detectAndCompute(img, None)
        kp = np.array([[_kp.pt[0]/scale[1], _kp.pt[1]/scale[0], _kp.response] for _kp in cv_kp]) # N*3
        index=np.flip(np.argsort(kp[:,2]))
        kp,desc=kp[index],desc[index]
        # print(kp[1],kp[2])
        if self.root:
            desc=np.sqrt(abs(desc/(np.linalg.norm(desc,axis=-1,ord=1)[:,np.newaxis]+1e-8)))
        return kp[:self.num_kp], desc[:self.num_kp]



class ExtractSuperpoint(object):
  def __init__(self,config):
    default_config = {
      'descriptor_dim': 256,
      'nms_radius': 4,
      'detection_threshold': config['det_th'],
      'max_keypoints': config['num_kpt'],
      'remove_borders': 4,
      'model_path':'../weights/sp/superpoint_v1.pth'
    }
    self.superpoint_extractor=SuperPoint(default_config)
    self.superpoint_extractor.eval(),self.superpoint_extractor.cuda()
    self.num_kp=config['num_kpt']
    self.resize=config['resize']

  def run(self,img_path):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    scale=1
    if self.resize[0]!=-1:
      img,scale=resize(img,self.resize)
    with torch.no_grad():
      result=self.superpoint_extractor(torch.from_numpy(img/255.).float()[None, None].cuda())
    score,kpt,desc=result['scores'][0],result['keypoints'][0],result['descriptors'][0]
    score,kpt,desc=score.cpu().numpy(),kpt.cpu().numpy(),desc.cpu().numpy().T
    kpt=np.concatenate([kpt/scale,score[:,np.newaxis]],axis=-1)
    #padding randomly
    if self.padding:
      if len(kpt)<self.num_kp:
        res=int(self.num_kp-len(kpt))
        pad_x,pad_desc=np.random.uniform(size=[res,2])*(img.shape[0]+img.shape[1])/2,np.random.uniform(size=[res,256])
        pad_kpt,pad_desc=np.concatenate([pad_x,np.zeros([res,1])],axis=-1),pad_desc/np.linalg.norm(pad_desc,axis=-1)[:,np.newaxis]
        kpt,desc=np.concatenate([kpt,pad_kpt],axis=0),np.concatenate([desc,pad_desc],axis=0)
    return kpt,desc


class ExtractDOGHN(object):
    def __init__(self,config):
        self.HN = HardNet()
        self.HN.load_state_dict(torch.load("../hardnet/checkpoint_liberty_with_aug.pth")['state_dict'])
        self.HN = self.HN.cuda()
        self.HN.eval()
        #self.sift = cv2.SIFT_create(contrastThreshold=config['det_th'])
        self.num_kp=config['num_kpt']
        self.resize=config['resize']
        self.det_th= config['det_th']
    def get_SIFT_keypoints(self, sift, img, lower_detection_th=False):

        # convert to gray-scale and compute SIFT keypoints
        keypoints = sift.detect(img, None)
        #print('hello')
        response = np.array([kp.response for kp in keypoints])
        respSort = np.argsort(response)[::-1]

        pt = np.array([kp.pt for kp in keypoints])[respSort]
        size = np.array([kp.size for kp in keypoints])[respSort]
        angle = np.array([kp.angle for kp in keypoints])[respSort]
        response = np.array([kp.response for kp in keypoints])[respSort]

        return pt, size, angle, response

    def feature_extract(self,img,num_kp):
        sift = cv2.SIFT_create(contrastThreshold=self.det_th)
        keypoints, scales, angles, responses = self.get_SIFT_keypoints(sift,
                                                                img)
        kpts = [
            cv2.KeyPoint(
                x=keypoints[i][0],
                y=keypoints[i][1],
                size=scales[i],
                angle=angles[i]) for i in range(min(num_kp,len(scales)))
        ]

        patches = extract_patches(
            kpts, img, 32, 16.0)
        patches = np.concatenate(patches).reshape(-1,32,32,1)*1.0/255
        patches = torch.FloatTensor(patches).permute(0,3,1,2).cuda()
        patches = (patches-0.4437)/0.201979
        with torch.no_grad():
            desc = self.HN(patches)
        return {'keypoints': keypoints[0:num_kp],'scores':responses[0:num_kp],'descriptors':desc.detach().cpu().numpy() }

    def run(self,img_path):
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        scale=[1,1]
        if self.resize[0]!=-1:
            img,scale=resize(img,self.resize)
        dict = self.feature_extract(img,self.num_kp)
        kpt,desc,score = dict['keypoints'],dict['descriptors'],np.expand_dims(dict['scores'],axis=-1)
        scale = np.expand_dims(np.flip(scale),0)
        kpt = kpt/scale
        kpt = np.concatenate([kpt,score],axis=-1)
        #print(kpt.shape)
        return kpt,desc
