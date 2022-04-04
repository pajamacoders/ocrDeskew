import os
import cv2
import glob
from numpy import uint64
from torch.utils.data import Dataset
import numpy as np

class OCRDataset3ch(Dataset):
    def __init__(self,dataroot, transformer=None, inverse_color=True):
        super(OCRDataset3ch, self).__init__()
        self.transformer=transformer
        #self.mode = mode
        self.img_pathes = glob.glob(dataroot, recursive=True)
        self.length = len(self.img_pathes)
       
    def set_transformer(self, tr):
        self.transformer=tr

    def __len__(self):
        return self.length
    
    def __getitem__(self, index: uint64):
        img = cv2.imread(self.img_pathes[index])
        if len(img.shape) == 2: 
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[2] == 4:   
            img = img[:,:,:3]
        res_dict = {'img':img, 'imgpath':self.img_pathes[index]}
      
        if self.transformer:
            res_dict = self.transformer(res_dict)
      
        return res_dict

    def compute_average(self):
        avg = np.zeros((512,512), dtype=np.uint64)
        for i in range(self.__len__()):
            data = self.__getitem__(i)
            img = data['img'] 
            cv2.imwrite(f'vis/'+os.path.basename(data['imgpath']), img)
            # avg+=img
            # if (i+1)%100==0:
            #     print(f'{i} image processed.')
        print(avg.mean())
        print(avg.std())
