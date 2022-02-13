import os
import cv2
import glob
from numpy import uint64
from torch.utils.data import Dataset
import numpy as np

class OCRDataset(Dataset):
    def __init__(self,dataroot, transformer=None, inverse_color=True):
        super(OCRDataset, self).__init__()
        self.transformer=transformer
        #self.mode = mode
        self.img_pathes = glob.glob(dataroot, recursive=True)
        self.length = len(self.img_pathes)
        self.inverse_color=inverse_color
       
    def set_transformer(self, tr):
        self.transformer=tr

    def __len__(self):
        return self.length
    
    def __getitem__(self, index: uint64):
        img = cv2.imread(self.img_pathes[index])
        # transform
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.inverse_color:
            img = 255-img # inverse image to apply dilate
        res_dict = {'img':img, 'imgpath':self.img_pathes[index]}
        if self.transformer:
            res_dict = self.transformer(res_dict)
        #cv2.imwrite('vis/'+os.path.basename(res_dict['imgpath']), res_dict['img'])
        return res_dict

    def compute_average(self):
        avg = np.zeros((512,512), dtype=np.uint64)
        for i in range(self.__len__()):
            data = self.__getitem__(i)
            img = data['img'] 
            #cv2.imwrite(f'vis/{img.mean().astype(int)}_'+os.path.basename(data['imgpath']), img)
            avg+=img
            if (i+1)%100==0:
                print(f'{i} image processed.')
        print(avg.mean())
        print(avg.std())


if __name__=="__main__":
    from ..utils import build_transformer
    
    tr = build_transformer({
        "Resize":{"scale":4},
        "RandomCrop":{"size":"(256,256)"},
        "RandomRotation":{"degree":30},
    })
    root = '/data/**/*.jpg'
    dset = OCRDataset(root,tr)
    dset.compute_average() 

