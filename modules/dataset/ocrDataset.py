import os
import cv2
import glob
from numpy import uint64
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    def __init__(self,dataroot, transformer=None):
        super(OCRDataset, self).__init__()
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
        # transform
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res_dict = {'img':img, 'imgpath':self.img_pathes[index]}
        if self.transformer:
            res_dict = self.transformer(res_dict)
        cv2.imwrite('vis/'+os.path.basename(res_dict['imgpath']), res_dict['img'])
        return res_dict



