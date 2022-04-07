import os
import cv2
import glob
from numpy import uint64

from torch.utils.data import Dataset
import shutil as sh

class FontDataSet(Dataset):
    def __init__(self,dataroot, transformer=None, inverse_color=True):
        super(FontDataSet, self).__init__()
        self.transformer=transformer
        #self.mode = mode
        self.img_pathes = glob.glob(dataroot, recursive=True)
        self.length = len(self.img_pathes)
        # print(f'before_move:{len(self.img_pathes)}')
        # for path in self.img_pathes :
        #     if os.path.isfile(path):
        #         name = os.path.basename(path)
        #         sh.move(path,f'/font_dataset/train/{name}')
        # ll = glob.glob('/font_dataset/train/*', recursive=True)
        # print(len(ll))
    def set_transformer(self, tr):
        self.transformer=tr

    def __len__(self):
        return self.length
    
    def __getitem__(self, index: uint64):
        img = cv2.imread(self.img_pathes[index])
        if len(img.shape)>2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res_dict = {'img':img, 'imgpath':self.img_pathes[index]}
      
        if self.transformer:
            res_dict = self.transformer(res_dict)
      
        return res_dict

