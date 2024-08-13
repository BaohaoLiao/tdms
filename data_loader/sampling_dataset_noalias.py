import os
import copy
import cv2
import random
from PIL import Image
from glob import glob
import numpy as np
from torch.utils.data import Dataset


def sampling(img_dir, number=10, image_size=512):
    img_paths = glob(f'{img_dir}/*.png')
    img_paths = sorted(img_paths)
    step = len(img_paths) / number
    st = len(img_paths) / 2.0 - 4.0 * step
    end = len(img_paths) + 0.0001
    
    imgs = np.zeros((image_size, image_size, number), dtype=np.uint8)
    for i, j in enumerate(np.arange(st, end, step)):
        p = img_paths[max(0, int((j-0.5001).round()))]
        img = Image.open(p).convert('L')
        img = np.array(img)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        imgs[..., i] = img.astype(np.uint8)
    return imgs
    

class RSNASamplingDataset(Dataset):
    def __init__(
            self, 
            df, 
            img_dir, 
            transform=None, 
            image_size=512, 
            in_channels=30, 
            phase="train", 
            flip_prob=0,
        ):
        self.df = df
        self.transform = transform
        self.image_size = image_size
        self.in_channels = in_channels
        self.img_dir = img_dir
        self.phase = phase
        self.flip_prob = flip_prob
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = np.zeros((self.image_size, self.image_size, self.in_channels), dtype=np.uint8)
        row = self.df.iloc[idx]
        st_id = int(row['study_id'])
        label = row[1:-1].values.astype(np.int64)
        new_label = copy.deepcopy(label)

        img_dir = os.path.join(self.img_dir, str(st_id))
        # Sagittal T1
        try:
            st1_vol  = sampling(
                f"{img_dir}/Sagittal_T1", number=self.in_channels//3, image_size=self.image_size
            )
            if self.phase == "train" and self.flip_prob > 0:
                if random.random() < self.flip_prob:
                    st1_vol = st1_vol[:, :, ::-1]
                    new_label[5:10] == label[10:15]
                    new_label[10:15] == label[5:10]
            x[..., :self.in_channels//3] = st1_vol  
        except:
            pass

        # Sagittal T2/STIR
        try:
            x[..., self.in_channels//3:2*self.in_channels//3] = sampling(
                f"{img_dir}/Sagittal_T2_STIR", number=self.in_channels//3, image_size=self.image_size
            )
        except:
            pass

        # Axial T2
        try:
            at2_vol = sampling(
                f"{img_dir}/Axial_T2", number=self.in_channels//3, image_size=self.image_size
            )
            if self.phase == "train" and self.flip_prob > 0:
                if random.random() < self.flip_prob:
                    at2_vol = at2_vol[:, ::-1, :]
                    new_label[15:20] == label[20:25]
                    new_label[20:25] == label[15:20]
            x[..., 2*self.in_channels//3:] = at2_vol
        except:
            pass
            
        assert np.sum(x)>0
        if self.transform is not None:
            x = self.transform(image=x)['image']
        #x = x.astype(np.float32)
        #x = x / 255
        x = x.transpose(2, 0, 1)
                
        return x, new_label