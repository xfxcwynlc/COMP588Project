import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class GlycogenDataset(Dataset):
    def __init__(self,image_dir, mask_dir, transform=None):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        img_path = os.path.join(self.image_dir, self.images[index])
        maskName = self.images[index]
        print(self.images[index])
        mask_path = os.path.join(self.mask_dir,maskName.replace("cr1","cm1")) #.replace(".jpg", "_mask.gif"))
        print(maskName.replace("cr1","cm1"))
        image = np.array(Image.open(img_path).convert("RGB"))
        #image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        #mask[mask == 255.0] = 1.0 #sigmoid facility

        #we will implement our own transform
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


