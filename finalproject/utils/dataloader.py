import os 
import re
import cv2
import numpy as np
import torch
from torch.utils.data  import DataLoader, Dataset
from operator import itemgetter
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0]+ worker_id)

def sorted_alphanumeric(data):

    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    
    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)",key)]
    
    return sorted(data,key=alphanum_key)

def make_dataset(train_path,val_path):
    image_train_dir = os.path.join(train_path,"images")
    image_val_dir = os.path.join(val_path,"images")
    mask_train_dir = os.path.join(train_path,"mask")
    mask_val_dir = os.path.join(val_path,"mask")

    

    image_train_names = sorted_alphanumeric(os.listdir(image_train_dir))
    val_image_names = sorted_alphanumeric(os.listdir(image_val_dir)) 
    if "Thumbs.db" in image_train_names or "Thumbs.db" in val_image_names:
        image_train_names.remove("Thumbs.db")
        val_image_names.remove("Thumbs.db")
    
    mask_train_names = image_train_names
    val_mask_names = val_image_names
    assert len(image_train_names) == len(mask_train_names) , "train image and mask is not equal"
    assert len(val_image_names) == len(val_mask_names) , "val image and mask is not equal"
    return image_train_names,val_image_names, mask_train_names, val_mask_names

class carsegmentation(Dataset):
    def __init__(self,root,img_names,mask_names,transform, train=False) -> None:
        super().__init__()
        self.root = root
        self.img_names = img_names
        self.mask_names = mask_names
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        img_path = os.path.join(self.root,"images",self.img_names[index])
        mask_path = img_path.replace('images','mask')
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        mask= cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            sample = self.transform(image=img,mask=mask)
            img , mask = sample['image'] , sample['mask']
        if self.train: 
            return img ,mask 
        return img
    
    def __len__(self):
        return len(self.img_names)
    
def data_loader(
        data_path,
        batch_size,
        img_size,
        train_aug,
        val_aug,


):
    train_path = os.path.join(data_path,"train")
    val_path = os.path.join(data_path,"val")
    train_img_names, valid_img_names, train_mask_names, valid_mask_names = make_dataset(train_path=train_path,val_path=val_path)

    train_dataset = carsegmentation(root=train_path,train=True,transform=train_aug,img_names=train_img_names,mask_names=train_mask_names)
    val_dataset = carsegmentation(root=val_path,img_names=valid_img_names,mask_names=valid_mask_names,transform=val_aug,train=True)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader