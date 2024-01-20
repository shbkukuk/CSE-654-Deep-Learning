import cv2
import numpy as np
import os
from glob import glob
import shutil
import os
from tqdm import tqdm

target_root = 'C:/Users/SBK/Desktop/Deep Learning/finalproject/dataset/bigger_dataset/'
root = "C:/Users/SBK/Downloads/bdd100k_sem_seg_labels_trainval/bdd100k/labels/sem_seg/masks/train/*.png"
images = 'C:/Users/SBK/Downloads/bdd100k_images_10k/bdd100k/images/10k/train/*.jpg'
masks = glob(root)
images_name = [os.path.basename(x)[:-4] for x in glob(images)]
images_dir = glob(images)
train_count = 1696
val_count = 1908
test_count = 2120
count = 0

for mask_path in tqdm(masks):
    msk = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    car_msk = (msk==13).astype('uint8')
    ratio_of_car = np.count_nonzero(car_msk) / (720*1280)
    if ratio_of_car >= 0.1:
        msk_name = os.path.basename(mask_path)[:-4]
        id=images_name.index(msk_name)
        image_path = images_dir[id] 
        if count<=train_count:
            count += 1
            target_dir = target_root + f"train/images/{str(count)}.png"
            mask_target_dir = target_dir.replace('images','mask')
            shutil.copy(image_path,target_dir)
            cv2.imwrite(mask_target_dir,car_msk)
        elif count>train_count and count<=val_count:
            count += 1
            target_dir = target_root + f"val/images/{str(count)}.png"
            mask_target_dir = target_dir.replace('images','mask')
            shutil.copy(image_path,target_dir)
            cv2.imwrite(mask_target_dir,car_msk)
        elif count>val_count and count<=test_count:
            count += 1
            target_dir = target_root + f"test/images/{str(count)}.png"
            mask_target_dir = target_dir.replace('images','mask')
            shutil.copy(image_path,target_dir)
            cv2.imwrite(mask_target_dir,car_msk)
