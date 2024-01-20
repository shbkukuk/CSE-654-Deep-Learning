from glob import glob
import cv2
import os

mask_gt = 'C:/Users/SBK/Desktop/Deep Learning/finalproject/dataset/bigger_dataset/test/mask/*.png'
mask_path = glob(mask_gt)
save_path = 'C:/Users/SBK/Desktop/Deep Learning/finalproject/dataset/bigger_dataset/test/mask_rgb/'
for path in mask_path:
    name = os.path.basename(path)
    gt=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    gt_rgb = cv2.cvtColor(gt*255,cv2.COLOR_GRAY2BGR)
    cv2.imwrite(save_path+name,gt_rgb)
